"""FUSE-14 Stage F: Reattempt all five blocked BF16 frontier configurations.

Refits conservative preflight directly from measured allocator + activation high-water
marks (never subtracting modeled cast bytes from an unrelated peak). Runs inside
VRAMorama strict FIFO (12 GiB lease, 2048 MiB headroom, >=60 W guard, exit 75 pause).
Reports GPU alloc/reserved, NVML, host RSS, p50/p95 step timings, tokens/s, energy,
and explicitly marks blocked native / FUSE-13 Arm G baselines as censored comparisons.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import psutil
import torch
from torch.utils.checkpoint import checkpoint as ckpt_fn

from mechinterp.bpd.adapters.trm import TRMBPDAdapter, build_mini_trm
from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import (
    AblationConfig,
    AdversaryConfig,
    BPDConfig,
    CIConfig,
    NullCalibrationConfig,
)
from mechinterp.bpd.trainer import BPDTrainer
from .bench_fuse12_cuda import percentile_95
from .bench_fuse13_frontier import preflight_estimate_mib as fuse13_preflight_mib
from .validate_fuse12_cuda import (
    EXIT_POWER_CAP,
    check_gpu_safety,
    configure_precision_mode,
    query_nvml_snapshot,
)

FRONTIER_SPECS_BF16 = (
    (2048, 48),
    (4096, 24),
    (4096, 48),
    (6144, 24),
    (8192, 12),
    (8192, 24),  # Attempted ONLY if conservative preflight fits inside 12 GiB lease
)


def fuse14_bf16_preflight_mib(width: int, n_passes: int, *, arm: str) -> dict[str, float]:
    """Conservative BF16 analytical peak VRAM model calibrated from measured FUSE-13/14 peaks.

    - Arm G (FUSE-13): shared BF16 residuals, but unwrapped frozen projections
      (qkv_proj, gate_up_proj) save 2 * T * unwrapped_elems * 2 bytes.
    - Arm H (FUSE-14): shared BF16 residuals + 1 shared detached BF16 cast per
      frozen projection (4 * (3*w*w + 2*i*w) bytes for 2 layers).
    - Arm H_ckpt (FUSE-14 + explicit recompute-safe checkpointing): eliminates
      per-pass activation retention (retains only boundary states + 1 block work buffer)
      with frozen projection casts shared across forward and recompute.
    """
    if arm == "G":
        return fuse13_preflight_mib(width, n_passes, arm="G", mode="bf16")

    raw_inter = int(round(4 * width * 2.0 / 3.0))
    inter = ((raw_inter + 255) // 256) * 256 if width >= 256 else width * 3
    site_elems = 2 * (width * width + width * inter)
    unwrapped_elems = 2 * (3 * width * width + 2 * width * inter)
    full_model_elems = site_elems + unwrapped_elems
    factor_elems = (
        width * (width // 32) + (width // 32) * width
        + inter * (width // 16) + (width // 16) * width
        + width * (3 * width // 32) + (3 * width // 32) * width
        + inter * (width // 8) + (width // 8) * width
    )
    # FP32 frozen model + FP32 factors + AdamW (m, v, grad = 3x)
    base_mib = (full_model_elems * 4 + factor_elems * 4 * 4) / (1024.0 ** 2)
    # 1 shared BF16 cast per frozen projection (qkv_proj + gate_up_proj across 2 layers)
    shared_proj_cast_mib = (unwrapped_elems * 2) / (1024.0 ** 2)
    # 1 shared BF16 residual per wrapped site + 1 transient FP32 reconstruction buffer
    res_mib = (site_elems * 2 + max(width * width, width * inter) * 4) / (1024.0 ** 2)

    if arm == "H":
        # Conservative allocator + activation high-water mark calibrated from measured
        # FP32 Arm G frontier peaks (w=2048 T=48: 2415 MiB; w=4096 T=24: 4703 MiB;
        # w=4096 T=48: 6305 MiB; w=6144 T=24: 9316 MiB).
        act_mib = (2 * n_passes * 2 * 20 * (4 * width + 3 * inter) * 4 * 2.4) / (1024.0 ** 2)
        total_est = base_mib + shared_proj_cast_mib + res_mib + act_mib + 750.0
    elif arm == "H_ckpt":
        # Checkpointing retains only 2 * T boundary states + 1 active block's activations
        act_mib = (2 * (n_passes * 2 * 20 * width * 4 + 2 * 20 * (4 * width + 3 * inter) * 4 * 3.0)) / (1024.0 ** 2)
        total_est = base_mib + shared_proj_cast_mib + act_mib + 750.0
    else:
        raise ValueError(f"unknown arm: {arm}")

    return {
        "base_weights_opt_mib": round(base_mib, 1),
        "shared_proj_cast_mib": round(shared_proj_cast_mib, 1),
        "live_residuals_mib": round(res_mib if arm == "H" else 0.0, 1),
        "activations_mib": round(act_mib, 1),
        "total_est_mib": round(total_est, 1),
    }


def run_frontier_arm(
    *,
    width: int,
    n_passes: int,
    arm: str,
    device: torch.device,
    lease_mib: float,
    min_headroom_mib: float,
    min_watts: float,
    n_roots: int = 2,
    warmup_steps: int = 1,
    timed_steps: int = 2,
) -> dict[str, Any]:
    snap = check_gpu_safety(min_watts=min_watts, min_free_mib=min_headroom_mib)
    est = fuse14_bf16_preflight_mib(width, n_passes, arm=arm)
    free_mib = snap.get("memory_free_mib")
    safe_budget = min(lease_mib, (free_mib - min_headroom_mib) if free_mib is not None else lease_mib)
    if est["total_est_mib"] > safe_budget:
        return {
            "arm": arm,
            "status": "memory_blocked_preflight",
            "preflight": est,
            "safe_budget_mib": round(safe_budget, 1),
            "nvml_free_mib": free_mib,
        }

    ac_fn = configure_precision_mode("bf16")
    proc = psutil.Process(os.getpid())
    cuda_ms_vals: list[float] = []
    wall_ms_vals: list[float] = []
    power_vals: list[float] = []
    peak_alloc = 0
    peak_res = 0
    nvml_peak = 0.0
    host_rss_mib = 0.0
    proj_receipt = None
    res_receipt = None
    total_steps = warmup_steps + timed_steps
    # Effective tokens processed per step across 2 retaining rollouts (batch=2, seq=20, T passes)
    tokens_per_step = 2 * 2 * 20 * n_passes

    try:
        for root in range(n_roots):
            seed = 14601 + root * 37 + width + n_passes * 100
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            model = build_mini_trm(
                hidden_size=width,
                num_heads=4,
                l_layers=2,
                h_cycles=1,
                l_cycles=2,
                puzzle_emb_len=4,
                seq_len=16,
                seed=seed,
            )
            paths = [f"layers.{i}.{n}" for i in range(2) for n in ("o_proj", "down_proj")]
            sites = dict(
                zip(
                    paths,
                    (
                        max(2, width // 32),
                        max(4, width // 16),
                        max(6, 3 * width // 32),
                        max(8, width // 8),
                    ),
                )
            )
            cfg = BPDConfig(
                c_per_module=sites,
                n_passes=n_passes,
                n_steps=total_steps,
                batch_size=2,
                seq_len=20,
                seed=seed,
                init_svd=False,
                faithfulness_warmup_steps=1,
                coeff_faith=1e5,
                coeff_stoch=0.5,
                coeff_adv=0.5,
                readout_policy="readout_or_boundary",
                rollout_mode="closed_loop",
                window_steps=max(1, n_passes // 2),
                ci=CIConfig(
                    variant="state_only",
                    d_model=32,
                    n_blocks=1,
                    n_heads=4,
                    mlp_hidden=64,
                    max_passes=n_passes,
                    max_seq_len=20,
                ),
                ablation=AblationConfig(mode="zero", sphere_rescale=False),
                adversary=AdversaryConfig(inner_steps=2),
                null_calibration=NullCalibrationConfig(use_tolerance_weights=False),
            )
            bpd = BPDModel(model, sites, max_passes=n_passes, ablation=cfg.ablation, init_svd=False)
            adapter = TRMBPDAdapter(model)

            use_res = arm in ("G", "H")
            use_proj = arm in ("H", "H_ckpt")
            use_ckpt = arm == "H_ckpt"
            if use_ckpt:
                from mechinterp.bpd.components import TiedComponentLinear

                def _ckpt_l(self_m, state, injection):
                    h = state + injection
                    for blk in self_m.layers:
                        if torch.is_grad_enabled():
                            wrappers = [m for m in (blk.o_proj, blk.down_proj) if isinstance(m, TiedComponentLinear)]
                            snap_state = [(w.mode, w.schedule, w.calls, w.pass_offset) for w in wrappers]

                            def _run_blk(h_in, _blk=blk, _ws=wrappers, _st=snap_state):
                                prev = [(w.mode, w.schedule, w.calls, w.pass_offset) for w in _ws]
                                for w, (md, sc, cl, po) in zip(_ws, _st):
                                    w.mode, w.schedule, w.calls, w.pass_offset = md, sc, cl, po
                                try:
                                    return _blk(h_in, self_m.cos, self_m.sin)
                                finally:
                                    for w, (md, sc, cl, po) in zip(_ws, prev):
                                        w.mode, w.schedule, w.calls, w.pass_offset = md, sc, cl, po

                            h = ckpt_fn(_run_blk, h, use_reentrant=False)
                            for w, (_, _, cl, _) in zip(wrappers, snap_state):
                                w.calls = cl + 1
                        else:
                            h = blk(h, self_m.cos, self_m.sin)
                    return h

                model.run_l_level = MethodType(_ckpt_l, model)

            cap_bytes = sum(w.W_target.numel() * 2 for w in bpd.wrappers.values())
            tr = BPDTrainer(
                bpd,
                cfg,
                adapter.rollout,
                clean_window_state_fn=adapter.sample_clean_window_state,
                visible_dim=adapter.visible_dim,
                prompt_dim=adapter.prompt_dim,
                device=device,
                cache_single_backward_residuals=use_res,
                residual_cache_max_bytes=cap_bytes,
                residual_cache_policy="step_exact" if use_res else "first_fit",
                cache_pgd_mask_residuals=use_res,
                cache_calibration_residuals=use_res,
                cache_pgd_final_residuals=use_res,
                allow_cuda_residual_cache=True,
                residual_cache_precision_mode="bf16",
                residual_cache_storage_mode="compute_dtype" if use_res else "fp32",
                cache_frozen_projection_casts=use_proj,
                projection_cache_targets="block_projs",
                projection_cache_allow_recompute=use_ckpt,
            )
            with ac_fn():
                tr.run_faithfulness_warmup(1)

            gen_b = torch.Generator().manual_seed(seed + 500)
            for s in range(total_steps):
                check_gpu_safety(min_watts=min_watts, min_free_mib=1024.0)
                inp = torch.randint(1, 11, (2, 16), generator=gen_b)
                batch = adapter.make_batch_with_clean_cache(inp, n_passes)
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                p_pre = query_nvml_snapshot().get("power_draw_w")
                t0 = time.perf_counter()
                ev0.record()
                with ac_fn():
                    tr.train_step(batch, s)
                ev1.record()
                torch.cuda.synchronize(device)
                wall_ms = (time.perf_counter() - t0) * 1000.0
                cuda_ms = float(ev0.elapsed_time(ev1))
                sn_post = query_nvml_snapshot()
                if s >= warmup_steps:
                    wall_ms_vals.append(wall_ms)
                    cuda_ms_vals.append(cuda_ms)
                    p_post = sn_post.get("power_draw_w")
                    if p_pre is not None and p_post is not None:
                        power_vals.append(0.5 * (p_pre + p_post))
                    elif p_post is not None:
                        power_vals.append(p_post)
                if sn_post.get("memory_used_mib") is not None:
                    nvml_peak = max(nvml_peak, float(sn_post["memory_used_mib"]))

            peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            peak_res = max(peak_res, int(torch.cuda.max_memory_reserved(device)))
            host_rss_mib = max(host_rss_mib, proc.memory_info().rss / (1024.0 ** 2))
            proj_receipt = tr.projection_cache_receipt
            res_receipt = tr.residual_step_receipt
            del tr, adapter, bpd, model, batch
            gc.collect()
            torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, getattr(torch, "AcceleratorError", RuntimeError)) as exc:
        msg = str(exc)
        if isinstance(exc, RuntimeError) and not (
            isinstance(exc, torch.cuda.OutOfMemoryError)
            or "out of memory" in msg.lower()
            or "Insufficient WDDM VRAM headroom" in msg
        ):
            raise
        gc.collect()
        torch.cuda.empty_cache()
        status = "memory_blocked_runtime_headroom" if "Insufficient WDDM VRAM headroom" in msg else "oom"
        return {"arm": arm, "status": status, "preflight": est, "error": msg}

    med_cuda = statistics.median(cuda_ms_vals)
    med_wall = statistics.median(wall_ms_vals)
    mean_w = statistics.mean(power_vals) if power_vals else None
    energy_mj = (mean_w * med_wall) if mean_w is not None else None
    toks_per_s = tokens_per_step / max(med_cuda / 1000.0, 1e-9)
    return {
        "arm": arm,
        "status": "ok",
        "preflight": est,
        "raw_cuda_ms": cuda_ms_vals,
        "raw_wall_ms": wall_ms_vals,
        "cuda_median_ms": med_cuda,
        "cuda_p95_ms": percentile_95(cuda_ms_vals),
        "wall_median_ms": med_wall,
        "tokens_per_s": toks_per_s,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_res,
        "nvml_used_mib": nvml_peak,
        "host_rss_mib": host_rss_mib,
        "mean_power_w": mean_w,
        "energy_per_step_mj": energy_mj,
        "residual_receipt": res_receipt,
        "projection_receipt": proj_receipt,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-cuda", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--lease-mib", type=float, default=12288.0)
    p.add_argument("--min-headroom-mib", type=float, default=2048.0)
    p.add_argument("--min-watts", type=float, default=60.0)
    args = p.parse_args()

    if not args.allow_cuda:
        p.error("CUDA frontier runner requires --allow-cuda inside VRAMorama lease")
    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.set_device(device)

    completed: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    completed.add(json.loads(line)["cell_id"])
                except Exception:
                    pass

    for width, n_passes in FRONTIER_SPECS_BF16:
        cid = f"fuse14_frontier__bf16__w{width}_p{n_passes}"
        if cid in completed:
            continue
        print(f"[fuse14-frontier] running {cid} ...", flush=True)
        # Always preflight FUSE-13 Arm G (blocked -> censored comparison, not a speedup)
        arm_g = run_frontier_arm(
            width=width, n_passes=n_passes, arm="G", device=device,
            lease_mib=args.lease_mib, min_headroom_mib=args.min_headroom_mib, min_watts=args.min_watts,
        )
        arm_h = run_frontier_arm(
            width=width, n_passes=n_passes, arm="H", device=device,
            lease_mib=args.lease_mib, min_headroom_mib=args.min_headroom_mib, min_watts=args.min_watts,
        )
        # Only test H_ckpt when H is blocked (e.g. w=8192) or on (4096, 48) for direct comparison
        arm_h_ckpt = None
        if arm_h["status"] != "ok" or (width == 4096 and n_passes == 48):
            arm_h_ckpt = run_frontier_arm(
                width=width, n_passes=n_passes, arm="H_ckpt", device=device,
                lease_mib=args.lease_mib, min_headroom_mib=args.min_headroom_mib, min_watts=args.min_watts,
            )
        rec = {
            "cell_id": cid,
            "mode": "bf16",
            "width": width,
            "n_passes": n_passes,
            "baseline_comparison": "censored_baseline_blocked" if arm_g["status"] != "ok" else "matched",
            "arm_g_fuse13": arm_g,
            "arm_h_fuse14": arm_h,
            "arm_h_ckpt": arm_h_ckpt,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with args.output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, allow_nan=False) + "\n")
        completed.add(cid)
        print(
            f"  -> G={arm_g['status']} | H={arm_h['status']} "
            f"({arm_h.get('cuda_median_ms') or arm_h['preflight']['total_est_mib']}) | "
            f"H_ckpt={arm_h_ckpt['status'] if arm_h_ckpt else 'skipped'}",
            flush=True,
        )

    configure_precision_mode("fp32")
    print("FUSE14_FRONTIER_DONE", json.dumps({"cells_completed": len(completed)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
