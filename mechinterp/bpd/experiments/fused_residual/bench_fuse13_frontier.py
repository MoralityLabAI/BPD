"""FUSE-13 Stage D: Preflighted frontier scaling (no forced WDDM spill) and checkpoint transfer audit."""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

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
from .bench_fuse12_cuda import bootstrap_ci_ratio, percentile_95
from .validate_fuse12_cuda import (
    EXIT_POWER_CAP,
    check_gpu_safety,
    configure_precision_mode,
    query_nvml_snapshot,
)
from .validate_fuse13_precision import stage_c_trainer_parity

FRONTIER_SPECS = (
    (2048, 48),
    (4096, 24),
    (4096, 48),
    (6144, 24),
    (8192, 12),
)


def preflight_estimate_mib(width: int, n_passes: int, *, arm: str, mode: str) -> dict[str, float]:
    """Conservative analytical peak VRAM estimate per arm before any CUDA allocation.

    Accounts for model parameters, optimizer states (AdamW 2x), activations across T passes,
    and live autograd-saved residual storages:
      - Arm A (native): 2 * T distinct saved residuals per site (stochastic + final PGD)
      - Arm G (cached compute_dtype): 1 shared saved residual per site
    """
    raw_inter = int(round(4 * width * 2.0 / 3.0))
    inter = ((raw_inter + 255) // 256) * 256 if width >= 256 else width * 3
    # 2 layers x (o_proj [D, D] + down_proj [D, inter])
    site_elems = 2 * (width * width + width * inter)
    # Full mini-TRM weights (qkv + o + gate_up + down per layer) + V/U factors + AdamW moments
    full_model_elems = 2 * (3 * width * width + width * width + 2 * width * inter + width * inter)
    # Factor ranks D/32, D/16, 3D/32, D/8
    factor_elems = (
        width * (width // 32) + (width // 32) * width
        + inter * (width // 16) + (width // 16) * width
        + width * (3 * width // 32) + (3 * width // 32) * width
        + inter * (width // 8) + (width // 8) * width
    )
    elem_bytes = 2 if mode in ("bf16", "fp16") else 4
    base_mib = (full_model_elems * 4 + factor_elems * 4 * 4) / (1024.0 ** 2)
    # Autograd activations across 2 backward-retaining rollouts (batch=2, seq=20, 2 layers)
    act_mib = (2 * n_passes * 2 * 20 * (4 * width + 3 * inter) * 4 * 2.2) / (1024.0 ** 2)
    if arm == "A":
        # Native saves one materialized residual per pass per backward rollout
        res_mib = (2 * n_passes * site_elems * elem_bytes) / (1024.0 ** 2)
    else:
        # Arm G shares 1 residual per site (+ 1 transient fp32 reconstruction buffer)
        res_mib = (site_elems * elem_bytes + max(width * width, width * inter) * 4) / (1024.0 ** 2)
    total_est = base_mib + act_mib + res_mib + 600.0  # allocator fragmentation + cuBLAS workspace
    return {
        "base_weights_opt_mib": round(base_mib, 1),
        "activations_mib": round(act_mib, 1),
        "live_residuals_mib": round(res_mib, 1),
        "total_est_mib": round(total_est, 1),
        "single_site_set_mib": round((site_elems * elem_bytes) / (1024.0 ** 2), 1),
    }


def check_local_pretrained_trm() -> dict[str, Any]:
    """Check whether pretrained 7M TRM checkpoints are cached locally (never downloads)."""
    candidates = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / "models",
    ]
    found = []
    for root in candidates:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.name in ("step_21700", "model.pt") and (
                "trm" in str(p).lower() or "tinyrecursive" in str(p).lower()
            ):
                found.append(str(p))
    if not found:
        return {
            "status": "blocked_no_local_checkpoint",
            "reason": "Neither gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att (step_21700) nor yagizdevre/trm-sudoku-extreme-att (model.pt) is present in local caches; offline/no-download policy respected.",
            "found": [],
        }
    return {"status": "available", "found": found}


def run_single_arm_frontier(
    *,
    width: int,
    n_passes: int,
    arm: str,
    mode: str,
    n_roots: int,
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
    lease_limit_mib: float,
    min_headroom_mib: float,
    min_watts: float,
) -> dict[str, Any]:
    """Preflight and run one frontier arm cleanly inside the VRAMorama lease without forcing spill."""
    snap = check_gpu_safety(min_watts=min_watts, min_free_mib=min_headroom_mib)
    est = preflight_estimate_mib(width, n_passes, arm=arm, mode=mode)
    free_mib = snap.get("memory_free_mib")
    safe_budget_mib = min(lease_limit_mib, (free_mib - min_headroom_mib) if free_mib is not None else lease_limit_mib)

    if est["total_est_mib"] > safe_budget_mib:
        return {
            "arm": arm,
            "status": "memory_blocked_preflight",
            "preflight": est,
            "safe_budget_mib": round(safe_budget_mib, 1),
            "nvml_free_mib": free_mib,
        }

    ac_fn = configure_precision_mode(mode)
    elem_bytes = 2 if mode in ("bf16", "fp16") else 4
    proc = psutil.Process(os.getpid())
    cuda_ms_vals: list[float] = []
    wall_ms_vals: list[float] = []
    power_vals: list[float] = []
    peak_alloc = 0
    peak_res = 0
    nvml_peak_used = 0.0
    host_rss_mib = 0.0
    payload_bytes = 0
    cuda_receipt = None
    total_steps = warmup_steps + timed_steps

    try:
        for root in range(n_roots):
            seed = 4013701 + root * 31 + width + n_passes * 100
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            model = build_mini_trm(
                hidden_size=width, num_heads=4, l_layers=2,
                h_cycles=1, l_cycles=2, puzzle_emb_len=4, seq_len=16, seed=seed,
            )
            paths = [f"layers.{i}.{n}" for i in range(2) for n in ("o_proj", "down_proj")]
            sites = dict(zip(paths, (max(2, width // 32), max(4, width // 16), max(6, 3 * width // 32), max(8, width // 8))))
            cfg = BPDConfig(
                c_per_module=sites, n_passes=n_passes, n_steps=total_steps, batch_size=2, seq_len=20,
                seed=seed, init_svd=False, faithfulness_warmup_steps=1,
                coeff_faith=1e5, coeff_stoch=0.5, coeff_adv=0.5,
                readout_policy="readout_or_boundary", rollout_mode="closed_loop", window_steps=max(1, n_passes // 2),
                ci=CIConfig(variant="state_only", d_model=32, n_blocks=1, n_heads=4, mlp_hidden=64,
                            max_passes=n_passes, max_seq_len=20),
                ablation=AblationConfig(mode="zero", sphere_rescale=False),
                adversary=AdversaryConfig(inner_steps=2),
                null_calibration=NullCalibrationConfig(use_tolerance_weights=False),
            )
            bpd = BPDModel(model, sites, max_passes=n_passes, ablation=cfg.ablation, init_svd=False)
            adapter = TRMBPDAdapter(model)
            is_g = arm == "G"
            cap_bytes = sum(w.W_target.numel() * elem_bytes for w in bpd.wrappers.values())
            tr = BPDTrainer(
                bpd, cfg, adapter.rollout,
                clean_window_state_fn=adapter.sample_clean_window_state,
                visible_dim=adapter.visible_dim, prompt_dim=adapter.prompt_dim,
                device=device,
                cache_single_backward_residuals=is_g,
                residual_cache_max_bytes=cap_bytes,
                residual_cache_policy="step_exact" if is_g else "first_fit",
                cache_pgd_mask_residuals=is_g,
                cache_calibration_residuals=is_g,
                cache_pgd_final_residuals=is_g,
                allow_cuda_residual_cache=True,
                residual_cache_precision_mode=mode,
                residual_cache_storage_mode="compute_dtype" if is_g else "fp32",
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
                    nvml_peak_used = max(nvml_peak_used, float(sn_post["memory_used_mib"]))

            peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            peak_res = max(peak_res, int(torch.cuda.max_memory_reserved(device)))
            host_rss_mib = max(host_rss_mib, proc.memory_info().rss / (1024.0 ** 2))
            if tr.residual_step_receipt:
                payload_bytes = max(payload_bytes, int(tr.residual_step_receipt["peak_bytes"]))
            cuda_receipt = tr.residual_cuda_receipt
            del tr, adapter, bpd, model, batch
            gc.collect()
            torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        gc.collect()
        return {"arm": arm, "status": "oom", "preflight": est, "error": str(exc)}

    med_cuda = statistics.median(cuda_ms_vals)
    med_wall = statistics.median(wall_ms_vals)
    mean_w = statistics.mean(power_vals) if power_vals else None
    energy_mj = (mean_w * med_wall) if mean_w is not None else None
    return {
        "arm": arm,
        "status": "ok",
        "preflight": est,
        "raw_cuda_ms": cuda_ms_vals,
        "raw_wall_ms": wall_ms_vals,
        "cuda_median_ms": med_cuda,
        "cuda_p95_ms": percentile_95(cuda_ms_vals),
        "wall_median_ms": med_wall,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_res,
        "nvml_used_mib": nvml_peak_used,
        "host_rss_mib": host_rss_mib,
        "payload_peak_bytes": payload_bytes,
        "mean_power_w": mean_w,
        "energy_per_step_mj": energy_mj,
        "cuda_receipt": cuda_receipt,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="Resumable JSONL frontier receipt path")
    p.add_argument("--allow-cuda", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--lease-mib", type=float, default=12288.0)
    p.add_argument("--min-headroom-mib", type=float, default=2048.0)
    p.add_argument("--min-watts", type=float, default=60.0)
    p.add_argument("--modes", default="fp32,bf16")
    args = p.parse_args()

    if not args.allow_cuda:
        p.error("CUDA frontier runner requires --allow-cuda inside VRAMorama lease")
    device = torch.device("cuda:0")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    completed: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    completed.add(json.loads(line)["cell_id"])
                except Exception:
                    pass

    # 1. Checkpoint transfer availability receipt
    ckpt_cid = "checkpoint_transfer_audit"
    if ckpt_cid not in completed:
        rec = {"cell_id": ckpt_cid, "kind": "checkpoint_transfer", **check_local_pretrained_trm()}
        with args.output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, allow_nan=False) + "\n")
        completed.add(ckpt_cid)

    # 2. Width-4096 full trainer parity validation at safe horizon (T=2 and T=6 fit cleanly inside 12 GiB lease!)
    for mode in modes:
        for t_par in (2, 6):
            cid = f"w4096_parity__{mode}__t{t_par}"
            if cid in completed:
                continue
            est_a = preflight_estimate_mib(4096, t_par, arm="A", mode=mode)
            snap = check_gpu_safety(min_watts=args.min_watts, min_free_mib=args.min_headroom_mib)
            safe_b = min(args.lease_mib, (snap["memory_free_mib"] - args.min_headroom_mib))
            if est_a["total_est_mib"] > safe_b:
                rec = {
                    "cell_id": cid, "kind": "w4096_parity", "mode": mode, "width": 4096, "n_passes": t_par,
                    "status": "memory_blocked_preflight", "preflight_arm_a": est_a, "safe_budget_mib": safe_b,
                }
            else:
                print(f"[frontier-parity] running {cid} ...", flush=True)
                res = stage_c_trainer_parity(device, mode, 4096, t_par, "off", "full", seed=4013801 + t_par)
                rec = {"cell_id": cid, "kind": "w4096_parity", "status": "ok", **res}
                print(f"  -> finite_bitwise_pass={res['finite_bitwise_pass']} ({res['parity']['finite_bitwise']}/{res['parity']['tensors']})", flush=True)
            with args.output.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, allow_nan=False) + "\n")
            completed.add(cid)

    # 3. Frontier scaling matrix with independent per-arm preflight
    for mode in modes:
        for width, n_passes in FRONTIER_SPECS:
            cid = f"frontier__{mode}__w{width}_p{n_passes}"
            if cid in completed:
                continue
            print(f"[frontier] preflight & running {cid} ...", flush=True)
            arm_a = run_single_arm_frontier(
                width=width, n_passes=n_passes, arm="A", mode=mode,
                n_roots=2, warmup_steps=1, timed_steps=2, device=device,
                lease_limit_mib=args.lease_mib, min_headroom_mib=args.min_headroom_mib, min_watts=args.min_watts,
            )
            arm_g = run_single_arm_frontier(
                width=width, n_passes=n_passes, arm="G", mode=mode,
                n_roots=2, warmup_steps=1, timed_steps=2, device=device,
                lease_limit_mib=args.lease_mib, min_headroom_mib=args.min_headroom_mib, min_watts=args.min_watts,
            )
            speedup = None
            ci95 = None
            if arm_a["status"] == "ok" and arm_g["status"] == "ok":
                speedup = arm_a["cuda_median_ms"] / max(arm_g["cuda_median_ms"], 1e-12)
                ci95 = bootstrap_ci_ratio(arm_a["raw_cuda_ms"], arm_g["raw_cuda_ms"])
            rec = {
                "cell_id": cid,
                "kind": "frontier_scaling",
                "mode": mode,
                "width": width,
                "n_passes": n_passes,
                "arm_a": arm_a,
                "arm_g": arm_g,
                "speedup_cuda": speedup,
                "speedup_ci95": ci95,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with args.output.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, allow_nan=False) + "\n")
            completed.add(cid)
            print(
                f"  -> A={arm_a['status']} ({arm_a.get('cuda_median_ms') or arm_a['preflight']['total_est_mib']}) | "
                f"G={arm_g['status']} ({arm_g.get('cuda_median_ms') or arm_g['preflight']['total_est_mib']})",
                flush=True,
            )

    configure_precision_mode("fp32")
    print("FUSE13_FRONTIER_DONE", json.dumps({"cells_completed": len(completed)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
