"""FUSE-13 Stage A: Native mixed-precision calibration (`observe_clean_pass`) verification."""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mechinterp.bpd.adapters.trm import TRMBPDAdapter, build_mini_trm
from mechinterp.bpd.components import BPDModel, TiedComponentLinear
from mechinterp.bpd.config import (
    AblationConfig,
    AdversaryConfig,
    BPDConfig,
    CIConfig,
    NullCalibrationConfig,
)
from mechinterp.bpd.trainer import BPDTrainer
from .validate_fuse12_cuda import EXIT_POWER_CAP, check_gpu_safety, configure_precision_mode


def _ac_for(device: torch.device, mode: str):
    if device.type == "cuda":
        return configure_precision_mode(mode)
    if mode == "fp32":
        torch.set_float32_matmul_precision("highest")
        return lambda: nullcontext()
    if mode == "bf16":
        torch.set_float32_matmul_precision("highest")
        return lambda: torch.autocast("cpu", enabled=True, dtype=torch.bfloat16)
    raise ValueError(f"Unsupported mode {mode!r} on {device.type}")


def verify_unit_calibration(device: torch.device, mode: str) -> dict[str, Any]:
    """Verify initial copy, repeated EMA, capacity expansion, repeated pass IDs, and dtype traces."""
    ac_factory = _ac_for(device, mode)
    torch.manual_seed(4013101)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(4013101)

    # Start with tiny max_passes=2 to force _ensure_pass_capacity expansion
    w = TiedComponentLinear(
        nn.Linear(64, 64),
        16,
        max_passes=2,
        ablation=AblationConfig(mode="pass_mean", ema_momentum=0.1, sphere_rescale=True),
        init_svd=False,
    ).to(device)
    xs = [torch.randn(2, 8, 64, device=device) for _ in range(4)]

    # Trace exact pre-cast reduction dtypes inside autocast
    with torch.no_grad(), ac_factory():
        xf = xs[0].detach().float()
        raw_acts = (xf @ w.V).reshape(-1, w.C).mean(dim=0)
        raw_delta = F.linear(xf, w.weight_delta()).reshape(-1, w.d_out).mean(dim=0)
        raw_clean = F.linear(xf, w.W_target.float()).reshape(-1, w.d_out)
        raw_rms = raw_clean.pow(2).mean(dim=-1).add(w.ablation.eps).sqrt().mean()
    dtype_trace = {
        "raw_acts_mean_dtype": str(raw_acts.dtype),
        "raw_delta_mean_dtype": str(raw_delta.dtype),
        "raw_write_rms_dtype": str(raw_rms.dtype),
    }

    # 1. Initial calibration on tau=0,1
    with torch.no_grad(), ac_factory():
        w.observe_clean_pass(xs[0], 0)
        w.observe_clean_pass(xs[1], 1)
    assert bool(w.calibrated_passes[0].item()) and bool(w.calibrated_passes[1].item())
    # Exact check that initial copy matches raw statistic cast to float32
    assert torch.equal(w.mean_comp_acts[0], raw_acts.to(torch.float32))
    assert torch.equal(w.mean_delta_out[0], raw_delta.to(torch.float32))
    assert torch.equal(w.mean_write_rms[0], raw_rms.to(torch.float32))
    snap_tau0_step0 = w.mean_comp_acts[0].clone()
    snap_tau1_step0 = w.mean_comp_acts[1].clone()

    # 2. Repeated EMA on same pass IDs (this previously crashed on bf16/fp16 lerp_!)
    with torch.no_grad(), ac_factory():
        w.observe_clean_pass(xs[2], 0)
        w.observe_clean_pass(xs[3], 0)
        # Verify exact reference lerp_ formula on cast-after-reduction statistic
        xf2 = xs[2].detach().float()
        a2 = (xf2 @ w.V).reshape(-1, w.C).mean(dim=0).to(torch.float32)
        xf3 = xs[3].detach().float()
        a3 = (xf3 @ w.V).reshape(-1, w.C).mean(dim=0).to(torch.float32)
    expected_tau0 = snap_tau0_step0.lerp(a2, 0.1).lerp(a3, 0.1)
    assert torch.equal(w.mean_comp_acts[0], expected_tau0), "EMA cast-after-reduction mismatch"

    # 3. Capacity expansion (tau=5 > max_passes=2) + repeated EMA on expanded slot
    with torch.no_grad(), ac_factory():
        w.observe_clean_pass(xs[0], 5)
        w.observe_clean_pass(xs[1], 5)
    assert w.max_passes >= 6 and w.mean_comp_acts.shape[0] >= 6
    assert torch.equal(w.mean_comp_acts[1], snap_tau1_step0), "expansion altered existing slot"
    assert torch.equal(w.mean_comp_acts[0], expected_tau0), "expansion altered EMA slot"
    assert w.mean_comp_acts.dtype == torch.float32
    assert w.mean_delta_out.dtype == torch.float32
    assert w.mean_write_rms.dtype == torch.float32
    assert bool(torch.isfinite(w.mean_comp_acts).all().item())
    assert bool(torch.isfinite(w.mean_delta_out).all().item())
    assert bool(torch.isfinite(w.mean_write_rms).all().item())

    dtype_trace["buffer_dtypes"] = {
        "mean_comp_acts": str(w.mean_comp_acts.dtype),
        "mean_delta_out": str(w.mean_delta_out.dtype),
        "mean_write_rms": str(w.mean_write_rms.dtype),
    }
    return {
        "status": "pass",
        "mode": mode,
        "dtype_trace": dtype_trace,
        "expanded_max_passes": int(w.max_passes),
    }


def verify_native_trainer_calibration(
    device: torch.device, mode: str, width: int = 128, n_passes: int = 6
) -> list[dict[str, Any]]:
    """Run native Arm A BPDTrainer across closed_loop/clean_window and cadences off/1/50."""
    ac_factory = _ac_for(device, mode)
    rows = []
    for regime in ("closed_loop", "clean_window"):
        for cal_cadence in ("off", "1", "50"):
            seed = 4013201 + (100 if regime == "clean_window" else 0) + {"off": 0, "1": 1, "50": 50}[cal_cadence]
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = build_mini_trm(
                hidden_size=width, num_heads=4, l_layers=2,
                h_cycles=1, l_cycles=2, puzzle_emb_len=4, seq_len=16, seed=seed,
            )
            paths = [f"layers.{i}.{n}" for i in range(2) for n in ("o_proj", "down_proj")]
            sites = dict(zip(paths, (max(2, width // 32), max(4, width // 16), max(6, 3 * width // 32), max(8, width // 8))))
            use_tol = cal_cadence != "off"
            cal_every = 1 if cal_cadence == "1" else (50 if cal_cadence == "50" else 0)
            cfg = BPDConfig(
                c_per_module=sites, n_passes=n_passes, n_steps=3, batch_size=2, seq_len=20,
                seed=seed, init_svd=False, faithfulness_warmup_steps=1,
                coeff_faith=1e5, coeff_stoch=0.5, coeff_adv=0.5,
                readout_policy="readout_or_boundary", rollout_mode=regime, window_steps=max(1, n_passes // 2),
                ci=CIConfig(variant="state_only", d_model=32, n_blocks=1, n_heads=4, mlp_hidden=64,
                            max_passes=n_passes, max_seq_len=20),
                ablation=AblationConfig(mode="pass_mean" if cal_cadence != "off" else "zero",
                                        sphere_rescale=(cal_cadence != "off")),
                adversary=AdversaryConfig(inner_steps=2),
                null_calibration=NullCalibrationConfig(use_tolerance_weights=use_tol,
                                                       n_rotations_train=1, calibrate_every=cal_every),
            )
            bpd = BPDModel(model, sites, max_passes=n_passes, ablation=cfg.ablation, init_svd=False)
            adapter = TRMBPDAdapter(model)
            # Native reference Arm A: ALL caches strictly OFF
            trainer = BPDTrainer(
                bpd, cfg, adapter.rollout,
                clean_window_state_fn=adapter.sample_clean_window_state,
                visible_dim=adapter.visible_dim, prompt_dim=adapter.prompt_dim,
                device=device,
            )
            with ac_factory():
                trainer.run_faithfulness_warmup(1)
            gen = torch.Generator().manual_seed(seed + 500)
            step_losses = []
            for s in range(3):
                inp = torch.randint(1, 11, (2, 16), generator=gen)
                batch = adapter.make_batch_with_clean_cache(inp, n_passes)
                with ac_factory():
                    m = trainer.train_step(batch, s)
                step_losses.append(m.total_loss)
            # Verify all calibration buffers stay float32 and finite
            for w in trainer.bpd_model.wrappers.values():
                assert w.mean_comp_acts.dtype == torch.float32
                assert w.mean_delta_out.dtype == torch.float32
                assert w.mean_write_rms.dtype == torch.float32
                assert bool(torch.isfinite(w.mean_comp_acts).all().item())
                assert bool(torch.isfinite(w.mean_delta_out).all().item())
                assert bool(torch.isfinite(w.mean_write_rms).all().item())
            rows.append({
                "mode": mode,
                "width": width,
                "n_passes": n_passes,
                "regime": regime,
                "cal_cadence": cal_cadence,
                "steps": 3,
                "all_finite_losses": all(torch.isfinite(torch.tensor(v)).item() for v in step_losses),
                "step_losses": step_losses,
                "status": "pass",
            })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--allow-cuda", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-watts", type=float, default=60.0)
    args = p.parse_args()

    if args.device == "cuda":
        if not args.allow_cuda:
            p.error("CUDA requires --allow-cuda inside VRAMorama lease")
        snap = check_gpu_safety(min_watts=args.min_watts, min_free_mib=2048.0)
        modes = ("fp32", "tf32", "bf16", "fp16")
        widths = (128, 512)
    else:
        torch.set_num_threads(1)
        snap = {}
        modes = ("fp32", "bf16")
        widths = (128,)

    device = torch.device(args.device)
    t0 = time.perf_counter()
    unit_results = {}
    trainer_results = []
    for mode in modes:
        if args.device == "cuda":
            check_gpu_safety(min_watts=args.min_watts, min_free_mib=2048.0)
        unit_results[mode] = verify_unit_calibration(device, mode)
        for w in widths:
            trainer_results.extend(verify_native_trainer_calibration(device, mode, width=w, n_passes=6))

    if args.device == "cuda":
        configure_precision_mode("fp32")

    out = {
        "stage": "A",
        "device": args.device,
        "torch": str(torch.__version__),
        "nvml": snap,
        "unit_tests": unit_results,
        "trainer_matrix": trainer_results,
        "all_pass": all(r["status"] == "pass" for r in trainer_results)
                    and all(v["status"] == "pass" for v in unit_results.values()),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        "FUSE13_STAGE_A_DONE",
        json.dumps({"device": args.device, "modes": list(modes), "trainer_cells": len(trainer_results), "all_pass": out["all_pass"]}),
        flush=True,
    )
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
