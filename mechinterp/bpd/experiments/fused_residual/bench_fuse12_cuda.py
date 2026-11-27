"""FUSE-12: Resumable matched 7-arm CUDA benchmark, phase profiler, and parity auditor."""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
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
from mechinterp.bpd.losses import deterministic_ci_schedules
from mechinterp.bpd.trainer import BPDTrainer
from .admission import Site, select
from .validate_fuse12_cuda import (
    ATOL,
    EXIT_POWER_CAP,
    MIN_VRAM_HEADROOM_MIB,
    RTOL,
    check_gpu_safety,
    configure_precision_mode,
    query_nvml_snapshot,
    tensor_error_metrics,
)

# Matched 7-arm specification (FUSE12_PROTOCOL.md Section 3)
ARM_SPECS: dict[str, dict[str, Any]] = {
    "A": {
        "label": "Native BPD",
        "cache_single_backward_residuals": False,
        "residual_cache_policy": "first_fit",
        "cache_pgd_mask_residuals": False,
        "cache_calibration_residuals": False,
        "cache_pgd_final_residuals": False,
    },
    "B": {
        "label": "FUSE-08 stochastic step cache",
        "cache_single_backward_residuals": True,
        "residual_cache_policy": "step_exact",
        "cache_pgd_mask_residuals": False,
        "cache_calibration_residuals": False,
        "cache_pgd_final_residuals": False,
    },
    "C": {
        "label": "FUSE-09 inner PGD cache",
        "cache_single_backward_residuals": True,
        "residual_cache_policy": "step_exact",
        "cache_pgd_mask_residuals": True,
        "cache_calibration_residuals": False,
        "cache_pgd_final_residuals": False,
    },
    "D": {
        "label": "FUSE-10 calibration cache only",
        "cache_single_backward_residuals": False,
        "residual_cache_policy": "first_fit",
        "cache_pgd_mask_residuals": False,
        "cache_calibration_residuals": True,
        "cache_pgd_final_residuals": False,
    },
    "E": {
        "label": "FUSE-09 + FUSE-10",
        "cache_single_backward_residuals": True,
        "residual_cache_policy": "step_exact",
        "cache_pgd_mask_residuals": True,
        "cache_calibration_residuals": True,
        "cache_pgd_final_residuals": False,
    },
    "F": {
        "label": "FUSE-09 + FUSE-11",
        "cache_single_backward_residuals": True,
        "residual_cache_policy": "step_exact",
        "cache_pgd_mask_residuals": True,
        "cache_calibration_residuals": False,
        "cache_pgd_final_residuals": True,
    },
    "G": {
        "label": "FUSE-09 + FUSE-10 + FUSE-11",
        "cache_single_backward_residuals": True,
        "residual_cache_policy": "step_exact",
        "cache_pgd_mask_residuals": True,
        "cache_calibration_residuals": True,
        "cache_pgd_final_residuals": True,
    },
}


def build_arm_trainer(
    *,
    arm: str,
    width: int,
    n_passes: int,
    regime: str,
    pgd_inner: int,
    cal_cadence: str,
    budget_mode: str,
    seed: int,
    device: torch.device,
    precision_mode: str,
    n_steps: int,
) -> tuple[BPDTrainer, TRMBPDAdapter, dict[str, Any]]:
    spec = ARM_SPECS[arm]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

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
    paths = [f"layers.{i}.{name}" for i in range(2) for name in ("o_proj", "down_proj")]
    ranks = (
        max(2, width // 32),
        max(4, width // 16),
        max(6, (3 * width) // 32),
        max(8, width // 8),
    )
    sites = dict(zip(paths, ranks))

    use_tol = cal_cadence != "off"
    cal_every = 1 if cal_cadence == "1" else (50 if cal_cadence == "50" else 0)
    ablation_mode = "pass_mean" if cal_cadence != "off" else "zero"

    cfg = BPDConfig(
        c_per_module=sites,
        n_passes=n_passes,
        n_steps=n_steps,
        batch_size=2,
        seq_len=20,
        seed=seed,
        init_svd=False,
        faithfulness_warmup_steps=1,
        coeff_faith=1e5,
        coeff_stoch=0.5,
        coeff_adv=0.5,
        readout_policy="readout_or_boundary",
        rollout_mode=regime,
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
        ablation=AblationConfig(mode=ablation_mode, sphere_rescale=(cal_cadence != "off")),
        adversary=AdversaryConfig(inner_steps=pgd_inner),
        null_calibration=NullCalibrationConfig(
            use_tolerance_weights=use_tol,
            n_rotations_train=1,
            calibrate_every=cal_every,
        ),
    )
    bpd = BPDModel(model, sites, max_passes=n_passes, ablation=cfg.ablation, init_svd=False)
    adapter = TRMBPDAdapter(model)
    cap_bytes = sum(
        w.W_target.numel() * 4
        for name, w in bpd.wrappers.items()
        if budget_mode == "full" or name.startswith("layers.0.")
    )
    allow_cuda = device.type == "cuda"
    trainer = BPDTrainer(
        bpd,
        cfg,
        adapter.rollout,
        clean_window_state_fn=adapter.sample_clean_window_state,
        visible_dim=adapter.visible_dim,
        prompt_dim=adapter.prompt_dim,
        device=device,
        cache_single_backward_residuals=spec["cache_single_backward_residuals"],
        residual_cache_max_bytes=cap_bytes,
        residual_cache_policy=spec["residual_cache_policy"],
        cache_pgd_mask_residuals=spec["cache_pgd_mask_residuals"],
        cache_calibration_residuals=spec["cache_calibration_residuals"],
        cache_pgd_final_residuals=spec["cache_pgd_final_residuals"],
        allow_cuda_residual_cache=allow_cuda,
        residual_cache_precision_mode=precision_mode if allow_cuda else "fp32",
    )
    meta = {
        "site_dims": {
            name: {"d_in": w.d_in, "d_out": w.d_out, "rank": w.C, "payload_bytes": w.W_target.numel() * 4}
            for name, w in bpd.wrappers.items()
        },
        "cap_bytes": cap_bytes,
        "full_payload_bytes": sum(w.W_target.numel() * 4 for w in bpd.wrappers.values()),
    }
    return trainer, adapter, meta


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _snapshot_rng(device: torch.device) -> dict[str, torch.Tensor]:
    out = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        out["cuda"] = torch.cuda.get_rng_state(device)
    return out


def _restore_rng(device: torch.device, snap: dict[str, torch.Tensor]) -> None:
    torch.set_rng_state(snap["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(snap["cuda"], device)


def _install_phase_timers(trainer: BPDTrainer, device: torch.device) -> dict[str, float]:
    """Instrument trainer phases with synchronized timers for clean phase attribution.

    NOTE: Phase instrumentation is run in a dedicated profiling pass separate from
    the uninstrumented end-to-end step timing so phase syncs do not perturb total step wall time.
    """
    phases = {
        "clean_calibration_ms": 0.0,
        "residual_reconstruction_ms": 0.0,
        "inner_pgd_ms": 0.0,
        "final_pgd_ms": 0.0,
        "joint_backward_ms": 0.0,
        "optimizer_ms": 0.0,
        "planner_overhead_ms": 0.0,
    }
    orig_cal = trainer.calibrate_baselines
    orig_clean = trainer.run_clean_rollout
    orig_inner = trainer.pgd.warmup_inner_steps
    orig_opt_step = trainer.optimizer.step
    orig_scope = trainer._single_backward_residual_scope

    def timed_cal(batch):
        _sync(device)
        t0 = time.perf_counter()
        out = orig_cal(batch)
        _sync(device)
        phases["clean_calibration_ms"] += (time.perf_counter() - t0) * 1000.0
        return out

    def timed_clean(*args, **kwargs):
        _sync(device)
        t0 = time.perf_counter()
        out = orig_clean(*args, **kwargs)
        _sync(device)
        phases["clean_calibration_ms"] += (time.perf_counter() - t0) * 1000.0
        return out

    def timed_inner(*args, **kwargs):
        _sync(device)
        t0 = time.perf_counter()
        out = orig_inner(*args, **kwargs)
        _sync(device)
        phases["inner_pgd_ms"] += (time.perf_counter() - t0) * 1000.0
        return out

    def timed_opt(*args, **kwargs):
        _sync(device)
        t0 = time.perf_counter()
        out = orig_opt_step(*args, **kwargs)
        _sync(device)
        phases["optimizer_ms"] += (time.perf_counter() - t0) * 1000.0
        return out

    def timed_scope(n_passes, *, mask_only=False):
        t0 = time.perf_counter()
        cm = orig_scope(n_passes, mask_only=mask_only)
        phases["planner_overhead_ms"] += (time.perf_counter() - t0) * 1000.0
        return cm

    trainer.calibrate_baselines = timed_cal  # type: ignore[method-assign]
    trainer.run_clean_rollout = timed_clean  # type: ignore[method-assign]
    trainer.pgd.warmup_inner_steps = timed_inner  # type: ignore[method-assign]
    trainer.optimizer.step = timed_opt  # type: ignore[method-assign]
    trainer._single_backward_residual_scope = timed_scope  # type: ignore[method-assign]

    # Time raw weight_delta calls on the class wrapper instances via monkeypatch-safe hook?
    # Better: measure standalone native residual reconstruction cost directly (exact per-call basis).
    return phases


def measure_residual_reconstruction_ms(
    trainer: BPDTrainer, device: torch.device, precision_mode: str, n_iters: int = 8
) -> dict[str, float]:
    """Measure exact per-reconstruction residual cost D = W_target.float() - (V @ U).T per site."""
    ac_factory = configure_precision_mode(precision_mode) if device.type == "cuda" else (lambda: nullcontext())
    per_site: dict[str, float] = {}
    with torch.no_grad():
        with ac_factory():
            for name in trainer.bpd_model.module_order:
                w = trainer.bpd_model.wrappers[name]
                # Warmup
                for _ in range(2):
                    _ = w.weight_delta()
                _sync(device)
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    _ = w.weight_delta()
                _sync(device)
                per_site[name] = ((time.perf_counter() - t0) * 1000.0) / n_iters
    return per_site


def measure_joint_backward_and_final_pgd(
    trainer: BPDTrainer,
    batch: Any,
    step: int,
    device: torch.device,
    precision_mode: str,
) -> dict[str, float]:
    """Run one instrumented phase-profiling copy of train_step to separate every phase cleanly."""
    phases = {
        "clean_calibration_ms": 0.0,
        "inner_pgd_ms": 0.0,
        "final_pgd_ms": 0.0,
        "joint_backward_ms": 0.0,
        "optimizer_ms": 0.0,
        "planner_overhead_ms": 0.0,
    }
    ac_factory = configure_precision_mode(precision_mode) if device.type == "cuda" else (lambda: nullcontext())

    # Instrument key trainer hooks for one step
    orig_clean = trainer.run_clean_rollout
    orig_inner = trainer.pgd.warmup_inner_steps
    orig_opt = trainer.optimizer.step
    orig_scope = trainer._single_backward_residual_scope
    orig_build_sched = trainer.pgd.build_schedules
    orig_autograd_grad = torch.autograd.grad
    orig_tensor_backward = torch.Tensor.backward

    state = {"in_inner": False, "final_t0": 0.0}

    def wrapped_clean(*a, **kw):
        _sync(device)
        t0 = time.perf_counter()
        res = orig_clean(*a, **kw)
        _sync(device)
        phases["clean_calibration_ms"] += (time.perf_counter() - t0) * 1000.0
        return res

    def wrapped_inner(*a, **kw):
        _sync(device)
        t0 = time.perf_counter()
        state["in_inner"] = True
        try:
            res = orig_inner(*a, **kw)
        finally:
            state["in_inner"] = False
        _sync(device)
        phases["inner_pgd_ms"] += (time.perf_counter() - t0) * 1000.0
        return res

    def wrapped_opt(*a, **kw):
        _sync(device)
        t0 = time.perf_counter()
        res = orig_opt(*a, **kw)
        _sync(device)
        phases["optimizer_ms"] += (time.perf_counter() - t0) * 1000.0
        return res

    def wrapped_scope(n_passes, *, mask_only=False):
        t0 = time.perf_counter()
        cm = orig_scope(n_passes, mask_only=mask_only)
        phases["planner_overhead_ms"] += (time.perf_counter() - t0) * 1000.0
        return cm

    def wrapped_build_sched(*a, **kw):
        res = orig_build_sched(*a, **kw)
        if not state["in_inner"]:
            _sync(device)
            state["final_t0"] = time.perf_counter()
        return res

    def wrapped_grad(*a, **kw):
        res = orig_autograd_grad(*a, **kw)
        if kw.get("retain_graph", False) and state["final_t0"] > 0.0:
            _sync(device)
            phases["final_pgd_ms"] += (time.perf_counter() - state["final_t0"]) * 1000.0
            state["final_t0"] = 0.0
        return res

    def wrapped_backward(self, *a, **kw):
        _sync(device)
        t0 = time.perf_counter()
        res = orig_tensor_backward(self, *a, **kw)
        _sync(device)
        phases["joint_backward_ms"] += (time.perf_counter() - t0) * 1000.0
        return res

    trainer.run_clean_rollout = wrapped_clean  # type: ignore[method-assign]
    trainer.pgd.warmup_inner_steps = wrapped_inner  # type: ignore[method-assign]
    trainer.optimizer.step = wrapped_opt  # type: ignore[method-assign]
    trainer._single_backward_residual_scope = wrapped_scope  # type: ignore[method-assign]
    trainer.pgd.build_schedules = wrapped_build_sched  # type: ignore[method-assign]
    torch.autograd.grad = wrapped_grad  # type: ignore[assignment]
    torch.Tensor.backward = wrapped_backward  # type: ignore[method-assign]
    try:
        with ac_factory():
            trainer.train_step(batch, step)
    finally:
        trainer.run_clean_rollout = orig_clean  # type: ignore[method-assign]
        trainer.pgd.warmup_inner_steps = orig_inner  # type: ignore[method-assign]
        trainer.optimizer.step = orig_opt  # type: ignore[method-assign]
        trainer._single_backward_residual_scope = orig_scope  # type: ignore[method-assign]
        trainer.pgd.build_schedules = orig_build_sched  # type: ignore[method-assign]
        torch.autograd.grad = orig_autograd_grad  # type: ignore[assignment]
        torch.Tensor.backward = orig_tensor_backward  # type: ignore[method-assign]
    return phases


def compare_nested_parity(u: Any, v: Any, path: str, stats: dict[str, Any]) -> None:
    if isinstance(u, torch.Tensor):
        em = tensor_error_metrics(u, v)
        stats["tensors"] += 1
        stats["bitwise_exact"] += int(em["bitwise_equal"])
        stats["within_tol"] += int(em["within_tol"])
        group = path.split("/")[0]
        stats["group_max_abs"][group] = max(stats["group_max_abs"].get(group, 0.0), em["max_abs_err"])
        stats["group_max_rel"][group] = max(stats["group_max_rel"].get(group, 0.0), em["max_rel_err"])
        stats["max_abs_err"] = max(stats["max_abs_err"], em["max_abs_err"])
        stats["max_rel_err"] = max(stats["max_rel_err"], em["max_rel_err"])
        stats["near_zero_max_abs_err"] = max(stats["near_zero_max_abs_err"], em["near_zero_max_abs_err"])
        if not em["within_tol"] and not (not em["finite_ref"] and not em["finite_cand"] and em["nonfinite_parity"]):
            stats["failures"] += 1
            if len(stats["failure_examples"]) < 16:
                stats["failure_examples"].append({"path": path, **em})
    elif dataclasses.is_dataclass(u):
        compare_nested_parity(dataclasses.asdict(u), dataclasses.asdict(v), path, stats)
    elif isinstance(u, dict):
        assert u.keys() == v.keys(), path
        for k in u:
            compare_nested_parity(u[k], v[k], f"{path}/{k}", stats)
    elif isinstance(u, (list, tuple)):
        assert len(u) == len(v), path
        for i, (a, b) in enumerate(zip(u, v)):
            compare_nested_parity(a, b, f"{path}/{i}", stats)
    elif isinstance(u, float):
        stats["scalars"] += 1
        ok = math.isfinite(u) and math.isfinite(v) and abs(u - v) <= ATOL + RTOL * abs(v)
        stats["scalar_bitwise"] += int(u == v)
        if not ok:
            stats["failures"] += 1
            if len(stats["failure_examples"]) < 16:
                stats["failure_examples"].append({"path": path, "ref": u, "cand": v})
    else:
        stats["scalars"] += 1
        stats["scalar_bitwise"] += int(u == v)
        if u != v:
            stats["failures"] += 1
            if len(stats["failure_examples"]) < 16:
                stats["failure_examples"].append({"path": path, "ref": str(u), "cand": str(v)})


def bootstrap_ci_ratio(ref_vals: list[float], cand_vals: list[float], n_boot: int = 2000, seed: int = 4012) -> list[float] | None:
    if len(ref_vals) < 4 or len(ref_vals) != len(cand_vals):
        return None
    rng = np.random.default_rng(seed)
    ref_a = np.asarray(ref_vals, dtype=np.float64)
    cand_a = np.asarray(cand_vals, dtype=np.float64)
    n = len(ref_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    ratios = ref_a[idx].sum(axis=1) / np.maximum(cand_a[idx].sum(axis=1), 1e-12)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return [float(lo), float(hi)]


def percentile_95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), 95))


def count_reconstructions(trainer: BPDTrainer, n_passes: int, regime: str, pgd_inner: int, cal_triggered: bool) -> dict[str, int]:
    """Compute exact native and actual cached reconstruction counts for one step."""
    n_sites = len(trainer.bpd_model.module_order)
    eff_passes = n_passes if regime == "closed_loop" else max(1, n_passes // 2)
    # Training rollouts: stoch (eff_passes) + adv_final (eff_passes) + adv_inner (pgd_inner * eff_passes if closed_loop)
    train_calls_per_site = eff_passes * (2 + (pgd_inner if regime == "closed_loop" else 0))
    # Calibration clean rollout calls (only when pass_mean is enabled)
    ablation_clean = trainer.cfg.ablation.mode == "pass_mean"
    cal_calls_per_site = (n_passes if cal_triggered and ablation_clean else 0) + (eff_passes if ablation_clean else 0)
    native_train = n_sites * train_calls_per_site
    native_cal = n_sites * cal_calls_per_site
    train_hits = sum(
        st.hits
        for st in (
            trainer.residual_cache_stats
            + trainer.residual_pgd_cache_stats
            + trainer.residual_final_pgd_stats
        )
    )
    cal_hits = sum(st["hits"] for st in trainer.residual_calibration_stats)
    return {
        "native_train_reconstructions": native_train,
        "cached_train_reconstructions": native_train - train_hits,
        "train_hits": train_hits,
        "native_cal_reconstructions": native_cal,
        "cached_cal_reconstructions": native_cal - cal_hits,
        "cal_hits": cal_hits,
        "native_total_reconstructions": native_train + native_cal,
        "cached_total_reconstructions": (native_train + native_cal) - (train_hits + cal_hits),
        "total_saved_reconstructions": train_hits + cal_hits,
    }


def run_matched_cell(
    *,
    tier: str,
    width: int,
    n_passes: int,
    regime: str,
    pgd_inner: int,
    cal_cadence: str,
    budget_mode: str,
    arms: list[str],
    n_roots: int,
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
    precision_mode: str,
    check_parity: bool,
    min_watts: float,
) -> dict[str, Any]:
    """Run all requested arms on matched weights/batches/RNG with alternating order."""
    assert "A" in arms
    if device.type == "cuda":
        nvml_pre = check_gpu_safety(min_watts=min_watts)
        ac_factory = configure_precision_mode(precision_mode)
    else:
        nvml_pre = {}
        ac_factory = lambda: nullcontext()

    proc = psutil.Process(os.getpid())
    per_arm_wall: dict[str, list[float]] = {a: [] for a in arms}
    per_arm_cuda: dict[str, list[float]] = {a: [] for a in arms}
    per_arm_power: dict[str, list[float]] = {a: [] for a in arms}
    per_arm_alloc: dict[str, int] = {a: 0 for a in arms}
    per_arm_reserved: dict[str, int] = {a: 0 for a in arms}
    per_arm_payload: dict[str, int] = {a: 0 for a in arms}
    per_arm_cal_payload: dict[str, int] = {a: 0 for a in arms}
    per_arm_rss: dict[str, float] = {a: 0.0 for a in arms}
    per_arm_nvml_used: dict[str, float] = {a: 0.0 for a in arms}
    per_arm_recons: dict[str, dict[str, int]] = {}
    per_arm_admitted: dict[str, list[str]] = {}
    per_arm_saved_macs: dict[str, int] = {}
    per_arm_phases: dict[str, dict[str, float]] = {}
    parity_reports: dict[str, dict[str, Any]] = {}
    site_recon_ms: dict[str, float] = {}
    meta_ref: dict[str, Any] = {}

    total_steps = warmup_steps + timed_steps

    for root in range(n_roots):
        seed = (
            4200001
            + root * 17
            + width * 100
            + n_passes * 10000
            + pgd_inner * 1000
            + (100000 if regime == "clean_window" else 0)
            + (200000 if budget_mode == "full" else 0)
        )
        # Alternating execution order across roots
        arm_order = list(arms) if root % 2 == 0 else list(reversed(arms))

        # Generate deterministic batches once on CPU
        gen_b = torch.Generator().manual_seed(seed + 500)
        raw_inputs = [torch.randint(1, 11, (2, 16), generator=gen_b) for _ in range(total_steps)]
        step_rng_seeds = [seed + 1000 + s for s in range(total_steps)]

        # Optional parity capture on root 0
        parity_snapshots: dict[str, Any] = {}

        for arm in arm_order:
            if device.type == "cuda":
                check_gpu_safety(min_watts=min_watts)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            trainer, adapter, meta = build_arm_trainer(
                arm=arm,
                width=width,
                n_passes=n_passes,
                regime=regime,
                pgd_inner=pgd_inner,
                cal_cadence=cal_cadence,
                budget_mode=budget_mode,
                seed=seed,
                device=device,
                precision_mode=precision_mode,
                n_steps=total_steps,
            )
            if not meta_ref:
                meta_ref = meta

            # Warmup faithfulness once
            with ac_factory():
                trainer.run_faithfulness_warmup(1)

            # Pre-build clean caches on device
            batches = [adapter.make_batch_with_clean_cache(inp, n_passes) for inp in raw_inputs]

            # Tap pre-clip gradients & PGD ascent gradients ONLY during parity check (root 0, final step)
            # so timing steps have zero tap overhead!
            for s in range(total_steps):
                torch.manual_seed(step_rng_seeds[s])
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(step_rng_seeds[s])

                is_timed = s >= warmup_steps
                if device.type == "cuda" and is_timed:
                    ev_start = torch.cuda.Event(enable_timing=True)
                    ev_end = torch.cuda.Event(enable_timing=True)
                    _sync(device)
                    p_pre = query_nvml_snapshot().get("power_draw_w")
                    t0 = time.perf_counter()
                    ev_start.record()
                    with ac_factory():
                        metrics = trainer.train_step(batches[s], s)
                    ev_end.record()
                    _sync(device)
                    wall_ms = (time.perf_counter() - t0) * 1000.0
                    cuda_ms = float(ev_start.elapsed_time(ev_end))
                    snap_post = query_nvml_snapshot()
                    p_post = snap_post.get("power_draw_w")
                    per_arm_wall[arm].append(wall_ms)
                    per_arm_cuda[arm].append(cuda_ms)
                    if p_pre is not None and p_post is not None:
                        per_arm_power[arm].append(0.5 * (p_pre + p_post))
                    elif p_post is not None:
                        per_arm_power[arm].append(p_post)
                    if snap_post.get("memory_used_mib") is not None:
                        per_arm_nvml_used[arm] = max(per_arm_nvml_used[arm], float(snap_post["memory_used_mib"]))
                else:
                    _sync(device)
                    t0 = time.perf_counter()
                    with ac_factory():
                        metrics = trainer.train_step(batches[s], s)
                    _sync(device)
                    if is_timed:
                        wall_ms = (time.perf_counter() - t0) * 1000.0
                        per_arm_wall[arm].append(wall_ms)
                        per_arm_cuda[arm].append(wall_ms)

                # Record cache accounting from last step
                if s == total_steps - 1 and root == 0:
                    cal_triggered = (
                        trainer.cfg.null_calibration.use_tolerance_weights
                        and n_passes > 1
                        and (
                            s == 0
                            or (
                                trainer.cfg.null_calibration.calibrate_every > 0
                                and s % trainer.cfg.null_calibration.calibrate_every == 0
                            )
                        )
                    )
                    per_arm_recons[arm] = count_reconstructions(
                        trainer, n_passes, regime, pgd_inner, cal_triggered
                    )
                    step_peak = (
                        trainer.residual_step_receipt["peak_bytes"]
                        if trainer.residual_step_receipt
                        else 0
                    )
                    cal_peak = max((st["peak_bytes"] for st in trainer.residual_calibration_stats), default=0)
                    per_arm_payload[arm] = max(per_arm_payload[arm], step_peak)
                    per_arm_cal_payload[arm] = max(per_arm_cal_payload[arm], cal_peak)
                    if trainer.residual_cache_plans:
                        per_arm_admitted[arm] = list(trainer.residual_cache_plans[0].get("names", []))
                        per_arm_saved_macs[arm] = int(trainer.residual_cache_plans[0].get("value", 0))
                    elif trainer.residual_calibration_stats:
                        per_arm_admitted[arm] = list(trainer.residual_calibration_stats[0].get("selected", []))
                        per_arm_saved_macs[arm] = int(trainer.residual_calibration_stats[0].get("planned_saved_macs", 0))
                    else:
                        per_arm_admitted[arm] = []
                        per_arm_saved_macs[arm] = 0

                    # Verify retained PGD source-query guard on F/G when closed_loop
                    if arm in ("F", "G") and regime == "closed_loop":
                        q = trainer.residual_final_pgd_query
                        if q is not None:
                            assert q["factor_nodes_before"] == 0 and q["factor_nodes_after"] == 0, q
                            assert q["forward_sealed"], q

            if device.type == "cuda":
                per_arm_alloc[arm] = max(per_arm_alloc[arm], int(torch.cuda.max_memory_allocated(device)))
                per_arm_reserved[arm] = max(per_arm_reserved[arm], int(torch.cuda.max_memory_reserved(device)))
            per_arm_rss[arm] = max(per_arm_rss[arm], proc.memory_info().rss / (1024.0 * 1024.0))

            # Dedicated phase profiling & residual reconstruction timing on root 0
            if root == 0:
                if arm == "A" and not site_recon_ms:
                    site_recon_ms = measure_residual_reconstruction_ms(trainer, device, precision_mode)
                # Run one phase-timed step on a fresh batch
                per_arm_phases[arm] = measure_joint_backward_and_final_pgd(
                    trainer, batches[-1], total_steps, device, precision_mode
                )

            # Dedicated parity audit on root 0 (fresh matched trainers, 1 step with pre-clip & ascent taps)
            if root == 0 and check_parity:
                tr_p, ad_p, _ = build_arm_trainer(
                    arm=arm,
                    width=width,
                    n_passes=n_passes,
                    regime=regime,
                    pgd_inner=pgd_inner,
                    cal_cadence=cal_cadence,
                    budget_mode=budget_mode,
                    seed=seed,
                    device=device,
                    precision_mode=precision_mode,
                    n_steps=2,
                )
                with ac_factory():
                    tr_p.run_faithfulness_warmup(1)
                ascent_grads: list[dict[str, torch.Tensor]] = []
                preclip_grads: dict[str, torch.Tensor] = {}
                orig_ascent = tr_p.pgd.ascent_step

                def tap_ascent(grads, *, lr, _orig=orig_ascent, _dest=ascent_grads):
                    _dest.append({k: v.detach().cpu().clone() for k, v in grads.items()})
                    return _orig(grads, lr=lr)

                tr_p.pgd.ascent_step = tap_ascent  # type: ignore[method-assign]
                hooks = []
                for tag, mod in (("bpd", tr_p.bpd_model), ("ci", tr_p.ci_net)):
                    for nm, param in mod.named_parameters():
                        if param.requires_grad:
                            def _cap(p_in, _k=f"{tag}/{nm}", _dst=preclip_grads):
                                _dst[_k] = p_in.grad.detach().cpu().clone()
                            hooks.append(param.register_post_accumulate_grad_hook(_cap))

                bp = ad_p.make_batch_with_clean_cache(raw_inputs[0], n_passes)
                torch.manual_seed(step_rng_seeds[0])
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(step_rng_seeds[0])
                with ac_factory():
                    m_p = tr_p.train_step(bp, 0)
                rng_after = _snapshot_rng(device)
                for h in hooks:
                    h.remove()
                with torch.no_grad():
                    with ac_factory():
                        residuals = {k: w.weight_delta().detach().cpu().clone() for k, w in tr_p.bpd_model.wrappers.items()}
                        ev_m, ev_ci, ev_tr = tr_p.evaluate(bp)
                        sched = deterministic_ci_schedules(ev_ci.lower_leaky, delta_value=0.0)
                        pred_tr = tr_p.run_masked_rollout(bp, sched, n_passes=n_passes)
                greedy = {k: v.argmax(-1).cpu() for k, v in pred_tr.readouts.items()}
                parity_snapshots[arm] = {
                    "metrics": dataclasses.asdict(m_p),
                    "residuals": residuals,
                    "ascent_grads": ascent_grads,
                    "preclip_grads": preclip_grads,
                    "bpd_state": {k: v.detach().cpu().clone() for k, v in tr_p.bpd_model.state_dict().items()},
                    "ci_state": {k: v.detach().cpu().clone() for k, v in tr_p.ci_net.state_dict().items()},
                    "opt_state": {
                        str(k): {
                            sk: (sv.detach().cpu().clone() if isinstance(sv, torch.Tensor) else sv)
                            for sk, sv in sd.items()
                        }
                        for k, sd in tr_p.optimizer.state_dict()["state"].items()
                    },
                    "pgd_sources": {k: v.detach().cpu().clone() for k, v in tr_p.pgd.sources.items()},
                    "pass_tolerances": tr_p.pass_tolerances,
                    "eval_metrics": ev_m,
                    "pred_readouts": {k: v.detach().cpu().clone() for k, v in pred_tr.readouts.items()},
                    "pred_boundaries": {k: v.detach().cpu().clone() for k, v in pred_tr.boundary_states.items()},
                    "greedy": greedy,
                    "trainer_rng": tr_p.rng.get_state().cpu(),
                    "ambient_rng": {k: v.cpu() for k, v in rng_after.items()},
                }
                del tr_p, ad_p, bp

            del trainer, adapter, batches
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if root == 0 and check_parity and "A" in parity_snapshots:
            ref_snap = parity_snapshots["A"]
            for arm in arms:
                if arm == "A":
                    continue
                p_stats = {
                    "tensors": 0,
                    "bitwise_exact": 0,
                    "within_tol": 0,
                    "scalars": 0,
                    "scalar_bitwise": 0,
                    "failures": 0,
                    "max_abs_err": 0.0,
                    "max_rel_err": 0.0,
                    "near_zero_max_abs_err": 0.0,
                    "group_max_abs": {},
                    "group_max_rel": {},
                    "failure_examples": [],
                }
                compare_nested_parity(ref_snap, parity_snapshots[arm], "parity", p_stats)
                parity_reports[arm] = p_stats
            parity_snapshots.clear()

    # Aggregate statistics per arm relative to Arm A
    ref_wall = per_arm_wall["A"]
    ref_cuda = per_arm_cuda["A"]
    ref_wall_med = statistics.median(ref_wall)
    ref_cuda_med = statistics.median(ref_cuda)
    arm_summaries: dict[str, Any] = {}
    for arm in arms:
        w_vals = per_arm_wall[arm]
        c_vals = per_arm_cuda[arm]
        w_med = statistics.median(w_vals)
        c_med = statistics.median(c_vals)
        speedup_wall = ref_wall_med / max(w_med, 1e-12)
        speedup_cuda = ref_cuda_med / max(c_med, 1e-12)
        payload_bytes = max(per_arm_payload[arm], per_arm_cal_payload[arm])
        payload_mib = payload_bytes / (1024.0 * 1024.0)
        speedup_per_mib = (speedup_cuda - 1.0) / payload_mib if payload_mib > 0 else 0.0
        ms_saved_per_mib = (ref_cuda_med - c_med) / payload_mib if payload_mib > 0 else 0.0
        mean_power = statistics.mean(per_arm_power[arm]) if per_arm_power[arm] else None
        energy_mj = (mean_power * w_med) if mean_power is not None else None  # W * ms == mJ

        # Estimate residual reconstruction phase ms from site_recon_ms and actual recon count
        rec = per_arm_recons.get(arm, {})
        avg_site_ms = statistics.mean(site_recon_ms.values()) if site_recon_ms else 0.0
        est_recon_ms = avg_site_ms * rec.get("cached_total_reconstructions", 0)
        phases = dict(per_arm_phases.get(arm, {}))
        phases["residual_reconstruction_est_ms"] = est_recon_ms

        arm_summaries[arm] = {
            "label": ARM_SPECS[arm]["label"],
            "raw_wall_ms": w_vals,
            "raw_cuda_ms": c_vals,
            "wall_median_ms": w_med,
            "wall_p95_ms": percentile_95(w_vals),
            "cuda_median_ms": c_med,
            "cuda_p95_ms": percentile_95(c_vals),
            "speedup_wall": speedup_wall,
            "speedup_cuda": speedup_cuda,
            "speedup_cuda_ci95": bootstrap_ci_ratio(ref_cuda, c_vals) if arm != "A" else [1.0, 1.0],
            "speedup_wall_ci95": bootstrap_ci_ratio(ref_wall, w_vals) if arm != "A" else [1.0, 1.0],
            "payload_step_bytes": per_arm_payload[arm],
            "payload_cal_bytes": per_arm_cal_payload[arm],
            "payload_peak_bytes": payload_bytes,
            "payload_peak_mib": payload_mib,
            "speedup_per_mib": speedup_per_mib,
            "ms_saved_per_mib": ms_saved_per_mib,
            "peak_allocated_bytes": per_arm_alloc[arm],
            "peak_reserved_bytes": per_arm_reserved[arm],
            "nvml_used_mib": per_arm_nvml_used[arm],
            "host_rss_mib": per_arm_rss[arm],
            "mean_power_w": mean_power,
            "energy_per_step_mj": energy_mj,
            "admitted_sites": per_arm_admitted.get(arm, []),
            "planned_saved_macs": per_arm_saved_macs.get(arm, 0),
            "reconstructions": rec,
            "phases_ms": phases,
            "parity": parity_reports.get(arm),
        }

    return {
        "cell_id": f"{tier}__{precision_mode}__w{width}_p{n_passes}_{regime}_k{pgd_inner}_cal{cal_cadence}_{budget_mode}",
        "tier": tier,
        "precision_mode": precision_mode,
        "device": str(device),
        "width": width,
        "n_passes": n_passes,
        "regime": regime,
        "pgd_inner": pgd_inner,
        "cal_cadence": cal_cadence,
        "budget_mode": budget_mode,
        "n_roots": n_roots,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "source_dataset_kind": "synthetic_native_trainer",
        "site_dims": meta_ref.get("site_dims", {}),
        "site_recon_ms": site_recon_ms,
        "cap_bytes": meta_ref.get("cap_bytes", 0),
        "full_payload_bytes": meta_ref.get("full_payload_bytes", 0),
        "nvml_pre": nvml_pre,
        "arms": arm_summaries,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def build_benchmark_matrix(tiers: list[str], arms_all: list[str]) -> list[dict[str, Any]]:
    """Staged fractional factorial design in strictly ascending computational cost."""
    cells: list[dict[str, Any]] = []

    if "smoke" in tiers:
        # Widths 64, 128; 6 passes; 2 fresh paired roots; full 7 arms
        for width in (64, 128):
            cells.append(dict(tier="smoke", width=width, n_passes=6, regime="closed_loop",
                              pgd_inner=2, cal_cadence="1", budget_mode="binding",
                              arms=arms_all, n_roots=2, warmup_steps=1, timed_steps=3,
                              precision_mode="fp32", check_parity=True))
            cells.append(dict(tier="smoke", width=width, n_passes=6, regime="closed_loop",
                              pgd_inner=2, cal_cadence="off", budget_mode="full",
                              arms=arms_all, n_roots=2, warmup_steps=1, timed_steps=3,
                              precision_mode="fp32", check_parity=True))
        # Also test K=0, K=4, clean_window in smoke
        cells.append(dict(tier="smoke", width=128, n_passes=6, regime="closed_loop",
                          pgd_inner=0, cal_cadence="1", budget_mode="full",
                          arms=arms_all, n_roots=2, warmup_steps=1, timed_steps=3,
                          precision_mode="fp32", check_parity=True))
        cells.append(dict(tier="smoke", width=128, n_passes=6, regime="closed_loop",
                          pgd_inner=4, cal_cadence="off", budget_mode="binding",
                          arms=arms_all, n_roots=2, warmup_steps=1, timed_steps=3,
                          precision_mode="fp32", check_parity=True))
        cells.append(dict(tier="smoke", width=128, n_passes=6, regime="clean_window",
                          pgd_inner=2, cal_cadence="1", budget_mode="binding",
                          arms=arms_all, n_roots=2, warmup_steps=1, timed_steps=3,
                          precision_mode="fp32", check_parity=True))

    if "medium" in tiers:
        # Widths 256, 512; passes 6 & 12; 4 paired roots; staged fractional factorial
        medium_design = [
            (256, 6, "closed_loop", 2, "1", "binding"),
            (256, 6, "closed_loop", 2, "off", "full"),
            (256, 12, "closed_loop", 0, "50", "binding"),
            (256, 12, "closed_loop", 4, "1", "full"),
            (256, 12, "clean_window", 2, "1", "binding"),
            (512, 6, "closed_loop", 2, "1", "binding"),
            (512, 6, "closed_loop", 2, "off", "full"),
            (512, 6, "closed_loop", 4, "50", "full"),
            (512, 12, "closed_loop", 0, "1", "full"),
            (512, 12, "closed_loop", 2, "1", "binding"),
            (512, 12, "closed_loop", 2, "off", "full"),
            (512, 12, "closed_loop", 4, "off", "binding"),
            (512, 12, "clean_window", 2, "1", "full"),
        ]
        for w, p, reg, k, cal, bud in medium_design:
            cells.append(dict(tier="medium", width=w, n_passes=p, regime=reg,
                              pgd_inner=k, cal_cadence=cal, budget_mode=bud,
                              arms=arms_all, n_roots=4, warmup_steps=1, timed_steps=3,
                              precision_mode="fp32", check_parity=(w == 256 or (w == 512 and p == 6 and k == 2))))

    if "wide" in tiers:
        # Widths 1024, 2048; passes 6 & 12; 4 paired roots; staged fractional factorial
        wide_design = [
            (1024, 6, "closed_loop", 2, "1", "binding"),
            (1024, 6, "closed_loop", 2, "off", "full"),
            (1024, 12, "closed_loop", 0, "50", "full"),
            (1024, 12, "closed_loop", 2, "1", "binding"),
            (1024, 12, "closed_loop", 2, "off", "full"),
            (1024, 12, "closed_loop", 4, "off", "full"),
            (1024, 12, "clean_window", 2, "1", "binding"),
            (2048, 6, "closed_loop", 2, "1", "binding"),
            (2048, 6, "closed_loop", 2, "off", "full"),
            (2048, 12, "closed_loop", 2, "1", "binding"),
            (2048, 12, "closed_loop", 2, "off", "full"),
            (2048, 12, "closed_loop", 4, "50", "full"),
            (2048, 12, "clean_window", 2, "1", "full"),
        ]
        for w, p, reg, k, cal, bud in wide_design:
            cells.append(dict(tier="wide", width=w, n_passes=p, regime=reg,
                              pgd_inner=k, cal_cadence=cal, budget_mode=bud,
                              arms=arms_all, n_roots=4, warmup_steps=1, timed_steps=2,
                              precision_mode="fp32", check_parity=(p == 6 and cal == "1")))

    if "frontier" in tiers:
        # Width 2048 / 4096 at 24 passes, strictly gated by >=2 GiB WDDM headroom
        frontier_arms = ["A", "C", "F", "G"]
        for w, k, cal, bud in [
            (2048, 2, "off", "binding"),
            (2048, 2, "off", "full"),
            (2048, 4, "1", "full"),
            (4096, 2, "off", "binding"),
            (4096, 2, "off", "full"),
        ]:
            cells.append(dict(tier="frontier", width=w, n_passes=24, regime="closed_loop",
                              pgd_inner=k, cal_cadence=cal, budget_mode=bud,
                              arms=frontier_arms, n_roots=2, warmup_steps=1, timed_steps=2,
                              precision_mode="fp32", check_parity=(w == 2048 and bud == "binding")))

    if "mixed_precision" in tiers:
        # FUSE-12B: separate registered arms for tf32, bf16, fp16
        mp_arms = ["A", "C", "F", "G"]
        for pm in ("tf32", "bf16", "fp16"):
            for w, p, k, cal, bud in [
                (128, 6, 2, "1", "full"),
                (512, 12, 2, "off", "full"),
                (1024, 12, 2, "1", "binding"),
                (2048, 12, 2, "off", "full"),
            ]:
                cells.append(dict(tier="mixed_precision", width=w, n_passes=p, regime="closed_loop",
                                  pgd_inner=k, cal_cadence=cal, budget_mode=bud,
                                  arms=mp_arms, n_roots=2, warmup_steps=1, timed_steps=2,
                                  precision_mode=pm, check_parity=(w in (128, 512))))

    return cells


def estimate_cell_vram_mib(width: int, n_passes: int) -> float:
    """Conservative upper-bound estimate of VRAM needed so we never breach 2 GiB headroom."""
    # 4 sites + target weights + activations + optimizer moments + cached residuals
    # Width 2048 full payload ~120 MiB; width 4096 full payload ~480 MiB + ~2.5 GiB activations at 24 passes
    weight_mib = (width * width * 4 * 24) / (1024.0 * 1024.0)
    act_mib = (width * n_passes * 2 * 20 * 4 * 24) / (1024.0 * 1024.0)
    return weight_mib + act_mib + 512.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL receipt file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tiers",
        default="smoke,medium,wide,frontier,mixed_precision",
        help="Comma-separated tiers to run",
    )
    parser.add_argument("--arms", default="A,B,C,D,E,F,G")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-watts", type=float, default=40.0)
    parser.add_argument("--min-headroom-mib", type=float, default=MIN_VRAM_HEADROOM_MIB)
    parser.add_argument("--max-cells", type=int, default=0, help="Optional cap for quick dry runs")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device.type == "cpu":
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    cells = build_benchmark_matrix(tiers, arms)
    if args.max_cells > 0:
        cells = cells[: args.max_cells]

    completed: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if "cell_id" in rec:
                    completed.add(rec["cell_id"])
            except json.JSONDecodeError:
                continue

    fp32_parity_failed = False
    frontier_aborted = False

    for idx, cell in enumerate(cells):
        cid = (
            f"{cell['tier']}__{cell['precision_mode']}__w{cell['width']}"
            f"_p{cell['n_passes']}_{cell['regime']}_k{cell['pgd_inner']}"
            f"_cal{cell['cal_cadence']}_{cell['budget_mode']}"
        )
        if cid in completed:
            print(f"[resume] skipping completed {cid}", flush=True)
            continue

        # Gate FUSE-12B strictly on FP32 parity pass
        if cell["tier"] == "mixed_precision" and fp32_parity_failed:
            rec = {"cell_id": cid, "status": "skipped_fp32_parity_failed", **cell}
            with args.output.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, allow_nan=False) + "\n")
            continue

        # WDDM VRAM headroom safety guard before escalating
        if device.type == "cuda":
            snap = check_gpu_safety(min_watts=args.min_watts, min_free_mib=args.min_headroom_mib)
            free_mib = snap.get("memory_free_mib")
            est_mib = estimate_cell_vram_mib(cell["width"], cell["n_passes"])
            if frontier_aborted and cell["tier"] == "frontier":
                rec = {"cell_id": cid, "status": "skipped_vram_headroom", "est_vram_mib": est_mib, **cell}
                with args.output.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, allow_nan=False) + "\n")
                continue
            if free_mib is not None and (free_mib - est_mib) < args.min_headroom_mib:
                print(
                    f"[vram-guard] skipping {cid}: free={free_mib:.0f} MiB, est={est_mib:.0f} MiB, "
                    f"headroom_floor={args.min_headroom_mib:.0f} MiB",
                    flush=True,
                )
                if cell["tier"] == "frontier":
                    frontier_aborted = True
                rec = {
                    "cell_id": cid,
                    "status": "skipped_vram_headroom",
                    "free_mib": free_mib,
                    "est_vram_mib": est_mib,
                    **cell,
                }
                with args.output.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, allow_nan=False) + "\n")
                continue

        print(f"[{idx + 1}/{len(cells)}] running {cid} ...", flush=True)
        try:
            rec = run_matched_cell(device=device, min_watts=args.min_watts, **cell)
            rec["status"] = "ok"
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            gc.collect()
            if cell["tier"] == "frontier":
                frontier_aborted = True
            rec = {"cell_id": cid, "status": "oom_skipped", "error": str(exc), **cell}

        with args.output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, allow_nan=False) + "\n")
        completed.add(cid)

        if rec.get("status") == "ok" and cell["precision_mode"] == "fp32":
            for arm_k, arm_v in rec["arms"].items():
                p = arm_v.get("parity")
                if p is not None and p.get("failures", 0) > 0:
                    fp32_parity_failed = True
                    print(f"[PARITY FAILURE] {cid} arm={arm_k} failures={p['failures']}", flush=True)

        if rec.get("status") == "ok":
            g = rec["arms"].get("G") or rec["arms"].get("F") or rec["arms"].get("C")
            print(
                f"  -> ok | A={rec['arms']['A']['cuda_median_ms']:.2f}ms "
                f"| best={g['cuda_median_ms']:.2f}ms ({g['speedup_cuda']:.3f}x) "
                f"| payload={g['payload_peak_mib']:.2f}MiB",
                flush=True,
            )

    print("FUSE12_BENCH_DONE", json.dumps({"cells_completed": len(completed), "fp32_parity_failed": fp32_parity_failed}), flush=True)


if __name__ == "__main__":
    main()
