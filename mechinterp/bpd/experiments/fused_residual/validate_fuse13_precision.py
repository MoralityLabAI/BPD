"""FUSE-13 Stages B & C: Precision-aware residual storage ablation, 96-case stress,
overflow/GradScaler parity, and full-trainer matched parity across FP32/TF32/BF16/FP16."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import itertools
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mechinterp.bpd.adapters.trm import TRMBPDAdapter, build_mini_trm
from mechinterp.bpd.components import BPDModel, PassMaskSchedule, TiedComponentLinear
from mechinterp.bpd.config import (
    AblationConfig,
    AdversaryConfig,
    BPDConfig,
    CIConfig,
    NullCalibrationConfig,
)
from mechinterp.bpd.losses import deterministic_ci_schedules
from mechinterp.bpd.trainer import BPDTrainer
from .adjoint_cache import _Primal, _SharedCastPrimal, _resolve_autocast_factors
from .step_cache import StepResidualPool
from .validate_fuse12_cuda import (
    ATOL,
    EXIT_POWER_CAP,
    RTOL,
    check_gpu_safety,
    configure_precision_mode,
    query_nvml_snapshot,
)


def byte_and_overflow_compare(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, Any]:
    """True byte equality (uint8 view), finite tolerance, and signed overflow/NaN mask parity."""
    assert ref.shape == cand.shape and ref.dtype == cand.dtype, (ref.shape, cand.shape, ref.dtype, cand.dtype)
    a = ref.detach().contiguous()
    b = cand.detach().contiguous()
    bitwise = bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
    if not (a.is_floating_point() or a.is_complex()):
        return {
            "bitwise": bitwise,
            "finite": True,
            "finite_bitwise": bitwise,
            "within_tol": bitwise,
            "overflow_parity": bitwise,
            "max_abs": 0.0,
            "nan_masks_equal": True,
            "posinf_masks_equal": True,
            "neginf_masks_equal": True,
        }
    af, bf = a.float(), b.float()
    fin_a = bool(torch.isfinite(af).all().item())
    fin_b = bool(torch.isfinite(bf).all().item())
    finite = fin_a and fin_b
    nan_eq = bool(torch.equal(torch.isnan(af), torch.isnan(bf)))
    pinf_eq = bool(torch.equal(torch.isposinf(af), torch.isposinf(bf)))
    ninf_eq = bool(torch.equal(torch.isneginf(af), torch.isneginf(bf)))
    if finite:
        diff = (af - bf).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        within_tol = bool(torch.allclose(af, bf, atol=ATOL, rtol=RTOL))
        return {
            "bitwise": bitwise,
            "finite": True,
            "finite_bitwise": bitwise,
            "within_tol": within_tol,
            "overflow_parity": True,
            "max_abs": max_abs,
            "nan_masks_equal": True,
            "posinf_masks_equal": True,
            "neginf_masks_equal": True,
        }
    # Separate overflow accounting: NEVER count nonfinite match as finite_bitwise or within_tol!
    fin_mask = torch.isfinite(af) & torch.isfinite(bf)
    if bool(fin_mask.any().item()):
        fin_diff = (af[fin_mask] - bf[fin_mask]).abs()
        max_abs_fin = float(fin_diff.max().item())
        fin_close = bool(torch.allclose(af[fin_mask], bf[fin_mask], atol=ATOL, rtol=RTOL))
    else:
        max_abs_fin = 0.0
        fin_close = True
    return {
        "bitwise": bitwise,
        "finite": False,
        "finite_bitwise": False,
        "within_tol": False,
        "overflow_parity": nan_eq and pinf_eq and ninf_eq and fin_close,
        "max_abs": max_abs_fin,
        "nan_masks_equal": nan_eq,
        "posinf_masks_equal": pinf_eq,
        "neginf_masks_equal": ninf_eq,
    }


def _ac_factory(device: torch.device, mode: str):
    if device.type == "cuda":
        return configure_precision_mode(mode)
    if mode == "fp32":
        torch.set_float32_matmul_precision("highest")
        return lambda: nullcontext()
    if mode == "bf16":
        torch.set_float32_matmul_precision("highest")
        return lambda: torch.autocast("cpu", enabled=True, dtype=torch.bfloat16)
    raise ValueError(f"Unsupported mode {mode!r} on {device.type}")


def stage_b_six_strategy_ablation(
    device: torch.device, mode: str, width: int = 128, passes: int = 12, seed: int = 4013001
) -> dict[str, Any]:
    """Compare all 6 Stage-B strategies inside TiedComponentLinear:
    1. native
    2. shared_fp32
    3. shared_compute_per_use_cast
    4. shared_compute_shared_cast (FUSE-13 candidate via _resolve_autocast_factors)
    5. shared_factor_casts_only
    6. native_graph_replay
    """
    ac_fn = _ac_factory(device, mode)
    comp_dt = {"fp32": torch.float32, "tf32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[mode]
    rank = max(8, width // 8)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    base_w = TiedComponentLinear(
        nn.Linear(width, width), rank, max_passes=passes,
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        init_svd=False,
    ).to(device)
    xs = [torch.randn(2, 8, width, device=device) for _ in range(passes)]
    ms = torch.randn(2, passes, 8, rank, device=device).sigmoid()
    ds = torch.randn(2, passes, 8, device=device).sigmoid()
    cots = [torch.randn(2, 8, width, device=device) for _ in range(passes)]
    with torch.no_grad(), ac_fn():
        for j in range(passes):
            base_w.observe_clean_pass(xs[j], j)

    class _ReplayCastPrimal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, vc, uc, payload):
            ctx.save_for_backward(vc, uc)
            ctx.dev_type = vc.device.type
            return payload

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, g):
            vc, uc = ctx.saved_tensors
            with torch.autocast(ctx.dev_type, enabled=False), torch.enable_grad():
                a = vc.detach().requires_grad_(True)
                b = uc.detach().requires_grad_(True)
                z = -(a @ b).T
                dz = g.float().to(vc.dtype)
                ga, gb = torch.autograd.grad(z, (a, b), dz)
            return ga, gb, None

    def run_strategy(strategy: str):
        q = copy.deepcopy(base_w)
        q.mode = "component"
        q.reset_pass_state()
        q.schedule = PassMaskSchedule(ms, ds)
        inps = [x.clone().requires_grad_(True) for x in xs]
        saved_sq: dict[int, tuple[int, str]] = {}
        saved_sq_calls = 0
        recons = 0

        def pack(t):
            nonlocal saved_sq_calls
            if t.ndim == 2 and tuple(t.shape) == (width, width):
                saved_sq_calls += 1
                s = t.untyped_storage()
                saved_sq[s.data_ptr()] = (s.nbytes(), str(t.dtype))
            return t

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        orig_wd = q.weight_delta
        dummy_stats = type("_S", (), {"backward_nodes": 0, "backward_replays": 0})()
        cached_p: list[torch.Tensor] = []

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            with ac_fn():
                if strategy == "native":
                    def wd():
                        nonlocal recons
                        recons += 1
                        return orig_wd()
                elif strategy == "shared_fp32":
                    def wd():
                        nonlocal recons
                        if not cached_p:
                            with torch.no_grad():
                                cached_p.append(orig_wd())
                            recons += 1
                        return _Primal.apply(q.V, q.U, cached_p[0], "residual", dummy_stats, False, None)
                elif strategy == "shared_compute_per_use_cast":
                    def wd():
                        nonlocal recons
                        if not cached_p:
                            with torch.no_grad():
                                cached_p.append(orig_wd().to(comp_dt))
                            recons += 1
                        vc, uc = q.V.to(comp_dt), q.U.to(comp_dt)
                        return _SharedCastPrimal.apply(vc, uc, cached_p[0], dummy_stats)
                elif strategy == "shared_compute_shared_cast":
                    def wd():
                        nonlocal recons
                        if not cached_p:
                            with torch.no_grad():
                                cached_p.append(orig_wd().to(comp_dt))
                            recons += 1
                        vc, uc = _resolve_autocast_factors(q.V, q.U, comp_dt)
                        return _SharedCastPrimal.apply(vc, uc, cached_p[0], dummy_stats)
                elif strategy == "shared_factor_casts_only":
                    def wd():
                        nonlocal recons
                        recons += 1
                        vc, uc = _resolve_autocast_factors(q.V, q.U, comp_dt)
                        return (q.W_target.float() - (vc @ uc).T).to(comp_dt)
                elif strategy == "native_graph_replay":
                    def wd():
                        nonlocal recons
                        if not cached_p:
                            with torch.no_grad():
                                cached_p.append(orig_wd().to(comp_dt))
                            recons += 1
                        vc, uc = _resolve_autocast_factors(q.V, q.U, comp_dt)
                        return _ReplayCastPrimal.apply(vc, uc, cached_p[0])
                else:
                    raise ValueError(strategy)

                q.weight_delta = wd  # type: ignore[method-assign]
                outs = [q(x) for x in inps]
                loss = sum((y.float() * c).sum() for y, c in zip(outs, cots))
            loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_alloc = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_alloc = None

        tensors = {
            **{f"y{i}": o.detach().cpu() for i, o in enumerate(outs)},
            **{f"gx{i}": x.grad.detach().cpu() for i, x in enumerate(inps)},
            "g_V": q.V.grad.detach().cpu(),
            "g_U": q.U.grad.detach().cpu(),
        }
        stats = {
            "reconstructions": recons,
            "saved_square_calls": saved_sq_calls,
            "unique_square_storages": len(saved_sq),
            "unique_square_bytes": sum(v[0] for v in saved_sq.values()),
            "saved_square_dtypes": sorted({v[1] for v in saved_sq.values()}),
            "peak_allocated_bytes": peak_alloc,
        }
        return tensors, stats

    ref_t, ref_s = run_strategy("native")
    strategies = {}
    for strat in (
        "shared_fp32",
        "shared_compute_per_use_cast",
        "shared_compute_shared_cast",
        "shared_factor_casts_only",
        "native_graph_replay",
    ):
        cand_t, cand_s = run_strategy(strat)
        comps = {k: byte_and_overflow_compare(ref_t[k], cand_t[k]) for k in ref_t}
        strategies[strat] = {
            "storage": cand_s,
            "finite_bitwise": all(c["finite_bitwise"] for c in comps.values()),
            "within_tol": all(c["within_tol"] for c in comps.values()),
            "max_abs": max(c["max_abs"] for c in comps.values()),
            "g_V_bitwise": comps["g_V"]["finite_bitwise"],
            "g_U_bitwise": comps["g_U"]["finite_bitwise"],
            "g_V_max_abs": comps["g_V"]["max_abs"],
            "g_U_max_abs": comps["g_U"]["max_abs"],
        }

    return {
        "mode": mode,
        "width": width,
        "passes": passes,
        "seed": seed,
        "native_storage": ref_s,
        "strategies": strategies,
    }


def stage_c_stress_suite(
    device: torch.device,
    mode: str,
    roots: tuple[int, ...],
    storage_mode: str = "compute_dtype",
) -> dict[str, Any]:
    """Stress suite testing StepResidualPool(storage_mode=...) across scales & targets."""
    ac_fn = _ac_factory(device, mode)
    total_cases = 0
    total_tensors = 0
    finite_bitwise = 0
    within_tol = 0
    overflow_parity_tensors = 0
    failures = 0
    max_abs = 0.0
    total_joint_nodes = 0
    total_hits = 0
    cells = []
    failure_examples = []

    for scale, target in itertools.product((1.0, 10.0, 100.0, 10000.0), ("ordinary", "exact", "near")):
        cell = {
            "mode": mode,
            "storage_mode": storage_mode,
            "scale": scale,
            "target": target,
            "cases": 0,
            "tensors": 0,
            "finite_bitwise": 0,
            "within_tol": 0,
            "overflow_parity_tensors": 0,
            "failures": 0,
            "max_abs": 0.0,
        }
        for seed in roots:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            w = TiedComponentLinear(
                nn.Linear(64, 64), 32, max_passes=3,
                ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
                init_svd=False,
            ).to(device)
            with torch.no_grad():
                w.V.mul_(scale)
                w.U.mul_(scale)
                if target != "ordinary":
                    w.W_target.copy_(
                        w.reconstructed_weight()
                        + (torch.randn_like(w.W_target) * 1e-4 if target == "near" else 0.0)
                    )
                x = torch.randn(3, 3, 1, 8, 64, device=device)
                m = torch.randn(3, 1, 3, 8, 32, device=device)
                d = torch.randn(3, 1, 3, 8, device=device)
                cot = torch.randn_like(x)
                with ac_fn():
                    for j in range(3):
                        w.observe_clean_pass(x[0, j], j)

            def run(fast: bool):
                q = copy.deepcopy(w)
                q.mode = "component"
                p = (
                    StepResidualPool(
                        [q],
                        max_bytes=q.W_target.numel() * 4,
                        allow_cuda=(device.type == "cuda"),
                        precision_mode=mode,
                        storage_mode=storage_mode,
                    )
                    if fast
                    else None
                )
                stats, vals, losses, leaves = [], [], [], []
                try:
                    with ac_fn():
                        for j in range(3):
                            xx = x[j].clone().requires_grad_()
                            mm = m[j].clone().requires_grad_()
                            dd = d[j].clone().requires_grad_()
                            q.reset_pass_state()
                            q.schedule = PassMaskSchedule(mm.sigmoid(), dd.sigmoid())
                            with (p.scope(mask_only=(j == 1)) if fast else nullcontext(None)) as st:
                                yy = torch.stack([q(xx[i]) for i in range(3)])
                            loss = (yy * cot[j]).sum()
                            vals.append(yy.detach())
                            stats.append(st)
                            if j == 1:
                                g = torch.autograd.grad(loss, (mm, dd))
                                vals.extend(t.detach() for t in g)
                            else:
                                losses.append(loss)
                                leaves.extend((xx, mm, dd))
                                if j == 2:
                                    if fast:
                                        p.seal_forward()
                                        assert stats[0].backward_nodes == stats[2].backward_nodes == 0
                                    g = torch.autograd.grad(loss, (mm, dd), retain_graph=True)
                                    vals.extend(t.detach() for t in g)
                                    if fast:
                                        assert stats[0].backward_nodes == stats[2].backward_nodes == 0
                        total = losses[0] + 0.5 * losses[1] + 0.01 * q.weight_delta().square().mean()
                    total.backward()
                    vals.extend(t.grad.detach().clone() for t in (q.V, q.U, *leaves))
                    if fast:
                        assert [s.backward_nodes for s in stats] == [3, 0, 3], [s.backward_nodes for s in stats]
                        assert [s.hits for s in stats] == [2, 3, 3], [s.hits for s in stats]
                        assert p.sealed and not p.poisoned
                        expected_n = q.W_target.numel() * p.storage_dtype.itemsize
                        assert p.peak_bytes == expected_n, (p.peak_bytes, expected_n)
                    return vals, stats
                finally:
                    if p is not None:
                        p.close()
                        assert p.closed and p.resident_bytes == 0
                        assert "weight_delta" not in q.__dict__

            ref_vals, _ = run(False)
            fast_vals, ss = run(True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            names = [
                "out_stoch", "out_inner", "g_inner_m", "g_inner_d",
                "out_final", "g_final_m", "g_final_d",
                "g_V", "g_U", "g_x0", "g_m0", "g_d0", "g_x2", "g_m2", "g_d2",
            ]
            for nm, u, v in zip(names, ref_vals, fast_vals):
                cm = byte_and_overflow_compare(u, v)
                total_tensors += 1
                cell["tensors"] += 1
                finite_bitwise += int(cm["finite_bitwise"])
                cell["finite_bitwise"] += int(cm["finite_bitwise"])
                within_tol += int(cm["within_tol"])
                cell["within_tol"] += int(cm["within_tol"])
                max_abs = max(max_abs, cm["max_abs"])
                cell["max_abs"] = max(cell["max_abs"], cm["max_abs"])
                if not cm["finite"]:
                    if cm["overflow_parity"]:
                        overflow_parity_tensors += 1
                        cell["overflow_parity_tensors"] += 1
                    else:
                        failures += 1
                        cell["failures"] += 1
                        if len(failure_examples) < 12:
                            failure_examples.append({"seed": seed, "scale": scale, "target": target, "tensor": nm, **cm})
                elif not cm["finite_bitwise"]:
                    failures += 1
                    cell["failures"] += 1
                    if len(failure_examples) < 12:
                        failure_examples.append({"seed": seed, "scale": scale, "target": target, "tensor": nm, **cm})
            total_cases += 1
            cell["cases"] += 1
            total_joint_nodes += sum(s.backward_nodes for s in ss)
            total_hits += sum(s.hits for s in ss)
        cells.append(cell)

    return {
        "mode": mode,
        "storage_mode": storage_mode,
        "roots": list(roots),
        "cases": total_cases,
        "tensors": total_tensors,
        "finite_bitwise": finite_bitwise,
        "within_tol": within_tol,
        "overflow_parity_tensors": overflow_parity_tensors,
        "failures": failures,
        "max_abs": max_abs,
        "joint_factor_nodes": total_joint_nodes,
        "hits": total_hits,
        "admission_pass": (failures == 0 and overflow_parity_tensors == 0 and finite_bitwise == total_tensors),
        "failure_examples": failure_examples,
        "cells": cells,
    }


def stage_c_fp16_gradscaler_parity(device: torch.device) -> dict[str, Any]:
    """Verify FP16 dynamic GradScaler skip/update parity under normal and overflow scales."""
    ac_fn = _ac_factory(device, "fp16")
    results = []
    for label, init_scale, factor_scale in (("safe_update", 64.0, 1.0), ("overflow_skip", 65536.0, 100.0)):
        torch.manual_seed(4013901)
        torch.cuda.manual_seed_all(4013901)
        w0 = TiedComponentLinear(
            nn.Linear(64, 64), 16, max_passes=3,
            ablation=AblationConfig(mode="zero", sphere_rescale=False),
            init_svd=False,
        ).to(device)
        with torch.no_grad():
            w0.V.mul_(factor_scale)
            w0.U.mul_(factor_scale)
        xs = [torch.randn(2, 8, 64, device=device) for _ in range(3)]
        ms = torch.randn(2, 3, 8, 16, device=device).sigmoid()
        ds = torch.randn(2, 3, 8, device=device).sigmoid()
        cots = [torch.randn(2, 8, 64, device=device) for _ in range(3)]

        def run_scaled(fast: bool):
            q = copy.deepcopy(w0)
            q.mode = "component"
            opt = torch.optim.AdamW([q.V, q.U], lr=1e-3)
            scaler = torch.amp.GradScaler("cuda", init_scale=init_scale)
            p = (
                StepResidualPool([q], max_bytes=q.W_target.numel() * 2,
                                 allow_cuda=True, precision_mode="fp16", storage_mode="compute_dtype")
                if fast else None
            )
            opt.zero_grad(set_to_none=True)
            q.reset_pass_state()
            q.schedule = PassMaskSchedule(ms, ds)
            try:
                with ac_fn():
                    with (p.scope() if fast else nullcontext(None)):
                        ys = [q(x) for x in xs]
                    loss = sum((y.float() * c).sum() for y, c in zip(ys, cots))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scale_before = float(scaler.get_scale())
                scaler.update()
                scale_after = float(scaler.get_scale())
            finally:
                if p is not None:
                    p.close()
            skipped = scale_after < scale_before
            return {
                "V": q.V.detach().cpu(),
                "U": q.U.detach().cpu(),
                "scale_before": scale_before,
                "scale_after": scale_after,
                "skipped": skipped,
            }

        ref = run_scaled(False)
        cand = run_scaled(True)
        cv = byte_and_overflow_compare(ref["V"], cand["V"])
        cu = byte_and_overflow_compare(ref["U"], cand["U"])
        results.append({
            "case": label,
            "ref_skipped": ref["skipped"],
            "cand_skipped": cand["skipped"],
            "skip_decision_match": ref["skipped"] == cand["skipped"],
            "scale_after_match": ref["scale_after"] == cand["scale_after"],
            "V_bitwise": cv["finite_bitwise"],
            "U_bitwise": cu["finite_bitwise"],
            "V_max_abs": cv["max_abs"],
            "U_max_abs": cu["max_abs"],
        })
    return {
        "cases": results,
        "all_match": all(r["skip_decision_match"] and r["scale_after_match"] and r["V_bitwise"] and r["U_bitwise"] for r in results),
    }


def stage_c_trainer_parity(
    device: torch.device,
    mode: str,
    width: int,
    n_passes: int,
    cal_cadence: str,
    budget_mode: str,
    seed: int,
    regime: str = "closed_loop",
) -> dict[str, Any]:
    """Full matched native Arm A vs Arm G (compute_dtype) parity + VRAM + timing."""
    ac_fn = _ac_factory(device, mode)
    elem_bytes = {"fp32": 4, "tf32": 4, "bf16": 2, "fp16": 2}[mode]

    def build_tr(arm_g: bool):
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
        cap_bytes = sum(
            w.W_target.numel() * elem_bytes
            for nm, w in bpd.wrappers.items()
            if budget_mode == "full" or nm.startswith("layers.0.")
        )
        tr = BPDTrainer(
            bpd, cfg, adapter.rollout,
            clean_window_state_fn=adapter.sample_clean_window_state,
            visible_dim=adapter.visible_dim, prompt_dim=adapter.prompt_dim,
            device=device,
            cache_single_backward_residuals=arm_g,
            residual_cache_max_bytes=cap_bytes,
            residual_cache_policy="step_exact" if arm_g else "first_fit",
            cache_pgd_mask_residuals=arm_g,
            cache_calibration_residuals=arm_g,
            cache_pgd_final_residuals=arm_g,
            allow_cuda_residual_cache=(device.type == "cuda"),
            residual_cache_precision_mode=mode if device.type == "cuda" else "fp32",
            residual_cache_storage_mode="compute_dtype" if (device.type == "cuda" and arm_g) else "fp32",
        )
        return tr, adapter, cap_bytes

    gen_b = torch.Generator().manual_seed(seed + 500)
    raw_inps = [torch.randint(1, 11, (2, 16), generator=gen_b) for _ in range(2)]

    snaps = {}
    metrics_out = {}
    for arm_name, is_g in (("A", False), ("G", True)):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        tr, ad, cap_b = build_tr(is_g)
        with ac_fn():
            tr.run_faithfulness_warmup(1)
        ascent_grads: list[dict[str, torch.Tensor]] = []
        preclip_grads: dict[str, torch.Tensor] = {}
        orig_ascent = tr.pgd.ascent_step

        def tap_ascent(grads, *, lr, _o=orig_ascent, _d=ascent_grads):
            _d.append({k: v.detach().cpu().clone() for k, v in grads.items()})
            return _o(grads, lr=lr)

        tr.pgd.ascent_step = tap_ascent  # type: ignore[method-assign]
        hooks = []
        for tag, mod in (("bpd", tr.bpd_model), ("ci", tr.ci_net)):
            for nm, param in mod.named_parameters():
                if param.requires_grad:
                    def _cap(pin, _k=f"{tag}/{nm}", _dst=preclip_grads):
                        _dst[_k] = pin.grad.detach().cpu().clone()
                    hooks.append(param.register_post_accumulate_grad_hook(_cap))

        step_ms = []
        for s in range(2):
            bp = ad.make_batch_with_clean_cache(raw_inps[s], n_passes)
            torch.manual_seed(seed + 1000 + s)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + 1000 + s)
            if device.type == "cuda":
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                ev0.record()
                with ac_fn():
                    m_step = tr.train_step(bp, s)
                ev1.record()
                torch.cuda.synchronize(device)
                step_ms.append(float(ev0.elapsed_time(ev1)))
            else:
                t0 = time.perf_counter()
                with ac_fn():
                    m_step = tr.train_step(bp, s)
                step_ms.append((time.perf_counter() - t0) * 1000.0)
            if is_g and tr.residual_final_pgd_query is not None:
                q_rec = tr.residual_final_pgd_query
                assert q_rec["factor_nodes_before"] == 0 and q_rec["factor_nodes_after"] == 0

        for h in hooks:
            h.remove()
        with torch.no_grad(), ac_fn():
            ev_m, ev_ci, _ = tr.evaluate(bp)
            sched = deterministic_ci_schedules(ev_ci.lower_leaky, delta_value=0.0)
            pred_tr = tr.run_masked_rollout(bp, sched, n_passes=n_passes)
        peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        peak_res = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        nvml_used = query_nvml_snapshot().get("memory_used_mib") if device.type == "cuda" else 0.0

        snaps[arm_name] = {
            "metrics": dataclasses.asdict(m_step),
            "ascent_grads": ascent_grads,
            "preclip_grads": preclip_grads,
            "bpd_state": {k: v.detach().cpu().clone() for k, v in tr.bpd_model.state_dict().items()},
            "ci_state": {k: v.detach().cpu().clone() for k, v in tr.ci_net.state_dict().items()},
            "opt_state": {
                str(k): {sk: (sv.detach().cpu().clone() if isinstance(sv, torch.Tensor) else sv)
                         for sk, sv in sd.items()}
                for k, sd in tr.optimizer.state_dict()["state"].items()
            },
            "pgd_sources": {k: v.detach().cpu().clone() for k, v in tr.pgd.sources.items()},
            "pass_tolerances": tr.pass_tolerances,
            "eval_metrics": ev_m,
            "pred_readouts": {k: v.detach().cpu().clone() for k, v in pred_tr.readouts.items()},
            "greedy": {k: v.argmax(-1).cpu() for k, v in pred_tr.readouts.items()},
            "trainer_rng": tr.rng.get_state().cpu(),
        }
        metrics_out[arm_name] = {
            "step_ms": step_ms,
            "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_res,
            "nvml_used_mib": nvml_used,
            "cap_bytes": cap_b,
            "payload_peak_bytes": tr.residual_step_receipt["peak_bytes"] if tr.residual_step_receipt else 0,
            "cuda_receipt": tr.residual_cuda_receipt,
        }
        del tr, ad, bp
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Recursive comparison
    counts = {
        "tensors": 0, "finite_bitwise": 0, "within_tol": 0,
        "overflow_parity": 0, "failures": 0, "max_abs": 0.0,
        "scalars": 0, "scalar_failures": 0, "failure_examples": [],
    }

    def _cmp(u, v, path):
        if isinstance(u, torch.Tensor):
            cm = byte_and_overflow_compare(u, v)
            counts["tensors"] += 1
            counts["finite_bitwise"] += int(cm["finite_bitwise"])
            counts["within_tol"] += int(cm["within_tol"])
            counts["max_abs"] = max(counts["max_abs"], cm["max_abs"])
            if not cm["finite"]:
                if cm["overflow_parity"]:
                    counts["overflow_parity"] += 1
                else:
                    counts["failures"] += 1
                    if len(counts["failure_examples"]) < 12:
                        counts["failure_examples"].append({"path": path, **cm})
            elif not cm["finite_bitwise"]:
                counts["failures"] += 1
                if len(counts["failure_examples"]) < 12:
                    counts["failure_examples"].append({"path": path, **cm})
        elif isinstance(u, dict):
            assert u.keys() == v.keys(), path
            for k in u:
                _cmp(u[k], v[k], f"{path}/{k}")
        elif isinstance(u, (list, tuple)):
            assert len(u) == len(v), path
            for i, (a, b) in enumerate(zip(u, v)):
                _cmp(a, b, f"{path}/{i}")
        elif isinstance(u, float):
            counts["scalars"] += 1
            if not (math.isfinite(u) and math.isfinite(v) and abs(u - v) <= ATOL + RTOL * abs(v)):
                counts["scalar_failures"] += 1
                if len(counts["failure_examples"]) < 12:
                    counts["failure_examples"].append({"path": path, "ref": u, "cand": v})
        else:
            counts["scalars"] += 1
            if u != v:
                counts["scalar_failures"] += 1
                if len(counts["failure_examples"]) < 12:
                    counts["failure_examples"].append({"path": path, "ref": str(u), "cand": str(v)})

    _cmp(snaps["A"], snaps["G"], "trainer")
    return {
        "mode": mode,
        "width": width,
        "n_passes": n_passes,
        "regime": regime,
        "cal_cadence": cal_cadence,
        "budget_mode": budget_mode,
        "seed": seed,
        "arms": metrics_out,
        "parity": counts,
        "finite_bitwise_pass": (
            counts["failures"] == 0
            and counts["scalar_failures"] == 0
            and counts["overflow_parity"] == 0
            and counts["finite_bitwise"] == counts["tensors"]
        ),
    }


def storage_mode_cpu_guards() -> dict[str, bool]:
    """Verify FUSE-13 storage_mode option guards and unchanged production defaults."""
    from .cuda_guard import STORAGE_MODES, target_storage_dtype, validate_cuda_options

    checks: dict[str, bool] = {}

    def raises(fn, exc=(ValueError, TypeError)):
        try:
            fn()
        except exc:
            return True
        return False

    validate_cuda_options(False, "fp32", "fp32")
    validate_cuda_options(True, "bf16", "compute_dtype")
    validate_cuda_options(True, "fp16", "compute_dtype")
    checks["storage_modes_enum"] = STORAGE_MODES == ("fp32", "compute_dtype")
    checks["reject_compute_dtype_without_cuda"] = raises(
        lambda: validate_cuda_options(False, "fp32", "compute_dtype"), ValueError
    )
    checks["reject_unknown_storage_mode"] = raises(
        lambda: validate_cuda_options(True, "bf16", "fp8"), ValueError
    )
    checks["target_dtype_fp32_default"] = target_storage_dtype("bf16", "fp32") == torch.float32
    checks["target_dtype_bf16_compute"] = target_storage_dtype("bf16", "compute_dtype") == torch.bfloat16
    checks["target_dtype_fp16_compute"] = target_storage_dtype("fp16", "compute_dtype") == torch.float16
    checks["target_dtype_tf32_compute"] = target_storage_dtype("tf32", "compute_dtype") == torch.float32
    assert all(checks.values()), f"Storage mode guard failure: {checks}"
    return checks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--allow-cuda", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-watts", type=float, default=60.0)
    args = p.parse_args()

    device = torch.device(args.device)
    t0 = time.perf_counter()
    guard_checks = storage_mode_cpu_guards()

    if device.type == "cuda":
        if not args.allow_cuda:
            p.error("CUDA requires --allow-cuda inside VRAMorama lease")
        snap = check_gpu_safety(min_watts=args.min_watts, min_free_mib=2048.0)
        modes = ("fp32", "tf32", "bf16", "fp16")
        stage_b = {m: stage_b_six_strategy_ablation(device, m, width=128, passes=12) for m in modes}

        reg_roots = tuple(range(4012001, 4012009))
        hold_roots = (4013001, 4013002, 4013003, 4013004)
        stress_reg = {}
        stress_hold = {}
        for m in modes:
            check_gpu_safety(min_watts=args.min_watts, min_free_mib=2048.0)
            stress_reg[m] = stage_c_stress_suite(device, m, reg_roots, storage_mode="compute_dtype")
            stress_hold[m] = stage_c_stress_suite(device, m, hold_roots, storage_mode="compute_dtype")

        gradscaler = stage_c_fp16_gradscaler_parity(device)

        trainer_cells = []
        trainer_grid = [
            ("fp32", 512, 12, "1", "full", "closed_loop"),
            ("tf32", 512, 12, "1", "full", "closed_loop"),
            ("bf16", 128, 6, "1", "full", "closed_loop"),
            ("bf16", 128, 6, "50", "full", "clean_window"),
            ("bf16", 512, 12, "1", "binding", "closed_loop"),
            ("bf16", 512, 12, "off", "full", "closed_loop"),
            ("bf16", 1024, 12, "1", "binding", "closed_loop"),
            ("bf16", 2048, 12, "off", "full", "closed_loop"),
            ("bf16", 2048, 24, "off", "full", "closed_loop"),
            ("fp16", 128, 6, "1", "full", "closed_loop"),
            ("fp16", 128, 6, "50", "full", "clean_window"),
            ("fp16", 512, 12, "1", "binding", "closed_loop"),
            ("fp16", 512, 12, "off", "full", "closed_loop"),
            ("fp16", 1024, 12, "1", "binding", "closed_loop"),
            ("fp16", 2048, 12, "off", "full", "closed_loop"),
        ]
        for idx, (m, w, t, cal, bud, reg) in enumerate(trainer_grid):
            check_gpu_safety(min_watts=args.min_watts, min_free_mib=2048.0)
            trainer_cells.append(stage_c_trainer_parity(device, m, w, t, cal, bud, seed=4013501 + idx, regime=reg))

        configure_precision_mode("fp32")
    else:
        torch.set_num_threads(1)
        snap = {}
        stage_b = {m: stage_b_six_strategy_ablation(device, m, width=64, passes=6) for m in ("fp32", "bf16")}
        stress_reg = {}
        stress_hold = {}
        gradscaler = None
        trainer_cells = []

    out = {
        "stages": "B_C",
        "device": args.device,
        "torch": str(torch.__version__),
        "nvml": snap,
        "storage_mode_guards": guard_checks,
        "stage_b_ablation": stage_b,
        "stage_c_stress_regression": stress_reg,
        "stage_c_stress_holdout": stress_hold,
        "stage_c_fp16_gradscaler": gradscaler,
        "stage_c_trainer_parity": trainer_cells,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {
        "device": args.device,
        "stage_b_shared_cast_bitwise": {
            m: v["strategies"]["shared_compute_shared_cast"]["finite_bitwise"]
            for m, v in stage_b.items()
        },
        "stress_reg_failures": {m: v["failures"] for m, v in stress_reg.items()},
        "stress_hold_failures": {m: v["failures"] for m, v in stress_hold.items()},
        "trainer_finite_bitwise_pass": f"{sum(c['finite_bitwise_pass'] for c in trainer_cells)}/{len(trainer_cells)}",
    }
    print("FUSE13_STAGE_BC_DONE", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
