"""FUSE-12: CUDA lifecycle, stream ownership, FP32 stress, and FUSE-12B mixed-precision gate."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import itertools
import json
import math
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
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
from mechinterp.bpd.trainer import BPDTrainer
from .adjoint_cache import independent_residuals
from .calibration_cache import calibration_residuals
from .cuda_guard import PRECISION_MODES, current_stream_token, validate_cuda_options
from .step_cache import StepResidualPool, planned_sites, supported

ATOL, RTOL = 5e-5, 5e-4
EXIT_POWER_CAP = 75
MIN_VRAM_HEADROOM_MIB = 2048.0


def query_nvml_snapshot() -> dict[str, Any]:
    """Query externally reported GPU telemetry via nvidia-smi (WDDM-aware)."""
    fields = [
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "memory.free",
        "enforced.power.limit",
        "power.default_limit",
        "power.draw",
        "utilization.gpu",
        "utilization.memory",
        "pstate",
    ]
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        def _flt(val: str) -> float | None:
            try:
                return float(val)
            except ValueError:
                return None
        return {
            "name": parts[0],
            "driver_version": parts[1],
            "memory_total_mib": _flt(parts[2]),
            "memory_used_mib": _flt(parts[3]),
            "memory_free_mib": _flt(parts[4]),
            "enforced_power_limit_w": _flt(parts[5]),
            "default_power_limit_w": _flt(parts[6]),
            "power_draw_w": _flt(parts[7]),
            "utilization_gpu_pct": _flt(parts[8]),
            "utilization_memory_pct": _flt(parts[9]),
            "pstate": parts[10] if len(parts) > 10 else None,
        }
    except Exception as exc:
        return {"error": str(exc)}


def check_gpu_safety(min_watts: float = 40.0, min_free_mib: float = MIN_VRAM_HEADROOM_MIB) -> dict[str, Any]:
    snap = query_nvml_snapshot()
    watts = snap.get("enforced_power_limit_w")
    if watts is not None and watts < min_watts:
        print(
            f"[power-guard] GPU enforced power limit {watts}W < {min_watts}W; exiting {EXIT_POWER_CAP}.",
            flush=True,
        )
        sys.exit(EXIT_POWER_CAP)
    free_mib = snap.get("memory_free_mib")
    if free_mib is not None and free_mib < min_free_mib:
        raise RuntimeError(
            f"Insufficient WDDM VRAM headroom: free={free_mib:.1f} MiB < {min_free_mib:.1f} MiB"
        )
    return snap


def configure_precision_mode(precision_mode: str):
    """Set global TF32/matmul flags and return a factory yielding a fresh autocast context."""
    if precision_mode == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        return lambda: nullcontext()
    if precision_mode == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return lambda: nullcontext()
    if precision_mode == "bf16":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        return lambda: torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    if precision_mode == "fp16":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        return lambda: torch.amp.autocast("cuda", enabled=True, dtype=torch.float16)
    raise ValueError(f"unknown precision_mode: {precision_mode!r}")


def tensor_error_metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, Any]:
    """Report bitwise equality separately from numerical tolerance & near-zero error."""
    assert ref.shape == cand.shape and ref.dtype == cand.dtype
    bitwise_equal = bool(torch.equal(ref, cand))
    if not (ref.is_floating_point() or ref.is_complex()):
        return {
            "bitwise_equal": bitwise_equal,
            "finite_ref": True,
            "finite_cand": True,
            "nonfinite_parity": bitwise_equal,
            "within_tol": bitwise_equal,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "near_zero_max_abs_err": 0.0,
        }
    ref_f = ref.detach().float()
    cand_f = cand.detach().float()
    finite_ref = bool(torch.isfinite(ref_f).all().item())
    finite_cand = bool(torch.isfinite(cand_f).all().item())
    if not (finite_ref and finite_cand):
        # Exact non-finite parity check (same NaN/Inf mask and identical finite entries)
        nan_match = bool(torch.equal(torch.isnan(ref_f), torch.isnan(cand_f)))
        inf_match = bool(torch.equal(torch.isinf(ref_f), torch.isinf(cand_f)))
        fin_mask = torch.isfinite(ref_f) & torch.isfinite(cand_f)
        if bool(fin_mask.any().item()):
            diff_fin = (ref_f[fin_mask] - cand_f[fin_mask]).abs()
            max_abs = float(diff_fin.max().item())
            rel_fin = diff_fin / ref_f[fin_mask].abs().clamp_min(1e-12)
            max_rel = float(rel_fin.max().item())
            fin_close = bool(torch.allclose(ref_f[fin_mask], cand_f[fin_mask], atol=ATOL, rtol=RTOL))
        else:
            max_abs = 0.0
            max_rel = 0.0
            fin_close = True
        return {
            "bitwise_equal": bitwise_equal,
            "finite_ref": finite_ref,
            "finite_cand": finite_cand,
            "nonfinite_parity": nan_match and inf_match and fin_close,
            "within_tol": False,
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "near_zero_max_abs_err": 0.0,
        }
    diff = (ref_f - cand_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = ref_f.abs().clamp_min(1e-8)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    near_mask = ref_f.abs() < 1e-4
    near_err = float(diff[near_mask].max().item()) if bool(near_mask.any().item()) else 0.0
    within_tol = bool(torch.allclose(ref_f, cand_f, atol=ATOL, rtol=RTOL))
    return {
        "bitwise_equal": bitwise_equal,
        "finite_ref": True,
        "finite_cand": True,
        "nonfinite_parity": True,
        "within_tol": within_tol,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "near_zero_max_abs_err": near_err,
    }


def cpu_guard_tests() -> dict[str, Any]:
    """Fast CPU-only tests verifying defaults, option validation, and 7-arm parity."""
    checks: dict[str, bool] = {}

    def check(name: str, cond: bool) -> None:
        checks[name] = bool(cond)
        if not cond:
            raise AssertionError(f"CPU guard check failed: {name}")

    def raises(fn, exc_type=(RuntimeError, ValueError, TypeError)) -> bool:
        try:
            fn()
        except exc_type:
            return True
        return False

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")

    # 1. Option validation
    validate_cuda_options(False, "fp32")
    validate_cuda_options(True, "fp32")
    for pm in PRECISION_MODES:
        validate_cuda_options(True, pm)
    check("reject_nonbool_allow_cuda", raises(lambda: validate_cuda_options(1, "fp32"), TypeError))  # type: ignore[arg-type]
    check("reject_unknown_precision", raises(lambda: validate_cuda_options(True, "fp8"), ValueError))
    check("reject_bf16_without_cuda", raises(lambda: validate_cuda_options(False, "bf16"), ValueError))
    check("reject_tf32_without_cuda", raises(lambda: validate_cuda_options(False, "tf32"), ValueError))
    check("reject_fp16_without_cuda", raises(lambda: validate_cuda_options(False, "fp16"), ValueError))

    # 2. CPU wrapper rejected for non-fp32 even with allow_cuda=True
    w = TiedComponentLinear(nn.Linear(16, 16), 4, max_passes=3, init_svd=False)
    check("cpu_fp32_supported_default", supported(w))
    check("cpu_fp32_supported_cuda_optin", supported(w, allow_cuda=True, precision_mode="fp32"))
    check("cpu_bf16_unsupported", not supported(w, allow_cuda=True, precision_mode="bf16"))
    check("cpu_tf32_unsupported", not supported(w, allow_cuda=True, precision_mode="tf32"))
    check("cpu_fp16_unsupported", not supported(w, allow_cuda=True, precision_mode="fp16"))

    # 3. BPDTrainer production defaults unchanged
    model = build_mini_trm(hidden_size=64, num_heads=4, l_layers=2, h_cycles=1, l_cycles=2,
                           puzzle_emb_len=4, seq_len=16, seed=4012901)
    sites = {f"layers.{i}.{n}": 8 for i in range(2) for n in ("o_proj", "down_proj")}
    cfg = BPDConfig(
        c_per_module=sites, n_passes=6, n_steps=2, batch_size=2, seq_len=20,
        seed=4012901, init_svd=False, faithfulness_warmup_steps=1,
        coeff_faith=1e5, coeff_stoch=0.5, coeff_adv=0.5,
        readout_policy="readout_or_boundary", rollout_mode="closed_loop", window_steps=3,
        ci=CIConfig(variant="state_only", d_model=32, n_blocks=1, n_heads=4, mlp_hidden=64,
                    max_passes=6, max_seq_len=20),
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        adversary=AdversaryConfig(inner_steps=2),
        null_calibration=NullCalibrationConfig(use_tolerance_weights=True, n_rotations_train=1, calibrate_every=1),
    )
    bpd = BPDModel(model, sites, max_passes=6, ablation=cfg.ablation, init_svd=False)
    adapter = TRMBPDAdapter(model)
    tr_def = BPDTrainer(bpd, cfg, adapter.rollout, clean_window_state_fn=adapter.sample_clean_window_state,
                        visible_dim=adapter.visible_dim, prompt_dim=adapter.prompt_dim, device="cpu")
    check("default_cache_off", not tr_def.cache_single_backward_residuals)
    check("default_cal_off", not tr_def.cache_calibration_residuals)
    check("default_pgd_mask_off", not tr_def.cache_pgd_mask_residuals)
    check("default_pgd_final_off", not tr_def.cache_pgd_final_residuals)
    check("default_cuda_optin_off", not tr_def.allow_cuda_residual_cache)
    check("default_precision_fp32", tr_def.residual_cache_precision_mode == "fp32")
    check(
        "trainer_rejects_bf16_without_cuda",
        raises(
            lambda: BPDTrainer(
                copy.deepcopy(bpd), cfg, adapter.rollout,
                residual_cache_precision_mode="bf16",
            ),
            ValueError,
        ),
    )

    # 4. CPU cumulative Arm A vs Arm G exact bitwise parity
    def make_trainer(arm_g: bool):
        torch.manual_seed(4012902)
        m = build_mini_trm(hidden_size=64, num_heads=4, l_layers=2, h_cycles=1, l_cycles=2,
                           puzzle_emb_len=4, seq_len=16, seed=4012902)
        b = BPDModel(m, sites, max_passes=6, ablation=cfg.ablation, init_svd=False)
        a = TRMBPDAdapter(m)
        cap = sum(wr.W_target.numel() * 4 for wr in b.wrappers.values())
        tr = BPDTrainer(
            b, cfg, a.rollout, clean_window_state_fn=a.sample_clean_window_state,
            visible_dim=a.visible_dim, prompt_dim=a.prompt_dim, device="cpu",
            cache_single_backward_residuals=arm_g,
            residual_cache_max_bytes=cap,
            residual_cache_policy="step_exact" if arm_g else "first_fit",
            cache_pgd_mask_residuals=arm_g,
            cache_calibration_residuals=arm_g,
            cache_pgd_final_residuals=arm_g,
        )
        return tr, a

    tra, ada = make_trainer(False)
    trb, adb = make_trainer(True)
    tra.run_faithfulness_warmup(1)
    trb.run_faithfulness_warmup(1)
    gen = torch.Generator().manual_seed(4012999)
    inp = torch.randint(1, 11, (2, 16), generator=gen)
    xa = ada.make_batch_with_clean_cache(inp, 6)
    xb = adb.make_batch_with_clean_cache(inp, 6)
    rng = torch.get_rng_state()
    ma = tra.train_step(xa, 0)
    torch.set_rng_state(rng)
    mb = trb.train_step(xb, 0)
    check("cpu_armA_vs_armG_metrics_exact", ma == mb)
    check(
        "cpu_armA_vs_armG_params_exact",
        all(torch.equal(p1, p2) for p1, p2 in zip(tra.bpd_model.parameters(), trb.bpd_model.parameters())),
    )
    check(
        "cpu_armA_vs_armG_ci_exact",
        all(torch.equal(p1, p2) for p1, p2 in zip(tra.ci_net.parameters(), trb.ci_net.parameters())),
    )
    check("cpu_armG_final_pgd_zero_source_adjoints",
          trb.residual_final_pgd_query is not None
          and trb.residual_final_pgd_query["factor_nodes_before"] == 0
          and trb.residual_final_pgd_query["factor_nodes_after"] == 0)
    check("cpu_armG_cuda_receipt_none", trb.residual_cuda_receipt is None)
    return {"status": "pass", "checks": checks}


def cuda_lifecycle_tests(device: torch.device) -> dict[str, Any]:
    """Comprehensive CUDA FP32 lifecycle, stream ownership, allocator, and guard tests."""
    checks: dict[str, bool] = {}

    def check(name: str, cond: bool) -> None:
        checks[name] = bool(cond)
        if not cond:
            raise AssertionError(f"CUDA lifecycle check failed: {name}")

    def raises(fn, exc_type=(RuntimeError, ValueError, TypeError)) -> bool:
        try:
            fn()
        except exc_type:
            return True
        return False

    configure_precision_mode("fp32")
    torch.manual_seed(4012101)
    torch.cuda.manual_seed_all(4012101)

    def make(width=32, rank=8):
        return TiedComponentLinear(
            nn.Linear(width, width),
            rank,
            max_passes=3,
            ablation=AblationConfig(mode="zero", sphere_rescale=False),
            init_svd=False,
        ).to(device)

    # 1. Default allow_cuda=False MUST bypass CUDA wrappers to native reference
    w = make()
    n = w.W_target.numel() * 4
    check("default_cuda_rejected_by_supported", not supported(w))
    chosen, plan = planned_sites([w], 6, n)
    check("default_cuda_rejected_by_planner", not chosen and plan.value == 0)
    p = StepResidualPool([w], max_bytes=n)
    with p.scope() as st:
        v = w.weight_delta()
        check("default_cuda_bypasses_in_pool", st.bypasses == 1 and st.misses == 0 and p.peak_bytes == 0)
    v.square().sum().backward()
    p.close()

    # 2. Explicit opt-in allow_cuda=True, precision_mode="fp32"
    w = make()
    check("optin_cuda_supported", supported(w, allow_cuda=True, precision_mode="fp32"))
    chosen, plan = planned_sites([w], 6, n, allow_cuda=True, precision_mode="fp32")
    check("optin_cuda_planned", len(chosen) == 1 and plan.size == n)
    p = StepResidualPool([w], max_bytes=n, allow_cuda=True, precision_mode="fp32")
    with p.scope(mask_only=True) as st_m:
        vm0 = w.weight_delta()
        vm1 = w.weight_delta()
        check("cuda_mask_only_detached", not vm0.requires_grad and torch.equal(vm0, vm1))
        check("cuda_mask_only_same_ptr", vm0.untyped_storage().data_ptr() == vm1.untyped_storage().data_ptr())
    check("cuda_stream_recorded", p.owner_stream == current_stream_token(device))
    with p.scope() as st_j:
        vj0 = w.weight_delta()
        vj1 = w.weight_delta()
        check("cuda_independent_grad_fns", vj0.grad_fn is not vj1.grad_fn)
        check("cuda_cross_scope_same_ptr", vj0.untyped_storage().data_ptr() == vm0.untyped_storage().data_ptr())
        check("cuda_cross_scope_hits", st_j.misses == 0 and st_j.cross_scope_hits == 2)
    p.seal_forward()
    check("cuda_sealed_blocks_reentry", p.sealed and raises(lambda: p.scope().__enter__()))
    (vj0 + vj1).square().sum().backward()
    check("cuda_adjoints_executed", st_j.backward_nodes == 2 and st_m.backward_nodes == 0)
    p.close()
    check("cuda_closed_restored", p.closed and p.resident_bytes == 0 and "weight_delta" not in w.__dict__)
    del vm0, vm1, vj0, vj1

    # 3. Single sequential CUDA stream ownership violation MUST raise & poison
    w = make()
    p = StepResidualPool([w], max_bytes=n, allow_cuda=True, precision_mode="fp32")
    with p.scope():
        _ = w.weight_delta()
    other_stream = torch.cuda.Stream(device=device)

    def cross_stream_call():
        with torch.cuda.stream(other_stream):
            with p.scope():
                w.weight_delta()

    check("cuda_stream_switch_rejected", raises(cross_stream_call, RuntimeError) and p.poisoned)
    p.close()
    check("cuda_stream_violation_restored", "weight_delta" not in w.__dict__)
    other_stream.synchronize()

    # 4. Calibration cache CUDA stream ownership violation
    w = make()
    w.update_running_means = True
    with torch.no_grad():
        def cal_stream_switch():
            with calibration_residuals([("s", w)], calls=3, max_bytes=n, allow_cuda=True, precision_mode="fp32"):
                w.weight_delta()
                with torch.cuda.stream(other_stream):
                    w.weight_delta()
        check("cuda_cal_stream_switch_rejected", raises(cal_stream_switch, RuntimeError))
        check("cuda_cal_restored_after_error", "weight_delta" not in w.__dict__)

    # 5. TF32 enabled while precision_mode="fp32" MUST bypass cleanly
    w = make()
    p = StepResidualPool([w], max_bytes=n, allow_cuda=True, precision_mode="fp32")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with p.scope() as st:
            v = w.weight_delta()
            check("cuda_tf32_bypassed_under_fp32_mode", st.bypasses == 1 and st.misses == 0 and p.peak_bytes == 0)
        v.square().sum().backward()
    finally:
        configure_precision_mode("fp32")
        p.close()

    # 6. CUDA autocast enabled while precision_mode="fp32" MUST bypass cleanly
    w = make()
    p = StepResidualPool([w], max_bytes=n, allow_cuda=True, precision_mode="fp32")
    with p.scope() as st:
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            v = w.weight_delta()
        check("cuda_autocast_bypassed_under_fp32_mode", st.bypasses == 1 and st.misses == 0)
    v.square().sum().backward()
    p.close()

    # 7. Precision/TF32 state mutation mid-pool MUST trigger identity mismatch & poison
    w = make()
    p = StepResidualPool([w], max_bytes=n, allow_cuda=True, precision_mode="fp32")
    with p.scope():
        _ = w.weight_delta()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        check("cuda_midpool_tf32_flip_poisoned", raises(lambda: p.seal_forward(), RuntimeError) and p.poisoned)
    finally:
        configure_precision_mode("fp32")
        p.close()

    # 8. Parameter mutation, identity swap, cached primal mutation, early backward, exception cleanup
    w = make()
    p = StepResidualPool([w], max_bytes=n, allow_cuda=True, precision_mode="fp32")
    with p.scope():
        w.weight_delta()
    with torch.no_grad():
        w.V.add_(0.01)
    check("cuda_param_mutation_rejected", raises(lambda: p.scope().__enter__(), RuntimeError) and p.poisoned)
    p.close()

    w = make()
    p = StepResidualPool([w], max_bytes=n, allow_cuda=True, precision_mode="fp32")
    with p.scope():
        w.weight_delta()
    w.U = nn.Parameter(w.U.detach().clone())
    check("cuda_identity_swap_rejected", raises(lambda: p.scope().__enter__(), RuntimeError) and p.poisoned)
    p.close()

    w = make()
    p = StepResidualPool([w], max_bytes=n, allow_cuda=True, precision_mode="fp32")
    with p.scope():
        w.weight_delta()
    with torch.no_grad():
        p.entries[id(w)][1].add_(1.0)
    check("cuda_cached_primal_mutation_rejected", raises(lambda: p.scope().__enter__(), RuntimeError) and p.poisoned)
    p.close()

    w = make()
    p = StepResidualPool([w], max_bytes=n, allow_cuda=True, precision_mode="fp32")
    def early_bwd():
        with p.scope():
            w.weight_delta().square().sum().backward()
    check("cuda_early_backward_rejected", raises(early_bwd, RuntimeError) and p.poisoned and p.sealed)
    check("cuda_early_backward_restored", "weight_delta" not in w.__dict__)
    p.close()

    # 9. independent_residuals CUDA stream violation & normal opt-in
    w = make()
    with independent_residuals([w], variant="residual", max_bytes=n, allow_cuda=True, precision_mode="fp32") as st_ind:
        d0 = w.weight_delta()
        d1 = w.weight_delta()
        check("cuda_independent_hits", st_ind.misses == 1 and st_ind.hits == 1)
    (d0 + d1).square().sum().backward()
    check("cuda_independent_adjoints", st_ind.backward_nodes == 2)
    del d0, d1

    # 10. Allocator lifetime & zero leak audit after close + backward
    w = make(width=256, rank=32)
    n_big = w.W_target.numel() * 4
    torch.cuda.synchronize(device)
    gc.collect()
    base_alloc = torch.cuda.memory_allocated(device)
    p = StepResidualPool([w], max_bytes=n_big, allow_cuda=True, precision_mode="fp32")
    with p.scope():
        d0 = w.weight_delta()
        d1 = w.weight_delta()
    peak_in_scope = torch.cuda.memory_allocated(device)
    loss = (d0 + d1).square().sum()
    loss.backward()
    p.close()
    del d0, d1, loss, p
    w.V.grad = None
    w.U.grad = None
    torch.cuda.synchronize(device)
    gc.collect()
    after_alloc = torch.cuda.memory_allocated(device)
    check("cuda_allocator_allocated_during_scope", peak_in_scope >= base_alloc + n_big)
    check("cuda_allocator_zero_leak_after_close", after_alloc == base_alloc)

    return {
        "status": "pass",
        "checks": checks,
        "allocator_audit": {
            "base_bytes": base_alloc,
            "peak_in_scope_bytes": peak_in_scope,
            "after_close_bytes": after_alloc,
            "expected_payload_bytes": n_big,
        },
    }


def cuda_stress_suite(device: torch.device, precision_mode: str = "fp32") -> dict[str, Any]:
    """96 CUDA stress fixtures (8 fresh roots x 4 scales x ordinary/exact/near)."""
    ac_ctx = configure_precision_mode(precision_mode)
    total_cases = 0
    total_tensors = 0
    bitwise_exact = 0
    within_tol = 0
    nonfinite_parity_cases = 0
    failures = 0
    max_abs = 0.0
    max_rel = 0.0
    near_zero_max_abs = 0.0
    total_joint_nodes = 0
    total_hits = 0
    per_cell_rows = []
    failure_examples = []

    for scale, target in itertools.product((1.0, 10.0, 100.0, 10000.0), ("ordinary", "exact", "near")):
        cell = {
            "precision_mode": precision_mode,
            "scale": scale,
            "target": target,
            "cases": 0,
            "tensors": 0,
            "bitwise_exact": 0,
            "within_tol": 0,
            "nonfinite_parity": 0,
            "failures": 0,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "near_zero_max_abs_err": 0.0,
        }
        for seed in range(4012001, 4012009):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            w = TiedComponentLinear(
                nn.Linear(64, 64),
                32,
                max_passes=3,
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
                for j in range(3):
                    w.observe_clean_pass(x[0, j], j)

            def run(fast: bool):
                q = copy.deepcopy(w)
                q.mode = "component"
                p = (
                    StepResidualPool(
                        [q],
                        max_bytes=q.W_target.numel() * 4,
                        allow_cuda=True,
                        precision_mode=precision_mode,
                    )
                    if fast
                    else None
                )
                stats = []
                vals = []
                losses = []
                leaves = []
                try:
                    with ac_ctx():
                        for j in range(3):
                            xx = x[j].clone().requires_grad_()
                            mm = m[j].clone().requires_grad_()
                            dd = d[j].clone().requires_grad_()
                            q.reset_pass_state()
                            q.schedule = PassMaskSchedule(mm.sigmoid(), dd.sigmoid())
                            with (p.scope(mask_only=(j == 1)) if fast else nullcontext(None)) as st:
                                # Also record direct weight_delta residual value parity on first pass
                                if j == 0:
                                    vals.append(q.weight_delta().detach().clone())
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
                                        try:
                                            with p.scope():
                                                pass
                                        except RuntimeError:
                                            pass
                                        else:
                                            raise AssertionError("scope after seal")
                        total = losses[0] + 0.5 * losses[1] + 0.01 * q.weight_delta().square().mean()
                    total.backward()
                    vals.extend(t.grad.detach().clone() for t in (q.V, q.U, *leaves))
                    if fast:
                        # 4 calls in scope 0 (1 direct residual + 3 forward), 3 in scope 1, 3 in scope 2
                        assert [s.backward_nodes for s in stats] == [4, 0, 3], [s.backward_nodes for s in stats]
                        assert [s.hits for s in stats] == [3, 3, 3], [s.hits for s in stats]
                        assert p.sealed and not p.poisoned
                        assert p.peak_bytes == q.W_target.numel() * 4
                        assert p.stream_checks > 0
                    return vals, stats
                finally:
                    if p is not None:
                        p.close()
                        assert p.closed and p.resident_bytes == 0
                        assert "weight_delta" not in q.__dict__

            ref_vals, _ = run(False)
            fast_vals, ss = run(True)
            torch.cuda.synchronize(device)
            names = [
                "residual_D", "out_stoch", "out_inner", "g_inner_m", "g_inner_d",
                "out_final", "g_final_m", "g_final_d",
                "g_V", "g_U", "g_x0", "g_m0", "g_d0", "g_x2", "g_m2", "g_d2",
            ]
            assert len(ref_vals) == len(fast_vals) == len(names)
            for nm, u, v in zip(names, ref_vals, fast_vals):
                em = tensor_error_metrics(u, v)
                total_tensors += 1
                cell["tensors"] += 1
                bitwise_exact += int(em["bitwise_equal"])
                cell["bitwise_exact"] += int(em["bitwise_equal"])
                within_tol += int(em["within_tol"])
                cell["within_tol"] += int(em["within_tol"])
                cell["max_abs_err"] = max(cell["max_abs_err"], em["max_abs_err"])
                cell["max_rel_err"] = max(cell["max_rel_err"], em["max_rel_err"])
                cell["near_zero_max_abs_err"] = max(cell["near_zero_max_abs_err"], em["near_zero_max_abs_err"])
                max_abs = max(max_abs, em["max_abs_err"])
                max_rel = max(max_rel, em["max_rel_err"])
                near_zero_max_abs = max(near_zero_max_abs, em["near_zero_max_abs_err"])
                if not em["finite_ref"] and not em["finite_cand"] and em["nonfinite_parity"]:
                    nonfinite_parity_cases += 1
                    cell["nonfinite_parity"] += 1
                elif not em["within_tol"]:
                    failures += 1
                    cell["failures"] += 1
                    if len(failure_examples) < 16:
                        failure_examples.append(
                            {"seed": seed, "scale": scale, "target": target, "tensor": nm, **em}
                        )
            total_cases += 1
            cell["cases"] += 1
            total_joint_nodes += sum(s.backward_nodes for s in ss)
            total_hits += sum(s.hits for s in ss)
        per_cell_rows.append(cell)

    configure_precision_mode("fp32")
    return {
        "precision_mode": precision_mode,
        "cases": total_cases,
        "tensors": total_tensors,
        "bitwise_exact": bitwise_exact,
        "within_tol": within_tol,
        "nonfinite_parity_tensors": nonfinite_parity_cases,
        "failures": failures,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "near_zero_max_abs_err": near_zero_max_abs,
        "joint_factor_nodes": total_joint_nodes,
        "hits": total_hits,
        "failure_examples": failure_examples,
        "cells": per_cell_rows,
    }


def cuda_mixed_precision_identity_tests(device: torch.device) -> dict[str, Any]:
    """Verify cache identity rules and native fallback across TF32, BF16, and FP16."""
    checks: dict[str, bool] = {}

    def check(name: str, cond: bool) -> None:
        checks[name] = bool(cond)
        if not cond:
            raise AssertionError(f"Mixed-precision identity check failed: {name}")

    def make():
        return TiedComponentLinear(
            nn.Linear(32, 32), 8, max_passes=3,
            ablation=AblationConfig(mode="zero", sphere_rescale=False),
            init_svd=False,
        ).to(device)

    for mode in ("tf32", "bf16", "fp16"):
        w = make()
        n = w.W_target.numel() * 4
        # Without matching ambient context, must NOT be supported
        configure_precision_mode("fp32")
        check(f"{mode}_rejected_in_fp32_context", not supported(w, allow_cuda=True, precision_mode=mode))

        ac_ctx = configure_precision_mode(mode)
        with ac_ctx():
            check(f"{mode}_admitted_in_matching_context", supported(w, allow_cuda=True, precision_mode=mode))
            p = StepResidualPool([w], max_bytes=n, allow_cuda=True, precision_mode=mode)
            with p.scope() as st:
                d0 = w.weight_delta()
                d1 = w.weight_delta()
                check(f"{mode}_primal_fp32", d0.dtype == torch.float32)
                check(f"{mode}_hit_and_alias", st.misses == 1 and st.hits == 1
                      and d0.untyped_storage().data_ptr() == d1.untyped_storage().data_ptr())
            (d0 + d1).square().sum().backward()
            p.close()

        # Verify mid-pool autocast exit triggers identity error for bf16/fp16
        if mode in ("bf16", "fp16"):
            w2 = make()
            p2 = StepResidualPool([w2], max_bytes=n, allow_cuda=True, precision_mode=mode)
            with ac_ctx():
                with p2.scope():
                    _ = w2.weight_delta()
            try:
                p2.seal_forward()
            except RuntimeError:
                check(f"{mode}_autocast_exit_detected", p2.poisoned)
            else:
                raise AssertionError(f"{mode}: autocast exit was not detected")
            p2.close()

    configure_precision_mode("fp32")
    return {"status": "pass", "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-guard-only", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--skip-mixed-precision", action="store_true")
    parser.add_argument("--min-watts", type=float, default=40.0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    result: dict[str, Any] = {
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_guard": cpu_guard_tests(),
    }

    if args.cuda and not args.cpu_guard_only:
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda requested but torch.cuda.is_available() is False")
        result["preflight_nvml"] = check_gpu_safety(min_watts=args.min_watts)
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        result["device_properties"] = {
            "name": props.name,
            "major": props.major,
            "minor": props.minor,
            "total_memory_bytes": int(props.total_memory),
            "multi_processor_count": int(props.multi_processor_count),
        }

        result["cuda_fp32_lifecycle"] = cuda_lifecycle_tests(device)
        fp32_stress = cuda_stress_suite(device, "fp32")
        result["cuda_fp32_stress"] = fp32_stress
        fp32_passed = (
            result["cuda_fp32_lifecycle"]["status"] == "pass"
            and fp32_stress["failures"] == 0
            and fp32_stress["nonfinite_parity_tensors"] == 0
        )
        result["fp32_gate_passed"] = fp32_passed

        if fp32_passed and not args.skip_mixed_precision:
            result["fuse12b_identity_tests"] = cuda_mixed_precision_identity_tests(device)
            mp_results = {}
            for pm in ("tf32", "bf16", "fp16"):
                check_gpu_safety(min_watts=args.min_watts)
                mp_results[pm] = cuda_stress_suite(device, pm)
            result["fuse12b_stress"] = mp_results
        elif not fp32_passed:
            result["fuse12b_status"] = "blocked_by_fp32_gate"

    result["elapsed_seconds"] = time.perf_counter() - t0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    summary = {
        "cpu_guard": result["cpu_guard"]["status"],
        "fp32_gate_passed": result.get("fp32_gate_passed"),
        "fp32_bitwise_exact": (
            f"{result['cuda_fp32_stress']['bitwise_exact']}/{result['cuda_fp32_stress']['tensors']}"
            if "cuda_fp32_stress" in result
            else None
        ),
        "fp32_failures": result.get("cuda_fp32_stress", {}).get("failures"),
        "elapsed_s": round(result["elapsed_seconds"], 2),
    }
    print("FUSE12_VALIDATE_DONE", json.dumps(summary), flush=True)
    if args.cuda and not args.cpu_guard_only and not result.get("fp32_gate_passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
