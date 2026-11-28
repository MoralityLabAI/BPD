"""FUSE-14 Stages A, B, C, D, E & G: Whole-block frozen projection BF16 cast validation.

- Stage A: Per-layer, per-visit, per-rollout saved-tensor cast audit + standalone probe replay.
- Stage B: Isolation ablation inside BPDTrainer (FUSE-13 G, autocast cache off, explicit
           per-call casts, qkv_only, gate_up_only, both block_projs).
- Stage C: 16-check lifecycle, mutation, layout, stream & default-off guard suite.
- Stage D: Frozen multi-step BF16 parity matrix (widths 128..2048 x visits 2..24,
           multiple roots, closed_loop/clean_window, calibration off/1/50, binding/full,
           plus FP16 HOLD confirmation).
- Stage E: Independent activation checkpointing parity & VRAM/latency evaluation.
- Stage G: Local checkpoint availability audit (TRM/DataGoblin) + real pretrained
           Qwen3-4B (hidden=2560, inter=9728) looped block transfer benchmark.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize as _parametrize
from torch.utils.checkpoint import checkpoint as ckpt_fn

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
from .fuse14_projection_probe import guards as probe_guards, run_cell as probe_run_cell
from .projection_cache import FrozenProjectionCastPool, eligible_frozen_projections
from .validate_fuse12_cuda import (
    ATOL,
    EXIT_POWER_CAP,
    RTOL,
    check_gpu_safety,
    configure_precision_mode,
    query_nvml_snapshot,
)
from .validate_fuse13_precision import byte_and_overflow_compare

def _find_local_qwen3_4b_shard2() -> Path | None:
    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-4B" / "snapshots"
    if not root.exists():
        return None
    for cand in sorted(root.rglob("model-00002-of-00003.safetensors")):
        if cand.is_file():
            return cand
    return None


def _bf16_ac(device: torch.device):
    if device.type == "cuda":
        return configure_precision_mode("bf16")
    torch.set_float32_matmul_precision("highest")
    return lambda: torch.autocast("cpu", enabled=True, dtype=torch.bfloat16)


def _build_trm_trainer(
    *,
    device: torch.device,
    width: int,
    n_passes: int,
    seed: int,
    regime: str = "closed_loop",
    cal_cadence: str = "off",
    budget_mode: str = "full",
    cache_residuals: bool = True,
    cache_projections: bool = False,
    projection_targets: str = "block_projs",
    allow_recompute: bool = False,
    precision_mode: str = "bf16",
    n_steps: int = 3,
):
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
    use_tol = cal_cadence != "off"
    cal_every = 1 if cal_cadence == "1" else (50 if cal_cadence == "50" else 0)
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
        ablation=AblationConfig(
            mode="pass_mean" if cal_cadence != "off" else "zero",
            sphere_rescale=(cal_cadence != "off"),
        ),
        adversary=AdversaryConfig(inner_steps=2),
        null_calibration=NullCalibrationConfig(
            use_tolerance_weights=use_tol,
            n_rotations_train=1,
            calibrate_every=cal_every,
        ),
    )
    bpd = BPDModel(model, sites, max_passes=n_passes, ablation=cfg.ablation, init_svd=False)
    adapter = TRMBPDAdapter(model)
    elem_bytes = 2 if precision_mode in ("bf16", "fp16") else 4
    cap_bytes = sum(
        w.W_target.numel() * elem_bytes
        for nm, w in bpd.wrappers.items()
        if budget_mode == "full" or nm.startswith("layers.0.")
    )
    is_cuda = device.type == "cuda"
    tr = BPDTrainer(
        bpd,
        cfg,
        adapter.rollout,
        clean_window_state_fn=adapter.sample_clean_window_state,
        visible_dim=adapter.visible_dim,
        prompt_dim=adapter.prompt_dim,
        device=device,
        cache_single_backward_residuals=cache_residuals and is_cuda,
        residual_cache_max_bytes=cap_bytes,
        residual_cache_policy="step_exact" if (cache_residuals and is_cuda) else "first_fit",
        cache_pgd_mask_residuals=cache_residuals and is_cuda,
        cache_calibration_residuals=cache_residuals and is_cuda,
        cache_pgd_final_residuals=cache_residuals and is_cuda,
        allow_cuda_residual_cache=is_cuda,
        residual_cache_precision_mode=precision_mode if is_cuda else "fp32",
        residual_cache_storage_mode="compute_dtype" if (cache_residuals and is_cuda) else "fp32",
        cache_frozen_projection_casts=cache_projections,
        projection_cache_targets=projection_targets,
        projection_cache_allow_recompute=allow_recompute,
    )
    return tr, adapter, model


def _compare_snapshots(snap_a: dict[str, Any], snap_b: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "tensors": 0,
        "finite_bitwise": 0,
        "within_tol": 0,
        "overflow_parity": 0,
        "failures": 0,
        "max_abs": 0.0,
        "scalars": 0,
        "scalar_failures": 0,
        "failure_examples": [],
    }

    def _cmp(u, v, path: str) -> None:
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
            if u.keys() != v.keys():
                counts["failures"] += 1
                if len(counts["failure_examples"]) < 12:
                    counts["failure_examples"].append(
                        {
                            "path": path,
                            "only_in_ref": sorted(set(u) - set(v))[:8],
                            "only_in_cand": sorted(set(v) - set(u))[:8],
                        }
                    )
                for k in sorted(set(u) & set(v)):
                    _cmp(u[k], v[k], f"{path}/{k}")
                return
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

    _cmp(snap_a, snap_b, "trainer")
    counts["finite_bitwise_pass"] = (
        counts["failures"] == 0
        and counts["scalar_failures"] == 0
        and counts["overflow_parity"] == 0
        and counts["finite_bitwise"] == counts["tensors"]
    )
    return counts


def _run_trainer_and_capture(
    *,
    device: torch.device,
    width: int,
    n_passes: int,
    seed: int,
    regime: str,
    cal_cadence: str,
    budget_mode: str,
    cache_residuals: bool,
    cache_projections: bool,
    projection_targets: str = "block_projs",
    allow_recompute: bool = False,
    precision_mode: str = "bf16",
    n_steps: int = 2,
    autocast_cache_enabled: bool = True,
    explicit_per_call_cast: bool = False,
    enable_checkpointing: bool = False,
    audit_casts: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        if precision_mode == "bf16":
            configure_precision_mode("bf16")
        else:
            configure_precision_mode(precision_mode)
        ac_dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision_mode)
        ac_fn = (
            (lambda: torch.amp.autocast("cuda", enabled=True, dtype=ac_dt, cache_enabled=autocast_cache_enabled))
            if ac_dt is not None
            else (lambda: nullcontext())
        )
    else:
        torch.set_float32_matmul_precision("highest")
        ac_fn = lambda: torch.autocast("cpu", enabled=True, dtype=torch.bfloat16, cache_enabled=autocast_cache_enabled)

    tr, ad, model = _build_trm_trainer(
        device=device,
        width=width,
        n_passes=n_passes,
        seed=seed,
        regime=regime,
        cal_cadence=cal_cadence,
        budget_mode=budget_mode,
        cache_residuals=cache_residuals,
        cache_projections=cache_projections,
        projection_targets=projection_targets,
        allow_recompute=allow_recompute,
        precision_mode=precision_mode,
        n_steps=n_steps + 1,
    )

    # Optional Stage B explicit per-call cast baseline on qkv_proj + gate_up_proj
    patched_per_call = []
    if explicit_per_call_cast:
        for nm, mod in eligible_frozen_projections(tr.bpd_model, targets="block_projs"):
            orig_f = mod.forward

            def _make(o):
                def _f(self_m, x):
                    dev = self_m.weight.device.type
                    if torch.is_autocast_enabled(dev) and torch.get_autocast_dtype(dev) == torch.bfloat16 and torch.is_grad_enabled():
                        return F.linear(x, self_m.weight.detach().to(torch.bfloat16))
                    return o(x)
                return MethodType(_f, mod)

            patched_per_call.append(mod)
            mod.forward = _make(orig_f)

    # Optional Stage E activation checkpointing on TRMBlock invocations.
    # TiedComponentLinear tracks (mode, schedule, calls, pass_offset) imperatively;
    # checkpoint recomputation runs after masked_components() exits, so we explicitly
    # capture and restore each block's wrapper state during recomputation.
    if enable_checkpointing:
        def _ckpt_run_l_level(self_m, state: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
            h = state + injection
            for blk in self_m.layers:
                if torch.is_grad_enabled():
                    wrappers = [m for m in (blk.o_proj, blk.down_proj) if isinstance(m, TiedComponentLinear)]
                    snap_state = [(w.mode, w.schedule, w.calls, w.pass_offset) for w in wrappers]

                    def _run_blk(h_in, *_deps, _blk=blk, _ws=wrappers, _st=snap_state):
                        prev = [(w.mode, w.schedule, w.calls, w.pass_offset) for w in _ws]
                        for w, (md, sc, cl, po) in zip(_ws, _st):
                            w.mode, w.schedule, w.calls, w.pass_offset = md, sc, cl, po
                        try:
                            return _blk(h_in, self_m.cos, self_m.sin)
                        finally:
                            for w, (md, sc, cl, po) in zip(_ws, prev):
                                w.mode, w.schedule, w.calls, w.pass_offset = md, sc, cl, po

                    deps_list = [t for w in wrappers for t in (w.V, w.U)]
                    for w in wrappers:
                        if w.schedule is not None:
                            deps_list.append(w.schedule.comp_mask)
                            if w.schedule.delta_mask is not None:
                                deps_list.append(w.schedule.delta_mask)
                    h = ckpt_fn(_run_blk, h, *deps_list, use_reentrant=False)
                    # Advance live forward call counters by 1 to match non-checkpointed forward
                    for w, (_, _, cl, _) in zip(wrappers, snap_state):
                        w.calls = cl + 1
                else:
                    h = blk(h, self_m.cos, self_m.sin)
            return h

        model.run_l_level = MethodType(_ckpt_run_l_level, model)

    with ac_fn():
        tr.run_faithfulness_warmup(1)

    gen_b = torch.Generator().manual_seed(seed + 500)
    raw_inps = [torch.randint(1, 11, (2, 16), generator=gen_b) for _ in range(n_steps)]

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

    # Optional Stage A saved-tensor cast audit by exact (shape, stride) of frozen projections.
    # Disambiguates qkv_proj [3*D, D] (saved transposed as [D, 3*D] stride (1, D)) from
    # down_proj [D, inter] (saved transposed as [inter, D] stride (1, inter)).
    proj_sigs: dict[tuple[tuple[int, int], tuple[int, int]], str] = {}
    for nm, mod in eligible_frozen_projections(tr.bpd_model, targets="block_projs"):
        out_f, in_f = int(mod.weight.shape[0]), int(mod.weight.shape[1])
        kind = nm.split(".")[-1]
        proj_sigs[((out_f, in_f), (in_f, 1))] = kind
        proj_sigs[((in_f, out_f), (1, in_f))] = kind
    audit_events: list[dict[str, Any]] = []
    saved_ptrs: dict[int, dict[str, Any]] = {}

    def _pack(t: torch.Tensor):
        if t.dtype == torch.bfloat16 and t.ndim == 2:
            sig = (tuple(t.shape), tuple(t.stride()))
            if sig in proj_sigs:
                st = t.untyped_storage()
                ptr = int(st.data_ptr())
                rec = {
                    "kind": proj_sigs[sig],
                    "shape": list(t.shape),
                    "stride": list(t.stride()),
                    "dtype": str(t.dtype),
                    "nbytes": int(st.nbytes()),
                    "ptr": ptr,
                }
                audit_events.append(rec)
                saved_ptrs[ptr] = rec
        return t

    step_ms: list[float] = []
    last_metrics = None
    last_bp = None
    for s in range(n_steps):
        bp = ad.make_batch_with_clean_cache(raw_inps[s], n_passes)
        last_bp = bp
        torch.manual_seed(seed + 1000 + s)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 1000 + s)
        ctx_hook = (
            torch.autograd.graph.saved_tensors_hooks(_pack, lambda t: t)
            if (audit_casts and s == n_steps - 1)
            else nullcontext()
        )
        if device.type == "cuda":
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            ev0.record()
            with ctx_hook, ac_fn():
                last_metrics = tr.train_step(bp, s)
            ev1.record()
            torch.cuda.synchronize(device)
            step_ms.append(float(ev0.elapsed_time(ev1)))
        else:
            t0 = time.perf_counter()
            with ctx_hook, ac_fn():
                last_metrics = tr.train_step(bp, s)
            step_ms.append((time.perf_counter() - t0) * 1000.0)

    for h in hooks:
        h.remove()
    for m in patched_per_call:
        if "forward" in m.__dict__:
            del m.forward

    with torch.no_grad(), ac_fn():
        ev_m, ev_ci, _ = tr.evaluate(last_bp)
        sched = deterministic_ci_schedules(ev_ci.lower_leaky, delta_value=0.0)
        pred_tr = tr.run_masked_rollout(last_bp, sched, n_passes=n_passes)

    peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_res = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    nvml_used = float(query_nvml_snapshot().get("memory_used_mib") or 0.0) if device.type == "cuda" else 0.0

    snap = {
        "metrics": dataclasses.asdict(last_metrics),
        "ascent_grads": ascent_grads,
        "preclip_grads": preclip_grads,
        "bpd_state": {k: v.detach().cpu().clone() for k, v in tr.bpd_model.state_dict().items()},
        "ci_state": {k: v.detach().cpu().clone() for k, v in tr.ci_net.state_dict().items()},
        "opt_state": {
            str(k): {sk: (sv.detach().cpu().clone() if isinstance(sv, torch.Tensor) else sv) for sk, sv in sd.items()}
            for k, sd in tr.optimizer.state_dict()["state"].items()
        },
        "pgd_sources": {k: v.detach().cpu().clone() for k, v in tr.pgd.sources.items()},
        "pass_tolerances": tr.pass_tolerances,
        "eval_metrics": ev_m,
        "pred_readouts": {k: v.detach().cpu().clone() for k, v in pred_tr.readouts.items()},
        "greedy": {k: v.argmax(-1).cpu() for k, v in pred_tr.readouts.items()},
        "trainer_rng": tr.rng.get_state().cpu(),
    }

    audit_summary = None
    if audit_casts:
        by_kind: dict[str, dict[str, int]] = {}
        for ev in audit_events:
            k = ev["kind"]
            b = by_kind.setdefault(k, {"pack_calls": 0, "unique_ptrs": 0, "unique_bytes": 0})
            b["pack_calls"] += 1
        for rec in saved_ptrs.values():
            k = rec["kind"]
            by_kind[k]["unique_ptrs"] += 1
            by_kind[k]["unique_bytes"] += rec["nbytes"]
        audit_summary = {
            "total_pack_calls": len(audit_events),
            "unique_storages": len(saved_ptrs),
            "unique_bytes": sum(r["nbytes"] for r in saved_ptrs.values()),
            "by_kind": by_kind,
        }

    meta = {
        "step_ms": step_ms,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_res,
        "nvml_used_mib": nvml_used,
        "residual_step_receipt": tr.residual_step_receipt,
        "residual_cuda_receipt": tr.residual_cuda_receipt,
        "projection_cache_receipt": tr.projection_cache_receipt,
        "cast_audit": audit_summary,
    }
    del tr, ad, model, last_bp
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return snap, meta


# ---- Stage C: Lifecycle & Fail-Closed Guard Suite ----
def stage_c_lifecycle_guards(device: torch.device) -> dict[str, bool]:
    checks: dict[str, bool] = {}

    def raises(fn, exc=(RuntimeError, ValueError, TypeError)) -> bool:
        try:
            fn()
        except exc:
            return True
        return False

    is_cuda = device.type == "cuda"
    ac_fn = _bf16_ac(device)

    # 1. Production defaults unchanged on BPDTrainer
    tr_def, _, _ = _build_trm_trainer(
        device=torch.device("cpu"), width=32, n_passes=2, seed=14001,
        cache_residuals=False, cache_projections=False, precision_mode="fp32",
    )
    checks["default_proj_cache_off"] = tr_def.cache_frozen_projection_casts is False
    checks["default_proj_recompute_off"] = tr_def.projection_cache_allow_recompute is False
    checks["default_cuda_cache_off"] = tr_def.allow_cuda_residual_cache is False

    # 2. Structural rejections
    m_bias = nn.Linear(8, 16, bias=True).to(device).requires_grad_(False)
    checks["reject_biased"] = raises(lambda: FrozenProjectionCastPool([("b", m_bias)], allow_cuda=is_cuda), ValueError)

    class SubLinear(nn.Linear):
        pass

    m_sub = SubLinear(8, 16, bias=False).to(device).requires_grad_(False)
    checks["reject_subclass"] = raises(lambda: FrozenProjectionCastPool([("s", m_sub)], allow_cuda=is_cuda), ValueError)

    m_par = nn.Linear(8, 16, bias=False).to(device).requires_grad_(False)

    class _IdParam(nn.Module):
        def forward(self, x):
            return x

    _parametrize.register_parametrization(m_par, "weight", _IdParam())
    checks["reject_parametrized"] = raises(lambda: FrozenProjectionCastPool([("p", m_par)], allow_cuda=is_cuda), ValueError)

    m_pre = nn.Linear(8, 16, bias=False).to(device).requires_grad_(False)
    m_pre.forward = MethodType(lambda s, x: F.linear(x, s.weight), m_pre)
    checks["reject_prepatched"] = raises(lambda: FrozenProjectionCastPool([("pre", m_pre)], allow_cuda=is_cuda), ValueError)

    # 3. Trainable & non-contiguous fall back safely
    m_tr = nn.Linear(8, 16, bias=False).to(device)
    x = torch.randn(2, 8, device=device, requires_grad=True)
    p_tr = FrozenProjectionCastPool([("tr", m_tr)], allow_cuda=is_cuda)
    with p_tr.scope() as st, ac_fn():
        m_tr(x).float().sum().backward()
    p_tr.close()
    checks["trainable_fallback_and_grad"] = (st.fallbacks == 1 and st.misses == 0 and m_tr.weight.grad is not None)

    m_nc = nn.Linear(8, 16, bias=False).to(device).requires_grad_(False)
    m_nc.weight = nn.Parameter(torch.randn(8, 16, device=device).T, requires_grad=False)
    p_nc = FrozenProjectionCastPool([("nc", m_nc)], allow_cuda=is_cuda)
    with p_nc.scope() as st_nc, ac_fn():
        m_nc(x)
    p_nc.close()
    checks["noncontiguous_fallback"] = (st_nc.fallbacks == 1 and st_nc.misses == 0)

    # 4. Versioned mutation & weight replacement fail closed + restore forward
    m_mut = nn.Linear(8, 16, bias=False).to(device).requires_grad_(False)
    p_mut = FrozenProjectionCastPool([("m", m_mut)], allow_cuda=is_cuda)
    mut_caught = False
    try:
        with p_mut.scope(), ac_fn():
            m_mut(x)
            with torch.no_grad():
                m_mut.weight.add_(1.0)
            m_mut(x)
    except RuntimeError as exc:
        mut_caught = "changed" in str(exc)
    checks["mutation_rejected"] = mut_caught and p_mut.poisoned and ("forward" not in m_mut.__dict__)
    p_mut.close()

    m_rep = nn.Linear(8, 16, bias=False).to(device).requires_grad_(False)
    p_rep = FrozenProjectionCastPool([("r", m_rep)], allow_cuda=is_cuda)
    rep_caught = False
    try:
        with p_rep.scope(), ac_fn():
            m_rep(x)
            m_rep.weight = nn.Parameter(torch.randn(16, 8, device=device), requires_grad=False)
            m_rep(x)
    except RuntimeError as exc:
        rep_caught = "changed" in str(exc)
    checks["weight_replacement_rejected"] = rep_caught and ("forward" not in m_rep.__dict__)
    p_rep.close()

    # 5. Nested scope rejected & closed pool rejects reentry
    m_ok = nn.Linear(8, 16, bias=False).to(device).requires_grad_(False)
    p1 = FrozenProjectionCastPool([("ok", m_ok)], allow_cuda=is_cuda)
    with p1.scope():
        checks["nested_rejected"] = raises(lambda: FrozenProjectionCastPool([("ok", m_ok)], allow_cuda=is_cuda), RuntimeError)
    p1.close()
    checks["closed_no_reentry"] = raises(lambda: p1.scope().__enter__(), RuntimeError)
    checks["restored_after_close"] = "forward" not in m_ok.__dict__

    # 6. Byte cap enforcement
    p_cap = FrozenProjectionCastPool([("ok", m_ok)], max_bytes=16, allow_cuda=is_cuda)
    with p_cap.scope() as st_cap, ac_fn():
        m_ok(x)
    p_cap.close()
    checks["byte_cap_enforced"] = (st_cap.cap_rejections == 1 and st_cap.misses == 0 and p_cap.peak_bytes == 0)

    # 7. FP32 (autocast off) & FP16 HOLD fall back cleanly
    p_hold = FrozenProjectionCastPool([("ok", m_ok)], allow_cuda=is_cuda)
    with p_hold.scope() as st_hold:
        m_ok(x)  # FP32 no autocast
        if is_cuda:
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                m_ok(x)
    p_hold.close()
    checks["fp32_and_fp16_hold_fallback"] = (
        st_hold.misses == 0 and st_hold.hits == 0 and st_hold.fallbacks == (2 if is_cuda else 1)
    )

    # 8. CUDA cross-stream ownership violation fails closed
    if is_cuda:
        p_st = FrozenProjectionCastPool([("ok", m_ok)], allow_cuda=True)
        s2 = torch.cuda.Stream(device=device)
        stream_caught = False
        try:
            with p_st.scope(), ac_fn():
                m_ok(x)
                with torch.cuda.stream(s2):
                    m_ok(x)
        except RuntimeError as exc:
            stream_caught = "stream ownership" in str(exc)
        torch.cuda.synchronize(device)
        p_st.close()
        checks["cuda_cross_stream_rejected"] = stream_caught and ("forward" not in m_ok.__dict__)

    assert all(checks.values()), f"Stage C guard failure: {checks}"
    return checks


# ---- Stage G: Pretrained Qwen3-4B Real-Weight Looped Transfer Block ----
class _Qwen3LoopedTransferModel(nn.Module):
    """2-layer looped block initialized with SHA-256-verified pretrained Qwen3-4B weights."""

    def __init__(self, layers_sd: list[dict[str, torch.Tensor]]) -> None:
        super().__init__()
        self.hidden_size = int(layers_sd[0]["q_proj"].shape[1])  # 2560
        self.q_dim = int(layers_sd[0]["q_proj"].shape[0])        # 4096
        self.kv_dim = int(layers_sd[0]["k_proj"].shape[0])       # 1024
        self.inter = int(layers_sd[0]["gate_proj"].shape[0])     # 9728
        self.num_heads = 32
        self.num_kv_heads = 8
        self.head_dim = self.q_dim // self.num_heads
        self.layers = nn.ModuleList()
        for lsd in layers_sd:
            blk = nn.Module()
            blk.q_proj = nn.Linear(self.hidden_size, self.q_dim, bias=False)
            blk.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
            blk.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
            blk.o_proj = nn.Linear(self.q_dim, self.hidden_size, bias=False)
            blk.gate_proj = nn.Linear(self.hidden_size, self.inter, bias=False)
            blk.up_proj = nn.Linear(self.hidden_size, self.inter, bias=False)
            blk.down_proj = nn.Linear(self.inter, self.hidden_size, bias=False)
            with torch.no_grad():
                for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                    getattr(blk, k).weight.copy_(lsd[k].float())
            self.layers.append(blk)

    def forward_block(self, blk: nn.Module, h: torch.Tensor) -> torch.Tensor:
        b, s, _ = h.shape
        q = blk.q_proj(h).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = blk.k_proj(h).view(b, s, self.num_kv_heads, self.head_dim).repeat_interleave(4, dim=2).transpose(1, 2)
        v = blk.v_proj(h).view(b, s, self.num_kv_heads, self.head_dim).repeat_interleave(4, dim=2).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, s, self.q_dim)
        h_mid = h + blk.o_proj(attn)
        h_norm = h_mid * torch.rsqrt(h_mid.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(h_mid.dtype)
        mlp = blk.down_proj(F.silu(blk.gate_proj(h_norm)) * blk.up_proj(h_norm))
        out = h_mid + mlp
        return out * torch.rsqrt(out.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(out.dtype)


def stage_g_pretrained_transfer(device: torch.device) -> dict[str, Any]:
    """Audit local TRM/DataGoblin availability and run real Qwen3-4B pretrained transfer."""
    # 1. Honest audit of TRM / DataGoblin checkpoints (no downloads)
    trm_status = {
        "status": "blocked_no_local_checkpoint",
        "reason": "No local gaoxin492/step_21700, yagizdevre/model.pt, complete HRM-Text-1B, or DataGoblin PyTorch checkpoint present; offline/no-download policy respected.",
    }
    shard2 = _find_local_qwen3_4b_shard2()
    if shard2 is None:
        return {"trm_or_datagoblin": trm_status, "pretrained_qwen3_4b": {"status": "unavailable"}}

    # Verify SHA-256 of shard2 on disk
    h = hashlib.sha256()
    with shard2.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    shard_sha256 = h.hexdigest()

    from safetensors import safe_open
    layers_sd = []
    with safe_open(str(shard2), framework="pt", device="cpu") as sf:
        for li in (16, 17):
            p = f"model.layers.{li}."
            layers_sd.append(
                {
                    "q_proj": sf.get_tensor(p + "self_attn.q_proj.weight"),
                    "k_proj": sf.get_tensor(p + "self_attn.k_proj.weight"),
                    "v_proj": sf.get_tensor(p + "self_attn.v_proj.weight"),
                    "o_proj": sf.get_tensor(p + "self_attn.o_proj.weight"),
                    "gate_proj": sf.get_tensor(p + "mlp.gate_proj.weight"),
                    "up_proj": sf.get_tensor(p + "mlp.up_proj.weight"),
                    "down_proj": sf.get_tensor(p + "mlp.down_proj.weight"),
                }
            )

    ac_fn = _bf16_ac(device)
    is_cuda = device.type == "cuda"
    results = []
    for visits in (6, 12):
        torch.manual_seed(14701 + visits)
        if is_cuda:
            torch.cuda.manual_seed_all(14701 + visits)
        xs = [torch.randn(2, 16, 2560, device=device) * 0.1 for _ in range(2)]
        cots = [torch.randn(2, 16, 2560, device=device) for _ in range(2)]

        arm_out = {}
        for arm_name, use_proj_cache in (("fuse13_arm_g", False), ("fuse14_arm_h", True)):
            if is_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            torch.manual_seed(14750 + visits)
            if is_cuda:
                torch.cuda.manual_seed_all(14750 + visits)
            base = _Qwen3LoopedTransferModel(layers_sd)
            sites = {f"layers.{i}.{n}": 64 for i in range(2) for n in ("o_proj", "down_proj")}
            bpd = BPDModel(base, sites, max_passes=visits, ablation=AblationConfig(mode="zero", sphere_rescale=False), init_svd=False).to(device)
            # Freeze base projections, keep BPD (V, U) trainable
            opt = torch.optim.AdamW(bpd.component_parameters(), lr=1e-3)
            from .step_cache import StepResidualPool
            res_cap = sum(w.W_target.numel() * 2 for w in bpd.wrappers.values())
            proj_mods = eligible_frozen_projections(bpd, targets=("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"))
            proj_sigs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            for _, m in proj_mods:
                out_f, in_f = int(m.weight.shape[0]), int(m.weight.shape[1])
                proj_sigs.add(((out_f, in_f), (in_f, 1)))
                proj_sigs.add(((in_f, out_f), (1, in_f)))

            saved_ptrs: set[int] = set()
            saved_bytes_map: dict[int, int] = {}

            def _pack(t: torch.Tensor):
                if t.dtype == torch.bfloat16 and (tuple(t.shape), tuple(t.stride())) in proj_sigs:
                    st = t.untyped_storage()
                    ptr = int(st.data_ptr())
                    saved_ptrs.add(ptr)
                    saved_bytes_map[ptr] = int(st.nbytes())
                return t

            # Run 1 warmup + 1 timed & audited step
            torch.manual_seed(14800 + visits)
            sched_map = {
                nm: PassMaskSchedule(
                    torch.rand(2, visits, 16, 64, device=device),
                    torch.rand(2, visits, 16, device=device),
                )
                for nm in bpd.module_order
            }
            step_times = []
            for s_idx in range(2):
                opt.zero_grad(set_to_none=True)
                inps = [x.detach().clone().requires_grad_(True) for x in xs]
                res_pool = StepResidualPool(
                    list(bpd.wrappers.values()),
                    max_bytes=res_cap,
                    allow_cuda=is_cuda,
                    precision_mode="bf16" if is_cuda else "fp32",
                    storage_mode="compute_dtype" if is_cuda else "fp32",
                ) if is_cuda else None
                proj_pool = (
                    FrozenProjectionCastPool(proj_mods, allow_cuda=is_cuda)
                    if use_proj_cache
                    else None
                )
                if s_idx == 1:
                    saved_ptrs.clear()
                    saved_bytes_map.clear()
                hook_ctx = torch.autograd.graph.saved_tensors_hooks(_pack, lambda t: t) if s_idx == 1 else nullcontext()
                if is_cuda:
                    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(device)
                    ev0.record()
                else:
                    t0 = time.perf_counter()
                try:
                    with hook_ctx, ac_fn():
                        ys = []
                        for r_idx, x_in in enumerate(inps):
                            with bpd.masked_components(sched_map, pass_offset=0):
                                with (proj_pool.scope() if proj_pool else nullcontext()), (res_pool.scope() if res_pool else nullcontext()):
                                    h_cur = x_in
                                    for _ in range(visits):
                                        for blk in base.layers:
                                            h_cur = base.forward_block(blk, h_cur)
                                    ys.append(h_cur)
                        loss = sum((y.float() * c).sum() for y, c in zip(ys, cots))
                    loss.backward()
                finally:
                    if res_pool is not None:
                        res_pool.close()
                    if proj_pool is not None:
                        proj_pool.close()
                opt.step()
                if is_cuda:
                    ev1.record()
                    torch.cuda.synchronize(device)
                    step_times.append(float(ev0.elapsed_time(ev1)))
                else:
                    step_times.append((time.perf_counter() - t0) * 1000.0)

            peak_alloc = int(torch.cuda.max_memory_allocated(device)) if is_cuda else 0
            peak_res = int(torch.cuda.max_memory_reserved(device)) if is_cuda else 0
            nvml_used = float(query_nvml_snapshot().get("memory_used_mib") or 0.0) if is_cuda else 0.0
            arm_out[arm_name] = {
                "tensors": {
                    "loss": loss.detach().cpu(),
                    **{f"y{i}": y.detach().cpu() for i, y in enumerate(ys)},
                    **{f"gx{i}": x_in.grad.detach().cpu() for i, x_in in enumerate(inps)},
                    **{f"p/{k}": v.detach().cpu() for k, v in bpd.state_dict().items()},
                },
                "step_ms": step_times,
                "unique_proj_cast_storages": len(saved_ptrs),
                "unique_proj_cast_bytes": sum(saved_bytes_map.values()),
                "peak_allocated_bytes": peak_alloc,
                "peak_reserved_bytes": peak_res,
                "nvml_used_mib": nvml_used,
                "proj_receipt": proj_pool.receipt() if proj_pool else None,
            }
            del bpd, base, opt
            gc.collect()
            if is_cuda:
                torch.cuda.empty_cache()

        cmp_res = _compare_snapshots(arm_out["fuse13_arm_g"]["tensors"], arm_out["fuse14_arm_h"]["tensors"])
        g_meta = {k: v for k, v in arm_out["fuse13_arm_g"].items() if k != "tensors"}
        h_meta = {k: v for k, v in arm_out["fuse14_arm_h"].items() if k != "tensors"}
        results.append(
            {
                "checkpoint_family": "pretrained_qwen3_4b_block",
                "layers": [16, 17],
                "hidden_size": 2560,
                "intermediate_size": 9728,
                "visits": visits,
                "fuse13_arm_g": g_meta,
                "fuse14_arm_h": h_meta,
                "parity": cmp_res,
                "vram_saved_alloc_mib": (g_meta["peak_allocated_bytes"] - h_meta["peak_allocated_bytes"]) / (1024.0 ** 2),
                "vram_saved_nvml_mib": g_meta["nvml_used_mib"] - h_meta["nvml_used_mib"],
            }
        )

    return {
        "trm_or_datagoblin": trm_status,
        "pretrained_qwen3_4b": {
            "status": "ok",
            "source_model": "Qwen/Qwen3-4B (local cached shard 2/3, layers 16-17, no download)",
            "shard_sha256": shard_sha256,
            "cells": results,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--allow-cuda", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-watts", type=float, default=60.0)
    p.add_argument("--stage-g-only", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    t0 = time.perf_counter()

    if device.type == "cuda":
        if not args.allow_cuda:
            p.error("CUDA suite requires --allow-cuda inside VRAMorama lease")
        torch.cuda.init()
        torch.cuda.set_device(device)
        snap_nvml = check_gpu_safety(min_watts=args.min_watts, min_free_mib=2048.0)
    else:
        torch.set_num_threads(1)
        snap_nvml = {}

    if args.stage_g_only:
        if device.type != "cuda" or not args.output.exists():
            raise RuntimeError("--stage-g-only requires --device cuda and an existing --output receipt")
        print("[fuse14] Stage G only: pretrained checkpoint transfer (Qwen3-4B) ...", flush=True)
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        stage_g = stage_g_pretrained_transfer(device)
        configure_precision_mode("fp32")
        existing["stage_g_pretrained_transfer"] = stage_g
        args.output.write_text(json.dumps(existing, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        cells = stage_g.get("pretrained_qwen3_4b", {}).get("cells", [])
        n_ok = sum(int(c["parity"]["finite_bitwise_pass"]) for c in cells)
        print("FUSE14_STAGE_G_DONE", json.dumps({"stage_g_qwen3_bitwise": f"{n_ok}/{len(cells)}"}), flush=True)
        for c in cells:
            if not c["parity"]["finite_bitwise_pass"]:
                raise AssertionError(f"Stage G parity failed visits={c['visits']}: {c['parity']['failure_examples'][:3]}")
        return 0

    # Stage C: Lifecycle & Fail-Closed Guards
    print("[fuse14] Stage C: lifecycle guards ...", flush=True)
    guards_res = stage_c_lifecycle_guards(device)

    # Stage A: Standalone probe replay + trainer per-layer cast audit
    print("[fuse14] Stage A: standalone probe + trainer cast audit ...", flush=True)
    probe_rows = [probe_run_cell(args.device, w, t, s) for w in (16, 32, 64) for t in (2, 6, 12) for s in (14001, 14002)]
    assert all(r["finite"] and r["bitwise"] and r["misses"] == 2 for r in probe_rows)
    probe_g = probe_guards(args.device)
    audit_w, audit_t = (512, 12) if device.type == "cuda" else (64, 6)
    sa_ref, ma_ref = _run_trainer_and_capture(
        device=device, width=audit_w, n_passes=audit_t, seed=14101,
        regime="closed_loop", cal_cadence="1", budget_mode="full",
        cache_residuals=True, cache_projections=False, n_steps=2, audit_casts=True,
    )
    sa_cand, ma_cand = _run_trainer_and_capture(
        device=device, width=audit_w, n_passes=audit_t, seed=14101,
        regime="closed_loop", cal_cadence="1", budget_mode="full",
        cache_residuals=True, cache_projections=True, n_steps=2, audit_casts=True,
    )
    stage_a = {
        "standalone_probe": {
            "cells": len(probe_rows),
            "finite_bitwise": sum(int(r["finite"] and r["bitwise"]) for r in probe_rows),
            "d12_storages": [(r["native_saved_shape_storages"], r["candidate_saved_shape_storages"]) for r in probe_rows if r["width"] == 16 and r["visits"] == 12],
            "guards": probe_g,
        },
        "trainer_cast_audit": {
            "width": audit_w,
            "visits": audit_t,
            "fuse13_arm_g": ma_ref,
            "fuse14_arm_h": ma_cand,
            "parity": _compare_snapshots(sa_ref, sa_cand),
        },
    }

    # Stage B: Isolation Ablation inside BPDTrainer
    print("[fuse14] Stage B: isolation ablation ...", flush=True)
    ab_w, ab_t = (512, 12) if device.type == "cuda" else (64, 6)
    ab_arms = [
        ("fuse13_arm_g", dict(cache_projections=False, autocast_cache_enabled=True, explicit_per_call_cast=False)),
        ("autocast_cache_disabled", dict(cache_projections=False, autocast_cache_enabled=False, explicit_per_call_cast=False)),
        ("explicit_per_call_cast", dict(cache_projections=False, autocast_cache_enabled=True, explicit_per_call_cast=True)),
        ("qkv_only", dict(cache_projections=True, projection_targets="qkv_only")),
        ("gate_up_only", dict(cache_projections=True, projection_targets="gate_up_only")),
        ("both_block_projs", dict(cache_projections=True, projection_targets="block_projs")),
    ]
    stage_b = {}
    ref_b_snap = None
    for aname, akw in ab_arms:
        if device.type == "cuda":
            check_gpu_safety(min_watts=args.min_watts, min_free_mib=2048.0)
        sn, mt = _run_trainer_and_capture(
            device=device, width=ab_w, n_passes=ab_t, seed=14201,
            regime="closed_loop", cal_cadence="1", budget_mode="full",
            cache_residuals=True, n_steps=2, audit_casts=True, **akw,
        )
        if aname == "fuse13_arm_g":
            ref_b_snap = sn
            par = None
        else:
            par = _compare_snapshots(ref_b_snap, sn)
        stage_b[aname] = {"meta": mt, "parity_vs_arm_g": par}

    # Stage D: Frozen Multi-Step Parity Matrix (uninstrumented timing + exact finite bytes)
    print("[fuse14] Stage D: frozen parity matrix ...", flush=True)
    if device.type == "cuda":
        d_grid = [
            # widths 128/512/1024/2048 x visits 2/6/12/24, two roots, both regimes, cal off/1/50, binding/full
            (128, 2, 14001, "closed_loop", "1", "full"),
            (128, 6, 14001, "closed_loop", "1", "full"),
            (128, 6, 14002, "clean_window", "50", "binding"),
            (512, 6, 14001, "closed_loop", "1", "binding"),
            (512, 12, 14001, "closed_loop", "off", "full"),
            (512, 12, 14002, "clean_window", "50", "full"),
            (1024, 6, 14001, "closed_loop", "1", "binding"),
            (1024, 12, 14001, "closed_loop", "off", "full"),
            (1024, 24, 14002, "closed_loop", "off", "full"),
            (2048, 6, 14001, "closed_loop", "1", "binding"),
            (2048, 12, 14001, "closed_loop", "off", "full"),
            (2048, 12, 14002, "clean_window", "50", "full"),
            (2048, 24, 14001, "closed_loop", "off", "full"),
        ]
    else:
        d_grid = [
            (64, 2, 14001, "closed_loop", "1", "full"),
            (64, 6, 14002, "clean_window", "50", "binding"),
        ]
    stage_d = []
    for w, t, seed, reg, cal, bud in d_grid:
        if device.type == "cuda":
            check_gpu_safety(min_watts=args.min_watts, min_free_mib=2048.0)
        sg, mg = _run_trainer_and_capture(
            device=device, width=w, n_passes=t, seed=seed,
            regime=reg, cal_cadence=cal, budget_mode=bud,
            cache_residuals=True, cache_projections=False, n_steps=2, audit_casts=False,
        )
        sh, mh = _run_trainer_and_capture(
            device=device, width=w, n_passes=t, seed=seed,
            regime=reg, cal_cadence=cal, budget_mode=bud,
            cache_residuals=True, cache_projections=True, n_steps=2, audit_casts=False,
        )
        par = _compare_snapshots(sg, sh)
        stage_d.append(
            {
                "width": w,
                "visits": t,
                "seed": seed,
                "regime": reg,
                "cal_cadence": cal,
                "budget_mode": bud,
                "fuse13_arm_g": mg,
                "fuse14_arm_h": mh,
                "parity": par,
                "vram_saved_alloc_mib": (mg["peak_allocated_bytes"] - mh["peak_allocated_bytes"]) / (1024.0 ** 2),
                "vram_saved_nvml_mib": mg["nvml_used_mib"] - mh["nvml_used_mib"],
            }
        )
        print(
            f"  [stage-d] w={w} t={t} r={seed} {reg} cal={cal} bud={bud}: "
            f"bitwise={par['finite_bitwise_pass']} ({par['finite_bitwise']}/{par['tensors']}) "
            f"dAlloc={stage_d[-1]['vram_saved_alloc_mib']:+.1f} MiB",
            flush=True,
        )

    # FP16 HOLD confirmation inside trainer
    fp16_hold_check = None
    if device.type == "cuda":
        _, m_fp16 = _run_trainer_and_capture(
            device=device, width=128, n_passes=6, seed=14999,
            regime="closed_loop", cal_cadence="off", budget_mode="full",
            cache_residuals=True, cache_projections=True, precision_mode="fp16", n_steps=1,
        )
        rec_fp16 = m_fp16["projection_cache_receipt"]
        fp16_hold_check = {
            "status": "HOLD_CONFIRMED" if (rec_fp16["hits"] == 0 and rec_fp16["misses"] == 0 and rec_fp16["fallbacks"] > 0) else "UNEXPECTED",
            "receipt": rec_fp16,
        }

    # Stage E: Independent Activation Checkpointing
    print("[fuse14] Stage E: independent activation checkpointing ...", flush=True)
    ckpt_grid = [(512, 12), (2048, 24)] if device.type == "cuda" else [(64, 6)]
    stage_e = []
    for w, t in ckpt_grid:
        if device.type == "cuda":
            check_gpu_safety(min_watts=args.min_watts, min_free_mib=2048.0)
        # Native checkpointing (residual cache OFF, projection cache OFF) vs Checkpointing + recompute-safe frozen projection casts
        sc_ref, mc_ref = _run_trainer_and_capture(
            device=device, width=w, n_passes=t, seed=14501,
            regime="closed_loop", cal_cadence="off", budget_mode="full",
            cache_residuals=False, cache_projections=False, enable_checkpointing=True, n_steps=2, audit_casts=False,
        )
        sc_cand, mc_cand = _run_trainer_and_capture(
            device=device, width=w, n_passes=t, seed=14501,
            regime="closed_loop", cal_cadence="off", budget_mode="full",
            cache_residuals=False, cache_projections=True, allow_recompute=True, enable_checkpointing=True, n_steps=2, audit_casts=False,
        )
        par_e = _compare_snapshots(sc_ref, sc_cand)
        stage_e.append(
            {
                "width": w,
                "visits": t,
                "ckpt_native": mc_ref,
                "ckpt_plus_frozen_proj": mc_cand,
                "parity": par_e,
                "vram_saved_alloc_mib": (mc_ref["peak_allocated_bytes"] - mc_cand["peak_allocated_bytes"]) / (1024.0 ** 2),
                "vram_saved_nvml_mib": mc_ref["nvml_used_mib"] - mc_cand["nvml_used_mib"],
            }
        )

    # Stage G: Pretrained Checkpoint Transfer (CUDA only; CPU skips 4B shard load)
    stage_g = None
    if device.type == "cuda":
        print("[fuse14] Stage G: pretrained checkpoint transfer (Qwen3-4B) ...", flush=True)
        check_gpu_safety(min_watts=args.min_watts, min_free_mib=2048.0)
        stage_g = stage_g_pretrained_transfer(device)
        configure_precision_mode("fp32")

    out = {
        "campaign": "FUSE-14",
        "device": args.device,
        "torch": str(torch.__version__),
        "nvml": snap_nvml,
        "stage_c_guards": guards_res,
        "stage_a": stage_a,
        "stage_b_ablation": stage_b,
        "stage_d_parity_matrix": stage_d,
        "fp16_hold_confirmation": fp16_hold_check,
        "stage_e_checkpointing": stage_e,
        "stage_g_pretrained_transfer": stage_g,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {
        "device": args.device,
        "guards_all_pass": all(guards_res.values()),
        "probe_bitwise": f"{stage_a['standalone_probe']['finite_bitwise']}/{stage_a['standalone_probe']['cells']}",
        "stage_d_bitwise": f"{sum(int(c['parity']['finite_bitwise_pass']) for c in stage_d)}/{len(stage_d)}",
        "stage_e_bitwise": f"{sum(int(c['parity']['finite_bitwise_pass']) for c in stage_e)}/{len(stage_e)}",
        "fp16_hold": fp16_hold_check["status"] if fp16_hold_check else "cpu_skipped",
        "stage_g_qwen3_bitwise": (
            f"{sum(int(c['parity']['finite_bitwise_pass']) for c in stage_g['pretrained_qwen3_4b']['cells'])}/{len(stage_g['pretrained_qwen3_4b']['cells'])}"
            if stage_g and stage_g.get("pretrained_qwen3_4b", {}).get("status") == "ok"
            else "skipped"
        ),
    }
    print("FUSE14_SUITE_DONE", json.dumps(summary), flush=True)
    if not stage_a["trainer_cast_audit"]["parity"]["finite_bitwise_pass"]:
        raise AssertionError(f"Stage A parity failed: {stage_a['trainer_cast_audit']['parity']['failure_examples'][:3]}")
    if not stage_b["both_block_projs"]["parity_vs_arm_g"]["finite_bitwise_pass"]:
        raise AssertionError(f"Stage B both_block_projs parity failed: {stage_b['both_block_projs']['parity_vs_arm_g']['failure_examples'][:3]}")
    for c in stage_d:
        if not c["parity"]["finite_bitwise_pass"]:
            raise AssertionError(f"Stage D parity failed w={c['width']} t={c['visits']}: {c['parity']['failure_examples'][:3]}")
    for c in stage_e:
        if not c["parity"]["finite_bitwise_pass"]:
            raise AssertionError(f"Stage E parity failed w={c['width']} t={c['visits']}: {c['parity']['failure_examples'][:3]}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as _exc:
        import traceback
        tb_last = traceback.format_exc().strip().splitlines()
        msg = " | ".join(tb_last[-4:])
        print(f"::error::{msg}", flush=True)
        raise
