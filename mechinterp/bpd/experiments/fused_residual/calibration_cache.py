"""FUSE-10: bounded native residual reuse for clean pass-mean observations.

One sequential no_grad target_capture only; no external writes, concurrent
access, checkpointing or parameter changes. Never shares a training pool.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch

from mechinterp.bpd.components import TiedComponentLinear
from .admission import Site, select
from .adjoint_cache import _key
from .cuda_guard import (
    current_stream_token,
    device_and_precision_supported,
    target_storage_dtype,
    validate_cuda_options,
)


def _supported(
    w,
    *,
    allow_cuda: bool = False,
    precision_mode: str = "fp32",
    storage_mode: str = "fp32",
):
    validate_cuda_options(allow_cuda, precision_mode, storage_mode)
    if type(w) is not TiedComponentLinear:
        return False
    ts = (w.W_target, w.V, w.U)
    return (
        not torch.is_grad_enabled()
        and not torch.is_inference_mode_enabled()
        and not w.W_target.requires_grad
        and not getattr(w, "use_fused_residual", False)
        and w.mode == "target" and w.update_running_means
        and w.intervention is None
        and device_and_precision_supported(
            ts, allow_cuda=allow_cuda, precision_mode=precision_mode
        )
    )


@contextmanager
def calibration_residuals(
    named_wrappers,
    *,
    calls,
    max_bytes,
    allow_cuda: bool = False,
    precision_mode: str = "fp32",
    storage_mode: str = "fp32",
):
    """Cache only original materialized residuals for one clean rollout.

    Planner values assume one residual request per site per pass. Actual
    hits/misses are measured. Cap is cached payload only, not total memory.
    """
    validate_cuda_options(allow_cuda, precision_mode, storage_mode)
    named = tuple(named_wrappers)
    if type(calls) is not int or calls < 0:
        raise ValueError("invalid planned calls")
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("invalid cache cap")
    if len({id(w) for _, w in named}) != len(named):
        raise ValueError("duplicate wrapper")
    if any(getattr(w, "_residual_cache_active", False) for _, w in named):
        raise RuntimeError("nested residual scope")
    storage_dt = target_storage_dtype(precision_mode, storage_mode)
    elem_bytes = storage_dt.itemsize
    use_low = storage_mode == "compute_dtype" and storage_dt != torch.float32
    stats = dict(status="empty", planned_calls=calls, selected=[], planned_bytes=0,
                 planned_saved_macs=0, calls=0, hits=0, misses=0, bypasses=0,
                 reconstructions=0, peak_bytes=0, resident_bytes=0, closed=False)
    eligible = [(name, w) for name, w in named
                if _supported(
                    w,
                    allow_cuda=allow_cuda,
                    precision_mode=precision_mode,
                    storage_mode=storage_mode,
                )
                and "weight_delta" not in w.__dict__
                and "reconstructed_weight" not in w.__dict__]
    try:
        plan = select([Site.residual(w.d_in, w.d_out, w.C, calls, elem_bytes=elem_bytes)
                       for _, w in eligible], max_bytes, "exact")
    except RuntimeError as exc:
        if str(exc) != "frontier limit exceeded":
            raise
        stats["status"] = "frontier_limit"
        try:
            yield stats
        finally:
            stats["closed"] = True
        return
    chosen = [eligible[i] for i in plan.indices]
    stats.update(status="selected" if chosen else "empty",
                 selected=[name for name, _ in chosen],
                 planned_bytes=plan.size, planned_saved_macs=plan.value)
    entries = {}
    originals = []
    active = True
    owner_stream = None

    def check_stream(device):
        nonlocal owner_stream
        if device.type != "cuda":
            return
        tok = current_stream_token(device)
        if owner_stream is None:
            owner_stream = tok
        elif tok != owner_stream:
            raise RuntimeError("CUDA stream ownership violation")

    def verify(w, entry):
        key, p, version, held = entry
        ts = (w.W_target, w.V, w.U)
        check_stream(w.W_target.device)
        if (any(a is not b for a, b in zip(ts, held))
                or any(t.is_inference() for t in ts)
                or key != _key(w) or p._version != version):
            raise RuntimeError("calibration residual parameter or cached primal mutated")

    def bind(w, original):
        def get(self_w):
            if not active:
                raise RuntimeError("closed calibration scope")
            stats["calls"] += 1
            wid = id(self_w)
            old = entries.get(wid)
            if old is not None:
                verify(self_w, old)
            if not _supported(
                self_w,
                allow_cuda=allow_cuda,
                precision_mode=precision_mode,
                storage_mode=storage_mode,
            ):
                stats["bypasses"] += 1
                stats["reconstructions"] += 1
                return original()
            check_stream(self_w.W_target.device)
            if old is not None:
                stats["hits"] += 1
                return old[1]
            n = self_w.W_target.numel() * elem_bytes
            if stats["resident_bytes"] + n > max_bytes:
                stats["bypasses"] += 1
                stats["reconstructions"] += 1
                return original()
            raw_p = original()
            p = raw_p.to(storage_dt) if use_low else raw_p
            if (p.requires_grad or p.is_inference()
                    or p.shape != self_w.W_target.shape
                    or p.dtype != storage_dt or p.device != self_w.W_target.device
                    or p.numel() * p.element_size() != n
                    or p.untyped_storage().nbytes() != n
                    or any(p.untyped_storage().data_ptr() == e[1].untyped_storage().data_ptr()
                           for e in entries.values())):
                raise RuntimeError("unexpected calibration residual storage")
            entries[wid] = (_key(self_w), p, p._version,
                            (self_w.W_target, self_w.V, self_w.U))
            stats["misses"] += 1
            stats["reconstructions"] += 1
            stats["resident_bytes"] += n
            stats["peak_bytes"] = max(stats["peak_bytes"], stats["resident_bytes"])
            return p
        return MethodType(get, w)

    missing = object()
    try:
        for _, w in chosen:
            original = w.weight_delta
            originals.append((w, w.__dict__.get("_residual_cache_active", missing)))
            w._residual_cache_active = True
            w.weight_delta = bind(w, original)
        yield stats
        for _, w in chosen:
            if id(w) in entries:
                verify(w, entries[id(w)])
    finally:
        active = False
        for w, flag in reversed(originals):
            if "weight_delta" in w.__dict__:
                del w.weight_delta
            if flag is missing:
                del w._residual_cache_active
            else:
                w._residual_cache_active = flag
        entries.clear()
        stats["resident_bytes"] = 0
        stats["closed"] = True
