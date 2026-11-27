"""FUSE-08: bounded, step-scoped detached residual primals.

Each native residual request gets its own first-order adjoint. Only the
original float32 primal storage is shared. Use a single sequential optimizer
step, no checkpoint/reentry/concurrency, no external storage writes.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType

import torch

from .adjoint_cache import Stats, _Primal, _key
from .admission import Site, select


_MISSING = object()


def supported(w):
    ts = (w.W_target, w.V, w.U)
    return (
        torch.is_grad_enabled()
        and not torch.is_inference_mode_enabled()
        and not torch.is_autocast_enabled("cpu")
        and torch.get_float32_matmul_precision() == "highest"
        and not w.W_target.requires_grad
        and not getattr(w, "use_fused_residual", False)
        and all(t.dtype == torch.float32 and t.device.type == "cpu"
                and t.ndim == 2 and t.is_contiguous() and not t.is_inference()
                for t in ts)
    )


def planned_sites(wrappers, passes, cap, *, max_frontier=65536):
    """Stable caller order; unsupported sites have zero value and stay native."""
    ws = tuple(wrappers)
    if len({id(w) for w in ws}) != len(ws):
        raise ValueError("duplicate wrapper")
    sites = tuple(Site.residual(w.d_in, w.d_out, w.C, passes)
                  if supported(w) else Site(4*w.d_in*w.d_out, 0)
                  for w in ws)
    plan = select(sites, cap, "exact", max_frontier=max_frontier)
    return tuple(ws[i] for i in plan.indices), plan


@dataclass
class StepScopeStats(Stats):
    cross_scope_hits: int = 0
    mask_only_calls: int = 0
    independent_calls: int = 0
    storage_checks: int = 0
    pool_bytes_after_scope: int = 0
    # Stable pool-local slot IDs, not process addresses.
    slots: list[int] = field(default_factory=list)


class StepResidualPool:
    """One fixed selected set and no eviction until joint backward completes.

    The cap counts distinct detached residual storages allocated by this pool,
    including their aliases retained by outstanding graphs. It does not count
    unselected native residuals, activations, autograd metadata, or total RSS.
    """

    def __init__(self, wrappers, *, max_bytes):
        self.ws = tuple(wrappers)
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("invalid cache cap")
        if len({id(w) for w in self.ws}) != len(self.ws):
            raise ValueError("duplicate wrapper")
        self.max_bytes = max_bytes
        self.entries = {}
        self.slots = {id(w): i for i, w in enumerate(self.ws)}
        self.resident_bytes = 0
        self.peak_bytes = 0
        self.scope_count = 0
        self.active = False
        self.closed = False
        self.sealed = False
        self.poisoned = False

    def _verify(self, w, entry):
        key, p, version, held, first_scope = entry
        ts = (w.W_target, w.V, w.U)
        if (any(a is not b for a, b in zip(ts, held))
                or any(t.is_inference() for t in ts)
                or key != _key(w) or version != p._version):
            self.poisoned = True
            raise RuntimeError("step residual parameter or cached primal mutated")

    def close(self):
        if self.active:
            raise RuntimeError("cannot close active residual scope")
        self.closed = True
        self.entries.clear()
        self.resident_bytes = 0

    @contextmanager
    def scope(self, *, mask_only: bool = False):
        """Private trainer contract: mask_only only for discarded source-gradient graphs.

        Detached residuals preserve derivatives with respect to activations and
        masks, but deliberately omit residual factor derivatives. Never admit
        such a graph to the joint training loss. No backward-seal hook exists
        for detached outputs; the native PGD source-only caller owns this scope.
        """
        if type(mask_only) is not bool:
            raise TypeError("mask_only must be bool")
        if self.closed or self.active or self.sealed or self.poisoned:
            raise RuntimeError("unavailable step residual pool")
        if any(getattr(w, "_residual_cache_active", False) for w in self.ws):
            raise RuntimeError("nested residual scope")
        for w in self.ws:
            old = self.entries.get(id(w))
            if old is not None:
                self._verify(w, old)
        self.scope_count += 1
        ordinal = self.scope_count
        stats = StepScopeStats(peak_bytes=self.resident_bytes)
        originals = []
        self.active = True

        def on_backward(g):
            self.sealed = True
            if self.active:
                self.poisoned = True
                stats.consumed = True
                raise RuntimeError("backward during step residual forward scope")
            return g

        def build(w, original):
            def get(self_w):
                stats.calls += 1
                if self.closed or self.sealed or self.poisoned or not self.active:
                    raise RuntimeError("unavailable step residual pool")
                if getattr(self_w, "use_fused_residual", False):
                    self.poisoned = True
                    raise RuntimeError("fusion enabled in step scope")
                wid = id(self_w)
                old = self.entries.get(wid)
                if old is not None:
                    self._verify(self_w, old)
                def bypass():
                    stats.bypasses += 1
                    stats.reconstructions += 1
                    return original()
                if not supported(self_w):
                    return bypass()
                if old is None:
                    n = self_w.W_target.numel()*self_w.W_target.element_size()
                    if self.resident_bytes+n > self.max_bytes:
                        stats.cap_rejections += 1
                        return bypass()
                    with torch.no_grad():
                        p = original()
                    if p.numel()*p.element_size() != n or p.untyped_storage().nbytes() != n:
                        self.poisoned = True
                        raise RuntimeError("unexpected residual storage size")
                    # New storages must be distinct from every live cached site.
                    if any(p.untyped_storage().data_ptr() == e[1].untyped_storage().data_ptr()
                           for e in self.entries.values()):
                        self.poisoned = True
                        raise RuntimeError("unexpected cross-site storage alias")
                    self.entries[wid] = (_key(self_w), p, p._version,
                                         (self_w.W_target,self_w.V,self_w.U), ordinal)
                    self.resident_bytes += n
                    self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
                    stats.misses += 1
                    stats.reconstructions += 1
                else:
                    p = old[1]
                    stats.hits += 1
                    stats.cross_scope_hits += int(old[4] != ordinal)
                stats.peak_bytes = max(stats.peak_bytes, self.resident_bytes)
                if mask_only:
                    # Only the native inner PGD loss requests source-leaf grads.
                    # Keep all input/mask dependencies in the native F.linear.
                    value = p
                    stats.mask_only_calls += 1
                else:
                    value = _Primal.apply(self_w.V, self_w.U, p, "residual", stats)
                    stats.independent_calls += 1
                if value.untyped_storage().data_ptr() != p.untyped_storage().data_ptr():
                    self.poisoned = True
                    raise RuntimeError("custom adjoint did not retain cached storage")
                stats.storage_checks += 1
                slot = self.slots[wid]
                if slot not in stats.slots:
                    stats.slots.append(slot)
                if value.requires_grad:
                    value.register_hook(on_backward)
                return value
            return MethodType(get, w)
        try:
            for w in self.ws:
                original = w.weight_delta
                originals.append((w, w.__dict__.get("weight_delta", _MISSING),
                                  w.__dict__.get("_residual_cache_active", _MISSING)))
                w._residual_cache_active = True
                w.weight_delta = build(w, original)
            yield stats
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self.active = False
            stats.pool_bytes_after_scope = self.resident_bytes
            # No scope-owned storage. The pool still owns its cached primals.
            stats.resident_bytes = 0
            for w, method, flag in reversed(originals):
                if method is _MISSING:
                    del w.weight_delta
                else:
                    w.weight_delta = method
                if flag is _MISSING:
                    del w._residual_cache_active
                else:
                    w._residual_cache_active = flag
