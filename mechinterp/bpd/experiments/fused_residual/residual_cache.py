"""Experimental single-rollout cache of the native residual reconstruction.

Only use for sequential, fixed-weight, non-checkpointed forward graphs. A train
scope must finish before its single backward; eval scope requires no_grad.
Unsafe .data writes and concurrent/reentrant use are outside the contract.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType

import torch


@dataclass
class CacheStats:
    calls: int = 0
    hits: int = 0
    misses: int = 0
    bypasses: int = 0
    reconstructions: int = 0
    invalidations: int = 0
    cap_rejections: int = 0
    resident_bytes: int = 0
    peak_bytes: int = 0
    consumed: bool = False


_MISSING = object()


def _key(w):
    return tuple((id(t), t._version, tuple(t.shape), t.dtype, t.device,
                  t.requires_grad) for t in (w.W_target, w.V, w.U))


@contextmanager
def cached_residuals(wrappers, *, mode='train', max_bytes=16 * 1024**2):
    """Cache the original weight_delta tensor, never the fused algebra.

    In train mode use exactly one forward graph per context, and backward only
    after exiting. An early residual-gradient hook marks the scope consumed.
    This is a diagnostic, not a complete autograd-lifecycle detector. In eval
    mode only no_grad calls cache. Other grad modes bypass, never poison cache.
    Dense numeric payload only is capped; autograd/allocator memory is not.
    """
    if mode not in ('train', 'eval') or type(max_bytes) is not int or max_bytes < 0:
        raise ValueError('invalid cache contract')
    ws = list(wrappers)
    if len(set(map(id, ws))) != len(ws):
        raise ValueError('duplicate wrapper')
    if any(getattr(w, '_residual_cache_active', False) for w in ws):
        raise RuntimeError('nested residual cache scope')
    if any(getattr(w, 'use_fused_residual', False) for w in ws):
        raise ValueError('disable reassociated fusion before caching')
    stats = CacheStats()
    entries = {}
    originals = []
    active = True

    def discard(wid):
        entry = entries.pop(wid, None)
        if entry is not None:
            stats.resident_bytes -= entry[2]

    def consumed(grad):
        if active:
            stats.consumed = True
            for wid in tuple(entries):
                discard(wid)
        return grad

    def build(w, original):
        def get(self):
            stats.calls += 1
            if stats.consumed:
                raise RuntimeError('residual cache used after backward in scope')
            def uncached():
                stats.bypasses += 1
                stats.reconstructions += 1
                return original()
            wid = id(self)
            if getattr(self, 'use_fused_residual', False):
                raise RuntimeError('fusion enabled inside residual cache scope')
            supported = (all(t.device.type == 'cpu' and t.dtype == torch.float32
                             and not t.is_inference() for t in (self.W_target,self.V,self.U))
                         and not torch.is_autocast_enabled('cpu')
                         and not torch.is_inference_mode_enabled()
                         and torch.get_float32_matmul_precision() == 'highest')
            # Check existing entry even on a bypass: a detectable mutation is
            # incompatible with the train scope's fixed-weight contract.
            old = entries.get(wid)
            if old is not None:
                if any(t.is_inference() for t in (self.W_target,self.V,self.U)) or old[0] != _key(self):
                    discard(wid)
                    stats.invalidations += 1
                    if mode == 'train':
                        raise RuntimeError('parameter/buffer changed during train cache scope')
                    old = None
            if not supported or torch.is_grad_enabled() != (mode == 'train'):
                return uncached()
            if old is not None:
                stats.hits += 1
                return old[1]
            nbytes = self.W_target.numel() * self.W_target.element_size()
            if stats.resident_bytes + nbytes > max_bytes:
                stats.cap_rejections += 1
                return uncached()
            value = original()
            stats.reconstructions += 1
            stats.misses += 1
            if value.numel()*value.element_size() != nbytes:
                raise RuntimeError('unexpected residual payload')
            if mode == 'train' and value.requires_grad:
                value.register_hook(consumed)
            entries[wid] = (_key(self), value, nbytes)
            stats.resident_bytes += nbytes
            stats.peak_bytes = max(stats.peak_bytes, stats.resident_bytes)
            return value
        return MethodType(get, w)

    try:
        for w in ws:
            saved = w.__dict__.get('weight_delta', _MISSING)
            marker = w.__dict__.get('_residual_cache_active', _MISSING)
            orig = w.weight_delta
            originals.append((w,saved,marker))
            w._residual_cache_active = True
            w.weight_delta = build(w, orig)
        yield stats
    finally:
        active = False
        for wid in tuple(entries):
            discard(wid)
        for w,saved,marker in reversed(originals):
            if saved is _MISSING:
                del w.weight_delta
            else:
                w.weight_delta = saved
            if marker is _MISSING:
                del w._residual_cache_active
            else:
                w._residual_cache_active = marker
