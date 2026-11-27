"""Bounded Mechanistic Memory Store (--mms-mib) with strict provenance binding.

Caches detached observer maps, immutable feature templates, reusable post-write
suffix states, and valid calibration gradients. Every entry is bound to:
  (checkpoint_sha256, module_id, visit_spec, input_fingerprint,
   objective_id, factor_frame_sha256).

Strictly rejects any tensor carrying an autograd graph (requires_grad or grad_fn)
and rejects stale entries whenever the active checkpoint or factor frame advances.
Charges payload bytes, key bytes, and Python container metadata overhead.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import sys
import time
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CacheKey:
    namespace: str
    checkpoint_sha256: str
    module_id: str
    visit_spec: str
    input_fingerprint: str
    objective_id: str
    factor_frame_sha256: str


def _assert_no_autograd_and_measure(obj: Any) -> tuple[int, int]:
    """Verify no autograd graph is retained and return (payload_bytes, meta_bytes)."""
    payload = 0
    meta = sys.getsizeof(obj)
    if isinstance(obj, torch.Tensor):
        if obj.requires_grad or obj.grad_fn is not None:
            raise ValueError('MechanisticCache refuses tensors with autograd graphs')
        payload += obj.numel() * obj.element_size()
        return payload, meta
    if isinstance(obj, np.ndarray):
        payload += int(obj.nbytes)
        return payload, meta
    if isinstance(obj, dict):
        for k, v in obj.items():
            p1, m1 = _assert_no_autograd_and_measure(k)
            p2, m2 = _assert_no_autograd_and_measure(v)
            payload += p1 + p2
            meta += m1 + m2
        return payload, meta
    if isinstance(obj, (list, tuple)):
        for item in obj:
            p, m = _assert_no_autograd_and_measure(item)
            payload += p
            meta += m
        return payload, meta
    if hasattr(obj, '__dataclass_fields__'):
        for fname in obj.__dataclass_fields__:
            p, m = _assert_no_autograd_and_measure(getattr(obj, fname))
            payload += p
            meta += m
        return payload, meta
    return payload, meta


class MechanisticCache:
    """LRU cache bounded by ``mms_mib`` total bytes (payload + metadata + keys)."""

    def __init__(self, mms_mib: float = 0.0) -> None:
        if not np.isfinite(mms_mib) or mms_mib < 0:
            raise ValueError('mms_mib must be finite and nonnegative')
        self.mms_mib = float(mms_mib)
        self.max_bytes = int(round(self.mms_mib * 1024 * 1024))
        self._store: OrderedDict[CacheKey, tuple[Any, int, int, float]] = OrderedDict()
        self.payload_bytes = 0
        self.metadata_bytes = 0
        self.peak_total_bytes = 0
        self.hits = 0
        self.misses = 0
        self.stale_rejections = 0
        self.evictions = 0
        self.oversize_rejections = 0
        self.recomputations_avoided = 0
        self.saved_cpu_seconds_est = 0.0
        self.lookup_cpu_seconds = 0.0
        self.insert_cpu_seconds = 0.0
        self.active_checkpoint: str | None = None
        self.active_factor_frame: str | None = None

    def bind_checkpoint(self, checkpoint_sha256: str, factor_frame_sha256: str, *,
                        evict_stale_model_entries: bool = True) -> int:
        """Advance active checkpoint & factor frame; evict stale model-bound entries."""
        self.active_checkpoint = checkpoint_sha256
        self.active_factor_frame = factor_frame_sha256
        evicted = 0
        if evict_stale_model_entries:
            # Keep checkpoint-independent immutable feature templates ('template')
            stale_keys = [
                k for k in list(self._store.keys())
                if k.namespace != 'template' and (
                    k.checkpoint_sha256 != checkpoint_sha256
                    or k.factor_frame_sha256 != factor_frame_sha256
                )
            ]
            for k in stale_keys:
                _, p, m, _ = self._store.pop(k)
                self.payload_bytes -= p
                self.metadata_bytes -= m
                self.evictions += 1
                evicted += 1
        return evicted

    def _is_stale(self, key: CacheKey) -> bool:
        if key.namespace == 'template':
            return False
        if self.active_checkpoint is not None and key.checkpoint_sha256 != self.active_checkpoint:
            return True
        if self.active_factor_frame is not None and key.factor_frame_sha256 != self.active_factor_frame:
            return True
        return False

    def get(self, key: CacheKey) -> Any | None:
        t0 = time.process_time()
        try:
            if self.max_bytes <= 0:
                self.misses += 1
                return None
            if self._is_stale(key):
                self.stale_rejections += 1
                return None
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            val, _, _, cost_est = entry
            self._store.move_to_end(key)
            self.hits += 1
            self.recomputations_avoided += 1
            self.saved_cpu_seconds_est += cost_est
            return val
        finally:
            self.lookup_cpu_seconds += time.process_time() - t0

    def put(self, key: CacheKey, value: Any, *, compute_cpu_seconds: float = 0.0) -> bool:
        t0 = time.process_time()
        try:
            if self.max_bytes <= 0:
                return False
            if self._is_stale(key):
                self.stale_rejections += 1
                return False
            p_bytes, m_bytes = _assert_no_autograd_and_measure(value)
            k_bytes = sum(sys.getsizeof(getattr(key, f)) for f in key.__dataclass_fields__) + sys.getsizeof(key)
            m_bytes += k_bytes
            total_entry = p_bytes + m_bytes
            if total_entry > self.max_bytes:
                self.oversize_rejections += 1
                return False
            if key in self._store:
                _, old_p, old_m, _ = self._store.pop(key)
                self.payload_bytes -= old_p
                self.metadata_bytes -= old_m
            while self._store and (self.payload_bytes + self.metadata_bytes + total_entry > self.max_bytes):
                _, (_, ev_p, ev_m, _) = self._store.popitem(last=False)
                self.payload_bytes -= ev_p
                self.metadata_bytes -= ev_m
                self.evictions += 1
            self._store[key] = (value, p_bytes, m_bytes, float(max(compute_cpu_seconds, 0.0)))
            self.payload_bytes += p_bytes
            self.metadata_bytes += m_bytes
            cur_total = self.payload_bytes + self.metadata_bytes
            if cur_total > self.peak_total_bytes:
                self.peak_total_bytes = cur_total
            return True
        finally:
            self.insert_cpu_seconds += time.process_time() - t0

    def stats(self) -> dict[str, Any]:
        total_req = self.hits + self.misses
        return {
            'mms_mib': self.mms_mib,
            'max_bytes': self.max_bytes,
            'entries': len(self._store),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': (self.hits / total_req) if total_req else 0.0,
            'stale_rejections': self.stale_rejections,
            'evictions': self.evictions,
            'oversize_rejections': self.oversize_rejections,
            'recomputations_avoided': self.recomputations_avoided,
            'payload_bytes': self.payload_bytes,
            'metadata_bytes': self.metadata_bytes,
            'total_bytes': self.payload_bytes + self.metadata_bytes,
            'peak_total_bytes': self.peak_total_bytes,
            'occupancy_fraction': (
                (self.payload_bytes + self.metadata_bytes) / self.max_bytes
                if self.max_bytes > 0 else 0.0
            ),
            'saved_cpu_seconds_est': self.saved_cpu_seconds_est,
            'lookup_cpu_seconds': self.lookup_cpu_seconds,
            'insert_cpu_seconds': self.insert_cpu_seconds,
            'overhead_cpu_seconds': self.lookup_cpu_seconds + self.insert_cpu_seconds,
        }
