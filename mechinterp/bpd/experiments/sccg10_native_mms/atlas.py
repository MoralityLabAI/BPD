"""Workstream C: provenance-aware cross-checkpoint mechanism atlas (MMS).

Separate namespace from the strict exact cache (trm_curation.cache). The atlas
stores only detached mechanism objects: support, physical metric (write Gram),
signed target/protected maps restricted to the support, the selected physical
direction, the reduced nullspace of the protected map on the support, singular
values, source parameter snapshot for drift screens, and conditional bounds.
It never stores or returns gradients, activations, optimizer state or model
state as current. Every lookup is a *proposal*; use requires the admission
ladder (ladder.py), which ends in native finite verification or refresh.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import sys
import time
from typing import Any

import numpy as np

FORBIDDEN_PAYLOAD = ('grad', 'gradient', 'activation', 'optimizer', 'exp_avg', 'state_dict', 'logits')


@dataclass(frozen=True)
class AtlasKey:
    module: str            # e.g. ComponentTRM.joint_state
    site: str              # e.g. post_write_prefix4
    phase: str             # e.g. suffix_step0_pre_low
    input_fingerprint: str
    objective: str         # e.g. sccg01_budgeted:dose=0.75:eta=0.25:delays=(15,)
    dtype: str
    dose: float


@dataclass
class AtlasEntry:
    key: AtlasKey
    source_checkpoint: str
    source_factor_frame: str
    metric_id: str
    support: tuple
    payload: dict           # numpy arrays only
    source_gain: float      # native finite target gain at source
    source_protected: float
    created_window: int
    validations: list = field(default_factory=list)  # (checkpoint_sha, window, outcome)
    uses: int = 0

    def nbytes(self) -> tuple[int, int]:
        p = sum(int(v.nbytes) for v in self.payload.values() if isinstance(v, np.ndarray))
        m = sys.getsizeof(self) + sys.getsizeof(self.payload) + sum(sys.getsizeof(k) for k in self.payload)
        m += sum(sys.getsizeof(getattr(self.key, f)) for f in self.key.__dataclass_fields__)
        m += sys.getsizeof(self.support) + 64 * (len(self.validations) + 4)
        return p, m


def metric_id(G: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(G, dtype=np.float64).tobytes()).hexdigest()[:16]


class MMSAtlas:
    """LRU atlas bounded by ``mib`` (payload + metadata). ``mib=0`` disables it."""

    def __init__(self, mib: float):
        self.mib = float(mib)
        self.max_bytes = int(round(self.mib * 2**20))
        self._store: OrderedDict[AtlasKey, AtlasEntry] = OrderedDict()
        self.payload_bytes = self.meta_bytes = self.peak_bytes = 0
        self.stats_ = {'lookups': 0, 'hits': 0, 'misses': 0, 'inserts': 0, 'evictions': 0,
                       'invalidations': 0, 'oversize': 0}
        self.ages: list[int] = []

    def _account(self, e: AtlasEntry, sign: int) -> None:
        p, m = e.nbytes()
        self.payload_bytes += sign * p
        self.meta_bytes += sign * m
        self.peak_bytes = max(self.peak_bytes, self.payload_bytes + self.meta_bytes)

    def lookup(self, key: AtlasKey, window: int) -> AtlasEntry | None:
        self.stats_['lookups'] += 1
        if self.max_bytes <= 0:
            self.stats_['misses'] += 1
            return None
        e = self._store.get(key)
        if e is None:
            self.stats_['misses'] += 1
            return None
        self._store.move_to_end(key)
        self.stats_['hits'] += 1
        self.ages.append(window - e.created_window)
        return e

    def insert(self, entry: AtlasEntry) -> bool:
        if self.max_bytes <= 0:
            return False
        for k, v in entry.payload.items():
            if any(f in k.lower() for f in FORBIDDEN_PAYLOAD):
                raise ValueError(f'atlas refuses non-mechanism payload {k}')
            if not isinstance(v, np.ndarray):
                raise ValueError('atlas payload must be detached numpy arrays')
        p, m = entry.nbytes()
        if p + m > self.max_bytes:
            self.stats_['oversize'] += 1
            return False
        if entry.key in self._store:
            self._account(self._store.pop(entry.key), -1)
        while self._store and self.payload_bytes + self.meta_bytes + p + m > self.max_bytes:
            _, old = self._store.popitem(last=False)
            self._account(old, -1)
            self.stats_['evictions'] += 1
        self._store[entry.key] = entry
        self._account(entry, +1)
        self.stats_['inserts'] += 1
        return True

    def invalidate(self, key: AtlasKey) -> None:
        e = self._store.pop(key, None)
        if e is not None:
            self._account(e, -1)
            self.stats_['invalidations'] += 1

    def stats(self) -> dict[str, Any]:
        return {**self.stats_, 'mib': self.mib, 'entries': len(self._store),
                'payload_bytes': self.payload_bytes, 'meta_bytes': self.meta_bytes,
                'total_bytes': self.payload_bytes + self.meta_bytes, 'peak_bytes': self.peak_bytes,
                'mean_age_at_hit': float(np.mean(self.ages)) if self.ages else None,
                'max_age_at_hit': int(max(self.ages)) if self.ages else None}
