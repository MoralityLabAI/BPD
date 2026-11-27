"""Provenance-aware cross-checkpoint mechanism atlas (MMS-TRM Workstream C).

The atlas is a separate namespace beside the strict exact cache
(``trm_curation.cache.MechanisticCache``), whose full keys are never relaxed.
Atlas entries are detached mechanism descriptions recorded at a *source*
checkpoint: physical write Gram matrices, signed observer maps, candidate
cancellation supports, reduced nullspaces, factor-frame provenance and any
conditional bounds. Reading an entry at another checkpoint never yields an
exact value; it yields a proposal whose scope is fixed by the decision
hierarchy in :func:`decide`:

1. cheap screen (age, parameter distance, factor-frame residual, trace residual);
2. compensated finite-protection certificate, only when uniform premises are
   supplied with provenance ``uniform`` (SCCG-10 reference, exact rationals);
3. exact multiplier audit, only when its estimated value exceeds its cost;
4. explicit native finite verification of the concrete candidate;
5. refresh or ordinary-training fallback.

A failed certificate or screen means reuse is not justified. It is never
reported as infeasibility of the underlying control problem.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, asdict
import hashlib
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

from mechinterp.bpd.experiments.trm_curation.cache import _assert_no_autograd_and_measure

# Unit roundoff (round to nearest) and half the smallest subnormal spacing.
STORAGE = {
    'float64': (np.float64, 2.0 ** -53, 2.0 ** -1075),
    'float32': (np.float32, 2.0 ** -24, 2.0 ** -150),
    'float16': (np.float16, 2.0 ** -11, 2.0 ** -25),
}

SCOPES = ('CERTIFIED_CONDITIONAL_IN_DECLARED_SPAN', 'VERIFIED_FINITE_ON_DECLARED_QUERIES',
          'HEURISTIC_PROPOSAL')


def array_digest(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(str((x.dtype.str, x.shape)).encode())
    h.update(x.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class AtlasKey:
    """Every field participates in identity; nothing is wildcarded."""
    source_checkpoint: str
    module: str
    site: str
    phase: str
    input_fingerprint: str
    support: tuple
    metric_id: str
    factor_frame: str
    objective: str
    dtype: str
    dose: str


@dataclass
class AtlasEntry:
    key: AtlasKey
    payload: dict[str, Any]
    storage_dtype: str
    storage_error_bounds: dict[str, float]
    build_cpu_seconds: float
    created_update: int
    validations: list[dict] = field(default_factory=list)
    trace_residuals: list[float] = field(default_factory=list)


def quantize(payload: dict[str, Any], dtype: str) -> tuple[dict[str, Any], dict[str, float]]:
    """Store arrays at ``dtype`` and return rigorous Frobenius rounding bounds.

    For round-to-nearest without overflow, |fl(x)-x| <= u|x| for normal
    results and <= eta for subnormal ones, so
    ||fl(X)-X||_2 <= ||fl(X)-X||_F <= u ||X||_F + eta sqrt(numel).
    Overflow is refused.
    """
    np_t, u, eta = STORAGE[dtype]
    out, bounds = {}, {}
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            if v.requires_grad or v.grad_fn is not None:
                raise ValueError('atlas refuses tensors with autograd graphs')
            v = v.detach().double().cpu().numpy()
        if isinstance(v, np.ndarray) and v.dtype.kind == 'f':
            if not np.isfinite(v).all():
                raise ValueError('nonfinite atlas payload')
            q = v.astype(np_t)
            if not np.isfinite(q).all():
                raise ValueError('storage overflow')
            out[k] = q
            bounds[k] = float(u * np.linalg.norm(v) + eta * np.sqrt(v.size))
        else:
            out[k] = v
    return out, bounds


class MechanismAtlas:
    """Byte-bounded LRU atlas; payload, keys and container metadata are charged."""

    def __init__(self, mib: float, *, storage_dtype: str = 'float64') -> None:
        if not np.isfinite(mib) or mib < 0 or storage_dtype not in STORAGE:
            raise ValueError('invalid atlas configuration')
        self.max_bytes = int(round(mib * 1024 * 1024))
        self.storage_dtype = storage_dtype
        self._store: OrderedDict[AtlasKey, tuple[AtlasEntry, int]] = OrderedDict()
        self.bytes = 0
        self.peak_bytes = 0
        self.counters = dict(inserts=0, lookups=0, hits=0, misses=0, evictions=0, oversize=0,
                             invalidations=0, validations=0)
        self.cpu = dict(insert=0.0, lookup=0.0, invalidate=0.0, evict=0.0)

    def _size(self, entry: AtlasEntry) -> int:
        p, m = _assert_no_autograd_and_measure(entry.payload)
        k = sum(sys.getsizeof(getattr(entry.key, f)) for f in entry.key.__dataclass_fields__)
        return p + m + k + sys.getsizeof(entry) + sys.getsizeof(entry.validations) + 64 * len(entry.validations)

    def insert(self, key: AtlasKey, payload: dict[str, Any], *, build_cpu_seconds: float,
               created_update: int) -> AtlasEntry | None:
        t0 = time.process_time()
        try:
            if self.max_bytes <= 0:
                return None
            data, bounds = quantize(payload, self.storage_dtype)
            entry = AtlasEntry(key, data, self.storage_dtype, bounds, float(build_cpu_seconds), created_update)
            size = self._size(entry)
            if size > self.max_bytes:
                self.counters['oversize'] += 1
                return None
            if key in self._store:
                self.bytes -= self._store.pop(key)[1]
            while self._store and self.bytes + size > self.max_bytes:
                t1 = time.process_time()
                _, (_, s) = self._store.popitem(last=False)
                self.bytes -= s
                self.counters['evictions'] += 1
                self.cpu['evict'] += time.process_time() - t1
            self._store[key] = (entry, size)
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.counters['inserts'] += 1
            return entry
        finally:
            self.cpu['insert'] += time.process_time() - t0

    def find(self, match: Callable[[AtlasKey], bool]) -> AtlasEntry | None:
        """Most recent entry whose key satisfies ``match`` (caller states every field it requires)."""
        t0 = time.process_time()
        try:
            self.counters['lookups'] += 1
            for key in reversed(self._store):
                if match(key):
                    self._store.move_to_end(key)
                    self.counters['hits'] += 1
                    return self._store[key][0]
            self.counters['misses'] += 1
            return None
        finally:
            self.cpu['lookup'] += time.process_time() - t0

    def invalidate(self, match: Callable[[AtlasKey], bool]) -> int:
        t0 = time.process_time()
        gone = [k for k in self._store if match(k)]
        for k in gone:
            self.bytes -= self._store.pop(k)[1]
        self.counters['invalidations'] += len(gone)
        self.cpu['invalidate'] += time.process_time() - t0
        return len(gone)

    def record_validation(self, entry: AtlasEntry, record: dict) -> None:
        if 'destination_checkpoint' not in record:
            raise ValueError('validation must name its destination checkpoint')
        entry.validations.append(dict(record))
        self.counters['validations'] += 1

    def stats(self) -> dict[str, Any]:
        return dict(max_bytes=self.max_bytes, bytes=self.bytes, peak_bytes=self.peak_bytes, entries=len(self._store),
                    storage_dtype=self.storage_dtype, **self.counters,
                    **{f'cpu_{k}': v for k, v in self.cpu.items()},
                    cpu_total=sum(self.cpu.values()))


# ------------------------------------------------------------------ decisions

@dataclass(frozen=True)
class Premises:
    """Inputs to the conditional certificate, with explicit provenance.

    provenance must be 'uniform' (an established bound over the whole dose
    ball and the declared response) for tier 2 to run. 'measured' or
    'sampled' inputs are recorded but never certify.
    """
    alpha: float
    gamma: float
    target_remainder: float
    protected_remainder: float
    provenance: str


@dataclass
class Decision:
    action: str            # 'reuse' | 'reuse_pending_verification' | 'refresh' | 'ordinary'
    tier: int              # last tier executed
    scope: str
    reason: str
    cost_cpu: dict[str, float]
    detail: dict[str, Any] = field(default_factory=dict)

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class Policy:
    max_age: int = 8
    max_param_distance: float = float('inf')
    max_frame_residual: float = 0.25
    max_trace_residual: float = 1.0
    gain_floor: float = 0.1
    collateral: float = 0.25
    audit_value_per_cpu: float = 0.0   # tier 3 runs only if estimated value > cost * this price


def decide(entry: AtlasEntry | None, *, age: int, param_distance: float, frame_residual: float = 0.0,
           premises: Premises | None = None, certificate: Callable[[Premises], dict] | None = None,
           audit: Callable[[], dict] | None = None, audit_value: float = 0.0, audit_cost: float = 0.0,
           verify: Callable[[], dict] | None = None, refresh_value: float = 0.0, refresh_cost: float = 0.0,
           policy: Policy = Policy()) -> Decision:
    cost: dict[str, float] = {}
    if entry is None:
        return _fallback(4, 'no atlas entry', cost, refresh_value, refresh_cost)
    # Tier 1: cheap screens.
    t0 = time.process_time()
    last_trace = entry.trace_residuals[-1] if entry.trace_residuals else 0.0
    screen = [(age > policy.max_age, f'age {age} > {policy.max_age}'),
              (param_distance > policy.max_param_distance, 'parameter distance'),
              (frame_residual > policy.max_frame_residual, f'factor-frame residual {frame_residual:.3g}'),
              (last_trace > policy.max_trace_residual, f'trace residual {last_trace:.3g}')]
    cost['screen'] = time.process_time() - t0
    for bad, why in screen:
        if bad:
            return _fallback(1, f'screen: {why}; reuse not justified', cost, refresh_value, refresh_cost)
    # Tier 2: conditional certificate, only with uniform premises.
    if premises is not None and certificate is not None and premises.provenance == 'uniform':
        t0 = time.process_time()
        cert = certificate(premises)
        cost['certificate'] = time.process_time() - t0
        if cert['verdict'] == 'CONDITIONAL_REUSE_CANDIDATE':
            return Decision('reuse', 2, SCOPES[0], 'conditional certificate passed', cost, dict(certificate=cert))
        # Tier 3: exact multiplier audit when worthwhile.
        if audit is not None and audit_value > audit_cost * policy.audit_value_per_cpu:
            t0 = time.process_time()
            aud = audit()
            cost['audit'] = time.process_time() - t0
            if aud.get('verdict') == 'CONDITIONAL_REUSE_CANDIDATE':
                return Decision('reuse', 3, SCOPES[0], 'multiplier audit passed', cost, dict(audit=aud))
    # Tier 4: explicit native finite verification of the concrete candidate.
    if verify is not None:
        t0 = time.process_time()
        v = verify()
        cost['verify'] = time.process_time() - t0
        ok = v['target_gain'] >= policy.gain_floor and v['protected_l2'] <= policy.collateral
        if ok:
            return Decision('reuse', 4, SCOPES[1], 'native finite verification passed', cost, dict(verify=v))
        return _fallback(4, 'native verification failed for this candidate; reuse not justified',
                         cost, refresh_value, refresh_cost, dict(verify=v))
    # No verifier supplied: the caller owns verification (e.g. the trainer's committed-update
    # acceptance check). The proposal is not usable until that verification runs.
    return Decision('reuse_pending_verification', 1, SCOPES[2], 'screen passed; heuristic proposal', cost)


def _fallback(tier, reason, cost, refresh_value, refresh_cost, detail=None):
    action = 'refresh' if refresh_value > refresh_cost else 'ordinary'
    return Decision(action, tier, SCOPES[2], reason, cost, detail or {})
