"""Workstream D: resource ledger and the reinvestment controller.

Every consequential expense is charged by wrapping it in ``ledger.charge(cat)``,
which records actual process CPU (time.process_time) and wall time. Deterministic
work counters (block visits, backward passes, solver calls) are recorded beside
real CPU so timer noise on a shared host can be separated from real work.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any

CATEGORIES = (
    'train_step',            # committed or discarded optimizer step fwd/bwd
    'candidate_preview',     # per-candidate gradient + analytic AdamW preview
    'cal_grad',              # calibration objective forward/backward
    'acceptance',            # post-commit calibration forward
    'mechanism_observe',     # VJPs / Jacobians for the mechanism
    'support_search',        # sparse support beam search
    'verification',          # native finite verification forwards
    'certificate',           # SCCG-10 / exact audit arithmetic
    'atlas_ops',             # atlas insert / lookup / invalidate / evict
    'cache_ops',             # exact cache keys, hashing, lookup, insert
    'transaction',           # optimizer/model clone and rollback
    'controller',            # value measurements for the policy
    'data',                  # batch construction
    'selection',             # scheduling bookkeeping
)


@dataclass
class ResourceLedger:
    cpu: dict = field(default_factory=lambda: defaultdict(float))
    wall: dict = field(default_factory=lambda: defaultdict(float))
    counts: dict = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def charge(self, cat: str):
        if cat not in CATEGORIES:
            raise ValueError(f'unknown cost category {cat}')
        c0, w0 = time.process_time(), time.perf_counter()
        try:
            yield
        finally:
            self.cpu[cat] += time.process_time() - c0
            self.wall[cat] += time.perf_counter() - w0

    def count(self, key: str, n: int = 1) -> None:
        self.counts[key] += n

    @property
    def total_cpu(self) -> float:
        return float(sum(self.cpu.values()))

    def as_dict(self) -> dict[str, Any]:
        return {'cpu': dict(self.cpu), 'wall': dict(self.wall), 'counts': dict(self.counts),
                'total_cpu': self.total_cpu}


@dataclass
class RateController:
    """Block-level choice between an ordinary step and an audited/curated step.

    Value = decrease of ordinary TRAIN-feedback cross-entropy per charged CPU
    second, measured on the arm's own blocks (multi-window estimate, EMA).
    Deterministic exploration keeps both estimates current. ``mode='always'``
    disables the controller (one-step greedy curated policy).
    """
    block: int = 8
    explore_every: int = 5
    ema: float = 0.5
    mode: str = 'rate'
    rates: dict = field(default_factory=lambda: {'ordinary': None, 'curated': None})
    history: list = field(default_factory=list)
    n_blocks: int = 0

    def choose(self) -> str:
        if self.mode == 'always':
            return 'curated'
        if self.mode == 'never':
            return 'ordinary'
        k = self.n_blocks
        if self.rates['ordinary'] is None:
            return 'ordinary'
        if self.rates['curated'] is None:
            return 'curated'
        best = 'curated' if self.rates['curated'] > self.rates['ordinary'] else 'ordinary'
        if k % self.explore_every == self.explore_every - 1:
            return 'ordinary' if best == 'curated' else 'curated'
        return best

    def update(self, action: str, ce_drop: float, cpu: float) -> None:
        rate = ce_drop / max(cpu, 1e-6)
        old = self.rates[action]
        self.rates[action] = rate if old is None else self.ema * rate + (1 - self.ema) * old
        self.history.append({'block': self.n_blocks, 'action': action, 'ce_drop': ce_drop,
                             'cpu': cpu, 'rate': rate, 'ema_ordinary': self.rates['ordinary'],
                             'ema_curated': self.rates['curated']})
        self.n_blocks += 1


def voi_refresh_upper(scores_stale, scores_fresh_err: float, score_drift_sup: float) -> float:
    """Upper bound on the one-window value of refreshing scores (MATH_REVIEW Prop. 6).

    If every stale score is within ``score_drift_sup`` of the fresh score and the
    fresh score is within ``scores_fresh_err`` of the realized gain, picking by stale
    scores loses at most 2*(score_drift_sup + scores_fresh_err) relative to the best
    candidate, and refreshing can recover at most 2*score_drift_sup + 2*fresh_err.
    """
    return 2.0 * (score_drift_sup + scores_fresh_err)
