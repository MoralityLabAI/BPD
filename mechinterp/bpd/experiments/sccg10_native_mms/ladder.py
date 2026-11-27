"""Workstream C: SCCG admission ladder for reusing an atlas entry at a new checkpoint.

Levels (cheapest first):
 L1 screen       provenance identity, age, relative parameter displacement.
 L2 certificate  SCCG-10 conditional admission. Its premises are uniform
                 derivative-error and finite-remainder bounds over the dose ball
                 for the full-suffix response. They are not available for the
                 native TRM, so in ``exact`` mode L2 records NOT_APPLICABLE and
                 never admits. In ``heuristic`` mode it evaluates SCCG-10 with a
                 calibrated drift envelope and labels the result HEURISTIC; a
                 heuristic pass still goes to L4.
 L3 exact audit  live observation restricted to the cached support and an exact
                 re-solve there (skips the combinatorial search). Invoked only for
                 borderline L4 outcomes when the audit is cheaper than a refresh.
 L4 verification native finite continuation of the cached direction at the
                 current checkpoint on the entry's own panel (read from the
                 exact-keyed readout of the calibration forward).
 L5 refresh      full observation + support search, or ordinary fallback.

A failed certificate or failed verification means reuse is not justified. It
never labels the edit infeasible: the fallback is refresh or ordinary training.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Fr
from typing import Any

import numpy as np

from .admission import assess


@dataclass(frozen=True)
class LadderConfig:
    mode: str = 'exact'            # exact | heuristic
    max_age_windows: int = 64
    max_rel_displacement: float = 0.05
    gain_ratio: float = 0.7        # L4: live gain >= ratio * source gain
    borderline_ratio: float = 0.4  # L3 band: [borderline, gain_ratio)
    protected_slack: float = 0.25  # L4: protected <= eta * (1 + slack)
    kappa_a: float = 0.0           # heuristic envelope alpha <= kappa_a * ||dtheta||
    kappa_c: float = 0.0           # heuristic envelope gamma <= kappa_c * ||dtheta||
    remainder: float = 0.0         # heuristic finite remainder
    audit: bool = True


def screen(entry, key, theta_now: np.ndarray, window: int, cfg: LadderConfig) -> tuple[bool, str, float]:
    if entry is None:
        return False, 'L1_MISS', 0.0
    if entry.key != key:
        return False, 'L1_PROVENANCE_MISMATCH', 0.0
    if window - entry.created_window > cfg.max_age_windows:
        return False, 'L1_AGE', 0.0
    src = entry.payload['theta_source']
    rel = float(np.linalg.norm(theta_now - src) / max(np.linalg.norm(src), 1e-30))
    if rel > cfg.max_rel_displacement:
        return False, 'L1_DISPLACEMENT', rel
    return True, 'L1_PASS', rel


def certificate(entry, theta_now: np.ndarray, eta: float, cfg: LadderConfig) -> dict[str, Any]:
    if cfg.mode == 'exact':
        return {'verdict': 'L2_NOT_APPLICABLE_NO_UNIFORM_BOUND', 'scope': 'NONE'}
    disp = float(np.linalg.norm(theta_now - entry.payload['theta_source']))
    sup = list(entry.support)
    if len(sup) > 8:
        return {'verdict': 'L2_SKIPPED_SUPPORT_TOO_LARGE_FOR_EXACT_ARITHMETIC', 'scope': 'NONE'}
    q = lambda x: Fr(float(x)).limit_denominator(10**8)
    a = [q(v) for v in entry.payload['a_support']]
    C = [[q(v) for v in row] for row in entry.payload['C_support']]
    u = [q(v) for v in entry.payload['delta'][sup]]
    dose = q(entry.key.dose)
    n2 = sum(x * x for x in u)
    if n2 > dose * dose:
        u = [x * dose * dose / n2 for x in u]
    I = [[Fr(int(i == j)) for j in range(len(sup))] for i in range(len(sup))]
    r = assess(I, a, C, u, dose=dose, collateral=q(eta), alpha=q(cfg.kappa_a * disp),
               gamma=q(cfg.kappa_c * disp), target_remainder=q(cfg.remainder),
               protected_remainder=q(cfg.remainder), min_gain=Fr(1, 1000))
    return {'verdict': r['verdict'], 'scope': 'HEURISTIC_ENVELOPE_NOT_A_CERTIFICATE',
            'target_lower': float(Fr(r['target_lower'])), 'target_upper': float(Fr(r['target_upper']))}


def verify(entry, readout: dict, eta: float, cfg: LadderConfig) -> tuple[str, float]:
    """L4 from a native readout {target_gain, protected_norm} of the cached direction."""
    # Reuse is judged against what native verification established at the source
    # checkpoint: the finite protected change of SCCG-01 directions can exceed the
    # linear budget eta already at source (finite-dose nonlinearity), so the bound is
    # max(eta, source protected change) * (1 + slack).
    g, p = readout['target_gain'], readout['protected_norm']
    src = max(entry.source_gain, 1e-9)
    ratio = g / src
    p_ok = p <= max(eta, entry.source_protected) * (1 + cfg.protected_slack)
    if g > 0 and ratio >= cfg.gain_ratio and p_ok:
        return 'L4_VERIFIED', ratio
    if g > 0 and ratio >= cfg.borderline_ratio and p_ok:
        return 'L4_BORDERLINE', ratio
    return 'L4_FAILED', ratio
