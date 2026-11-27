"""Deterministic sparse-support search over physical intervention columns.

Single-column beam growth with the SCCG-01 budgeted solver. No global optimality
claim (SCCG-03 shows exact sparse control is combinatorial). Records the best
support found at every size so k=4 and k=8 come from one search.
"""
from __future__ import annotations

import time

import numpy as np

from ..trm_curation.reference import budgeted_selective_direction


def solve_on(a, C, B, dose, eta):
    return budgeted_selective_direction(a, C, B, dose, collateral_budget=eta)


def beam_support(a: np.ndarray, C: np.ndarray, B: np.ndarray, dose: float, eta: float,
                 k: int, beam: int) -> dict:
    t0 = time.process_time()
    ncol = B.shape[1]
    if not 1 <= k <= ncol or beam < 1:
        raise ValueError('invalid support search budget')
    solved: dict[tuple, dict] = {}
    layer = [()]
    best_by_size = {}
    for size in range(1, k + 1):
        cands = sorted({tuple(sorted(s + (j,))) for s in layer for j in range(ncol) if j not in s})
        for s in cands:
            if s not in solved:
                solved[s] = solve_on(a, C, B[:, list(s)], dose, eta)
        ranked = sorted(cands, key=lambda s: (-solved[s]['gain'], s))
        layer = ranked[:beam]
        best_by_size[size] = (layer[0], solved[layer[0]])
    return {'best_by_size': best_by_size, 'solves': len(solved),
            'cpu_seconds': time.process_time() - t0}


def resolve_fixed(a, C, B, support, dose, eta) -> dict:
    """Re-solve the direction on a stored support at new derivatives (cheap)."""
    return solve_on(a, C, B[:, list(support)], dose, eta)
