"""Shared fixed-atom control for matched-rank comparisons."""
from itertools import combinations

import numpy as np


def best_atom_subset(actions: np.ndarray, target: np.ndarray, rank: int) -> tuple[int, ...]:
    """Pick the least squared residual on training actions, then lexical ties.

    actions[i, j] is the projected physical action of atom j on training
    sample i. A bank of rank-one atoms gives a summed operator of rank at most
    the chosen count. The caller separately checks the achieved rank.
    """
    if actions.ndim != 3 or target.shape != (actions.shape[0], actions.shape[2]):
        raise ValueError("atom actions and target must share samples and write width")
    if not 0 <= rank <= actions.shape[1]:
        raise ValueError("invalid atom count")
    candidates = combinations(range(actions.shape[1]), rank)
    return min(candidates, key=lambda keep: (float(np.sum((target - actions[:, keep].sum(axis=1)) ** 2)), keep))
