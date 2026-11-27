"""Equal-weight additive minibatch screen; no live optimizer or model mutation."""
from __future__ import annotations

import math

import torch

from .adamw_screen import screen_adamw


@torch.no_grad()
def mean_gradient(rows):
    """Mean raw (unclipped) per-edge gradients; None only if all are missing."""
    if not rows or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError('invalid gradient roster')
    result = []
    for col in zip(*rows, strict=True):
        present = [g for g in col if g is not None]
        if not present:
            result.append(None)
            continue
        first = present[0]
        if any(g.shape != first.shape or g.dtype != first.dtype or g.device != first.device
               for g in present):
            raise ValueError('incompatible gradients')
        result.append(torch.stack(present).sum(0) / len(rows))
    return tuple(result)


@torch.no_grad()
def search_batch(optimizer, gradients, calibration, initial, *, pinned=(),
                 allowed=lambda batch: True, clip=1.0, rounds=1):
    """Deterministic best-improving single swaps, with fixed exploration pins.

    Caller owns checkpoint/optimizer provenance, equal edge weights, exposure
    legality, and an independent native whole-batch acceptance transaction.
    The returned score is predicted gain, never realized gain.
    """
    n = len(gradients)
    initial = tuple(sorted(initial))
    pins = frozenset(pinned)
    if (not initial or len(set(initial)) != len(initial) or
            any(type(i) is not int or not 0 <= i < n for i in initial) or
            not pins <= set(initial) or type(rounds) is not int or rounds < 0 or
            not allowed(initial)):
        raise ValueError('invalid initial batch')
    cache = {}

    def score(batch):
        if batch not in cache:
            g = mean_gradient([gradients[i] for i in batch])
            val = screen_adamw(optimizer, g, calibration, clip)['predicted_gain']
            if not math.isfinite(val):
                raise ValueError('nonfinite score')
            cache[batch] = val
        return cache[batch]

    current = initial
    initial_score = score(current)
    for _ in range(rounds):
        options = [current]
        for i in current:
            if i in pins:
                continue
            for j in range(n):
                if j not in current:
                    candidate = tuple(sorted((set(current) - {i}) | {j}))
                    if allowed(candidate):
                        options.append(candidate)
        best = min(options, key=lambda b: (-score(b), b))
        if score(best) <= score(current):
            break
        current = best
    return dict(batch=current, predicted_gain=score(current),
                initial_gain=initial_score, scored_batches=len(cache))
