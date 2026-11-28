"""Toy cross-loop reader math for designing native, dose-matched BPD controls.

All computations use visible synthetic vectors and a frozen linear reader.
No conclusion about Inkling, component transfer, or any trained model.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def _finite_vector(values: Sequence[float], *, width: int | None = None) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if not result or (width is not None and len(result) != width):
        raise ValueError("expected nonempty vectors of identical width")
    if any(not math.isfinite(x) for x in result):
        raise ValueError("all coordinates must be finite")
    return result


def rmsnorm(values: Sequence[float], *, epsilon: float = 1e-6) -> tuple[float, ...]:
    """Normalize vector to RMS one. epsilon acts on mean square."""
    x = _finite_vector(values)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive finite")
    scale = math.sqrt(sum(v * v for v in x) / len(x) + epsilon)
    if not math.isfinite(scale):
        raise ValueError("RMS overflow")
    return tuple(v / scale for v in x)


def reader_gap(
    reader: Sequence[float], block2_raw: Sequence[float], block3_raw: Sequence[float],
) -> dict[str, object]:
    """Cross-loop block-3 score gap and score contributions, before intervention."""
    u = _finite_vector(reader)
    v2 = rmsnorm(_finite_vector(block2_raw, width=len(u)))
    v3 = rmsnorm(_finite_vector(block3_raw, width=len(u)))
    droot = math.sqrt(len(u))
    contributions = tuple(u_i * (b - a) / droot for u_i, a, b in zip(u, v2, v3, strict=True))
    gap = sum(contributions)
    beta3 = 1.0 / (1.0 + math.exp(-gap)) if gap >= 0 else math.exp(gap) / (1.0 + math.exp(gap))
    return {"score_gap": gap, "block3_weight": beta3, "coordinate_contributions": contributions}


def coordinate_patch(
    reader: Sequence[float], block2_raw: Sequence[float], block3_raw: Sequence[float],
    *, coordinate: int, delta: float,
) -> dict[str, object]:
    """Patch RAW block-3 state, then recompute normalization and cross-loop reader."""
    v3 = list(_finite_vector(block3_raw))
    if type(coordinate) is not int or not 0 <= coordinate < len(v3):
        raise ValueError("invalid coordinate")
    if not math.isfinite(delta):
        raise ValueError("finite delta required")
    before = reader_gap(reader, block2_raw, v3)
    v3[coordinate] += delta
    after = reader_gap(reader, block2_raw, v3)
    return {
        "before": before,
        "after": after,
        "actual_raw_displacement_l2": abs(delta),
        "change_in_block3_weight": after["block3_weight"] - before["block3_weight"],
    }


def norm_matched_random_patch(
    reader: Sequence[float], block2_raw: Sequence[float], block3_raw: Sequence[float],
    *, dose: float, seed: int,
) -> dict[str, object]:
    """Random-direction sham, L2 matched to coordinate-patch dose."""
    v3 = list(_finite_vector(block3_raw))
    if not math.isfinite(dose) or dose < 0 or type(seed) is not int:
        raise ValueError("invalid dose or seed")
    rng = random.Random(seed)
    direction = [rng.gauss(0.0, 1.0) for _ in v3]
    norm = math.sqrt(sum(v * v for v in direction))
    if norm == 0:
        raise ValueError("unexpected zero random direction")
    before = reader_gap(reader, block2_raw, v3)
    v3 = [x + dose * d / norm for x, d in zip(v3, direction, strict=True)]
    after = reader_gap(reader, block2_raw, v3)
    return {
        "before": before, "after": after,
        "actual_raw_displacement_l2": dose,
        "change_in_block3_weight": after["block3_weight"] - before["block3_weight"],
        "seed": seed,
    }
