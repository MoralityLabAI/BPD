"""INK-BPD-04: RMS-normalized reader geometry for Goodfire-style writers.

These are exact algebraic derivatives of a frozen two-block reader, not
measurements of an Inkling checkpoint or proof of a downstream mechanism.
For Inkling's idealized score s_i = u·RMSNorm(v_i)/sqrt(d), the factors
sqrt(d) cancel and s_i = u·v_i/||v_i||. The pre-normalization gradient
therefore lies in the tangent space of the block-output sphere.

Keep weight-space cosine, coordinate attribution, tangent derivative, and
full native intervention outcomes in distinct evidence classes.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

try:
    from .routing_math import route_gap
except ImportError:  # Standalone pure-stdlib unittest discovery skips bpd/__init__ (torch).
    from routing_math import route_gap


def _vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    if not values or any(
        isinstance(v, bool)
        or not isinstance(v, (float, int))
        or not math.isfinite(v)
        for v in values
    ):
        raise ValueError(f"{name} must be a nonempty finite real vector")
    return tuple(float(v) for v in values)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(_dot(values, values))


def reader_tangent(
    reader: Sequence[float],
    block2: Sequence[float],
    block3: Sequence[float],
) -> dict:
    """Exact frozen-reader score attribution and raw-state Jacobian.

    The block-3 derivative includes the RMS-normalizer's dependence on
    the physical raw block vector. It is a *local* derivative before
    recalculating upstream states, other readers, or later recurrent loops.
    """
    u = _vector(reader, "reader")
    a = _vector(block2, "block2")
    b = _vector(block3, "block3")
    if len(u) != len(a) or len(u) != len(b):
        raise ValueError("reader and blocks must share a width")
    radius = _norm(b)
    if radius == 0:
        raise ValueError("zero block3 has undefined reader gradient")
    observed = route_gap(u, a, b)
    score3 = _dot(u, b) / radius
    gradient = tuple(
        (ui - score3 * vi / radius) / radius
        for ui, vi in zip(u, b)
    )
    beta = observed["beta3"]
    beta_gradient = tuple(beta * (1 - beta) * g for g in gradient)
    return {
        "schema": "bpd/ink-bpd04-frozen-reader-tangent/v1",
        "evidence_class": "exact_source_only_reader_geometry",
        "gap": observed["gap"],
        "beta3": beta,
        "score3": score3,
        "block3_l2": radius,
        "gap_coordinate_terms": observed["per_coordinate"],
        "score3_coordinate_terms": tuple(
            ui * vi / radius for ui, vi in zip(u, b)
        ),
        "rms_energy_shares": tuple((vi / radius) ** 2 for vi in b),
        "score3_gradient": gradient,
        "beta3_gradient": beta_gradient,
        "radial_score_derivative": _dot(gradient, b),
    }


def writer_scan(
    reader: Sequence[float],
    block2: Sequence[float],
    block3: Sequence[float],
    writers: Mapping[str, Sequence[float]],
) -> dict[str, dict]:
    """Compare Goodfire-like writer/reader cosine with tangent sensitivity.

    A high absolute raw cosine can coexist with zero local routing gain:
    a writer parallel to the block state changes magnitude without
    changing an RMS-normalized two-block reader score.
    """
    if not writers:
        raise ValueError("at least one writer is required")
    u = _vector(reader, "reader")
    geom = reader_tangent(u, block2, block3)
    u_norm = _norm(u)
    if u_norm == 0:
        raise ValueError("zero reader has undefined writer cosine")
    g = geom["score3_gradient"]
    g_norm = _norm(g)
    output = {}
    for name, direction in sorted(writers.items()):
        w = _vector(direction, f"writer {name}")
        if len(w) != len(u):
            raise ValueError("writer width mismatch")
        w_norm = _norm(w)
        if w_norm == 0:
            raise ValueError("zero writer direction")
        raw_cosine = _dot(u, w) / (u_norm * w_norm)
        tangent_projection = _dot(g, w) / w_norm
        output[name] = {
            "raw_reader_cosine": raw_cosine,
            "absolute_raw_reader_cosine": abs(raw_cosine),
            "tangent_reader_cosine": (
                tangent_projection / g_norm if g_norm > 0 else None
            ),
            "predicted_score_shift_per_l2": tangent_projection,
            "predicted_gate_shift_per_l2": (
                geom["beta3"] * (1 - geom["beta3"]) * tangent_projection
            ),
        }
    return output


def physical_patch_audit(
    reader: Sequence[float],
    block2: Sequence[float],
    block3: Sequence[float],
    patched3: Sequence[float],
) -> dict:
    """Compare a first-order prediction to a physically recomputed gate.

    First-order residuals measure local linearization error. Changes to
    downstream model accuracy require a separate native continuation.
    """
    b = _vector(block3, "block3")
    p = _vector(patched3, "patched3")
    if len(b) != len(p):
        raise ValueError("patch width mismatch")
    before = reader_tangent(reader, block2, b)
    after = reader_tangent(reader, block2, p)
    dv = tuple(y - x for x, y in zip(b, p))
    predicted_gap = _dot(before["score3_gradient"], dv)
    predicted_gate = _dot(before["beta3_gradient"], dv)
    actual_gap = after["gap"] - before["gap"]
    actual_gate = after["beta3"] - before["beta3"]
    return {
        "schema": "bpd/ink-bpd04-frozen-reader-patch/v1",
        "evidence_class": "exact_recomputed_reader_with_local_linearization",
        "physical_dose_l2": _norm(dv),
        "rms_before": before["block3_l2"] / math.sqrt(len(b)),
        "rms_after": after["block3_l2"] / math.sqrt(len(p)),
        "predicted_gap_change": predicted_gap,
        "actual_gap_change": actual_gap,
        "gap_linearization_residual": actual_gap - predicted_gap,
        "predicted_beta3_change": predicted_gate,
        "actual_beta3_change": actual_gate,
        "gate_linearization_residual": actual_gate - predicted_gate,
    }
