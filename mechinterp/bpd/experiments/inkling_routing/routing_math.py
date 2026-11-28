"""INK-BPD-01: source-only cross-loop routing attribution and dose controls.

Pure standard-library algebraic fixture motivated by Inkling 2 Mini / Goodfire.
The equations model an idealized two-block attention-residual message. They are
NOT a hooked Inkling forward pass and contain NO model or causal measurements.
"""
from __future__ import annotations

import math
from typing import Sequence


def _vector(v: Sequence[float], name: str) -> tuple[float, ...]:
    if not v or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
                    for x in v):
        raise ValueError(f"{name} requires a nonempty finite real vector")
    return tuple(float(x) for x in v)


def rms(v: Sequence[float]) -> float:
    x = _vector(v, "rms")
    return math.sqrt(sum(t * t for t in x) / len(x))


def rms_normalize(v: Sequence[float]) -> tuple[float, ...]:
    x = _vector(v, "rms_normalize")
    scale = rms(x)
    if scale <= 0:
        raise ValueError("zero vector cannot be RMS normalized")
    return tuple(t / scale for t in x)


def l2(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("length mismatch")
    a, b = _vector(x, "x"), _vector(y, "y")
    return math.sqrt(sum((u - v)**2 for u, v in zip(a, b)))


def route_gap(reader: Sequence[float], block2: Sequence[float], block3: Sequence[float]) -> dict:
    """Return logit gap s3-s2 and its exact coordinate-wise additive terms.

    Idealized Goodfire equation: s_i = u dot RMSNorm(v_i) / sqrt(d);
    beta_3 = sigmoid(s3-s2). No residual gate or transformer is simulated.
    """
    u = _vector(reader, "reader")
    v2, v3 = rms_normalize(block2), rms_normalize(block3)
    if len(u) != len(v2) or len(u) != len(v3):
        raise ValueError("reader, block2 and block3 must have equal width")
    terms = tuple(ui * (b - a) / math.sqrt(len(u)) for ui, a, b in zip(u, v2, v3))
    gap = sum(terms)
    beta3 = (1. / (1. + math.exp(-gap)) if gap >= 0
             else math.exp(gap) / (1. + math.exp(gap)))
    return {"gap": gap, "beta3": beta3, "per_coordinate": terms, "dimension": len(u)}


def rotate_pair_at_fixed_dose(v: Sequence[float], i: int, j: int, dose: float,
                              orientation: int = 1) -> tuple[float, ...]:
    """Orthogonal two-coordinate rotation with specified delivered L2 distance.

    Preserves the full block's RMS and matches the *same* physical L2 dose
    for distinct coordinate pairs. Feasible only when 0 <= dose <= 2*r_ij.
    Orthogonality alone does not make the causal arms equally informative.
    """
    x = list(_vector(v, "block"))
    if not (0 <= i < len(x) and 0 <= j < len(x) and i != j):
        raise ValueError("distinct valid coordinates required")
    if orientation not in (-1, 1):
        raise ValueError("orientation must be +1 or -1")
    if isinstance(dose, bool) or not isinstance(dose, (int, float)) or not math.isfinite(dose) or dose < 0:
        raise ValueError("finite nonnegative dose required")
    radius = math.hypot(x[i], x[j])
    if radius == 0 or dose > 2 * radius:
        raise ValueError("unachievable dose for selected coordinate pair")
    angle = orientation * 2. * math.asin(min(1., float(dose) / (2. * radius)))
    co, si = math.cos(angle), math.sin(angle)
    xi, xj = x[i], x[j]
    x[i], x[j] = co * xi - si * xj, si * xi + co * xj
    return tuple(x)


def message(v2: Sequence[float], v3: Sequence[float], beta3: float) -> tuple[float, ...]:
    """Idealized previous-loop message under beta2=1-beta3."""
    a, b = rms_normalize(v2), rms_normalize(v3)
    if len(a) != len(b) or not 0 <= beta3 <= 1:
        raise ValueError("invalid mixture")
    return tuple((1 - beta3) * x + beta3 * y for x, y in zip(a, b))


def path_decomposition(reader: Sequence[float], v2: Sequence[float], before3: Sequence[float],
                       after3: Sequence[float]) -> dict:
    """Telescoping route-vs-content decomposition for a frozen v2.

    This is a counterfactual algebraic decomposition under a *chosen*
    intervention graph. In a real model, changing a block also changes
    upstream activations/gates, so hook and verify that graph independently.
    """
    old = route_gap(reader, v2, before3)["beta3"]
    new = route_gap(reader, v2, after3)["beta3"]
    a, b = rms_normalize(v2), rms_normalize(before3)
    c = rms_normalize(after3)
    clean = tuple((1-old)*x + old*y for x, y in zip(a,b))
    full = tuple((1-new)*x + new*y for x, y in zip(a,c))
    # Carry changed selector over clean content, then changed content over clean selector.
    selector_only = tuple((new-old)*(y-x) for x,y in zip(a,b))
    content_only = tuple(old*(z-y) for y,z in zip(b,c))
    interaction = tuple((new-old)*(z-y) for y,z in zip(b,c))
    assert all(abs(v-u-(r+s+t)) < 1e-10 for u,v,r,s,t in
               zip(clean,full,selector_only,content_only,interaction))
    return {"clean_beta3":old,"patched_beta3":new,"clean_message":clean,"full_message":full,
            "selector_only_delta":selector_only,"content_only_delta":content_only,
            "interaction_delta":interaction,
            "total_delta":tuple(x-y for x,y in zip(full,clean))}


def projective_shares(contributions: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Goodfire-style contribution shares s_j=(c_j dot x)/||x||^2.

    Signs may be negative and sums can cancel. The unit-sum identity requires
    exactly x=sum_j c_j, including any bias, bypass or residual contributions.
    """
    if not contributions:
        raise ValueError("at least one contribution required")
    cols = [_vector(col,"contribution") for col in contributions]
    width = len(cols[0])
    if any(len(col) != width for col in cols):
        raise ValueError("width mismatch")
    x = [sum(col[k] for col in cols) for k in range(width)]
    denom = sum(t*t for t in x)
    if denom == 0:
        raise ValueError("zero summed input has undefined shares")
    return tuple(sum(c*t for c,t in zip(col,x))/denom for col in cols)


def paired_dose_audit(reader: Sequence[float], v2: Sequence[float], v3: Sequence[float],
                      target_pair: tuple[int,int], sham_pair: tuple[int,int], dose: float) -> dict:
    """Deterministic source-only physical controls; absent model endpoints."""
    clean = _vector(v3,"v3")
    target = rotate_pair_at_fixed_dose(clean,*target_pair,dose)
    sham = rotate_pair_at_fixed_dose(clean,*sham_pair,dose)
    tr,sh = path_decomposition(reader,v2,clean,target),path_decomposition(reader,v2,clean,sham)
    def desc(v,path):
        return {"dose_l2":l2(clean,v),"rms_before":rms(clean),"rms_after":rms(v),
                "beta3_change":path["patched_beta3"]-path["clean_beta3"],
                "message_change_l2":l2(path["full_message"],path["clean_message"]),
                "selector_only_l2":l2(tuple(0. for _ in clean),path["selector_only_delta"]),
                "content_only_l2":l2(tuple(0. for _ in clean),path["content_only_delta"])}
    return {"status":"source_only_math_fixture","target":desc(target,tr),"sham":desc(sham,sh),
            "target_minus_sham_beta_change":tr["patched_beta3"]-sh["patched_beta3"],
            "uncertainty":"Algebraic fixture, not a physical Inkling causal mediation effect."}
