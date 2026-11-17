"""BPD-OUT03: task-gradient decomposition for exact selective SAE kills.

A full row rank physical encoder A defines a minimum-norm coordinate kill
delta in row(A) and a same-dose sham in ker(A). The task-margin difference
splits exactly into the coordinate dual and unconstrained tangent gradient.

This is local linear mathematics; no real-model gradient is inferred.
"""
from __future__ import annotations

from fractions import Fraction as Q
import math
import numpy as np


def exact_rank2_counterexample(tangent_gradient: int) -> dict:
    """Exact Fractions: same selected code/kill/sham, tunable task verdict."""
    if not isinstance(tangent_gradient,int):
        raise TypeError("exact witness accepts integer tangent gradient")
    a=((Q(1),Q(0),Q(0)),(Q(0),Q(1),Q(0)))
    delta=(Q(1),Q(0),Q(0))
    sham=(Q(0),Q(0),Q(1))
    g=(Q(1),Q(0),Q(tangent_gradient))
    b=(Q(1),Q(0))
    assert tuple(sum(a[i][j]*delta[j] for j in range(3)) for i in range(2))==b
    assert all(sum(a[i][j]*sham[j] for j in range(3))==0 for i in range(2))
    d=sum(g[j]*(delta[j]-sham[j]) for j in range(3))
    return {"axial_term":Q(1),"tangent_sham_term":Q(tangent_gradient),
            "paired_margin_damage":d,"kill_norm_squared":Q(1),
            "sham_norm_squared":Q(1)}


def certificate(A, g, selected_index, overshoot, sham, *,
                curvature_bound=0.0, tolerance=1e-9) -> dict:
    """Compute a minimum-norm coordinate erasure and its gradient split.

    Given full-row-rank A (k×d), e_j, b=overshoot*e_j, define
        delta=A.T @ (A@A.T)^-1 @ b   in row(A)
        g_R=A.T @ (A@A.T)^-1 @ A@g
        g_T=g-g_R
        lambda=(A@A.T)^-1 @ A@g.

    For sham s in ker(A) with ||s||=||delta||, the task-local linear effect
    D=g.T@(delta-s)=overshoot*lambda[j]-g_T.T@s. In the nonlinear case
    with ||H_m||op<=curvature_bound on both edit segments, |D_true-D|
    <= curvature_bound * ||delta||**2.

    Returns bounds applying to any same-dose tangent sham and this sham.
    """
    A=np.asarray(A,dtype=np.float64)
    g=np.asarray(g,dtype=np.float64)
    s=np.asarray(sham,dtype=np.float64)
    if A.ndim!=2 or g.ndim!=1 or s.ndim!=1:
        raise ValueError("expected A matrix, gradient vector and sham vector")
    k,d=A.shape
    if not 1<=k<d or g.shape!=(d,) or s.shape!=(d,) or not 0<=selected_index<k:
        raise ValueError("invalid physical or index dimensions")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(g)) and
            np.all(np.isfinite(s)) and math.isfinite(overshoot) and
            math.isfinite(curvature_bound)):
        raise ValueError("nonfinite input")
    if overshoot<=0 or curvature_bound<0:
        raise ValueError("overshoot positive and curvature bound nonnegative")
    sigma=np.linalg.svd(A,compute_uv=False)
    if sigma[-1]<=1e-10:
        raise ValueError("physical encoder rows rank deficient")
    gram=A@A.T
    b=np.zeros(k,dtype=np.float64)
    b[selected_index]=overshoot
    delta=A.T@np.linalg.solve(gram,b)
    lamb=np.linalg.solve(gram,A@g)
    g_row=A.T@lamb
    g_tan=g-g_row
    dose=float(np.linalg.norm(delta))
    if (abs(float(np.linalg.norm(s))-dose)>tolerance or
        float(np.linalg.norm(A@s,ord=np.inf))>tolerance):
        raise ValueError("sham must be same physical dose and tangent to all encoder rows")
    if float(np.linalg.norm(A@delta-b,ord=np.inf))>tolerance:
        raise ArithmeticError("minimum-norm target erasure did not meet active constraints")
    if float(np.linalg.norm(delta))<1e-12:
        raise ArithmeticError("target erasure has zero dose")
    if float(abs(delta@g_tan))>tolerance or float(abs(s@g_row))>tolerance:
        raise ArithmeticError("row/tangent score geometry failed")
    axial=float(overshoot*lamb[selected_index])
    tang=float(g_tan@s)
    d_linear=float(g@(delta-s))
    if abs(d_linear-(axial-tang))>tolerance:
        raise ArithmeticError("paired effect does not split into encoder dual and task tangent")
    radius=dose*float(np.linalg.norm(g_tan))
    remainder=curvature_bound*dose*dose
    return {
        "selected_feature":int(selected_index),
        "encoder_rows":int(k),"hidden_width":int(d),
        "physical_rank":int(k),
        "dose":dose,
        "dual_gradient_coeff":float(lamb[selected_index]),
        "task_gradient_row_norm":float(np.linalg.norm(g_row)),
        "task_gradient_tangent_norm":float(np.linalg.norm(g_tan)),
        "axial_term":axial,
        "tangent_sham_term":tang,
        "paired_linear_margin_damage":d_linear,
        "all_tangent_sham_linear_damage_range":[axial-radius,axial+radius],
        "nonlinear_curvature_budget":remainder,
        "certified_nonlinear_interval_for_this_sham":[d_linear-remainder,d_linear+remainder],
        "any_sham_positive_sufficient":bool(axial>radius+remainder),
        "delta_min_norm":delta.tolist(),
        "sham":s.tolist(),
    }


def sample_tangent_sham(A, physical_dose, seed=202623):
    """Independent seedable representative of the exact physical tangent sphere."""
    A=np.asarray(A,dtype=np.float64)
    k,d=A.shape
    if not 1<=k<d or physical_dose<=0:
        raise ValueError("nonzero nullspace and positive dose required")
    _,sigma,vh=np.linalg.svd(A,full_matrices=True)
    if sigma[-1]<=1e-10:
        raise ValueError("encoder rank deficient")
    tangent=vh[k:,:].T
    generator=np.random.default_rng(seed)
    r=tangent@generator.normal(size=(d-k,))
    norm=float(np.linalg.norm(r))
    if norm<=1e-12:
        raise ValueError("sham sampling degenerate")
    sham=r*(physical_dose/norm)
    if float(np.linalg.norm(A@sham,ord=np.inf))>1e-8:
        raise ArithmeticError("sampled vector is not tangent")
    return sham
