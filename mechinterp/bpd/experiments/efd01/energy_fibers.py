"""BPD-EFD-01: physical energy-tangent dark-channel linear algebra.

Source-only NumPy instrument. See PROTOCOL.md, committed before this file.
All projections use the physical Euclidean state metric. They do not certify
finite nonlinear invariance, causal mediation, or nonorthogonal gauge transport.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

RTOL = 1e-9
ATOL = 1e-12


def vector(x, *, width=None):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError("expected a finite one-dimensional physical vector")
    if width is not None and len(arr) != width:
        raise ValueError("vector width mismatch")
    return arr


def matrix(x, *, width=None):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError("expected a finite two-dimensional physical matrix")
    if width is not None and arr.shape[1] != width:
        raise ValueError("matrix physical input width mismatch")
    return arr


def row_nullspace(a, *, rtol=RTOL, atol=ATOL):
    """Full right singular-vector kernel basis, including wide-matrix nulls."""
    a = matrix(a)
    if a.shape[1] < 1:
        raise ValueError("physical state width must be positive")
    if rtol < 0 or atol < 0:
        raise ValueError("negative numerical rank cutoff")
    if a.shape[0] == 0:
        return np.eye(a.shape[1]), np.zeros(0), 0, atol
    _, singular, vh = np.linalg.svd(a, full_matrices=True)
    threshold = max(atol, rtol * float(singular[0]))
    rank = int(np.count_nonzero(singular > threshold))
    return vh[rank:, :].T.copy(), singular, rank, threshold


@dataclass(frozen=True)
class EnergyFiber:
    """Exact physical kernel under a stated numerical rank tolerance."""
    basis: np.ndarray
    projector: np.ndarray
    current_rank: int
    joint_rank: int
    dimension: int
    lower_bound: int
    singular_values: np.ndarray
    threshold: float
    constraint_residual: float
    projector_error: float


def energy_fiber(current_jacobian, energy_gradient, *, rtol=RTOL, atol=ATOL):
    """K=ker(A) intersect ker(g) in physical state coordinates."""
    g = vector(energy_gradient)
    if g.size == 0:
        raise ValueError("empty physical state")
    a = matrix(current_jacobian, width=len(g))
    _, _, rank_a, _ = row_nullspace(a, rtol=rtol, atol=atol)
    constraints = np.concatenate((a, g[None, :]), axis=0)
    basis, singular, joint_rank, threshold = row_nullspace(
        constraints, rtol=rtol, atol=atol
    )
    projector = basis @ basis.T
    dimension = len(g) - joint_rank
    lower_bound = len(g) - rank_a - 1
    if dimension < lower_bound:
        raise ArithmeticError("rank-nullity dimension bound violated")
    residual = float(np.linalg.norm(constraints @ basis, ord="fro"))
    idempotence = float(np.linalg.norm(projector @ projector - projector, ord="fro"))
    return EnergyFiber(
        basis=basis, projector=projector, current_rank=rank_a,
        joint_rank=joint_rank, dimension=dimension, lower_bound=lower_bound,
        singular_values=singular, threshold=threshold,
        constraint_residual=residual, projector_error=idempotence,
    )


def delayed_rank(future_jacobian, fiber: EnergyFiber, *,
                 rtol=RTOL, atol=ATOL):
    """Rank of future Jacobian restricted to present-energy-dark K."""
    b = matrix(future_jacobian, width=fiber.projector.shape[0])
    restricted = b @ fiber.basis
    _, sv, rank, threshold = row_nullspace(
        restricted, rtol=rtol, atol=atol
    ) if restricted.shape[1] else (
        np.zeros((0, 0)), np.zeros(0), 0, atol
    )
    return {"rank": rank, "singular_values": sv, "threshold": threshold,
            "restricted_operator": restricted}


def split_native_update(native_update, energy_gradient, *, atol=ATOL):
    """v=-alpha*g+r, where g dot r=0. Undefined if ||g|| is zero."""
    v = vector(native_update)
    g = vector(energy_gradient, width=len(v))
    norm2 = float(g @ g)
    if norm2 <= atol * atol:
        return {"identified": False, "alpha": None, "tangent": v.copy(),
                "descent": np.zeros_like(v), "orthogonality": None}
    alpha = -float(g @ v) / norm2
    descent = -alpha * g
    tangent = v - descent
    return {"identified": True, "alpha": alpha, "descent": descent,
            "tangent": tangent, "orthogonality": float(g @ tangent)}


def skew_representer(tangent, energy_gradient, *, atol=ATOL):
    """Algebraic Qg=r. Any tangent admits this Q; NOT mechanism identification."""
    r = vector(tangent)
    g = vector(energy_gradient, width=len(r))
    norm2 = float(g @ g)
    if norm2 <= atol * atol:
        raise ValueError("skew representer undefined at zero energy gradient")
    if abs(float(r @ g)) > 1e-9 * max(1.0, float(np.linalg.norm(r) * np.linalg.norm(g))):
        raise ValueError("tangent residual is not perpendicular to gradient")
    return (np.outer(r, g) - np.outer(g, r)) / norm2


def tangent_shams(fiber: EnergyFiber, *, radius, count=8, seed):
    """Label-blind, equal-physical-dose isotropic K directions for CPU tests."""
    if radius <= 0 or count < 1 or not np.isfinite(radius):
        raise ValueError("invalid positive radius or sham count")
    k = fiber.dimension
    if k == 0:
        raise ValueError("no dark tangent directions")
    rng = np.random.default_rng(int(seed))
    out = []
    for _ in range(count):
        coeff = rng.standard_normal(k)
        norm = float(np.linalg.norm(coeff))
        if norm <= ATOL:
            raise ArithmeticError("random null coefficient unexpectedly zero")
        out.append(fiber.basis @ (coeff * (radius / norm)))
    return np.stack(out, axis=0)
