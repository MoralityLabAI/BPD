"""RT-BPD-02 source-only weighted rank approximation reference implementation.

All functions are NumPy CPU operations over supplied arrays. No model execution
or external data access is performed. This module describes physical operators,
not learned individual BPD atom identities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankSolution:
    operator: np.ndarray
    error: float
    tail_oracle: float
    singular_values: np.ndarray
    input_support_rank: int
    reader_rank: int


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return out


def uncentered_second_moment(rows: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return E[x x^T]. Rows must be the intended independent weighted units."""
    x = _matrix(rows, "rows")
    if not x.shape[0]:
        raise ValueError("at least one row is required")
    if weights is None:
        return x.T @ x / x.shape[0]
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (x.shape[0],) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, and have positive total")
    return x.T @ (w[:, None] * x) / w.sum()


def paired_contrast_moment(a: np.ndarray, b: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Second moment of paired A/B differences. Caller guarantees root balance."""
    aa = _matrix(a, "a")
    bb = _matrix(b, "b")
    if aa.shape != bb.shape:
        raise ValueError("paired worlds need the same shape")
    return uncentered_second_moment(aa - bb, weights)


def truncated_svd(operator: np.ndarray, rank: int) -> np.ndarray:
    w = _matrix(operator, "operator")
    if not isinstance(rank, (int, np.integer)) or isinstance(rank, (bool, np.bool_)) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    k = min(rank, len(s))
    return (u[:, :k] * s[:k]) @ vt[:k, :]


def _moment_sqrt_and_pinv(covariance: np.ndarray, n: int, rtol: float) -> tuple[np.ndarray, np.ndarray, int]:
    cov = _matrix(covariance, "covariance")
    if cov.shape != (n, n):
        raise ValueError("covariance shape must match operator inputs")
    scale = max(1.0, float(np.linalg.norm(cov, ord=2)))
    if not np.allclose(cov, cov.T, atol=1e-12 * scale, rtol=1e-12):
        raise ValueError("covariance must be symmetric")
    sym = (cov + cov.T) / 2
    eigval, eigvec = np.linalg.eigh(sym)
    if eigval[0] < -1e-12 * scale:
        raise ValueError("covariance must be positive semidefinite")
    positive = eigval > rtol * max(float(eigval[-1]), np.finfo(np.float64).tiny)
    basis = eigvec[:, positive]
    root = np.sqrt(eigval[positive])
    sqrt = (basis * root) @ basis.T
    pinv_sqrt = (basis / root) @ basis.T
    return sqrt, pinv_sqrt, int(positive.sum())


def optimal_weighted_rank(
    operator: np.ndarray,
    covariance: np.ndarray,
    rank: int,
    reader: np.ndarray | None = None,
    *,
    rtol: float = 1e-12,
) -> RankSolution:
    """Minimize ||reader @ (W - K) @ sqrt(covariance)||_F^2, rank(K)<=rank.

    With reader=None, the identity map on physical write space is used.
    A truncated SVD of reader@W@sqrt(covariance) gives the exact global
    optimum. Pseudoinverses pick a minimum-norm representative, including
    when covariance or reader is rank deficient.

    Eigenmodes below rtol * largest covariance eigenvalue are treated as
    numerical null directions. Readers use the same relative cutoff.
    """
    w = _matrix(operator, "operator")
    if not isinstance(rank, (int, np.integer)) or isinstance(rank, (bool, np.bool_)) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    if not np.isfinite(rtol) or rtol <= 0 or rtol >= 1:
        raise ValueError("rtol must be in (0,1)")
    m, n = w.shape
    a = np.eye(m) if reader is None else _matrix(reader, "reader")
    if a.shape[1] != m:
        raise ValueError("reader input width must match operator output width")
    sqrt, pinv_sqrt, n_support = _moment_sqrt_and_pinv(covariance, n, rtol)
    target = a @ w @ sqrt
    u, s, vt = np.linalg.svd(target, full_matrices=False)
    k = min(rank, len(s))
    target_rank_k = (u[:, :k] * s[:k]) @ vt[:k, :]
    if reader is None:
        estimate = target_rank_k @ pinv_sqrt
    else:
        estimate = np.linalg.pinv(a, rcond=rtol) @ target_rank_k @ pinv_sqrt
    error = float(np.linalg.norm(a @ (w - estimate) @ sqrt, ord="fro") ** 2)
    tail = float(np.sum(s[k:] ** 2))
    reader_rank = m if reader is None else int(np.linalg.matrix_rank(a, tol=rtol * (np.linalg.svd(a, compute_uv=False)[0] if a.size else 0)))
    return RankSolution(estimate, error, tail, s, n_support, reader_rank)


def weighted_error(operator: np.ndarray, estimate: np.ndarray, covariance: np.ndarray, reader: np.ndarray | None = None) -> float:
    """Evaluate a rank candidate with the same physical reached-state metric."""
    w = _matrix(operator, "operator")
    k = _matrix(estimate, "estimate")
    if k.shape != w.shape:
        raise ValueError("estimate must have the operator shape")
    a = np.eye(w.shape[0]) if reader is None else _matrix(reader, "reader")
    if a.shape[1] != w.shape[0]:
        raise ValueError("reader input width must match operator output width")
    sqrt, _, _ = _moment_sqrt_and_pinv(covariance, w.shape[1], 1e-12)
    return float(np.linalg.norm(a @ (w - k) @ sqrt, ord="fro") ** 2)
