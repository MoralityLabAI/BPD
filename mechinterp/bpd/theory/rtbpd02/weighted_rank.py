"""CPU reference for rank-constrained physical operators in fixed metrics."""
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
    full_error: float = float("nan")


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return out


def _rank(rank: int) -> int:
    if not isinstance(rank, (int, np.integer)) or isinstance(rank, (bool, np.bool_)) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    return int(rank)


def _tolerance(rtol: float) -> float:
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError("rtol must be in (0,1)")
    return float(rtol)


def _scaled_rows(rows: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    x = _matrix(rows, "rows")
    if not len(x):
        raise ValueError("at least one row is required")
    if weights is None:
        return x / np.sqrt(len(x))
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (len(x),) or not np.isfinite(w).all() or np.any(w < 0) or w.max() == 0:
        raise ValueError("weights must be finite, nonnegative, and have positive total")
    scaled = w / w.max()
    return np.sqrt(scaled / scaled.sum())[:, None] * x


def uncentered_second_moment(rows: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return E[x x^T] in the caller's specified sampling measure."""
    if weights is None:
        x = _matrix(rows, "rows")
        if not len(x):
            raise ValueError("at least one row is required")
        return (x.T @ x) / len(x)
    x = _scaled_rows(rows, weights)
    return x.T @ x


def paired_contrast_moment(a: np.ndarray, b: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return the second moment of paired A/B differences."""
    aa = _matrix(a, "a")
    bb = _matrix(b, "b")
    if aa.shape != bb.shape:
        raise ValueError("paired worlds need the same shape")
    return uncentered_second_moment(aa - bb, weights)


def truncated_svd(operator: np.ndarray, rank: int) -> np.ndarray:
    w = _matrix(operator, "operator")
    rank = _rank(rank)
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]


def _moment_modes(covariance: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    cov = _matrix(covariance, "covariance")
    if cov.shape != (width, width):
        raise ValueError("covariance shape must match operator inputs")
    if not width:
        return cov, np.empty(0)
    scale = np.max(np.abs(cov))
    if not scale:
        return np.eye(width), np.zeros(width)
    scaled = cov / scale
    # Validate PSD relative to the declared matrix scale before selecting support.
    if np.max(np.abs(scaled - scaled.T)) > 1e-12:
        raise ValueError("covariance must be symmetric")
    eigenvalues, vectors = np.linalg.eigh((scaled + scaled.T) / 2)
    if eigenvalues[0] < -1e-12 * np.max(np.abs(eigenvalues)):
        raise ValueError("covariance must be positive semidefinite")
    roots = np.sqrt(np.maximum(eigenvalues, 0)) * np.sqrt(scale)
    return vectors, roots


def _retained(values: np.ndarray, cutoff: float) -> np.ndarray:
    if not len(values) or values.max() == 0:
        return np.zeros(len(values), dtype=bool)
    return values / values.max() > cutoff


def _reader_modes(reader: np.ndarray | None, width: int, rtol: float):
    if reader is None:
        return None, None, None
    a = _matrix(reader, "reader")
    if a.shape[1] != width:
        raise ValueError("reader input width must match operator output width")
    _, values, vt = np.linalg.svd(a, full_matrices=False)
    kept = _retained(values, rtol)
    return a, vt[kept, :].T, values[kept]


def _squared(values: np.ndarray) -> float:
    return float(np.sum(np.square(values)))


def _solve(w, basis, roots, rank, reader, rtol) -> RankSolution:
    kept = _retained(roots, np.sqrt(rtol))
    input_basis = basis[:, kept]
    input_roots = roots[kept]
    a, output_basis, output_values = _reader_modes(reader, w.shape[0], rtol)
    weighted_w = (w @ input_basis) * input_roots
    target = weighted_w if a is None else output_values[:, None] * (output_basis.T @ weighted_w)
    u, s, vt = np.linalg.svd(target, full_matrices=False)
    projected = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    lift = projected if a is None else output_basis @ (projected / output_values[:, None])
    estimate = (lift / input_roots) @ input_basis.T
    defect = w - estimate
    effective = (defect @ input_basis) * input_roots
    full = (defect @ basis) * roots
    if a is not None:
        effective = output_values[:, None] * (output_basis.T @ effective)
        full = a @ full
    reader_rank = w.shape[0] if a is None else len(output_values)
    return RankSolution(
        estimate, _squared(effective), _squared(s[rank:]), s,
        len(input_roots), reader_rank, _squared(full),
    )


def optimal_weighted_rank(
    operator: np.ndarray,
    covariance: np.ndarray,
    rank: int,
    reader: np.ndarray | None = None,
    *,
    rtol: float = 1e-12,
) -> RankSolution:
    """Solve the rank-constrained loss on the retained input and reader supports.

    Covariance eigenvalues and reader singular values must exceed rtol times
    their respective maxima. The effective loss, its spectral tail and the
    lift share those supports. full_error evaluates the same estimate using
    all modes of the symmetrized PSD moment and the supplied reader.
    """
    w = _matrix(operator, "operator")
    rank = _rank(rank)
    rtol = _tolerance(rtol)
    basis, roots = _moment_modes(covariance, w.shape[1])
    return _solve(w, basis, roots, rank, reader, rtol)


def optimal_sample_rank(
    operator: np.ndarray,
    rows: np.ndarray,
    rank: int,
    reader: np.ndarray | None = None,
    *,
    weights: np.ndarray | None = None,
    rtol: float = 1e-12,
) -> RankSolution:
    """Solve the same problem using a thin SVD of weighted sample rows."""
    w = _matrix(operator, "operator")
    rank = _rank(rank)
    rtol = _tolerance(rtol)
    x = _scaled_rows(rows, weights)
    if x.shape[1] != w.shape[1]:
        raise ValueError("row width must match operator inputs")
    _, roots, vt = np.linalg.svd(x, full_matrices=False)
    return _solve(w, vt.T, roots, rank, reader, rtol)


def weighted_error(
    operator: np.ndarray,
    estimate: np.ndarray,
    covariance: np.ndarray,
    reader: np.ndarray | None = None,
    *,
    rtol: float = 1e-12,
    full: bool = False,
) -> float:
    """Evaluate a candidate in the effective metric or the original PSD metric."""
    w = _matrix(operator, "operator")
    k = _matrix(estimate, "estimate")
    if k.shape != w.shape:
        raise ValueError("estimate must have the operator shape")
    rtol = _tolerance(rtol)
    basis, roots = _moment_modes(covariance, w.shape[1])
    a, output_basis, output_values = _reader_modes(reader, w.shape[0], rtol)
    if not full:
        kept = _retained(roots, np.sqrt(rtol))
        basis, roots = basis[:, kept], roots[kept]
    residual = ((w - k) @ basis) * roots
    if a is not None:
        residual = a @ residual if full else output_values[:, None] * (output_basis.T @ residual)
    return _squared(residual)
