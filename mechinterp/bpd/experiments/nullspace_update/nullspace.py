"""Protected low-rank linear updates in NumPy (no torch).

Setting: a linear layer W (d_out x d_in). Keep inputs X_keep (k x d_in) must map unchanged,
new-task inputs X_new (m x d_in) must move toward targets R (m x d_out). Each solver returns
D (d_in x d_out), the transpose of the weight edit. All solves run in float64 internally and
cast the result to the input dtype, so float32 inputs give X_keep D at float32 rounding level.

Stable protected form: D = N (X_new N)^+ R, with N an orthonormal basis of null(X_keep) taken
from a rank-revealing SVD of X_keep. The literal form P = I - X_keep^+ X_keep, U = pinv(X_new P) R
keeps rounding-level singular values of X_new P, so U explodes along span(X_keep) and D = P U leaks;
`literal_projector_update` reproduces that and is a diagnostic only.
"""
from __future__ import annotations

import numpy as np


def _out_dtype(*arrays):
    return np.result_type(*[a.dtype for a in arrays if np.issubdtype(a.dtype, np.floating)] or [np.float64])


def numerical_rank_tol(Xk: np.ndarray, tol: float | None = None) -> float:
    """Relative singular-value cutoff; default max(shape) * eps(dtype)."""
    if tol is not None:
        return float(tol)
    eps = np.finfo(Xk.dtype if np.issubdtype(Xk.dtype, np.floating) else np.float64).eps
    return float(max(Xk.shape) * eps)


def keep_spectrum(Xk: np.ndarray):
    """Return (U, S, Vt) of X_keep in float64 with full_matrices=True."""
    return np.linalg.svd(np.asarray(Xk, dtype=np.float64), full_matrices=True)


def null_basis(Xk: np.ndarray, tol: float | None = None):
    """Orthonormal basis N (d_in x (d_in - r)) of null(X_keep) and numerical rank r (float64)."""
    _, S, Vt = keep_spectrum(Xk)
    cut = numerical_rank_tol(Xk, tol) * (S.max() if S.size else 0.0)
    r = int((S > cut).sum()) if S.size else 0
    return Vt[r:].T.copy(), r


def keep_leak(Xk: np.ndarray, D: np.ndarray) -> float:
    """Scale-free leak ||X_keep D||_F / (||X_keep||_2 ||D||_F), in [0, 1]."""
    Xk64, D64 = np.asarray(Xk, np.float64), np.asarray(D, np.float64)
    nd = np.linalg.norm(D64)
    if nd == 0:
        return 0.0
    return float(np.linalg.norm(Xk64 @ D64) / (np.linalg.norm(Xk64, 2) * nd))


def _pinv_solve(A: np.ndarray, R: np.ndarray, rcond: float) -> np.ndarray:
    return np.linalg.pinv(A, rcond=rcond) @ R


def _truncate(C: np.ndarray, rank: int | None) -> np.ndarray:
    if rank is None or rank >= min(C.shape):
        return C
    if rank <= 0:
        return np.zeros_like(C)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    return (U[:, :rank] * S[:rank]) @ Vt[:rank]


def ordinary_ridge(Xn, R, mu: float = 1e-3):
    """argmin |Xn D - R|^2 + mu |D|^2 (dual form when m < d_in)."""
    X, Rr = np.asarray(Xn, np.float64), np.asarray(R, np.float64)
    m, d = X.shape
    if m < d:
        D = X.T @ np.linalg.solve(X @ X.T + mu * np.eye(m), Rr)
    else:
        D = np.linalg.solve(X.T @ X + mu * np.eye(d), X.T @ Rr)
    return D.astype(_out_dtype(Xn, R))


def soft_protected_ridge(Xn, Xk, R, lam: float = 10.0, mu: float = 1e-3):
    """argmin |Xn D - R|^2 + lam |Xk D|^2 + mu |D|^2."""
    X, K, Rr = (np.asarray(a, np.float64) for a in (Xn, Xk, R))
    A = X.T @ X + lam * (K.T @ K) + mu * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Rr).astype(_out_dtype(Xn, Xk, R))


def nullspace_update(Xn, Xk, R, rank: int | None = None, rcond: float = 1e-10,
                     N: np.ndarray | None = None, tol: float | None = None):
    """Stable protected update D = N (Xn N)^+ R; with `rank`, rank-r inside null(X_keep).

    Empty null space returns zeros. Truncation acts on the coordinates C = (Xn N)^+ R, so the
    result stays inside range(N) and X_keep D stays at rounding level for any rank.
    """
    if N is None:
        N, _ = null_basis(Xk, tol)
    d_out = np.asarray(R).shape[1]
    dt = _out_dtype(Xn, Xk, R)
    if N.shape[1] == 0:
        return np.zeros((np.asarray(Xn).shape[1], d_out), dt)
    C = _pinv_solve(np.asarray(Xn, np.float64) @ N, np.asarray(R, np.float64), rcond)
    return (N @ _truncate(C, rank)).astype(dt)


def truncated_svd_update(Xn, R, rank: int, rcond: float = 1e-10):
    """Unprotected least squares update truncated to rank r."""
    D = _pinv_solve(np.asarray(Xn, np.float64), np.asarray(R, np.float64), rcond)
    return _truncate(D, rank).astype(_out_dtype(Xn, R))


def random_matched(shape, rank: int, fro: float, rng: np.random.Generator, dtype=np.float64):
    """Random rank-r matrix scaled to Frobenius norm `fro`."""
    D = rng.standard_normal((shape[0], rank)) @ rng.standard_normal((rank, shape[1]))
    return (D * (fro / max(np.linalg.norm(D), 1e-300))).astype(dtype)


def approx_protected_update(Xn, Xk, R, energy: float = 0.9, k: int | None = None,
                            rank: int | None = None, rcond: float = 1e-10):
    """Protect only the top-k energy directions of X_keep (k from the cumulative squared
    singular-value fraction `energy` unless given). Larger null space, so a better new-task fit,
    at the price of a leak through the discarded directions. Returns (D, info) with info holding
    k, the protected energy fraction and keep_leak against the full X_keep."""
    _, S, Vt = keep_spectrum(Xk)
    cum = np.cumsum(S ** 2) / max((S ** 2).sum(), 1e-300)
    if k is None:
        k = int(np.searchsorted(cum, energy) + 1)
    k = int(min(max(k, 0), len(S)))
    N = Vt[k:].T.copy()
    D = nullspace_update(Xn, Xk, R, rank=rank, rcond=rcond, N=N)
    return D, {"k": k, "protected_energy": float(cum[k - 1]) if k else 0.0, "leak": keep_leak(Xk, D)}


def literal_projector_update(Xn, Xk, R, rcond: float | None = None):
    """Diagnostic: D = P pinv(Xn P) R, P = I - pinv(Xk) Xk, NumPy default rcond."""
    X, K, Rr = (np.asarray(a, np.float64) for a in (Xn, Xk, R))
    P = np.eye(K.shape[1]) - np.linalg.pinv(K) @ K
    U = np.linalg.pinv(X @ P) if rcond is None else np.linalg.pinv(X @ P, rcond=rcond)
    U = U @ Rr
    return (P @ U).astype(_out_dtype(Xn, Xk, R)), U


def adaptive_rank_allocation(spectra, budget: int):
    """Greedy rank split across modules under a total budget by marginal squared singular value.
    Optimal for the captured-energy objective because each spectrum is sorted descending."""
    spectra = [np.sort(np.asarray(s, np.float64))[::-1] for s in spectra]
    alloc = [0] * len(spectra)
    for _ in range(budget):
        gains = [(s[a] ** 2 if a < len(s) else -1.0) for s, a in zip(spectra, alloc)]
        i = int(np.argmax(gains))
        if gains[i] <= 0:
            break
        alloc[i] += 1
    return alloc


def uniform_rank_allocation(spectra, budget: int):
    n = len(spectra)
    return [min(budget // n + (1 if i < budget % n else 0), len(s)) for i, s in enumerate(spectra)]


def null_coordinate_spectrum(Xn, Xk, R, rcond: float = 1e-10, tol: float | None = None):
    """Singular values of the full null-space update coordinates (input to rank allocation)."""
    N, _ = null_basis(Xk, tol)
    if N.shape[1] == 0:
        return np.zeros(0)
    C = _pinv_solve(np.asarray(Xn, np.float64) @ N, np.asarray(R, np.float64), rcond)
    return np.linalg.svd(C, compute_uv=False)
