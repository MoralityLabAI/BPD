"""Finite-dimensional identities used by RT-BPD-02."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weighted_rank import _matrix, _moment_modes, _retained, _tolerance


@dataclass(frozen=True)
class Disagreement:
    operator: np.ndarray
    state: np.ndarray
    reader: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.operator + self.state + self.reader

    def energies(self) -> dict[str, float]:
        return {
            "operator": float(self.operator @ self.operator),
            "state": float(self.state @ self.state),
            "reader": float(self.reader @ self.reader),
            "operator_state_cross": float(2 * (self.operator @ self.state)),
            "operator_reader_cross": float(2 * (self.operator @ self.reader)),
            "state_reader_cross": float(2 * (self.state @ self.reader)),
            "total": float(self.total @ self.total),
        }


def disagreement(k_a, k_b, x_a, x_b, reader_a=None, reader_b=None) -> Disagreement:
    """Split b minus a in common input, write and measured frames."""
    a = _matrix(k_a, "k_a")
    b = _matrix(k_b, "k_b")
    xa, xb = np.asarray(x_a, dtype=np.float64), np.asarray(x_b, dtype=np.float64)
    if a.shape != b.shape or xa.shape != (a.shape[1],) or xa.shape != xb.shape:
        raise ValueError("operators and reached states need common dimensions")
    if not np.isfinite(xa).all() or not np.isfinite(xb).all():
        raise ValueError("reached states must be finite")
    op = (b - a) @ ((xa + xb) / 2)
    state = ((a + b) / 2) @ (xb - xa)
    if reader_a is None and reader_b is None:
        return Disagreement(op, state, np.zeros_like(op))
    ar, br = _matrix(reader_a, "reader_a"), _matrix(reader_b, "reader_b")
    if ar.shape != br.shape or ar.shape[1] != a.shape[0]:
        raise ValueError("both readers must map writes into the same measured frame")
    mean_reader = (ar + br) / 2
    reader = (br - ar) @ ((a @ xa + b @ xb) / 2)
    return Disagreement(mean_reader @ op, mean_reader @ state, reader)


def fixed_future_reader(write, transitions, readers, weights) -> np.ndarray:
    """Stack weighted readouts at horizons zero through len(transitions)."""
    propagated = _matrix(write, "write")
    w = np.asarray(weights, dtype=np.float64)
    if len(readers) != 1 + len(transitions) or w.shape != (len(readers),):
        raise ValueError("supply one reader and weight per horizon, including zero")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("reader weights must be nonnegative and finite")
    blocks = []
    for horizon, (reader, weight) in enumerate(zip(readers, w)):
        if horizon:
            propagated = _matrix(transitions[horizon - 1], "transition") @ propagated
        block = np.sqrt(weight) * (_matrix(reader, "reader") @ propagated)
        blocks.append(block)
    return np.vstack(blocks)


def _singular_rank(singular_values, rtol: float) -> int:
    return int(np.count_nonzero(_retained(singular_values, rtol)))


def delayed_dark_rank(present, future, domain, *, rtol: float = 1e-12) -> int:
    """Numerical rank of the future reader restricted to currently dark writes.

    First orthonormalize the supplied write domain. Identify the kernel in
    current-reader units, then threshold its future image in future-reader
    units on the whole supplied domain.
    """
    rtol = _tolerance(rtol)
    d = _matrix(domain, "domain")
    current = _matrix(present, "present")
    later = _matrix(future, "future")
    q, s, _ = np.linalg.svd(d, full_matrices=False)
    q = q[:, :_singular_rank(s, rtol)]
    _, now_s, now_vt = np.linalg.svd(current @ q, full_matrices=True)
    null = now_vt[_singular_rank(now_s, rtol):, :].T
    forward = later @ q
    scale = np.linalg.norm(forward, ord=2) if forward.size else 0.0
    delayed_s = np.linalg.svd(forward @ null, compute_uv=False)
    return int(np.count_nonzero(delayed_s / scale > rtol)) if scale else 0


def direction(observed, reference, *, zero_atol: float = 0.0) -> dict[str, float | str | None]:
    """Measured length, signed parallel gain, and cosine against a fixed direction."""
    x = np.asarray(observed, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("observed and reference must be finite vectors of equal shape")
    if not np.isfinite(zero_atol) or zero_atol < 0:
        raise ValueError("zero_atol must be finite and nonnegative")
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if ny <= zero_atol:
        return {"status": "zero_reference", "gain": None, "cosine": None, "length": float(nx)}
    if nx <= zero_atol:
        return {"status": "zero_observed", "gain": 0.0, "cosine": None, "length": float(nx)}
    dot = float((x / nx) @ (y / ny))
    cosine = float(np.clip(dot, -1, 1))
    return {"status": "ok", "gain": float(cosine * (nx / ny)), "cosine": cosine, "length": float(nx)}


def dominance_constant(source, target, *, rtol: float = 1e-12) -> float:
    """Smallest PSD domination constant on numerical support, or infinity.

    Intended for diagnostics at modest dimension. A finite numerical result
    is not a certificate for omitted or poorly resolved support directions.
    """
    rtol = _tolerance(rtol)
    p = _matrix(source, "source")
    n = p.shape[0]
    q, sp = _moment_modes(p, n)
    r, sq = _moment_modes(target, n)
    kept = _retained(sp, np.sqrt(rtol))
    target_kept = _retained(sq, np.sqrt(rtol))
    factor = r[:, target_kept] * sq[target_kept]
    if not factor.size:
        return 0.0
    if np.any(~kept):
        null_action = q[:, ~kept].T @ factor
        if np.linalg.norm(null_action, ord=2) / sq.max() > np.sqrt(rtol):
            return float("inf")
    if not np.any(kept):
        return float("inf")
    whitened = (q[:, kept].T @ factor) / sp[kept, None]
    return float(np.linalg.norm(whitened, ord=2) ** 2)


def signed_paths(edges, source: int, target: int) -> tuple[float, float]:
    """Return signed sum and sum of absolute products in a time-ordered DAG."""
    graph = _matrix(edges, "edges")
    n = len(graph)
    if graph.shape != (n, n) or np.any(np.triu(graph)):
        raise ValueError("edges[dst, src] must be strictly lower triangular")
    if not 0 <= source <= target < n:
        raise ValueError("node indices must lie in causal order")
    signed = np.zeros(n)
    absolute = np.zeros(n)
    signed[source] = absolute[source] = 1.0
    for dst in range(source + 1, target + 1):
        signed[dst] = graph[dst, source:dst] @ signed[source:dst]
        absolute[dst] = np.abs(graph[dst, source:dst]) @ absolute[source:dst]
    return float(signed[target]), float(absolute[target])


def subspace_angles(a, b, *, rtol: float = 1e-12) -> np.ndarray:
    """Angles between actual column supports, in decreasing order.

    Pad missing dimensions with right angles to expose rank loss.
    """
    rtol = _tolerance(rtol)
    aa, bb = _matrix(a, "a"), _matrix(b, "b")
    if aa.shape[0] != bb.shape[0]:
        raise ValueError("subspaces need a common output frame")
    ua, sa, _ = np.linalg.svd(aa, full_matrices=False)
    ub, sb, _ = np.linalg.svd(bb, full_matrices=False)
    ka, kb = _singular_rank(sa, rtol), _singular_rank(sb, rtol)
    overlap = np.linalg.svd(ua[:, :ka].T @ ub[:, :kb], compute_uv=False)
    angles = np.arccos(np.clip(overlap, 0, 1))
    return np.r_[np.repeat(np.pi / 2, abs(ka - kb)), angles[::-1]]
