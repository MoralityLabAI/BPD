"""SCCG-01 numerical reference; float diagnostics, not rigorous intervals."""
from __future__ import annotations

import numpy as np


def array(x, ndim, name):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != ndim or not np.isfinite(x).all():
        raise ValueError(f"invalid {name}")
    return x


def scalar(x, name, positive=False):
    x = float(x)
    if not np.isfinite(x) or x < 0 or (positive and x == 0):
        raise ValueError(f"invalid {name}")
    return x


def rank_svd(C, rcond):
    _, s, vh = np.linalg.svd(C, full_matrices=True)
    rank = int(np.sum(s > rcond * s[0])) if s.size else 0
    return s, vh, rank


def frontier(a, C, eps, eta, *, rcond=1e-12, iterations=90):
    """Max a@u, ||u||<=eps, ||C@u||<=eta, Euclidean whitened dose.

    Positive budgets: numerical primal/dual bracket from normalized
    quadratic constraints. Zero collateral: numerical SVD nullspace
    (rank cutoff explicit; residual on actual C reported).
    """
    a, C = array(a, 1, "a"), array(C, 2, "C")
    eps, eta = scalar(eps, "eps"), scalar(eta, "eta")
    rcond = scalar(rcond, "rcond")
    if not a.size or C.shape[1] != a.size or rcond >= 1:
        raise ValueError("shape or rank cutoff")
    if type(iterations) is not int or not 1 <= iterations <= 200:
        raise ValueError("iterations must be an integer in [1,200]")
    s, vh, rank = rank_svd(C, rcond)
    best = np.zeros_like(a)
    upper = 0.0
    alpha = None
    status = "NUMERICAL_POSITIVE_BUDGET"
    if eps == 0 or not np.any(a):
        status = "ZERO_DOSE_OR_OBJECTIVE"
    elif eta == 0:
        # Computed rank, not a certified exact nullspace.
        p = vh[rank:].T @ (vh[rank:] @ a)
        gain = np.linalg.norm(p)
        if gain:
            best = eps * p / gain
        upper = eps * gain
        status = "NUMERICAL_NULLSPACE"
    else:
        # In SVD coordinates Q(alpha) is diagonal; retain ALL singular
        # values in positive-budget arithmetic, irrespective of rank label.
        z = vh @ a
        K = np.zeros_like(a)
        K[:s.size] = (s / eta) ** 2
        E = 1 / eps**2
        if not np.isfinite(K).all() or not np.isfinite(E) or E <= 0:
            raise ValueError("rescale ill-conditioned budgets")
        upper = np.inf

        def evaluate(t):
            nonlocal best, upper, alpha
            diag = t * E + (1-t) * K
            if np.any(diag <= 0):
                raise ValueError("nonpositive mixed metric")
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                w = z / diag
                value = float(z @ w)
                if value <= 0 or not np.isfinite(value):
                    raise ValueError("rescale ill-conditioned objective")
                bound = np.sqrt(value)
                raw = vh.T @ (w / bound)
                norm = np.linalg.norm(raw)
                collateral = np.linalg.norm(C @ raw)
                if not np.isfinite(norm + collateral):
                    raise ValueError("nonfinite primal candidate")
                scale = min(1., eps/norm if norm else 1.,
                            eta/collateral if collateral else 1.)
                candidate = raw * scale
                if a @ candidate > a @ best:
                    best = candidate
                if bound < upper:
                    upper, alpha = bound, t
                deriv = -float(np.sum(w*w*(E-K)))
            if not np.isfinite(deriv):
                raise ValueError("nonfinite dual derivative")
            return deriv

        # Convex squared upper bound; derivative >= 0 to right of minimizer.
        if evaluate(1.) > 0:
            lo, hi = 0., 1.
            for _ in range(iterations):
                mid = (lo + hi)/2
                if not lo < mid < hi:
                    break
                if evaluate(mid) > 0:
                    hi = mid
                else:
                    lo = mid
            if hi > 0:
                evaluate(hi)
    lower = float(a @ best)
    dose = float(np.linalg.norm(best))
    collateral = float(np.linalg.norm(C @ best))
    if not np.isfinite([lower, upper, dose, collateral]).all():
        raise ValueError("nonfinite result")
    return dict(u=best, lower=lower, upper=float(upper),
                gap=float(upper-lower), dose=dose, collateral=collateral,
                rank=rank, rcond=rcond, alpha=alpha, status=status)


def transport(Js, Bs):
    """J[t] precedes B[t] injection into h[t+1]; chronological suffixes."""
    if len(Js) != len(Bs) or not Js:
        raise ValueError("nonempty equal-length sequences required")
    js = [array(j, 2, "J") for j in Js]
    bs = [array(b, 2, "B") for b in Bs]
    d = js[0].shape[0]
    if not d or any(j.shape != (d,d) for j in js) or any(
            b.shape[0] != d for b in bs):
        raise ValueError("incompatible state widths")
    suffix = np.eye(d)
    blocks = []
    for j,b in reversed(list(zip(js,bs))):
        blocks.append(suffix @ b)
        suffix = suffix @ j
    result = np.concatenate(blocks[::-1], axis=1)
    return array(result, 2, "H")


def reader_boundary_problem(flow, components, targets, protected, *,
                            rcond=1e-12):
    """Adapt INK-BPD-06 reader-local flow; unit L2 actual boundary dose.

    Use only named component writes (exclude residual/bias/remainder).
    Restrict to their physical span, quotient redundant coefficients.
    Target/protected rows are gap gradients, not full task responses.
    Accepts serialized source fixture or detached CPU NumPy-compatible arrays.
    """
    writes = array(flow["physical_writes"], 2, "physical_writes")
    gradients = array(flow["reader"]["gap_gradient_block3"], 2, "gradient")
    labels = list(flow["term_labels"])
    readers = list(flow["reader_ids"])
    if len(labels) != writes.shape[0] or len(readers) != gradients.shape[0]:
        raise ValueError("metadata dimensions")
    if writes.shape[1] != gradients.shape[1]:
        raise ValueError("width mismatch")
    if len(set(labels)) != len(labels) or len(set(readers)) != len(readers):
        raise ValueError("duplicate metadata")
    rcond = scalar(rcond, "rcond")
    if rcond >= 1:
        raise ValueError("invalid rank cutoff")
    groups = [list(components), list(targets), list(protected)]
    if not groups[0] or not groups[1] or any(len(set(g)) != len(g) for g in groups):
        raise ValueError("empty or duplicate component/target choices")
    if set(groups[1]) & set(groups[2]):
        raise ValueError("overlapping reader sets")
    if any(c not in labels or not c.startswith("component:") for c in groups[0]):
        raise ValueError("only named component writes are admissible")
    if any(r not in readers for r in groups[1]+groups[2]):
        raise ValueError("unknown reader")
    B = writes[[labels.index(c) for c in groups[0]]].T
    U, s, vh = np.linalg.svd(B, full_matrices=False)
    rank = int(np.sum(s > rcond*s[0])) if s.size else 0
    if rank == 0:
        raise ValueError("zero physical component span")
    Q = U[:,:rank]
    coeff = vh[:rank].T / s[:rank]
    A = gradients[[readers.index(r) for r in groups[1]]] @ Q
    C = gradients[[readers.index(r) for r in groups[2]]] @ Q
    return dict(A=A,C=C,physical_basis=Q,coefficient_map=coeff,
                rank=rank,rcond=rcond,
                component_labels=groups[0],target_readers=groups[1],
                protected_readers=groups[2],
                scope="LOCAL_BOUNDARY_GAP_RESPONSE")


def protected_correction(g, C, v, L, domain_radius, *, rcond=1e-12, steps=32):
    """Numerical fixed-point replay under the theorem's externally supplied
    Lipschitz bound. Does not certify smoothness or exact floating equality."""
    C, v = array(C,2,"C"), array(v,1,"v")
    L, D = scalar(L,"L"),scalar(domain_radius,"domain_radius",positive=True)
    rcond = scalar(rcond,"rcond")
    if C.shape[1] != v.size or C.shape[0] == 0 or rcond >= 1:
        raise ValueError("shape or cutoff")
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError("invalid iteration count")
    s, _, rank = rank_svd(C,rcond)
    if rank != C.shape[0]:
        raise ValueError("protection derivative must have full row rank")
    R = np.linalg.pinv(C,rcond=rcond)
    r = float(np.linalg.norm(v))
    kappa = 1/float(s[rank-1])
    b = kappa*L*r*r
    if kappa*L*r > .25 or r+b > D:
        raise ValueError("theorem radius condition fails")
    def evaluate_g(x):
        val = array(g(x),1,"g")
        if val.shape != (C.shape[0],):
            raise ValueError("protection output shape")
        return val
    if np.linalg.norm(evaluate_g(np.zeros_like(v))) > 1e-12 or np.linalg.norm(C@v) > 1e-12:
        raise ValueError("origin or tangent fails numeric check")
    w = np.zeros_like(v)
    for _ in range(steps):
        x = v+w
        w = -R @ (evaluate_g(x)-C@x)
        if np.linalg.norm(w) > b + 1e-12:
            raise ValueError("iterate violates supplied correction bound")
    x = v+w
    return dict(u=x, correction=w, correction_bound=b, dose_bound=np.hypot(r,b),
                dose=float(np.linalg.norm(x)),
                protected_residual=float(np.linalg.norm(evaluate_g(x))),
                contraction_bound=kappa*L*(r+b),
                status="CONDITIONAL_NUMERICAL_REPLAY")
