"""Small deterministic instruments for delayed-memory curriculum discovery."""
from __future__ import annotations

from collections import Counter
from itertools import combinations, product

import numpy as np


def legal(memory, split):
    if split not in ('train', 'eval'):
        raise ValueError('invalid split')
    if len(memory) != 4 or any(type(v) is not int or not 0 <= v < 4 for v in memory):
        raise ValueError('invalid memory')
    return (sum(v * 4**i for i, v in enumerate(memory)) % 5 >= 2) == (split == 'train')


def semantic_edges(split='train'):
    """Return directed (receiver, hybrid, edited_slot) triples, canonical order."""
    if split not in ('train', 'eval'):
        raise ValueError('invalid split')
    edges = []
    for r in product(range(4), repeat=4):
        if not legal(r, split):
            continue
        for slot in range(4):
            for v in range(4):
                h = r[:slot] + (v,) + r[slot+1:]
                if v != r[slot] and legal(h, split):
                    edges.append((r, h, slot))
    return tuple(edges)


def graph_summary(split):
    vertices = {m for m in product(range(4), repeat=4) if legal(m, split)}
    adj = {m: set() for m in vertices}
    transitions = Counter()
    for r, h, s in semantic_edges(split):
        adj[r].add(h)
        transitions[s, r[s], h[s]] += 1
    pending = set(vertices)
    components = []
    while pending:
        queue = [min(pending)]
        pending.remove(queue[0])
        for r in queue:
            for h in sorted(adj[r] & pending):
                pending.remove(h)
                queue.append(h)
        components.append(len(queue))
    return {'vertices': len(vertices), 'directed_edges': sum(transitions.values()),
            'slot_edges': [sum(n for (s, _, _), n in transitions.items() if s == j)
                           for j in range(4)], 'components': sorted(components),
            'transition_types': len(transitions)}


def selective_direction(a, C, B, dose, rtol=1e-10):
    """Maximize a @ delta over span(B), C @ delta=0, ||delta||<=dose.

    Numerical rank is a declared tolerance, not an exact certificate.
    Returns a feasible up-to-rounding physical direction, not coefficients.
    """
    a, C, B = (np.asarray(x, dtype=np.float64) for x in (a, C, B))
    if (a.ndim != 1 or C.ndim != 2 or B.ndim != 2 or len(a) == 0
            or C.shape[1] != len(a) or B.shape[0] != len(a)
            or not all(np.isfinite(x).all() for x in (a, C, B))
            or not np.isfinite(dose) or dose < 0 or not 0 < rtol < 1):
        raise ValueError('invalid control geometry')
    q, s, _ = np.linalg.svd(B, full_matrices=False)
    rank = int(np.count_nonzero(s > rtol * s[0])) if len(s) and s[0] > 0 else 0
    q = q[:, :rank]
    D = C @ q
    _, s, vh = np.linalg.svd(D, full_matrices=True)
    cr = int(np.count_nonzero(s > rtol * s[0])) if len(s) and s[0] > 0 else 0
    Z = q @ vh[cr:].T
    v = Z @ (Z.T @ a)
    n = np.linalg.norm(v)
    delta = dose * v / n if n > rtol * np.linalg.norm(a) else np.zeros_like(a)
    return {'delta': delta, 'gain': float(a @ delta), 'support_rank': rank,
            'null_rank': rank-cr, 'dose': float(np.linalg.norm(delta)),
            'collateral': float(np.linalg.norm(C @ delta)),
            'feasible': bool(n > rtol * np.linalg.norm(a))}


def budgeted_selective_direction(a, C, B, dose, collateral_budget=0.0, rtol=1e-10):
    """SCCG-01 constrained selective-control frontier solver (float64).

    Maximizes ``a @ delta`` over ``delta in span(B)`` subject to:
        ``||delta||_2 <= dose`` and ``||C @ delta||_2 <= collateral_budget``.

    When ``collateral_budget == 0``, delegates identically to ``selective_direction``.
    When ``collateral_budget > 0``, solves the exact convex QCQP in the SVD frame
    of ``D = C @ Q`` via monotonic scalar bisection on the Lagrange multiplier ratio.
    Retains infeasible / zero-rank cases explicitly in the returned dict.
    """
    if not np.isfinite(collateral_budget) or collateral_budget < 0:
        raise ValueError('collateral_budget must be finite and nonnegative')
    if collateral_budget == 0.0:
        res = selective_direction(a, C, B, dose, rtol=rtol)
        res['collateral_budget'] = 0.0
        res['budget_active'] = res['null_rank'] < res['support_rank']
        return res

    a, C, B = (np.asarray(x, dtype=np.float64) for x in (a, C, B))
    if (a.ndim != 1 or C.ndim != 2 or B.ndim != 2 or len(a) == 0
            or C.shape[1] != len(a) or B.shape[0] != len(a)
            or not all(np.isfinite(x).all() for x in (a, C, B))
            or not np.isfinite(dose) or dose < 0 or not 0 < rtol < 1):
        raise ValueError('invalid control geometry')

    q, sb, _ = np.linalg.svd(B, full_matrices=False)
    rank = int(np.count_nonzero(sb > rtol * sb[0])) if len(sb) and sb[0] > 0 else 0
    if rank == 0 or dose == 0.0:
        return {'delta': np.zeros_like(a), 'gain': 0.0, 'support_rank': rank,
                'null_rank': 0, 'dose': 0.0, 'collateral': 0.0,
                'collateral_budget': float(collateral_budget),
                'budget_active': False, 'feasible': False}
    q = q[:, :rank]
    g = q.T @ a
    g_norm = float(np.linalg.norm(g))
    if g_norm <= rtol * max(float(np.linalg.norm(a)), 1e-30):
        return {'delta': np.zeros_like(a), 'gain': 0.0, 'support_rank': rank,
                'null_rank': rank, 'dose': 0.0, 'collateral': 0.0,
                'collateral_budget': float(collateral_budget),
                'budget_active': False, 'feasible': False}

    D = C @ q
    _, sd, vh = np.linalg.svd(D, full_matrices=True)
    # Pad singular values to length `rank` with zeros for null directions
    s_full = np.zeros(rank, dtype=np.float64)
    s_full[:len(sd)] = sd
    cr = int(np.count_nonzero(s_full > rtol * s_full[0])) if s_full[0] > 0 else 0
    null_rank = rank - cr

    # Coordinates in V_d basis: u = vh.T @ y, ||u|| = ||y||, ||D u||^2 = sum s_full^2 y^2
    h = vh @ g
    # Case 1: Unconstrained dose-sphere optimum is already within collateral budget
    y_unc = dose * h / np.linalg.norm(h)
    col_unc = float(np.linalg.norm(s_full * y_unc))
    if col_unc <= collateral_budget * (1.0 + 1e-12):
        delta = q @ (vh.T @ y_unc)
        return {'delta': delta, 'gain': float(a @ delta), 'support_rank': rank,
                'null_rank': null_rank, 'dose': float(np.linalg.norm(delta)),
                'collateral': float(np.linalg.norm(C @ delta)),
                'collateral_budget': float(collateral_budget),
                'budget_active': False, 'feasible': True}

    # Case 2: Both constraints active or pure collateral-ellipsoid optimum (dose inactive)
    # Pure collateral-ellipsoid optimum when ker(D) == 0 (i.e. cr == rank):
    # max h^T y s.t. sum s_i^2 y_i^2 <= B^2  =>  y_i = B * (h_i / s_i^2) / sqrt(sum h_j^2 / s_j^2)
    s2 = s_full * s_full
    if cr == rank:
        inv_s2_h = h / s2
        denom = float(np.sqrt(np.sum((h * h) / s2)))
        if denom > 0:
            y_ell = collateral_budget * inv_s2_h / denom
            if float(np.linalg.norm(y_ell)) <= dose * (1.0 + 1e-12):
                delta = q @ (vh.T @ y_ell)
                return {'delta': delta, 'gain': float(a @ delta), 'support_rank': rank,
                        'null_rank': null_rank, 'dose': float(np.linalg.norm(delta)),
                        'collateral': float(np.linalg.norm(C @ delta)),
                        'collateral_budget': float(collateral_budget),
                        'budget_active': True, 'feasible': True}

    # Otherwise, both ||y|| = dose and ||s * y|| = collateral_budget are active.
    # Parameterize y(t) = dose * (h / (1 + t * s^2)) / ||h / (1 + t * s^2)||, t >= 0.
    # ||s * y(t)|| is strictly decreasing in t from col_unc (t=0) toward col_inf (t -> inf).
    def eval_t(t: float) -> tuple[np.ndarray, float]:
        w = h / (1.0 + t * s2)
        wn = float(np.linalg.norm(w))
        if wn <= 0.0:
            return np.zeros_like(h), 0.0
        y = dose * w / wn
        return y, float(np.linalg.norm(s_full * y))

    lo, hi = 0.0, 1.0 / max(float(s2[0]), 1e-30)
    _, col_hi = eval_t(hi)
    for _ in range(100):
        if col_hi <= collateral_budget:
            break
        hi *= 4.0
        _, col_hi = eval_t(hi)
        if hi > 1e30:
            break

    if col_hi > collateral_budget:
        # Fallback: scale down along minimal-collateral direction
        y, col = eval_t(hi)
        if col > 0:
            y = y * min(1.0, collateral_budget / col)
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            _, col_mid = eval_t(mid)
            if col_mid > collateral_budget:
                lo = mid
            else:
                hi = mid
        y, _ = eval_t(hi)

    delta = q @ (vh.T @ y)
    return {'delta': delta, 'gain': float(a @ delta), 'support_rank': rank,
            'null_rank': null_rank, 'dose': float(np.linalg.norm(delta)),
            'collateral': float(np.linalg.norm(C @ delta)),
            'collateral_budget': float(collateral_budget),
            'budget_active': True, 'feasible': bool(np.linalg.norm(delta) > 0)}


def sccg_frontier(a, C, B, dose, budgets=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0), rtol=1e-10):
    """Evaluate SCCG-01 constrained selective-control frontier across collateral budgets."""
    return [budgeted_selective_direction(a, C, B, dose, collateral_budget=float(b), rtol=rtol)
            for b in budgets]


def joint_support_search(a, C, B, dose, rank=4, beam=8, collateral_budget=0.0):
    """Joint single/pair expansion; deterministic beam, no global optimality claim."""
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2 or not 1 <= rank <= B.shape[1] or beam < 1:
        raise ValueError('invalid search budget')
    cache = {}
    layers = {0: [()]}
    for size in range(1, rank + 1):
        candidates = set()
        for step in (1, 2):
            for old in layers.get(size - step, []):
                for new in combinations((i for i in range(B.shape[1]) if i not in old), step):
                    candidates.add(tuple(sorted(old + new)))
        for support in sorted(candidates):
            if collateral_budget == 0.0:
                cache[support] = selective_direction(a, C, B[:, support], dose)
            else:
                cache[support] = budgeted_selective_direction(
                    a, C, B[:, support], dose, collateral_budget=collateral_budget
                )
        layers[size] = sorted(candidates, key=lambda k: (-cache[k]['gain'], k))[:beam]
    winner = min(cache, key=lambda k: (-cache[k]['gain'], len(k), k))
    return winner, cache[winner], len(cache)


def signed_observer_matched_sham(observer_matrix, target_delta, dose, *,
                                 seed=0, rtol=1e-10):
    """MOLT-06-style sham matching both physical dose and signed linear observer response.

    Finds a random direction ``delta_sham`` with ``observer_matrix @ delta_sham ==
    observer_matrix @ target_delta`` and ``||delta_sham||_2 == dose`` by drawing
    a random vector in ``ker(observer_matrix)`` when feasible. Logs infeasibility
    explicitly if ``ker(observer_matrix)`` is trivial.
    """
    J = np.asarray(observer_matrix, dtype=np.float64)
    d = np.asarray(target_delta, dtype=np.float64)
    if J.ndim != 2 or d.ndim != 1 or J.shape[1] != len(d):
        raise ValueError('invalid observer/direction shapes')
    y = J @ d
    _, s, vh = np.linalg.svd(J, full_matrices=True)
    rank = int(np.count_nonzero(s > rtol * s[0])) if len(s) and s[0] > 0 else 0
    null_dim = J.shape[1] - rank
    # Minimum-norm solution to J @ d_min = y
    d_min, _, _, _ = np.linalg.lstsq(J, y, rcond=rtol)
    min_norm = float(np.linalg.norm(d_min))
    rem_sq = dose * dose - min_norm * min_norm
    if null_dim <= 0 or rem_sq <= 1e-14:
        return {
            'feasible': False,
            'null_dim': null_dim,
            'min_norm': min_norm,
            'delta': np.zeros_like(d),
            'observer_error': float('nan'),
            'dose': 0.0,
        }
    Z = vh[rank:].T  # [dim, null_dim]
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(null_dim)
    # Avoid aligning with original null component if null_dim >= 2
    orig_null = Z.T @ (d - d_min)
    if null_dim >= 2 and np.linalg.norm(orig_null) > 1e-12:
        u_orig = orig_null / np.linalg.norm(orig_null)
        z = z - float(z @ u_orig) * u_orig
    zn = float(np.linalg.norm(z))
    if zn <= 1e-12:
        return {'feasible': False, 'null_dim': null_dim, 'min_norm': min_norm,
                'delta': np.zeros_like(d), 'observer_error': float('nan'), 'dose': 0.0}
    sham = d_min + np.sqrt(rem_sq) * (Z @ (z / zn))
    return {
        'feasible': True,
        'null_dim': null_dim,
        'min_norm': min_norm,
        'delta': sham,
        'observer_error': float(np.linalg.norm(J @ sham - y)),
        'dose': float(np.linalg.norm(sham)),
        'cosine_to_target': float((sham @ d) / max(np.linalg.norm(sham) * np.linalg.norm(d), 1e-30)),
    }


def predicted_gain(gradient, update, curvature=None):
    """First-order loss improvement; optional conditional Taylor lower bound."""
    g, d = (np.asarray(x, dtype=np.float64) for x in (gradient, update))
    if g.shape != d.shape or g.ndim != 1 or not np.isfinite(g).all() or not np.isfinite(d).all():
        raise ValueError('invalid update geometry')
    gain = -float(g @ d)
    if curvature is None:
        return gain, None
    if not np.isfinite(curvature) or curvature < 0:
        raise ValueError('invalid curvature')
    return gain, gain - float(curvature) * float(d @ d) / 2
