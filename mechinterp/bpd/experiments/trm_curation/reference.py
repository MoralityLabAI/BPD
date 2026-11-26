"""Small deterministic instruments for delayed-memory curriculum discovery."""
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
    rank = int(np.count_nonzero(s > rtol * s[0])) if len(s) else 0
    q = q[:, :rank]
    D = C @ q
    _, s, vh = np.linalg.svd(D, full_matrices=True)
    cr = int(np.count_nonzero(s > rtol * s[0])) if len(s) else 0
    Z = q @ vh[cr:].T
    v = Z @ (Z.T @ a)
    n = np.linalg.norm(v)
    delta = dose * v / n if n > rtol * np.linalg.norm(a) else np.zeros_like(a)
    return {'delta': delta, 'gain': float(a @ delta), 'support_rank': rank,
            'null_rank': rank-cr, 'dose': float(np.linalg.norm(delta)),
            'collateral': float(np.linalg.norm(C @ delta))}


def joint_support_search(a, C, B, dose, rank=4, beam=8):
    """Joint single/pair expansion; deterministic beam, no global optimality claim."""
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2 or not 1 <= rank <= B.shape[1] or beam < 1:
        raise ValueError('invalid search budget')
    cache = {}
    layers = {0: [()]}
    for size in range(1, rank+1):
        candidates = set()
        for step in (1, 2):
            for old in layers.get(size-step, []):
                for new in combinations((i for i in range(B.shape[1]) if i not in old), step):
                    candidates.add(tuple(sorted(old+new)))
        for support in sorted(candidates):
            cache[support] = selective_direction(a, C, B[:, support], dose)
        layers[size] = sorted(candidates, key=lambda k: (-cache[k]['gain'], k))[:beam]
    winner = min(cache, key=lambda k: (-cache[k]['gain'], len(k), k))
    return winner, cache[winner], len(cache)


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
