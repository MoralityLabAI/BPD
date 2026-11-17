"""Exact-rational finite vector-matroid greedy audit for BPD hyperplane weights.

Complements FiniteFiltration.lean's general conditional profile theorem.
Independent of neural training; runs exact Fraction Gaussian elimination.
"""
from __future__ import annotations

from fractions import Fraction as Q
from itertools import combinations
from random import Random


def rank(rows):
    """Exact row rank over Q by forward Gaussian elimination."""
    if not rows:
        return 0
    a = [[Q(x) for x in row] for row in rows]
    n, d = len(a), len(a[0])
    pivots = 0
    for col in range(d):
        pivot = next((i for i in range(pivots, n) if a[i][col]), None)
        if pivot is None:
            continue
        a[pivots], a[pivot] = a[pivot], a[pivots]
        scale = a[pivots][col]
        a[pivots] = [x / scale for x in a[pivots]]
        for i in range(pivots + 1, n):
            q = a[i][col]
            if q:
                a[i] = [x - q * y for x, y in zip(a[i], a[pivots])]
        pivots += 1
        if pivots == n:
            break
    return pivots


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def basis_certification(directions, atoms):
    d = len(directions[0])
    assert len(atoms) and all(len(v) == d and any(v) for v in directions)
    assert all(len(a) == d for a in atoms)
    masses = tuple(Q(sum(dot(m, a) == 0 for a in atoms), len(atoms))
                   for m in directions)
    bases = tuple(c for c in combinations(range(len(directions)), d)
                  if rank([directions[i] for i in c]) == d)
    assert bases
    scores = {b: sum((masses[i] for i in b), Q(0)) for b in bases}
    top = max(scores.values())
    optimum = {b for b, score in scores.items() if score == top}
    thresholds = sorted({w for w in masses if w > 0}, reverse=True)
    level_rank = {
        t: rank([v for v, w in zip(directions, masses) if w >= t])
        for t in thresholds
    }
    for basis in bases:
        matches = all(
            rank([directions[i] for i in basis if masses[i] >= t]) == level_rank[t]
            for t in thresholds
        )
        assert matches == (basis in optimum), (basis, top, masses, level_rank)
    greedy = []
    for i in sorted(range(len(directions)), key=lambda k: (-masses[k], k)):
        if rank([directions[j] for j in (*greedy, i)]) > len(greedy):
            greedy.append(i)
        if len(greedy) == d:
            break
    assert len(greedy) == d
    assert sum((masses[i] for i in greedy), Q(0)) == top
    return {
        "bases": len(bases), "optima": len(optimum), "weight": str(top),
        "levels": len(thresholds), "greedy": greedy,
        "max_rank_by_level": {str(t): level_rank[t] for t in thresholds},
    }


def run_audit():
    result = {}
    for d in (2, 3, 4):
        rows = []
        for seed in range(2026111700, 2026111732):
            g = Random(17 + 1000 * d + seed)
            directions = [tuple(int(i == j) for j in range(d)) for i in range(d)]
            while len(directions) < d + 3:
                v = tuple(g.choice((-2, -1, 0, 1, 2)) for _ in range(d))
                if any(v):
                    directions.append(v)
            atoms = [tuple(g.choice((-1, 0, 1)) for _ in range(d)) for _ in range(7)]
            rows.append(basis_certification(directions, atoms))
        result[str(d)] = {
            "cases": len(rows),
            "total_bases": sum(r["bases"] for r in rows),
            "tied_optima_cases": sum(r["optima"] > 1 for r in rows),
            "total_optima": sum(r["optima"] for r in rows),
            "max_levels": max(r["levels"] for r in rows),
        }
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(run_audit(), indent=2, sort_keys=True))
