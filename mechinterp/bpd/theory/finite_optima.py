"""Exact-rational BPD idealized expected-L0 optimal-set fixtures.

Finite five-direction audit. Does not prove that a neural BPD learns these
minimizers; all-real-direction results are proved separately in the note.
"""
from __future__ import annotations

from fractions import Fraction as Q
from itertools import combinations
from typing import Sequence

Vec = tuple[int, int]
READS: tuple[Vec, ...] = ((1, 0), (0, 1), (1, 1), (1, -1), (1, 2))


def dot(x: Vec, y: Vec) -> int:
    return x[0] * y[0] + x[1] * y[1]


def determinant(x: Vec, y: Vec) -> int:
    return x[0] * y[1] - x[1] * y[0]


def hyperplane_mass(read: Vec, atoms: Sequence[tuple[Vec, Q]]) -> Q:
    assert read != (0, 0)
    assert sum((p for _, p in atoms), Q(0)) == 1
    return sum((p for a, p in atoms if dot(read, a) == 0), Q(0))


def optimal_pairs(atoms: Sequence[tuple[Vec, Q]]):
    masses = tuple(hyperplane_mass(read, atoms) for read in READS)
    bases = [(i, j) for i, j in combinations(range(len(READS)), 2)
             if determinant(READS[i], READS[j])]
    scores = {b: masses[b[0]] + masses[b[1]] for b in bases}
    top = max(scores.values())
    return masses, top, tuple(b for b in bases if scores[b] == top), scores


def mat_inverse(a: Vec, b: Vec) -> tuple[tuple[Q, Q], tuple[Q, Q]]:
    det = determinant(a, b)
    if not det:
        raise ValueError("singular candidate")
    return ((Q(b[1], det), -Q(a[1], det)),
            (-Q(b[0], det), Q(a[0], det)))


def relative(anchor: tuple[int, int], candidate: tuple[int, int]):
    inv = mat_inverse(READS[anchor[0]], READS[anchor[1]])
    return tuple(tuple(sum((Q(READS[i][k]) * inv[k][j]
                            for k in range(2)), Q(0))
                       for j in range(2)) for i in candidate)


def monomial(m) -> bool:
    """Allows row permutations and nonzero row rescaling as mere relabeling."""
    return (all(sum(x != 0 for x in row) == 1 for row in m)
            and all(sum(m[i][j] != 0 for i in range(2)) == 1 for j in range(2)))


def relative_remix_blocks(top_pairs: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], ...]:
    """Exact two-coordinate block audit; higher-dimensional theory uses centralizers."""
    anchor = top_pairs[0]
    if all(monomial(relative(anchor, other)) for other in top_pairs):
        return ((0,), (1,))
    return ((0, 1),)


def masked_weight(pair: tuple[int, int], mask: tuple[int, int]):
    """Exact represented edit M^{-1} diag(mask) M with W=identity."""
    a, b = READS[pair[0]], READS[pair[1]]
    inverse = mat_inverse(a, b)
    masked_rows = ((Q(mask[0]) * a[0], Q(mask[0]) * a[1]),
                   (Q(mask[1]) * b[0], Q(mask[1]) * b[1]))
    return tuple(tuple(sum((inverse[i][k] * masked_rows[k][j]
                            for k in range(2)), Q(0))
                       for j in range(2)) for i in range(2))


def unstable_singleton_masks(optimal_bases: tuple[tuple[int, int], ...]):
    """Compare represented edits across all listed optimal dictionaries."""
    anchor = optimal_bases[0]
    return tuple(mask for mask in ((1, 0), (0, 1))
                 if any(masked_weight(pair, mask) != masked_weight(anchor, mask)
                        for pair in optimal_bases))


def threshold_certificate(atoms: Sequence[tuple[Vec, Q]], basis: tuple[int, int]):
    """Verify the finite maximum-weight basis filtration level by level."""
    masses = tuple(hyperplane_mass(read, atoms) for read in READS)
    rank_of = lambda items: min(2, max((1 if items else 0),
                                    2 if any(determinant(READS[i], READS[j])
                                             for i, j in combinations(items, 2)) else 0))
    levels = sorted({m for m in masses if m > 0}, reverse=True)
    return all(rank_of([i for i in basis if masses[i] >= t])
               == rank_of([i for i, m in enumerate(masses) if m >= t])
               for t in levels)


def fixtures():
    return {
        "rotated_unique": (((1, 1), Q(1, 2)), ((1, -1), Q(1, 2))),
        "atomic_cancellation": (((1, 0), Q(1, 3)), ((0, 1), Q(1, 3)),
                                ((1, 1), Q(1, 3))),
        "nested": (((1, 0), Q(1, 2)), ((0, 0), Q(1, 2))),
        "equal_all_zero": (((0, 0), Q(1)),),
    }


def audit():
    expected = {
        "rotated_unique": {"optimum": Q(1), "pairs": ((2, 3),), "blocks": ((0,), (1,))},
        "atomic_cancellation": {"optimum": Q(2, 3),
                                "pairs": ((0, 1), (0, 3), (1, 3)), "blocks": ((0, 1),)},
        "nested": {"optimum": Q(3, 2),
                   "pairs": ((0, 1), (1, 2), (1, 3), (1, 4)), "blocks": ((0, 1),)},
        "equal_all_zero": {"optimum": Q(2),
                           "pairs": tuple(combinations(range(5), 2)), "blocks": ((0, 1),)},
    }
    results = {}
    for name, atoms in fixtures().items():
        masses, top, pairs, scores = optimal_pairs(atoms)
        exp = expected[name]
        assert top == exp["optimum"], (name, top)
        assert pairs == exp["pairs"], (name, pairs)
        assert relative_remix_blocks(pairs) == exp["blocks"], name
        for b in scores:
            assert threshold_certificate(atoms, b) == (b in pairs), (name, b)
        assert all(masked_weight(p, (1, 1)) == ((1, 0), (0, 1))
                   for p in pairs), name
        assert all(masked_weight(p, (0, 0)) == ((0, 0), (0, 0))
                   for p in pairs), name
        expected_unstable = () if name == "rotated_unique" else ((1, 0), (0, 1))
        assert unstable_singleton_masks(pairs) == expected_unstable, name
        results[name] = {
            "weights": [str(x) for x in masses],
            "optimal_mass_total": str(top),
            "optimal_expected_L0": str(Q(2) - top),
            "optimal_pairs": [list(pair) for pair in pairs],
            "invariant_masks_blocks": [list(block) for block in exp["blocks"]],
            "checked_bases": len(scores),
            "unstable_singleton_masks": [list(v) for v in unstable_singleton_masks(pairs)],
        }
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(audit(), sort_keys=True, indent=2))
