"""Exact finite checks for the nonzero cube-root bound and its composition.

The oracle enumerates residue-class sums.  Reaching target sums (a, b, c)
requires exactly half their L1 distance from the current sums, since flips
within each residue class can realize every sum of the appropriate parity.
This covers every binary row at each tested length through its residue sums.
"""

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from had05 import direct, ldt, power3, sectors, spectral_lower


def scalar_power(triple):
    a, b, c = (int(value) for value in triple)
    return a * a + b * b + c * c - a * b - b * c - c * a


def residue_rows(m):
    class_size = m // 3
    triples = np.asarray(
        list(itertools.product(range(-class_size, class_size + 1, 2), repeat=3)),
        dtype=np.int64,
    )
    powers = np.asarray([scalar_power(t) for t in triples], dtype=np.int64)
    return triples, powers


def row_with_residue_sums(m, triple):
    row = np.empty(m, dtype=np.int64)
    class_size = m // 3
    for residue, total in enumerate(triple):
        positives = (class_size + int(total)) // 2
        row[residue::3] = [1] * positives + [-1] * (class_size - positives)
    return row


class SpectralBoundTests(unittest.TestCase):
    def test_spectral_bound_against_exact_residue_distance(self):
        # m=9 checks the zero-bound domain.  m=15 and 27 exercise positive
        # bounds, which the historical exhaustive m=9 test never reached.
        for m in (9, 15, 27):
            with self.subTest(m=m):
                triples, powers = residue_rows(m)
                feasible = triples[powers <= 4 * m]
                positive = 0
                for triple, power in zip(triples, powers):
                    exact = int((np.abs(feasible - triple).sum(axis=1) // 2).min())
                    bound = int(spectral_lower(np.asarray([power]), m)[0])
                    self.assertLessEqual(bound, exact, (m, triple.tolist(), power))
                    self.assertGreaterEqual(bound, 0)
                    positive += int(bound > 0)
                if m == 9:
                    self.assertEqual(positive, 0)
                else:
                    self.assertGreater(positive, 0)

    def test_power_matches_scalar_residue_formula(self):
        for m in (15, 27):
            triples, powers = residue_rows(m)
            for triple, expected in zip(triples, powers):
                row = row_with_residue_sums(m, triple)
                observed = power3(np.tile(row, (4, 1)))
                np.testing.assert_array_equal(observed, np.full(4, expected))

    def test_joint_bound_against_exact_row_and_spectral_relaxation(self):
        m = 15
        current = [(-5, 5, -5), (1, 1, -1), (1, 1, 1), (5, 5, 5)]
        x = np.asarray([row_with_residue_sums(m, t) for t in current])
        result = ldt(x, direct(x), x.sum(axis=1), horizon=16)

        # This oracle enumerates signed row-sum solutions directly, so it
        # also checks the magnitude/sign reduction used by sectors().
        signed = [
            sums
            for sums in itertools.product(range(-m, m + 1, 2), repeat=4)
            if sum(value * value for value in sums) == 4 * m
        ]
        row_distance = min(
            sum(abs(int(a) - b) // 2 for a, b in zip(x.sum(axis=1), sums))
            for sums in signed
        )
        triples, powers = residue_rows(m)
        feasible = triples[powers <= 4 * m]
        target_sums = sorted({s for sums in signed for s in sums})
        exact_row_cost = [
            {
                s: int(
                    (
                        np.abs(feasible[feasible.sum(axis=1) == s] - triple).sum(axis=1)
                        // 2
                    ).min()
                )
                for s in target_sums
            }
            for triple in current
        ]
        exact_relaxed_distance = min(
            sum(exact_row_cost[i][s] for i, s in enumerate(sums)) for sums in signed
        )
        self.assertEqual(row_distance, 5)
        self.assertEqual(exact_relaxed_distance, 7)
        self.assertEqual(result["row"], row_distance)
        self.assertEqual(result["spectral"], 2)
        self.assertEqual(result["joint"], 6)
        self.assertLessEqual(result["joint"], exact_relaxed_distance)
        self.assertGreater(result["joint"], max(result["row"], result["spectral"]))
        self.assertGreaterEqual(result["lower"], result["joint"])
        self.assertGreaterEqual(result["lower"], result["corr"])

    def test_sector_domain_rejects_even_or_too_short_lengths(self):
        for m in (0, 1, 2, 4, 14):
            with self.subTest(m=m), self.assertRaises(ValueError):
                sectors(m)


if __name__ == "__main__":
    unittest.main()
