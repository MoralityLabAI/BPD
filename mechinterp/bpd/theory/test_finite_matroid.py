"""Unit tests for rational greedy vector-matroid rank-filtration audit."""
from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'bpd_finite_matroid', Path(__file__).with_name('finite_matroid_audit.py'))
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)
basis_certification, rank, run_audit = mod.basis_certification, mod.rank, mod.run_audit


class TestFiniteMatroid(unittest.TestCase):
    def test_exact_fraction_rank(self):
        self.assertEqual(rank([(1, 2, 3), (2, 4, 6), (0, 1, 1)]), 2)
        self.assertEqual(rank([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), 3)

    def test_rotated_two_atom_optimum(self):
        c = basis_certification(
            [(1, 0), (0, 1), (1, 1), (1, -1), (1, 2)],
            [(1, 1), (1, -1)],
        )
        self.assertEqual(c["weight"], "1")
        self.assertEqual(c["optima"], 1)
        self.assertEqual(set(c["greedy"]), {2, 3})

    def test_cancellation_three_way_tie(self):
        c = basis_certification(
            [(1, 0), (0, 1), (1, 1), (1, -1), (1, 2)],
            [(1, 0), (0, 1), (1, 1)],
        )
        self.assertEqual(c["weight"], "2/3")
        self.assertEqual(c["optima"], 3)

    def test_96_cases_across_dimensions(self):
        result = run_audit()
        self.assertEqual(
            result, {
                "2": {"cases": 32, "total_bases": 280,
                      "tied_optima_cases": 27, "total_optima": 97, "max_levels": 4},
                "3": {"cases": 32, "total_bases": 527,
                      "tied_optima_cases": 20, "total_optima": 78, "max_levels": 4},
                "4": {"cases": 32, "total_bases": 894,
                      "tied_optima_cases": 20, "total_optima": 75, "max_levels": 5},
            },
        )


if __name__ == "__main__":
    unittest.main()
