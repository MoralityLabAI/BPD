"""Source-only exact-rational tests for idealized BPD basis minimizers."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bpd_finite_optima", HERE / "finite_optima.py")
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


class TestFiniteOptima(unittest.TestCase):
    def test_four_exact_models(self):
        r = mod.audit()
        self.assertEqual(set(r), {"rotated_unique", "atomic_cancellation", "nested", "equal_all_zero"})
        self.assertEqual(r["rotated_unique"]["optimal_pairs"], [[2, 3]])
        self.assertEqual(r["atomic_cancellation"]["optimal_pairs"], [[0, 1], [0, 3], [1, 3]])
        self.assertEqual(r["atomic_cancellation"]["weights"], ["1/3", "1/3", "0", "1/3", "0"])

    def test_rotated_reference_not_optimal(self):
        atoms = mod.fixtures()["rotated_unique"]
        masses, top, pairs, scores = mod.optimal_pairs(atoms)
        self.assertEqual(masses[0] + masses[1], 0)
        self.assertGreater(top, scores[(0, 1)])
        self.assertEqual(mod.relative_remix_blocks(pairs), ((0,), (1,)))

    def test_held_reference_monomial_gauge(self):
        for name, atoms in mod.fixtures().items():
            _, _, pairs, _ = mod.optimal_pairs(atoms)
            anchor = pairs[0]
            self.assertTrue(mod.monomial(mod.relative(anchor, anchor)), name)

    def test_atomic_cancellation_not_free_remix(self):
        atoms = mod.fixtures()["atomic_cancellation"]
        masses, top, pairs, scores = mod.optimal_pairs(atoms)
        self.assertEqual(masses[2], 0)
        self.assertEqual(masses[4], 0)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(len(scores), 10)
        self.assertEqual(top, mod.Q(2, 3))


if __name__ == "__main__":
    unittest.main()
