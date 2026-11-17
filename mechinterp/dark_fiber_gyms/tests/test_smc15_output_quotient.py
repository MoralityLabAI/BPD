"""BPD-OUT01 independent NumPy CPU linear algebra checks."""
from __future__ import annotations

import unittest
import numpy as np

from mechinterp.dark_fiber_gyms.smc15_output_quotient import (
    two_atom_observable_null_certificate,
)


class TestOutputQuotient(unittest.TestCase):
    def test_orthogonal_physical_acts_as_one_observable(self):
        U = np.eye(3, dtype=np.float64)[:, :2]
        L = np.array([[1.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        rec = two_atom_observable_null_certificate(U, L)
        self.assertAlmostEqual(rec["gamma_input_sigma_min"], 1.0, places=12)
        self.assertLess(rec["epsilon_normalised_output_sigma_min"], 1e-12)
        self.assertLess(rec["actual_output_norm"], 1e-12)
        w = np.asarray(rec["physical_unit_nearnull"])
        self.assertAlmostEqual(np.linalg.norm(w), 1.0, places=12)

    def test_near_parallel_output_distinct_physical_features(self):
        U = np.eye(3, dtype=np.float64)[:, :2]
        L = np.array([[1, 1, 0], [0, 0.001, 0]], dtype=float)
        rec = two_atom_observable_null_certificate(U, L)
        self.assertGreater(rec["gamma_input_sigma_min"], 0.99)
        self.assertLess(rec["epsilon_normalised_output_sigma_min"], 0.001)
        self.assertGreater(rec["actual_output_norm"], 0.0)
        self.assertLessEqual(rec["actual_output_norm"],
                             rec["certified_output_upper_bound"] + 1e-12)

    def test_random_full_rank_bounds(self):
        rng = np.random.default_rng(20261517)
        for _ in range(64):
            U = rng.standard_normal((48, 2))
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            L = rng.standard_normal((9, 48))
            rec = two_atom_observable_null_certificate(U, L)
            self.assertLessEqual(rec["actual_output_norm"],
                                 rec["certified_output_upper_bound"] + 1e-10)
            self.assertGreater(rec["gamma_input_sigma_min"], 0)

    def test_orthogonal_coordinate_rotation_covariance(self):
        rng = np.random.default_rng(105)
        U = rng.standard_normal((5, 2))
        L = rng.standard_normal((3, 5))
        Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        a = two_atom_observable_null_certificate(U, L)
        b = two_atom_observable_null_certificate(Q @ U, L @ Q.T)
        for k in ("epsilon_normalised_output_sigma_min", "gamma_input_sigma_min",
                  "actual_output_norm", "certified_output_upper_bound"):
            self.assertAlmostEqual(a[k], b[k], places=12)

    def test_reject_unsupported_rank_or_silent_columns(self):
        U = np.eye(3)[:, :2]
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            two_atom_observable_null_certificate(
                np.column_stack([U[:, 0], U[:, 0]]), np.eye(3))
        with self.assertRaisesRegex(ValueError, "zero downstream output"):
            two_atom_observable_null_certificate(U, np.array([[0., 0., 1.]]))


if __name__ == "__main__":
    unittest.main()
