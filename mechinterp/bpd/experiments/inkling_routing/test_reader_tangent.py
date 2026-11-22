"""Exact Goodfire reader geometry tests; standard library, no model weights."""
from __future__ import annotations

import math
import unittest

from mechinterp.bpd.experiments.inkling_routing.reader_tangent import (
    physical_patch_audit,
    reader_tangent,
    writer_scan,
)


class ReaderTangentTests(unittest.TestCase):
    def setUp(self):
        self.u = (0.7, -0.2, 0.3, 1.1)
        self.v2 = (0.6, 0.1, -0.4, 0.3)
        self.v3 = (0.5, 1.0, 0.4, -0.2)

    def test_exact_coordinate_sum_and_energy_shares(self):
        g = reader_tangent(self.u, self.v2, self.v3)
        self.assertAlmostEqual(sum(g["gap_coordinate_terms"]), g["gap"], places=12)
        self.assertAlmostEqual(sum(g["score3_coordinate_terms"]), g["score3"], places=12)
        self.assertAlmostEqual(sum(g["rms_energy_shares"]), 1, places=12)
        self.assertAlmostEqual(g["radial_score_derivative"], 0, places=12)

    def test_finite_difference_matches_full_rms_jacobian(self):
        direction = (0.3, -0.1, 0.2, 0.5)
        eps = 1e-5
        plus = tuple(x + eps * d for x, d in zip(self.v3, direction))
        minus = tuple(x - eps * d for x, d in zip(self.v3, direction))
        start = reader_tangent(self.u, self.v2, self.v3)
        gp = reader_tangent(self.u, self.v2, plus)
        gm = reader_tangent(self.u, self.v2, minus)
        analytic_gap = sum(g * d for g, d in zip(start["score3_gradient"], direction))
        analytic_gate = sum(g * d for g, d in zip(start["beta3_gradient"], direction))
        self.assertAlmostEqual((gp["gap"] - gm["gap"]) / (2 * eps),
                               analytic_gap, places=8)
        self.assertAlmostEqual((gp["beta3"] - gm["beta3"]) / (2 * eps),
                               analytic_gate, places=8)

    def test_high_writer_alignment_can_be_radial_and_gate_inert(self):
        u = (1.0, 0.1, 0.0, 0.0)
        v2 = (0.0, 1.0, 0.0, 0.0)
        v3 = (12.0, 0.0, 0.0, 0.0)
        score = reader_tangent(u, v2, v3)
        self.assertAlmostEqual(score["rms_energy_shares"][0], 1, places=12)
        self.assertAlmostEqual(score["score3_gradient"][0], 0, places=12)
        writers = writer_scan(u, v2, v3, {
            "radial": (1.0, 0.0, 0.0, 0.0),
            "tangent": (0.0, 1.0, 0.0, 0.0),
        })
        self.assertGreater(writers["radial"]["absolute_raw_reader_cosine"], 0.99)
        self.assertAlmostEqual(writers["radial"]["predicted_gate_shift_per_l2"],
                               0, places=12)
        self.assertGreater(writers["tangent"]["predicted_gate_shift_per_l2"], 0)
        radial_patch = physical_patch_audit(u, v2, v3, (13.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(radial_patch["actual_beta3_change"], 0, places=12)

    def test_positive_rescaling_preserves_gate_and_scales_gradient(self):
        base = reader_tangent(self.u, self.v2, self.v3)
        scaled = reader_tangent(self.u, self.v2, tuple(3 * v for v in self.v3))
        self.assertAlmostEqual(base["beta3"], scaled["beta3"], places=12)
        for a, b in zip(base["score3_gradient"], scaled["score3_gradient"]):
            self.assertAlmostEqual(a, 3 * b, places=12)

    def test_physical_patch_has_separate_linearization_residual(self):
        patch = tuple(v + 1e-4 * d for v, d in zip(
            self.v3, (0.3, -0.1, 0.2, 0.5)
        ))
        out = physical_patch_audit(self.u, self.v2, self.v3, patch)
        self.assertGreater(out["physical_dose_l2"], 0)
        self.assertLess(abs(out["gate_linearization_residual"]), 1e-7)
        self.assertNotIn("task_accuracy", out)
        self.assertNotIn("p_value", out)

    def test_reject_invalid_vectors_and_controls(self):
        with self.assertRaises(ValueError):
            reader_tangent(self.u, self.v2, (0, 0, 0, 0))
        with self.assertRaises(ValueError):
            reader_tangent(self.u, self.v2, (1, math.nan, 2, 3))
        with self.assertRaises(ValueError):
            reader_tangent(self.u, self.v2, (1, 2))
        with self.assertRaises(ValueError):
            writer_scan(self.u, self.v2, self.v3, {})
        with self.assertRaises(ValueError):
            writer_scan(self.u, self.v2, self.v3, {"zero": (0, 0, 0, 0)})
        with self.assertRaises(ValueError):
            physical_patch_audit(self.u, self.v2, self.v3, (1, 2))


if __name__ == "__main__":
    unittest.main()
