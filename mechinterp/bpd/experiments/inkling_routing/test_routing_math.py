"""CPU source-only checks for INK-BPD-01; no network, model or torch."""
from __future__ import annotations

import math
import unittest

from mechinterp.bpd.experiments.inkling_routing.routing_math import (
    l2, message, paired_dose_audit, path_decomposition,
    projective_shares, rms, rms_normalize, rotate_pair_at_fixed_dose, route_gap,
)


class InkyRoutingMathTests(unittest.TestCase):
    def setUp(self):
        self.u = (0.6, -0.3, 0.8, 0.5)
        self.v2 = (0.2, 0.4, 0.5, 0.7)
        self.v3 = (0.7, 0.3, 0.6, 0.4)

    def test_route_gap_coordinate_attribution_sums_to_logit_gap(self):
        r = route_gap(self.u, self.v2, self.v3)
        self.assertAlmostEqual(sum(r["per_coordinate"]), r["gap"], places=12)
        self.assertGreater(r["beta3"], 0)
        self.assertLess(r["beta3"], 1)
        with self.assertRaises(ValueError):
            route_gap((1.0,), self.v2, self.v3)

    def test_norm_matched_orthogonal_rotation_matches_physical_dose(self):
        for pair in ((0, 1), (2, 3)):
            x = rotate_pair_at_fixed_dose(self.v3, *pair, 0.25)
            self.assertAlmostEqual(rms(self.v3), rms(x), places=12)
            self.assertAlmostEqual(l2(self.v3, x), 0.25, places=12)
        with self.assertRaisesRegex(ValueError, "unachievable"):
            rotate_pair_at_fixed_dose(self.v3, 0, 1, 4.0)
        with self.assertRaisesRegex(ValueError, "distinct"):
            rotate_pair_at_fixed_dose(self.v3, 0, 0, 0.1)

    def test_path_components_telescope_to_full_message_change(self):
        edited = rotate_pair_at_fixed_dose(self.v3, 0, 1, 0.20)
        r = path_decomposition(self.u, self.v2, self.v3, edited)
        total = tuple(a+b+c for a,b,c in zip(
            r["selector_only_delta"],r["content_only_delta"],r["interaction_delta"]))
        for t,d in zip(total,r["total_delta"]):
            self.assertAlmostEqual(t,d,places=12)
        self.assertAlmostEqual(sum((r["clean_beta3"],1-r["clean_beta3"])),1,places=12)
        self.assertEqual(len(message(self.v2,self.v3,0.5)),4)

    def test_projective_shares_preserve_unit_sum_with_negative_contributions(self):
        vals = projective_shares(((2,0),(0,1),(-0.5,0)))
        self.assertAlmostEqual(sum(vals),1,places=12)
        with self.assertRaisesRegex(ValueError, "undefined"):
            projective_shares(((1,0),(-1,0)))
        with self.assertRaises(ValueError):
            rms_normalize((0,0))

    def test_pair_summary_is_source_only_and_has_separate_dose_receipts(self):
        out=paired_dose_audit(self.u,self.v2,self.v3,(0,1),(2,3),0.25)
        self.assertEqual(out["status"],"source_only_math_fixture")
        for arm in ("target","sham"):
            self.assertAlmostEqual(out[arm]["dose_l2"],0.25,places=12)
            self.assertAlmostEqual(out[arm]["rms_before"],out[arm]["rms_after"],places=12)
            self.assertTrue(math.isfinite(out[arm]["beta3_change"]))
        self.assertNotIn("p_value",out)
        self.assertNotIn("native_task_success",out)


if __name__ == "__main__":
    unittest.main()
