"""PD05 independent replay of immutable first-run seed-paired CSV.

Scientific input: GitHub Actions artifact 14335015466, source-pinned
study BPD-PD-05, run 44096809925. This test runs no neural optimization.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parent

SUCCESS_ARMS = (
    "supervised_sigmoid", "unsupervised_sigmoid", "supervised_identity",
    "unsupervised_identity", "unsupervised_tempered2",
    "supervised_frozen", "oracle_count", "calibrated_independent",
)
EXPECTED_COUNTS = (21,0,14,0,0,23,24,6)


class TestPublishedPD05(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
        with (ROOT / "PER_SEED_20261117.csv").open(newline="", encoding="utf-8") as f:
            cls.rows = list(csv.DictReader(f))

    def test_complete_fresh_cohort_and_fixed_arm_order(self):
        self.assertEqual(len(self.rows), 24)
        self.assertEqual(
            [int(r["seed"]) for r in self.rows], self.cfg["seeds"]
        )
        self.assertEqual(
            len(self.cfg["arms"]), len(SUCCESS_ARMS)
        )
        self.assertEqual(len({r["seed"] for r in self.rows}), 24)

    def test_first_registered_primary_result_and_exact_pairing(self):
        for arm, expected in zip(SUCCESS_ARMS, EXPECTED_COUNTS):
            got = sum(int(r[arm + "_success"]) for r in self.rows)
            self.assertEqual(got, expected, arm)
        supervised = [int(r["supervised_sigmoid_success"]) for r in self.rows]
        original = [int(r["unsupervised_sigmoid_success"]) for r in self.rows]
        sup_only = sum(a and not b for a,b in zip(supervised,original))
        orig_only = sum(b and not a for a,b in zip(supervised,original))
        self.assertEqual((sup_only, orig_only), (21,0))
        n = sup_only + orig_only
        p = min(1.0, 2 * sum(math.comb(n,k)
                             for k in range(min(sup_only,orig_only)+1)) / 2**n)
        self.assertEqual(p, 9.5367431640625e-7)
        self.assertGreaterEqual(sum(supervised), self.cfg["success"]["primary_min_success_24"])
        self.assertGreaterEqual(sup_only-orig_only, self.cfg["success"]["primary_net_seed_wins_min"])
        self.assertLessEqual(p, self.cfg["success"]["primary_exact_mcnemar_p_max"])

    def test_family_level_support_and_readout_recovery(self):
        def avg(field):
            return statistics.mean(float(x[field]) for x in self.rows)
        self.assertAlmostEqual(avg("primary_disjoint_route"), .9342447917, places=7)
        self.assertAlmostEqual(avg("primary_overlap_route"), 1.0, places=7)
        self.assertAlmostEqual(avg("primary_nested_route"), 1.0, places=7)
        self.assertAlmostEqual(avg("control_disjoint_route"), .1705729167, places=7)
        self.assertAlmostEqual(avg("control_overlap_route"), .7438151042, places=7)
        self.assertAlmostEqual(avg("control_nested_route"), .54296875, places=7)
        self.assertAlmostEqual(avg("primary_disjoint_count_accuracy"), .9948730469, places=7)
        self.assertAlmostEqual(avg("control_disjoint_count_accuracy"), .5359700521, places=7)
        self.assertEqual(
            [int(r["seed"]) for r in self.rows if r["supervised_sigmoid_success"]=="0"],
            [631,673,691],
        )
        for r in self.rows:
            if r["supervised_sigmoid_success"]=="1":
                self.assertLessEqual(float(r["primary_mse"]),.02)
                self.assertGreaterEqual(float(r["primary_axis"]),.9)
                self.assertTrue(all(float(r[f"primary_{f}_route"])>=.9
                                    for f in ("disjoint","overlap","nested")))

    def test_secondary_identity_comparison_labelled_unadjusted(self):
        sig = [int(r["supervised_sigmoid_success"]) for r in self.rows]
        ident = [int(r["supervised_identity_success"]) for r in self.rows]
        self.assertEqual(sum(a and not b for a,b in zip(sig,ident)),7)
        self.assertEqual(sum(b and not a for a,b in zip(sig,ident)),0)
        self.assertEqual(2.0/2**7,.015625)


if __name__=="__main__":
    unittest.main()
