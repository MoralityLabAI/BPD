"""Stdlib replay of PD10 first-run 24-seed paired CSV.

Original immutable GitHub Actions run 44124926867, artifact 14347856096;
full uncompressed JSON SHA256 cfbac7207a7e354769bffa2620f94ef7d3379c7018de0d737b7e4209a9649b81.
The test uses no model training or additional observed held-out examples.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parent
ARM_COLS={
    "cold015":"cold_success",
    "anneal060_030_015":"anneal_success",
    "cold030":"cold030_success",
    "warm030_to015":"warm_success",
    "privileged_pairmix015":"pairmix_success",
    "count_supervised_positive":"supervised_success",
}
EXPECTED={"cold015":2,"anneal060_030_015":0,"cold030":2,
          "warm030_to015":2,"privileged_pairmix015":0,
          "count_supervised_positive":24}


class TestPublishedPD10(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads((ROOT/"CONFIG.json").read_text(encoding="utf-8"))
        with (ROOT/"PER_SEED_20261117.csv").open(newline="",encoding="utf-8") as f:
            cls.rows=list(csv.DictReader(f))

    def test_complete_new_cohort_and_original_binary_outcomes(self):
        self.assertEqual(len(self.rows),24)
        self.assertEqual([int(r["seed"]) for r in self.rows],self.cfg["seeds"])
        self.assertEqual(len({r["seed"] for r in self.rows}),24)
        self.assertNotIn(self.cfg["excluded_engineering_seed"],self.cfg["seeds"])
        for arm,column in ARM_COLS.items():
            self.assertEqual(sum(int(r[column]) for r in self.rows),EXPECTED[arm])

    def test_preregistered_failure_remains_failure(self):
        cold=[int(r["cold_success"]) for r in self.rows]
        annealed=[int(r["anneal_success"]) for r in self.rows]
        cold_only=sum(a and not b for a,b in zip(cold,annealed))
        annealed_only=sum(b and not a for a,b in zip(cold,annealed))
        self.assertEqual((cold_only,annealed_only),(2,0))
        n=cold_only+annealed_only
        p=min(1.0,2*sum(math.comb(n,k) for k in range(min(cold_only,annealed_only)+1))/2**n)
        self.assertEqual(p,.5)
        sc=self.cfg["scoring"]
        self.assertFalse(sum(cold)>=sc["primary_min_successes"]
                         and cold_only-annealed_only>=sc["primary_net_paired_wins_min"]
                         and p<=sc["primary_exact_two_sided_McNemar_p_max"])
        self.assertEqual([int(r["seed"]) for r in self.rows if r["cold_success"]=="1"],
                         [1493,1613])
        self.assertEqual([int(r["seed"]) for r in self.rows if r["cold030_success"]=="1"],
                         [1489,1613])
        self.assertEqual([int(r["seed"]) for r in self.rows if r["warm_success"]=="1"],
                         [1489,1613])

    def test_mixed_pair_stress_unchanged_and_strict_source_margin(self):
        avg=lambda name:statistics.mean(float(r[name]) for r in self.rows)
        self.assertAlmostEqual(avg("cold_cos"),.7828357,places=5)
        self.assertAlmostEqual(avg("anneal_cos"),.8593072,places=5)
        self.assertAlmostEqual(avg("cold_min_cos"),.654106,places=5)
        self.assertAlmostEqual(avg("anneal_min_cos"),.718637,places=5)
        self.assertAlmostEqual(avg("pairmix_cos"),.858258,places=5)
        self.assertAlmostEqual(avg("cold_mse"),.0133266,places=5)
        self.assertAlmostEqual(avg("anneal_mse"),.0008310,places=5)
        self.assertEqual(sum(float(r["anneal_ood_pair"])>0 for r in self.rows),0)
        self.assertEqual(sum(float(r["cold_ood_pair"])>=.7 for r in self.rows),2)
        self.assertEqual(sum(float(r["pairmix_success"]) for r in self.rows),0)

    def test_privileged_controls_are_not_counted_as_unlabelled(self):
        cfg={a["key"]:a for a in self.cfg["arms"]}
        for arm in ("cold015","cold030","warm030_to015","anneal060_030_015"):
            self.assertFalse(cfg[arm]["uses_true_Q_for_initialization"])
            self.assertFalse(cfg[arm]["uses_count_labels"])
        self.assertTrue(cfg["privileged_pairmix015"]["uses_true_Q_for_initialization"])
        self.assertTrue(cfg["count_supervised_positive"]["uses_count_labels"])


if __name__=="__main__":
    unittest.main()
