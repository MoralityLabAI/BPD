"""Source-only replay of published PD04 seed-level negative scientific verdict."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import unittest

HERE=Path(__file__).resolve().parent


class TestPublishedPD04(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (HERE/"PER_SEED_20261117.csv").open(newline="",encoding="utf-8") as f:
            cls.rows=list(csv.DictReader(f))
        cls.config=json.loads((HERE/"CONFIG.json").read_text(encoding="utf-8"))

    def test_full_fresh_cohort_and_seed_pairing(self):
        seeds=[int(r["seed"]) for r in self.rows]
        self.assertEqual(seeds,self.config["seed_list"])
        self.assertEqual(len(set(seeds)),24)
        self.assertEqual(seeds[0],419)
        self.assertEqual(seeds[-1],569)
        self.assertNotIn(1009,seeds)

    def test_preregistered_comparison_fails_with_exact_p(self):
        a=[int(r["adaptive_success"]) for r in self.rows]
        b=[int(r["calibrated_success"]) for r in self.rows]
        o=[int(r["oracle_success"]) for r in self.rows]
        self.assertEqual(sum(a),0)
        self.assertEqual(sum(b),6)
        self.assertEqual(sum(o),22)
        aonly=sum(x and not y for x,y in zip(a,b))
        bonly=sum(y and not x for x,y in zip(a,b))
        self.assertEqual((aonly,bonly),(0,6))
        n=aonly+bonly
        p=min(1.,2*sum(math.comb(n,k)
                        for k in range(min(aonly,bonly)+1))/2**n)
        self.assertEqual(p,.03125)
        self.assertFalse(aonly-bonly>=self.config["success"]["primary_net_paired_seed_wins_min"])
        self.assertLess(sum(a),self.config["success"]["primary_adaptive_success_count_min"])

    def test_per_family_support_matches_seed_success_flags(self):
        cutoff=self.config["success"]["per_family_exact_four_pass_support_fraction_min"]
        for r in self.rows:
            for arm in ("adaptive","calibrated"):
                routes=[float(r[f"{arm}_{fam}_support"]) for fam in ("disjoint","overlap","nested")]
                v=float(r[f"{arm}_mse"])
                axis=float(r[f"{arm}_axis"])
                computed=(
                    all(f>=cutoff for f in routes)
                    and v<=self.config["success"]["overall_heldout_mse_max"]
                    and axis>=self.config["success"]["read_axis_cos_min"]
                )
                self.assertEqual(int(computed),int(r[f"{arm}_success"]),r["seed"])

    def test_mean_support_and_oracle_recovery(self):
        def avg(column):
            return statistics.mean(float(r[column]) for r in self.rows)
        self.assertAlmostEqual(avg("adaptive_disjoint_support"),.19921875,places=6)
        self.assertAlmostEqual(avg("adaptive_overlap_support"),.7073567708,places=6)
        self.assertAlmostEqual(avg("adaptive_nested_support"),.6025390625,places=6)
        self.assertAlmostEqual(avg("calibrated_disjoint_support"),.25,places=6)
        self.assertAlmostEqual(avg("calibrated_overlap_support"),.9954427083,places=6)
        self.assertAlmostEqual(avg("calibrated_nested_support"),.4658203125,places=6)
        self.assertEqual(sum(float(r["adaptive_axis"])>=.9 for r in self.rows),21)
        self.assertEqual(sum(float(r["calibrated_axis"])>=.9 for r in self.rows),23)
        self.assertEqual(sum(float(r["calibrated_mse"])<=.02 for r in self.rows),24)
        self.assertEqual(sum(int(r["oracle_success"]) for r in self.rows),22)

    def test_calibration_tie_and_selected_thresholds(self):
        vals=[float(r["calibrated_threshold"]) for r in self.rows]
        self.assertEqual(vals.count(.45),18)
        self.assertEqual(vals.count(.5),6)
        self.assertTrue(all(v in self.config["optimization"]["independent_threshold_candidates"]
                            for v in vals))


if __name__=="__main__":
    unittest.main()
