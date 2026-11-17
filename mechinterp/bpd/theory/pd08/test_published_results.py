"""Independent stdlib replay of PD08 first-run per-seed CSV.

Original first-run 120-arm JSON: GitHub Actions 44114726057, artifact
14342451949; SHA256 73704fd7eeef67930b7913ebdcf81d0fc748039898f7e797d86bc410ae683483.

No model training or source-randomness is involved in these checks.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent


class TestPublishedPD08(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (HERE/"PER_SEED_20261117.csv").open(encoding="utf-8",newline="") as f:
            cls.rows=list(csv.DictReader(f))
        cls.cfg=json.loads((HERE/"CONFIG.json").read_text(encoding="utf-8"))

    def test_source_locked_24_independent_new_rotations(self):
        self.assertEqual(len(self.rows),24)
        self.assertEqual([int(r["seed"]) for r in self.rows],self.cfg["seeds"])
        self.assertEqual(len({int(r["seed"]) for r in self.rows}),24)
        self.assertNotIn(self.cfg["smoke_seed"],self.cfg["seeds"])

    def test_preserve_preregistered_23_vs_0_pass_and_exact_p(self):
        counts={
            arm:sum(int(r[arm+"_success"]) for r in self.rows)
            for arm in ("read","raw","fixed","oracle","planted")
        }
        self.assertEqual(counts,{"read":23,"raw":0,"fixed":23,"oracle":23,"planted":24})
        read=[int(r["read_success"]) for r in self.rows]
        raw=[int(r["raw_success"]) for r in self.rows]
        aonly=sum(a and not b for a,b in zip(read,raw))
        bonly=sum(b and not a for a,b in zip(read,raw))
        self.assertEqual((aonly,bonly),(23,0))
        n=aonly+bonly
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(aonly,bonly)+1))/2**n)
        self.assertEqual(p,2.384185791015625e-7)
        sc=self.cfg["scoring"]
        self.assertGreaterEqual(counts["read"],sc["primary_read_successes_min"])
        self.assertGreaterEqual(aonly-bonly,sc["primary_net_seed_wins_min"])
        self.assertLessEqual(p,sc["primary_paired_two_sided_mcnemar_p_max"])

    def test_stage1_margin_and_single_factor_failure(self):
        high=[int(r["seed"]) for r in self.rows
              if int(r["stage1_margin_sufficient"])==1]
        self.assertEqual(len(high),23)
        self.assertEqual([int(r["seed"]) for r in self.rows if r["read_success"]=="0"],[1237])
        self.assertEqual([int(r["seed"]) for r in self.rows if r["fixed_success"]=="0"],[1237])
        self.assertEqual([int(r["seed"]) for r in self.rows if r["oracle_success"]=="0"],[1237])
        self.assertTrue(all(float(r["stage1_min_cosine"])>.99333
                            for r in self.rows if r["stage1_margin_sufficient"]=="1"))
        fail=next(r for r in self.rows if r["seed"]=="1237")
        self.assertAlmostEqual(float(fail["stage1_mean_cosine"]),.749858811,places=7)
        self.assertAlmostEqual(float(fail["stage1_min_cosine"]),.602009356,places=7)

    def test_read_gate_ood_transfer_vs_raw_and_threshold_drift(self):
        def avg(name):
            return statistics.mean(float(r[name]) for r in self.rows)
        for key,expected in (
            ("read_ood_pair",.9578993055),
            ("raw_ood_pair",.2734375),
            ("read_ood_triple",.9544270833),
            ("raw_ood_triple",.0342881944),
            ("read_ood_transition",.9574652778),
            ("raw_ood_transition",.1848958333),
            ("read_threshold_mean",.7342457250),
        ):
            self.assertAlmostEqual(avg(key),expected,places=5)
        self.assertAlmostEqual(avg("read_mse"),.0006701677,places=7)
        self.assertAlmostEqual(avg("raw_mse"),.0397298733,places=7)
        self.assertTrue(all(float(r["read_threshold_mean"])>.74
                            for r in self.rows if r["read_success"]=="1"))

    def test_fixed_read_gate_passes_at_margin_sufficient_axes(self):
        for r in self.rows:
            if int(r["stage1_margin_sufficient"]):
                self.assertEqual(r["read_success"],"1")
                self.assertEqual(r["fixed_success"],"1")
                self.assertEqual(r["oracle_success"],"1")


if __name__=="__main__":
    unittest.main()
