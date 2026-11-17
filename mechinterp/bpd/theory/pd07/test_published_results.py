"""Independent standard-library replay of PD07 first-run paired CSV.

Source: original immutable 144-arm JSON artifact 14340351331 from GitHub
Action 44109446626. This test does no neural model training.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent
ARM_NAMES=(
    "read_coupled","raw_independent","read_detached",
    "raw_count_supervised","read_planted_frozen","oracle_support",
)
EXPECTED=(1,0,1,0,24,24)


class TestPublishedPD07(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT/"PER_SEED_20261117.csv").open(
            encoding="utf-8",newline=""
        ) as stream:
            cls.rows=list(csv.DictReader(stream))
        cls.cfg=json.loads((ROOT/"CONFIG.json").read_text(encoding="utf-8"))

    def test_complete_source_locked_new_seed_cohort(self):
        self.assertEqual(len(self.rows),24)
        self.assertEqual([int(r["seed"]) for r in self.rows],self.cfg["seeds"])
        self.assertEqual(len({int(r["seed"]) for r in self.rows}),24)
        self.assertEqual(ARM_NAMES,tuple(a["key"] for a in self.cfg["arms"]))

    def test_preserve_preregistered_fail_and_exact_paired_test(self):
        for name, count in zip(ARM_NAMES,EXPECTED):
            self.assertEqual(sum(int(r[name+"_success"]) for r in self.rows),
                             count,name)
        primary=[int(r["read_coupled_success"]) for r in self.rows]
        other=[int(r["raw_independent_success"]) for r in self.rows]
        one=sum(a and not b for a,b in zip(primary,other))
        two=sum(b and not a for a,b in zip(primary,other))
        self.assertEqual((one,two),(1,0))
        n=one+two
        p=min(1.0,2*sum(math.comb(n,k)
                       for k in range(min(one,two)+1))/2**n)
        self.assertEqual(p,1.0)
        self.assertFalse(
            sum(primary)>=self.cfg["success"]["primary_min_successes"] and
            one-two>=self.cfg["success"]["primary_net_seed_wins_min"] and
            p<=self.cfg["success"]["primary_exact_mcnemar_p_max"]
        )
        self.assertEqual([int(r["seed"]) for r in self.rows
                          if r["read_coupled_success"]=="1"],[1063])

    def test_planted_factor_success_on_every_heldout_combination(self):
        for r in self.rows:
            for arm in ("read_planted_frozen","oracle_support"):
                self.assertEqual(int(r[arm+"_success"]),1)
                for f in ("k2_unseen","k3_unseen","transition_unseen"):
                    self.assertEqual(float(r[f"{arm}_{f}"]),1.0)

    def test_recovered_axis_diagnostic_and_ood_failure(self):
        for arm, axis_mean, pair_mean, triple_mean in (
            ("read_coupled",.8049979893,.0892,.0822),
            ("raw_independent",.8352031466,.0063,.0009),
            ("raw_count_supervised",.9759559774,.2222,.3639),
        ):
            def avg(key):
                return statistics.mean(float(r[key]) for r in self.rows)
            self.assertAlmostEqual(avg(arm+"_axis"),axis_mean,places=5)
            self.assertAlmostEqual(avg(arm+"_k2_unseen"),pair_mean,delta=.0001)
            self.assertAlmostEqual(avg(arm+"_k3_unseen"),triple_mean,delta=.0001)
        self.assertEqual(sum(float(r["read_coupled_axis"])>=.9
                             for r in self.rows),4)
        self.assertEqual(sum(float(r["raw_count_supervised_axis"])>=.9
                             for r in self.rows),21)


if __name__=="__main__":
    unittest.main()
