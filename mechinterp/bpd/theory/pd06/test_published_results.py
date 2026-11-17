"""PD06 independent source-only replay of the registered 32-seed result.

The original full 224-arm JSON is pinned to GitHub Actions artifact
14338021049 (run 44103217529), SHA256:
6d33689b84ac40f169afc04d1cae50d35bc74ff319d820a56d02143d14fb9d0b.

This test never retrains or reads the heldout data generator.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import unittest

HERE=Path(__file__).resolve().parent
FAMILIES=("k1_seen","k2_seen","k2_unseen","k3_seen","k3_unseen","transition_unseen")

class TestPublishedPD06(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads((HERE/"CONFIG.json").read_text(encoding="utf-8"))
        with (HERE/"PER_SEED_20261117.csv").open(newline="",encoding="utf-8") as f:
            cls.rows=list(csv.DictReader(f))

    def test_locked_independent_rotations_and_data_completeness(self):
        self.assertEqual(len(self.rows),32)
        self.assertEqual([int(r["seed"]) for r in self.rows],self.cfg["seeds"])
        self.assertEqual(len({int(r["seed"]) for r in self.rows}),32)
        self.assertEqual(set(FAMILIES),set(self.cfg["source"]["eval_families"]))
        for row in self.rows:
            for arm in ("count","support"):
                for family in FAMILIES:
                    n=int(row[f"{arm}_{family}_boards"])
                    self.assertTrue(0<=n<=192)

    def test_primary_preregistered_fail_and_exact_McNemar(self):
        fields=("unsup_ok","count_ok","support_ok","oracle_count_ok",
                "oracle_support_ok","planted_ok","dense_ok")
        expected=(0,1,17,0,32,6,0)
        for name,count in zip(fields,expected):
            self.assertEqual(sum(int(r[name]) for r in self.rows),count,name)
        sup=[int(r["count_ok"]) for r in self.rows]
        unsup=[int(r["unsup_ok"]) for r in self.rows]
        aonly=sum(a and not b for a,b in zip(sup,unsup))
        bonly=sum(b and not a for a,b in zip(sup,unsup))
        self.assertEqual((aonly,bonly),(1,0))
        n=aonly+bonly
        p=min(1.,2*sum(math.comb(n,k)
                       for k in range(min(aonly,bonly)+1))/2**n)
        self.assertEqual(p,1.)
        sc=self.cfg["scoring"]
        self.assertFalse(
            sum(sup)>=sc["primary_min_successes"]
            and aonly-bonly>=sc["primary_net_paired_wins_min"]
            and p<=sc["primary_mcnemar_two_sided_p_max"]
        )
        self.assertEqual(
            [int(r["seed"]) for r in self.rows if r["count_ok"]=="1"],[751]
        )

    def test_full_rotated_read_axis_but_missing_novel_support(self):
        self.assertTrue(all(float(r["count_axis"])>=.9 for r in self.rows))
        self.assertTrue(all(float(r["support_axis"])>=.9 for r in self.rows))
        mean=lambda arm,f:statistics.mean(
            int(r[f"{arm}_{f}_boards"])/192 for r in self.rows
        )
        known={
            "k1_seen":(.9790,1.0),
            "k2_seen":(.9347,.9624),
            "k2_unseen":(.3118,.7108),
            "k3_seen":(.8958,.9388),
            "k3_unseen":(.5620,.8224),
            "transition_unseen":(.4207,.8026),
        }
        for fam,(count,support) in known.items():
            self.assertAlmostEqual(mean("count",fam),count,delta=.00025)
            self.assertAlmostEqual(mean("support",fam),support,delta=.00025)
        self.assertGreater(
            mean("count","k2_seen")-mean("count","k2_unseen"),.60
        )
        self.assertGreater(
            mean("support","k2_unseen")-mean("count","k2_unseen"),.39
        )
        self.assertGreater(
            mean("support","transition_unseen")-mean("count","transition_unseen"),.38
        )

    def test_success_requires_all_six_source_families(self):
        score=self.cfg["scoring"]
        for row in self.rows:
            for arm,col in (("count","count_ok"),("support","support_ok")):
                if int(row[col]):
                    self.assertGreaterEqual(float(row[f"{arm}_axis"]),score["axis_cosine_min"])
                    self.assertLessEqual(float(row[f"{arm}_mse"]),score["full_recon_MSE_max"]+1e-7)
                    for fam in FAMILIES:
                        unseen="unseen" in fam
                        minimum=(score["exact_support_unseen_min"] if unseen
                                 else score["exact_support_seen_min"])
                        self.assertGreaterEqual(
                            int(row[f"{arm}_{fam}_boards"])/192,minimum
                        )

if __name__=="__main__":
    unittest.main()
