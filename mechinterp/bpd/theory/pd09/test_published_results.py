"""Source-only replay of the original 24-seed BPD-PD-09 scientific FAIL.

This file reads only the committed compact CSV extracted from the original
full GitHub Actions 120-arm JSON. It conducts no model training.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class TestPublishedPD09(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT/"PER_SEED_20261117.csv").open(
            encoding="utf-8",newline=""
        ) as stream:
            cls.rows=list(csv.DictReader(stream))
        cls.cfg=json.loads((ROOT/"CONFIG.json").read_text(encoding="utf-8"))

    def test_all_24_fresh_seed_pairs_and_five_arms(self):
        self.assertEqual(len(self.rows),24)
        self.assertEqual([int(x["seed"]) for x in self.rows],self.cfg["seeds"])
        self.assertEqual(len(set(self.cfg["seeds"])),24)
        self.assertEqual(len(self.cfg["arms"]),5)
        self.assertNotIn(self.cfg["smoke_seed"],self.cfg["seeds"])

    def test_registered_orthogonal_annealed_fail_and_exact_p(self):
        expected={
            "annealed_success":0, "fixed_success":0, "untied_success":0,
            "random_success":0, "supervised_success":24
        }
        for name,n in expected.items():
            self.assertEqual(sum(int(r[name]) for r in self.rows),n)
        a=[int(r["annealed_success"]) for r in self.rows]
        b=[int(r["fixed_success"]) for r in self.rows]
        ao=sum(x and not y for x,y in zip(a,b))
        bo=sum(y and not x for x,y in zip(a,b))
        self.assertEqual((ao,bo),(0,0))
        n=ao+bo
        p=1.0 if n==0 else min(1.0,2*sum(
            math.comb(n,k) for k in range(min(ao,bo)+1)
        )/2**n)
        self.assertEqual(p,1.0)
        self.assertFalse(sum(a)>=self.cfg["scoring"]["primary_successes_min"])

    def test_pairwise_45_degree_remix_in_all_annealed_models(self):
        pairs=Counter(r["annealed_mixed_latent_pair"] for r in self.rows)
        self.assertEqual(pairs,Counter({"0-1":11,"1-2":13}))
        self.assertEqual(sum(float(r["annealed_cos"])>=.9 for r in self.rows),0)
        self.assertAlmostEqual(
            statistics.mean(float(r["annealed_cos"]) for r in self.rows),
            .8586637477,places=6
        )
        self.assertAlmostEqual(
            statistics.mean(float(r["annealed_min_cos"]) for r in self.rows),
            .7173517098,places=6
        )
        self.assertTrue(all(.700<float(r["annealed_min_cos"])<.740
                            for r in self.rows))

    def test_unseen_support_zero_and_privileged_supervised_exact(self):
        for family in ("ood_pair","ood_triple","ood_transition"):
            self.assertTrue(all(float(r["annealed_"+family])==0 for r in self.rows))
            self.assertTrue(all(float(r["fixed_"+family])==0 for r in self.rows))
        self.assertTrue(all(r["supervised_success"]=="1" for r in self.rows))
        self.assertTrue(all(float(r["supervised_cos"])>.9999 for r in self.rows))
        self.assertAlmostEqual(
            statistics.mean(float(r["annealed_mse"]) for r in self.rows),
            .0008329411,places=7
        )
        self.assertAlmostEqual(
            statistics.mean(float(r["supervised_mse"]) for r in self.rows),
            .0000139884,places=7
        )


if __name__=="__main__":
    unittest.main()
