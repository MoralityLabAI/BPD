"""Independent stdlib replay of PD12 published first-run seed-level CSV.

Original registered 256-arm JSON, run 44135442855, artifact 14352700682:
SHA256 27d521c5f60d1bb28fe1d38b27eebb0a5b07af8c2f63160ce7561a28f5469186.
This suite does not sample source data or run trained models.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parent


class TestPublishedPD12(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/"CONFIG.json").read_text(encoding="utf-8"))
        with (ROOT/"PER_SEED_20261118.csv").open(
            encoding="utf-8",newline=""
        ) as f:
            cls.rows=list(csv.DictReader(f))
        cls.cells={c["id"]:c for c in cls.config["factorial"]["cells"]}

    def test_exact_64_units_in_eight_frozen_cells(self):
        self.assertEqual(len(self.rows),64)
        known={int(seed):c["id"] for c in self.cells.values() for seed in c["seed_list"]}
        self.assertEqual(len(known),64)
        self.assertEqual(
            {int(row["seed"]):row["cell"] for row in self.rows}, known
        )
        for row in self.rows:
            cell=self.cells[row["cell"]]
            self.assertEqual(int(row["dimension"]),cell["dimension"])
            self.assertEqual(row["support_distribution"],cell["support_distribution"])
            self.assertEqual(row["margin"],cell["margin"])
        self.assertNotIn(self.config["execution"]["smoke_seed"],known)

    def test_registered_scientific_fail_and_exact_paired_advantage(self):
        rows=self.rows
        counts={
            name:sum(int(x[name+"_success"]) for x in rows)
            for name in ("global37","local37","global13","oracle")
        }
        self.assertEqual(counts,{"global37":35,"local37":0,"global13":15,"oracle":64})
        global_ok=[int(x["global37_success"]) for x in rows]
        local_ok=[int(x["local37_success"]) for x in rows]
        global_only=sum(a and not b for a,b in zip(global_ok,local_ok))
        local_only=sum(b and not a for a,b in zip(global_ok,local_ok))
        self.assertEqual((global_only,local_only),(35,0))
        n=global_only+local_only
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(global_only,local_only)+1))/2**n)
        self.assertEqual(p,5.820766091346741e-11)
        s=self.config["scoring"]
        dim8=sum(int(x["global37_success"]) for x in rows if x["dimension"]=="8")
        dim16=sum(int(x["global37_success"]) for x in rows if x["dimension"]=="16")
        self.assertEqual((dim8,dim16),(22,13))
        self.assertGreaterEqual(global_only-local_only,s["primary_net_paired_seed_wins_min"])
        self.assertLessEqual(p,s["primary_two_sided_exact_McNemar_p_max"])
        self.assertFalse(counts["global37"]>=s["primary_global_total_successes_min"])
        self.assertFalse(dim8>=s["primary_global_d8_successes_min"])
        self.assertTrue(dim16>=s["primary_global_d16_successes_min"])

    def test_each_factorial_cell_outcome(self):
        expected={
            "d8_uniform_strong":(8,0,7,8),
            "d8_uniform_weak":(8,0,0,8),
            "d8_clustered_strong":(0,0,0,8),
            "d8_clustered_weak":(6,0,0,8),
            "d16_uniform_strong":(8,0,8,8),
            "d16_uniform_weak":(5,0,0,8),
            "d16_clustered_strong":(0,0,0,8),
            "d16_clustered_weak":(0,0,0,8),
        }
        for name,tuple_expect in expected.items():
            row=[r for r in self.rows if r["cell"]==name]
            self.assertEqual(len(row),8)
            counts=tuple(sum(int(r[a+"_success"]) for r in row)
                         for a in ("global37","local37","global13","oracle"))
            self.assertEqual(counts,tuple_expect,name)

    def test_equal_budget_cost_counts_and_strict_support(self):
        for row in self.rows:
            d=int(row["dimension"])
            pairs=d*(d-1)//2
            self.assertEqual(int(row["global37_scores"]),3*pairs*37)
            self.assertEqual(int(row["global13_scores"]),3*pairs*13)
            if row["global37_success"]=="1":
                self.assertGreaterEqual(float(row["global37_cosine"]),.9)
                self.assertGreaterEqual(float(row["global37_k2_unseen"]),.7)
                self.assertGreaterEqual(float(row["global37_k3_unseen"]),.7)
                self.assertGreaterEqual(float(row["global37_k4_unseen"]),.7)
                self.assertGreaterEqual(float(row["global37_transition"]),.7)
                self.assertLessEqual(float(row["global37_mse"]),.035)
            self.assertGreater(float(row["global37_seconds"]),0)
            self.assertGreater(float(row["global13_seconds"]),0)
        self.assertTrue(all(r["oracle_success"]=="1" for r in self.rows))

    def test_compute_cost_and_factorial_mean_alignment(self):
        def average(dim,key):
            return statistics.mean(float(r[key]) for r in self.rows
                                   if int(r["dimension"])==dim)
        self.assertAlmostEqual(average(8,"global37_seconds"),.2016,places=3)
        self.assertAlmostEqual(average(16,"global37_seconds"),.5941,places=3)
        self.assertAlmostEqual(average(8,"global13_seconds"),.1077,places=3)
        self.assertAlmostEqual(average(16,"global13_seconds"),.1913,places=3)
        self.assertEqual(
            sum(int(r["global37_success"]) for r in self.rows
                if r["support_distribution"]=="uniform"),29
        )
        self.assertEqual(
            sum(int(r["global37_success"]) for r in self.rows
                if r["support_distribution"]=="clustered"),6
        )
        for cell in ("d8_clustered_strong","d16_clustered_strong"):
            rows=[r for r in self.rows if r["cell"]==cell]
            self.assertLess(statistics.mean(float(r["global37_min_cosine"])
                                            for r in rows),.75)
            self.assertTrue(all(float(r["global37_k4_unseen"])==0 for r in rows))


if __name__=="__main__":
    unittest.main()
