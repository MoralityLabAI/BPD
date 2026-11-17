"""Independent stdlib replay of the frozen PD13 32-seed scientific failure.

This reads committed per-seed CSV and CONFIG only. It never regenerates
training data or refits a model. The original complete 160-arm JSON receipt
is archived at GitHub Actions run 44140294730, artifact 14353309264,
SHA256 04bbda829276662e1fb37edbbd6cf5c49a2b8321cd55a712a2cd79c09b88ad20.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parent
ARM_KEYS=("priority2048","lex2048","priority256","lex256")
EXPECTED_COUNTS=(2,2,1,0)


class TestPublishedPD13(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT/"PER_SEED_20261118.csv").open(
            newline="",encoding="utf-8",
        ) as f:
            cls.rows=list(csv.DictReader(f))
        cls.cfg=json.loads((ROOT/"CONFIG.json").read_text(encoding="utf-8"))

    def test_exact_new_32_seed_roster_and_source_cells(self):
        expected=[
            (seed,cell["id"]) for cell in self.cfg["seed_cells"]
            for seed in cell["seeds"]
        ]
        self.assertEqual(
            [(int(r["seed"]),r["cell"]) for r in self.rows], expected,
        )
        self.assertEqual(len(expected),32)
        self.assertEqual(len({r["seed"] for r in self.rows}),32)
        self.assertNotIn(str(self.cfg["smoke_seed"]),
                         [r["seed"] for r in self.rows])

    def test_registered_primary_failure_and_exact_seed_paired_mcnemar(self):
        rows=self.rows
        observed=tuple(
            sum(int(r[arm+"_success"]) for r in rows)
            for arm in ARM_KEYS
        )
        self.assertEqual(observed,EXPECTED_COUNTS)
        self.assertEqual(sum(int(r["oracle_success"]) for r in rows),32)
        a=[int(r["priority2048_success"]) for r in rows]
        b=[int(r["lex2048_success"]) for r in rows]
        only_a=sum(x and not y for x,y in zip(a,b))
        only_b=sum(y and not x for x,y in zip(a,b))
        self.assertEqual((only_a,only_b),(1,1))
        n=only_a+only_b
        p=min(1.,2*sum(math.comb(n,k)
                        for k in range(min(only_a,only_b)+1))/2**n)
        self.assertEqual(p,1.0)
        sc=self.cfg["scoring"]
        self.assertFalse(
            observed[0]>=sc["primary_min_successes"]
            and only_a-only_b>=sc["primary_net_paired_wins_min"]
            and p<=sc["primary_exact_mcnemar_p_max"]
        )

    def test_regime_strata_and_seed_level_nonmonotonic_volume(self):
        for margin,expected in (
            ("strong",(0,0,0,0,16)),
            ("weak",(2,2,1,0,16)),
        ):
            rows=[r for r in self.rows if r["cell"]=="clustered16d_"+margin]
            self.assertEqual(len(rows),16)
            self.assertEqual(
                tuple(sum(int(r[arm+"_success"]) for r in rows)
                      for arm in ARM_KEYS+("oracle",)),
                expected,
            )
        self.assertEqual(
            [int(r["seed"]) for r in self.rows if r["priority2048_success"]=="1"],
            [2477,2551],
        )
        self.assertEqual(
            [int(r["seed"]) for r in self.rows if r["lex2048_success"]=="1"],
            [2477,2521],
        )
        self.assertEqual(
            [int(r["seed"]) for r in self.rows if r["priority256_success"]=="1"],
            [2579],
        )

    def test_read_axis_and_ood_summary_matches_source_receipt(self):
        rows=self.rows
        def avg(key,subset=rows):
            return statistics.mean(float(r[key]) for r in subset)
        for key,value in (
            ("priority2048_cos",.8867979127141834),
            ("lex2048_cos",.8813097142936834),
            ("priority256_cos",.9029632696218509),
            ("lex256_cos",.8622607109094738),
            ("priority2048_unseen_k4",.064127604),
            ("lex2048_unseen_k4",.069335938),
            ("priority2048_mse",.0013350706328765227),
            ("lex2048_mse",.0013983099285042044),
        ):
            self.assertAlmostEqual(avg(key),value,places=5,key)
        strong=[r for r in rows if r["cell"]=="clustered16d_strong"]
        self.assertTrue(all(float(r["priority2048_unseen_k4"])==0 for r in strong))
        self.assertTrue(all(float(r["lex2048_unseen_k4"])==0 for r in strong))
        self.assertAlmostEqual(avg("priority2048_cos",strong),.85956,places=4)

    def test_registered_angle_and_sample_comparison_costs(self):
        objective=self.cfg["objective"]
        self.assertEqual(objective["sweeps"],3)
        self.assertEqual(objective["pair_count"],120)
        self.assertEqual(objective["candidate_grid"],37)
        self.assertEqual(3*120*37,13320)
        self.assertEqual(
            13320*4*self.cfg["source"]["train_full_boards"],109117440,
        )
        self.assertEqual(
            13320*4*self.cfg["source"]["train_small_boards"],13639680,
        )
        self.assertEqual(
            self.cfg["source"]["train_full_boards"] //
            self.cfg["source"]["train_small_boards"],8,
        )
        self.assertEqual(self.cfg["scoring"]["primary_arm"],"priority2048")
        self.assertEqual(self.cfg["scoring"]["primary_control"],"lex2048")


if __name__=="__main__":
    unittest.main()
