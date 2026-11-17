"""Independent source-only replay of PD08 first-run paired 24-seed CSV.

The original 144-arm JSON is immutable in GitHub Actions run 44114011511,
artifact 14340839503. No neural training occurs in this audit.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import unittest

HERE=Path(__file__).resolve().parent
ARMS=(
    "read_refit","raw_refit","read_nofit",
    "read_shuffled_refit","stage1_count_topk","oracle_support",
)
EXPECTED=(23,0,23,0,2,23)
FAMILIES=("k1_seen","k2_seen","k2_unseen",
          "k3_seen","k3_unseen","transition_unseen")


class TestPublishedPD08(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((HERE/"CONFIG.json").read_text(encoding="utf-8"))
        with (HERE/"PER_SEED_20261117.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            cls.rows=list(csv.DictReader(stream))

    def test_exact_fixed_new_seed_list_and_all_arms(self):
        self.assertEqual(len(self.rows),24)
        self.assertEqual([int(r["seed"]) for r in self.rows],self.config["seeds"])
        self.assertEqual(ARMS,tuple(a["key"] for a in self.config["arms"]))

    def test_registered_primary_pass_and_exact_seed_pairs(self):
        for arm,expected in zip(ARMS,EXPECTED):
            self.assertEqual(
                sum(int(r[f"{arm}_success"]) for r in self.rows),
                expected,arm,
            )
        a=[int(r["read_refit_success"]) for r in self.rows]
        b=[int(r["raw_refit_success"]) for r in self.rows]
        aonly=sum(i and not j for i,j in zip(a,b))
        bonly=sum(j and not i for i,j in zip(a,b))
        self.assertEqual((aonly,bonly),(23,0))
        n=aonly+bonly
        p=min(1.0,2*sum(math.comb(n,k)
              for k in range(min(aonly,bonly)+1))/(2**n))
        self.assertEqual(p,2.384185791015625e-7)
        c=self.config["success"]
        self.assertGreaterEqual(sum(a),c["primary_min_successes"])
        self.assertGreaterEqual(aonly-bonly,c["primary_net_seed_wins_min"])
        self.assertLessEqual(p,c["primary_two_sided_McNemar_p_max"])

    def test_23_aligned_seed_perfect_ood_supports_without_refit(self):
        failures=[int(r["seed"]) for r in self.rows if r["read_refit_success"]=="0"]
        self.assertEqual(failures,[1237])
        for row in self.rows:
            if int(row["seed"])==1237:
                continue
            self.assertEqual(int(row["read_nofit_success"]),1)
            for arm in ("read_refit","read_nofit"):
                for f in FAMILIES:
                    self.assertEqual(float(row[f"{arm}_{f}_support"]),1.0)
        self.assertEqual(sum(float(r["read_refit_axis"])>=.9 for r in self.rows),23)

    def test_component_permutation_sham_and_legacy_gate_fail(self):
        for r in self.rows:
            for arm in ("raw_refit","read_shuffled_refit"):
                self.assertEqual(r[f"{arm}_success"],"0")
            for f in FAMILIES:
                self.assertEqual(float(r[f"read_shuffled_refit_{f}_support"]),0.0)
                self.assertEqual(float(r[f"oracle_support_{f}_support"]),1.0)
        self.assertEqual(sum(int(r["stage1_count_topk_success"]) for r in self.rows),2)
        self.assertAlmostEqual(
            statistics.mean(float(r["stage1_count_topk_k2_unseen_support"])
                            for r in self.rows),
            .27474,delta=.0001,
        )
        self.assertAlmostEqual(
            statistics.mean(float(r["raw_refit_k3_unseen_support"])
                            for r in self.rows),
            .015625,delta=.00001,
        )

    def test_source_alignment_and_reconstruction_gap(self):
        ax=[float(r["read_refit_axis"]) for r in self.rows]
        self.assertAlmostEqual(statistics.mean(ax),.991395,delta=.00001)
        failed=next(r for r in self.rows if int(r["seed"])==1237)
        self.assertAlmostEqual(float(failed["read_refit_axis"]),.79375495,places=6)
        for arm,expected in (
            ("read_refit",.000468),
            ("read_nofit",.001059),
            ("raw_refit",.051998),
        ):
            avg=statistics.mean(float(r[f"{arm}_overall_mse"]) for r in self.rows)
            self.assertAlmostEqual(avg,expected,delta=.00001)


if __name__=="__main__":
    unittest.main()
