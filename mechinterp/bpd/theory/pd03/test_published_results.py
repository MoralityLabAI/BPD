"""Independent stdlib source-only replay of PD03's published first-run CSV.

The original full 168-arm JSON is immutable in GitHub Actions artifact
14321589616, run 44070377352. This test checks the compact per-seed
extraction separately from neural-model execution.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import math
import statistics
import unittest

ROOT = Path(__file__).resolve().parent


class TestPublishedPD03(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = list(csv.DictReader(
            (ROOT / "PER_SEED_20261117.csv").open(encoding="utf-8", newline="")
        ))

    def test_complete_new_seed_set(self):
        expected = [
            269,271,277,281,283,293,307,311,313,317,331,337,
            347,349,353,359,367,373,379,383,389,397,401,409,
        ]
        self.assertEqual([int(r["seed"]) for r in self.rows], expected)

    def test_failure_is_preserved_with_exact_paired_statistic(self):
        top1 = [int(r["top1_success"]) for r in self.rows]
        margin = [int(r["margin_success"]) for r in self.rows]
        hard = [int(r["hard_success"]) for r in self.rows]
        self.assertEqual(sum(top1), 24)
        self.assertEqual(sum(margin), 19)
        self.assertEqual(sum(hard), 3)
        top1_only = sum(x and not y for x, y in zip(top1, margin))
        margin_only = sum(y and not x for x, y in zip(top1, margin))
        self.assertEqual((top1_only, margin_only), (5, 0))
        n = top1_only + margin_only
        p = min(1.0, 2 * sum(
            math.comb(n, k) for k in range(min(top1_only, margin_only) + 1)
        ) / (2 ** n))
        self.assertEqual(p, 0.0625)
        self.assertFalse(top1_only - margin_only >= 6 and p <= 0.05)

    def test_recovery_individual_thresholds(self):
        for row in self.rows:
            for key in ("top1", "margin"):
                expected = (
                    float(row[f"{key}_mse"]) <= .02
                    and float(row[f"{key}_axis"]) >= .90
                    and float(row[f"{key}_route"]) >= .95
                )
                self.assertEqual(int(expected), int(row[f"{key}_success"]), row["seed"])
        self.assertEqual(
            [int(row["seed"]) for row in self.rows if row["margin_success"] == "0"],
            [277,293,307,317,397],
        )

    def test_posthoc_gate_error_direction_and_thresholds(self):
        self.assertEqual(
            sum(int(r["margin_false_off"]) for r in self.rows), 0
        )
        self.assertEqual(
            sum(int(r["margin_false_on"]) for r in self.rows), 2048
        )
        self.assertEqual(Counter(r["margin_threshold"] for r in self.rows),
                         Counter({"0.45":21,"0.5":3}))
        self.assertTrue(all(float(r["top1_route"]) == 1 for r in self.rows))
        self.assertEqual(
            statistics.mean(float(r["margin_route"]) for r in self.rows),
            19/24
        )
        self.assertAlmostEqual(
            statistics.mean(float(r["top1_mse"]) for r in self.rows),
            1.3272794337252739e-05, places=9,
        )


if __name__ == "__main__":
    unittest.main()
