"""Posthoc integrity checks for immutable PD02 per-seed extract.

The receipt is the locked GitHub Actions JSON artifact. These stdlib tests
recompute registered metrics from its separately published per-seed CSV.
They are not additional training and do not change the first-run verdict.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parent


class TestPublishedPD02(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = list(csv.DictReader((ROOT / "PER_SEED_20261117.csv").open()))
        cls.config = json.loads((ROOT / "CONFIG.json").read_text())

    def test_pinned_24_seeds_and_exact_success_counts(self):
        rows = self.rows
        self.assertEqual(len(rows), 24)
        self.assertEqual(
            [int(r["seed"]) for r in rows],
            self.config["seeds"],
        )
        self.assertEqual(sum(int(r["hard_success"]) for r in rows), 6)
        self.assertEqual(sum(int(r["soft_success"]) for r in rows), 10)
        hard_only = sum(r["hard_success"] == "1" and r["soft_success"] == "0" for r in rows)
        soft_only = sum(r["hard_success"] == "0" and r["soft_success"] == "1" for r in rows)
        self.assertEqual((hard_only, soft_only), (3, 7))
        n = hard_only + soft_only
        p = min(1., 2 * sum(math.comb(n, k) for k in range(min(hard_only, soft_only)+1)) / 2**n)
        self.assertEqual(p, .34375)

    def test_registered_component_thresholds_remain_failed(self):
        rows = self.rows
        for a in ("hard", "soft"):
            recovered = 0
            for r in rows:
                condition = (float(r[a + "_mse"]) <= .02
                             and float(r[a + "_axis_cos"]) >= .9
                             and float(r[a + "_route_fraction"]) >= .95)
                self.assertEqual(condition, r[a + "_success"] == "1", r["seed"])
                recovered += condition
            self.assertEqual(recovered, 6 if a == "hard" else 10)
        self.assertLess(6, self.config["primary"]["hard_success_min"])
        self.assertLess(6 - 10, self.config["primary"]["net_paired_success_wins_min"])

    def test_measured_gate_activity_and_error_breakdown(self):
        rows = self.rows
        self.assertEqual(sum(float(r["hard_axis_cos"]) >= .9 for r in rows), 24)
        self.assertEqual(sum(float(r["soft_axis_cos"]) >= .9 for r in rows), 19)
        self.assertEqual(sum(float(r["hard_mse"]) <= .02 for r in rows), 20)
        self.assertEqual(sum(float(r["soft_mse"]) <= .02 for r in rows), 24)
        for key, target in (
            ("hard_mse", .0127498123), ("soft_mse", .0081411199),
            ("hard_axis_cos", .9992459863), ("soft_axis_cos", .9433755005),
            ("hard_route_fraction", .8509114583),
            ("soft_route_fraction", .4166666667),
            ("hard_gate_fraction", .4797363281),
            ("soft_gate_fraction", .7003987630),
        ):
            self.assertAlmostEqual(statistics.mean(float(r[key]) for r in rows), target, places=6)
        # 24 seeds * 128 boards * 4 passes * 2 gate candidates.
        actual_on = sum(round(1024 * float(r["hard_gate_fraction"])) for r in rows)
        self.assertEqual(actual_on, 11790)
        self.assertEqual(24 * 128 * 4 - actual_on, 498)


if __name__ == "__main__":
    unittest.main()
