"""Independent stdlib replay of PD11's published seed-level result.

Original 144-arm raw JSON: GitHub Actions 44130218981,
artifact 14350115809, SHA256
4d2b7a75b2f917254c141a828253557f801552cf8f655c7ed0bd4323d91a5df3.

This source-only suite does not train any model or regenerate heldout data.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parent


class TestPublishedPD11(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "PER_SEED_20261118.csv").open(
            newline="", encoding="utf-8"
        ) as f:
            cls.rows = list(csv.DictReader(f))
        cls.config = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))

    def test_all_24_fresh_seed_rotation_pairs(self):
        rows = self.rows
        self.assertEqual(len(rows), 24)
        self.assertEqual(
            [int(x["seed"]) for x in rows],
            self.config["seed_list"],
        )
        self.assertEqual(len({int(x["seed"]) for x in rows}), 24)
        self.assertNotIn(self.config["smoke_seed"], self.config["seed_list"])
        self.assertEqual(self.config["objective"]["angles_per_pair"], 73)
        self.assertEqual(self.config["objective"]["sweep_count"], 6)
        self.assertEqual(len(self.config["objective"]["ordered_pairs"]), 6)

    def test_registered_success_and_exact_paired_mcnemar(self):
        rows = self.rows
        counts = {
            name: sum(int(x[name]) for x in rows)
            for name in (
                "global_success", "local_success",
                "stress_global_success", "stress_local_success",
                "random_success", "supervised_success",
            )
        }
        self.assertEqual(counts, {
            "global_success": 24, "local_success": 1,
            "stress_global_success": 24, "stress_local_success": 0,
            "random_success": 0, "supervised_success": 22,
        })
        glob = [int(x["global_success"]) for x in rows]
        local = [int(x["local_success"]) for x in rows]
        a_only = sum(a and not b for a, b in zip(glob, local))
        b_only = sum(b and not a for a, b in zip(glob, local))
        self.assertEqual((a_only, b_only), (23, 0))
        n = a_only + b_only
        p = min(1., 2 * sum(math.comb(n, k)
                            for k in range(min(a_only, b_only)+1)) / 2**n)
        self.assertEqual(p, 2.384185791015625e-7)
        config = self.config["scoring"]
        self.assertGreaterEqual(
            counts["global_success"], config["primary_successes_min"]
        )
        self.assertGreaterEqual(
            a_only-b_only, config["primary_net_paired_seed_wins_min"]
        )
        self.assertLessEqual(p, config["primary_two_sided_McNemar_p_max"])

    def test_every_global_seed_generalizes_all_unseen_supports(self):
        for row in self.rows:
            self.assertEqual(int(row["global_success"]), 1)
            for fam in ("global_ood_pair", "global_ood_triple",
                        "global_ood_transition"):
                self.assertEqual(float(row[fam]), 1., (row["seed"], fam))
            self.assertGreater(float(row["global_min_cosine"]), .99333)
            self.assertGreater(float(row["global_mass_gain"]), 0)
            self.assertGreater(int(row["global_update_count"]), 0)
            self.assertGreaterEqual(float(row["global_read_cosine"]), .90)

    def test_local_failure_preserved_with_low_support(self):
        rows = self.rows
        self.assertEqual(
            [int(x["seed"]) for x in rows if int(x["local_success"])],
            [1723],
        )
        def avg(key):
            return statistics.mean(float(x[key]) for x in rows)
        self.assertAlmostEqual(avg("global_read_cosine"), .99978818, places=6)
        self.assertAlmostEqual(avg("local_read_cosine"), .78286542, places=6)
        self.assertAlmostEqual(avg("global_mse"), .00012428, places=6)
        self.assertAlmostEqual(avg("local_mse"), .01396859, places=6)
        self.assertAlmostEqual(avg("local_ood_pair"), .04578993, places=5)
        self.assertAlmostEqual(avg("local_ood_triple"), .06618924, places=5)
        self.assertGreater(avg("global_mass_gain"), avg("local_mass_gain"))
        self.assertTrue(all(int(x["local_update_count"]) > 0 for x in rows))

    def test_privileged_controls_never_relabelled_unlabelled(self):
        arms = {a["key"]: a for a in self.config["arms"]}
        for name in ("global_givens", "local_givens", "random_orthogonal"):
            self.assertFalse(arms[name]["uses_true_rotation_in_initialization"])
            self.assertFalse(arms[name]["uses_count_labels"])
        for name in ("global_pairmix_privileged", "local_pairmix_privileged"):
            self.assertTrue(arms[name]["uses_true_rotation_in_initialization"])
        self.assertTrue(arms["count_supervised_positive"]["uses_count_labels"])


if __name__ == "__main__":
    unittest.main()
