"""Posthoc support guard keeps harder tasks at fixed depth on new source roots."""

from __future__ import annotations

import unittest

import torch

from mechinterp.bpd.adapters.trm import build_mini_trm
from mechinterp.bpd.experiments.inkling_reader_v0.exit_difficulty_transfer import MODEL_SHA256
from mechinterp.bpd.experiments.inkling_reader_v0.exit_support_guard import run


class DifficultyGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        cls.model = build_mini_trm(
            hidden_size=32, num_heads=4, l_layers=1, h_cycles=2,
            l_cycles=1, puzzle_emb_len=4, seq_len=81, seed=43,
        )
        cls.frozen = {
            "model_sha256": MODEL_SHA256,
            "training": {
                "features": [
                    "mean_blank_logit_margin", "min_blank_logit_margin",
                    "p10_blank_logit_margin", "median_blank_logit_margin",
                    "mean_blank_entropy", "mean_blank_top_probability",
                ],
                "weights": [0.0] * 6, "mean": [0.0] * 6,
                "scale": [1.0] * 6, "bias": 20.0,
            },
            "calibration": {
                "threshold": 0.5656608144205763,
                "random_stop_probability": 0.4583333333333333,
            },
        }

    def test_fixed_support_cutoff_is_applied_online(self) -> None:
        report, rows = run(
            self.model, self.frozen, set(),
            roots_per_panel=2,
        )
        self.assertEqual(sorted(rows), [40, 52])
        per = {p["holes"]: p for p in report["panels"]}
        self.assertEqual(per[40]["support_guarded"]["outer_steps"], 2)
        self.assertEqual(per[52]["support_guarded"]["outer_steps"], 4)
        self.assertEqual(per[52]["support_guarded"]["incorrect_early_exits"], 0)
        self.assertEqual(per[52]["original_frozen_learned"]["total_outer_steps"], 2)
        self.assertEqual(report["guard_rule_frozen_before_new_sources"], "holes <= 50 else fixed_2")
        self.assertTrue(all(not row["training_eligible"] for group in rows.values() for row in group))

    def test_old_seeds_and_missing_cohorts_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "both predeclared"):
            run(
                self.model, self.frozen, set(), roots_per_panel=2,
                cohorts=((40, 20261411),),
            )
        with self.assertRaisesRegex(ValueError, "reuse"):
            run(
                self.model, self.frozen, set(), roots_per_panel=2,
                cohorts=((40, 20261311), (52, 20261312)),
            )


if __name__ == "__main__":
    unittest.main()
