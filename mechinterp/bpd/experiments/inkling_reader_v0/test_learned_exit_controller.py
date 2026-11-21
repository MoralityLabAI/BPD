"""Separate TRAIN/CAL roots and real recurrent-state policy-path controls."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from mechinterp.bpd.adapters.trm import build_mini_trm
from mechinterp.bpd.experiments.inkling_reader_v0.learned_exit_controller import (
    FEATURES, _fixed_rows, _logit_model, calibrate, evaluate, fit_logistic,
)
from mechinterp.bpd.experiments.inkling_reader_v0.native_trm_exit import native_rollout
from mechinterp.bpd.experiments.inkling_reader_v0.native_trm_pilot import _permute_sudoku


def fake_rows(split: str = "train") -> list[dict]:
    return [
        {
            "root_id": f"{split}-{i}", "split": split,
            "features": [float(i), i / 2, i / 3, i / 4, 2 / (i + 1), i / 10],
            "first_pass_verified": i >= 6,
        }
        for i in range(12)
    ]


class LearnedExitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        cls.model = build_mini_trm(
            hidden_size=32, num_heads=4, l_layers=1, h_cycles=2,
            l_cycles=1, puzzle_emb_len=4, seq_len=81, seed=43,
        )

    def test_training_calibration_and_feature_contract(self) -> None:
        params = fit_logistic(fake_rows(), epochs=200)
        self.assertEqual(len(params["weights"]), len(FEATURES))
        self.assertGreater(params["train_accuracy_at_half"], 0.8)
        cal = calibrate(fake_rows("calibration"), params)
        self.assertEqual(cal["false_stops"], 0)
        self.assertGreater(cal["stop_count"], 0)
        self.assertLessEqual(cal["stop_count"], cal["calibration_valid_first"])
        scores = _logit_model(np.array([r["features"] for r in fake_rows()]), params)
        self.assertGreater(scores[-1], scores[0])

    def test_split_custody_prohibits_training_on_dev(self) -> None:
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            fit_logistic(fake_rows("development"))
        with self.assertRaisesRegex(ValueError, "CAL"):
            calibrate(fake_rows("train"), {
                "weights": [0.0] * len(FEATURES), "bias": 0.0,
                "mean": [0.0] * len(FEATURES), "scale": [1.0] * len(FEATURES),
            })
        duplicated = fake_rows()
        duplicated[1]["root_id"] = duplicated[0]["root_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fit_logistic(duplicated)

    def test_hidden_checker_not_available_to_callback(self) -> None:
        q, _ = _permute_sudoku(3333, holes=22)
        features_seen = []
        fixed, _ = native_rollout(self.model, q, max_outer_steps=2, stop_when_verified=False)
        stopped, receipt = native_rollout(
            self.model, q, max_outer_steps=2, stop_when_verified=False,
            stop_decider=lambda features: (features_seen.append(dict(features)), True)[1],
        )
        self.assertEqual(len(stopped), 1)
        self.assertEqual(stopped[0]["candidate"], fixed[0]["candidate"])
        self.assertEqual(receipt["stop_reason"], "learned_exit")
        self.assertEqual(len(features_seen), 1)
        self.assertEqual(set(features_seen[0]), set(FEATURES))
        self.assertTrue(all(math.isfinite(v) for v in features_seen[0].values()))
        with self.assertRaisesRegex(ValueError, "separate"):
            native_rollout(
                self.model, q, max_outer_steps=2, stop_when_verified=True,
                stop_decider=lambda _: True,
            )

    def test_real_mini_forward_paired_policy_traces(self) -> None:
        sha = "0" * 64
        rows = _fixed_rows(self.model, split="development", n=4, holes=22, sha=sha)
        params = {
            "weights": [0.0] * len(FEATURES), "bias": 0.0,
            "mean": [0.0] * len(FEATURES), "scale": [1.0] * len(FEATURES),
        }
        results, details = evaluate(
            self.model, rows, params,
            {"threshold": 1.000001, "random_stop_probability": 0.0},
        )
        self.assertEqual(results["independent_root_count"], 4)
        self.assertEqual(results["scores"]["fixed2"]["total_outer_steps"], 8)
        self.assertEqual(results["scores"]["learned"]["total_outer_steps"], 8)
        self.assertEqual(len(details), 4)
        self.assertTrue(all(not r["training_eligible"] for r in rows))
        self.assertTrue(all(len(r["policies"]) == 4 for r in details))


if __name__ == "__main__":
    unittest.main()
