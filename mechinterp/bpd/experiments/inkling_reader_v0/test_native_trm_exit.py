"""Test the actual recurrent-state continuation contract without downloading weights."""

from __future__ import annotations

import hashlib
import math
import unittest

import torch

from mechinterp.bpd.adapters.trm import build_mini_trm
from mechinterp.bpd.experiments.inkling_reader_v0.native_trm_exit import native_rollout, run
from mechinterp.bpd.experiments.inkling_reader_v0.native_trm_pilot import _permute_sudoku


class ResumableNativeExitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        cls.model = build_mini_trm(
            hidden_size=32, num_heads=4, l_layers=1,
            h_cycles=2, l_cycles=1, puzzle_emb_len=4, seq_len=81, seed=43,
        )

    def test_fixed_and_verifier_arms_replay(self) -> None:
        question, _ = _permute_sudoku(20261123, holes=22)
        fixed, fixed_result = native_rollout(
            self.model, question, max_outer_steps=2, stop_when_verified=False,
        )
        conditional, conditional_result = native_rollout(
            self.model, question, max_outer_steps=2, stop_when_verified=True,
        )
        self.assertEqual(len(fixed), 2)
        self.assertTrue(1 <= len(conditional) <= 2)
        self.assertEqual(fixed[0]["candidate"], conditional[0]["candidate"])
        self.assertEqual(
            fixed[len(conditional) - 1]["candidate"], conditional[-1]["candidate"],
        )
        self.assertEqual(fixed_result["realized_outer_steps"], 2)
        self.assertEqual(conditional_result["realized_outer_steps"], len(conditional))
        self.assertGreaterEqual(fixed_result["elapsed_seconds"], 0)
        self.assertTrue(all(len(row["candidate"]) == 81 for row in fixed))

    def test_development_run_and_handoff_boundary(self) -> None:
        result = run(
            self.model, model_id="mini", model_sha=hashlib.sha256(b"fixture").hexdigest(),
            pretrained=False, seed=20261123, holes=22, roots=2,
        )
        self.assertEqual(result["evidence_class"], "untrained_plumbing_smoke")
        self.assertEqual(result["roots"], 2)
        self.assertFalse(result["student_training_eligible"])
        self.assertFalse(result["check_uses_planted_solution"])
        self.assertEqual(result["summary"]["fixed_2"]["outer_steps_total"], 4)
        self.assertLessEqual(result["summary"]["verifier_exit"]["outer_steps_total"], 4)
        self.assertTrue(math.isfinite(result["summary"]["step_count_fraction_saved"]))
        self.assertTrue(all(
            row["source_split"] == "development" for row in result["records"]
        ))

    def test_limits_reject_invalid_settings(self) -> None:
        question, _ = _permute_sudoku(1, holes=22)
        with self.assertRaisesRegex(ValueError, "max_outer_steps"):
            native_rollout(self.model, question, max_outer_steps=9, stop_when_verified=True)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            native_rollout(self.model, question.replace(".", "1"),
                           max_outer_steps=2, stop_when_verified=False)
        with self.assertRaisesRegex(ValueError, "wall"):
            run(self.model, model_id="mini", model_sha="x" * 64,
                pretrained=False, max_wall_seconds=-1)


if __name__ == "__main__":
    unittest.main()
