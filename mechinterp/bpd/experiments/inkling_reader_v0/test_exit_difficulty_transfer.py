"""Train/cal frozen boundary and no-checker learned stopping on fresh shifts."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from mechinterp.bpd.adapters.trm import build_mini_trm
from mechinterp.bpd.experiments.inkling_reader_v0.exit_difficulty_transfer import (
    MODEL_SHA256, _add_fixed_one, generate_verified_rows, run_transfer,
)
from mechinterp.bpd.experiments.inkling_reader_v0.native_trm_pilot import (
    _permute_sudoku, count_solutions, verify_sudoku,
)


class FrozenTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        cls.model = build_mini_trm(
            hidden_size=32, num_heads=4, l_layers=1,
            h_cycles=2, l_cycles=1, puzzle_emb_len=4, seq_len=81, seed=43,
        )

    def test_predeclared_difficulties_have_unique_solutions(self) -> None:
        for holes in (40, 52):
            question, solved = _permute_sudoku(20261311 + holes, holes=holes)
            self.assertTrue(verify_sudoku(question, solved))
            self.assertEqual(question.count("."), holes)
            self.assertEqual(count_solutions(question, limit=2), 1)

    def test_shift_records_use_development_and_new_roots(self) -> None:
        rows = generate_verified_rows(
            self.model, sha256=MODEL_SHA256, holes=40,
            seed=20261311, roots=2, prohibited=set(),
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["split"] == "development" and not r["training_eligible"] for r in rows))
        self.assertTrue(all(r["root_id"].startswith("transfer-h40") for r in rows))
        self.assertEqual(len({r["task_digest"] for r in rows}), len(rows))
        with self.assertRaisesRegex(ValueError, "repeated source"):
            generate_verified_rows(
                self.model, sha256=MODEL_SHA256, holes=40,
                seed=20261311, roots=2, prohibited={rows[0]["task_digest"]},
            )

    def test_run_mini_transfer_with_fixed_controller(self) -> None:
        fixed = {
            "model_sha256": MODEL_SHA256,
            "training": {
                "features": [
                    "mean_blank_logit_margin", "min_blank_logit_margin",
                    "p10_blank_logit_margin", "median_blank_logit_margin",
                    "mean_blank_entropy", "mean_blank_top_probability",
                ],
                "weights": [0.0] * 6, "mean": [0.0] * 6,
                "scale": [1.0] * 6, "bias": 0.0,
            },
            "calibration": {"threshold": 1.000001, "random_stop_probability": 0.0},
        }
        report, data = run_transfer(
            self.model, fixed, set(), roots_per_panel=2,
            cohorts=((40, 20261311), (52, 20261312)),
        )
        self.assertEqual(report["source_roots_per_panel"], 2)
        self.assertEqual(sorted(data), [40, 52])
        for c in report["cohorts"]:
            self.assertEqual(c["results"]["scores"]["fixed2"]["total_outer_steps"], 4)
            self.assertEqual(c["results"]["scores"]["learned"]["total_outer_steps"], 4)
            self.assertEqual(c["results"]["scores"]["random_matched"]["total_outer_steps"], 4)
            self.assertEqual(c["results"]["scores"]["fixed1"]["total_outer_steps"], 2)
        self.assertTrue(all(not r["training_eligible"] for values in data.values() for r in values))

    def test_bad_threshold_and_bad_hole_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "difficulty"):
            generate_verified_rows(
                self.model, sha256=MODEL_SHA256, holes=25,
                seed=20261311, roots=2, prohibited=set(),
            )
        with self.assertRaisesRegex(ValueError, "controller"):
            run_transfer(
                self.model, {"model_sha256": MODEL_SHA256,
                             "calibration": {"threshold": -1.0}}, set(),
                roots_per_panel=2, cohorts=((40, 20261311),),
            )


if __name__ == "__main__":
    unittest.main()
