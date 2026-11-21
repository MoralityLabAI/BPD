"""CPU-only tests of real TRM blocks with randomly initialized mini weights."""

from __future__ import annotations

import hashlib
import json
import math
import unittest

import torch

from mechinterp.bpd.adapters.trm import build_mini_trm
from mechinterp.bpd.experiments.inkling_reader_v0.native_trm_pilot import (
    ARMS, _permute_sudoku, run_campaign, run_one, verify_clean_parity, verify_sudoku,
)


class NativeTRMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        cls.model = build_mini_trm(
            hidden_size=32, num_heads=4, l_layers=1, h_cycles=2,
            l_cycles=1, puzzle_emb_len=4, seq_len=81, seed=9,
        )

    def test_generator_validity_and_root_change(self) -> None:
        q, target = _permute_sudoku(20261121)
        self.assertEqual(len(q), 81)
        self.assertEqual(q.count("."), 22)
        self.assertTrue(verify_sudoku(q, target))
        self.assertFalse(verify_sudoku(q, target[:-1] + "0"))
        q2, target2 = _permute_sudoku(20261122)
        self.assertNotEqual(target, target2)
        self.assertNotEqual(q, q2)

    def test_native_forward_parity(self) -> None:
        q, _ = _permute_sudoku(123)
        self.assertLessEqual(verify_clean_parity(self.model, q), 1e-5)

    def test_patching_changes_real_hidden_state(self) -> None:
        q, _ = _permute_sudoku(543)
        with torch.no_grad():
            base, clean = run_one(self.model, q, arm="clean")
            zero, zero_row = run_one(self.model, q, arm="zero")
            head, readout = run_one(self.model, q, arm="head_direction")
            other, sham = run_one(self.model, q, arm="random_direction", root_seed=3)
        self.assertTrue(torch.equal(base, zero))
        self.assertEqual(clean["actual_raw_l2"], 0.0)
        self.assertEqual(zero_row["actual_raw_l2"], 0.0)
        self.assertTrue(math.isclose(
            readout["actual_raw_l2"], readout["nominal_raw_l2"], rel_tol=1e-5,
        ))
        self.assertTrue(math.isclose(
            sham["actual_raw_l2"], sham["nominal_raw_l2"], rel_tol=1e-5,
        ))
        self.assertGreater(float((head - base).abs().sum()), 0)
        self.assertGreater(float((other - base).abs().sum()), 0)
        self.assertEqual(readout["site"], "z_H_after_h_cycle")

    def test_untrained_campaign_marked_smoke(self) -> None:
        checkpoint_sha = hashlib.sha256(b"mini-test").hexdigest()
        result = run_campaign(
            self.model, model_id="mini-test", model_sha=checkpoint_sha,
            pretrained=False, roots=2, seed=20261121,
        )
        self.assertEqual(result["evidence_class"], "untrained_native_plumbing_smoke")
        self.assertEqual(len(result["rows"]), 2 * len(ARMS))
        self.assertEqual(result["max_clean_parity_difference"], 0.0)
        self.assertTrue(result["all_sudoku_targets_verified"])
        self.assertTrue(all(x["experiment_split"] == "development" for x in result["rows"]))
        self.assertTrue(all(x["root_id"] and x["model_sha256"] for x in result["rows"]))
        self.assertTrue(all(x["target_matches_verified_sudoku"] for x in result["rows"]))

    def test_invalid_patch_limits(self) -> None:
        q, _ = _permute_sudoku(1)
        with self.assertRaisesRegex(ValueError, "outer_steps"):
            run_one(self.model, q, outer_steps=0)
        with self.assertRaisesRegex(ValueError, "subsequent H-cycle"):
            run_one(self.model, q, patch_h_cycle=self.model.h_cycles)
        with self.assertRaisesRegex(ValueError, "relative_dose"):
            run_one(self.model, q, relative_dose=float("nan"))
        with self.assertRaisesRegex(ValueError, "blank"):
            run_one(self.model, q.replace(".", "1"))


if __name__ == "__main__":
    unittest.main()
