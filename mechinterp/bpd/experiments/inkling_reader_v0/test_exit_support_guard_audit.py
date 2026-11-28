"""Independent verifier guard-audit smoke and sealed-tamper checks."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mechinterp.bpd.adapters.trm import build_mini_trm
from mechinterp.bpd.experiments.inkling_reader_v0 import audit_exit_support_guard as auditor
from mechinterp.bpd.experiments.inkling_reader_v0.exit_support_guard import run


class GuardReceiptReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        cls.model = build_mini_trm(
            hidden_size=32, num_heads=4, l_layers=1, h_cycles=2,
            l_cycles=1, puzzle_emb_len=4, seq_len=81, seed=43,
        )
        cls.frozen = {
            "model_sha256": auditor.MODEL_SHA,
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

    def _write(self, base: Path) -> tuple[Path, Path, str]:
        original_path = base / "frozen.json"
        original_path.write_text(json.dumps(self.frozen, sort_keys=True) + "\n")
        digest = hashlib.sha256(original_path.read_bytes()).hexdigest()
        report, cohorts = run(self.model, self.frozen, set(), roots_per_panel=2)
        report["frozen_controller_report_sha256"] = digest
        out = base / "source"
        out.mkdir()
        hashes = {}
        for holes, rows in cohorts.items():
            target = out / f"guard_h{holes}.jsonl"
            target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            hashes[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
        report["files_sha256"] = hashes
        (out / "report.json").write_text(json.dumps(report, sort_keys=True) + "\n")
        return out, original_path, digest

    def test_replay_all_logged_policies_and_raw_sudoku(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            out, original, digest = self._write(Path(temp))
            with patch.object(auditor, "FROZEN_SHA", digest):
                report = auditor.audit(out, original)
            self.assertEqual(report["unique_source_roots"], 4)
            self.assertTrue(report["independent_exact_checker_replay"])
            self.assertTrue(report["learned_stop_policy_replay"])
            self.assertEqual(sorted(item["holes"] for item in report["panels"]), [40, 52])

    def test_mutated_classifier_score_detected_even_if_resealed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            out, original, digest = self._write(Path(temp))
            filename = out / "guard_h40.jsonl"
            data = [json.loads(line) for line in filename.read_text().splitlines()]
            data[0]["frozen_controller_score"] = 0.03
            data[0]["reference_learned_score"] = 0.03
            filename.write_text("".join(json.dumps(v, sort_keys=True) + "\n" for v in data))
            report_path = out / "report.json"
            manifest = json.loads(report_path.read_text())
            manifest["files_sha256"][filename.name] = hashlib.sha256(filename.read_bytes()).hexdigest()
            report_path.write_text(json.dumps(manifest) + "\n")
            with patch.object(auditor, "FROZEN_SHA", digest):
                with self.assertRaisesRegex(ValueError, "classifier score"):
                    auditor.audit(out, original)

    def test_mutated_candidate_detected_even_if_resealed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            out, original, digest = self._write(Path(temp))
            filename = out / "guard_h52.jsonl"
            data = [json.loads(line) for line in filename.read_text().splitlines()]
            data[0]["first_pass_candidate"] = "1" * 81
            filename.write_text("".join(json.dumps(v, sort_keys=True) + "\n" for v in data))
            report_path = out / "report.json"
            manifest = json.loads(report_path.read_text())
            manifest["files_sha256"][filename.name] = hashlib.sha256(filename.read_bytes()).hexdigest()
            report_path.write_text(json.dumps(manifest) + "\n")
            with patch.object(auditor, "FROZEN_SHA", digest):
                with self.assertRaisesRegex(ValueError, "output disagrees"):
                    auditor.audit(out, original)


if __name__ == "__main__":
    unittest.main()
