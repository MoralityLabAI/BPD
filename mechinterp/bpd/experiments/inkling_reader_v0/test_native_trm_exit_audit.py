"""Adversarial tamper fixtures for the development verifier-exit receipt auditor."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mechinterp.bpd.experiments.inkling_reader_v0.audit_native_trm_exit import (
    PINNED_GAOXIN,
    _digest,
    audit,
)


def fixture() -> dict:
    solved = "".join(str((r * 3 + r // 3 + c) % 9 + 1) for r in range(9) for c in range(9))
    question = "." + solved[1:]
    wrong = "1" * 81
    trace = [
        {
            "outer_step": 1, "candidate": wrong,
            "sudoku_checker_passed": False,
            "native_step_seconds": 0.2, "exact_check_seconds": 0.001,
        },
        {
            "outer_step": 2, "candidate": solved,
            "sudoku_checker_passed": True,
            "native_step_seconds": 0.2, "exact_check_seconds": 0.001,
        },
    ]
    entry = {
        "root_id": "dev-1", "question": question, "blank_count": 1,
        "model_sha256": PINNED_GAOXIN,
        "task_digest": _digest({"question": question}),
        "measured_arm_order": ["verifier_exit", "fixed_2"],
        "policies": {
            "fixed_2": {
                "realized_outer_steps": 2, "stop_reason": "verified",
                "elapsed_seconds": 0.45, "trace": trace,
            },
            "verifier_exit": {
                "realized_outer_steps": 2, "stop_reason": "verified",
                "elapsed_seconds": 0.44, "trace": [dict(t) for t in trace],
            },
        },
    }
    summary = {}
    for policy in ("fixed_2", "verifier_exit"):
        row = entry["policies"][policy]
        summary[policy] = {
            "verified_solved": 1, "cases": 1, "outer_steps_total": 2,
            "elapsed_wall_seconds": row["elapsed_seconds"],
            "total_native_step_seconds": 0.4, "total_checker_seconds": 0.002,
            "stop_reasons": {"verified": 1},
        }
    summary["step_count_fraction_saved"] = 0.0
    summary["measured_native_step_fraction_saved"] = 0.0
    return {
        "schema_version": "bpd.native_trm_verifier_exit.v1",
        "evidence_class": "native_pretrained_development",
        "model": "gaoxin492", "model_sha256": PINNED_GAOXIN,
        "split": "development", "student_training_eligible": False,
        "check_uses_planted_solution": False, "max_outer_steps": 2,
        "roots": 1, "holes": 1, "records": [entry], "summary": summary,
    }


class ReceiptAuditTest(unittest.TestCase):
    def audit_data(self, data: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return audit(path)

    def test_independent_valid_solution_and_timing(self) -> None:
        report = self.audit_data(fixture())
        self.assertTrue(report["root_and_checker_replay_passed"])
        self.assertEqual(report["source_roots"], 1)
        self.assertEqual(report["depth_one_verified"], 0)
        self.assertEqual(report["depth_two_verified"], 1)
        self.assertEqual(report["adaptive_steps"], 2)

    def test_reject_mutated_solution_grade(self) -> None:
        x = fixture()
        x["records"][0]["policies"]["fixed_2"]["trace"][1]["sudoku_checker_passed"] = False
        with self.assertRaisesRegex(ValueError, "Sudoku verifier"):
            self.audit_data(x)

    def test_reject_skipping_valid_stop(self) -> None:
        x = fixture()
        x["records"][0]["policies"]["verifier_exit"]["trace"][0]["candidate"] = (
            x["records"][0]["policies"]["verifier_exit"]["trace"][1]["candidate"]
        )
        x["records"][0]["policies"]["verifier_exit"]["trace"][0]["sudoku_checker_passed"] = True
        x["records"][0]["policies"]["fixed_2"]["trace"][0]["candidate"] = (
            x["records"][0]["policies"]["fixed_2"]["trace"][1]["candidate"]
        )
        x["records"][0]["policies"]["fixed_2"]["trace"][0]["sudoku_checker_passed"] = True
        with self.assertRaisesRegex(ValueError, "kept computing"):
            self.audit_data(x)

    def test_reject_tampered_step_totals(self) -> None:
        x = fixture()
        x["summary"]["verifier_exit"]["outer_steps_total"] = 1
        with self.assertRaisesRegex(ValueError, "summary mismatch"):
            self.audit_data(x)

    def test_reject_bad_model(self) -> None:
        x = fixture()
        x["model_sha256"] = hashlib.sha256(b"wrong").hexdigest()
        with self.assertRaisesRegex(ValueError, "wrong model"):
            self.audit_data(x)


if __name__ == "__main__":
    unittest.main()
