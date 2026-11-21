"""Synthetic custody fixtures for train-only DataGoblin RLT export."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from export_datagoblin import MODEL_SHA, _digest, _hash, export


def _solution(index: int) -> str:
    return "".join(
        str(((r * 3 + r // 3 + c + index) % 9) + 1)
        for r in range(9) for c in range(9)
    )


def _row(split: str, index: int, first_valid: bool) -> dict:
    solved = _solution(index)
    q = "." + solved[1:]
    return {
        "schema_version": "bpd.trm_exit_training_row.v1",
        "root_id": f"{split}-sudoku-root-{index}",
        "task_digest": _digest({"question": q}),
        "model_sha256": MODEL_SHA, "split": split, "question": q,
        "features": [1.0 + index] * 6,
        "first_pass_verified": first_valid,
        "second_pass_verified": True,
        "second_pass_rescue": not first_valid,
        "first_candidate": solved if first_valid else "1" * 81,
        "second_candidate": solved,
        "training_eligible": split == "train",
        "oracle_target_visible_to_controller": False,
        "future_step_visible_to_controller": False,
    }


def make_source(root: Path) -> Path:
    source = root / "model_run"
    source.mkdir(parents=True)
    rows = {
        "train": [_row("train", 0, True), _row("train", 1, False)],
        "calibration": [_row("calibration", 2, True), _row("calibration", 3, True)],
        "development": [_row("development", 4, False), _row("development", 5, False)],
        "evaluation": [{"root_id": "development-4"}, {"root_id": "development-5"}],
    }
    digests = {}
    for split, values in rows.items():
        path = source / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(v, sort_keys=True, separators=(",", ":")) + "\n" for v in values),
            encoding="utf-8",
        )
        digests[path.name] = _hash(path)
    report = {
        "schema_version": "bpd.trm_learned_exit.v1",
        "model_sha256": MODEL_SHA, "evidence_class": "native_pretrained_development_learned_exit",
        "root_counts": {"train": 2, "calibration": 2, "development": 2},
        "feature_names": [
            "mean_blank_logit_margin", "min_blank_logit_margin",
            "p10_blank_logit_margin", "median_blank_logit_margin",
            "mean_blank_entropy", "mean_blank_top_probability",
        ],
        "data_sha256": digests,
    }
    (source / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return source


def reseal(source: Path) -> None:
    report = json.loads((source / "report.json").read_text())
    for split in ("train", "calibration", "development", "evaluation"):
        name = f"{split}.jsonl"
        report["data_sha256"][name] = _hash(source / name)
    (source / "report.json").write_text(json.dumps(report), encoding="utf-8")


class TrainExportTests(unittest.TestCase):
    def test_train_only_export_and_verified_repairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = make_source(folder)
            result = export(source, folder / "out")
            self.assertEqual(result["training_root_count"], 2)
            self.assertEqual(result["verified_repair_pairs"], 1)
            self.assertEqual(result["development_rows_exported"], 0)
            records = [
                json.loads(line) for line in (folder / "out" / "controller_sft.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(
                [r["target"]["decision"] for r in records], ["stop", "continue"],
            )
            self.assertTrue(all(r["training_eligible"] and r["split"] == "train" for r in records))
            content = (folder / "out" / "controller_sft.jsonl").read_text()
            self.assertNotIn("development-sudoku-root", content)

    def test_reject_altered_verifier_label_despite_resealed_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = make_source(folder)
            path = source / "train.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["first_pass_verified"] = False
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            reseal(source)
            with self.assertRaisesRegex(ValueError, "verifier label"):
                export(source, folder / "out")

    def test_reject_cross_split_duplicates_and_changed_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = make_source(folder)
            path = source / "calibration.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["question"] = _solution(0)[0:1].replace("1", ".") + _solution(0)[1:]
            rows[0]["task_digest"] = _digest({"question": rows[0]["question"]})
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                export(source, folder / "out")
            reseal(source)
            with self.assertRaisesRegex(ValueError, "lineage"):
                export(source, folder / "out")


if __name__ == "__main__":
    unittest.main()
