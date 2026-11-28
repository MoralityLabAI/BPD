"""Export verified, lineage-disjoint TRAIN controller rollouts into DataGoblin format.

Do not export calibration or development labels. The exported targets are
Sudoku-verifier-backed stop/continue decisions, not hidden teacher thoughts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SOURCE_SCHEMA = "bpd.trm_learned_exit.v1"
SOURCE_ROW = "bpd.trm_exit_training_row.v1"
MODEL_SHA = "932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.stat().st_size > 24_000_000:
        raise ValueError("oversized training corpus")
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected JSON object")
                rows.append(row)
    return rows


def _sudoku_valid(question: str, candidate: str) -> bool:
    if not isinstance(question, str) or not isinstance(candidate, str):
        return False
    if len(question) != 81 or len(candidate) != 81:
        return False
    if any(v not in "123456789." for v in question):
        return False
    if any(v not in "123456789" for v in candidate):
        return False
    if any(v != "." and candidate[i] != v for i, v in enumerate(question)):
        return False
    for r in range(9):
        if {candidate[r * 9 + c] for c in range(9)} != set("123456789"):
            return False
    for c in range(9):
        if {candidate[r * 9 + c] for r in range(9)} != set("123456789"):
            return False
    for r in (0, 3, 6):
        for c in (0, 3, 6):
            if {
                candidate[(r + dr) * 9 + c + dc]
                for dr in range(3) for dc in range(3)
            } != set("123456789"):
                return False
    return True


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def export(source: Path, out: Path) -> dict[str, Any]:
    manifest_path = source / "report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SOURCE_SCHEMA
        or manifest.get("model_sha256") != MODEL_SHA
        or manifest.get("evidence_class") != "native_pretrained_development_learned_exit"
    ):
        raise ValueError("requires real, pinned pretrained model campaign")
    all_rows = {}
    all_roots = set()
    all_digests = set()
    for split in ("train", "calibration", "development"):
        path = source / f"{split}.jsonl"
        if manifest["data_sha256"].get(path.name) != _hash(path):
            raise ValueError(f"{split} digest mismatch")
        rows = _rows(path)
        if len(rows) != manifest["root_counts"][split]:
            raise ValueError(f"{split} root count mismatch")
        for row in rows:
            if row.get("schema_version") != SOURCE_ROW or row.get("split") != split:
                raise ValueError("row split/schema mismatch")
            if row.get("model_sha256") != MODEL_SHA:
                raise ValueError("row checkpoint mismatch")
            if row.get("training_eligible") is not (split == "train"):
                raise ValueError("training data custody mismatch")
            if row.get("oracle_target_visible_to_controller") is not False:
                raise ValueError("controller observed target oracle")
            if row.get("future_step_visible_to_controller") is not False:
                raise ValueError("controller observed future step")
            if type(row.get("root_id")) is not str or not row["root_id"]:
                raise ValueError("root ID required")
            if row["root_id"] in all_roots:
                raise ValueError("root reused across splits")
            all_roots.add(row["root_id"])
            q = row.get("question")
            if not isinstance(q, str) or len(q) != 81:
                raise ValueError("invalid question")
            digest = _digest({"question": q})
            if row.get("task_digest") != digest or digest in all_digests:
                raise ValueError("duplicate or mismatched lineage")
            all_digests.add(digest)
            for step in ("first", "second"):
                candidate = row.get(f"{step}_candidate")
                outcome = row.get(f"{step}_pass_verified")
                if type(outcome) is not bool or _sudoku_valid(q, candidate) is not outcome:
                    raise ValueError("verifier label does not replay")
            if row.get("second_pass_rescue") is not (
                row["second_pass_verified"] and not row["first_pass_verified"]
            ):
                raise ValueError("rescue label mismatch")
            features = row.get("features")
            if (
                not isinstance(features, list) or len(features) != len(manifest["feature_names"])
                or any(type(v) not in {int, float} or not math.isfinite(v) for v in features)
            ):
                raise ValueError("nonfinite or incomplete feature array")
        all_rows[split] = rows
    # Evaluation/online policy records are not eligible training examples.
    eval_path = source / "evaluation.jsonl"
    if manifest["data_sha256"].get(eval_path.name) != _hash(eval_path):
        raise ValueError("evaluation digest mismatch")
    if len(_rows(eval_path)) != manifest["root_counts"]["development"]:
        raise ValueError("incomplete development policy outcomes")
    if out.exists():
        raise ValueError("output must be a new directory")
    out.mkdir(parents=True)
    sft, repair, outcomes = [], [], []
    for row in all_rows["train"]:
        base = {
            "root_id": row["root_id"], "task_digest": row["task_digest"],
            "model_sha256": MODEL_SHA, "family": "sudoku9",
            "split": "train", "training_eligible": True,
            "source_sha256": _hash(source / "train.jsonl"),
            "source_model": "pretrained_gaoxin492_trm",
            "teacher_hidden_state_access": False,
        }
        sft.append({
            "schema_version": "rlt_gym.controller_sft.v1", **base,
            "kind": "verified_stop_decision",
            "input": {
                "task_question": row["question"],
                "first_candidate": row["first_candidate"],
                "logits_only_feature_names": manifest["feature_names"],
                "logits_only_features": row["features"],
            },
            "target": {"decision": "stop" if row["first_pass_verified"] else "continue"},
            "verification": {"first_pass_verified": row["first_pass_verified"]},
        })
        outcomes.append({
            "schema_version": "rlt_gym.controller_outcome.v1", **base,
            "kind": "future_verifier_outcome",
            "first_pass_verified": row["first_pass_verified"],
            "second_pass_verified": row["second_pass_verified"],
            "second_pass_rescue": row["second_pass_rescue"],
            "observed_second_candidate": row["second_candidate"],
        })
        if row["second_pass_rescue"]:
            repair.append({
                "schema_version": "rlt_gym.controller_preference.v1", **base,
                "kind": "verified_repair",
                "input": {"task_question": row["question"]},
                "rejected": {"candidate": row["first_candidate"]},
                "chosen": {"candidate": row["second_candidate"]},
                "verification": {"chosen": True, "rejected": False},
            })
    _jsonl(out / "controller_sft.jsonl", sft)
    _jsonl(out / "outcomes.jsonl", outcomes)
    _jsonl(out / "repairs.jsonl", repair)
    report = {
        "schema_version": "rlt_gym.pretrained_trm_train_export.v1",
        "model_sha256": MODEL_SHA,
        "source_report_sha256": _hash(manifest_path),
        "source_train_sha256": _hash(source / "train.jsonl"),
        "source_calibration_sha256": _hash(source / "calibration.jsonl"),
        "source_development_sha256": _hash(source / "development.jsonl"),
        "source_evaluation_sha256": _hash(eval_path),
        "training_root_count": len(sft), "verified_repair_pairs": len(repair),
        "train_only": True, "development_rows_exported": 0,
        "files_sha256": {
            n: _hash(out / n) for n in ("controller_sft.jsonl", "outcomes.jsonl", "repairs.jsonl")
        },
        "claim_boundary": (
            "Fresh TRAIN source roots only, exact independent Sudoku verification and "
            "logits-derived controller features. No exposed CAL/DEV labels in this training export."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = export(args.source, args.out)
    print(json.dumps({"training_roots": report["training_root_count"],
                      "repair_pairs": report["verified_repair_pairs"],
                      "train_only": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
