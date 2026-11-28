"""Independent stdlib audit of BPD learned recurrent exit outcomes.

Recomputes exact Sudoku verifier labels, lineage custody, logistic scores,
calibration selection, deterministic random controls and policy outcome totals
from archived JSON. Does not rerun native model weights or certify generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

MODEL_SHA = "932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"
SPLITS = ("train", "calibration", "development")
POLICIES = ("fixed2", "exact_verifier", "learned", "random_matched")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.stat().st_size > 24_000_000:
        raise ValueError("oversized evidence file")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("evidence row must be an object")
            rows.append(record)
    return rows


def _verify_sudoku(q: str, candidate: str) -> bool:
    if not isinstance(q, str) or not isinstance(candidate, str):
        return False
    if len(q) != 81 or len(candidate) != 81:
        return False
    if any(c not in "123456789." for c in q):
        return False
    if any(c not in "123456789" for c in candidate):
        return False
    if any(c != "." and candidate[i] != c for i, c in enumerate(q)):
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


def _score(features: list[float], training: dict[str, Any]) -> float:
    weights = training["weights"]
    mean = training["mean"]
    scale = training["scale"]
    if any(len(seq) != len(features) for seq in (weights, mean, scale)):
        raise ValueError("classifier dimensionality mismatch")
    z = training["bias"]
    for x, w, mu, sigma in zip(features, weights, mean, scale, strict=True):
        if not all(math.isfinite(v) for v in (x, w, mu, sigma)) or sigma <= 0:
            raise ValueError("invalid logistic feature/weight")
        z += ((x - mu) / sigma) * w
    z = min(40, max(-40, z))
    return 1 / (1 + math.exp(-z))


def audit(source: Path) -> dict[str, Any]:
    meta_path = source / "report.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta.get("schema_version") != "bpd.trm_learned_exit.v1"
        or meta.get("evidence_class") != "native_pretrained_development_learned_exit"
        or meta.get("model_sha256") != MODEL_SHA
    ):
        raise ValueError("unrecognized experiment model or evidence class")
    feature_count = len(meta["feature_names"])
    if feature_count != 6:
        raise ValueError("expected six logits-only features")
    stages = {}
    roots = set()
    questions = set()
    for split in SPLITS:
        path = source / f"{split}.jsonl"
        if meta["data_sha256"].get(path.name) != _sha(path):
            raise ValueError(f"{split} source checksum mismatch")
        rows = _rows(path)
        if len(rows) != meta["root_counts"][split]:
            raise ValueError(f"{split} root-count mismatch")
        for index, row in enumerate(rows):
            q = row.get("question")
            root = row.get("root_id")
            expected_root = f"{split}-sudoku-{meta['source_seeds'][split] + 101 * index}"
            if root != expected_root or root in roots or row.get("split") != split:
                raise ValueError("duplicate or mismatched source-root lineage")
            roots.add(root)
            if not isinstance(q, str) or q.count(".") != meta["holes"] or q in questions:
                raise ValueError("duplicate or invalid source question")
            questions.add(q)
            if row.get("task_digest") != _digest({"question": q}):
                raise ValueError("task digest mismatch")
            if row.get("model_sha256") != MODEL_SHA:
                raise ValueError("row uses different checkpoint")
            if row.get("training_eligible") is not (split == "train"):
                raise ValueError("train/cal/development custody mismatch")
            if row.get("oracle_target_visible_to_controller") is not False or (
                row.get("future_step_visible_to_controller") is not False
            ):
                raise ValueError("controller-observation provenance invalid")
            first = _verify_sudoku(q, row.get("first_candidate"))
            second = _verify_sudoku(q, row.get("second_candidate"))
            if (
                row.get("first_pass_verified") is not first
                or row.get("second_pass_verified") is not second
                or row.get("second_pass_rescue") is not (second and not first)
            ):
                raise ValueError("recomputed Sudoku outcome disagrees with row")
            feature = row.get("features")
            if (
                not isinstance(feature, list) or len(feature) != feature_count
                or any(type(v) not in {int, float} or not math.isfinite(v) for v in feature)
            ):
                raise ValueError("invalid logits feature row")
        stages[split] = rows

    training = meta["training"]
    if training["train_count"] != len(stages["train"]) or training["train_positive"] != sum(
        row["first_pass_verified"] for row in stages["train"]
    ):
        raise ValueError("trained-model label counters inconsistent")
    train_predictions = [
        _score(row["features"], training) >= 0.5 for row in stages["train"]
    ]
    train_accuracy = sum(
        pred == row["first_pass_verified"]
        for pred, row in zip(train_predictions, stages["train"], strict=True)
    ) / len(stages["train"])
    if not math.isclose(train_accuracy, training["train_accuracy_at_half"], abs_tol=1e-10):
        raise ValueError("training accuracy disagrees with frozen classifier")

    cal = meta["calibration"]
    candidates = [1.000001] + sorted(
        {_score(row["features"], training) for row in stages["calibration"]}, reverse=True
    )
    best = (1.000001, 0)
    for t in candidates:
        stopped = [
            row for row in stages["calibration"] if _score(row["features"], training) >= t
        ]
        if all(row["first_pass_verified"] for row in stopped) and len(stopped) > best[1]:
            best = (t, len(stopped))
    if (
        not math.isclose(cal["threshold"], best[0], abs_tol=1e-10)
        or cal["stop_count"] != best[1]
        or cal["false_stops"] != 0
        or cal["calibration_cases"] != len(stages["calibration"])
        or cal["calibration_valid_first"] != sum(
            row["first_pass_verified"] for row in stages["calibration"]
        )
        or not math.isclose(cal["random_stop_probability"], best[1] / len(stages["calibration"]),
                            abs_tol=1e-12)
    ):
        raise ValueError("calibration zero-false-stop selection does not replay")

    eval_path = source / "evaluation.jsonl"
    if meta["data_sha256"].get(eval_path.name) != _sha(eval_path):
        raise ValueError("development policy evidence checksum mismatch")
    evaluations = _rows(eval_path)
    if len(evaluations) != len(stages["development"]):
        raise ValueError("incomplete evaluation root matrix")
    totals = {
        name: {"solved": 0, "total_outer_steps": 0, "stopped_after_first": 0,
               "incorrect_early_exits": 0, "native_seconds": 0.0}
        for name in POLICIES
    }
    for index, (row, recorded) in enumerate(zip(stages["development"], evaluations, strict=True)):
        if (
            recorded.get("root_id") != row["root_id"]
            or recorded.get("task_digest") != row["task_digest"]
            or recorded.get("split") != "development"
            or recorded.get("training_eligible") is not False
            or set(recorded.get("policies", {})) != set(POLICIES)
        ):
            raise ValueError("evaluation source root/custody mismatch")
        score = _score(row["features"], training)
        if not math.isclose(recorded["learned_score"], score, abs_tol=1e-10):
            raise ValueError("learned probability receipt mismatch")
        if not math.isclose(recorded["calibrated_threshold"], cal["threshold"], abs_tol=1e-12):
            raise ValueError("learned threshold differs from frozen calibration")
        rand = random.Random((meta["source_seeds"]["development"] + 101 * index) ^ 0xF107)
        random_stop = rand.random() < cal["random_stop_probability"]
        decisions = {
            "fixed2": False,
            "exact_verifier": row["first_pass_verified"],
            "learned": score >= cal["threshold"],
            "random_matched": random_stop,
        }
        for policy in POLICIES:
            actual = recorded["policies"][policy]
            step_count = 1 if decisions[policy] else 2
            correct = row["first_pass_verified"] if step_count == 1 else row["second_pass_verified"]
            candidate = row["first_candidate"] if step_count == 1 else row["second_candidate"]
            if (
                actual.get("visits") != step_count
                or actual.get("correct") is not correct
                or actual.get("final_candidate") != candidate
                or actual.get("first_candidate") != row["first_candidate"]
            ):
                raise ValueError(f"{policy} decision/continuation does not replay")
            secs = actual.get("native_seconds")
            if type(secs) not in {int, float} or not math.isfinite(secs) or secs < 0:
                raise ValueError("invalid native execution time")
            totals[policy]["solved"] += int(correct)
            totals[policy]["total_outer_steps"] += step_count
            totals[policy]["stopped_after_first"] += int(step_count == 1)
            totals[policy]["incorrect_early_exits"] += int(step_count == 1 and not correct)
            totals[policy]["native_seconds"] += secs
    for policy, computed in totals.items():
        registered = meta["evaluation"]["scores"][policy]
        for field in ("solved", "total_outer_steps", "stopped_after_first", "incorrect_early_exits"):
            if computed[field] != registered[field]:
                raise ValueError(f"{policy} summary {field} differs from raw evidence")
        if not math.isclose(computed["native_seconds"], registered["native_seconds"], abs_tol=0.001):
            raise ValueError(f"{policy} timing differs from raw evidence")

    return {
        "schema_version": "bpd.learned_exit_independent_audit.v1",
        "source_report_sha256": _sha(meta_path),
        "training_count": len(stages["train"]),
        "calibration_count": len(stages["calibration"]),
        "development_count": len(stages["development"]),
        "calibration_threshold_replayed": best[0],
        "calibration_zero_false_stops": True,
        "policy_totals": totals,
        "source_lineage_disjoint": True,
        "independent_sudoku_grade_replay": True,
        "model_is_reexecuted_by_this_auditor": False,
        "evidence_class": "native_pretrained_development_independent_receipt_replay",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("audit output exists")
    result = audit(args.source)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "training_roots": result["training_count"],
        "development_roots": result["development_count"],
        "learned_solves": result["policy_totals"]["learned"]["solved"],
        "learned_steps": result["policy_totals"]["learned"]["total_outer_steps"],
        "replayed": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
