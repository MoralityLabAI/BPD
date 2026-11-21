"""Independently replay exact Sudoku outcomes and policy decisions in RLT-EXIT-04.

Uses archived text/data and immutable originally trained controller parameters.
This rechecks saved receipts; it does not reexecute pretrained TRM weights.
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
FROZEN_SHA = "dda473da08434484f4d67e8f0e6fa77b4296d9fa1489a810f529c699dc94a7e1"
FEATURE_COUNT = 6
SUPPORT_MAX_HOLES = 50
PANELS = ((40, 20261411), (52, 20261412))
POLICIES = ("fixed2", "exact_verifier", "learned", "random_matched")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def valid_sudoku(question: str, candidate: str) -> bool:
    if not isinstance(question, str) or not isinstance(candidate, str):
        return False
    if len(question) != 81 or len(candidate) != 81:
        return False
    if any(x not in "123456789." for x in question):
        return False
    if any(x not in "123456789" for x in candidate):
        return False
    if any(x != "." and candidate[i] != x for i, x in enumerate(question)):
        return False
    digits = set("123456789")
    for r in range(9):
        if {candidate[r * 9 + c] for c in range(9)} != digits:
            return False
    for c in range(9):
        if {candidate[r * 9 + c] for r in range(9)} != digits:
            return False
    for r in (0, 3, 6):
        for c in (0, 3, 6):
            if {candidate[(r + dr) * 9 + c + dc] for dr in range(3) for dc in range(3)} != digits:
                return False
    return True


def _probability(values: list[float], controller: dict[str, Any]) -> float:
    if not isinstance(values, list) or len(values) != FEATURE_COUNT:
        raise ValueError("wrong frozen controller feature count")
    arrays = [controller[name] for name in ("mean", "scale", "weights")]
    if any(not isinstance(a, list) or len(a) != FEATURE_COUNT for a in arrays):
        raise ValueError("invalid controller parameters")
    z = controller["bias"]
    for x, mu, scale, weight in zip(values, *arrays, strict=True):
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (x, mu, scale, weight)):
            raise ValueError("nonfinite classifier features")
        if scale <= 0:
            raise ValueError("classifier scale must be positive")
        z += ((x - mu) / scale) * weight
    z = min(40.0, max(-40.0, z))
    return 1 / (1 + math.exp(-z))


def _expected_visits(policy: str, *, first_valid: bool, score: float, threshold: float,
                     index: int, random_rate: float) -> int:
    if policy == "fixed2":
        return 2
    if policy == "exact_verifier":
        return 1 if first_valid else 2
    if policy == "learned":
        return 1 if score >= threshold else 2
    if policy == "random_matched":
        # Retain the historical development-index seed to replay its controls.
        mask = random.Random((20261213 + 101 * index) ^ 0xF107).random() < random_rate
        return 1 if mask else 2
    raise ValueError("unrecognized comparator policy")


def audit(source: Path, frozen_report: Path) -> dict[str, Any]:
    if _sha(frozen_report) != FROZEN_SHA:
        raise ValueError("frozen controller artifact changed")
    frozen = json.loads(frozen_report.read_text(encoding="utf-8"))
    result_path = source / "report.json"
    if result_path.stat().st_size > 1_000_000:
        raise ValueError("oversized summary")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("schema_version") != "bpd.trm_exit_support_guard.v1"
        or result.get("model_sha256") != MODEL_SHA
        or result.get("frozen_controller_report_sha256") != FROZEN_SHA
        or result.get("evidence_class") != "posthoc_registered_support_guard_fresh_development_roots"
        or result.get("guard_rule_frozen_before_new_sources") != "holes <= 50 else fixed_2"
        or not math.isclose(result.get("frozen_threshold"), frozen["calibration"]["threshold"], abs_tol=1e-12)
    ):
        raise ValueError("source/model/support rule mismatch")
    threshold = frozen["calibration"]["threshold"]
    random_rate = frozen["calibration"]["random_stop_probability"]
    controller = frozen["training"]
    expected_panels = {holes: seed for holes, seed in PANELS}
    if {p.get("holes"): p.get("seed") for p in result["panels"]} != expected_panels:
        raise ValueError("wrong registered cohort roster")
    seen_roots, seen_questions = set(), set()
    reports = []
    for panel in result["panels"]:
        holes, seed, n = panel["holes"], panel["seed"], panel["roots"]
        path = source / f"guard_h{holes}.jsonl"
        if path.stat().st_size > 16 * 1024 * 1024 or _sha(path) != result["files_sha256"][path.name]:
            raise ValueError("source evidence digest/size mismatch")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != n or type(n) is not int or not 1 <= n <= 64:
            raise ValueError("incomplete cohort")
        grade = {
            key: {"solved": 0, "steps": 0, "incorrect_early_exits": 0, "native_seconds": 0.0}
            for key in (*POLICIES, "support_guarded")
        }
        first_solves = 0
        for index, row in enumerate(rows):
            expected_root = f"transfer-h{holes}-s{seed + 101 * index}"
            q = row["question"]
            if (
                row.get("root_id") != expected_root or expected_root in seen_roots
                or not isinstance(q, str) or q.count(".") != holes or q in seen_questions
                or row.get("split") != "development" or row.get("training_eligible") is not False
                or row.get("holes") != holes or row.get("controller_support_max_holes") != SUPPORT_MAX_HOLES
                or row.get("task_digest") != _digest({"question": q})
            ):
                raise ValueError("root, task, or development custody mismatch")
            seen_roots.add(expected_root)
            seen_questions.add(q)
            first, second = row["first_pass_candidate"], row["second_pass_candidate"]
            first_valid, second_valid = valid_sudoku(q, first), valid_sudoku(q, second)
            if row["first_pass_verified"] is not first_valid or row["second_pass_verified"] is not second_valid:
                raise ValueError("independent Sudoku grade differs from source")
            first_solves += int(first_valid)
            score = row["frozen_controller_score"]
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError("invalid controller output")
            recomputed_score = _probability(row.get("first_step_features"), controller)
            if not math.isclose(score, recomputed_score, abs_tol=1e-10):
                raise ValueError("frozen trained classifier score does not replay")
            if not math.isclose(score, row["reference_learned_score"], abs_tol=1e-12):
                raise ValueError("inconsistent frozen controller score")
            policies = row["reference_policies"]
            if not isinstance(policies, dict) or set(policies) != set(POLICIES):
                raise ValueError("incomplete policy matrix")
            for policy in POLICIES:
                visits = _expected_visits(
                    policy, first_valid=first_valid, score=score,
                    threshold=threshold, index=index, random_rate=random_rate,
                )
                selected = first if visits == 1 else second
                source_row = policies[policy]
                outcome = first_valid if visits == 1 else second_valid
                if (
                    source_row.get("visits") != visits
                    or source_row.get("correct") is not outcome
                    or source_row.get("first_candidate") != first
                    or source_row.get("final_candidate") != selected
                ):
                    raise ValueError(f"{policy} output disagrees with verifier replay")
                cost = source_row.get("native_seconds")
                if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                    raise ValueError("missing actual native seconds")
                metrics = grade[policy]
                metrics["steps"] += visits
                metrics["solved"] += int(outcome)
                metrics["incorrect_early_exits"] += int(visits == 1 and not outcome)
                metrics["native_seconds"] += cost
            guard_visits = 2 if holes > SUPPORT_MAX_HOLES else (
                1 if score >= threshold else 2
            )
            guard_pass = first_valid if guard_visits == 1 else second_valid
            if (
                row.get("guarded_outer_steps") != guard_visits
                or row.get("guarded_correct") is not guard_pass
            ):
                raise ValueError("guarded outcome violates frozen threshold/support rule")
            native_seconds = row.get("guarded_native_seconds")
            if type(native_seconds) not in (int, float) or not math.isfinite(native_seconds) or native_seconds < 0:
                raise ValueError("invalid guard execution cost")
            gm = grade["support_guarded"]
            gm["steps"] += guard_visits
            gm["solved"] += int(guard_pass)
            gm["incorrect_early_exits"] += int(guard_visits == 1 and not guard_pass)
            gm["native_seconds"] += native_seconds
        for key, totals in grade.items():
            reported = panel["support_guarded"] if key == "support_guarded" else panel[
                {"fixed2": "fixed_2", "exact_verifier": "exact_verifier",
                 "learned": "original_frozen_learned", "random_matched": "calibration_rate_random"}[key]
            ]
            report_steps = reported["outer_steps"] if key == "support_guarded" else reported["total_outer_steps"]
            if (report_steps != totals["steps"] or reported["solved"] != totals["solved"]
                    or reported["incorrect_early_exits"] != totals["incorrect_early_exits"]
                    or not math.isclose(reported["native_seconds"], totals["native_seconds"], abs_tol=0.0002)):
                raise ValueError(f"{key} aggregate differs from source outcomes and timing")
        reports.append({
            "holes": holes, "roots": n, "first_pass_solved": first_solves,
            "verified_policy_totals": grade,
        })
    return {
        "schema_version": "bpd.rlt_exit_support_guard_audit.v1",
        "source_report_sha256": _sha(result_path),
        "pinned_controller_sha256": FROZEN_SHA,
        "checkpoint_sha256": MODEL_SHA,
        "unique_source_roots": len(seen_roots),
        "panels": reports,
        "independent_exact_checker_replay": True,
        "learned_stop_policy_replay": True,
        "limitation": (
            "Rechecks written Sudoku candidate outputs and policy decisions from native run; "
            "does not reexecute pretrained TRM weights or certify cross-task generalization."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists")
    report = audit(args.source, args.frozen)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"checked_roots": report["unique_source_roots"],
                      "all_passed": report["independent_exact_checker_replay"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
