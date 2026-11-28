"""Independent, CPU-free audit of pretrained TRM verifier-gated recurrence receipts.

Recomputes Sudoku legality and budget consistency from saved text outputs.
This audits the *reported* native trajectories; it does not reexecute weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "bpd.native_trm_verifier_exit.v1"
PINNED_GAOXIN = "932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sudoku_valid(question: str, candidate: str) -> bool:
    if len(question) != 81 or len(candidate) != 81:
        return False
    if any(c not in "123456789." for c in question):
        return False
    if any(c not in "123456789" for c in candidate):
        return False
    if any(c != "." and c != candidate[i] for i, c in enumerate(question)):
        return False
    for r in range(9):
        if {candidate[r * 9 + c] for c in range(9)} != set("123456789"):
            return False
    for c in range(9):
        if {candidate[r * 9 + c] for r in range(9)} != set("123456789"):
            return False
    for r in (0, 3, 6):
        for c in (0, 3, 6):
            if {candidate[(r + dr) * 9 + c + dc] for dr in range(3) for dc in range(3)} != set(
                "123456789"
            ):
                return False
    return True


def _nonnegative(value: Any) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError("missing or invalid measured nonnegative number")
    return float(value)


def audit(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("receipt too large")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or obj.get("schema_version") != SCHEMA:
        raise ValueError("invalid receipt schema")
    if (
        obj.get("evidence_class") != "native_pretrained_development"
        or obj.get("model") != "gaoxin492"
        or obj.get("model_sha256") != PINNED_GAOXIN
        or obj.get("split") != "development"
        or obj.get("check_uses_planted_solution") is not False
        or obj.get("student_training_eligible") is not False
    ):
        raise ValueError("wrong model, provenance or task custody")
    budget = obj.get("max_outer_steps")
    roots = obj.get("roots")
    if type(roots) is not int or not 1 <= roots <= 64:
        raise ValueError("invalid root count")
    if type(budget) is not int or budget < 2 or budget > 8:
        raise ValueError("invalid outer depth")
    records = obj.get("records")
    if not isinstance(records, list) or len(records) != roots:
        raise ValueError("incomplete root roster")
    identities = set()
    totals: dict[str, Any] = {}
    for policy in ("fixed_2", "verifier_exit"):
        totals[policy] = {
            "verified_solved": 0, "cases": roots,
            "outer_steps_total": 0,
            "elapsed_wall_seconds": 0.0,
            "total_native_step_seconds": 0.0,
            "total_checker_seconds": 0.0,
            "stop_reasons": {},
        }
    first_pass = 0
    second_pass = 0
    regrets = 0
    for entry in records:
        root, question = entry.get("root_id"), entry.get("question")
        if not isinstance(root, str) or root in identities:
            raise ValueError("duplicate/invalid source root")
        identities.add(root)
        if not isinstance(question, str) or len(question) != 81:
            raise ValueError("invalid question")
        if question.count(".") != obj.get("holes"):
            raise ValueError("clue count differs from manifest")
        if entry.get("task_digest") != _digest({"question": question}):
            raise ValueError("task digest mismatch")
        if entry.get("model_sha256") != PINNED_GAOXIN:
            raise ValueError("root changed checkpoint")
        if set(entry.get("policies", {})) != {"fixed_2", "verifier_exit"}:
            raise ValueError("missing policy")
        order = entry.get("measured_arm_order")
        if not isinstance(order, list) or set(order) != {"fixed_2", "verifier_exit"}:
            raise ValueError("missing policy run order")
        fixed_trace = entry["policies"]["fixed_2"]["trace"]
        conditional_trace = entry["policies"]["verifier_exit"]["trace"]
        if len(fixed_trace) != budget or not 1 <= len(conditional_trace) <= budget:
            raise ValueError("policy executed wrong depth")
        if [r.get("candidate") for r in fixed_trace[:len(conditional_trace)]] != [
            r.get("candidate") for r in conditional_trace
        ]:
            raise ValueError("mismatched model trajectory between policy arms")
        first_pass += int(fixed_trace[0]["sudoku_checker_passed"])
        second_pass += int(fixed_trace[-1]["sudoku_checker_passed"])
        regrets += int(fixed_trace[0]["sudoku_checker_passed"] and not fixed_trace[-1]["sudoku_checker_passed"])
        for policy in ("fixed_2", "verifier_exit"):
            row = entry["policies"][policy]
            steps = row["trace"]
            if row.get("realized_outer_steps") != len(steps):
                raise ValueError("reported visits disagree with trace")
            if [s.get("outer_step") for s in steps] != list(range(1, len(steps) + 1)):
                raise ValueError("non-contiguous recurrent visits")
            for step in steps:
                if type(step.get("sudoku_checker_passed")) is not bool:
                    raise ValueError("missing checker boolean")
                if _sudoku_valid(question, step["candidate"]) != step["sudoku_checker_passed"]:
                    raise ValueError("independent exact Sudoku verifier disagrees")
                _nonnegative(step.get("native_step_seconds"))
                _nonnegative(step.get("exact_check_seconds"))
            if policy == "verifier_exit":
                if any(step["sudoku_checker_passed"] for step in steps[:-1]):
                    raise ValueError("adaptive policy kept computing after verified success")
                if len(steps) < budget and not steps[-1]["sudoku_checker_passed"]:
                    raise ValueError("adaptive policy quit without a valid solution")
            if row.get("stop_reason") not in {"verified", "depth_budget", "fixed_depth"}:
                raise ValueError("invalid stopping status")
            if policy == "verifier_exit" and (row["stop_reason"] == "verified") != steps[-1][
                "sudoku_checker_passed"
            ]:
                raise ValueError("stopping status mismatches verification")
            total = totals[policy]
            total["verified_solved"] += int(steps[-1]["sudoku_checker_passed"])
            total["outer_steps_total"] += len(steps)
            total["elapsed_wall_seconds"] += _nonnegative(row.get("elapsed_seconds"))
            total["total_native_step_seconds"] += sum(_nonnegative(s["native_step_seconds"]) for s in steps)
            total["total_checker_seconds"] += sum(_nonnegative(s["exact_check_seconds"]) for s in steps)
            reason = row["stop_reason"]
            total["stop_reasons"][reason] = total["stop_reasons"].get(reason, 0) + 1
    for policy in totals:
        rec = obj["summary"][policy]
        actual = totals[policy]
        for field in ("verified_solved", "cases", "outer_steps_total", "stop_reasons"):
            if actual[field] != rec[field]:
                raise ValueError(f"summary mismatch: {policy}/{field}")
        for field in ("elapsed_wall_seconds", "total_native_step_seconds", "total_checker_seconds"):
            if not math.isclose(actual[field], rec[field], abs_tol=0.0001):
                raise ValueError(f"timing total mismatch: {policy}/{field}")
    fixed, adaptive = totals["fixed_2"], totals["verifier_exit"]
    step_savings = 1 - adaptive["outer_steps_total"] / fixed["outer_steps_total"]
    native_savings = 1 - adaptive["total_native_step_seconds"] / fixed["total_native_step_seconds"]
    if not math.isclose(step_savings, obj["summary"]["step_count_fraction_saved"], abs_tol=1e-9):
        raise ValueError("incorrect reported saved steps")
    if not math.isclose(
        native_savings, obj["summary"]["measured_native_step_fraction_saved"], abs_tol=1e-6
    ):
        raise ValueError("incorrect reported native time savings")
    return {
        "schema_version": "bpd.native_trm_verifier_exit_audit.v1",
        "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_sha256": PINNED_GAOXIN,
        "source_roots": roots,
        "depth_one_verified": first_pass,
        "depth_two_verified": second_pass,
        "later_regressions": regrets,
        "fixed_solved": fixed["verified_solved"],
        "adaptive_solved": adaptive["verified_solved"],
        "fixed_steps": fixed["outer_steps_total"],
        "adaptive_steps": adaptive["outer_steps_total"],
        "step_fraction_saved": step_savings,
        "measured_native_fraction_saved": native_savings,
        "measured_wall_fraction_saved": (
            1 - adaptive["elapsed_wall_seconds"] / fixed["elapsed_wall_seconds"]
        ),
        "root_and_checker_replay_passed": True,
        "evidence_class": "independently_rechecked_development_receipts",
        "limitations": (
            "Exact solution checked from public puzzle/output and timing totals; native "
            "weights were not reexecuted by this auditor. Model and roots are development-exposed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("audit destination already exists")
    report = audit(args.input)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"replayed": True, "roots": report["source_roots"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
