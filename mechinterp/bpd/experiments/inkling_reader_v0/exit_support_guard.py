"""Predeclared support guard for the frozen 50-blank trained TRM exit controller.

A difficulty shift to 52 blanks produced an unsafe exit in RLT-EXIT-03.
This posthoc-designed guard allows learned stopping only for <=50 blanks
and evaluates it on NEW development source roots, without refitting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mechinterp.bpd.adapters.trm import build_mini_trm, load_pretrained_trm
from mechinterp.bpd.experiments.inkling_reader_v0.exit_difficulty_transfer import (
    BASELINE_REPORT_SHA256, COHORTS as OLD_COHORTS, MODEL_SHA256,
    generate_verified_rows, load_frozen, sha,
)
from mechinterp.bpd.experiments.inkling_reader_v0.learned_exit_controller import (
    FEATURES, _logit_model, evaluate,
)
from mechinterp.bpd.experiments.inkling_reader_v0.native_trm_exit import native_rollout
from mechinterp.loopscope02.scripts.trm_sudoku import TRM_CHECKPOINTS

SCHEMA = "bpd.trm_exit_support_guard.v1"
EASY_HOLES = 40
HARD_HOLES = 52
ALLOWED_HOLES = 50
FRESH_COHORTS = ((EASY_HOLES, 20261411), (HARD_HOLES, 20261412))


def _policy_details(
    model: torch.nn.Module, rows: list[dict[str, Any]],
    frozen: dict[str, Any], *, holes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params, calibration = frozen["training"], frozen["calibration"]
    threshold = calibration["threshold"]
    records = []
    for row in rows:
        q = row["question"]

        def guarded_stop(features: dict[str, float]) -> bool:
            if holes > ALLOWED_HOLES:
                return False
            feature = np.array([features[name] for name in FEATURES], dtype=np.float64)
            return bool(_logit_model(feature, params) >= threshold)

        trace, receipt = native_rollout(
            model, q, max_outer_steps=2, stop_when_verified=False,
            stop_decider=guarded_stop,
        )
        if trace[0]["candidate"] != row["first_candidate"]:
            raise ValueError("guarded first visit diverged from frozen task")
        if len(trace) == 2 and trace[1]["candidate"] != row["second_candidate"]:
            raise ValueError("guarded second visit changed fixed-depth trajectory")
        if holes > ALLOWED_HOLES and len(trace) != 2:
            raise ValueError("out-of-support case stopped early")
        valid = trace[-1]["sudoku_checker_passed"]
        if valid != (
            row["first_pass_verified"] if len(trace) == 1 else row["second_pass_verified"]
        ):
            raise ValueError("guarded output disagrees with independent Sudoku check")
        records.append({
            "root_id": row["root_id"],
            "task_digest": row["task_digest"],
            "question": q,
            "split": "development",
            "training_eligible": False,
            "holes": holes,
            "controller_support_max_holes": ALLOWED_HOLES,
            "first_pass_verified": row["first_pass_verified"],
            "second_pass_verified": row["second_pass_verified"],
            "first_pass_candidate": row["first_candidate"],
            "second_pass_candidate": row["second_candidate"],
            "frozen_controller_score": float(_logit_model(
                np.array(row["features"], dtype=np.float64), params,
            )),
            "guarded_outer_steps": len(trace),
            "guarded_correct": valid,
            "guarded_native_seconds": round(sum(s["native_step_seconds"] for s in trace), 6),
            "guarded_checker_seconds": round(sum(s["exact_check_seconds"] for s in trace), 6),
            "guarded_stop_reason": receipt["stop_reason"],
        })
    return records, {
        "solved": sum(row["guarded_correct"] for row in records),
        "roots": len(records),
        "outer_steps": sum(row["guarded_outer_steps"] for row in records),
        "native_seconds": round(sum(row["guarded_native_seconds"] for row in records), 6),
        "incorrect_early_exits": sum(
            row["guarded_outer_steps"] == 1 and not row["guarded_correct"] for row in records
        ),
        "stopped_after_first": sum(row["guarded_outer_steps"] == 1 for row in records),
        "support_rule": f"learned_exit_if_holes_lte_{ALLOWED_HOLES}_else_fixed_2",
    }


def run(
    model: torch.nn.Module, frozen: dict[str, Any], prohibited: set[str],
    *, roots_per_panel: int = 24,
    cohorts: tuple[tuple[int, int], ...] = FRESH_COHORTS,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    if set(holes for holes, _ in cohorts) != {EASY_HOLES, HARD_HOLES}:
        raise ValueError("both predeclared ease/hardness arms are required")
    if any(seed in {s for _, s in OLD_COHORTS} for _, seed in cohorts):
        raise ValueError("cannot reuse the prior difficulty-shift seeds")
    used = set(prohibited)
    report_rows = {}
    panel_summaries = []
    started = time.monotonic()
    for holes, seed in cohorts:
        rows = generate_verified_rows(
            model, sha256=MODEL_SHA256, holes=holes, seed=seed,
            roots=roots_per_panel, prohibited=used,
        )
        used.update(r["task_digest"] for r in rows)
        unguarded, seen = evaluate(
            model, rows, frozen["training"], frozen["calibration"],
        )
        if len(seen) != len(rows):
            raise ValueError("unguarded policy matrix incomplete")
        guarded_rows, guard = _policy_details(model, rows, frozen, holes=holes)
        for index, receipt in enumerate(guarded_rows):
            reference = seen[index]
            if receipt["root_id"] != reference["root_id"]:
                raise ValueError("guard and policy comparison source roots differ")
            receipt["reference_policies"] = reference["policies"]
            receipt["reference_learned_score"] = reference["learned_score"]
        scores = unguarded["scores"]
        report_rows[holes] = guarded_rows
        panel_summaries.append({
            "holes": holes, "seed": seed, "roots": len(rows),
            "original_frozen_learned": scores["learned"],
            "fixed_2": scores["fixed2"],
            "exact_verifier": scores["exact_verifier"],
            "calibration_rate_random": scores["random_matched"],
            "support_guarded": guard,
            "guarded_vs_fixed_steps_saved": (
                1 - guard["outer_steps"] / scores["fixed2"]["total_outer_steps"]
            ),
            "guarded_vs_fixed_native_time_saved": (
                1 - guard["native_seconds"] / scores["fixed2"]["native_seconds"]
            ),
        })
    return {
        "schema_version": SCHEMA,
        "evidence_class": "posthoc_registered_support_guard_fresh_development_roots",
        "model_sha256": MODEL_SHA256,
        "frozen_controller_report_sha256": BASELINE_REPORT_SHA256,
        "frozen_threshold": frozen["calibration"]["threshold"],
        "guard_rule_frozen_before_new_sources": "holes <= 50 else fixed_2",
        "root_counts": roots_per_panel,
        "panels": panel_summaries,
        "source_train_cal_dev_roots_excluded": len(prohibited),
        "total_wall_seconds": round(time.monotonic() - started, 4),
        "claim_boundary": (
            "The guard is a posthoc response to the already observed 52-blank failure. "
            "It is evaluated on new independent development source seeds with no "
            "retraining or new threshold tuning. No preregistered scientific confirmation."
        ),
    }, report_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=("gaoxin492", "mini"), default="gaoxin492")
    parser.add_argument("--roots-per-panel", type=int, default=24)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists")
    torch.set_num_threads(2)
    original, seen = load_frozen(args.source)
    if args.model == "gaoxin492":
        model = load_pretrained_trm("gaoxin492", dtype=torch.float32, verify_sha256=True)
        if TRM_CHECKPOINTS["gaoxin492"]["sha256"] != MODEL_SHA256:
            raise ValueError("pretrained checkpoint changed")
    else:
        model = build_mini_trm(
            hidden_size=32, num_heads=4, l_layers=1,
            h_cycles=2, l_cycles=1, puzzle_emb_len=4, seq_len=81, seed=43,
        )
    report, rows = run(model, original, seen, roots_per_panel=args.roots_per_panel)
    if args.model == "mini":
        report["evidence_class"] = "untrained_mini_plumbing"
        report["model_sha256"] = "mini_untrained_43"
    args.out.mkdir(parents=True)
    digests = {}
    for holes, data in rows.items():
        p = args.out / f"guard_h{holes}.jsonl"
        p.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                    for row in data), encoding="utf-8",
        )
        digests[p.name] = sha(p)
    report["files_sha256"] = digests
    report["source_sha256"] = sha(Path(__file__))
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "panels": report["panels"],
        "output": str(args.out),
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
