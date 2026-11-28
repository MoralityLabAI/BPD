"""Frozen learned-exit transfer to different Sudoku clue regimes on actual pretrained TRM.

Only the upstream TRAIN/CAL controller weights and threshold are used. This
development experiment independently rechecks original model source custody
and runs new, disjoint uniquely solvable puzzles at two difficulty levels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from mechinterp.bpd.adapters.trm import build_mini_trm, load_pretrained_trm
from mechinterp.bpd.experiments.inkling_reader_v0.learned_exit_controller import (
    _row, evaluate,
)
from mechinterp.bpd.experiments.inkling_reader_v0.native_trm_exit import native_rollout
from mechinterp.bpd.experiments.inkling_reader_v0.native_trm_pilot import (
    _permute_sudoku, verify_sudoku,
)
from mechinterp.loopscope02.scripts.trm_sudoku import TRM_CHECKPOINTS

BASELINE_REPORT_SHA256 = "dda473da08434484f4d67e8f0e6fa77b4296d9fa1489a810f529c699dc94a7e1"
MODEL_SHA256 = "932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"
SCHEMA = "bpd.trm_exit_difficulty_transfer.v1"
COHORTS = ((40, 20261311), (52, 20261312))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen(source: Path) -> tuple[dict[str, Any], set[str]]:
    report = source / "report.json"
    if sha(report) != BASELINE_REPORT_SHA256:
        raise ValueError("learned controller source report does not match pinned artifact")
    obj = json.loads(report.read_text(encoding="utf-8"))
    if (
        obj.get("schema_version") != "bpd.trm_learned_exit.v1"
        or obj.get("model_sha256") != MODEL_SHA256
        or obj.get("evidence_class") != "native_pretrained_development_learned_exit"
        or obj.get("holes") != 50
        or obj.get("feature_names") != obj.get("training", {}).get("features")
        or obj.get("calibration", {}).get("false_stops") != 0
    ):
        raise ValueError("invalid original checkpoint/controller evidence")
    prior = set()
    for split in ("train", "calibration", "development"):
        path = source / f"{split}.jsonl"
        if sha(path) != obj["data_sha256"][path.name]:
            raise ValueError(f"frozen {split} corpus digest changed")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("split") != split or row.get("model_sha256") != MODEL_SHA256:
                raise ValueError("bad source split/model identity")
            digest = row.get("task_digest")
            if not isinstance(digest, str) or digest in prior:
                raise ValueError("duplicate source task across original splits")
            prior.add(digest)
    expected = sum(obj["root_counts"].values())
    if len(prior) != expected:
        raise ValueError("source root census changed")
    return obj, prior


def generate_verified_rows(
    model: torch.nn.Module, *, sha256: str, holes: int, seed: int, roots: int,
    prohibited: set[str],
) -> list[dict[str, Any]]:
    if type(roots) is not int or not 2 <= roots <= 48:
        raise ValueError("root count must lie in 2..48")
    if holes not in (40, 52):
        raise ValueError("difficulty transfer accepts only predeclared 40/52 blank panels")
    seen = set()
    rows = []
    for index in range(roots):
        q, target = _permute_sudoku(seed + 101 * index, holes=holes)
        if not verify_sudoku(q, target):
            raise ValueError("generated Sudoku solution invalid")
        observed, _ = native_rollout(model, q, max_outer_steps=2, stop_when_verified=False)
        if any(s["sudoku_checker_passed"] != (s["candidate"] == target) for s in observed):
            raise ValueError("solver target and exact Sudoku checker disagreed")
        row = _row("development", index, q, observed, sha256)
        if row["task_digest"] in seen or row["task_digest"] in prohibited:
            raise ValueError("repeated source world across difficulty or fit splits")
        seen.add(row["task_digest"])
        row["root_id"] = f"transfer-h{holes}-s{seed + 101 * index}"
        row["difficulty_holes"] = holes
        row["split"] = "development"
        row["training_eligible"] = False
        rows.append(row)
    return rows


def _add_fixed_one(rows: list[dict[str, Any]], result: dict[str, Any]) -> None:
    """First-visit baseline is computed from the separately checked fixed trace."""
    correct = sum(row["first_pass_verified"] for row in rows)
    result["scores"]["fixed1"] = {
        "solved": correct, "total": len(rows),
        "total_outer_steps": len(rows), "stopped_after_first": len(rows),
        "incorrect_early_exits": len(rows) - correct,
        "native_seconds": None, "checker_seconds": None,
        "source": "same checkpoint frozen first-visit predictions; no additional model calls",
    }


def run_transfer(
    model: torch.nn.Module, original: dict[str, Any], prohibited: set[str],
    *, roots_per_panel: int = 24, cohorts: tuple[tuple[int, int], ...] = COHORTS,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    if not cohorts or len({holes for holes, _ in cohorts}) != len(cohorts):
        raise ValueError("distinct declared transfer difficulties required")
    if original.get("model_sha256") != MODEL_SHA256:
        raise ValueError("model checkpoint changed")
    if original["calibration"]["threshold"] <= 0 or original["calibration"]["threshold"] > 1.1:
        raise ValueError("frozen stop threshold invalid")
    model.eval()
    blocked = set(prohibited)
    outputs: dict[int, list[dict[str, Any]]] = {}
    receipts = []
    begin = time.monotonic()
    for holes, seed in cohorts:
        rows = generate_verified_rows(
            model, sha256=MODEL_SHA256, holes=holes, seed=seed,
            roots=roots_per_panel, prohibited=blocked,
        )
        blocked.update(r["task_digest"] for r in rows)
        results, evaluated = evaluate(
            model, rows, original["training"], original["calibration"],
        )
        if len(evaluated) != len(rows):
            raise ValueError("incomplete policy comparison")
        _add_fixed_one(rows, results)
        outputs[holes] = [
            {
                "schema_version": "bpd.trm_exit_transfer_case.v1",
                "source_split": "development",
                "training_eligible": False,
                "root_id": row["root_id"],
                "task_digest": row["task_digest"],
                "question": row["question"],
                "holes": holes, "seed": seed,
                "first_pass_valid": row["first_pass_verified"],
                "second_pass_valid": row["second_pass_verified"],
                "first_pass_candidate": row["first_candidate"],
                "second_pass_candidate": row["second_candidate"],
                "frozen_controller_score": evaluated[i]["learned_score"],
                "policies": evaluated[i]["policies"],
            }
            for i, row in enumerate(rows)
        ]
        receipts.append({
            "holes": holes, "seed": seed, "roots": len(rows),
            "results": results,
            "source_digest": hashlib.sha256(
                json.dumps([(r["root_id"], r["task_digest"]) for r in rows],
                           sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        })
    return {
        "schema_version": SCHEMA,
        "evidence_class": "pretrained_native_out_of_difficulty_development",
        "checkpoint_sha256": MODEL_SHA256,
        "frozen_training_report_sha256": BASELINE_REPORT_SHA256,
        "source_training_roots_excluded": len(prohibited),
        "calibration_threshold_frozen": original["calibration"]["threshold"],
        "calibration_stop_rate_frozen": original["calibration"]["random_stop_probability"],
        "model_observation": "first_pass_digit_logits_only",
        "evaluation_uses_first_pass_verifier_result": False,
        "source_roots_per_panel": roots_per_panel,
        "cohorts": receipts,
        "total_wall_seconds": round(time.monotonic() - begin, 4),
        "claim_boundary": (
            "Two development-only Sudoku difficulty shifts with frozen single-model "
            "weights and an already calibrated stop classifier. No online retraining, "
            "threshold tuning, Lean/MCP generalization or Inkling routing was performed."
        ),
    }, outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=("gaoxin492", "mini"), default="gaoxin492")
    parser.add_argument("--roots-per-panel", type=int, default=24)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output must be a fresh directory")
    torch.set_num_threads(2)
    original, protected = load_frozen(args.source)
    if args.model == "gaoxin492":
        model = load_pretrained_trm("gaoxin492", dtype=torch.float32, verify_sha256=True)
        if TRM_CHECKPOINTS["gaoxin492"]["sha256"] != MODEL_SHA256:
            raise ValueError("pretrained model SHA changed")
    else:
        model = build_mini_trm(
            hidden_size=32, num_heads=4, l_layers=1, h_cycles=2,
            l_cycles=1, puzzle_emb_len=4, seq_len=81, seed=43,
        )
    report, rows = run_transfer(
        model, original, protected, roots_per_panel=args.roots_per_panel,
    )
    if args.model == "mini":
        report["evidence_class"] = "untrained_mini_plumbing_smoke"
        report["checkpoint_sha256"] = "untrained_mini_seed_43"
    args.out.mkdir(parents=True)
    files = {}
    for holes, examples in rows.items():
        path = args.out / f"transfer_h{holes}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in examples:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        files[path.name] = sha(path)
    report["files_sha256"] = files
    report["source_sha256"] = sha(Path(__file__))
    (args.out / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(args.out),
        "cohorts": [
            {
                "holes": c["holes"],
                "roots": c["roots"],
                "fixed2": c["results"]["scores"]["fixed2"],
                "learned": c["results"]["scores"]["learned"],
                "random_matched": c["results"]["scores"]["random_matched"],
            } for c in report["cohorts"]
        ],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
