"""Learned logits-only exit controller for real pretrained 7M TRM recurrence.

Fit train-only first-visit correctness, choose threshold on separate calibration
roots, and evaluate genuine resumed model execution on new development roots.
All material is a research pilot; no sealed confirmation data is accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mechinterp.bpd.adapters.trm import build_mini_trm, load_pretrained_trm
from mechinterp.bpd.experiments.inkling_reader_v0.native_trm_exit import native_rollout
from mechinterp.bpd.experiments.inkling_reader_v0.native_trm_pilot import (
    _permute_sudoku, verify_sudoku,
)
from mechinterp.loopscope02.scripts.trm_sudoku import TRM_CHECKPOINTS, PureSudokuTRM

SCHEMA = "bpd.trm_learned_exit.v1"
FEATURES = (
    "mean_blank_logit_margin", "min_blank_logit_margin",
    "p10_blank_logit_margin", "median_blank_logit_margin",
    "mean_blank_entropy", "mean_blank_top_probability",
)
SPLIT_SEEDS = {"train": 20261211, "calibration": 20261212, "development": 20261213}
DEFAULT_COUNTS = {"train": 96, "calibration": 48, "development": 48}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _features(obs: dict[str, Any]) -> list[float]:
    values = [obs[name] for name in FEATURES]
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("invalid logits-only feature")
    return [float(x) for x in values]


def _logit_model(features: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    mu, scale = np.array(params["mean"]), np.array(params["scale"])
    weights = np.array(params["weights"])
    x = (features - mu) / scale
    z = np.clip(x @ weights + params["bias"], -40, 40)
    return 1 / (1 + np.exp(-z))


def fit_logistic(rows: list[dict[str, Any]], *, epochs: int = 500, ridge: float = 0.08) -> dict[str, Any]:
    """Deterministic six-dimensional logistic, optimizing only train-root labels."""
    if not rows or any(item["split"] != "train" for item in rows):
        raise ValueError("logistic training requires TRAIN rows exclusively")
    roots = [item["root_id"] for item in rows]
    if len(roots) != len(set(roots)):
        raise ValueError("duplicate training roots")
    x = np.array([row["features"] for row in rows], dtype=np.float64)
    y = np.array([float(row["first_pass_verified"]) for row in rows], dtype=np.float64)
    if not (np.any(y == 0) and np.any(y == 1)):
        raise ValueError("training requires positive and negative labels")
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("invalid feature array")
    mu, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    x = (x - mu) / scale
    w = np.zeros(len(FEATURES), dtype=np.float64)
    bias = 0.0
    for epoch in range(epochs):
        z = np.clip(x @ w + bias, -40, 40)
        error = 1 / (1 + np.exp(-z)) - y
        eta = 0.3 / (1 + epoch / 200)
        w -= eta * ((x.T @ error) / len(rows) + ridge * w)
        bias -= eta * float(error.mean())
    scores = 1 / (1 + np.exp(-np.clip(x @ w + bias, -40, 40)))
    return {
        "features": list(FEATURES), "mean": mu.tolist(), "scale": scale.tolist(),
        "weights": w.tolist(), "bias": bias,
        "ridge": ridge, "epochs": epochs,
        "train_count": len(rows), "train_positive": int(y.sum()),
        "train_accuracy_at_half": float(((scores >= 0.5) == y).mean()),
        "source_only": "logits_features_without_checker_or_oracle",
    }


def calibrate(rows: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    """Choose maximum calibration stops subject to zero observed false stops."""
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("calibration must use disjoint CAL roots")
    x = np.array([row["features"] for row in rows], dtype=np.float64)
    scores = _logit_model(x, params)
    labels = np.array([row["first_pass_verified"] for row in rows], dtype=bool)
    # No valid threshold always stops incorrectly less often than an unsafe one.
    candidates = [1.000001] + [float(s) for s in sorted(set(scores), reverse=True)]
    chosen = {"threshold": 1.000001, "stop_count": 0, "false_stops": 0}
    for cutoff in candidates:
        should_stop = scores >= cutoff
        false_stops = int((should_stop & ~labels).sum())
        stop_count = int(should_stop.sum())
        if false_stops == 0 and stop_count > chosen["stop_count"]:
            chosen = {"threshold": cutoff, "stop_count": stop_count, "false_stops": 0}
    return {
        **chosen, "calibration_cases": len(rows),
        "calibration_valid_first": int(labels.sum()),
        "random_stop_probability": chosen["stop_count"] / len(rows),
        "threshold_selection": "maximum_calibration_stops_with_zero_false_stops",
        "calibration_is_scientific_confirmation": False,
    }


def _root_seed(split: str, index: int) -> int:
    return SPLIT_SEEDS[split] + 101 * index


def _row(
    split: str, idx: int, q: str, trace: list[dict[str, Any]], checkpoint_sha: str,
) -> dict[str, Any]:
    if len(trace) != 2:
        raise ValueError("two-depth source required for first-step valuation")
    first, second = trace
    return {
        "schema_version": "bpd.trm_exit_training_row.v1",
        "root_id": f"{split}-sudoku-{_root_seed(split, idx)}",
        "task_digest": _sha({"question": q}),
        "split": split,
        "model_sha256": checkpoint_sha,
        "question": q,
        "features": _features(first),
        "first_pass_verified": bool(first["sudoku_checker_passed"]),
        "second_pass_verified": bool(second["sudoku_checker_passed"]),
        "second_pass_rescue": bool(second["sudoku_checker_passed"] and not first["sudoku_checker_passed"]),
        "first_candidate": first["candidate"],
        "second_candidate": second["candidate"],
        "training_eligible": split == "train",
        "oracle_target_visible_to_controller": False,
        "future_step_visible_to_controller": False,
    }


def _fixed_rows(model: PureSudokuTRM, *, split: str, n: int, holes: int, sha: str) -> list[dict[str, Any]]:
    rows = []
    for idx in range(n):
        q, solution = _permute_sudoku(_root_seed(split, idx), holes=holes)
        trace, _ = native_rollout(model, q, max_outer_steps=2, stop_when_verified=False)
        if any(v["sudoku_checker_passed"] != (v["candidate"] == solution) for v in trace):
            raise ValueError("task oracle and exact verifier disagree")
        rows.append(_row(split, idx, q, trace, sha))
    if len({row["task_digest"] for row in rows}) != len(rows):
        raise ValueError("duplicate task source inside split")
    return rows


def _metrics(traces: list[tuple[str, list[dict[str, Any]], dict[str, Any]]]) -> dict[str, Any]:
    solved = sum(steps[-1]["sudoku_checker_passed"] for _, steps, _ in traces)
    invalid_exits = sum(len(steps) == 1 and not steps[0]["sudoku_checker_passed"]
                        for _, steps, _ in traces)
    return {
        "solved": solved, "total": len(traces),
        "total_outer_steps": sum(len(steps) for _, steps, _ in traces),
        "stopped_after_first": sum(len(steps) == 1 for _, steps, _ in traces),
        "incorrect_early_exits": invalid_exits,
        "native_seconds": round(sum(step["native_step_seconds"] for _, steps, _ in traces
                                    for step in steps), 6),
        "checker_seconds": round(sum(step["exact_check_seconds"] for _, steps, _ in traces
                                     for step in steps), 6),
    }


def evaluate(
    model: PureSudokuTRM,
    rows: list[dict[str, Any]],
    params: dict[str, Any],
    calibration: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute policies independently with real model-state continuation."""
    if any(row["split"] != "development" or row["training_eligible"] for row in rows):
        raise ValueError("test evaluation requires development-only new roots")
    policies = ("fixed2", "exact_verifier", "learned", "random_matched")
    traces: dict[str, list[tuple[str, list[dict[str, Any]], dict[str, Any]]]] = {
        name: [] for name in policies
    }
    records = []
    cutoff = calibration["threshold"]
    p_random = calibration["random_stop_probability"]
    for index, row in enumerate(rows):
        q = row["question"]
        model_outputs = {}
        # Rotate execution order deterministically across roots.
        order = policies[index % len(policies):] + policies[:index % len(policies)]
        for policy in order:
            if policy == "fixed2":
                steps, result = native_rollout(model, q, max_outer_steps=2, stop_when_verified=False)
            elif policy == "exact_verifier":
                steps, result = native_rollout(model, q, max_outer_steps=2, stop_when_verified=True)
            elif policy == "learned":
                def decide(features: dict[str, float]) -> bool:
                    vector = np.array([features[name] for name in FEATURES], dtype=np.float64)
                    return bool(_logit_model(vector, params) >= cutoff)
                steps, result = native_rollout(
                    model, q, max_outer_steps=2, stop_when_verified=False, stop_decider=decide,
                )
            else:
                control_rng = random.Random(_root_seed("development", index) ^ 0xF107)
                random_decision = control_rng.random() < p_random
                steps, result = native_rollout(
                    model, q, max_outer_steps=2, stop_when_verified=False,
                    stop_decider=lambda _features, s=random_decision: s,
                )
            model_outputs[policy] = (steps, result)
            traces[policy].append((row["root_id"], steps, result))
        fixed_steps = model_outputs["fixed2"][0]
        for policy, (steps, _) in model_outputs.items():
            if steps[0]["candidate"] != fixed_steps[0]["candidate"]:
                raise ValueError("first-visit model output differs between paired policies")
            if len(steps) == 2 and steps[1]["candidate"] != fixed_steps[1]["candidate"]:
                raise ValueError("second-visit continuation changed across policies")
        records.append({
            "root_id": row["root_id"], "task_digest": row["task_digest"],
            "split": "development", "training_eligible": False,
            "first_step_valid": row["first_pass_verified"],
            "second_step_valid": row["second_pass_verified"],
            "learned_score": float(_logit_model(np.array(row["features"]), params)),
            "calibrated_threshold": cutoff, "execution_order": list(order),
            "policies": {
                name: {
                    "visits": len(steps), "correct": bool(steps[-1]["sudoku_checker_passed"]),
                    "first_candidate": steps[0]["candidate"],
                    "final_candidate": steps[-1]["candidate"],
                    "stop_reason": result["stop_reason"],
                    "native_seconds": round(sum(x["native_step_seconds"] for x in steps), 6),
                } for name, (steps, result) in model_outputs.items()
            },
        })
    scores = {name: _metrics(traces[name]) for name in policies}
    return {
        "scores": scores,
        "learned_vs_fixed_steps_saved": 1 - scores["learned"]["total_outer_steps"] / scores["fixed2"]["total_outer_steps"],
        "learned_vs_fixed_correct_delta": scores["learned"]["solved"] - scores["fixed2"]["solved"],
        "verifier_vs_fixed_steps_saved": 1 - scores["exact_verifier"]["total_outer_steps"] / scores["fixed2"]["total_outer_steps"],
        "independent_root_count": len(rows),
    }, records


def run(
    model: PureSudokuTRM,
    *,
    sha: str,
    pretrained: bool,
    counts: dict[str, int] | None = None,
    holes: int = 50,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    counts = dict(counts or DEFAULT_COUNTS)
    if set(counts) != set(SPLIT_SEEDS) or any(type(n) is not int or not 4 <= n <= 160
                                              for n in counts.values()):
        raise ValueError("counts must be train/calibration/development with 4-160 roots")
    seen = set()
    started = time.monotonic()
    stages = {}
    for split in ("train", "calibration", "development"):
        rows = _fixed_rows(model, split=split, n=counts[split], holes=holes, sha=sha)
        overlap = seen & {r["task_digest"] for r in rows}
        if overlap:
            raise ValueError("cross-split duplicated root questions")
        seen.update(r["task_digest"] for r in rows)
        stages[split] = rows
    parameters = fit_logistic(stages["train"])
    calibration = calibrate(stages["calibration"], parameters)
    results, records = evaluate(model, stages["development"], parameters, calibration)
    for row in stages["development"]:
        row["training_eligible"] = False
    report = {
        "schema_version": SCHEMA,
        "evidence_class": "native_pretrained_development_learned_exit" if pretrained else "untrained_smoke",
        "model_sha256": sha, "source_seeds": dict(SPLIT_SEEDS),
        "root_counts": counts, "holes": holes,
        "feature_names": list(FEATURES),
        "training": parameters, "calibration": calibration,
        "evaluation": results,
        "seconds_total_including_data_collection": round(time.monotonic() - started, 3),
        "claim_boundary": (
            "Real pretrained TRM with a learned logits-only stop model; exact verifier labels "
            "supervise TRAIN and CAL but are not provided to the learned controller at inference. "
            "Evaluation remains development only and is not a sealed scientific confirmation."
        ),
    }
    return report, {**stages, "evaluation": records}


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=("mini", "gaoxin492"), default="mini")
    parser.add_argument("--train-roots", type=int, default=96)
    parser.add_argument("--cal-roots", type=int, default=48)
    parser.add_argument("--dev-roots", type=int, default=48)
    parser.add_argument("--holes", type=int, default=50)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("create-only output directory required")
    torch.set_num_threads(2)
    if args.model == "gaoxin492":
        model = load_pretrained_trm("gaoxin492", dtype=torch.float32, verify_sha256=True)
        sha = TRM_CHECKPOINTS["gaoxin492"]["sha256"]
        pretrained = True
    else:
        model = build_mini_trm(hidden_size=32, num_heads=4, l_layers=1,
                               h_cycles=2, l_cycles=1, puzzle_emb_len=4, seq_len=81, seed=43)
        sha = _sha({"untrained_mini_seed": 43})
        pretrained = False
    report, rows = run(
        model, sha=sha, pretrained=pretrained,
        counts={"train": args.train_roots, "calibration": args.cal_roots, "development": args.dev_roots},
        holes=args.holes,
    )
    args.out.mkdir(parents=True)
    paths = {}
    for key, values in rows.items():
        target = args.out / f"{key}.jsonl"
        _save_rows(target, values)
        paths[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    report["data_sha256"] = paths
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                        encoding="utf-8")
    print(json.dumps({
        "out": str(args.out), "evidence_class": report["evidence_class"],
        "calibration": report["calibration"],
        "evaluation": report["evaluation"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
