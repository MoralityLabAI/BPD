"""Run genuine pretrained TRM with a resumable, exact-verifier recurrent exit.

Each outer step uses the saved z_H/z_L states of the actual pretrained
attention/MLP recurrent model. A Sudoku checker can stop the model once
it returns a complete legal solution without seeing the planted target.
This is development research, not an Inkling Attention Residual study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from mechinterp.bpd.adapters.trm import build_mini_trm, load_pretrained_trm
from mechinterp.bpd.experiments.inkling_reader_v0.native_trm_pilot import (
    _digest, _permute_sudoku, verify_sudoku,
)
from mechinterp.loopscope02.scripts.trm_sudoku import TRM_CHECKPOINTS, PureSudokuTRM, encode_questions

SCHEMA = "bpd.native_trm_verifier_exit.v1"


@torch.no_grad()
def native_rollout(
    model: PureSudokuTRM,
    question: str,
    *,
    max_outer_steps: int,
    stop_when_verified: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One uninterrupted recurrence; later steps reuse the preceding state."""
    if type(max_outer_steps) is not int or not 1 <= max_outer_steps <= 8:
        raise ValueError("max_outer_steps must be in 1..8")
    if len(question) != 81 or "." not in question:
        raise ValueError("a bounded incomplete Sudoku question is required")
    tokens = encode_questions([question], model.embed_tokens.weight.device)
    embedded = model.encode_inputs(tokens)
    z_h, z_l = model.init_states(1)
    blank = [index for index, value in enumerate(question) if value == "."]
    observations: list[dict[str, Any]] = []
    start = time.perf_counter()
    for step in range(1, max_outer_steps + 1):
        began = time.perf_counter()
        z_h, z_l = model.act_step(z_h, z_l, embedded)
        step_compute_sec = time.perf_counter() - began
        logits = model.digit_logits(z_h)[0].float()
        prediction = "".join(str(int(v) + 1) for v in logits.argmax(dim=-1).tolist())
        verify_started = time.perf_counter()
        checked = verify_sudoku(question, prediction)
        check_sec = time.perf_counter() - verify_started
        margin = torch.topk(logits[blank], k=2, dim=-1).values
        margins = (margin[:, 0] - margin[:, 1]).tolist()
        observations.append({
            "outer_step": step,
            "candidate": prediction,
            "sudoku_checker_passed": checked,
            "mean_blank_logit_margin": sum(margins) / len(margins),
            "min_blank_logit_margin": min(margins),
            "native_step_seconds": round(step_compute_sec, 6),
            "exact_check_seconds": round(check_sec, 6),
        })
        if checked and stop_when_verified:
            break
    return observations, {
        "realized_outer_steps": len(observations),
        "stop_reason": "verified" if observations[-1]["sudoku_checker_passed"] else "depth_budget",
        "elapsed_seconds": round(time.perf_counter() - start, 6),
    }


def run(
    model: PureSudokuTRM,
    *,
    model_id: str,
    model_sha: str,
    pretrained: bool,
    seed: int = 20261123,
    roots: int = 24,
    holes: int = 50,
    max_outer_steps: int = 2,
    max_wall_seconds: float = 600.0,
) -> dict[str, Any]:
    if type(roots) is not int or not 1 <= roots <= 32:
        raise ValueError("roots must be in 1..32")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid root seed")
    if not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError("positive finite wall limit required")
    if len(model_sha) != 64:
        raise ValueError("checkpoint identity requires SHA-256")
    if max_outer_steps != 2:
        raise ValueError("v1 comparison uses exactly two permitted outer steps")
    model.eval()
    started = time.monotonic()
    records = []
    for root_index in range(roots):
        if time.monotonic() - started > max_wall_seconds:
            raise TimeoutError("wall budget exceeded between independent roots")
        root_seed = seed + 101 * root_index
        question, planted_solution = _permute_sudoku(root_seed, holes=holes)
        evaluations = {}
        # Alternate run order to avoid assigning warm hardware to one arm.
        policies = ("fixed_2", "verifier_exit") if root_index % 2 == 0 else ("verifier_exit", "fixed_2")
        for policy in policies:
            trace, result = native_rollout(
                model, question, max_outer_steps=max_outer_steps,
                stop_when_verified=policy == "verifier_exit",
            )
            evaluations[policy] = {**result, "trace": trace}
        fixed, adaptive = evaluations["fixed_2"], evaluations["verifier_exit"]
        if adaptive["trace"][0]["candidate"] != fixed["trace"][0]["candidate"]:
            raise RuntimeError("first-visit predictions differed across matched policies")
        if not all(
            item["sudoku_checker_passed"] == (item["candidate"] == planted_solution)
            for policy in evaluations.values() for item in policy["trace"]
        ):
            raise RuntimeError("checker disagrees with planted uniquely solvable solution")
        if adaptive["trace"][-1]["candidate"] != fixed["trace"][
            adaptive["realized_outer_steps"] - 1
        ]["candidate"]:
            raise RuntimeError("resumed recurrent state diverged from fixed-depth control")
        records.append({
            "root_id": f"sudoku-root-{root_seed}",
            "source_split": "development",
            "task_digest": _digest({"question": question}),
            "question": question,
            "planted_solution_sha256": hashlib.sha256(planted_solution.encode()).hexdigest(),
            "blank_count": holes,
            "model_id": model_id,
            "model_sha256": model_sha,
            "measured_arm_order": list(policies),
            "policies": evaluations,
        })
    summary: dict[str, Any] = {}
    for name in ("fixed_2", "verifier_exit"):
        rows = [row["policies"][name] for row in records]
        summary[name] = {
            "verified_solved": sum(row["trace"][-1]["sudoku_checker_passed"] for row in rows),
            "cases": roots,
            "outer_steps_total": sum(row["realized_outer_steps"] for row in rows),
            "mean_outer_steps": sum(row["realized_outer_steps"] for row in rows) / roots,
            "elapsed_wall_seconds": round(sum(row["elapsed_seconds"] for row in rows), 6),
            "total_native_step_seconds": round(sum(
                v["native_step_seconds"] for row in rows for v in row["trace"]
            ), 6),
            "total_checker_seconds": round(sum(
                v["exact_check_seconds"] for row in rows for v in row["trace"]
            ), 6),
            "stop_reasons": dict(sorted(Counter(row["stop_reason"] for row in rows).items())),
        }
    summary["step_count_fraction_saved"] = (
        1 - summary["verifier_exit"]["outer_steps_total"] / summary["fixed_2"]["outer_steps_total"]
    )
    summary["measured_native_step_fraction_saved"] = (
        1 - summary["verifier_exit"]["total_native_step_seconds"]
        / summary["fixed_2"]["total_native_step_seconds"]
    )
    return {
        "schema_version": SCHEMA,
        "evidence_class": "native_pretrained_development" if pretrained else "untrained_plumbing_smoke",
        "model": model_id,
        "model_sha256": model_sha,
        "seed": seed,
        "roots": roots,
        "holes": holes,
        "max_outer_steps": max_outer_steps,
        "checker": "sudoku_9x9_full_grid_clues_and_constraints",
        "check_uses_planted_solution": False,
        "student_training_eligible": False,
        "split": "development",
        "configuration": {
            "threads": torch.get_num_threads(),
            "h_cycles": model.h_cycles,
            "l_cycles": model.l_cycles,
            "internal_weight_tying": True,
        },
        "summary": summary,
        "records": records,
        "study_limitations": (
            "Task selection is development exposed. Sudoku legal-solution verification "
            "provides strong external feedback and does not directly transfer to "
            "general language tasks. No Inkling Attention Residual intervention performed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", choices=("mini", "gaoxin492"), default="mini")
    parser.add_argument("--roots", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--holes", type=int, default=50)
    parser.add_argument("--wall-limit", type=float, default=600)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists")
    torch.set_num_threads(2)
    if args.model == "mini":
        model = build_mini_trm(
            hidden_size=32, num_heads=4, l_layers=1,
            h_cycles=2, l_cycles=1, puzzle_emb_len=4, seq_len=81, seed=43,
        )
        pretrained = False
        sha = _digest({"mini_seed": 43, "weights": "untrained"})
    else:
        model = load_pretrained_trm("gaoxin492", dtype=torch.float32, verify_sha256=True)
        pretrained = True
        sha = TRM_CHECKPOINTS["gaoxin492"]["sha256"]
    result = run(
        model, model_id=args.model, model_sha=sha, pretrained=pretrained,
        roots=args.roots, seed=args.seed, holes=args.holes,
        max_wall_seconds=args.wall_limit,
    )
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({
        "out": str(args.out), "evidence_class": result["evidence_class"],
        "summary": result["summary"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
