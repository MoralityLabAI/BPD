"""Pretrained 7M Attention-TRM: CPU-bounded native recurrent-state intervention pilot.

Uses the existing SHA-pinned PureSudokuTRM checkpoint loader. This model
contains learned recurrent attention and MLP blocks; it does not implement
Inkling's Attention Residual cross-loop block readers.

The selected readout direction is fixed from the digit head and is NOT
chosen using the solution, model's outcome, or held-out contrast.
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

import torch
from torch import Tensor

from mechinterp.bpd.adapters.trm import build_mini_trm, load_pretrained_trm
from mechinterp.loopscope02.scripts.trm_sudoku import PureSudokuTRM, encode_questions

SCHEMA = "bpd.native_trm_boundary_pilot.v1"
ARMS = ("clean", "zero", "head_direction", "random_direction")
DEFAULT_SEED = 20261121


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def count_solutions(question: str, *, limit: int = 2) -> int:
    """Independent Sudoku constraint solver; stop after limit valid completions."""
    if len(question) != 81 or any(ch not in "123456789." for ch in question):
        raise ValueError("invalid Sudoku question")
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("invalid solution count limit")
    board = [0 if ch == "." else int(ch) for ch in question]
    row_masks, col_masks, box_masks = [0] * 9, [0] * 9, [0] * 9
    blank = []
    for index, value in enumerate(board):
        r, col = divmod(index, 9)
        box = (r // 3) * 3 + col // 3
        if not value:
            blank.append(index)
            continue
        bit = 1 << (value - 1)
        if row_masks[r] & bit or col_masks[col] & bit or box_masks[box] & bit:
            return 0
        row_masks[r] |= bit
        col_masks[col] |= bit
        box_masks[box] |= bit

    solutions = 0

    def search() -> None:
        nonlocal solutions
        if solutions >= limit:
            return
        target, options, fewest = -1, 0, 10
        for index in blank:
            if board[index]:
                continue
            r, col = divmod(index, 9)
            box = (r // 3) * 3 + col // 3
            mask = 0x1FF & ~(row_masks[r] | col_masks[col] | box_masks[box])
            n = mask.bit_count()
            if n < fewest:
                target, options, fewest = index, mask, n
            if n == 0:
                return
            if n == 1:
                break
        if target < 0:
            solutions += 1
            return
        r, col = divmod(target, 9)
        box = (r // 3) * 3 + col // 3
        while options and solutions < limit:
            bit = options & -options
            options -= bit
            board[target] = bit.bit_length()
            row_masks[r] |= bit
            col_masks[col] |= bit
            box_masks[box] |= bit
            search()
            row_masks[r] ^= bit
            col_masks[col] ^= bit
            box_masks[box] ^= bit
            board[target] = 0

    search()
    return solutions


def _permute_sudoku(seed: int, holes: int = 22) -> tuple[str, str]:
    """Produce reproducible clue/target pairs from a Sudoku-valid solved board."""
    if type(seed) is not int or type(holes) is not int or not 0 <= holes <= 40:
        raise ValueError("seed and hole count out of bounds")
    rng = random.Random(seed)
    bands = list(range(3))
    stacks = list(range(3))
    rng.shuffle(bands)
    rng.shuffle(stacks)
    rows = []
    cols = []
    for group in bands:
        members = list(range(3 * group, 3 * group + 3))
        rng.shuffle(members)
        rows.extend(members)
    for group in stacks:
        members = list(range(3 * group, 3 * group + 3))
        rng.shuffle(members)
        cols.extend(members)
    digits = list("123456789")
    rng.shuffle(digits)
    solution = "".join(
        digits[((rows[r] * 3 + rows[r] // 3 + cols[c]) % 9)]
        for r in range(9) for c in range(9)
    )
    question = list(solution)
    removed = 0
    # Each removal retains a uniquely solvable puzzle independent of the model.
    for position in rng.sample(range(81), 81):
        if removed >= holes:
            break
        question[position] = "."
        if count_solutions("".join(question)) == 1:
            removed += 1
        else:
            question[position] = solution[position]
    if removed != holes:
        raise ValueError("could not construct uniquely solvable puzzle at requested clue count")
    q = "".join(question)
    assert verify_sudoku(q, solution) and count_solutions(q) == 1
    return q, solution


def verify_sudoku(question: str, candidate: str) -> bool:
    if len(question) != 81 or len(candidate) != 81:
        return False
    if any(ch not in "123456789." for ch in question):
        return False
    if any(ch not in "123456789" for ch in candidate):
        return False
    for p, clue in enumerate(question):
        if clue != "." and candidate[p] != clue:
            return False
    groups = []
    for r in range(9):
        groups.append([r * 9 + c for c in range(9)])
    for c in range(9):
        groups.append([r * 9 + c for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            groups.append([(br + r) * 9 + bc + c for r in range(3) for c in range(3)])
    return all(set(candidate[p] for p in indices) == set("123456789") for indices in groups)


def _direction(model: PureSudokuTRM, arm: str, *, root_seed: int) -> Tensor:
    width = model.hidden_size
    if arm == "head_direction":
        direction = (
            model.lm_head.weight[2].detach().float()
            - model.lm_head.weight[3].detach().float()
        )
    elif arm == "random_direction":
        rng = torch.Generator(device="cpu").manual_seed(root_seed)
        direction = torch.randn(width, generator=rng)
    else:
        raise ValueError("unknown direction")
    return direction / direction.norm().clamp_min(1e-12)


@torch.no_grad()
def run_one(
    model: PureSudokuTRM,
    question: str,
    *,
    outer_steps: int = 1,
    arm: str = "clean",
    root_seed: int = 0,
    relative_dose: float = 0.2,
    patch_h_cycle: int = 1,
) -> tuple[Tensor, dict[str, Any]]:
    """Replay actual pretrained recurrent blocks; patch z_H after H-cycle commit."""
    if arm not in ARMS:
        raise ValueError("unknown intervention arm")
    if type(outer_steps) is not int or not 1 <= outer_steps <= 3:
        raise ValueError("outer_steps must be 1..3")
    if type(patch_h_cycle) is not int or not 1 <= patch_h_cycle < model.h_cycles:
        raise ValueError("patch must leave at least one subsequent H-cycle")
    if not math.isfinite(relative_dose) or not 0 < relative_dose <= 1:
        raise ValueError("relative_dose must be finite in (0,1]")
    if question.count(".") == 0:
        raise ValueError("patch cell must be a blank Sudoku location")
    idx = question.index(".")
    tokens = encode_questions([question], model.embed_tokens.weight.device)
    embeddings = model.encode_inputs(tokens)
    z_h, z_l = model.init_states(1)
    patch_receipt: dict[str, Any] | None = None
    early_logits = None
    for outer in range(outer_steps):
        for cycle in range(model.h_cycles):
            for _ in range(model.l_cycles):
                z_l = model.run_l_level(z_l, z_h + embeddings)
            z_h = model.run_l_level(z_h, z_l)
            if outer == 0 and cycle + 1 == patch_h_cycle:
                pre = z_h[:, model.puzzle_emb_len + idx, :].float().clone()
                early_logits = model.lm_head(pre.to(model.lm_head.weight.dtype))[0, 2:11].float()
                dose = float(pre.norm().item() * relative_dose)
                if arm in {"head_direction", "random_direction"}:
                    direction = _direction(model, arm, root_seed=root_seed).to(
                        pre.device, dtype=pre.dtype,
                    )
                    delta = direction * dose
                    patched = z_h.clone()
                    patched[:, model.puzzle_emb_len + idx, :] += delta.to(z_h.dtype)[None, :]
                    z_h = patched
                    actual = float(
                        (z_h[:, model.puzzle_emb_len + idx, :].float() - pre).norm().item()
                    )
                else:
                    actual = 0.0
                patch_receipt = {
                    "site": "z_H_after_h_cycle",
                    "outer_step": 1,
                    "h_cycle": patch_h_cycle,
                    "selected_cell": idx,
                    "selected_from": "first_blank_in_question",
                    "dose_relative_to_raw_state": relative_dose,
                    "nominal_raw_l2": dose if arm in {"head_direction", "random_direction"} else 0.0,
                    "actual_raw_l2": actual,
                    "pre_state_norm_l2": float(pre.norm()),
                }
    assert early_logits is not None and patch_receipt is not None
    logits = model.digit_logits(z_h)[0].float()
    prediction = "".join(str(int(v) + 1) for v in logits.argmax(dim=-1).tolist())
    summary = {
        **patch_receipt, "arm": arm,
        "predicted": prediction,
        "initial_selected_cell_logits": early_logits.tolist(),
        "final_selected_cell_logits": logits[idx].tolist(),
        "recurrent_calls": outer_steps * model.h_cycles * (model.l_cycles + 1),
        "model_is_pretrained": None,
    }
    return logits, summary


@torch.no_grad()
def verify_clean_parity(model: PureSudokuTRM, question: str) -> float:
    """Independent manual unroll must reproduce the native act_step output."""
    inputs = encode_questions([question], model.embed_tokens.weight.device)
    x_emb = model.encode_inputs(inputs)
    zh0, zl0 = model.init_states(1)
    zh_native, zl_native = model.act_step(zh0.clone(), zl0.clone(), x_emb)
    zh, zl = zh0, zl0
    for _ in range(model.h_cycles):
        for _ in range(model.l_cycles):
            zl = model.run_l_level(zl, zh + x_emb)
        zh = model.run_l_level(zh, zl)
    max_difference = max(
        float((zh - zh_native).float().abs().max().item()),
        float((zl - zl_native).float().abs().max().item()),
    )
    if max_difference > 1e-5:
        raise ValueError(f"manual native recurrence parity failed: {max_difference}")
    return max_difference


def run_campaign(
    model: PureSudokuTRM,
    *,
    pretrained: bool,
    model_id: str,
    model_sha: str,
    seed: int = DEFAULT_SEED,
    roots: int = 2,
    outer_steps: int = 1,
    relative_dose: float = 0.2,
    holes: int = 22,
    patch_h_cycle: int = 1,
) -> dict[str, Any]:
    if type(roots) is not int or not 1 <= roots <= 8:
        raise ValueError("roots must be in 1..8")
    if len(model_sha) != 64:
        raise ValueError("checkpoint SHA-256 required")
    model.eval()
    measurements = []
    parity = None
    started = time.monotonic()
    for i in range(roots):
        root_seed = seed + 101 * i
        question, target = _permute_sudoku(root_seed, holes=holes)
        if parity is None:
            parity = verify_clean_parity(model, question)
        baseline = None
        for arm in ARMS:
            begun = time.monotonic()
            logits, row = run_one(
                model, question, outer_steps=outer_steps,
                arm=arm, root_seed=root_seed + 11,
                relative_dose=relative_dose, patch_h_cycle=patch_h_cycle,
            )
            if arm == "clean":
                baseline = logits
            assert baseline is not None
            row.update(
                schema_version=SCHEMA,
                experiment_split="development",
                case_id=f"sudoku-{seed}-{i}",
                root_id=f"source-sudoku-{root_seed}",
                question=question,
                target_sha256=hashlib.sha256(target.encode()).hexdigest(),
                model_id=model_id,
                model_sha256=model_sha,
                model_pretrained=pretrained,
                clean_exact_solution=target,
                target_matches_verified_sudoku=verify_sudoku(question, target),
                board_exact=row["predicted"] == target,
                board_valid=verify_sudoku(question, row["predicted"]),
                cell_correct=sum(a == b for a, b in zip(row["predicted"], target)),
                selected_cell_correct=row["predicted"][row["selected_cell"]] == target[row["selected_cell"]],
                target_digit_at_selected_cell=target[row["selected_cell"]],
                mean_abs_logit_change=float((logits - baseline).abs().mean()),
                selected_cell_logits_l2_change=float(
                    (logits[row["selected_cell"]] - baseline[row["selected_cell"]]).norm()
                ),
                seconds=round(time.monotonic() - begun, 4),
            )
            measurements.append(row)
    return {
        "schema_version": SCHEMA,
        "evidence_class": "native_pretrained_development" if pretrained else "untrained_native_plumbing_smoke",
        "model": model_id, "model_sha256": model_sha, "pretrained": pretrained,
        "roots": roots, "arms": list(ARMS), "outer_steps": outer_steps,
        "actual_model_h_cycles": model.h_cycles, "actual_model_l_cycles": model.l_cycles,
        "relative_dose": relative_dose, "holes": holes, "patch_h_cycle": patch_h_cycle,
        "max_clean_parity_difference": parity,
        "all_sudoku_targets_verified": all(row["target_matches_verified_sudoku"] for row in measurements),
        "rows": measurements, "elapsed_seconds": round(time.monotonic() - started, 3),
        "claim_boundary": (
            "TRM native recurrent z_H intervention; no cross-loop Attention Residual reader. "
            "No native Inkling weights, no independent representation discovery or scientific confirmation. "
            "Model outcomes are descriptive on fresh generated development Sudoku worlds."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=("mini", "gaoxin492", "yagizdevre"), default="mini")
    parser.add_argument("--roots", type=int, default=2)
    parser.add_argument("--outer-steps", type=int, default=1)
    parser.add_argument("--relative-dose", type=float, default=0.2)
    parser.add_argument("--holes", type=int, default=22)
    parser.add_argument("--patch-h-cycle", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output file exists")
    torch.set_num_threads(2)
    torch.manual_seed(32)
    if args.model == "mini":
        model = build_mini_trm(
            hidden_size=32, num_heads=4, l_layers=1,
            h_cycles=2, l_cycles=1, puzzle_emb_len=4, seq_len=81, seed=32,
        )
        trained = False
        model_sha = _digest({
            "architecture": "build_mini_trm_32x1",
            "seed": 32,
            "training": False,
        })
    else:
        model = load_pretrained_trm(args.model, dtype=torch.float32, verify_sha256=True)
        from mechinterp.loopscope02.scripts.trm_sudoku import TRM_CHECKPOINTS
        model_sha = TRM_CHECKPOINTS[args.model]["sha256"]
        trained = True
    result = run_campaign(
        model, pretrained=trained, model_id=args.model, model_sha=model_sha,
        roots=args.roots, seed=args.seed, outer_steps=args.outer_steps,
        relative_dose=args.relative_dose, holes=args.holes, patch_h_cycle=args.patch_h_cycle,
    )
    result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out), "evidence_class": result["evidence_class"],
        "roots": result["roots"], "seconds": result["elapsed_seconds"],
        "parity_max": result["max_clean_parity_difference"],
        "arms": {
            arm: {
                "mean_cell_correct": sum(
                    row["cell_correct"] for row in result["rows"] if row["arm"] == arm
                ) / result["roots"],
                "mean_logit_abs_delta": sum(
                    row["mean_abs_logit_change"] for row in result["rows"] if row["arm"] == arm
                ) / result["roots"],
            } for arm in ARMS
        },
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
