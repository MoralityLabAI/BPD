"""BPD-PD-02: 24-new-seed hard-gate trained-dictionary cohort.

Real BPDModel/TiedComponentLinear masks on a shared physical W=I2. Source
contract: CONFIG.json and PROTOCOL.md, pinned before registered execution.
No neural mechanism identification claim is made from this toy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.theory.phase_disjoint_control import (
    PhaseTeacher, SharedPhaseGate, draw_batch, read_axis_match,
)

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
SEEDS = tuple(CONFIG["seeds"])
ARMS = tuple(a["key"] for a in CONFIG["arms"])


class BinarySTE(torch.autograd.Function):
    """Exact binary forward; identity gradient through p and sigmoid logits."""

    @staticmethod
    def forward(ctx, probability: torch.Tensor) -> torch.Tensor:
        return (probability >= 0.5).to(dtype=probability.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


def make_model(seed: int, *, frozen: bool) -> tuple[BPDModel, SharedPhaseGate, str]:
    torch.manual_seed(seed)
    bpd = BPDModel(
        PhaseTeacher(),
        {"core": 2},
        max_passes=4,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=frozen,
    )
    gate = SharedPhaseGate()
    if frozen:
        for p in bpd.component_parameters():
            p.requires_grad_(False)
    state = {
        "V": bpd.wrappers["core"].V.detach().cpu().tolist(),
        "U": bpd.wrappers["core"].U.detach().cpu().tolist(),
        "gate": {k: v.detach().cpu().tolist() for k, v in gate.state_dict().items()},
    }
    fingerprint = hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return bpd, gate, fingerprint


def masks_for_probabilities(p: torch.Tensor, *, hard: bool) -> torch.Tensor:
    if hard:
        return BinarySTE.apply(p)
    return p


def masked_rollout(
    bpd: BPDModel, gate: SharedPhaseGate, x: torch.Tensor, *, hard: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = gate(x)
    masks = masks_for_probabilities(probabilities, hard=hard)
    if hard and not torch.all((masks.detach() == 0) | (masks.detach() == 1)):
        raise AssertionError("hard gate is not exactly binary")
    schedule = PassMaskSchedule(
        comp_mask=masks,
        delta_mask=torch.zeros_like(masks[..., 0]),
    )
    with bpd.masked_components({"core": schedule}):
        out = torch.stack(
            [bpd.target_model(x[:, t]) for t in range(4)], dim=1
        )
    return out, masks, probabilities


def objective(
    bpd: BPDModel,
    gate: SharedPhaseGate,
    x: torch.Tensor,
    *,
    hard: bool,
    switch_coefficient: float = 0.0,
) -> dict[str, torch.Tensor]:
    y, mask, p = masked_rollout(bpd, gate, x, hard=hard)
    recon = (y - x).square().mean()
    count = mask.mean()
    faith = bpd.wrappers["core"].weight_delta().square().mean()
    switch = (mask[:, 1:] - mask[:, :-1]).abs().mean()
    return {
        "total": recon + .30 * faith + .05 * count + switch_coefficient * switch,
        "reconstruction": recon,
        "gate_fraction": count,
        "faithfulness": faith,
        "switch": switch,
        "forward_binary": ((mask.detach() == 0) | (mask.detach() == 1)).all(),
        "probabilities": p,
    }


def assignment_fraction(mask: torch.Tensor) -> float:
    """Fraction of boards satisfying all 4 hard gates under one global permutation."""
    assert mask.ndim == 4 and mask.shape[1:] == (4, 1, 2)
    grid = mask.detach()[:, :, 0, :]
    p0 = torch.tensor(
        [[1, 0], [1, 0], [0, 1], [0, 1]],
        dtype=grid.dtype, device=grid.device,
    )
    p1 = 1.0 - p0
    a = (grid == p0.unsqueeze(0)).all(dim=(1, 2)).float().mean()
    b = (grid == p1.unsqueeze(0)).all(dim=(1, 2)).float().mean()
    return max(float(a), float(b))


@torch.no_grad()
def evaluate(
    bpd: BPDModel,
    gate: SharedPhaseGate,
    x: torch.Tensor,
) -> dict:
    binary_y, binary_mask, probability = masked_rollout(bpd, gate, x, hard=True)
    soft_y, _, _ = masked_rollout(bpd, gate, x, hard=False)
    mean = binary_mask.mean(dim=(0, 2)).tolist()
    return {
        "binary_mse": float((binary_y - x).square().mean()),
        "soft_mse": float((soft_y - x).square().mean()),
        "axis_match": read_axis_match(bpd),
        "binary_gate_fraction": float(binary_mask.mean()),
        "probability_mean": float(probability.mean()),
        "binary_switch": float((binary_mask[:, 1:] - binary_mask[:, :-1]).abs().mean()),
        "phase_assignment_fraction": assignment_fraction(binary_mask),
        "mean_hard_gate_by_pass": mean,
        "mean_probability_by_pass": probability.mean(dim=(0, 2)).tolist(),
        "hard_forward_exact_binary": bool(
            ((binary_mask == 0) | (binary_mask == 1)).all()
        ),
        "faithfulness_mse": float(bpd.wrappers["core"].weight_delta().square().mean()),
    }


def gate_success(
    row: dict,
    *,
    factor_update_norm: float,
    first_grad_norm: float,
) -> bool:
    m = CONFIG["metrics"]
    return bool(
        row["binary_mse"] <= m["heldout_binary_mse_max"]
        and row["axis_match"] >= m["axis_cosine_min"]
        and row["phase_assignment_fraction"] >= m["phase_consistent_fraction_min"]
        and row["hard_forward_exact_binary"]
        and factor_update_norm > 0
        and first_grad_norm > 0
    )


def run_arm(
    seed: int, key: str, *, stage1_steps: int | None = None,
    stage2_steps: int | None = None,
    batch: int | None = None,
    calibration_count: int | None = None,
    heldout_count: int | None = None,
) -> dict:
    if key not in ARMS:
        raise ValueError(f"unknown arm {key}")
    torch.set_num_threads(1)
    conf = CONFIG["training"]
    s2 = CONFIG["stage2"]
    source = CONFIG["source"]
    stage1_steps = conf["stage1_steps"] if stage1_steps is None else stage1_steps
    stage2_steps = s2["updates_per_nonzero_coefficient"] if stage2_steps is None else stage2_steps
    batch = source["batch"] if batch is None else batch
    calibration_count = source["calibration_count"] if calibration_count is None else calibration_count
    heldout_count = source["heldout_count"] if heldout_count is None else heldout_count
    frozen = key == "hard_frozen"
    hard = key != "soft_disjoint"
    mixed = key == "hard_mixed"
    bpd, gate, initial_hash = make_model(seed, frozen=frozen)
    factors = bpd.component_parameters()
    original = [p.detach().clone() for p in factors]
    training = torch.Generator().manual_seed(source["train_generator_offset"] + seed)
    optimizer = torch.optim.Adam(
        list(gate.parameters()) + ([] if frozen else factors),
        lr=conf["learning_rate"], weight_decay=0.0,
    )
    first_factor_grad = 0.0
    first_gate_grad = 0.0
    start = time.monotonic()
    for step in range(stage1_steps):
        x = draw_batch(training, batch, mixed=mixed)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(bpd, gate, x, hard=hard)
        loss["total"].backward()
        if step == 0:
            first_factor_grad = math.sqrt(sum(
                float(p.grad.square().sum()) for p in factors if p.grad is not None
            ))
            first_gate_grad = math.sqrt(sum(
                float(p.grad.square().sum()) for p in gate.parameters()
                if p.grad is not None
            ))
        optimizer.step()
    factor_update_norm = math.sqrt(sum(
        float((p.detach() - before).square().sum())
        for p, before in zip(factors, original)
    ))
    stage1_gate = copy.deepcopy(gate.state_dict())
    factor_snapshot = [p.detach().clone() for p in factors]

    # Stage-2 uses calibration only. The held-out generator is not touched
    # until after all stage-two candidates and selection are frozen.
    calibration = draw_batch(
        torch.Generator().manual_seed(source["calibration_generator_offset"] + seed),
        calibration_count, mixed=mixed,
    )
    stage1_calibration = evaluate(bpd, gate, calibration)
    candidate_rows = []
    selected_coefficient = 0.0
    chosen_switch = stage1_calibration["binary_switch"]
    chosen_gate = copy.deepcopy(stage1_gate)
    if key == s2["applies_to"]:
        for p in factors:
            p.requires_grad_(False)
        for coeff in s2["coefficients"]:
            if coeff == 0:
                continue
            gate.load_state_dict(stage1_gate)
            stage2_optimizer = torch.optim.Adam(
                gate.parameters(), lr=s2["learning_rate"], weight_decay=0.0,
            )
            stage2_train = torch.Generator().manual_seed(400000 + seed)
            for _ in range(stage2_steps):
                x = draw_batch(stage2_train, batch, mixed=mixed)
                stage2_optimizer.zero_grad(set_to_none=True)
                objective(
                    bpd, gate, x, hard=True, switch_coefficient=coeff,
                )["total"].backward()
                stage2_optimizer.step()
            candidate = evaluate(bpd, gate, calibration)
            feasible = bool(
                candidate["binary_mse"] <= stage1_calibration["binary_mse"]
                + s2["calibration_mse_tolerance_absolute"]
                and candidate["binary_gate_fraction"]
                <= stage1_calibration["binary_gate_fraction"]
                + s2["calibration_gate_count_tolerance_absolute"]
            )
            candidate_rows.append({
                "coefficient": coeff,
                "calibration_feasible": feasible,
                "calibration_binary_mse": candidate["binary_mse"],
                "calibration_gate_fraction": candidate["binary_gate_fraction"],
                "calibration_switch": candidate["binary_switch"],
            })
            if feasible and candidate["binary_switch"] < chosen_switch:
                chosen_switch = candidate["binary_switch"]
                chosen_gate = copy.deepcopy(gate.state_dict())
                selected_coefficient = coeff
    factors_frozen = all(
        torch.equal(p.detach(), before) for p, before in zip(factors, factor_snapshot)
    )
    if not factors_frozen:
        raise AssertionError("dictionary changed during stage-two gate-only sweep")

    # Freeze switch coefficient selection; evaluate held-out exactly once for
    # primary stage-1 and selected secondary stage-2 parameter states.
    heldout = draw_batch(
        torch.Generator().manual_seed(source["heldout_generator_offset"] + seed),
        heldout_count, mixed=mixed,
    )
    gate.load_state_dict(stage1_gate)
    primary = evaluate(bpd, gate, heldout)
    gate.load_state_dict(chosen_gate)
    secondary = evaluate(bpd, gate, heldout)
    passed = (
        gate_success(
            primary,
            factor_update_norm=factor_update_norm,
            first_grad_norm=first_factor_grad,
        ) if not frozen else False
    )

    return {
        "seed": seed,
        "arm": key,
        "initial_parameter_sha256": initial_hash,
        "trained_dictionary": not frozen,
        "training_steps": stage1_steps,
        "calibration_count": calibration_count,
        "heldout_count": heldout_count,
        "factor_first_step_grad_norm": first_factor_grad,
        "gate_first_step_grad_norm": first_gate_grad,
        "factor_stage1_update_norm": factor_update_norm,
        "factors_bitwise_unchanged_in_stage2": factors_frozen,
        "stage1_calibration": stage1_calibration,
        "stage1_primary_heldout": primary,
        "selected_stage2_heldout": secondary,
        "stage2_candidates": candidate_rows,
        "stage2_selected_coefficient": selected_coefficient,
        "directional_success": passed,
        "elapsed_seconds": round(time.monotonic() - start, 4),
    }


def exact_mcnemar_p(hard_only: int, soft_only: int) -> float:
    n = hard_only + soft_only
    if n == 0:
        return 1.0
    smaller = min(hard_only, soft_only)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(smaller + 1)) / (2 ** n))


def wilson_ci(success: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        raise ValueError("positive seed count required")
    p = success / total
    d = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / d
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [max(0, mid - spread), min(1, mid + spread)]


def summarize(arms: list[dict], *, registered: bool) -> dict:
    by_key = {(r["seed"], r["arm"]): r for r in arms}
    seeds = sorted({r["seed"] for r in arms})
    hard = [by_key[(seed, "hard_disjoint")] for seed in seeds]
    soft = [by_key[(seed, "soft_disjoint")] for seed in seeds]
    hard_s = [r["directional_success"] for r in hard]
    soft_s = [r["directional_success"] for r in soft]
    hard_only = sum(h and not s for h, s in zip(hard_s, soft_s))
    soft_only = sum(s and not h for h, s in zip(hard_s, soft_s))
    n = len(seeds)
    wins = hard_only - soft_only
    p_value = exact_mcnemar_p(hard_only, soft_only)
    criterion = CONFIG["primary"]
    valid = registered and n == criterion["total"]
    return {
        "hard_success_count": sum(hard_s),
        "soft_success_count": sum(soft_s),
        "hard_wilson_95": wilson_ci(sum(hard_s), n),
        "soft_wilson_95": wilson_ci(sum(soft_s), n),
        "hard_only": hard_only,
        "soft_only": soft_only,
        "net_paired_success_wins": wins,
        "exact_two_sided_mcnemar_p": p_value,
        "registered_primary_criterion_passed": bool(
            valid
            and sum(hard_s) >= criterion["hard_success_min"]
            and wins >= criterion["net_paired_success_wins_min"]
            and p_value <= criterion["mcnemar_two_sided_p_max"]
        ),
        "is_full_registered_run": valid,
        "seed_count": n,
        "per_seed": [
            {
                "seed": seed,
                "hard_success": h,
                "soft_success": s,
                "hard_binary_mse": by_key[(seed, "hard_disjoint")]["stage1_primary_heldout"]["binary_mse"],
                "soft_binary_mse": by_key[(seed, "soft_disjoint")]["stage1_primary_heldout"]["binary_mse"],
                "hard_axis_match": by_key[(seed, "hard_disjoint")]["stage1_primary_heldout"]["axis_match"],
                "soft_axis_match": by_key[(seed, "soft_disjoint")]["stage1_primary_heldout"]["axis_match"],
            }
            for seed, h, s in zip(seeds, hard_s, soft_s)
        ],
    }


def run_cohort() -> dict:
    started = time.monotonic()
    arms = []
    for seed in SEEDS:
        for key in ARMS:
            arms.append(run_arm(seed, key))
    for seed in SEEDS:
        hashes = {
            r["initial_parameter_sha256"]
            for r in arms if r["seed"] == seed and r["trained_dictionary"]
        }
        if len(hashes) != 1:
            raise AssertionError(f"trainable arms have different initialization for seed {seed}")
    report = {
        "study_id": CONFIG["study_id"],
        "study_status": "registered_first_full_run",
        "source_protocol_sha256": hashlib.sha256(
            (ROOT / "PROTOCOL.md").read_bytes()
        ).hexdigest(),
        "source_config_sha256": hashlib.sha256(
            (ROOT / "CONFIG.json").read_bytes()
        ).hexdigest(),
        "source_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "summary": summarize(arms, registered=True),
        "arms": arms,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_cohort()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
