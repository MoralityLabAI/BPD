"""BPD-PD-03 frozen 24-seed top-1 vs calibrated binary margin experiment.

The actual BPDModel/TiedComponentLinear carries two trainable rank-one factors
through every disjoint/mixed trainable arm. No 7M checkpoint or held-out labels
are accessed during training or calibration.
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
import torch.nn.functional as F

from mechinterp.bpd.components import PassMaskSchedule
from mechinterp.bpd.theory.phase_disjoint_control import draw_batch, read_axis_match
from mechinterp.bpd.theory.pd02.experiment import (
    BinarySTE, exact_mcnemar_p, make_model, wilson_ci,
)

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
SEEDS = tuple(CONFIG["seed_list"])
ARMS = tuple(arm["name"] for arm in CONFIG["arms"])


def routing_mode(arm: str) -> str:
    if arm.startswith("top1_"):
        return "top1"
    if arm.startswith("margin_"):
        return "margin"
    if arm == "hard_disjoint":
        return "hard"
    raise ValueError(f"Unknown PD03 arm: {arm}")


def source_mixed(arm: str) -> bool:
    return arm.endswith("_mixed")


def dictionary_frozen(arm: str) -> bool:
    return arm.endswith("_frozen")


class TopOneSTE(torch.autograd.Function):
    """Exact one-hot forward, softmax-Jacobian straight-through gradient."""

    @staticmethod
    def forward(ctx, logits):
        soft = logits.softmax(dim=-1)
        ctx.save_for_backward(soft)
        return F.one_hot(logits.argmax(dim=-1), num_classes=2).to(logits.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (soft,) = ctx.saved_tensors
        mean_grad = (grad_output * soft).sum(dim=-1, keepdim=True)
        return soft * (grad_output - mean_grad)


class ThresholdSTE(torch.autograd.Function):
    """Exact zero-or-one forward, sigmoid-probability STE in backward."""

    @staticmethod
    def forward(ctx, probability, threshold):
        return (probability >= threshold).to(probability.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def gate_mask(logits: torch.Tensor, mode: str, *, threshold: float = .5) -> torch.Tensor:
    if mode == "top1":
        return TopOneSTE.apply(logits)
    if mode in ("margin", "hard"):
        return ThresholdSTE.apply(logits.sigmoid(), threshold)
    raise ValueError(mode)


def masked_forward(bpd, gate, x: torch.Tensor, mode: str, *, threshold: float = .5):
    logits = gate.fc(x.abs())
    probs = logits.sigmoid()
    mask = gate_mask(logits, mode, threshold=threshold)
    if not bool(((mask.detach() == 0) | (mask.detach() == 1)).all()):
        raise AssertionError("Every PD03 execution mask must be exact 0/1")
    sched = PassMaskSchedule(
        comp_mask=mask,
        delta_mask=torch.zeros_like(mask[..., 0]),
    )
    with bpd.masked_components({"core": sched}):
        predicted = torch.stack(
            [bpd.target_model(x[:, step]) for step in range(x.shape[1])],
            dim=1,
        )
    return predicted, mask, probs


def loss_parts(bpd, gate, x, mode: str):
    y, mask, probs = masked_forward(bpd, gate, x, mode, threshold=.5)
    recon = (y - x).square().mean()
    faith = bpd.wrappers["core"].weight_delta().square().mean()
    activity = mask.mean()
    near_boundary = F.relu(
        CONFIG["optimization"]["margin_prob_gap"] - (probs - .5).abs()
    ).mean()
    margin_coeff = CONFIG["optimization"]["margin_coeff"] if mode == "margin" else 0.0
    loss = (recon + CONFIG["optimization"]["faithfulness_coeff"] * faith
            + CONFIG["optimization"]["gate_count_coeff"] * activity
            + margin_coeff * near_boundary)
    return {"total": loss, "recon": recon, "faith": faith,
            "activity": activity, "margin": near_boundary}


def read_assignment(bpd) -> int:
    """Which component (0 or 1) corresponds to input coordinate 0?"""
    v = F.normalize(bpd.wrappers["core"].V.detach(), dim=0)
    identity = v[0, 0].abs() + v[1, 1].abs()
    swapped = v[0, 1].abs() + v[1, 0].abs()
    return 0 if float(identity) >= float(swapped) else 1


@torch.no_grad()
def evaluate(bpd, gate, x, mode: str, *, threshold: float = .5) -> dict:
    y, mask, probs = masked_forward(bpd, gate, x, mode, threshold=threshold)
    # This is a separate descriptive continuous probability-path diagnostic.
    original = (probs @ bpd.wrappers["core"].U).float()
    count = x.shape[0]
    assignment = read_assignment(bpd)
    target = [assignment, assignment, 1 - assignment, 1 - assignment]
    observed = mask.detach()[:, :, 0, :]
    expected = torch.zeros_like(observed)
    for t, i in enumerate(target):
        expected[:, t, i] = 1.
    all_correct = (observed == expected).all(dim=(1, 2))
    false_off = int(((expected == 1) & (observed == 0)).sum().item())
    false_on = int(((expected == 0) & (observed == 1)).sum().item())
    assert false_off + false_on >= 0
    if mode == "top1":
        assert bool((observed.sum(dim=-1) == 1).all())
    return {
        "binary_mse": float((y - x).square().mean()),
        "axis_match": read_axis_match(bpd),
        "phase_assignment_fraction": float(all_correct.float().mean()),
        "phase_correct_boards": int(all_correct.sum().item()),
        "false_off_count": false_off,
        "false_on_count": false_on,
        "required_gate_count": int(count * 4),
        "gate_fraction": float(mask.mean()),
        "gate_switch": float((mask[:, 1:] - mask[:, :-1]).abs().mean()),
        "margin_probability_mean": float((probs - .5).abs().mean()),
        "mean_probabilities_by_pass": probs.mean(dim=(0, 2)).tolist(),
        "mean_masks_by_pass": observed.mean(dim=0).tolist(),
        "threshold": threshold,
        "component_assignment_for_input0": assignment,
        "execution_exact_binary": bool(((mask == 0) | (mask == 1)).all()),
        "faithfulness_mse": float(bpd.wrappers["core"].weight_delta().square().mean()),
    }


def choose_threshold(bpd, gate, calibration, *, mode: str):
    if mode != "margin":
        return .5, []
    best = None
    rows = []
    for threshold in CONFIG["optimization"]["calibration_thresholds"]:
        result = evaluate(bpd, gate, calibration, mode, threshold=threshold)
        score = (result["binary_mse"] +
                 CONFIG["optimization"]["gate_count_coeff"] * result["gate_fraction"])
        row = {"threshold": threshold, "calibration_score": score,
               "calibration_binary_mse": result["binary_mse"],
               "calibration_gate_fraction": result["gate_fraction"]}
        rows.append(row)
        # Preference for higher threshold on exact ties: scan ascending,
        # replace on <= with numerical tolerance.
        if best is None or score < best[0] - 1e-12 or (
            abs(score - best[0]) <= 1e-12 and threshold > best[1]
        ):
            best = (score, threshold)
    assert best is not None
    return best[1], rows


def success(row, factor_update_norm: float, factor_initial_grad: float) -> bool:
    c = CONFIG["success"]
    return bool(
        row["binary_mse"] <= c["heldout_binary_mse_le"]
        and row["axis_match"] >= c["read_axis_cos_ge"]
        and row["phase_assignment_fraction"] >= c["board_phase_route_fraction_ge"]
        and row["execution_exact_binary"]
        and factor_update_norm > 0
        and factor_initial_grad > 0
    )


def run_arm(
    seed: int, arm: str, *, steps: int | None = None,
    batch_size: int | None = None, calibration_size: int | None = None,
    heldout_size: int | None = None,
) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    torch.set_num_threads(1)
    config = CONFIG
    opt = config["optimization"]
    gen_cfg = config["generation"]
    steps = opt["stage1_steps"] if steps is None else steps
    batch_size = gen_cfg["train_batch"] if batch_size is None else batch_size
    calibration_size = gen_cfg["calibration_boards"] if calibration_size is None else calibration_size
    heldout_size = gen_cfg["heldout_boards"] if heldout_size is None else heldout_size
    frozen, mixed = dictionary_frozen(arm), source_mixed(arm)
    mode = routing_mode(arm)
    bpd, gate, initial_sha = make_model(seed, frozen=frozen)
    params = bpd.component_parameters()
    initial_params = [p.detach().clone() for p in params]
    train_generator = torch.Generator().manual_seed(gen_cfg["training_seed_offset"] + seed)
    optimizer = torch.optim.Adam(
        list(gate.parameters()) + ([] if frozen else params),
        lr=opt["Adam_lr"], weight_decay=opt["Adam_weight_decay"],
    )
    factor_initial_grad = 0.
    gate_initial_grad = 0.
    start = time.monotonic()
    for step in range(steps):
        x = draw_batch(train_generator, batch_size, mixed=mixed)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_parts(bpd, gate, x, mode)
        loss["total"].backward()
        if step == 0:
            factor_initial_grad = math.sqrt(sum(
                float(p.grad.square().sum()) for p in params if p.grad is not None
            ))
            gate_initial_grad = math.sqrt(sum(
                float(p.grad.square().sum()) for p in gate.parameters() if p.grad is not None
            ))
        optimizer.step()
    factor_update_norm = math.sqrt(sum(
        float((p.detach() - before).square().sum())
        for p, before in zip(params, initial_params)
    ))
    calibration = draw_batch(
        torch.Generator().manual_seed(gen_cfg["calibration_seed_offset"] + seed),
        calibration_size, mixed=mixed,
    )
    threshold, candidates = choose_threshold(bpd, gate, calibration, mode=mode)
    # No model or threshold changes after this point. Heldout is opened once.
    heldout = draw_batch(
        torch.Generator().manual_seed(gen_cfg["heldout_seed_offset"] + seed),
        heldout_size, mixed=mixed,
    )
    heldout_metrics = evaluate(bpd, gate, heldout, mode, threshold=threshold)
    return {
        "seed": seed, "arm": arm,
        "mode": mode, "mixed": mixed, "frozen": frozen,
        "initial_parameter_sha256": initial_sha,
        "updates": steps, "batch_size": batch_size,
        "factor_first_gradient_norm": factor_initial_grad,
        "gate_first_gradient_norm": gate_initial_grad,
        "factor_update_norm": factor_update_norm,
        "calibration_threshold": threshold,
        "calibration_candidates": candidates,
        "heldout": heldout_metrics,
        "success": False if frozen or mixed else success(
            heldout_metrics, factor_update_norm, factor_initial_grad
        ),
        "elapsed_seconds": round(time.monotonic() - start, 4),
    }


def cohort_summary(arms: list[dict], *, registered: bool) -> dict:
    by_key = {(a["seed"], a["arm"]): a for a in arms}
    seeds = sorted({a["seed"] for a in arms})
    names = ("top1_disjoint", "margin_disjoint")
    a = [by_key[(seed, names[0])]["success"] for seed in seeds]
    b = [by_key[(seed, names[1])]["success"] for seed in seeds]
    aonly = sum(x and not y for x, y in zip(a, b))
    bonly = sum(y and not x for x, y in zip(a, b))
    n = len(seeds)
    c = CONFIG["success"]
    p = exact_mcnemar_p(aonly, bonly)
    valid = registered and n == len(SEEDS) and tuple(seeds) == SEEDS
    counts = {
        key: sum(bool(by_key[(seed, key)]["success"]) for seed in seeds)
        for key in ("top1_disjoint", "margin_disjoint", "hard_disjoint")
    }
    return {
        "study_id": "BPD-PD-03", "registered": valid, "seeds": n, "arms": len(arms),
        "success_counts": counts,
        "top1_wilson95": wilson_ci(counts["top1_disjoint"], n),
        "margin_wilson95": wilson_ci(counts["margin_disjoint"], n),
        "top1_only": aonly, "margin_only": bonly,
        "net_top1_seed_wins": aonly - bonly,
        "mcnemar_two_sided_p": p,
        "primary_passed": bool(
            valid and counts["top1_disjoint"] >= c["primary_top1_min_successes"]
            and aonly - bonly >= c["primary_net_paired_wins_min"]
            and p <= c["mcnemar_two_sided_p_le"]
        ),
        "per_seed": [
            {"seed": seed,
             "top1_success": a[i], "margin_success": b[i],
             "top1_mse": by_key[(seed,"top1_disjoint")]["heldout"]["binary_mse"],
             "margin_mse": by_key[(seed,"margin_disjoint")]["heldout"]["binary_mse"],
             "top1_axis": by_key[(seed,"top1_disjoint")]["heldout"]["axis_match"],
             "margin_axis": by_key[(seed,"margin_disjoint")]["heldout"]["axis_match"],
             "top1_route": by_key[(seed,"top1_disjoint")]["heldout"]["phase_assignment_fraction"],
             "margin_route": by_key[(seed,"margin_disjoint")]["heldout"]["phase_assignment_fraction"],
             "calibrated_threshold": by_key[(seed,"margin_disjoint")]["calibration_threshold"],
            } for i, seed in enumerate(seeds)
        ],
    }


def run_registered() -> dict:
    started=time.monotonic()
    arm_rows=[]
    for seed in SEEDS:
        for name in ARMS:
            arm_rows.append(run_arm(seed,name))
    for seed in SEEDS:
        cohort = [x for x in arm_rows if x["seed"] == seed and not x["frozen"]]
        assert len({x["initial_parameter_sha256"] for x in cohort}) == 1, seed
        assert all(x["factor_update_norm"] > 0 for x in cohort)
    return {
        "study_id":"BPD-PD-03",
        "status":"first_prospective_full_run",
        "sha256_sources":{
            name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("CONFIG.json","PROTOCOL.md","experiment.py")
        },
        "torch_version":torch.__version__,
        "python_version":platform.python_version(),
        "elapsed_seconds":round(time.monotonic()-started,4),
        "summary":cohort_summary(arm_rows,registered=True),
        "arms":arm_rows
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    receipt=run_registered()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(receipt["summary"],indent=2,sort_keys=True))


if __name__=="__main__":
    main()
