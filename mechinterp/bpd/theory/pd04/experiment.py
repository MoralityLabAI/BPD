"""BPD-PD-04 prospective adaptive cardinality experiment.

Uses actual BPDModel/TiedComponentLinear and fully trainable shared (V,U).
The sole label-exposed arm is named oracle_cardinality. All other gates
observe only |x|, never source-family labels or planted support masks.

Frozen source contract: CONFIG.json and PROTOCOL.md, Nov 17, 2026.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.bpd.components import PassMaskSchedule
from mechinterp.bpd.theory.pd02.experiment import (
    make_model, exact_mcnemar_p, wilson_ci,
)
from mechinterp.bpd.theory.pd03.experiment import (
    TopOneSTE, ThresholdSTE, read_assignment,
)
from mechinterp.bpd.theory.phase_disjoint_control import read_axis_match

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
SEEDS = tuple(CONFIG["seed_list"])
ARMS = tuple(a["key"] for a in CONFIG["arms"])
FAMILIES = tuple(CONFIG["heldout_families"])


class AdaptiveGate(nn.Module):
    """Shared affine read logits and independent learned cardinality logit."""

    def __init__(self, previous_gate: nn.Module):
        super().__init__()
        self.fc = previous_gate.fc
        self.count = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude = x.abs()
        return self.fc(magnitude), self.count(magnitude)


class AdaptiveTopKSTE(torch.autograd.Function):
    """Exact top-1 or top-2 forward with stated differentiable surrogate."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, count_logits: torch.Tensor):
        soft = logits.softmax(dim=-1)
        count_prob = count_logits.sigmoid()
        hard_count = (count_prob >= .5).to(dtype=logits.dtype)
        top = F.one_hot(logits.argmax(dim=-1), num_classes=2).to(logits.dtype)
        ctx.save_for_backward(soft, count_prob, hard_count)
        return top + hard_count * (1.0 - top)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        soft, count_prob, hard_count = ctx.saved_tensors
        mean = (grad_output * soft).sum(dim=-1, keepdim=True)
        grad_logits = (1.0 - hard_count) * soft * (grad_output - mean)
        grad_count = ((grad_output * (1.0 - soft)).sum(dim=-1, keepdim=True)
                      * count_prob * (1.0 - count_prob))
        return grad_logits, grad_count


def mode_for_arm(key: str) -> str:
    return {
        "adaptive_disjoint_overlap": "adaptive",
        "calibrated_independent": "calibrated",
        "independent_hard": "hard",
        "top1_fixed": "top1",
        "all2_fixed": "all2",
        "oracle_cardinality": "oracle",
        "adaptive_frozen": "adaptive",
    }[key]


def frozen_for_arm(key: str) -> bool:
    return key == "adaptive_frozen"


def build(seed: int, *, frozen: bool):
    bpd, previous_gate, _ = make_model(seed, frozen=frozen)
    gate = AdaptiveGate(previous_gate)
    payload = {
        "factor_V": bpd.wrappers["core"].V.detach().tolist(),
        "factor_U": bpd.wrappers["core"].U.detach().tolist(),
        "gate": {k: v.detach().tolist() for k, v in gate.state_dict().items()},
    }
    fingerprint = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return bpd, gate, fingerprint


def draw_family(
    generator: torch.Generator, boards: int, family: str
) -> torch.Tensor:
    src = CONFIG["source"]
    if family not in FAMILIES:
        raise ValueError(family)
    support = src[f"{family}_supports"]
    mag = src["amplitude_min"] + (
        src["amplitude_max"] - src["amplitude_min"]
    ) * torch.rand(boards, 4, 1, 2, generator=generator)
    sign = torch.where(
        torch.randint(0, 2, (boards, 4, 1, 2), generator=generator).bool(),
        1.0, -1.0,
    )
    values = mag * sign
    for step, active in enumerate(support):
        for component in range(2):
            if component not in active:
                values[:, step, :, component] = 0.0
    return values


def draw_train_batch(generator: torch.Generator, boards_per_family: int = 8):
    stacked = torch.cat([
        draw_family(generator, boards_per_family, name) for name in FAMILIES
    ], dim=0)
    order = torch.randperm(len(stacked), generator=generator)
    return stacked[order]


def draw_stratified(generator: torch.Generator, boards_per_family: int):
    return {
        name: draw_family(generator, boards_per_family, name)
        for name in FAMILIES
    }


def gate_mask(
    logits: torch.Tensor, count_logits: torch.Tensor, x: torch.Tensor,
    mode: str, threshold: float = .5,
):
    if mode == "adaptive":
        return AdaptiveTopKSTE.apply(logits, count_logits)
    if mode in ("calibrated", "hard"):
        return ThresholdSTE.apply(logits.sigmoid(), threshold)
    if mode == "top1":
        return TopOneSTE.apply(logits)
    if mode == "all2":
        return torch.ones_like(logits)
    if mode == "oracle":
        # Known cardinality is permitted ONLY in this named upper-bound arm.
        top = TopOneSTE.apply(logits)
        both = ((x != 0).sum(dim=-1, keepdim=True) == 2).to(logits.dtype)
        return top + both * (1.0 - top)
    raise ValueError(mode)


def masked_forward(bpd, gate, x, mode: str, *, threshold: float = .5):
    logits, count_logits = gate(x)
    mask = gate_mask(logits, count_logits, x, mode, threshold)
    if not bool(((mask.detach() == 0) | (mask.detach() == 1)).all()):
        raise AssertionError("PD04 gate forward must be exactly binary")
    schedule = PassMaskSchedule(
        comp_mask=mask,
        delta_mask=torch.zeros_like(mask[..., 0]),
    )
    with bpd.masked_components({"core": schedule}):
        output = torch.stack(
            [bpd.target_model(x[:, step]) for step in range(4)], dim=1
        )
    return output, mask, logits, count_logits


def objective(bpd, gate, x, mode: str):
    predicted, mask, logits, _ = masked_forward(bpd, gate, x, mode)
    mse = (predicted - x).square().mean()
    faith = bpd.wrappers["core"].weight_delta().square().mean()
    density = mask.mean()
    margin = F.relu(
        CONFIG["optimization"]["calibrated_margin_band"] -
        (logits.sigmoid() - .5).abs()
    ).mean()
    coeff = (
        CONFIG["optimization"]["calibrated_margin_coeff"]
        if mode == "calibrated" else 0.0
    )
    total = (mse +
             CONFIG["optimization"]["matrix_faithfulness_coeff"] * faith +
             CONFIG["optimization"]["gate_activity_coeff"] * density +
             coeff * margin)
    return total


def expected_component_mask(bpd, x: torch.Tensor) -> torch.Tensor:
    assignment = read_assignment(bpd)
    support = (x != 0).to(torch.float32)
    if assignment == 0:
        return support
    return support.flip(dims=(-1,))


@torch.no_grad()
def evaluate_family(bpd, gate, x, mode: str, threshold: float):
    predicted, mask, logits, count_logits = masked_forward(
        bpd, gate, x, mode, threshold=threshold
    )
    expected = expected_component_mask(bpd, x)
    exact_per_board = (mask.detach() == expected).all(dim=(-1, -2, -3))
    return {
        "mse": float((predicted - x).square().mean()),
        "support_accuracy": float(exact_per_board.float().mean()),
        "exact_support_boards": int(exact_per_board.sum().item()),
        "board_count": int(x.shape[0]),
        "false_off_count": int(((mask == 0) & (expected == 1)).sum().item()),
        "false_on_count": int(((mask == 1) & (expected == 0)).sum().item()),
        "gate_density": float(mask.mean()),
        "mean_cardinality": float(mask.sum(dim=-1).mean()),
        "mean_count_probability": float(count_logits.sigmoid().mean()),
        "expected_cardinality": float((x != 0).sum(dim=-1).float().mean()),
        "mask_is_binary": bool(((mask == 0) | (mask == 1)).all()),
        "onehot_fraction": float((mask.sum(dim=-1) == 1).float().mean()),
        "twohot_fraction": float((mask.sum(dim=-1) == 2).float().mean()),
    }


@torch.no_grad()
def evaluate_stratified(bpd, gate, strata, mode: str, threshold: float):
    per = {
        name: evaluate_family(bpd, gate, strata[name], mode, threshold)
        for name in FAMILIES
    }
    return {
        "mse": sum(x["mse"] for x in per.values()) / len(FAMILIES),
        "gate_density": sum(x["gate_density"] for x in per.values()) / len(FAMILIES),
        "axis_match": read_axis_match(bpd),
        "threshold": threshold,
        "per_family": per,
    }


def choose_threshold(bpd, gate, calibration, mode: str):
    if mode != "calibrated":
        return .5, []
    rows = []
    best = None
    weight = CONFIG["optimization"]["gate_activity_coeff"]
    for candidate in CONFIG["optimization"]["independent_threshold_candidates"]:
        m = evaluate_stratified(bpd, gate, calibration, mode, candidate)
        score = m["mse"] + weight * m["gate_density"]
        rows.append({"threshold": candidate, "score": score,
                     "mse": m["mse"], "gate_density": m["gate_density"]})
        if (best is None or score < best[0] - 1e-12 or
                abs(score - best[0]) <= 1e-12 and candidate > best[1]):
            best = (score, candidate)
    assert best is not None
    return best[1], rows


def arm_success(metrics, *, factor_gradient: float,
                factor_update: float, frozen: bool) -> bool:
    c = CONFIG["success"]
    return bool(
        metrics["axis_match"] >= c["read_axis_cos_min"]
        and metrics["mse"] <= c["overall_heldout_mse_max"]
        and all(
            m["mse"] <= c["per_family_binary_mse_max"]
            and m["support_accuracy"] >= c["per_family_exact_four_pass_support_fraction_min"]
            and m["mask_is_binary"]
            for m in metrics["per_family"].values()
        )
        and (frozen or (factor_gradient > 0 and factor_update > 0))
    )


def run_arm(
    seed: int, arm: str, *,
    steps: int | None = None,
    boards_per_family: int | None = None,
    train_per_family: int | None = None,
):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.set_num_threads(1)
    cfg = CONFIG
    src = cfg["source"]
    opt = cfg["optimization"]
    steps = opt["steps"] if steps is None else steps
    boards_per_family = (
        src["heldout_per_family"] if boards_per_family is None else boards_per_family
    )
    train_per_family = (
        src["train_batch_per_family"] if train_per_family is None else train_per_family
    )
    frozen = frozen_for_arm(arm)
    mode = mode_for_arm(arm)
    bpd, gate, original_sha = build(seed, frozen=frozen)
    factors = bpd.component_parameters()
    initial = [p.detach().clone() for p in factors]
    train_gen = torch.Generator().manual_seed(src["train_seed_offset"] + seed)
    optimizer = torch.optim.Adam(
        list(gate.parameters()) + ([] if frozen else factors),
        lr=opt["lr"], weight_decay=opt["weight_decay"],
    )
    factor_grad = 0.
    gate_grad = 0.
    count_grad = 0.
    start = time.monotonic()
    for update in range(steps):
        x = draw_train_batch(train_gen, train_per_family)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(bpd, gate, x, mode)
        loss.backward()
        if update == 0:
            factor_grad = math.sqrt(sum(
                float(p.grad.square().sum()) for p in factors if p.grad is not None
            ))
            gate_grad = math.sqrt(sum(
                float(p.grad.square().sum()) for p in gate.fc.parameters()
                if p.grad is not None
            ))
            count_grad = math.sqrt(sum(
                float(p.grad.square().sum()) for p in gate.count.parameters()
                if p.grad is not None
            ))
        optimizer.step()
    factor_update = math.sqrt(sum(
        float((p.detach() - init).square().sum())
        for p, init in zip(factors, initial)
    ))
    calib = draw_stratified(
        torch.Generator().manual_seed(src["calibration_seed_offset"] + seed),
        src["calibration_per_family"]
    )
    threshold, calibration_candidates = choose_threshold(
        bpd, gate, calib, mode
    )
    heldout = draw_stratified(
        torch.Generator().manual_seed(src["heldout_seed_offset"] + seed),
        boards_per_family
    )
    metrics = evaluate_stratified(bpd, gate, heldout, mode, threshold)
    return {
        "seed":seed,"arm":arm,"mode":mode,"frozen":frozen,
        "initial_parameter_sha256":original_sha,
        "updates":steps,
        "factor_first_gradient_norm":factor_grad,
        "gate_first_gradient_norm":gate_grad,
        "count_first_gradient_norm":count_grad,
        "factor_update_norm":factor_update,
        "threshold":threshold,
        "calibration_candidates":calibration_candidates,
        "heldout":metrics,
        "success":arm_success(metrics,factor_gradient=factor_grad,
                              factor_update=factor_update,frozen=frozen),
        "seconds":round(time.monotonic()-start,4),
    }


def summarize(rows: list[dict], registered: bool):
    by={(r["seed"],r["arm"]):r for r in rows}
    seeds=sorted({r["seed"] for r in rows})
    expected_keys={(seed,arm) for seed in seeds for arm in ARMS}
    if set(by) != expected_keys or len(rows) != len(by):
        raise ValueError("missing or duplicate PD04 arm rows")
    primary=[by[(s,"adaptive_disjoint_overlap")]["success"] for s in seeds]
    comparator=[by[(s,"calibrated_independent")]["success"] for s in seeds]
    aonly=sum(a and not b for a,b in zip(primary,comparator))
    bonly=sum(b and not a for a,b in zip(primary,comparator))
    counts={arm:sum(by[(s,arm)]["success"] for s in seeds) for arm in ARMS}
    stat=CONFIG["success"]
    genuine=bool(registered and tuple(seeds)==SEEDS)
    return {
        "study_id":"BPD-PD-04",
        "is_registered_run":genuine,
        "seed_count":len(seeds),
        "arm_count":len(rows),
        "success_count_by_arm":counts,
        "adaptive_Wilson95":wilson_ci(counts["adaptive_disjoint_overlap"],len(seeds)),
        "calibrated_Wilson95":wilson_ci(counts["calibrated_independent"],len(seeds)),
        "adaptive_only":aonly,
        "calibrated_only":bonly,
        "net_adaptive_seed_wins":aonly-bonly,
        "McNemar_exact_two_sided_p":exact_mcnemar_p(aonly,bonly),
        "registered_primary_passed":bool(
            genuine
            and counts["adaptive_disjoint_overlap"]>=stat["primary_adaptive_success_count_min"]
            and aonly-bonly>=stat["primary_net_paired_seed_wins_min"]
            and exact_mcnemar_p(aonly,bonly)<=stat["primary_mcnemar_two_sided_p_max"]
        ),
        "per_seed":[{
            "seed":s,
            "adaptive_success":by[(s,"adaptive_disjoint_overlap")]["success"],
            "calibrated_success":by[(s,"calibrated_independent")]["success"],
            "adaptive_mse":by[(s,"adaptive_disjoint_overlap")]["heldout"]["mse"],
            "calibrated_mse":by[(s,"calibrated_independent")]["heldout"]["mse"],
            "adaptive_axis":by[(s,"adaptive_disjoint_overlap")]["heldout"]["axis_match"],
            "calibrated_axis":by[(s,"calibrated_independent")]["heldout"]["axis_match"],
            "adaptive_support": {
                f:by[(s,"adaptive_disjoint_overlap")]["heldout"]["per_family"][f]["support_accuracy"]
                for f in FAMILIES
            },
            "calibrated_support": {
                f:by[(s,"calibrated_independent")]["heldout"]["per_family"][f]["support_accuracy"]
                for f in FAMILIES
            },
        } for s in seeds],
    }


def run_registered():
    start=time.monotonic()
    rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(run_arm(seed,arm))
    for seed in SEEDS:
        shared={row["initial_parameter_sha256"] for row in rows
                if row["seed"]==seed and not row["frozen"]}
        if len(shared)!=1:
            raise AssertionError(f"initial state mismatch for seed {seed}")
    return {
        "study_id":CONFIG["study_id"],
        "status":"first_frozen_prospective_cohort",
        "source_hashes": {
            n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
            for n in ("CONFIG.json","PROTOCOL.md","experiment.py")
        },
        "torch_version":torch.__version__,
        "python_version":platform.python_version(),
        "elapsed_seconds":round(time.monotonic()-start,4),
        "summary":summarize(rows,registered=True),
        "rows":rows,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    r=run_registered()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(r["summary"],indent=2,sort_keys=True))


if __name__=="__main__":
    main()
