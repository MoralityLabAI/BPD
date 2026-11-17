"""PD-05: supervised cardinality versus STE gradients on shared BPD factors.

Source-locked against CONFIG.json and PROTOCOL.md. This experiment calls
the actual BPDModel/TiedComponentLinear four times per input. All non-frozen
arms optimize both V and U jointly with the gate parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from mechinterp.bpd.components import PassMaskSchedule
from mechinterp.bpd.theory.pd02.experiment import exact_mcnemar_p, wilson_ci
from mechinterp.bpd.theory.pd03.experiment import ThresholdSTE, read_assignment
from mechinterp.bpd.theory.pd04.experiment import (
    build, draw_train_batch, draw_stratified, expected_component_mask,
)

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
SEEDS = tuple(CONFIG["seeds"])
ARMS = tuple(a["key"] for a in CONFIG["arms"])
FAMILIES = tuple(CONFIG["source"]["families"])
ARM_BY_KEY = {a["key"]: a for a in CONFIG["arms"]}


class AdaptiveCountSTE(torch.autograd.Function):
    """PD-04 exact binary forward with one of three declared count derivatives."""

    @staticmethod
    def forward(ctx, assignment_logits, count_logits, derivative):
        soft = assignment_logits.softmax(dim=-1)
        p = count_logits.sigmoid()
        h = (count_logits >= 0).to(assignment_logits.dtype)
        top = F.one_hot(
            assignment_logits.argmax(dim=-1), num_classes=2
        ).to(assignment_logits.dtype)
        ctx.save_for_backward(soft, p, h, count_logits)
        ctx.derivative = derivative
        return top + h * (1.0 - top)

    @staticmethod
    def backward(ctx, grad_output):
        soft, p, h, count_logit = ctx.saved_tensors
        grad_assign = (1.0 - h) * soft * (
            grad_output - (grad_output * soft).sum(dim=-1, keepdim=True)
        )
        base = (grad_output * (1.0 - soft)).sum(dim=-1, keepdim=True)
        if ctx.derivative == "sigmoid_logit_jacobian":
            multiplier = p * (1.0 - p)
        elif ctx.derivative == "identity_logit_jacobian":
            multiplier = torch.ones_like(p)
        elif ctx.derivative == "temperature2_logit_jacobian":
            q = (count_logit / 2.0).sigmoid()
            multiplier = q * (1.0 - q) / 2.0
        else:
            raise ValueError(ctx.derivative)
        return grad_assign, base * multiplier, None


def mask_for(bpd, logits, count_logits, x, arm, *, threshold=.5):
    spec = ARM_BY_KEY[arm]
    variant = spec["ste"]
    if variant in (
        "sigmoid_logit_jacobian", "identity_logit_jacobian",
        "temperature2_logit_jacobian",
    ):
        return AdaptiveCountSTE.apply(logits, count_logits, variant)
    if variant == "oracle_count":
        from mechinterp.bpd.theory.pd03.experiment import TopOneSTE
        top = TopOneSTE.apply(logits)
        both = ((x != 0).sum(dim=-1, keepdim=True) == 2).to(logits.dtype)
        return top + both * (1.0 - top)
    if variant == "independent_sigmoid":
        return ThresholdSTE.apply(logits.sigmoid(), threshold)
    raise ValueError(variant)


def masked_forward(bpd, gate, x, arm, *, threshold=.5):
    logits, count_logits = gate(x)
    mask = mask_for(bpd, logits, count_logits, x, arm, threshold=threshold)
    if not bool(((mask.detach() == 0) | (mask.detach() == 1)).all()):
        raise AssertionError("PD05 execution must be exactly binary")
    schedule = PassMaskSchedule(
        comp_mask=mask, delta_mask=torch.zeros_like(mask[..., 0]),
    )
    with bpd.masked_components({"core": schedule}):
        predicted = torch.stack(
            [bpd.target_model(x[:, step]) for step in range(4)], dim=1
        )
    return predicted, mask, logits, count_logits


def count_truth(x):
    return ((x != 0).sum(dim=-1, keepdim=True) == 2).float()


def objective(bpd, gate, x, arm):
    y, mask, logits, count_logits = masked_forward(bpd, gate, x, arm)
    mse = (y - x).square().mean()
    faith = bpd.wrappers["core"].weight_delta().square().mean()
    gate_frac = mask.mean()
    o = CONFIG["training"]
    total = (
        o["loss_reconstruction_mse_weight"] * mse
        + o["loss_matrix_faithfulness_weight"] * faith
        + o["loss_mean_binary_gate_usage_weight"] * gate_frac
    )
    supervised = ARM_BY_KEY[arm]["count_supervision"] > 0
    count_bce = F.binary_cross_entropy_with_logits(
        count_logits, count_truth(x)
    ) if supervised else torch.zeros((), dtype=total.dtype, device=total.device)
    if supervised:
        total = total + o["auxiliary_count_BCE_weight"] * count_bce
    if arm == "calibrated_independent":
        band = F.relu(.10 - (logits.sigmoid() - .5).abs()).mean()
        total = total + .02 * band
    return {
        "total": total,
        "mse": mse,
        "faith": faith,
        "gate_fraction": gate_frac,
        "count_supervised": supervised,
        "count_bce": count_bce,
    }


@torch.no_grad()
def evaluate_family(bpd, gate, x, arm, threshold):
    pred, mask, logits, count_logits = masked_forward(
        bpd, gate, x, arm, threshold=threshold
    )
    expected = expected_component_mask(bpd, x)
    entire = (mask.detach() == expected).all(dim=(-1, -2, -3))
    true_count = count_truth(x)
    variant = ARM_BY_KEY[arm]["ste"]
    if variant in ("oracle_count", "independent_sigmoid"):
        # A count-head logit is unused by these controls.
        count_accuracy = None
        signed_count_margin = None
        saturation = None
    else:
        count_accuracy = float(
            ((count_logits >= 0).float() == true_count).float().mean()
        )
        signed_count_margin = float(
            ((2 * true_count - 1) * count_logits).mean()
        )
        saturation = float((count_logits.abs() >= 4).float().mean())
    return {
        "binary_mse": float((pred-x).square().mean()),
        "exact_four_pass_support": float(entire.float().mean()),
        "exact_four_pass_boards": int(entire.sum()),
        "boards": int(x.shape[0]),
        "false_off_count": int(((mask == 0) & (expected == 1)).sum()),
        "false_on_count": int(((mask == 1) & (expected == 0)).sum()),
        "gate_fraction": float(mask.mean()),
        "average_cardinality": float(mask.sum(dim=-1).mean()),
        "count_head_accuracy": count_accuracy,
        "signed_count_logit_margin": signed_count_margin,
        "saturation_abs_logit_ge4": saturation,
        "actual_forward_binary": bool(((mask == 0) | (mask == 1)).all()),
    }


@torch.no_grad()
def evaluate_all(bpd, gate, strata, arm, *, threshold):
    fam = {
        name: evaluate_family(bpd, gate, strata[name], arm, threshold)
        for name in FAMILIES
    }
    from mechinterp.bpd.theory.phase_disjoint_control import read_axis_match
    return {
        "overall_mse": sum(v["binary_mse"] for v in fam.values()) / len(fam),
        "overall_gate_fraction": sum(v["gate_fraction"] for v in fam.values()) / len(fam),
        "read_axis_cosine": read_axis_match(bpd),
        "per_family": fam,
        "threshold": threshold,
    }


def choose_threshold(bpd, gate, strata, arm):
    if arm != "calibrated_independent":
        return .5, []
    candidates = CONFIG["training"]["separate_calibration_thresholds"]
    weight = CONFIG["training"]["loss_mean_binary_gate_usage_weight"]
    best = None
    rows = []
    for th in candidates:
        val = evaluate_all(bpd, gate, strata, arm, threshold=th)
        score = val["overall_mse"] + weight * val["overall_gate_fraction"]
        rows.append({
            "threshold": th,
            "score": score,
            "mse": val["overall_mse"],
            "gate_fraction": val["overall_gate_fraction"],
        })
        if (best is None or score < best[0] - 1e-12 or
                (abs(score - best[0]) <= 1e-12 and th > best[1])):
            best = (score, th)
    return best[1], rows


def succeeds(m, *, factor_grad, factor_update, frozen):
    s = CONFIG["success"]
    return bool(
        m["read_axis_cosine"] >= s["read_axis_cosine_min"]
        and m["overall_mse"] <= s["overall_binary_mse_max"]
        and all(
            v["binary_mse"] <= s["per_family_binary_mse_max"]
            and v["exact_four_pass_support"] >= s["per_family_full_trajectory_support_min"]
            and v["actual_forward_binary"]
            for v in m["per_family"].values()
        )
        and (frozen or factor_grad > 0 and factor_update > 0)
    )


def run_arm(seed, arm, *, steps=None, boards_per_family=None, train_per_family=None):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.set_num_threads(1)
    spec = ARM_BY_KEY[arm]
    frozen = not spec["train_factors"]
    bpd, gate, fingerprint = build(seed, frozen=frozen)
    factors = bpd.component_parameters()
    before = [p.detach().clone() for p in factors]
    source = CONFIG["source"]
    training = CONFIG["training"]
    steps = training["steps"] if steps is None else steps
    train_per_family = (
        source["balanced_train_per_family"]
        if train_per_family is None else train_per_family
    )
    boards_per_family = (
        source["heldout_count_per_family"]
        if boards_per_family is None else boards_per_family
    )
    rng_train = torch.Generator().manual_seed(
        source["train_generator_offset"] + seed
    )
    optimizer = torch.optim.Adam(
        list(gate.parameters()) + ([] if frozen else factors),
        lr=training["learning_rate"], weight_decay=training["weight_decay"],
    )
    first_factor_grad = 0.0
    first_count_grad = 0.0
    first_assign_grad = 0.0
    start = time.monotonic()
    for k in range(steps):
        x = draw_train_batch(rng_train, train_per_family)
        optimizer.zero_grad(set_to_none=True)
        d = objective(bpd, gate, x, arm)
        d["total"].backward()
        if k == 0:
            first_factor_grad = math.sqrt(sum(
                float(p.grad.square().sum()) for p in factors if p.grad is not None
            ))
            first_count_grad = math.sqrt(sum(
                float(p.grad.square().sum()) for p in gate.count.parameters()
                if p.grad is not None
            ))
            first_assign_grad = math.sqrt(sum(
                float(p.grad.square().sum()) for p in gate.fc.parameters()
                if p.grad is not None
            ))
        optimizer.step()
    factor_update = math.sqrt(sum(
        float((p.detach() - b).square().sum())
        for p, b in zip(factors, before)
    ))
    calib = draw_stratified(
        torch.Generator().manual_seed(
            source["calibration_generator_offset"] + seed
        ), source["calibration_count_per_family"]
    )
    threshold, sweep = choose_threshold(bpd, gate, calib, arm)
    heldout = draw_stratified(
        torch.Generator().manual_seed(
            source["heldout_generator_offset"] + seed
        ), boards_per_family
    )
    measurements = evaluate_all(
        bpd, gate, heldout, arm, threshold=threshold
    )
    if spec["count_supervision"] and first_count_grad <= 0:
        raise AssertionError("Count-supervised arm has zero count-head gradient")
    return {
        "seed": seed, "arm": arm, "frozen_factors": frozen,
        "first_factor_grad_norm": first_factor_grad,
        "first_count_grad_norm": first_count_grad,
        "first_assignment_grad_norm": first_assign_grad,
        "factor_update_norm": factor_update,
        "initialization_sha256": fingerprint,
        "training_steps": steps,
        "calibration_threshold": threshold,
        "calibration_candidates": sweep,
        "heldout": measurements,
        "success": succeeds(
            measurements, factor_grad=first_factor_grad,
            factor_update=factor_update, frozen=frozen,
        ),
        "elapsed_seconds": round(time.monotonic() - start, 4),
    }


def summary(rows, *, registered):
    by = {(r["seed"], r["arm"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    if len(rows) != len(by) or set(by) != {
        (s, a) for s in seeds for a in ARMS
    }:
        raise ValueError("duplicate or incomplete PD05 run")
    success_by_arm = {
        arm: sum(by[(s, arm)]["success"] for s in seeds)
        for arm in ARMS
    }
    arm_primary = CONFIG["success"]["primary_arm"]
    arm_control = CONFIG["success"]["primary_control"]
    pair_primary = [by[(s, arm_primary)]["success"] for s in seeds]
    pair_control = [by[(s, arm_control)]["success"] for s in seeds]
    primary_only = sum(a and not b for a, b in zip(pair_primary, pair_control))
    control_only = sum(b and not a for a, b in zip(pair_primary, pair_control))
    p = exact_mcnemar_p(primary_only, control_only)
    registered = bool(registered and tuple(seeds) == SEEDS)
    sc = CONFIG["success"]
    return {
        "study_id": CONFIG["study_id"],
        "full_registered_cohort": registered,
        "seeds": len(seeds),
        "arm_count": len(rows),
        "success_by_arm": success_by_arm,
        "seed_success_Wilson95_by_arm": {
            arm: wilson_ci(n, len(seeds))
            for arm, n in success_by_arm.items()
        },
        "primary_only": primary_only,
        "control_only": control_only,
        "net_primary_seed_wins": primary_only - control_only,
        "McNemar_exact_two_sided_p": p,
        "registered_primary_passed": bool(
            registered
            and success_by_arm[arm_primary] >= sc["primary_min_success_24"]
            and primary_only - control_only >= sc["primary_net_seed_wins_min"]
            and p <= sc["primary_exact_mcnemar_p_max"]
        ),
        "per_seed_primary": [
            {
                "seed": s, "primary_success": pair_primary[i],
                "control_success": pair_control[i],
                "primary_mse": by[(s, arm_primary)]["heldout"]["overall_mse"],
                "control_mse": by[(s, arm_control)]["heldout"]["overall_mse"],
                "primary_axis": by[(s, arm_primary)]["heldout"]["read_axis_cosine"],
                "control_axis": by[(s, arm_control)]["heldout"]["read_axis_cosine"],
                "primary_support": {
                    f: by[(s, arm_primary)]["heldout"]["per_family"][f]["exact_four_pass_support"]
                    for f in FAMILIES
                },
                "control_support": {
                    f: by[(s, arm_control)]["heldout"]["per_family"][f]["exact_four_pass_support"]
                    for f in FAMILIES
                },
            } for i, s in enumerate(seeds)
        ],
    }


def run_registered():
    start = time.monotonic()
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(run_arm(seed, arm))
            # Unix runner reports ru_maxrss in KiB.
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            if mem >= 3:
                raise MemoryError(f"BPD-PD-05 process RSS {mem:.3f} GiB exceeds 3 GiB cap")
        hashes = {
            row["initialization_sha256"] for row in rows
            if row["seed"] == seed and not row["frozen_factors"]
        }
        if len(hashes) != 1:
            raise AssertionError(f"Within-seed initial weights mismatch: {seed}")
    return {
        "study_id": CONFIG["study_id"],
        "status": "first_full_source_locked_cohort",
        "source_sha256": {
            f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
            for f in ("CONFIG.json", "PROTOCOL.md", "experiment.py")
        },
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "elapsed_seconds": round(time.monotonic() - start, 4),
        "peak_rss_gib": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss / 1024**2,
        "summary": summary(rows, registered=True),
        "arms": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_registered()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
