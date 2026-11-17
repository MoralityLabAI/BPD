"""Excluded-seed engineering checks for preregistered BPD-PD-05.

No PD-05 science seed is trained by this suite. GitHub CI must pass before
a one-time EXECUTE_LOCK.json is committed.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.theory.pd05.experiment import (
    CONFIG, ARMS, SEEDS, FAMILIES, AdaptiveCountSTE, ARM_BY_KEY,
    build, count_truth, draw_train_batch, draw_stratified, masked_forward,
    objective, run_arm, summary, evaluate_all,
)
from mechinterp.bpd.theory.pd04.experiment import draw_family

SMOKE = CONFIG["runtime"]["smoke_seed"]


def test_source_lock_and_seed_independence():
    assert CONFIG["study_id"] == "BPD-PD-05"
    assert len(SEEDS) == len(set(SEEDS)) == 24
    assert all(s > 569 for s in SEEDS)
    assert SMOKE not in SEEDS and 71 not in SEEDS and 89 not in SEEDS
    assert len(ARMS) == 8
    assert FAMILIES == ("disjoint", "overlap", "nested")
    assert CONFIG["training"]["steps"] == 240
    assert CONFIG["success"]["primary_min_success_24"] == 18
    assert CONFIG["predecessor"]["pd04_adaptive_successes"] == 0
    source = CONFIG["source"]
    for seed in SEEDS:
        assert len({
            source["train_generator_offset"] + seed,
            source["calibration_generator_offset"] + seed,
            source["heldout_generator_offset"] + seed,
        }) == 3


def test_balanced_source_cardinality_labels_match_exact_support():
    src = CONFIG["source"]
    for name in FAMILIES:
        x = draw_family(
            torch.Generator().manual_seed(src["heldout_generator_offset"] + SMOKE),
            16, name,
        )
        supports = src["supports"][name]
        assert x.shape == (16, 4, 1, 2)
        assert torch.equal(count_truth(x), (
            ((x != 0).sum(dim=-1, keepdim=True) == 2).float()
        ))
        for t in range(4):
            for c in (0, 1):
                if c in supports[t]:
                    assert (x[:, t, :, c].abs() >= .75).all()
                    assert (x[:, t, :, c].abs() <= 1.25).all()
                else:
                    assert (x[:, t, :, c] == 0).all()
    a = draw_train_batch(
        torch.Generator().manual_seed(src["train_generator_offset"] + SMOKE), 4,
    )
    b = draw_train_batch(
        torch.Generator().manual_seed(src["train_generator_offset"] + SMOKE), 4,
    )
    assert torch.equal(a, b)
    assert a.shape == (12, 4, 1, 2)


def test_identical_binary_forward_and_distinct_count_backward():
    a = torch.tensor([[[[1.0, -0.4]]]], requires_grad=True)
    c = torch.tensor([[[[-7.0]]]], requires_grad=True)
    reference_mask = None
    grads = {}
    for scheme in (
        "sigmoid_logit_jacobian", "identity_logit_jacobian",
        "temperature2_logit_jacobian",
    ):
        a.grad = None
        c.grad = None
        mask = AdaptiveCountSTE.apply(a, c, scheme)
        assert torch.equal(mask.detach(), torch.tensor([[[[1., 0.]]]]))
        if reference_mask is not None:
            assert torch.equal(mask.detach(), reference_mask)
        reference_mask = mask.detach().clone()
        (mask[..., 0] + 2.0 * mask[..., 1]).sum().backward()
        grads[scheme] = abs(float(c.grad))
    assert grads["identity_logit_jacobian"] > grads["temperature2_logit_jacobian"]
    assert grads["temperature2_logit_jacobian"] > grads["sigmoid_logit_jacobian"]
    assert grads["sigmoid_logit_jacobian"] > 0


def test_gradient_and_real_physical_wrapper_for_supervised_and_unsupervised():
    for arm in ("unsupervised_sigmoid", "unsupervised_identity",
                "unsupervised_tempered2", "supervised_sigmoid",
                "supervised_identity", "oracle_count",
                "calibrated_independent"):
        bpd, gate, _ = build(SMOKE, frozen=False)
        assert isinstance(bpd, BPDModel)
        x = draw_train_batch(torch.Generator().manual_seed(101009), 4)
        output, mask, _, _ = masked_forward(bpd, gate, x, arm)
        assert output.shape == x.shape
        assert mask.shape == x.shape
        assert bpd.wrappers["core"].mode == "target"
        assert bpd.wrappers["core"].calls == 4
        assert bool(((mask.detach() == 0) | (mask.detach() == 1)).all())
        d = objective(bpd, gate, x, arm)
        assert d["count_supervised"] == (ARM_BY_KEY[arm]["count_supervision"] > 0)
        if d["count_supervised"]:
            assert d["count_bce"].item() > 0
        else:
            assert d["count_bce"].item() == 0
        d["total"].backward()
        assert all(p.grad is not None for p in bpd.component_parameters())
        assert sum(float(p.grad.square().sum()) for p in bpd.component_parameters()) > 0
        if d["count_supervised"]:
            assert sum(float(p.grad.square().sum()) for p in gate.count.parameters()) > 0


def test_training_only_bce_and_no_label_gradient_for_unsupervised():
    x = draw_train_batch(torch.Generator().manual_seed(101009), 4)
    bpd1, gate1, fp1 = build(SMOKE, frozen=False)
    bpd2, gate2, fp2 = build(SMOKE, frozen=False)
    assert fp1 == fp2
    raw1 = objective(bpd1, gate1, x, "unsupervised_sigmoid")
    raw2 = objective(bpd2, gate2, x, "supervised_sigmoid")
    assert torch.allclose(
        raw2["total"] - raw1["total"],
        F.binary_cross_entropy_with_logits(
            gate2.count(x.abs()), count_truth(x)
        ),
    )


def test_real_wrapper_smokes_with_frozen_factor_and_calibration():
    rows = [
        run_arm(SMOKE, arm, steps=5, boards_per_family=16, train_per_family=4)
        for arm in ARMS
    ]
    assert len(rows) == 8
    assert len({r["initialization_sha256"] for r in rows
                if not r["frozen_factors"]}) == 1
    for row in rows:
        assert row["seed"] == SMOKE
        assert row["factor_update_norm"] > 0 or row["frozen_factors"]
        assert row["first_factor_grad_norm"] > 0 or row["frozen_factors"]
        assert set(row["heldout"]["per_family"]) == set(FAMILIES)
        assert all(v["actual_forward_binary"]
                   for v in row["heldout"]["per_family"].values())
        if row["frozen_factors"]:
            assert row["factor_update_norm"] == 0
        if ARM_BY_KEY[row["arm"]]["count_supervision"]:
            assert row["first_count_grad_norm"] > 0
        if row["arm"] == "calibrated_independent":
            assert len(row["calibration_candidates"]) == 7
            chosen = next(c for c in row["calibration_candidates"]
                          if c["threshold"] == row["calibration_threshold"])
            assert chosen["score"] <= min(
                c["score"] for c in row["calibration_candidates"]
            ) + 1e-10
        else:
            assert row["calibration_candidates"] == []
            assert row["calibration_threshold"] == .5
    s = summary(rows, registered=False)
    assert s["seeds"] == 1 and s["arm_count"] == 8
    assert not s["registered_primary_passed"]


def test_exact_paired_inference_is_seed_based():
    from mechinterp.bpd.theory.pd02.experiment import exact_mcnemar_p
    assert exact_mcnemar_p(0,0) == 1.0
    assert exact_mcnemar_p(6,0) == .03125
    assert exact_mcnemar_p(3,7) == .34375
