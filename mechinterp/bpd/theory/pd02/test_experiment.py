"""Engineering gates for BPD-PD-02; registered model seeds remain untouched."""
from __future__ import annotations

import math

import torch

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.theory.phase_disjoint_control import draw_batch
from mechinterp.bpd.theory.pd02.experiment import (
    ARMS, CONFIG, SEEDS, BinarySTE, assignment_fraction,
    exact_mcnemar_p, gate_success, make_model, masked_rollout,
    objective, run_arm, summarize, wilson_ci,
)

SMOKE_SEED = 1009


def test_frozen_source_and_independent_splits():
    assert CONFIG["study_id"] == "BPD-PD-02"
    assert len(SEEDS) == len(set(SEEDS)) == 24
    assert 71 not in SEEDS and 89 not in SEEDS and SMOKE_SEED not in SEEDS
    assert ARMS == ("hard_disjoint", "soft_disjoint", "hard_mixed", "hard_frozen")
    for seed in SEEDS:
        assert len({
            CONFIG["source"][f"{x}_generator_offset"] + seed
            for x in ("train", "calibration", "heldout")
        }) == 3
    assert CONFIG["training"]["stage1_steps"] == 200
    assert CONFIG["primary"]["hard_success_min"] == 18
    assert CONFIG["stage2"]["applies_to"] == "hard_disjoint"


def test_true_binary_forward_and_surrogate_backward():
    p = torch.tensor([.1, .499, .5, .9], requires_grad=True)
    hard = BinarySTE.apply(p)
    assert torch.equal(hard, torch.tensor([0., 0., 1., 1.]))
    hard.sum().backward()
    assert torch.equal(p.grad, torch.ones_like(p))

    bpd, gate, _ = make_model(SMOKE_SEED, frozen=False)
    assert isinstance(bpd, BPDModel)
    x = draw_batch(torch.Generator().manual_seed(102009), 8, mixed=False)
    y, h, probability = masked_rollout(bpd, gate, x, hard=True)
    assert y.shape == (8, 4, 1, 2)
    assert h.shape == probability.shape == (8, 4, 1, 2)
    assert torch.equal(h.detach(), (probability >= .5).float())
    assert bpd.wrappers["core"].calls == 4
    assert bpd.wrappers["core"].mode == "target"
    parts = objective(bpd, gate, x, hard=True)
    assert parts["forward_binary"].item()
    parts["total"].backward()
    assert all(p.grad is not None for p in bpd.component_parameters())
    assert sum(float(p.grad.square().sum()) for p in bpd.component_parameters()) > 0
    assert sum(float(p.grad.square().sum()) for p in gate.parameters()) > 0


def test_source_support_and_identical_init_between_matched_arms():
    hard, _, h = make_model(SMOKE_SEED, frozen=False)
    soft, _, s = make_model(SMOKE_SEED, frozen=False)
    mixed, _, m = make_model(SMOKE_SEED, frozen=False)
    assert h == s == m
    assert torch.equal(hard.wrappers["core"].V, soft.wrappers["core"].V)
    assert torch.equal(soft.wrappers["core"].U, mixed.wrappers["core"].U)
    x1 = draw_batch(torch.Generator().manual_seed(101009), 16, mixed=False)
    x2 = draw_batch(torch.Generator().manual_seed(101009), 16, mixed=False)
    assert torch.equal(x1, x2)
    assert torch.all(x1[:, :2, :, 1] == 0)
    assert torch.all(x1[:, 2:, :, 0] == 0)
    assert torch.all(x1[:, :2, :, 0] != 0)
    assert torch.all(x1[:, 2:, :, 1] != 0)
    xm = draw_batch(torch.Generator().manual_seed(101009), 16, mixed=True)
    assert torch.all(xm != 0)


def test_phase_assignment_needs_one_global_mapping():
    good = torch.tensor([[[[1.,0.]],[[1.,0.]],[[0.,1.]],[[0.,1.]]]]).expand(8,-1,-1,-1)
    bad = torch.cat([good[:4], 1-good[4:]], dim=0)
    dense = torch.ones_like(good)
    assert assignment_fraction(good) == 1
    assert assignment_fraction(bad) == .5
    assert assignment_fraction(dense) == 0
    example = {
        "binary_mse": 0.01,
        "axis_match": .97,
        "phase_assignment_fraction": 1.0,
        "hard_forward_exact_binary": True,
    }
    assert gate_success(example, factor_update_norm=1., first_grad_norm=1.)
    assert not gate_success(example, factor_update_norm=0., first_grad_norm=1.)


def test_exact_paired_statistics_and_wilson():
    assert exact_mcnemar_p(0, 0) == 1
    assert exact_mcnemar_p(6, 0) == 0.03125
    assert exact_mcnemar_p(3, 3) == 1
    assert exact_mcnemar_p(8, 1) == 2 * (1 + 9) / 512
    for a, b in ((0, 24), (12, 24), (24, 24)):
        lo, hi = wilson_ci(a, b)
        assert 0 <= lo <= a / b <= hi <= 1


def test_excluded_seed_smoke_with_full_wrapper_and_stage2():
    rows = [
        run_arm(SMOKE_SEED, key, stage1_steps=5, stage2_steps=2,
                batch=8, calibration_count=16, heldout_count=16)
        for key in ARMS
    ]
    assert len({r["initial_parameter_sha256"] for r in rows[:3]}) == 1
    assert len(rows[0]["stage2_candidates"]) == 3
    assert all(r["factors_bitwise_unchanged_in_stage2"] for r in rows)
    assert rows[0]["factor_stage1_update_norm"] > 0
    assert rows[1]["factor_stage1_update_norm"] > 0
    assert rows[2]["factor_stage1_update_norm"] > 0
    assert rows[3]["factor_stage1_update_norm"] == 0
    assert rows[0]["factor_first_step_grad_norm"] > 0
    assert rows[0]["gate_first_step_grad_norm"] > 0
    assert rows[3]["factor_first_step_grad_norm"] == 0
    assert all(r["stage1_primary_heldout"]["hard_forward_exact_binary"] for r in rows)
    assert not summarize(rows, registered=False)["is_full_registered_run"]
    assert not summarize(rows, registered=False)["registered_primary_criterion_passed"]
    cal = rows[0]["stage1_calibration"]
    selected = rows[0]["stage2_selected_coefficient"]
    chosen = next((r for r in rows[0]["stage2_candidates"] if r["coefficient"] == selected), None)
    if chosen:
        assert chosen["calibration_feasible"]
        assert chosen["calibration_binary_mse"] <= cal["binary_mse"] + .005 + 1e-9
        assert chosen["calibration_gate_fraction"] <= cal["binary_gate_fraction"] + .005 + 1e-9
