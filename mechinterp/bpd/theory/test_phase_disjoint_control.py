"""Source/engineering tests for the registered BPD-PD-01 training control."""
from __future__ import annotations

import torch

from mechinterp.bpd.theory.phase_disjoint_control import (
    PhaseTeacher, SharedPhaseGate, draw_batch, loss_parts,
    run_arm, masked_forward,
)
from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig


def test_phase_support_is_disjoint_and_mixed_control_is_dense():
    for seed in (71, 89):
        x = draw_batch(torch.Generator().manual_seed(100000 + seed), 32, mixed=False)
        assert x.shape == (32, 4, 1, 2)
        assert torch.all(x[:, :2, :, 1] == 0)
        assert torch.all(x[:, 2:, :, 0] == 0)
        assert torch.all(x[:, :2, :, 0].abs() >= .75)
        assert torch.all(x[:, 2:, :, 1].abs() >= .75)
        assert torch.all(x.abs() <= 1.25)
        mixed = draw_batch(torch.Generator().manual_seed(100000 + seed), 32, mixed=True)
        assert torch.all(mixed.abs() >= .75)
        assert torch.all(mixed.abs() <= 1.25)


def test_real_bpd_shared_wrapper_and_gradient_reaches_factors():
    torch.manual_seed(71)
    bpd = BPDModel(
        PhaseTeacher(), {"core": 2}, max_passes=4,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=False,
    )
    gate = SharedPhaseGate()
    x = draw_batch(torch.Generator().manual_seed(100071), 16, mixed=False)
    y, masks = masked_forward(bpd, gate, x)
    assert y.shape == x.shape and masks.shape == (16, 4, 1, 2)
    assert bpd.wrappers["core"].calls == 4
    assert bpd.wrappers["core"].mode == "target"  # context restores original
    parts = loss_parts(bpd, gate, x)
    parts["stage1"].backward()
    params = bpd.component_parameters()
    assert all(p.grad is not None for p in params)
    assert sum(float(p.grad.square().sum()) for p in params) > 0.0


def test_factor_training_and_stage2_freezing_with_sealed_eval():
    for mixed, frozen in ((False, False), (True, False), (False, True)):
        r = run_arm(71, mixed=mixed, freeze_dictionary=frozen,
                    steps=4, stage2_steps=2, batch_size=8, calibration_size=16, heldout_size=16)
        assert r["study"] == "BPD-PD-01"
        assert r["stage2_dictionary_bitwise_frozen"]
        assert len(r["stage2_candidates"]) == 3
        assert r["selected_switch_coeff"] in (0.0, .01, .05, .20)
        if frozen:
            assert r["factor_stage1_update_norm"] == 0.0
        else:
            assert r["factor_stage1_update_norm"] > 0.0
            assert r["factor_first_step_grad_norm"] > 0.0
        assert r["calibration_selected"]["reconstruction_mse"] <= (
            r["calibration_stage1"]["reconstruction_mse"] + .005 + 1e-8)
        assert r["calibration_selected"]["sparsity"] <= (
            r["calibration_stage1"]["sparsity"] + .005 + 1e-8)
        assert len(r["heldout_selected"]["gate_mean_by_pass"]) == 4
