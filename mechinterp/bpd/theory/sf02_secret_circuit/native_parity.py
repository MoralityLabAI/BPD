"""Compare independent trained factor execution to actual native BPDModel wrapper."""

from __future__ import annotations
import json
import torch
from torch import nn
from mechinterp.bpd.components import BPDModel, PassMaskSchedule, ComponentIntervention
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.theory.sf02_secret_circuit.experiment import (
    D, WIDTH, C, train_teacher, train_factors, sample, donor_pairs, factor_writes,
    calibrate,
)


class _Core(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.core = nn.Linear(WIDTH, D, bias=False)
        with torch.no_grad():
            self.core.weight.copy_(teacher.W2.T)

    def forward(self, x):
        return self.core(x)


def native_parity(seed=901, n=128, teacher_steps=850, factor_steps=400):
    torch.set_num_threads(1)
    teacher = train_teacher(seed, teacher_steps)
    factors = train_factors(seed, teacher, factor_steps)
    bpd = BPDModel(
        _Core(teacher), {"core": C}, max_passes=2,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=False,
    )
    wrapper = bpd.wrappers["core"]
    with torch.no_grad():
        wrapper.V.copy_(factors.V)
        wrapper.U.copy_(factors.U)
        h, d, _, _, _ = donor_pairs(seed + 710300, n)
        a = teacher.features(h)
        independent_one = a @ factors.weight()
        independent_two = teacher.features(independent_one) @ factors.weight()
        ones = torch.ones((n, 2, 1, C))
        zeros = torch.zeros((n, 2, 1))
        with bpd.masked_components({"core": PassMaskSchedule(ones, zeros)}):
            native_one = bpd.target_model(a.unsqueeze(1))[:, 0]
            native_two = bpd.target_model(teacher.features(native_one).unsqueeze(1))[:, 0]
        rollout_max = max(float((native_one-independent_one).abs().max()),
                          float((native_two-independent_two).abs().max()))
        # Target-mode intervention must reproduce the additive native component
        # write *before* the separate current-readout dark projection.
        selected, _ = calibrate(seed, teacher, factors)
        recipient_features = teacher.features(h)
        donor_acts = (teacher.features(d) @ factors.V[:, selected]).unsqueeze(1)
        wrapper.mode = "target"
        wrapper.reset_pass_state(pass_offset=0)
        wrapper.intervention = ComponentIntervention(
            components=tuple(selected), passes=frozenset({0}),
            mode="swap_acts", donor_acts=donor_acts)
        try:
            native_write = bpd.target_model(recipient_features.unsqueeze(1))[:, 0]
        finally:
            wrapper.intervention = None
        expected_write = teacher(h) + factor_writes(teacher, factors, h, d)[:, selected, :].sum(1)
        swap_max = float((native_write - expected_write).abs().max())
    result = {
        "seed": seed,
        "native_fullmask_rollout_max_error": rollout_max,
        "native_swap_acts_max_error": swap_max,
        "passed": rollout_max <= 1e-5 and swap_max <= 1e-5
    }
    if not result["passed"]:
        raise AssertionError(f"native BPD parity failed: {result}")
    return result


if __name__ == "__main__":
    print(json.dumps(native_parity(), indent=2))
