"""Native BPDModel/TiedComponentLinear parity on excluded engineering seed 901."""

import json
import torch
from torch import nn

from mechinterp.bpd.components import (
    BPDModel, ComponentIntervention, PassMaskSchedule,
)
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.theory.sf04_source_binding.experiment import (
    COMPONENTS, H, WIDTH, component_writes, paired_source_swap,
    select_components, train_factors, train_teacher,
)


class Core(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.core = nn.Linear(WIDTH, H, bias=False)
        with torch.no_grad():
            self.core.weight.copy_(teacher.C.detach().T)

    def forward(self, x):
        return self.core(x)


def check_native(seed=901):
    torch.set_num_threads(1)
    teacher = train_teacher(seed)
    factors = train_factors(seed, teacher)
    bpd = BPDModel(Core(teacher), {"core": COMPONENTS}, max_passes=3,
                   ablation=AblationConfig(mode="zero", sphere_rescale=False),
                   init_svd=False)
    wrapper = bpd.wrappers["core"]
    with torch.no_grad():
        wrapper.V.copy_(factors.V)
        wrapper.U.copy_(factors.U)
        recipient, donor, donor_target = paired_source_swap(seed + 710300, 72)
        first, second, query, _, _ = recipient
        h0 = torch.zeros(72, H)
        h1 = teacher.step(h0, first)
        selected, _ = select_components(seed, teacher, factors)
        comp = factors.weight()

        # Three tied physical writes executed under the native component wrapper.
        schedule = PassMaskSchedule(torch.ones(72, 3, 1, COMPONENTS),
                                    torch.zeros(72, 3, 1))
        with bpd.masked_components({"core": schedule}):
            native1 = h0 + bpd.target_model(teacher.features(h0, first).unsqueeze(1))[:, 0]
            native2 = native1 + bpd.target_model(teacher.features(native1, second).unsqueeze(1))[:, 0]
            native3 = native2 + bpd.target_model(teacher.features(native2, query).unsqueeze(1))[:, 0]
        independent1 = teacher.features(h0, first) @ comp
        independent2 = independent1 + teacher.features(independent1, second) @ comp
        independent3 = independent2 + teacher.features(independent2, query) @ comp
        forward_error = max(float((native1-independent1).abs().max()),
                            float((native2-independent2).abs().max()),
                            float((native3-independent3).abs().max()))

        # Target-mode native component donor swap on second write, before
        # downstream dark projection. Both source histories share h1.
        read_rec = teacher.features(h1, second)
        read_donor = teacher.features(h1, donor)
        donor_acts = (read_donor @ factors.V[:, selected]).unsqueeze(1)
        wrapper.mode = "target"
        wrapper.reset_pass_state(pass_offset=1)
        wrapper.intervention = ComponentIntervention(
            components=tuple(selected), passes=frozenset({1}),
            mode="swap_acts", donor_acts=donor_acts,
        )
        try:
            native_raw = bpd.target_model(read_rec.unsqueeze(1))[:, 0]
        finally:
            wrapper.intervention = None
        expected_raw = read_rec @ teacher.C + component_writes(
            teacher, factors, h1, second, donor)[:, selected, :].sum(1)
        swap_error = float((native_raw - expected_raw).abs().max())

    outcome = {
        "seed": seed, "native_tied_three_pass_error": forward_error,
        "native_source_swap_error": swap_error,
        "passed": forward_error <= 1e-5 and swap_error <= 1e-5,
    }
    if not outcome["passed"]:
        raise AssertionError(outcome)
    return outcome


if __name__ == "__main__":
    print(json.dumps(check_native(), indent=2))
