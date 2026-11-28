"""Native BPD component-mask integration for a planted compositional reader.

This is an exact engineering fixture; no learned decomposition or causal
model evidence is claimed. The full identity bank retains rank k.
"""
import itertools
import torch
from torch import nn
from mechinterp.bpd.components import PassMaskSchedule, TiedComponentLinear
from mechinterp.bpd.config import AblationConfig


def native_read(x, indices, signs, operator):
    """x [batch,k] bipolar. Return reply and per-pass selected writes."""
    if operator not in ("parity", "and") or not indices or len(indices) != len(signs):
        raise ValueError("invalid nonempty program")
    if len(set(indices)) != len(indices) or any(i < 0 or i >= x.shape[1] for i in indices):
        raise ValueError("invalid coordinates")
    if any(s not in (-1, 1) for s in signs) or not bool(((x == 1) | (x == -1)).all()):
        raise ValueError("expected bipolar data")
    batch, k = x.shape
    linear = nn.Linear(k, k, bias=False, device=x.device, dtype=torch.float32)
    with torch.no_grad():
        eye = torch.eye(k, device=x.device)
        linear.weight.copy_(eye)
        layer = TiedComponentLinear(linear, k, max_passes=len(indices),
            ablation=AblationConfig(mode="zero", sphere_rescale=False))
        layer.V.copy_(eye)
        layer.U.copy_(eye)
    assert torch.equal(layer.weight_delta(), torch.zeros_like(eye))
    masks = x.new_zeros((batch, len(indices), 1, k), dtype=torch.float32)
    for t, i in enumerate(indices):
        masks[:, t, 0, i] = 1
    layer.mode = "component"
    layer.schedule = PassMaskSchedule(comp_mask=masks)
    layer.reset_pass_state()
    a = torch.ones(batch, device=x.device)
    writes = []
    with torch.no_grad():
        for i, s in zip(indices, signs):
            write = layer(x.float().unsqueeze(1)).squeeze(1)
            expected = torch.zeros_like(write)
            expected[:, i] = x[:, i]
            assert torch.equal(write, expected)
            v = write.sum(-1) * s
            a = a * v if operator == "parity" else torch.minimum(a, v)
            writes.append(write)
        assert layer.calls == len(indices)
        # Full-head fidelity and zero-mask control on the identical target.
        layer.reset_pass_state()
        layer.schedule = PassMaskSchedule(comp_mask=torch.ones_like(masks))
        assert torch.equal(layer(x.float().unsqueeze(1)).squeeze(1), x.float())
        layer.reset_pass_state()
        layer.schedule = PassMaskSchedule(comp_mask=torch.zeros_like(masks))
        assert torch.equal(layer(x.float().unsqueeze(1)).squeeze(1), torch.zeros_like(x.float()))
    return a, writes


def smoke():
    k = 4
    x = torch.tensor(list(itertools.product((-1., 1.), repeat=k)))
    programs = 0
    decisions = 0
    for d in (1, 2, 3):
        for indices in itertools.combinations(range(k), d):
            for signs in itertools.product((-1, 1), repeat=d):
                literals = x[:, indices] * torch.tensor(signs)
                for op in ("parity", "and"):
                    actual, _ = native_read(x, indices, signs, op)
                    expected = literals.prod(-1) if op == "parity" else literals.min(-1).values
                    assert torch.equal(actual, expected)
                    programs += 1
                    decisions += len(x)
    print({"programs": programs, "decisions": decisions, "status": "PASS"})


if __name__ == "__main__":
    torch.set_num_threads(1)
    smoke()
