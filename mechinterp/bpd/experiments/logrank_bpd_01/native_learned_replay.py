"""Replay a saved learned 08 component bank through native BPD.

Engineering test only: externally learned V, fixed scalar U, frozen routes.
This does not run BPDTrainer or retrain the router.
"""
import base64
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mechinterp.bpd.components import PassMaskSchedule, TiedComponentLinear
from mechinterp.bpd.config import AblationConfig


@torch.no_grad()
def replay():
    data = json.loads(Path(__file__).with_name("learned_fixture.json").read_text())
    k = data["k"]
    v = torch.from_numpy(np.frombuffer(base64.b64decode(data["V_b64"]),
                                        dtype="<f4").copy().reshape(k, k))
    x = 2 * ((torch.tensor(data["ids"])[:, None] >> torch.arange(k)) & 1).float() - 1
    n = len(x)
    linear = nn.Linear(k, 1, bias=False)
    linear.weight.copy_(v.sum(1).reshape(1, k))
    layer = TiedComponentLinear(
        linear, k, max_passes=3, init_svd=False,
        ablation=AblationConfig(mode="zero", sphere_rescale=False))
    layer.V.copy_(v)
    layer.U.fill_(1)
    layer.mode = "component"
    ones = torch.ones(n, 1, 1, k)
    layer.reset_pass_state()
    layer.schedule = PassMaskSchedule(comp_mask=ones)
    full = layer(x[:, None, :])
    torch.testing.assert_close(full, linear(x[:, None, :]), atol=2e-5, rtol=2e-5)
    layer.reset_pass_state()
    layer.schedule = PassMaskSchedule(comp_mask=torch.zeros_like(ones),
                                      delta_mask=torch.zeros(n, 1, 1))
    assert torch.equal(layer(x[:, None, :]), torch.zeros(n, 1, 1))
    responses = []
    for route in data["routes"]:
        indices, signs, op = route["indices"], route["signs"], route["op"]
        t = len(indices)
        mask = torch.zeros(n, t, 1, k)
        for step, i in enumerate(indices):
            mask[:, step, 0, i] = 1
        layer.reset_pass_state()
        layer.schedule = PassMaskSchedule(comp_mask=mask,
                                          delta_mask=torch.zeros(n, t, 1))
        a = torch.ones(n)
        for i, s in zip(indices, signs):
            read = layer(x[:, None, :]).reshape(n)
            direct = x @ v[:, i]
            torch.testing.assert_close(read, direct, atol=2e-5, rtol=2e-5)
            lit = torch.where(read >= 0, 1., -1.) * s
            a = a * lit if op == "parity" else torch.minimum(a, lit)
        assert layer.calls == t
        responses.append(a)
    got = "".join("1" if b else "0" for b in
                  (torch.stack(responses, 1) > 0).flatten().tolist())
    assert got == data["predictions"]
    print({"root": data["root"], "native_saved_predictions": len(got), "status": "PASS"})


if __name__ == "__main__":
    torch.set_num_threads(1)
    replay()
