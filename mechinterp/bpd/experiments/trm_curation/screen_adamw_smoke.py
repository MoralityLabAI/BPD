"""Independent comparison with live torch.optim.AdamW, no model training claim."""
import copy
import json
import itertools
import torch
from adamw_screen import screen_adamw

torch.set_num_threads(1)
torch.manual_seed(20261127)

def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b, strict=True))
    return a == b

cases = 0
max_delta = max_score = 0.0
for dtype, ams, maximize, warmup, clip, missing in itertools.product(
    [torch.float32, torch.float64], [False, True], [False, True], [0, 4],
    [0.05, 1.0, 100.0], [False, True]
):
    ps = [torch.nn.Parameter(torch.randn(n, dtype=dtype)) for n in (9, 7, 3)]
    opt = torch.optim.AdamW([{'params': ps[:2], 'lr': 0.003, 'weight_decay': 0.01},
                            {'params': ps[2:], 'lr': 0.001, 'weight_decay': 0.2,
                             'betas': (0.7, 0.91), 'eps': 1e-6}],
                           amsgrad=ams, maximize=maximize, foreach=False, fused=False)
    for _ in range(warmup):
        for p in ps:
            p.grad = torch.randn_like(p)
        opt.step()
    old = [p.detach().clone() for p in ps]
    old_grads = [None if p.grad is None else p.grad.clone() for p in ps]
    state = copy.deepcopy(opt.state_dict())
    rng = torch.get_rng_state().clone()
    g = [torch.randn_like(p) for p in ps]
    if missing:
        g[1] = None
    else:
        g[1].zero_()
    c = [torch.randn_like(p) for p in ps]
    rng = torch.get_rng_state().clone()
    pred = screen_adamw(opt, g, c, clip)
    assert all(torch.equal(p, q) for p, q in zip(ps, old, strict=True))
    assert equal_state(opt.state_dict(), state)
    assert all((p.grad is None and x is None) or (x is not None and torch.equal(p.grad, x))
               for p, x in zip(ps, old_grads, strict=True))
    assert torch.equal(torch.get_rng_state(), rng)
    for p, x in zip(ps, g, strict=True):
        p.grad = None if x is None else x.clone()
    torch.nn.utils.clip_grad_norm_(ps, clip, error_if_nonfinite=True, foreach=False)
    opt.step()
    actual = [p.detach() - q for p, q in zip(ps, old, strict=True)]
    d = max(float((x - y).abs().max()) for x, y in zip(actual, pred['deltas'], strict=True))
    score = -sum(float((x.double() * y.double()).sum()) for x, y in zip(c, actual, strict=True))
    err = abs(score - pred['predicted_gain'])
    assert d <= (2e-7 if dtype == torch.float32 else 1e-14), (dtype, ams, maximize, warmup, clip, missing, d)
    assert err <= (2e-6 if dtype == torch.float32 else 1e-13), err
    cases += 1
    max_delta = max(max_delta, d)
    max_score = max(max_score, err)
print(json.dumps({'scope':'synthetic dense CPU AdamW parity, not ComponentTRM training',
                  'torch_version':torch.__version__,'cases':cases,'max_abs_displacement_error':max_delta,
                  'max_abs_linear_score_error':max_score,'parameter_optimizer_grad_rng_restoration':'PASS'}))
