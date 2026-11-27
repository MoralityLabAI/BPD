"""Batched analytic AdamW previews (engineering benchmark for MATH_REVIEW open problem 5).

Per-candidate gradients for all window proposals come from one ``torch.func``
vmap call; the clipped AdamW step is then evaluated in closed form from the
live optimizer moments, without copying or restoring model/optimizer state.
Agreement with the transactional ``preview_update`` is up to float32 rounding,
not bitwise; the benchmark reports max abs difference, rank agreement and
top-k set agreement. Not used by any frozen arm.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


def candidate_gradients(model, xs: torch.Tensor, ys: torch.Tensor):
    """xs [K, B, T, D], ys [K, B] -> dict name -> [K, *param.shape]."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}

    def loss(p, x, y):
        logits = functional_call(model, (p, buffers), (x,))
        return F.cross_entropy(logits, y)

    model.train()
    return vmap(grad(loss), in_dims=(None, 0, 0))(params, xs, ys)


def analytic_adamw_gain(model, optimizer, grads: dict, cal_grad, *, clip=1.0):
    """Closed-form clipped AdamW step per candidate and predicted gain -g_cal . dtheta."""
    group = optimizer.param_groups[0]
    lr, (b1, b2), eps, wd = group['lr'], group['betas'], group['eps'], group['weight_decay']
    names = [k for k, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    K = next(iter(grads.values())).shape[0]
    norms = torch.sqrt(sum(grads[n].reshape(K, -1).double().square().sum(1) for n in names))
    scale = torch.clamp(clip / (norms + 1e-6), max=1.0)
    gain = torch.zeros(K, dtype=torch.float64)
    sq = torch.zeros(K, dtype=torch.float64)
    for n, p, gc in zip(names, params, cal_grad):
        st = optimizer.state.get(p, {})
        m = st.get('exp_avg', torch.zeros_like(p))
        v = st.get('exp_avg_sq', torch.zeros_like(p))
        t = float(st['step']) + 1 if 'step' in st else 1.0
        g = grads[n] * scale.view(-1, *([1] * p.dim())).to(p.dtype)
        m1 = b1 * m + (1 - b1) * g
        v1 = b2 * v + (1 - b2) * g * g
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        step = -lr * wd * p.detach() - (lr / bc1) * m1 / (v1.sqrt() / (bc2 ** .5) + eps)
        gain -= (gc.double() * step.double()).reshape(K, -1).sum(1)
        sq += step.double().reshape(K, -1).square().sum(1)
    return gain, sq.sqrt()


def batched_previews(model, optimizer, batches, cal_grad, *, clip=1.0):
    from mechinterp.bpd.experiments.trm_curation.transaction import _restore_native_state, _snapshot_native_state
    t0 = time.process_time()
    xs = torch.stack([torch.cat((b.receiver, b.hybrid)) for b in batches])
    ys = torch.stack([torch.cat((b.labels_r, b.labels_h)) for b in batches])
    native = _snapshot_native_state(model)
    flags = [(m, m.training) for m in model.modules()]
    try:
        grads = candidate_gradients(model, xs, ys)
    finally:
        _restore_native_state(native)
        for m, f in flags:
            m.training = f
    gain, norm = analytic_adamw_gain(model, optimizer, grads, cal_grad, clip=clip)
    return dict(predicted_gain=gain.tolist(), update_norm=norm.tolist(), cpu_seconds=time.process_time() - t0)
