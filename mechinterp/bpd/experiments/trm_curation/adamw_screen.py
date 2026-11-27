"""Read-only dense CPU AdamW step prediction for candidate screening.

Gradients must come from the exact parent checkpoint and the candidate loss.
Forward/RNG/native mutable-state isolation belongs to the caller. This predicts
parameter displacement, not the nonlinear calibration loss after a step.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch


@torch.no_grad()
def screen_adamw(optimizer: torch.optim.AdamW,
                 gradients: Sequence[torch.Tensor | None],
                 calibration: Sequence[torch.Tensor | None],
                 clip: float = 1.0) -> dict:
    """Predict one clipped AdamW step without modifying live parameters/state."""
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError('AdamW required')
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError('invalid clip')
    params = [p for group in optimizer.param_groups for p in group['params']]
    if len(params) != len(gradients) or len(params) != len(calibration):
        raise ValueError('gradient list length')
    if len({id(p) for p in params}) != len(params):
        raise ValueError('duplicate optimizer parameter')
    for p, g, c in zip(params, gradients, calibration, strict=True):
        if p.device.type != 'cpu' or p.dtype not in (torch.float32, torch.float64):
            raise ValueError('dense CPU float32/float64 required')
        if not torch.isfinite(p).all():
            raise ValueError('nonfinite parameter')
        for x in (g, c):
            if x is not None and (x.shape != p.shape or x.dtype != p.dtype
                                  or x.device != p.device or x.layout != torch.strided
                                  or not torch.isfinite(x).all()):
                raise ValueError('invalid gradient')
    present = [g.detach() for g in gradients if g is not None]
    # Match PyTorch L2 clipping, including its 1e-6 stabilizer.
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(g) for g in present
    ])) if present else torch.tensor(0.0)
    coef = torch.clamp(clip / (norm + 1e-6), max=1.0)
    if not torch.isfinite(norm):
        raise ValueError('nonfinite gradient norm')
    deltas = []
    offset = 0
    for group in optimizer.param_groups:
        if any(group.get(k, False) for k in ('fused', 'capturable', 'differentiable')):
            raise ValueError('unsupported optimizer mode')
        if isinstance(group['lr'], torch.Tensor):
            raise ValueError('scalar learning rate required')
        lr, wd, eps = map(float, (group['lr'], group['weight_decay'], group['eps']))
        b1, b2 = map(float, group['betas'])
        if not all(map(math.isfinite, (lr, wd, eps, b1, b2))) or not (
            lr >= 0 and wd >= 0 and eps > 0 and 0 <= b1 < 1 and 0 <= b2 < 1
        ):
            raise ValueError('invalid optimizer hyperparameters')
        for p in group['params']:
            g = gradients[offset]
            offset += 1
            if g is None:
                # PyTorch skips weight decay, moments and step for missing grads.
                deltas.append(torch.zeros_like(p))
                continue
            g = g.detach() * coef.to(dtype=g.dtype)
            if group.get('maximize', False):
                g = -g
            st = optimizer.state.get(p, {})
            if st:
                m, v = st['exp_avg'].clone(), st['exp_avg_sq'].clone()
                step = float(st['step']) + 1
                if step < 1 or not step.is_integer():
                    raise ValueError('invalid optimizer step')
                vmax = st['max_exp_avg_sq'].clone() if group['amsgrad'] else None
            else:
                m, v, step = torch.zeros_like(p), torch.zeros_like(p), 1.0
                vmax = torch.zeros_like(p) if group['amsgrad'] else None
            m.lerp_(g, 1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            if vmax is not None:
                torch.maximum(vmax, v, out=vmax)
                v = vmax
            denom = v.sqrt().div_(math.sqrt(1 - b2 ** step)).add_(eps)
            candidate = p.detach().clone().mul_(1 - lr * wd)
            candidate.addcdiv_(m, denom, value=-lr / (1 - b1 ** step))
            delta = candidate - p.detach()
            if not torch.isfinite(delta).all():
                raise ValueError('nonfinite displacement')
            deltas.append(delta)
    score = -sum(float((c.detach().double() * d.double()).sum())
                 for c, d in zip(calibration, deltas, strict=True) if c is not None)
    return {'predicted_gain': score, 'deltas': tuple(deltas),
            'update_norm': math.sqrt(sum(float(d.double().square().sum()) for d in deltas)),
            'gradient_norm': float(norm), 'clip_scale': float(coef)}
