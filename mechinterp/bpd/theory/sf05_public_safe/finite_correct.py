"""SF05 A2: privileged nonlinear public-output level-set correction.

This solver may query and differentiate the future public output. It is a
research oracle, not a deployable privacy barrier or evidence of a canonical
BPD component.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import torch


@dataclass
class FiniteRepair:
    delta: torch.Tensor
    public_error_before: torch.Tensor
    public_error_after: torch.Tensor
    converged: torch.Tensor
    correction_norm: torch.Tensor
    iterations: int
    failed_zero_gradient: torch.Tensor


def finite_public_repair(
    state: torch.Tensor,
    dark_delta: torch.Tensor,
    public_future: Callable[[torch.Tensor], torch.Tensor],
    *,
    max_iterations: int = 8,
    tolerance: float = 1e-4,
    per_step_relative_cap: float = .25,
    per_step_absolute_cap: float = .5,
    gradient_tol: float = 1e-9,
) -> FiniteRepair:
    """Damped projected Newton correction onto a finite public output level set.

    Each step preserves current q(h)=h[:,0] exactly. Backtracking only
    accepts a reduction in actual future public readout difference.
    Rows with small gradients or insufficient correction dose are retained
    and reported, including all nonconverged cases.
    """
    if state.shape != dark_delta.shape or state.ndim != 2:
        raise ValueError("state and dark_delta must be [B,D]")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("max_iterations >=1 and tolerance >0 required")
    if not torch.isfinite(state).all() or not torch.isfinite(dark_delta).all():
        raise ValueError("nonfinite state or edit")
    v = dark_delta.detach().clone()
    v[:, 0] = 0
    original = v.clone()
    dose = v.norm(dim=1, keepdim=True)
    cap = torch.minimum(torch.full_like(dose, per_step_absolute_cap),
                        per_step_relative_cap * dose)
    with torch.no_grad():
        base = public_future(state).detach()
        before = (public_future(state+v)-base).detach()
    failed = torch.zeros(len(v), dtype=torch.bool, device=v.device)
    actual_iter = 0
    for it in range(max_iterations):
        actual_iter = it + 1
        x = (state+v).detach().requires_grad_(True)
        with torch.enable_grad():
            y = public_future(x)
            grad = torch.autograd.grad(y.sum(), x)[0].detach()
        residual = (y.detach()-base)
        active = residual.abs() > tolerance
        if not bool(active.any()):
            break
        gp = grad.clone()
        gp[:, 0] = 0
        gnorm2 = gp.square().sum(1, keepdim=True)
        viable = (gnorm2[:, 0] > gradient_tol**2) & (cap[:, 0] > 0)
        failed |= active & ~viable
        rawstep = -residual[:, None] * gp / gnorm2.clamp_min(1e-20)
        stepnorm = rawstep.norm(dim=1, keepdim=True)
        scaledstep = rawstep * torch.minimum(
            torch.ones_like(cap), cap / stepnorm.clamp_min(1e-12))
        scaledstep[~(active & viable)] = 0
        best = v
        with torch.no_grad():
            best_error = residual.abs()
            for rate in (1.0, .5, .25, .125):
                cand = v + rate*scaledstep
                cand[:, 0] = 0
                err = (public_future(state+cand)-base).abs()
                better = err < best_error
                best = torch.where(better[:, None], cand, best)
                best_error = torch.where(better, err, best_error)
        v = best.detach()
    with torch.no_grad():
        after = (public_future(state+v)-base).detach()
        change = (v-original).norm(dim=1)
    if not torch.equal((state+v)[:, 0], state[:, 0]):
        raise AssertionError("finite correction changed the current readout")
    return FiniteRepair(v, before, after, after.abs()<=tolerance,
                        change, actual_iter, failed)
