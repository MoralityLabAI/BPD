"""Readout-/public-Jacobian-safe interventions for the SF05 toy model.

For SF04 the actual current readout is state coordinate zero. The public
future readout is a scalar, so at most one additional independent tangent
constraint is imposed. Projections use physical hidden-state vectors;
no specific BPD singleton basis enters their definition.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class Projection:
    value: torch.Tensor
    removed: torch.Tensor
    public_tangent: torch.Tensor
    zero_public_gradient: torch.Tensor


def public_safe_projection(delta: torch.Tensor, public_gradient: torch.Tensor, *,
                           gradient_tol: float = 1e-8) -> Projection:
    """Closest current-readout-dark edit orthogonal to the public gradient."""
    if delta.ndim != 2 or delta.shape != public_gradient.shape or delta.shape[-1] < 3:
        raise ValueError("delta and public gradient must be [B,D] with D>=3")
    if not torch.isfinite(delta).all() or not torch.isfinite(public_gradient).all():
        raise ValueError("nonfinite intervention or public gradient")
    dark = delta.double().clone()
    dark[:, 0] = 0.0
    g = public_gradient.double().clone()
    g[:, 0] = 0.0
    g2 = g.square().sum(dim=-1, keepdim=True)
    tiny = g2 <= float(gradient_tol)**2
    amount = torch.where(tiny, torch.zeros_like(g2),
                         (dark*g).sum(dim=-1, keepdim=True) / g2.clamp_min(1e-300))
    safe = dark - amount*g
    safe[:, 0] = 0.0
    out = safe.to(delta.dtype)
    return Projection(value=out,
                      removed=(dark-safe).to(delta.dtype),
                      public_tangent=(g*safe).sum(dim=-1).to(delta.dtype),
                      zero_public_gradient=tiny.squeeze(-1))


def random_public_safe_sham(delta: torch.Tensor, public_gradient: torch.Tensor,
                            generator: torch.Generator) -> torch.Tensor:
    """Per-example L2 dose-matched random direction in the intersection K."""
    noise = torch.randn(delta.shape, generator=generator, dtype=delta.dtype,
                        device=delta.device)
    noise = public_safe_projection(noise, public_gradient).value
    dose = delta.norm(dim=-1, keepdim=True)
    return noise * (dose / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def signed_authorized_matched_sham(
    safe_delta: torch.Tensor, public_gradient: torch.Tensor,
    authorized_gradient: torch.Tensor, generator: torch.Generator,
) -> torch.Tensor:
    """Match physical dose and signed authorized tangent inside public-safe K.

    The sham is privileged: its component along the true downstream
    authorized gradient is copied from the selected edit. Only the
    remaining intersection-subspace direction is independently randomized.
    """
    if safe_delta.shape != authorized_gradient.shape:
        raise ValueError("authorized_gradient has wrong shape")
    guide = public_safe_projection(authorized_gradient, public_gradient).value.double()
    val = safe_delta.double()
    unit = guide / guide.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    alpha = (val*unit).sum(dim=-1, keepdim=True)
    residual_sq = (val.square().sum(dim=-1, keepdim=True)-alpha.square()).clamp_min(0)
    noise = random_public_safe_sham(val, public_gradient.double(), generator).double()
    noise -= (noise*unit).sum(dim=-1, keepdim=True)*unit
    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sham = alpha*unit + residual_sq.sqrt()*noise
    sham[:, 0] = 0.0
    return sham.to(safe_delta.dtype)


def physical_output_subspace_write(
    feature_delta: torch.Tensor, physical_weight: torch.Tensor, rank: int,
) -> tuple[torch.Tensor, float | None]:
    """Gauge-invariant rank-k output-state SVD projection of a physical write."""
    if physical_weight.ndim != 2 or feature_delta.shape[-1] != physical_weight.shape[0]:
        raise ValueError("feature delta/physical weight dimensions are incompatible")
    if rank < 1 or rank > min(physical_weight.shape):
        raise ValueError("rank outside physical operator range")
    _, singular, vh = torch.linalg.svd(physical_weight, full_matrices=False)
    write = feature_delta @ physical_weight
    basis = vh[:rank, :]
    projected = (write @ basis.T) @ basis
    gap = (float(singular[rank-1]-singular[rank]) if rank < len(singular) else None)
    return projected, gap


def linearized_authorized_capacity(authorized_gradient: torch.Tensor,
                                   public_gradient: torch.Tensor,
                                   safe_delta: torch.Tensor) -> torch.Tensor:
    """Max |authorized future linear response| at specified safe-edit norm."""
    a = public_safe_projection(authorized_gradient, public_gradient).value
    return safe_delta.norm(dim=-1) * a.norm(dim=-1)
