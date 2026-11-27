"""Expected one-dimensional shared-label width for BPD pass gates."""
from __future__ import annotations
import torch
from torch import Tensor


def expected_span(gates: Tensor, sites: Tensor) -> Tensor:
    """Independent Bernoulli sites; exact for hard gates. Last axis is visit.

    Empty and singleton subsets have zero width. Sites are fixed and sorted.
    For binary gates this is the span of all selected visits.
    """
    if sites.ndim != 1 or gates.ndim < 1 or gates.shape[-1] != sites.numel() or sites.numel() == 0:
        raise ValueError("invalid site axis")
    if sites.requires_grad or not torch.isfinite(sites).all() or (sites.diff() < 0).any():
        raise ValueError("sites must be fixed finite sorted coordinates")
    if not torch.is_floating_point(gates) or not torch.isfinite(gates).all() or ((gates < 0) | (gates > 1)).any():
        raise ValueError("gates must be probabilities")
    sites = sites.to(device=gates.device, dtype=gates.dtype)
    q = 1 - gates
    left = 1 - q.cumprod(-1)[..., :-1]
    right = 1 - q.flip(-1).cumprod(-1).flip(-1)[..., 1:]
    return (left * right * sites.diff()).sum(-1)


def spectral_span_penalty(comp_mask: Tensor, sites: Tensor, bits: Tensor) -> Tensor:
    """PassMaskSchedule.comp_mask [B,T,P,C]; bits [C]. Mean over B,P.

    A producer is included only if its gate is explicitly included in comp_mask.
    Bits are assigned costs, not estimates of mutual information.
    """
    if comp_mask.ndim != 4 or bits.shape != (comp_mask.shape[-1],) or comp_mask.numel() == 0:
        raise ValueError("invalid schedule or bits")
    if not torch.isfinite(bits).all() or (bits < 0).any():
        raise ValueError("invalid bits")
    spans = expected_span(comp_mask.permute(0, 2, 3, 1), sites)
    return (spans * bits.to(device=spans.device, dtype=spans.dtype)).sum(-1).mean()
