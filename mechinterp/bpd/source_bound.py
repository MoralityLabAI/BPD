"""Producer-bound span costs for BPD pass masks; coordinates are fixed."""
from __future__ import annotations

import torch
from torch import Tensor


def anchored_width(gates: Tensor, sites: Tensor) -> Tensor:
    """Expected span with a mandatory producer at sites[0], independent consumers."""
    if gates.ndim < 1 or sites.ndim != 1 or sites.numel() != gates.shape[-1] + 1:
        raise ValueError("sites must contain producer and consumers")
    if sites.requires_grad or not torch.isfinite(sites).all() or (sites.diff() < 0).any():
        raise ValueError("sites must be fixed, finite and sorted")
    if not torch.is_floating_point(gates) or not torch.isfinite(gates).all() or ((gates < 0) | (gates > 1)).any():
        raise ValueError("invalid gate probabilities")
    suffix = 1 - (1 - gates).flip(-1).cumprod(-1).flip(-1)
    return (suffix * sites.to(gates).diff()).sum(-1)


def source_bound_penalty(comp_mask: Tensor, sites: Tensor, bits: Tensor) -> Tensor:
    """Schedule [B,T,P,C]; fixed sites [T+1] and assigned bits [C].

    Soft gates use an independent-Bernoulli surrogate. For sampled hard masks,
    evaluate each joint sample directly and average across samples externally.
    """
    if comp_mask.ndim != 4 or comp_mask.numel() == 0 or bits.shape != (comp_mask.shape[-1],):
        raise ValueError("invalid schedule or bits")
    if bits.requires_grad or not torch.isfinite(bits).all() or (bits < 0).any():
        raise ValueError("bits must be fixed finite nonnegative prices")
    spans = anchored_width(comp_mask.permute(0, 2, 3, 1), sites)
    return (spans * bits.to(spans)).sum(-1).mean()
