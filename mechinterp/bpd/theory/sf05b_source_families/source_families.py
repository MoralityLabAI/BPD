"""Deterministic, structurally disjoint multi-user source-history generators.

F_train: iid source tags and 2/3/4 writes. F_heldout: 5/6 writes
in alternating/overwrite templates with independent random permutations
of source names. No private user data or external APIs are used.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.nn import functional as F

SOURCES, ROLES, WIDTH, MAX_WRITES = 3, 4, 11, 6
HELDOUT_TEMPLATES = {5: (0, 1, 0, 2, 1), 6: (0, 1, 2, 0, 1, 2)}


@dataclass
class HistoryBatch:
    events: torch.Tensor
    event_mask: torch.Tensor
    query: torch.Tensor
    target: torch.Tensor
    source_ids: torch.Tensor
    values: torch.Tensor
    query_role: torch.Tensor
    public_value: torch.Tensor
    lengths: torch.Tensor
    source_blind_conditional_mean: torch.Tensor
    family: str


def _generator(seed: int):
    return torch.Generator().manual_seed(int(seed))


def target_from_records(tags, values, lengths, query_role, public):
    """Exact last-write-wins target; role 3 returns the public scalar."""
    if tags.shape != values.shape or tags.ndim != 2:
        raise ValueError("tags/values must be [N,L]")
    n, L = tags.shape
    if lengths.shape != (n,) or query_role.shape != (n,) or public.shape != (n,):
        raise ValueError("invalid role, length or public shape")
    target = torch.zeros(n, dtype=values.dtype)
    for j in range(L):
        valid = (j < lengths) & (query_role != 3) & (query_role == tags[:, j])
        target = torch.where(valid, values[:, j], target)
    return torch.where(query_role == 3, public, target)


def _blind_mean(values, lengths, query_role, public, family):
    n, L = values.shape
    out = torch.zeros(n, dtype=values.dtype)
    if family == "train":
        for j in range(L):
            steps_after = (lengths-j-1).clamp_min(0)
            weight = (1/3)*torch.pow(
                torch.full((n,), 2/3, dtype=values.dtype), steps_after)
            out += torch.where(j < lengths, weight*values[:, j], torch.zeros_like(out))
    elif family == "heldout":
        for length, template in HELDOUT_TEMPLATES.items():
            locs = [max(i for i, t in enumerate(template) if t == k) for k in range(3)]
            estimate = (values[:, locs[0]]+values[:, locs[1]]+values[:, locs[2]])/3
            out = torch.where(lengths == length, estimate, out)
    else:
        raise ValueError("unknown family")
    return torch.where(query_role == 3, public, out)


def make_batch(seed: int, n: int, family: str) -> HistoryBatch:
    if n < 1 or family not in ("train", "heldout"):
        raise ValueError("family must be train/heldout and n>=1")
    g = _generator(seed)
    if family == "train":
        lengths = torch.randint(2, 5, (n,), generator=g)
        tags = torch.randint(0, 3, (n, MAX_WRITES), generator=g)
    else:
        lengths = torch.randint(5, 7, (n,), generator=g)
        permutations = torch.rand((n, SOURCES), generator=g).argsort(dim=-1)
        templates = torch.tensor([
            [*HELDOUT_TEMPLATES[5], 0], HELDOUT_TEMPLATES[6]
        ], dtype=torch.long)
        t = templates[lengths-5]
        tags = permutations.gather(1, t)
    values = torch.rand((n, MAX_WRITES), generator=g)*1.6-.8
    query_role = torch.randint(0, ROLES, (n,), generator=g)
    public = torch.rand(n, generator=g)*1.6-.8
    mask = torch.arange(MAX_WRITES)[None, :] < lengths[:, None]
    events = torch.zeros((n, MAX_WRITES, WIDTH))
    events[:, :, :SOURCES] = F.one_hot(tags, SOURCES).float()*mask[:, :, None]
    events[:, :, 3] = values*mask
    events[:, :, 4] = mask.float()
    query = torch.zeros((n, WIDTH))
    query[:, 5:9] = F.one_hot(query_role, ROLES).float()
    query[:, 9] = public
    query[:, 10] = 1.
    target = target_from_records(tags, values, lengths, query_role, public)
    blind = _blind_mean(values, lengths, query_role, public, family)
    return HistoryBatch(events, mask, query, target, tags, values,
                        query_role, public, lengths, blind, family)


def change_last_source(batch: HistoryBatch, seed: int):
    """Counterfactual donor flips only the last actual write's source tag."""
    g = _generator(seed)
    n = len(batch.target)
    row = torch.arange(n)
    col = batch.lengths-1
    old = batch.source_ids[row, col]
    offsets = 1+torch.randint(0, 2, (n,), generator=g)
    new = (old+offsets) % SOURCES
    tags = batch.source_ids.clone()
    tags[row, col] = new
    events = batch.events.clone()
    events[row, col, :SOURCES] = F.one_hot(new, SOURCES).float()
    target = target_from_records(tags, batch.values, batch.lengths,
                                 batch.query_role, batch.public_value)
    return events, target, tags
