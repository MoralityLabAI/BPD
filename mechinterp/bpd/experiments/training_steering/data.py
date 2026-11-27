"""Synthetic delayed-query memory with disjoint semantic tuple partitions.

Each tuple is encoded canonically in base ``values`` before any root-specific
input permutation. Evaluation uses residues {0, 1} modulo five; training uses
{2, 3, 4}. Receiver, donor and hybrid tuples must all belong to the requested
partition. Two evaluation residues are necessary: a single changed base-four
digit cannot preserve one residue modulo five.

All three members of a pair share write order, distractors, query and noise.
Only the values appearing in the initial write tokens change. The model sees
feature tensors; labels and semantic metadata remain with the host.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import math

import torch
from torch import Tensor


TRAIN_RESIDUES = (2, 3, 4)
EVAL_RESIDUES = (0, 1)


@dataclass
class PairBatch:
    receiver: Tensor
    donor: Tensor
    hybrid: Tensor
    labels_r: Tensor
    labels_d: Tensor
    labels_h: Tensor
    edit_slot: Tensor
    query: Tensor
    values_r: Tensor
    values_d: Tensor
    values_h: Tensor
    prefix_steps: int

    def to(self, device: torch.device | str) -> "PairBatch":
        return PairBatch(**{
            field.name: (getattr(self, field.name).to(device)
                         if isinstance(getattr(self, field.name), Tensor)
                         else getattr(self, field.name))
            for field in fields(self)
        })

    def fingerprint(self) -> str:
        """Bind every model input, host label and semantic pair identifier."""
        digest = hashlib.sha256()
        for field in fields(self):
            value = getattr(self, field.name)
            digest.update(field.name.encode("ascii") + b"\0")
            if isinstance(value, Tensor):
                tensor = value.detach().contiguous().cpu()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(str(value).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()


def canonical_codes(memory: Tensor, values: int = 4) -> Tensor:
    """Encode semantic values with slot zero as the least significant digit."""
    if memory.ndim < 1 or values < 2:
        raise ValueError("A nonempty memory axis and at least two values are required")
    if memory.shape[-1] < 1:
        raise ValueError("Memory must contain at least one slot")
    powers = values ** torch.arange(memory.shape[-1], device=memory.device,
                                    dtype=torch.long)
    return (memory.long() * powers).sum(dim=-1)


def partition_mask(memory: Tensor, split: str, values: int = 4) -> Tensor:
    if split not in ("train", "eval"):
        raise ValueError("split must be 'train' or 'eval'")
    residue = canonical_codes(memory, values).remainder(5)
    evaluation = residue <= 1
    return evaluation if split == "eval" else ~evaluation


def root_encoding(root_seed: int, slots: int = 4, values: int = 4) -> tuple[Tensor, Tensor]:
    """Semantic-index to input-channel permutations, fixed within each root."""
    generator = torch.Generator(device="cpu").manual_seed(
        (int(root_seed) * 104729 + 9029) % (2 ** 63 - 1))
    return (torch.randperm(slots, generator=generator),
            torch.randperm(values, generator=generator))


def make_pairs(seed: int, n: int, root_seed: int, delay: int,
               split: str = "train", slots: int = 4, values: int = 4,
               noise: float = .03) -> PairBatch:
    """Generate a balanced edit/query cohort by explicit rejection sampling.

    Edit slots differ in count by at most one. Exactly ``ceil(n/2)`` queries
    address the edited slot. Every remaining query addresses another slot.
    Receiver/donor tuples are independently proposed before conditioning on
    the split and the requirement that their edited values differ.
    """
    if (type(n) is not int or n < 1 or type(delay) is not int or delay < 0
            or type(slots) is not int or slots < 2
            or type(values) is not int or values < 2):
        raise ValueError("Require n >= 1, delay >= 0, slots >= 2 and values >= 2")
    if split not in ("train", "eval"):
        raise ValueError("split must be 'train' or 'eval'")
    if not math.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if values ** slots > 2 ** 62:
        raise ValueError("Canonical memory code would exceed the integer domain")

    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2 ** 63 - 1))
    edit_slot = torch.arange(n).remainder(slots)
    edit_slot = edit_slot[torch.randperm(n, generator=generator)]
    target_query = torch.arange(n) < (n + 1) // 2
    target_query = target_query[torch.randperm(n, generator=generator)]
    offset = torch.randint(1, slots, (n,), generator=generator)
    query = torch.where(target_query, edit_slot, (edit_slot + offset).remainder(slots))

    memories_r = torch.empty((n, slots), dtype=torch.long)
    memories_d = torch.empty_like(memories_r)
    pending = torch.arange(n)
    # Sixteen proposals per pending row reduce small late rejection batches.
    # Choosing the first admitted proposal retains ordinary rejection sampling.
    proposals_per_row = 16
    for _ in range(128):
        if pending.numel() == 0:
            break
        count = len(pending)
        candidate_r = torch.randint(values, (count, proposals_per_row, slots),
                                    generator=generator)
        candidate_d = torch.randint(values, (count, proposals_per_row, slots),
                                    generator=generator)
        edits = edit_slot[pending, None, None].expand(-1, proposals_per_row, 1)
        candidate_h = candidate_r.scatter(2, edits, candidate_d.gather(2, edits))
        different = (candidate_r.gather(2, edits) !=
                     candidate_d.gather(2, edits)).squeeze(-1)
        admitted = (different & partition_mask(candidate_r, split, values)
                    & partition_mask(candidate_d, split, values)
                    & partition_mask(candidate_h, split, values))
        found = admitted.any(dim=1)
        chosen = admitted.long().argmax(dim=1)
        row = torch.arange(count)[found]
        memories_r[pending[found]] = candidate_r[row, chosen[found]]
        memories_d[pending[found]] = candidate_d[row, chosen[found]]
        pending = pending[~found]
    if pending.numel():
        raise RuntimeError("Pair constraints yielded no admitted proposal within the bounded sampler")

    memories_h = memories_r.scatter(1, edit_slot[:, None],
                                    memories_d.gather(1, edit_slot[:, None]))
    address_encoding, value_encoding = root_encoding(root_seed, slots, values)
    # Continuous random scores generate a uniform permutation per row with
    # deterministic tie handling in the exceptional equal-score case.
    order = torch.rand((n, slots), generator=generator).argsort(dim=1, stable=True)
    length = slots + delay + 1
    input_dim = 3 + slots + values
    base = torch.zeros((n, length, input_dim), dtype=torch.float32)
    base[:, :slots, 0] = 1.0
    base[:, slots:slots + delay, 1] = 1.0
    base[:, -1, 2] = 1.0
    row = torch.arange(n)[:, None]
    write_steps = torch.arange(slots)[None, :]
    base[row, write_steps, 3 + address_encoding[order]] = 1.0
    base[torch.arange(n), -1, 3 + address_encoding[query]] = 1.0
    if delay:
        distractor_addresses = torch.randint(slots, (n, delay), generator=generator)
        distractor_values = torch.randint(values, (n, delay), generator=generator)
        distractor_steps = torch.arange(slots, slots + delay)[None, :]
        base[row, distractor_steps, 3 + address_encoding[distractor_addresses]] = 1.0
        base[row, distractor_steps,
             3 + slots + value_encoding[distractor_values]] = 1.0
    if noise:
        base[:, :, 3:] += noise * torch.randn((n, length, slots + values),
                                             generator=generator)

    def features(memory: Tensor) -> Tensor:
        result = base.clone()
        written_values = memory.gather(1, order)
        # Add into the common noisy background, preserving identical noise.
        result[row, write_steps, 3 + slots + value_encoding[written_values]] += 1.0
        return result

    return PairBatch(
        receiver=features(memories_r), donor=features(memories_d),
        hybrid=features(memories_h),
        labels_r=memories_r.gather(1, query[:, None]).squeeze(1),
        labels_d=memories_d.gather(1, query[:, None]).squeeze(1),
        labels_h=memories_h.gather(1, query[:, None]).squeeze(1),
        edit_slot=edit_slot, query=query, values_r=memories_r,
        values_d=memories_d, values_h=memories_h, prefix_steps=slots,
    )
