"""RP03 domains over RP02's unchanged, disjoint semantic tuple partitions."""
from __future__ import annotations

import hashlib
import torch

from ..training_steering.data import PairBatch, make_pairs, root_encoding


def seed_for(root: int, domain: str, index: int = 0) -> int:
    raw = f"BPD-RP-03:{root}:{domain}:{index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def make_batch(root: int, domain: str, index: int, n: int, delay: int,
               split: str, recipe) -> PairBatch:
    return make_pairs(seed_for(root, domain, index), n, root_seed=root,
                      delay=delay, split=split, slots=recipe.slots,
                      values=recipe.values, noise=recipe.noise)


def binding_batch(root: int, domain: str, n: int, delay: int, split: str,
                  recipe) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """All-distinct memories queried at every address, within the given split."""
    if recipe.slots != recipe.values or n < 4 or n % recipe.slots:
        raise ValueError("Binding cohorts require equal slots/values and whole query groups")
    wanted = n // recipe.slots
    features, memories, old_queries, hashes = [], [], [], []
    accepted = 0
    for attempt in range(16):
        pairs = make_batch(root, domain + '_candidates', attempt, 2048,
                           delay, split, recipe)
        hashes.append(pairs.fingerprint())
        distinct = pairs.values_r.sort(dim=1).values.eq(torch.arange(recipe.values)).all(1)
        chosen = torch.where(distinct)[0][:wanted - accepted]
        if len(chosen):
            features.append(pairs.receiver[chosen])
            memories.append(pairs.values_r[chosen])
            old_queries.append(pairs.query[chosen])
            accepted += len(chosen)
        if accepted == wanted:
            break
    if accepted != wanted:
        raise RuntimeError("Binding cohort exhausted the bounded admission cap")
    x = torch.cat(features).repeat_interleave(recipe.slots, dim=0)
    memory = torch.cat(memories).repeat_interleave(recipe.slots, dim=0)
    old = torch.cat(old_queries).repeat_interleave(recipe.slots)
    query = torch.arange(recipe.slots).repeat(wanted)
    address, _ = root_encoding(root, recipe.slots, recipe.values)
    row = torch.arange(n)
    x[row, -1, 3 + address[old]] -= 1
    x[row, -1, 3 + address[query]] += 1
    labels = memory.gather(1, query[:, None]).squeeze(1)
    digest = hashlib.sha256(x.contiguous().numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
    return x, labels, {"batch_sha256": digest, "candidate_batch_hashes": hashes,
                       "split": split, "n_base_sequences": wanted,
                       "queries": query.tolist(), "memories": memory.tolist(),
                       "modal_baseline_accuracy": 1 / recipe.values}
