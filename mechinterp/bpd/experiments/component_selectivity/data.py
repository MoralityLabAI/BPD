"""Fresh RP04 input draws, querying every slot of each semantic pair."""
from __future__ import annotations
from dataclasses import fields
import hashlib
import torch
from ..training_steering.data import PairBatch, make_pairs, root_encoding
from .common import Recipe


def seed_for(root, phase):
    if phase not in ('smoke', 'development'):
        raise ValueError('Unknown phase')
    # Retain the archived smoke seed; amended development must use a fresh domain.
    domain = 'slot-selectivity-v2-audit-arithmetic' if phase == 'development' else 'slot-selectivity-v1'
    raw = f'BPD-RP-04:{root}:{phase}:{domain}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little') % (2**63-1)


def expand_queries(base, root, slots=4, values=4):
    n = len(base.query)
    address, _ = root_encoding(root, slots, values)
    query = torch.arange(slots).repeat(n)
    old_query = base.query.repeat_interleave(slots)
    rows = torch.arange(n*slots)
    kwargs = {}
    for field in fields(base):
        value = getattr(base, field.name)
        kwargs[field.name] = value.repeat_interleave(slots, dim=0) if isinstance(value, torch.Tensor) else value
    for name in ('receiver', 'donor', 'hybrid'):
        x = kwargs[name].clone()
        x[rows, -1, 3+address[old_query]] -= 1.
        x[rows, -1, 3+address[query]] += 1.
        kwargs[name] = x
    kwargs['query'] = query
    for suffix in ('r', 'd', 'h'):
        kwargs['labels_'+suffix] = kwargs['values_'+suffix].gather(1, query[:, None]).squeeze(1)
    return PairBatch(**kwargs)


def make_batch(root, phase, recipe=None):
    recipe = recipe or Recipe()
    n = recipe.smoke_base_pairs if phase == 'smoke' else recipe.base_pairs
    base = make_pairs(seed_for(root, phase), n, root_seed=root, delay=recipe.delay,
                      split='eval', slots=recipe.slots, values=recipe.values, noise=recipe.noise)
    expanded = expand_queries(base, root, recipe.slots, recipe.values)
    address, value = root_encoding(root, recipe.slots, recipe.values)
    extra = (base.values_r != base.values_d).sum(-1)-1
    metadata = {
        'schema_version': 1, 'root': root, 'phase': phase, 'seed': seed_for(root, phase),
        'n_base_pairs': n, 'n_query_cases': n*recipe.slots,
        'semantic_partition': 'eval residues 0 and 1 modulo 5; same finite tuple domain as RP03',
        'sampling_scope': 'new purpose-specific input draw on exposed model roots',
        'base_batch_sha256': base.fingerprint(), 'batch_sha256': expanded.fingerprint(),
        'query_order': list(range(recipe.slots)), 'edit_slot_by_pair': base.edit_slot.tolist(),
        'values_r': base.values_r.tolist(), 'values_d': base.values_d.tolist(),
        'values_h': base.values_h.tolist(), 'extra_donor_slot_changes': extra.tolist(),
        'address_encoding': address.tolist(), 'value_encoding': value.tolist(),
        'labels_r': expanded.labels_r.tolist(), 'labels_h': expanded.labels_h.tolist(),
        'targeted': expanded.query.eq(expanded.edit_slot).tolist(),
        'edited_slot': expanded.edit_slot.tolist(), 'query': expanded.query.tolist(),
        'prefix_steps': expanded.prefix_steps,
    }
    return base, expanded, metadata
