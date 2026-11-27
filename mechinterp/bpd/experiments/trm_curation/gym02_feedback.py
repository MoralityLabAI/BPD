"""GYM-02 TRAIN feedback panels and exact semantic sampling; GYM-01 remains frozen."""
from __future__ import annotations

from functools import lru_cache
from itertools import product

import torch

from .gym_campaign import gen_gym_task as legacy_sample


E27 = 'E27_copy_retrieve_2slot'
E28 = 'E28_dynamic_overwrite_register'
E29 = 'E29_dyck1_bounded_depth'
E31 = 'E31_2hop_pointer_chase'
G05 = 'G05_3slot_copy_retrieve'
TASKS = (E29, E31, E28, E27, G05)


@lru_cache(maxsize=6)
def support(task, split):
    if split not in ('train', 'eval'):
        raise ValueError('invalid split')
    width = {E27: 2, G05: 3, E31: 4}[task]
    ev = split == 'eval'
    def valid(m):
        pred = (sum(v * 4**i for i, v in enumerate(m)) % 5 <= 1
                if task == E31 else sum(m) % 3 == 0)
        return pred == ev
    return tuple(m for m in product(range(4), repeat=width) if valid(m))


def _generator(seed):
    return torch.Generator(device='cpu').manual_seed(int(seed) % (2**63 - 1))


def _encode(task, maps, q, split, g):
    n, width = maps.shape
    delay = {E27: (6, 14), G05: (5, 10), E31: (2, 2)}[task][split == 'eval']
    query = width if task == E31 else width + delay
    x = torch.zeros((n, width + delay + 1, 11))
    row = torch.arange(n)
    x[:, :width, 0] = 1.
    for i in range(width):
        x[:, i, 3+i] = 1.
        x[row, i, 7+maps[:, i]] = 1.
    x[:, query, 2] = 1.
    x[row, query, 3+q] = 1.
    if task == E31:
        x[:, query+1:, 1] = 1.
        y = maps[row, maps[row, q]]
    else:
        x[:, width:query, 1] = 1.
        for i in range(width, query):
            x[row, i, 3+torch.randint(width, (n,), generator=g)] = 1.
            x[row, i, 7+torch.randint(4, (n,), generator=g)] = 1.
        y = maps[row, q]
    return x, y, 4


def sample_task(task, seed, n, split):
    if task not in TASKS or split not in ('train', 'eval') or type(n) is not int or n < 1:
        raise ValueError('invalid sampling recipe')
    if task in (E28, E29):
        # Frozen generators have disjoint sequence lengths, no bounded map rejection.
        return legacy_sample(task, seed, n, split)
    g = _generator(seed)
    roster = torch.tensor(support(task, split), dtype=torch.long)
    maps = roster[torch.randint(len(roster), (n,), generator=g)]
    q = torch.randint(maps.shape[1], (n,), generator=g)
    return _encode(task, maps, q, split, g)


def exact_e31(seed=0):
    roster = torch.tensor(support(E31, 'train'), dtype=torch.long)
    return _encode(E31, roster.repeat_interleave(4, dim=0),
                   torch.arange(4).repeat(len(roster)), 'train', _generator(seed))


def feedback_panel(task, seed, n=128, *, mode='iid'):
    """TRAIN only. Map panels preserve uniform E31 map/query objective in expectation.

    For other tasks balanced means equal target-class weight, explicitly a
    different objective from natural IID. All proposal generation is charged.
    """
    if mode == 'iid':
        return sample_task(task, seed, n, 'train')
    if mode == 'population_or_large':
        return exact_e31(seed) if task == E31 else sample_task(task, seed, 4*n, 'train')
    if mode != 'balanced' or n < 4 or n % 4:
        raise ValueError('invalid panel mode/size')
    g = _generator(seed)
    if task == E31:
        roster = torch.tensor(support(E31, 'train'), dtype=torch.long)
        if n//4 > len(roster):
            raise ValueError('panel too large')
        chosen = roster[torch.randperm(len(roster), generator=g)[:n//4]]
        return _encode(task, chosen.repeat_interleave(4, dim=0),
                       torch.arange(4).repeat(n//4), 'train', g)
    # Bounded fail-closed class-balanced draw from a fresh TRAIN pool.
    x, y, nc = sample_task(task, seed, 8*n, 'train')
    ids = []
    for c in range(nc):
        pool = torch.nonzero(y == c, as_tuple=True)[0]
        if len(pool) < n//nc:
            raise ValueError('insufficient class support for balanced feedback')
        ids.append(pool[torch.randperm(len(pool), generator=g)[:n//nc]])
    pick = torch.cat(ids)
    pick = pick[torch.randperm(len(pick), generator=g)]
    if len(pick) != n:
        raise ValueError('invalid balanced roster')
    return x[pick], y[pick], nc
