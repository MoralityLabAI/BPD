"""Train-only edge partitions and bounded candidate scheduling."""
from collections import defaultdict, Counter
from dataclasses import dataclass
import hashlib
import math

from .reference import semantic_edges


def _digest(seed, domain, key):
    return hashlib.sha256(repr((seed, domain, key)).encode()).digest()


def pair_key(edge):
    r, h, slot = edge
    return (min(r, h), max(r, h), slot)


def edge_views(seed):
    """Split both orientations together, stratifying unordered value transitions.

    Vertices may recur across views; only the edge and reverse are disjoint.
    No historical evaluation edges are loaded by this function.
    """
    groups = defaultdict(list)
    for edge in semantic_edges('train'):
        r, h, slot = edge
        if r < h:
            groups[(slot, min(r[slot], h[slot]), max(r[slot], h[slot]))].append(edge)
    out = {k: [] for k in ('fit', 'feedback', 'tune')}
    for stratum, edges in sorted(groups.items()):
        ordered = sorted(edges, key=lambda e: (_digest(seed, 'split', e), e))
        n = len(ordered)
        hold = max(1, n // 6)
        if n < 2 * hold + 1:
            raise ValueError('stratum too small')
        for name, part in [('feedback', ordered[:hold]), ('tune', ordered[hold:2*hold]),
                           ('fit', ordered[2*hold:])]:
            for r, h, slot in part:
                out[name].extend([(r, h, slot), (h, r, slot)])
    return {k: tuple(sorted(v)) for k, v in out.items()}


@dataclass(frozen=True)
class Candidate:
    edge: tuple
    loss: float
    predicted_gain: float
    cpu_seconds: float
    epoch: str


def select_candidates(rows, *, fit_edges, epoch, mode, count, seed, window,
                      exposures=None, cap=4, exploration=.2):
    """Deterministic bounded ranking; charge proposal work outside this routine.

    `predicted_gain` is a signed train-feedback estimate, not observed held-out
    gain. Stale candidates are rejected rather than silently relabelled.
    """
    if (mode not in ('uniform', 'loss', 'mechanistic', 'shuffled') or
            type(count) is not int or count < 0 or type(cap) is not int or cap < 1 or
            not math.isfinite(exploration) or not 0 <= exploration <= 1 or not epoch):
        raise ValueError('invalid scheduling recipe')
    allowed = set(fit_edges)
    train = set(semantic_edges('train'))
    if not allowed <= train:
        raise ValueError('fit pool outside training partition')
    used = Counter(exposures or {})
    if any(type(n) is not int or n < 0 for n in used.values()):
        raise ValueError('invalid exposure ledger')
    by_edge = {}
    for row in rows:
        if (row.edge not in allowed or row.edge in by_edge or row.epoch != epoch or
                not all(math.isfinite(v) for v in (row.loss, row.predicted_gain, row.cpu_seconds)) or
                row.cpu_seconds <= 0):
            raise ValueError('invalid, duplicate, stale or non-fit candidate')
        by_edge[row.edge] = row
    available = [r for r in by_edge.values() if used[pair_key(r.edge)] < cap]
    canonical = sorted(available, key=lambda r: r.edge)
    random_order = sorted(canonical, key=lambda r: (_digest(seed, ('explore', window), r.edge), r.edge))
    scores = {r.edge: r.predicted_gain / r.cpu_seconds for r in canonical}
    if mode == 'shuffled':
        vals = [scores[r.edge] for r in sorted(canonical, key=lambda r: (_digest(seed, ('sham', window), r.edge), r.edge))]
        scores = dict(zip((r.edge for r in canonical), vals))
    if mode == 'loss':
        scores = {r.edge: r.loss for r in canonical}
    ranked = sorted(canonical, key=lambda r: (-scores[r.edge], r.edge))
    target = min(count, len(available))
    n_explore = target if mode == 'uniform' else math.ceil(exploration * target)
    selected, seen = [], set()

    def take(order, limit, reason, positive=False):
        for row in order:
            if len(selected) >= limit:
                break
            key = pair_key(row.edge)
            if row.edge in seen or used[key] >= cap:
                continue
            if positive and scores[row.edge] <= 0:
                continue
            selected.append((row.edge, reason))
            seen.add(row.edge)
            used[key] += 1

    take(random_order, n_explore, 'explore')
    if mode != 'uniform':
        take(ranked, target, 'ranked', positive=mode in ('mechanistic', 'shuffled'))
    take(random_order, target, 'ordinary_fallback')
    return tuple(selected), dict(used)
