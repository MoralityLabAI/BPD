"""Sampled exact structural tests. Atlas is an opt-in secondary backend."""
from collections import Counter
from itertools import combinations
import argparse
import json
import random
import time

if __package__:
    from .dynamic_atlas import Atlas, N, K, EPOCHS, digest, scan, verify, world
else:
    from dynamic_atlas import Atlas, N, K, EPOCHS, digest, scan, verify, world


def fixture(seed, draws, drift):
    D, epochs = world(seed)
    universe = list(combinations(range(N), K))
    rng = random.Random(seed + 1000003)
    roster = [tuple(rng.choice(universe) for _ in range(draws))
              for _ in epochs]
    # Exact quotient invalidation challenge; identical roster for every mode.
    changed = [list(row) for row in D]
    changed[-1][rng.randrange(N)] += 1
    changed = tuple(map(tuple, changed))
    matrices = [(changed if drift and 5 <= t < 9 else D, E)
                for t, E in enumerate(epochs)]
    return roster, matrices


def run_root(roster, matrices, mode):
    start = time.perf_counter()
    # The sampled schedule is public before scoring. Both modes deduplicate
    # within each epoch. Atlas compiles only the union of sampled supports.
    unique = [tuple(sorted(set(row))) for row in roster]
    union = tuple(sorted(set().union(*map(set, unique))))
    counts = Counter()
    result = []
    atlas = None
    for S, (D, E) in zip(unique, matrices):
        if mode == 'rebuild':
            answer = scan(D + E, S, counts)
        else:
            if atlas is None or atlas.D != D:
                if atlas is not None:
                    counts.update(atlas.counts)
                    counts['quotient_rebuilds'] += 1
                atlas = Atlas(D, union)
                counts['atlas_compiles'] += 1
            live = atlas.update(D, E, sparse=True)
            answer = {s: live[s] for s in S if s in live}
        result.append(answer)
    if atlas is not None:
        counts.update(atlas.counts)
    return result, dict(counts), time.perf_counter() - start


def run(seed=51000, roots=8, draws=256, mode='rebuild', drift=False):
    if mode not in ('rebuild', 'atlas', 'paired'):
        raise ValueError('unknown backend')
    if not 1 <= roots <= 128 or not 1 <= draws <= 4096:
        raise ValueError('roots or draws outside engineering cap')
    modes = ('rebuild', 'atlas') if mode == 'paired' else (mode,)
    totals = {m: Counter() for m in modes}
    seconds = {m: 0. for m in modes}
    rows = []
    positives = {m: 0 for m in modes}
    for root in range(seed, seed + roots):
        roster, matrices = fixture(root, draws, drift)
        answers = {}
        for m in modes if root % 2 == 0 else modes[::-1]:
            a, c, elapsed = run_root(roster, matrices, m)
            answers[m] = a
            totals[m].update(c)
            seconds[m] += elapsed
        for t, (sample, (D, E)) in enumerate(zip(roster, matrices)):
            if mode == 'paired':
                verify(D, E, answers['rebuild'][t], answers['atlas'][t])
            row = dict(root=root, epoch=t, draws=len(sample),
                       unique=len(set(sample)), backends={})
            for m in modes:
                a = answers[m][t]
                hit = sum(s in a for s in sample)
                positives[m] += hit
                row['backends'][m] = dict(positive_draws=hit, digest=digest(a))
            rows.append(row)
    return dict(scope='MONTE_CARLO_CONSTRUCTED_STRUCTURAL_ONLY',
                seed=seed, roots=roots, epochs=EPOCHS, draws_per_epoch=draws,
                total_draws=roots*EPOCHS*draws,
                unique_support_epoch_checks=sum(r['unique'] for r in rows),
                mode=mode, quotient_drift=drift,
                paired_parity_checked=mode == 'paired',
                arms={m: dict(counts=dict(totals[m]), seconds=seconds[m],
                              positive_draws=positives[m]) for m in modes},
                rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('rebuild', 'atlas', 'paired'), default='rebuild')
    p.add_argument('--seed', type=int, default=51000)
    p.add_argument('--roots', type=int, default=8)
    p.add_argument('--draws', type=int, default=256)
    p.add_argument('--quotient-drift', action='store_true')
    p.add_argument('--output')
    a = p.parse_args()
    result = run(a.seed, a.roots, a.draws, a.mode, a.quotient_drift)
    if a.output:
        with open(a.output, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, sort_keys=True))


if __name__ == '__main__':
    main()
