"""Exact active sensing pilot: sparse fixed-weight directed linear couplings."""
import json
from itertools import combinations
import numpy as np


def run(n=6, sparsity=3, probes=64, budget=12, roots=32, seed=261126):
    edges = [(i,j) for i in range(n) for j in range(n) if i != j]
    supports = np.array(list(combinations(range(len(edges)), sparsity)))
    stats = {p: [] for p in ('random', 'entropy', 'static_entropy')}
    for root in range(roots):
        rng = np.random.default_rng(np.random.SeedSequence([seed, root]))
        # Readout c^T A x, signs generated independently of hidden support.
        # A_ij=1/(sparsity+1) on support; integer measurements use known scale.
        x = rng.choice([-1, 1], size=(probes,n))
        c = rng.choice([-1, 1], size=(probes,n))
        features = np.stack([c[:,i]*x[:,j] for i,j in edges], axis=1)
        obs = features[:,supports].sum(axis=2).T.astype(np.int8)
        hidden = int(rng.integers(len(supports)))
        # Score partitions of uniform remaining hypotheses in bits.
        def entropy(ids):
            counts = np.stack([(obs[ids] == v).sum(axis=0)
                               for v in range(-sparsity,sparsity+1,2)])
            p = counts / len(ids)
            return -(p*np.log2(np.maximum(p,1e-300))).sum(axis=0)
        static = entropy(np.arange(len(supports)))
        order = rng.permutation(probes)
        for policy in stats:
            ids = np.arange(len(supports))
            used = np.zeros(probes,dtype=bool)
            sizes = []
            for step in range(budget):
                if policy == 'random':
                    a = order[step]
                else:
                    score = (entropy(ids) if policy == 'entropy' else static).copy()
                    score[used] = -np.inf
                    a = int(np.argmax(score))
                used[a] = True
                ids = ids[obs[ids,a] == obs[hidden,a]]
                assert hidden in ids
                sizes.append(len(ids))
            stats[policy].append(sizes)
    return dict(n=n, sparsity=sparsity, candidates=len(supports), probes=probes,
                budget=budget, roots=roots, seed=seed,
                prediction_cache_bytes=obs.nbytes,
                arms={k: dict(unique_counts=(np.array(v)==1).sum(axis=0).tolist(),
                              mean_remaining=np.mean(v,axis=0).tolist(),
                              median_remaining=np.median(v,axis=0).tolist(),
                              root_remaining=v) for k,v in stats.items()})


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
