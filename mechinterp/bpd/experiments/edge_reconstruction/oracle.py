"""Exact small graph edge-deck oracle. Run with Python 3.10+."""
from collections import Counter
from itertools import combinations, permutations
from functools import lru_cache

@lru_cache(None)
def pairs(n):
    return tuple(combinations(range(n), 2))

@lru_cache(None)
def canonical(n, edges):
    return min(tuple(sorted(tuple(sorted((p[a], p[b]))) for a,b in edges))
               for p in permutations(range(n)))

def deck(n, edges):
    return Counter(canonical(n, edges[:i]+edges[i+1:]) for i in range(len(edges)))

def candidates(n, m, observations):
    if not 0 <= m <= len(pairs(n)) or n > 6:
        raise ValueError("oracle restricted to n <= 6 and valid edge count")
    seen = set()
    for raw in combinations(pairs(n), m):
        g = canonical(n, raw)
        if g in seen:
            continue
        seen.add(g)
        if not observations - deck(n, g):
            yield g

def self_test():
    triangle = ((0,1),(0,2),(1,2))
    star = ((0,1),(0,2),(0,3))
    assert deck(4, triangle) == deck(4, star)
    assert len(list(candidates(4,3,deck(4,triangle)))) == 2
    path = ((0,1),(1,2),(2,3))
    assert list(candidates(4,3,deck(4,path))) == [canonical(4,path)]
    assert not list(candidates(4,3,Counter({canonical(4,path):1})))
    print("four oracle checks passed")

if __name__ == "__main__":
    self_test()
