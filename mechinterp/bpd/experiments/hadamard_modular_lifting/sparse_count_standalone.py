"""Standalone exact support-space count for the order-12 B4 group at 2396.
Counts labelled masks satisfying mod4 and M<=18, before exact mass.
Standard library only. No Hadamard existence or UNSAT claim.
"""
from collections import Counter
from math import comb, log2
import json

Q = 23
D = 12
FULL = (1 << D) - 1
ORBITS = [[0]] + [[i, Q-i] for i in range(1, D)]
BASIS = [{x for x in o} for o in ORBITS]


def multiply(a, b):
    x = [t for k, o in enumerate(ORBITS) if (a >> k) & 1 for t in o]
    y = [t for k, o in enumerate(ORBITS) if (b >> k) & 1 for t in o]
    c = 0
    for i in x:
        for j in y:
            c ^= 1 << ((i+j) % Q)
    assert all(((c >> i) & 1) == ((c >> ((-i) % Q)) & 1) for i in range(Q))
    return c & FULL


def rank(columns):
    pivots = {}
    for c in columns:
        while c:
            k = c.bit_length() - 1
            if k not in pivots:
                pivots[k] = c
                break
            c ^= pivots[k]
    return len(pivots)


def audit_ranks():
    counts = Counter()
    for x in range(1 << D):
        r = rank(multiply(x, 1 << i) for i in range(D))
        expected = 0 if x == 0 else 1 if x == FULL else 12 if x & 1 else 11
        assert r == expected
        counts[r] += 1
    return dict(counts)


def count_closed():
    total = 1 << 48  # ds=dw=0, hence dt=0
    for a in (0, 1):
        for k in range(5):
            if not (a or k):
                continue
            W = a + 2*k
            extra = (18-2*W)//8
            diffs = comb(11, k)*3**(a+k)
            targets = 2**(a+k)*sum(comb(11-k, j) for j in range(extra+1))
            total += diffs*targets*2**(36 if a else 37)
    return total


def count_direct():
    total = 1 << 48
    unions = 0
    for u in range(1, 1 << D):
        W = (u & 1) + 2*(u >> 1).bit_count()
        if W > 9:
            continue
        unions += 1
        allowed = 0
        for t in range(1 << D):
            if not u & 1 and t & 1:
                continue
            off = t & ~u
            M = 2*W + 4*((off & 1) + 2*(off >> 1).bit_count())
            allowed += M <= 18
        total += 3**u.bit_count()*allowed*2**(36 if u & 1 else 37)
    return total, unions


def main():
    ranks = audit_ranks()
    n = count_closed()
    n2, unions = count_direct()
    assert n == n2 == 266306251732811776
    print(json.dumps(dict(rank_counts=ranks, union_masks=unions, count=n,
                          log2_count=log2(n), scope="BEFORE_MASS"), indent=2))


if __name__ == "__main__":
    main()
