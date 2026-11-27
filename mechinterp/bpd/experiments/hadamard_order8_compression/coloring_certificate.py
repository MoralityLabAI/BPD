"""Exact local checks for HAD-12. No learned model, probabilistic exclusion, or SAT claim."""
from collections import Counter, defaultdict
from itertools import product
from math import log2

SIGNS = tuple(product((-1, 1), repeat=4))

def compress(a):
    return a[0] + 4 * sum(a[1:])

def cost(s):
    return (s*s-9)//16

def color_feasible(majorities, flexible, margins):
    """Each flexible slot has one minority sign in an arbitrary one of 3 columns."""
    M = sum(majorities)
    P = sum(m == 1 and f for m, f in zip(majorities, flexible))
    N = sum(m == -1 and f for m, f in zip(majorities, flexible))
    if any((M-b) % 2 for b in margins):
        return False
    d = [(M-b)//2 for b in margins]
    return sum(d) == P-N and sum(max(x,0) for x in d) <= P

def norm_count():
    counts = Counter(cost(compress(a)) for a in SIGNS)
    assert counts == {0:6,1:6,7:2,10:2}
    dp = [1] + [0]*43
    for w,n in ((1,8),(2,88)):
        for _ in range(n):
            dp = [sum(m*dp[j-w*g] for g,m in counts.items() if j>=w*g) for j in range(44)]
    return dp[43]

def small_controls():
    tested = 0
    for n in range(1,4):
        actual = defaultdict(set)
        for rows in product(SIGNS, repeat=n):
            s = tuple(map(compress,rows))
            actual[s].add(tuple(sum(r[j] for r in rows) for j in range(4)))
        values = tuple(sorted({compress(a) for a in SIGNS}))
        for s in product(values,repeat=n):
            # The central signs are determined uniquely by s modulo 8.
            A = [1 if x%8==5 else -1 for x in s]
            D = [(x-a)//4 for x,a in zip(s,A)]
            M = [1 if x>0 else -1 for x in D]
            flexible = [abs(x)==1 for x in D]
            expected = {(sum(A),*b) for b in product(range(-n,n+1,2),repeat=3)
                        if color_feasible(M,flexible,b)}
            assert actual[s] == expected
            tested += 1
    return tested

if __name__ == '__main__':
    n = norm_count()
    print('exact norm count', n)
    print('log2',log2(n),'bits removed',384-log2(n))
    print('exhaustive small compressed strings',small_controls())
