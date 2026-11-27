"""Exact arithmetic checks of the reduction and circuit theorem."""
from fractions import Fraction
from itertools import combinations
from random import Random
import unittest


def rank(rows, width):
    a = [[Fraction(v) for v in row] for row in rows]
    r = 0
    for j in range(width):
        p = next((i for i in range(r,len(a)) if a[i][j]),None)
        if p is None:
            continue
        a[r],a[p] = a[p],a[r]
        z = a[r][j]
        a[r] = [x/z for x in a[r]]
        for i in range(r+1,len(a)):
            z = a[i][j]
            if z:
                a[i] = [x-z*y for x,y in zip(a[i],a[r])]
        r += 1
    return r


def restrict(rows,S):
    return [[row[j] for j in S] for row in rows]


def positive(C,a,S):
    rows=restrict(C,S)
    return rank(rows+[[a[j] for j in S]],len(S))>rank(rows,len(S))


def min_support(C,a,k):
    for j in range(1,min(k,len(a))+1):
        for S in combinations(range(len(a)),j):
            if positive(C,a,S):
                return S
    return None


class ExactComplexityTests(unittest.TestCase):
    def test_x3c_reduction(self):
        universe=set(range(6))
        triples=list(combinations(range(6),3))
        count=yes=0
        for size in range(4):
            for family in combinations(triples,size):
                cover=any(set(x).isdisjoint(y) and set(x)|set(y)==universe
                          for x,y in combinations(family,2))
                C=[[int(i in t) for t in family]+[-1] for i in range(6)]
                a=[0]*size+[1]
                S=min_support(C,a,3)
                self.assertEqual(S is not None,cover)
                if S is not None:
                    self.assertEqual(len(S),3)
                    self.assertEqual(S[-1],size)
                    yes+=1
                count+=1
        self.assertEqual(count,1351)
        self.assertEqual(yes,190)

    def test_rank_bounded_circuits(self):
        rng=Random(1729)
        for i in range(100):
            n=1+i%6
            C=[[rng.randrange(-1,2) for _ in range(n)] for _ in range(i%4)]
            a=[rng.randrange(-1,2) for _ in range(n)]
            r=rank(C,n)
            S=min_support(C,a,n)
            T=min_support(C,a,r+1)
            self.assertEqual(S,T)
            if S is not None:
                self.assertLessEqual(len(S),r+1)
                self.assertEqual(rank(restrict(C,S),len(S)),len(S)-1)
                for j in S:
                    U=tuple(x for x in S if x!=j)
                    self.assertEqual(rank(restrict(C,U),len(U)),len(U))


if __name__=="__main__":
    unittest.main()
