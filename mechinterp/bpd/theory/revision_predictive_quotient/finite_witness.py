"""BPD-REV-01 exact tiny delayed observer witness.

A complete finite Moore-style signature for old match-bit histories when
a hidden future replacement set is revealed. No model inference, BPD factor
fit or physical intervention takes place.
"""
from __future__ import annotations
from fractions import Fraction as Q
from itertools import combinations
from math import comb

PI=(Q(9,20),Q(7,20),Q(1,5))
T=(Q(19,20),Q(13,20),Q(3,20))


def binom(n,r,p):
    if not 0<=r<=n:return Q(0)
    return comb(n,r)*p**r*(1-p)**(n-r)


def normalize(weights):
    total=sum(weights,Q(0))
    if total<=0:raise ValueError("impossible observation")
    return tuple(x/total for x in weights)


def old_posterior(m,n,j):
    """Initial Pr(K|old m) under an undisclosed uniform correction set."""
    if not(1<=j<n<=8 and 0<=m<=n):raise ValueError("invalid domain")
    return normalize(tuple(
        pi*sum((binom(n-j,u,t)*binom(j,m-u,Q(1,2))
                for u in range(n-j+1)),Q(0))
        for pi,t in zip(PI,T)))


def revised_posterior(mask,n,corrected):
    """Replace the selected reports with identical new bits all zero."""
    if not(1<=n<=8 and type(mask)==int and 0<=mask<(1<<n)):
        raise ValueError("unknown source history")
    kept=sum((mask>>i)&1 for i in range(n) if i not in corrected)
    return normalize(tuple(p*t**kept*(1-t)**(n-kept)
                           for p,t in zip(PI,T)))


def signature(mask,n,j):
    """Current posterior followed by all allowed post-revision readouts."""
    if not(1<=j<n<=8 and type(mask)==int and 0<=mask<(1<<n)):
        raise ValueError("invalid signature query")
    return (old_posterior(mask.bit_count(),n,j),
            tuple(revised_posterior(mask,n,s) for s in combinations(range(n),j)))


def witness(left,right,n,j):
    """Independent first distinguishing one-disclosure event."""
    if not(1<=j<n<=8 and 0<=left<(1<<n) and 0<=right<(1<<n)
           and left!=right and left.bit_count()==right.bit_count()):
        raise ValueError("requires distinct present-count-equivalent histories")
    for s in combinations(range(n),j):
        if revised_posterior(left,n,s)!=revised_posterior(right,n,s):
            return s
    raise AssertionError("source histories unexpectedly continuation-equivalent")


def audit(n=4,j=2):
    signatures={signature(s,n,j) for s in range(1<<n)}
    witnesses=0
    for a,b in combinations(range(1<<n),2):
        if a.bit_count()!=b.bit_count():continue
        event=witness(a,b,n,j)
        if revised_posterior(a,n,event)==revised_posterior(b,n,event):
            raise AssertionError("witness defective")
        witnesses+=1
    return {"old_histories":1<<n,"present_labels":n+1,
            "future_classes":len(signatures),
            "checked_present_collision_pairs":witnesses,
            "exact_fraction":True,"neural_mechanism_tested":False}
