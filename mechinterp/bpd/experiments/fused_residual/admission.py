"""FUSE-07: deterministic static admission for a fixed-weight residual cache."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key


@dataclass(frozen=True)
class Site:
    size: int
    value: int

    @classmethod
    def residual(cls, din: int, dout: int, rank: int, calls: int, elem_bytes: int = 4) -> Site:
        if (any(type(x) is not int or x <= 0 for x in (din, dout, rank, elem_bytes))
                or type(calls) is not int or calls < 0):
            raise ValueError('invalid residual shape or calls')
        return cls(elem_bytes*din*dout, max(0,calls-1)*din*rank*dout)


@dataclass(frozen=True)
class Plan:
    indices: tuple[int, ...]
    size: int
    value: int
    transitions: int
    peak_frontier: int


def select(sites, cap, mode='exact', *, max_frontier=65536):
    """Optimize saved reconstruction MACs, not elapsed time.

    Exact tie order: max value, min bytes, lexicographically smallest indices.
    Bound retained frontier states; transient candidate map has <=2*limit.
    Raise on frontier overflow without issuing an approximate plan.
    """
    sites=tuple(sites)
    if type(cap) is not int or cap<0 or type(max_frontier) is not int or max_frontier<1:
        raise ValueError('invalid cap or frontier limit')
    if any(type(s.size) is not int or s.size<=0 or type(s.value) is not int or s.value<0 for s in sites):
        raise ValueError('invalid site')
    ids=[i for i,s in enumerate(sites) if s.value>0 and s.size<=cap]
    if mode!='exact':
        if mode=='smallest':ids.sort(key=lambda i:(sites[i].size,i))
        elif mode=='density':
            def cmp(i,j):
                a,b=sites[i],sites[j]
                z=a.value*b.size-b.value*a.size
                return -1 if z>0 else 1 if z<0 else (i>j)-(i<j)
            ids.sort(key=cmp_to_key(cmp))
        elif mode!='first':raise ValueError('unknown policy')
        size=value=0; chosen=[]
        for i in ids:
            s=sites[i]
            if size+s.size<=cap:
                size+=s.size;value+=s.value;chosen.append(i)
        return Plan(tuple(sorted(chosen)),size,value,len(ids),1)
    frontier={0:(0,())}; transitions=0;peak=1
    for i in ids:
        s=sites[i]; candidates=dict(frontier)
        for size,(v,chosen) in frontier.items():
            transitions+=1; w=size+s.size
            if w>cap:continue
            proposal=(v+s.value,chosen+(i,));prev=candidates.get(w)
            if prev is None or proposal[0]>prev[0] or (proposal[0]==prev[0] and proposal[1]<prev[1]):
                candidates[w]=proposal
        # A smaller weight with >= value dominates all future extensions.
        # Positive sizes ensure equal-weight witnesses cannot be proper prefixes.
        frontier={}; best=-1
        for w,(v,chosen) in sorted(candidates.items()):
            if v>best:
                frontier[w]=(v,chosen);best=v
                if len(frontier)>max_frontier:raise RuntimeError('frontier limit exceeded')
        peak=max(peak,len(frontier))
    size,(value,chosen)=max(frontier.items(),key=lambda x:x[1][0])
    return Plan(chosen,size,value,transitions,peak)
