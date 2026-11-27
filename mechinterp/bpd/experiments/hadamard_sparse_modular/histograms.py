"""Exact minimum-energy compressed histogram table, independently checked against HAD12 counts."""
from itertools import product
from pathlib import Path
from math import factorial
from collections import Counter, defaultdict
import json
P=Path(__file__).parent; prior=P.parent/'HAD12'
T=tuple(sorted(a+4*m*(1 if f else 3) for a,m,f in product((-1,1),(-1,1),(0,1))))
A=[1 if c%8==5 else -1 for c in T]; D=[(c-a)//4 for c,a in zip(T,A)]; M=[1 if d>0 else -1 for d in D]; F=[abs(d)==1 for d in D]; E=[(c*c-9)//16 for c in T]
def compositions(n,k):
    if k==1: yield (n,);return
    for i in range(n+1):
        for z in compositions(n-i,k-1):yield (i,*z)
def key(v):
    k=0
    for x in v:assert -11<=x<=11 and x%2;k=k*12+(x+11)//2
    return k

def main():
    expected=[list(map(int,l.split())) for l in open(prior/'compressed_counts.txt')]; assert len(expected)==12**4
    rows=defaultdict(Counter)
    for n in compositions(11,8):
        e=2*sum(x*y for x,y in zip(n,E))
        if e>43:continue
        a=sum(x*y for x,y in zip(n,A));m=sum(x*y for x,y in zip(n,M));p=sum(x for x,y,f in zip(n,M,F) if y==1 and f);q=sum(x for x,y,f in zip(n,M,F) if y==-1 and f)
        ways=factorial(11)//__import__('math').prod(factorial(x) for x in n)
        for x in range(-q,p+1):
            for y in range(-q,p+1):
                z=p-q-x-y
                if -q<=z<=p and sum(max(0,t) for t in (x,y,z))<=p:
                    rows[key((a,m-2*x,m-2*y,m-2*z))][(e,n)]+=ways
    hist=[];hs={};out=[];min_unique=0;checks=0
    for k,r in enumerate(expected):
        got=Counter()
        for (e,n),v in rows[k].items():got[e]+=v
        assert [got[e] for e in range(44)]==r;checks+=44
        if not got:out.append((-1,-1));continue
        e=min(got);bank=[n for ee,n in rows[k] if ee==e];assert len(bank)==1;n=bank[0];min_unique+=1
        if n not in hs:hs[n]=len(hist);hist.append(n)
        out.append((e,hs[n]))
    assert min_unique==16426 and len(hist)==186
    with (P/'min_hist.txt').open('w') as f:
        print(len(hist),file=f);print(*T,file=f)
        for h in hist:print(*h,file=f)
        for r in out:print(*r,file=f)
    print(json.dumps(dict(margin_cost_cells=checks,unique_minimum_cells=min_unique,distinct_min_histograms=len(hist)),indent=2))
if __name__=='__main__':main()
