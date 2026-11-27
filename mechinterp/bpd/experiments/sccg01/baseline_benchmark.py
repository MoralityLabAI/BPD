"""SCCG-09 constructed development benchmark, frozen before execution."""
from fractions import Fraction as F
from collections import Counter
import argparse
import json
import random
import time

if __package__:
    from .baseline_atlas import Problem,dot,principal_psd
else:
    from baseline_atlas import Problem,dot,principal_psd


def fixture(seed,n):
    rng=random.Random(seed)
    W=[[F(2*int(i==j)+(rng.choice((-1,1)) if i==j+1 else 0))
        for j in range(n)] for i in range(n)]
    cols=list(zip(*W))
    G=[[dot(u,v) for v in cols] for u in cols]
    x=[F(rng.randint(-2,2)) for _ in range(n)]
    h=[dot(u,x) for u in cols]
    rho2=dot(x,x)+5
    c=[F(rng.randint(-3,3)) for _ in range(1+seed%3)]
    if not any(c): c[0]=F(1)
    # Half the roots have small compensated derivatives, half larger.
    scale=F(1,16) if seed%2 else F(1,2)
    H=[[ci*hj/rho2+scale*rng.randint(-3,3) for hj in h] for ci in c]
    return Problem(G,H,h,c,rho2,F(1+seed%4,2),F(rng.randint(1,16),16))


def edges():
    p=Problem([[1]],[[0]],[0],[1],1,1,F(1,4))
    assert p.violates([1]) and not p.exact_line()['safe']
    p=Problem([[1]],[[1]],[1],[2],2,F(1,4),F(1,16))
    assert not p.separate() and p.compensated() and p.exact_line()['safe']
    p=Problem([[1]],[[1]],[0],[0],1,1,F(3,4))
    assert p.exact_line()['safe']
    p=Problem([[1]],[[0]],[0],[0],1,1,0)
    assert p.exact_line()['safe'] and p.envelope_grid() is not None
    p=Problem([[1]],[[1]],[0],[0],1,1,0)
    assert not p.exact_line()['safe']
    # Crossing the boundary s2=4; H=0,c=2, eta=1.
    p=Problem([[3]],[[0]],[0],[2],1,F(7,4),1)
    assert not p.exact_line()['safe'] # dose>sqrt(3) allows strictly larger |w|
    p=Problem([[3]],[[0]],[0],[2],1,F(3,2),1)
    assert p.exact_line()['safe']
    # Norm change can exactly meet the budget at the physical boundary.
    p=Problem([[1]],[[0]],[0],[5],1,F(3,4),1)
    assert p.exact_line()['safe'] and not p.violates([F(3,4)])
    assert Problem([[1]],[[0]],[0],[5],1,F(3,4),F(99,100)).violates([F(3,4)])
    # Nonpositive affine lower endpoint: conservative envelope refuses.
    p=Problem([[1]],[[0]],[2],[1],5,3,2)
    assert p.lower<=0 and p.envelope_grid() is None and p.exact_line()['safe']
    return dict(zero_direct_write_breach=True,cancellation_safe=True,boundary_cases=True)


def run():
    edge=edges(); rows=[]; totals={}; t0=time.perf_counter()
    for name,start,count in [('line',71000,96),('multi',72000,96)]:
        counts=Counter()
        for seed in range(start,start+count):
            n=1 if name=='line' else 2+seed%3
            p=fixture(seed,n)
            sep=p.separate();comp=p.compensated();wit=p.envelope_grid()
            safe=sep or comp or wit is not None
            if wit is not None:
                assert all(principal_psd(p.matrix(b,lam)) for b,lam in zip((F(0),p.b),wit))
            oracle=p.exact_line() if n==1 else None
            if oracle is not None:
                assert not safe or oracle['safe']
                if not oracle['safe']:
                    assert p.violates([oracle['witness']])
                counts['exact_safe']+=int(oracle['safe'])
                counts['exact_unsafe']+=int(not oracle['safe'])
                counts['safe_missed']+=int(oracle['safe'] and not safe)
            rng=random.Random(seed+310001);breach=False
            for _ in range(64):
                w=[F(rng.randint(-8,8),8) for _ in range(n)]
                while dot(w,[dot(row,w) for row in p.G])>p.e**2:
                    w=[v/2 for v in w]
                breach|=p.violates(w)
            assert not safe or not breach
            counts['separate']+=int(sep);counts['compensated']+=int(comp)
            counts['psd_grid']+=int(wit is not None);counts['combined']+=int(safe)
            counts['extra_over_separate']+=int(safe and not sep)
            counts['extra_psd_over_trace']+=int(wit is not None and not (sep or comp))
            counts['sample_breach']+=int(breach)
            counts['unresolved']+=int(not safe and not breach and oracle is None)
            counts['sampled_edits']+=64
            rows.append(dict(seed=seed,dimension=n,separate=sep,compensated=comp,
                             multipliers=None if wit is None else list(map(str,wit)),
                             safe=safe,oracle=oracle,sample_breach=breach))
        totals[name]=dict(counts)
    return dict(scope='CONSTRUCTED_DEVELOPMENT_ONLY',edges=edge,totals=totals,
                rows=rows,seconds=time.perf_counter()-t0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()
    r=run()
    if a.output:
        with open(a.output,'w') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k!='rows'},sort_keys=True))
