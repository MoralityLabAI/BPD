"""Replay the exposed SCCG-07 roots and exact boundary fixtures."""
from collections import Counter
from fractions import Fraction as F
import argparse
import json
import random
import time

if __package__:
    from .exact_multiplier import decide, replay_witness, certificate_pencil
    from .robust_atlas import Certificate, fixture, psd
else:
    from exact_multiplier import decide, replay_witness, certificate_pencil
    from robust_atlas import Certificate, fixture, psd


def edges():
    p=Certificate([[1]],[[1]],[F(1,2)],3,1,1)
    assert p.find() is None
    r=decide(*certificate_pencil(p))
    assert r['witness']=={'kind':'rational','lambda':'5/2'}
    assert replay_witness(*certificate_pencil(p),r['witness'])
    A=[[0,1,0,0],[1,0,0,0],[0,0,2,0],[0,0,0,1]]
    B=[[1,0,0,0],[0,F(1,2),0,0],[0,0,0,1],[0,0,1,0]]
    a=(A,B,0,2); ir=decide(*a)
    assert ir['feasible'] and ir['witness']['kind']=='algebraic'
    assert replay_witness(*a,ir['witness']) and decide(*a,principal=True)['feasible']
    assert not decide(A,B,0,F(7,5))['feasible']
    assert not decide(A,B,F(3,2),2)['feasible']
    assert decide([[0,0],[0,1]],[[0,0],[0,0]],0,0)['feasible']
    assert not decide([[-1,0],[0,-2]],[[0,0],[0,0]],0,1)['feasible']
    assert decide([[0,0],[0,0]],[[0,0],[0,0]],0,1)['feasible']
    violations={61035:['-41679/65536','13893/32768','13893/32768','41679/65536'],
                61045:['58617/65536','-58617/65536']}
    for seed,w in violations.items():
        p=fixture(seed,False)
        assert p.within_budget(w) and p.violates(w)
    return {'grid_miss':r,'irrational_singleton':ir,'recovered_violations':violations}


def run():
    edge=edges(); rows=[]; totals={};start=time.perf_counter()
    for orthogonal in (True,False):
        counts=Counter()
        for seed in range(61000,61096):
            p=fixture(seed,orthogonal);a=certificate_pencil(p)
            trace=p.trace_safe();grid=p.find()
            r=decide(*a);safe=r['feasible']
            assert not (grid is not None or trace) or safe
            if safe:
                assert replay_witness(*a,r['witness'])
            if orthogonal:
                M=[[p.eta**2*(p.rho2+p.e**2)*p.G[i][j]-p.rho2*p.e**2*p.H2[i][j]
                    for j in range(p.n)] for i in range(p.n)]
                assert psd(M)==safe
            rng=random.Random(seed+300000);breach=False
            for _ in range(64):
                w=[F(rng.randint(-8,8),8) for _ in range(p.n)]
                while not p.within_budget(w):
                    w=[v/2 for v in w]
                breach |= p.violates(w)
            assert not (safe and breach)
            audit=not orthogonal and grid is None and not trace and not breach
            if audit:
                assert decide(*a,principal=True)['feasible']==safe
                counts['principal_decision_audits']+=1
            counts['safe']+=int(safe);counts['unsafe']+=int(not safe)
            counts['grid_safe']+=int(grid is not None);counts['trace_safe']+=int(trace)
            counts['new_safe']+=int(safe and grid is None)
            counts['sample_breach']+=int(breach)
            counts['prior_unresolved_now_unsafe']+=int(audit and not safe)
            counts['witness_replays']+=int(safe)
            counts['cells_tested']+=r['tested']
            rows.append(dict(seed=seed,orthogonal=orthogonal,trace_safe=trace,
                             grid_witness=None if grid is None else str(grid),
                             sample_breach=breach,decision=r,principal_audit=audit))
        totals['orthogonal' if orthogonal else 'nonorthogonal']=dict(counts)
    return dict(scope='EXPOSED_CONSTRUCTED_REPLAY',edges=edge,totals=totals,
                rows=rows,seconds=time.perf_counter()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()
    r=run()
    if a.output:
        with open(a.output,'w') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k!='rows'},sort_keys=True))
