"""Independent exact bound and certificate audit, plus transfer/cache sanity checks."""
import argparse, json, collections, itertools, math
from pathlib import Path
import numpy as np
from had05 import State, Trajectory, direct, certify, ldt, power3, spectral_lower, sectors, transfer, preflight, payload_per, evaluate, make_bank, F, HISTORY

def audit_bounds():
    m=3; n=12; xs=np.array([[1 if k>>i&1 else -1 for i in range(n)] for k in range(1<<n)],dtype=np.int64).reshape(-1,4,m)
    # Independent scalar autocorrelation; multi-source BFS exact Hamming distance.
    valid=[k for k,x in enumerate(xs) if sum(int(s[i]*s[(i+1)%m]) for s in x for i in range(m))==0]
    dist=np.full(1<<n,100,dtype=int); dist[valid]=0; q=collections.deque(valid)
    while q:
        k=q.popleft()
        for i in range(n):
            j=k^(1<<i)
            if dist[j]>dist[k]+1: dist[j]=dist[k]+1; q.append(j)
    signed=[s for s in itertools.product(range(-m,m+1,2),repeat=4) if sum(v*v for v in s)==4*m]
    # Exact row-sum and spectral lower-bound checks.
    count=0
    for k,x in enumerate(xs):
        u=x.sum(1); r=direct(x); b=ldt(x,r,u,1)
        brute=min(sum(abs(int(a)-v)//2 for a,v in zip(u,s)) for s in signed)
        assert brute==b['row']; assert b['lower']<=dist[k]; count+=2
        if b['lower']>1: assert dist[k]>1; count+=1
    # At m=9, exhaustive individual-row Hamming distance to order-3 power feasibility.
    m=9; arr=np.array([[1 if k>>i&1 else -1 for i in range(m)] for k in range(1<<m)],dtype=np.int64)
    power=lambda x: (sum(x[::3])**2+sum(x[1::3])**2+sum(x[2::3])**2-sum(x[::3])*sum(x[1::3])-sum(x[1::3])*sum(x[2::3])-sum(x[::3])*sum(x[2::3]))
    feasible=[k for k,x in enumerate(arr) if power(x)<=4*m]
    for k,x in enumerate(arr):
        dist0=min((k^z).bit_count() for z in feasible); q0=power(x); assert q0==int(power3(np.tile(x,(4,1)))[0]); assert spectral_lower(np.array([q0]),m)[0]<=dist0; count+=2
    rng=np.random.default_rng(99101); tested=0
    for m in (9,19,27):
        for j in range(8):
            ts=[Trajectory(rng.choice([-1,1],size=(4,m)),rng.integers(2**60),16) for _ in range(5)]
            for t in ts:
                for _ in range(3): t.advance(); assert np.array_equal(t.state.r,direct(t.state.matrix())); assert np.array_equal(t.state.d,State(t.state.matrix()).d); count+=2
            for i,t in enumerate(ts):
                child,work,offers=transfer(ts,i,4*m*100)
                assert work==offers*4*m; count+=1
                if child is not None:
                    x,_,_,_,e=child; assert e<t.state.energy; assert np.array_equal(x.sum(1),t.state.s)
                    a=power3(x); b=power3(t.state.matrix()); assert a is None or np.array_equal(a,b); tested+=1; count+=3
    # Positive control: construct a donor with exact solution and corrupt one row by
    # swapping opposite signs within a residue class, keeping the guarded commitments.
    rng=np.random.default_rng(10101); fixture=None
    for _ in range(50):
        t=Trajectory(rng.choice([-1,1],size=(4,9)),rng.integers(2**60),16)
        for _ in range(12):
            t.advance()
            if t.state.energy==0: break
        if t.state.energy: continue
        z=t.state.matrix().copy()
        for a in range(4):
            for i,j in itertools.combinations(range(9),2):
                if (i-j)%3 or z[a,i]==z[a,j]: continue
                y=z.copy(); y[a,[i,j]]*=-1
                if np.any(direct(y)): fixture=(y,z,a); break
            if fixture: break
        if fixture: break
    assert fixture is not None
    y,z,a=fixture; recipient=Trajectory(y,1,16); donor=Trajectory(z,2,16)
    child,work,offers=transfer([recipient,donor],0,4*9*16)
    assert child is not None and child[-1]==0; assert certify(child[0]); assert work==offers*36
    assert np.array_equal(y.sum(1),z.sum(1)) and np.array_equal(power3(y),power3(z)); count+=4
    try: preflight(599,1000,1); raise AssertionError('budget should reject')
    except ValueError: count+=1
    try: make_bank(1,[19],8,16,100,100,1); raise AssertionError('bank should reject')
    except ValueError: count+=1
    return dict(bound_and_cache_checks=count,exhaustive_m3_states=len(xs),m3_exact_solutions=len(valid),exhaustive_m9_rows=len(arr),improving_guarded_transfer_tests=tested,positive_control_transfer_solved=True)

def bitgram(x):
    m=len(x[0]); A,B,C,D=x; rows=[]
    specs=[[(A,1,False,False),(B,1,False,True),(C,1,False,True),(D,1,False,True)],[(B,-1,False,True),(A,1,False,False),(D,-1,True,True),(C,1,True,True)],[(C,-1,False,True),(D,1,True,True),(A,1,False,False),(B,-1,True,True)],[(D,-1,False,True),(C,-1,True,True),(B,1,True,True),(A,1,False,False)]]
    for br in range(4):
        for i in range(m):
            row=0
            for s,sgn,trans,rev in specs[br]:
                for j in range(m):
                    k=m-1-j if rev else j; idx=(i-k if trans else k-i)%m; row=(row<<1)|(sgn*s[idx]==1)
            rows.append(row)
    checks=0
    for i,u in enumerate(rows):
        for v in rows[i+1:]: assert (u^v).bit_count()==2*m; checks+=1
    return checks

def receipt(path):
    result=json.loads(Path(path).read_text()); a=result['args']; seen={}; certs=checks=0
    for r in result['rows']:
        key=(r['root'],r['m'],r['case']); prev=seen.setdefault(key,r['start_hash']); assert r['start_hash']==prev; assert r['candidate_scores']<=a['rounds']*a['chunk']*4*r['m']; assert r['tracked_population_bytes']==a['population']*payload_per(r['m']); assert r['solved']==(r['certificate'] is not None); checks+=4
        if r['solved']:
            x=r['certificate']; m=r['m']; assert len(x)==4 and all(len(s)==m and set(s)<={-1,1} for s in x)
            for t in range(1,m): assert sum(s[i]*s[(i+t)%m] for s in x for i in range(m))==0; checks+=1
            checks+=bitgram(x); certs+=1
    return dict(receipt_checks=checks,verified_certificates=certs,paired_populations=len(seen))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--results'); p.add_argument('--out',default='audit.json'); a=p.parse_args(); r=audit_bounds();
    if a.results: r.update(receipt(a.results))
    Path(a.out).write_text(json.dumps(r,indent=2)); print(r)
