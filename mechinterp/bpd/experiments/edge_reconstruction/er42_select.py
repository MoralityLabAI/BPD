"""Exact top-K by score descending, last access descending, key ascending."""
import itertools,json,time
import numpy as np
from numba import njit

@njit(inline='always')
def better(a,b,v,t,k):
    if v[a]!=v[b]: return v[a]>v[b]
    if t[a]!=t[b]: return t[a]>t[b]
    return k[a]<k[b]

@njit(cache=True,fastmath=False)
def reference(v,t,k,nkeep):
    ix=np.argsort(k,kind='mergesort'); ix=ix[np.argsort(-t[ix],kind='mergesort')]; ix=ix[np.argsort(-v[ix],kind='mergesort')]
    out=np.zeros(len(v),np.bool_); out[ix[:nkeep]]=True; return out

@njit(cache=True,fastmath=False)
def choose(v,t,k,nkeep):
    n=len(v); assert len(t)==n and len(k)==n and 0<=nkeep<=n
    out=np.zeros(n,np.bool_)
    if nkeep==0: return out
    if nkeep==n:
        out[:]=True; return out
    ix=np.arange(n); lo,hi=0,n-1
    for _ in range(64):
        if lo>=hi: break
        mid=(lo+hi)//2; pivot=ix[mid]; ix[mid]=ix[hi]; ix[hi]=pivot; q=lo
        for j in range(lo,hi):
            if better(ix[j],pivot,v,t,k):
                ix[j],ix[q]=ix[q],ix[j]; q+=1
        ix[q],ix[hi]=ix[hi],ix[q]
        if q==nkeep-1: break
        if q<nkeep-1: lo=q+1
        else: hi=q-1
    else:
        return reference(v,t,k,nkeep)
    out[ix[:nkeep]]=True; return out

if __name__=='__main__':
    rng=np.random.default_rng(261145); checks=0
    for n in range(1,5):
        for v in itertools.product(range(2),repeat=n):
            for t in itertools.product(range(2),repeat=n):
                keys=rng.permutation(n); vs=np.array(v,dtype=np.float64); ts=np.array(t,dtype=np.int32)
                for k in range(n+1):
                    assert np.array_equal(choose(vs,ts,keys,k),reference(vs,ts,keys,k)); checks+=1
    for _ in range(2000):
        n=int(rng.integers(1,600)); v=rng.integers(-6,7,n).astype(float); t=rng.integers(-1,10,n,dtype=np.int32); keys=rng.permutation(n); k=int(rng.integers(0,n+1))
        assert np.array_equal(choose(v,t,keys,k),reference(v,t,keys,k)); checks+=1
    cases=[(rng.normal(size=160),rng.integers(0,500,160,dtype=np.int32),rng.permutation(160),128) for _ in range(2048)]
    times={}
    for name,fn in [('sort',reference),('partition',choose),('partition2',choose),('sort2',reference)]:
        start=time.perf_counter()
        for v,t,k,c in cases: fn(v,t,k,c)
        times[name]=(time.perf_counter()-start)*1000
    print(json.dumps(dict(checks=checks,timing_cases=len(cases),times_ms=times)))
