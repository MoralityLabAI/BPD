"""ER-09: direct bit-plane construction vs dense temporary."""
import json,time,tracemalloc,gc
from itertools import combinations
import numpy as np
from moments import world,pack,P

def dense(f,sup):
    obs=f[:,sup].sum(axis=2).T.astype(np.uint8)
    return [(pack(obs[:,a]&1),pack((obs[:,a]&2)!=0)) for a in range(P)]
def direct(f,sup):
    masks=[]
    for a in range(P):
        v=f[a,sup[:,0]].astype(np.uint8)
        v+=f[a,sup[:,1]]; v+=f[a,sup[:,2]]
        masks.append((pack(v&1),pack((v&2)!=0)))
    return masks

def measure(fn,f,sup):
    gc.collect(); tracemalloc.start(); m=fn(f,sup); _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    return m,peak

def main():
    rows=[]
    for n in (6,7,8,9,10):
        sup=np.array(list(combinations(range(n*(n-1)),3))); f,_,_=world(n,600+n,sup)
        d,dp=measure(dense,f,sup); b,bp=measure(direct,f,sup); assert d==b; del d,b
        times=[]
        for r in range(5):
            ts={}; outputs={}
            for fn in ((dense,direct) if r%2==0 else (direct,dense)):
                t=time.perf_counter(); outputs[fn.__name__]=fn(f,sup); ts[fn.__name__]=(time.perf_counter()-t)*1000
            assert outputs['dense']==outputs['direct']; times.append([ts['dense'],ts['direct']]); del outputs
        rows.append(dict(n=n,hypotheses=len(sup),raw_payload=2*P*((len(sup)+7)//8),dense_peak_tracemalloc=dp,direct_peak_tracemalloc=bp,median_ms=np.median(times,axis=0).tolist(),times_ms=times))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
