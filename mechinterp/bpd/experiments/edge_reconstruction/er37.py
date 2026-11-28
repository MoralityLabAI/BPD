"""ER-37: equal-byte prefix-only FIFO versus full-projection FIFO."""
import argparse,json,time
from pathlib import Path
import numpy as np
from numba import njit
from er23 import Bank,EPS,cost,decision
from er35 import StreamBank,BLOCK
from er33 import rss
from er32 import load

MODES=('full512','prefix512','prefix_equal')

@njit(fastmath=False,cache=True)
def finish(wt,vt,v,u,z2,reads,bias,r,y,floors,lower,order,cols,best,score,capacity,prefix_only,eps):
    d,width=u.shape; p=len(y); work=p; calls=1
    loc=np.full(d*p,-1,np.int64); keys=np.full(capacity,-1,np.int64)
    cz=np.empty((capacity,width)); cp=np.empty((0 if prefix_only else capacity,d)); cursor=0
    hits=misses=dedup=evictions=h2_terms=projection_terms=0
    for start in range(0,len(order),64):
        if lower[order[start]]>score+eps: break
        ids=order[start:start+64].copy(); ids=ids[lower[ids]<=score+eps]
        terms=floors[ids].copy()
        for a in range(p):
            if not len(ids): break
            k=cols[a]; m=len(ids)
            group=np.full(d,-1,np.int64); first=np.empty(m,np.int64); gi=np.empty(m,np.int64); ng=0
            for t in range(m):
                i=ids[t]//d
                if group[i]<0: group[i]=ng; first[ng]=i; ng+=1
                gi[t]=group[i]
            dedup+=m-ng
            z=np.empty((ng,width)); pr=np.empty((0 if prefix_only else ng,d)); h2=np.empty((ng if prefix_only else 0,width))
            miss=np.empty(ng,np.int64); nm=0
            # Read hits before inserting misses (FIFO may evict the hit slots).
            for g in range(ng):
                key=first[g]*p+k; slot=loc[key]
                if slot>=0:
                    z[g]=cz[slot]; hits+=1
                    if not prefix_only: pr[g]=cp[slot]
                else: miss[nm]=g; nm+=1; misses+=1
                if prefix_only:
                    i=first[g]
                    for l in range(width): h2[g,l]=np.tanh(z2[k,l]-reads[i,k]*u[i,l])
            if prefix_only: h2_terms+=ng
            else: h2_terms+=nm
            if nm:
                hh=np.empty((nm,width))
                for t in range(nm):
                    g=miss[t]; i=first[g]
                    if prefix_only: hh[t]=h2[g]
                    else:
                        for l in range(width): hh[t,l]=np.tanh(z2[k,l]-reads[i,k]*u[i,l])
                zz=hh@wt
                pp=np.empty((0,d)) if prefix_only else hh@vt
                if not prefix_only: projection_terms+=nm*d
                for t in range(nm):
                    g=miss[t]; z[g]=zz[t]+bias[k]
                    if not prefix_only: pr[g]=pp[t]
                    slot=cursor; cursor=(cursor+1)%capacity; old=keys[slot]
                    if old>=0: loc[old]=-1; evictions+=1
                    key=first[g]*p+k; keys[slot]=key; loc[key]=slot; cz[slot]=z[g]
                    if not prefix_only: cp[slot]=pr[g]
            h=np.empty((m,width))
            for t in range(m):
                j=ids[t]%d; g=gi[t]
                if prefix_only:
                    s=0.
                    for l in range(width): s+=h2[g,l]*v[j,l]
                else: s=pr[g,j]
                for l in range(width): h[t,l]=np.tanh(z[g,l]-s*u[j,l])
            if prefix_only: projection_terms+=m
            zz=h@wt; energy=np.empty(m)
            for t in range(m):
                f=0.
                for l in range(width): f+=np.tanh(zz[t,l]+bias[k,l])*r[k,l]
                value=(f-y[k])**2; assert terms[t,k]<=value+eps
                terms[t,k]=value; energy[t]=terms[t].sum()
            work+=m
            if a+1>=p:
                calls+=m
                for t in range(m):
                    value=energy[t]; key=ids[t]
                    if value<score or (value==score and key<best): best,score=key,value
                break
            keep=energy<=score+eps; ids=ids[keep]; terms=terms[keep]
    assert hits+misses+dedup==work-p
    return best,score,calls,work,hits,misses,dedup,evictions,h2_terms,projection_terms

def capacity_for(mode,d,width):
    if mode=='prefix_equal': return 512*(width+d+1)//(width+1)
    return 512

def solve(args,y,mode):
    t=time.perf_counter(); b=StreamBank(*args,'replay')
    n,p=b.d*b.d,b.p; floors=np.empty((n,p)); gap=np.zeros(p); upper=np.inf; best=-1
    for i in range(0,b.d,BLOCK):
        end=min(i+BLOCK,b.d); pred,err=b.bound(i,end); delta=abs(pred-y)
        l=np.maximum(0,delta-err)**2; h=(delta+err)**2; floors[i*b.d:end*b.d]=l; gap+=(h-l).sum(0)
        hi=h.sum(1); j=int(hi.argmin())
        if hi[j]<upper: upper=float(hi[j]); best=i*b.d+j
    del pred,err,delta,l,h,hi
    lower=floors.sum(1); score=float(cost(b.exact_incumbent(best),y)); assert lower[best]<=score+EPS<=upper+2*EPS
    order=np.argsort(lower,kind='stable'); order=order[order!=best]; cols=np.argsort(-gap,kind='stable'); incumbent=best
    wt,vt,v,u,z2,reads,bias,r,y,floors,lower,order,cols=[np.ascontiguousarray(a) for a in [b.w.T,b.v.T,b.v,b.u,b.z2,b.reads,b.bias,b.r,y,floors,lower,order,cols]]
    del b
    d,width=u.shape; cap=capacity_for(mode,d,width); only=mode!='full512'
    before=time.perf_counter(); best,score,calls,work,hits,misses,dedup,evictions,h2_terms,projection_terms=finish(wt,vt,v,u,z2,reads,bias,r,y,floors,lower,order,cols,best,score,cap,only,EPS); end=time.perf_counter()
    cache_bytes=8*(cap*(width+1+(0 if only else d))+d*p)
    return dict(best=int(best),score=float(score),incumbent=int(incumbent),calls=int(calls),pairprobes=int(work),hits=int(hits),misses=int(misses),dedup=int(dedup),evictions=int(evictions),h2_terms=int(h2_terms),projection_terms=int(projection_terms),capacity=cap,cache_bytes=cache_bytes,screen_ms=(before-t)*1000,finish_ms=(end-before)*1000,ms=(end-t)*1000)

def warm():
    rng=np.random.default_rng(37); d=6
    w,o,u,v=[rng.standard_normal((d,d))*.1 for _ in range(4)]; x,c=rng.standard_normal((2,12,d)); y=rng.standard_normal(12)
    t=time.perf_counter()
    for mode in MODES: solve((w,o,u,v,x,c),y,mode)
    return (time.perf_counter()-t)*1000

def bench(paths,mode,reps):
    cold=warm(); baseline=rss()
    for path in paths:
        args,y,hidden,noise=load(path); rows=[solve(args,y,mode) for _ in range(reps)]; out=rows[0].copy()
        for x in rows:
            assert x['best']==out['best'] and abs(x['score']-out['score'])<EPS
            assert all(x[k]==out[k] for k in ('pairprobes','hits','misses','dedup','evictions','h2_terms','projection_terms'))
        for k in ('ms','screen_ms','finish_ms'): out[k]=float(np.median([x[k] for x in rows]))
        print(json.dumps(dict(path=path.name,width=args[0].shape[0],mode=mode,hidden=hidden,noise=noise,baseline=baseline,peak_kib=rss()['peak_kib'],warm_ms=cold,times_ms=[x['ms'] for x in rows],**out)),flush=True)

def audit(paths,receipt):
    rows=[json.loads(s) for s in receipt.read_text().splitlines()]
    for path in paths:
        args,y,hidden,noise=load(path); best,score,n,_=decision(Bank(*args),y,'eager',batch=64)
        rr=[r for r in rows if r['path']==path.name]; assert len(rr)==len(MODES) and {r['mode'] for r in rr}==set(MODES)
        assert all(r['best']==best and abs(r['score']-score)<EPS for r in rr)
        assert len({r['pairprobes'] for r in rr})==len({r['incumbent'] for r in rr})==1
        assert len({r['dedup'] for r in rr})==1
        by={r['mode']:r for r in rr}; assert by['full512']['misses']==by['prefix512']['misses']; assert by['prefix_equal']['cache_bytes']<=by['full512']['cache_bytes']
        print(json.dumps(dict(path=path.name,best=best,score=score,hidden=hidden,exact_pairs=n,comparisons=len(rr),pairprobes=rr[0]['pairprobes'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=MODES+('audit',)); p.add_argument('paths',type=Path,nargs='+'); p.add_argument('--reps',type=int,default=2); p.add_argument('--receipt',type=Path); a=p.parse_args()
    if a.mode=='audit': audit(a.paths,a.receipt)
    else: bench(a.paths,a.mode,a.reps)
