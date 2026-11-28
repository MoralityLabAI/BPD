"""ER-36: batch-local prefix deduplication and bounded FIFO reuse."""
import argparse,json,time
from pathlib import Path
import numpy as np
from numba import njit
from er23 import EPS,cost
from er35 import StreamBank,finish as old_finish,BLOCK
from er33 import rss
from er32 import load

@njit(fastmath=False,cache=True)
def cached_finish(wt,vt,u,z2,reads,bias,r,y,floors,lower,order,cols,best,score,capacity,eps):
    d,width=u.shape; p=len(y); work=p; calls=1
    loc=np.full(d*p,-1,np.int64); keys=np.full(capacity,-1,np.int64)
    cz=np.empty((capacity,width)); cp=np.empty((capacity,d)); cursor=0
    misses=hits=dedup=evictions=unique=0
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
            unique+=ng; dedup+=m-ng
            z=np.empty((ng,width)); proj=np.empty((ng,d)); miss=np.empty(ng,np.int64); nm=0
            # Snapshot all hits before FIFO writes can evict them.
            for g in range(ng):
                key=first[g]*p+k; slot=loc[key]
                if slot>=0:
                    z[g]=cz[slot]; proj[g]=cp[slot]; hits+=1
                else: miss[nm]=g; nm+=1; misses+=1
            if nm:
                h2=np.empty((nm,width))
                for t in range(nm):
                    i=first[miss[t]]
                    for l in range(width): h2[t,l]=np.tanh(z2[k,l]-reads[i,k]*u[i,l])
                zz=h2@wt; pp=h2@vt
                for t in range(nm):
                    g=miss[t]; z[g]=zz[t]+bias[k]; proj[g]=pp[t]
                    if capacity:
                        slot=cursor; cursor=(cursor+1)%capacity; old=keys[slot]
                        if old>=0: loc[old]=-1; evictions+=1
                        key=first[g]*p+k; keys[slot]=key; loc[key]=slot; cz[slot]=z[g]; cp[slot]=proj[g]
            h=np.empty((m,width))
            for t in range(m):
                j=ids[t]%d; g=gi[t]; s=proj[g,j]
                for l in range(width): h[t,l]=np.tanh(z[g,l]-s*u[j,l])
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
    assert hits+misses==unique and unique+dedup==work-p
    return best,score,calls,work,hits,misses,dedup,evictions

def solve(args,y,mode):
    t=time.perf_counter(); b=StreamBank(*args,'stream' if mode=='stream' else 'replay')
    n,p=b.d*b.d,b.p; floors=np.empty((n,p)); gap=np.zeros(p); upper=np.inf; best=-1
    for i in range(0,b.d,BLOCK):
        end=min(i+BLOCK,b.d); pred,err=b.bound(i,end); delta=abs(pred-y)
        l=np.maximum(0,delta-err)**2; h=(delta+err)**2; floors[i*b.d:end*b.d]=l; gap+=(h-l).sum(0)
        hi=h.sum(1); j=int(hi.argmin())
        if hi[j]<upper: upper=float(hi[j]); best=i*b.d+j
    del pred,err,delta,l,h,hi
    lower=floors.sum(1); score=float(cost(b.exact_incumbent(best),y)); assert lower[best]<=score+EPS<=upper+2*EPS
    order=np.argsort(lower,kind='stable'); order=order[order!=best]; cols=np.argsort(-gap,kind='stable'); incumbent=best
    wt,u,v,z3,proj,z2,reads,bias,r,y,floors,lower,order,cols=[np.ascontiguousarray(a) for a in [b.w.T,b.u,b.v,b.first_z3,b.proj,b.z2,b.reads,b.bias,b.r,y,floors,lower,order,cols]]
    del b
    vt=np.ascontiguousarray(v.T); before=time.perf_counter(); d,width=u.shape
    if mode in ('stream','replay'):
        best,score,calls,work,_,prefix=old_finish(wt,u,v,z3,proj,z2,reads,bias,r,y,floors,lower,order,cols,best,score,0 if mode=='stream' else 2,EPS)
        hits=dedup=evictions=0; misses=prefix; capacity=0; cache_bytes=0
    else:
        capacity=int(mode[5:]); best,score,calls,work,hits,misses,dedup,evictions=cached_finish(wt,vt,u,z2,reads,bias,r,y,floors,lower,order,cols,best,score,capacity,EPS)
        cache_bytes=8*(capacity*(width+d+1)+d*p)
    end=time.perf_counter()
    return dict(best=int(best),score=float(score),incumbent=int(incumbent),calls=int(calls),pairprobes=int(work),hits=int(hits),misses=int(misses),dedup=int(dedup),evictions=int(evictions),capacity=capacity,cache_bytes=cache_bytes,screen_ms=(before-t)*1000,finish_ms=(end-before)*1000,ms=(end-t)*1000)

def warm(mode):
    rng=np.random.default_rng(36); d=6
    w,o,u,v=[rng.standard_normal((d,d))*.1 for _ in range(4)]; x,c=rng.standard_normal((2,12,d)); y=rng.standard_normal(12)
    t=time.perf_counter(); solve((w,o,u,v,x,c),y,mode); return (time.perf_counter()-t)*1000

def bench(paths,mode,reps):
    cold=warm(mode); baseline=rss()
    for path in paths:
        args,y,hidden,noise=load(path); rows=[solve(args,y,mode) for _ in range(reps)]; out=rows[0].copy()
        for x in rows:
            assert x['best']==out['best'] and abs(x['score']-out['score'])<EPS
            assert all(x[k]==out[k] for k in ('pairprobes','hits','misses','dedup','evictions'))
        for k in ('ms','screen_ms','finish_ms'): out[k]=float(np.median([x[k] for x in rows]))
        print(json.dumps(dict(path=path.name,width=args[0].shape[0],mode=mode,hidden=hidden,noise=noise,baseline=baseline,peak_kib=rss()['peak_kib'],cold_ms=cold,times_ms=[x['ms'] for x in rows],**out)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('stream','replay','cache0','cache128','cache512')); p.add_argument('paths',type=Path,nargs='+'); p.add_argument('--reps',type=int,default=2); a=p.parse_args(); bench(a.paths,a.mode,a.reps)
