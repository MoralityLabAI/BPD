"""ER-38: equal-capacity FIFO, LRU and probe-segmented full prefix cache."""
import argparse,json,time
from pathlib import Path
import numpy as np
from numba import njit
from er23 import Bank,EPS,cost,decision
from er35 import StreamBank,BLOCK
from er33 import rss
from er32 import load

MODES=('fifo','lru','segfifo','seglru')
CAP=512

@njit(inline='always')
def unlink(slot,seg,prev,nxt,head,tail):
    a,b=prev[slot],nxt[slot]
    if a>=0: nxt[a]=b
    else: head[seg]=b
    if b>=0: prev[b]=a
    else: tail[seg]=a

@njit(inline='always')
def append(slot,seg,prev,nxt,head,tail):
    last=tail[seg]; prev[slot]=last; nxt[slot]=-1
    if last>=0: nxt[last]=slot
    else: head[seg]=slot
    tail[seg]=slot

@njit(cache=True,fastmath=False)
def finish(wt,vt,u,z2,reads,bias,r,y,floors,lower,order,cols,best,score,capacity,policy,eps):
    d,width=u.shape; p=len(y); work=p; calls=1
    loc=np.full(d*p,-1,np.int64); keys=np.full(capacity,-1,np.int64)
    prev=np.full(capacity,-1,np.int64); nxt=prev.copy()
    head=np.full(p,-1,np.int64); tail=head.copy(); used=np.zeros(p,np.int64)
    cz=np.empty((capacity,width)); cp=np.empty((capacity,d))
    hits=misses=dedup=evictions=0; probe_hits=np.zeros(p,np.int64); probe_misses=probe_hits.copy()
    segmented=policy>=2; touch=policy%2==1
    for start in range(0,len(order),64):
        if lower[order[start]]>score+eps: break
        ids=order[start:start+64].copy(); ids=ids[lower[ids]<=score+eps]
        terms=floors[ids].copy()
        for a in range(p):
            if not len(ids): break
            k=cols[a]; m=len(ids); seg=k if segmented else 0
            limit=capacity//p+int(seg<capacity%p) if segmented else capacity
            offset=seg*(capacity//p)+min(seg,capacity%p) if segmented else 0
            group=np.full(d,-1,np.int64); first=np.empty(m,np.int64); gi=np.empty(m,np.int64); ng=0
            for t in range(m):
                i=ids[t]//d
                if group[i]<0: group[i]=ng; first[ng]=i; ng+=1
                gi[t]=group[i]
            dedup+=m-ng
            z=np.empty((ng,width)); pr=np.empty((ng,d)); miss=np.empty(ng,np.int64); nm=0
            # Snapshot all batch hits before insertion. LRU touches in first-use order.
            for g in range(ng):
                key=first[g]*p+k; slot=loc[key]
                if slot>=0:
                    z[g]=cz[slot]; pr[g]=cp[slot]; hits+=1; probe_hits[k]+=1
                    if touch:
                        unlink(slot,seg,prev,nxt,head,tail); append(slot,seg,prev,nxt,head,tail)
                else: miss[nm]=g; nm+=1; misses+=1; probe_misses[k]+=1
            if nm:
                hh=np.empty((nm,width))
                for t in range(nm):
                    i=first[miss[t]]
                    for l in range(width): hh[t,l]=np.tanh(z2[k,l]-reads[i,k]*u[i,l])
                zz=hh@wt; pp=hh@vt
                for t in range(nm):
                    g=miss[t]; z[g]=zz[t]+bias[k]; pr[g]=pp[t]
                    if limit==0: continue
                    if used[seg]<limit:
                        slot=offset+used[seg]; used[seg]+=1
                    else:
                        slot=head[seg]; old=keys[slot]; assert old>=0 and loc[old]==slot
                        loc[old]=-1; evictions+=1; unlink(slot,seg,prev,nxt,head,tail)
                    key=first[g]*p+k; keys[slot]=key; loc[key]=slot; cz[slot]=z[g]; cp[slot]=pr[g]
                    append(slot,seg,prev,nxt,head,tail)
            h=np.empty((m,width))
            for t in range(m):
                j=ids[t]%d; g=gi[t]; s=pr[g,j]
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
    assert hits+misses+dedup==work-p and probe_hits.sum()==hits and probe_misses.sum()==misses
    # Validate each occupied list and its reverse map at exit.
    seen=np.zeros(capacity,np.bool_)
    for seg in range(p):
        slot=head[seg]; last=-1; count=0
        while slot>=0:
            assert not seen[slot] and prev[slot]==last and loc[keys[slot]]==slot
            if segmented: assert keys[slot]%p==seg
            seen[slot]=True; count+=1; last=slot; slot=nxt[slot]
        assert count==used[seg] and last==tail[seg]
    assert seen.sum()==(keys>=0).sum()
    return best,score,calls,work,hits,misses,dedup,evictions,probe_hits,probe_misses

def solve(args,y,mode,cap=CAP):
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
    wt,vt,u,z2,reads,bias,r,y,floors,lower,order,cols=[np.ascontiguousarray(a) for a in [b.w.T,b.v.T,b.u,b.z2,b.reads,b.bias,b.r,y,floors,lower,order,cols]]
    del b
    before=time.perf_counter(); d,width=u.shape
    best,score,calls,work,hits,misses,dedup,evictions,ph,pm=finish(wt,vt,u,z2,reads,bias,r,y,floors,lower,order,cols,best,score,cap,MODES.index(mode),EPS); end=time.perf_counter()
    return dict(best=int(best),score=float(score),incumbent=int(incumbent),calls=int(calls),pairprobes=int(work),hits=int(hits),misses=int(misses),dedup=int(dedup),evictions=int(evictions),probe_hits=ph.tolist(),probe_misses=pm.tolist(),capacity=cap,cache_bytes=8*(cap*(width+d+3)+d*p+3*p),screen_ms=(before-t)*1000,finish_ms=(end-before)*1000,ms=(end-t)*1000)

def warm():
    rng=np.random.default_rng(38); d=6
    w,o,u,v=[rng.standard_normal((d,d))*.1 for _ in range(4)]; x,c=rng.standard_normal((2,12,d)); y=rng.standard_normal(12)
    t=time.perf_counter()
    for mode in MODES: solve((w,o,u,v,x,c),y,mode,5)
    return (time.perf_counter()-t)*1000

def bench(paths,mode,reps,cap):
    cold=warm(); baseline=rss()
    for path in paths:
        args,y,hidden,noise=load(path); rows=[solve(args,y,mode,cap) for _ in range(reps)]; out=rows[0].copy()
        for x in rows:
            assert x['best']==out['best'] and abs(x['score']-out['score'])<EPS
            assert all(x[k]==out[k] for k in ('pairprobes','hits','misses','dedup','evictions','probe_hits','probe_misses'))
        for k in ('ms','screen_ms','finish_ms'): out[k]=float(np.median([x[k] for x in rows]))
        print(json.dumps(dict(path=path.name,width=args[0].shape[0],mode=mode,hidden=hidden,noise=noise,baseline=baseline,peak_kib=rss()['peak_kib'],warm_ms=cold,times_ms=[x['ms'] for x in rows],**out)),flush=True)

def audit(paths,receipt):
    rows=[json.loads(s) for s in receipt.read_text().splitlines()]
    for path in paths:
        args,y,hidden,noise=load(path); best,score,n,_=decision(Bank(*args),y,'eager',batch=64)
        rr=[r for r in rows if r['path']==path.name]; assert len(rr)==len(MODES) and {r['mode'] for r in rr}==set(MODES)
        assert all(r['best']==best and abs(r['score']-score)<EPS for r in rr)
        for k in ('pairprobes','incumbent','dedup','cache_bytes','capacity'): assert len({r[k] for r in rr})==1
        assert len({tuple(a+b for a,b in zip(r['probe_hits'],r['probe_misses'])) for r in rr})==1
        print(json.dumps(dict(path=path.name,best=best,score=score,hidden=hidden,exact_pairs=n,comparisons=len(rr),pairprobes=rr[0]['pairprobes'],max_score_error=max(abs(r['score']-score) for r in rr))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=MODES+('audit',)); p.add_argument('paths',type=Path,nargs='+'); p.add_argument('--reps',type=int,default=2); p.add_argument('--receipt',type=Path); p.add_argument('--capacity',type=int,default=CAP); a=p.parse_args()
    if a.mode=='audit': audit(a.paths,a.receipt)
    else: bench(a.paths,a.mode,a.reps,a.capacity)
