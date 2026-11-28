"""ER-34: compact exact floors; fixed incumbent and probe order."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from numba import njit
from er23 import EPS,cost
from er33 import BlockBank,rss
from er32 import load,payload
MODES=('dense','recompute','provisional')

@njit(fastmath=False)
def finish(wt,u,z3,proj,bias,r,y,floors,fidx,lower,order,cols,best,score,step,eps):
    d=len(u); p=len(y); work=p; calls=1; checks=0
    for start in range(0,len(order),64):
        if lower[order[start]]>score+eps: break
        ids=order[start:start+64].copy(); ids=ids[lower[ids]<=score+eps]
        ix=fidx[ids]; assert np.all(ix>=0)
        terms=floors[ix].copy()
        for a in range(0,p,step):
            if not len(ids): break
            kcols=cols[a:a+step]; m=len(ids); q=len(kcols)
            h=np.empty((m*q,d),np.float64)
            for t in range(m):
                i,j=ids[t]//d,ids[t]%d
                for c in range(q):
                    k=kcols[c]; v=proj[i,k,j]
                    for l in range(d): h[t*q+c,l]=np.tanh(z3[i,k,l]-v*u[j,l])
            zz=h@wt; energy=np.empty(m,np.float64)
            for t in range(m):
                for c in range(q):
                    k=kcols[c]; f=0.
                    for l in range(d): f+=np.tanh(zz[t*q+c,l]+bias[k,l])*r[k,l]
                    v=(f-y[k])**2
                    assert terms[t,k]<=v+eps
                    terms[t,k]=v; checks+=1
                energy[t]=terms[t].sum()
            work+=m*q
            if a+step>=p:
                calls+=m
                for t in range(m):
                    v=energy[t]; k=ids[t]
                    if v<score or (v==score and k<best): best,score=k,v
                break
            keep=energy<=score+eps; ids=ids[keep]; terms=terms[keep]
    return best,score,calls,work,checks

def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def solve(args,y,mode,audit=False):
    t=time.perf_counter(); stages={'start':rss()}; b=BlockBank(*args,block=8); b.setup_rank()
    stages['setup']=rss(); setup_ms=(time.perf_counter()-t)*1000
    n,p=b.d*b.d,b.p; lower=np.empty(n); gap=np.zeros(p); upper=np.inf; best=-1
    floors=np.empty((n,p)) if mode=='dense' else None
    provisional=[]; provisional_rows=0; recompute_rows=0
    for i in range(0,b.d,8):
        end=min(i+8,b.d); lo,hi=i*b.d,end*b.d
        pred,err=b.bound(i,end); delta=abs(pred-y)
        l=np.maximum(0,delta-err)**2; h=(delta+err)**2
        low=l.sum(1); lower[lo:hi]=low; gap+=(h-l).sum(0)
        hu=h.sum(1); j=int(hu.argmin())
        if hu[j]<upper: upper=float(hu[j]); best=lo+j
        if mode=='dense': floors[lo:hi]=l
        elif mode=='provisional':
            # Final exact incumbent <= current minimum upper in real arithmetic.
            keep=np.flatnonzero(low<=upper+2*EPS)
            provisional.append((lo+keep,l[keep])); provisional_rows+=len(keep)
    del pred,err,delta,l,h,hu,low
    incumbent=best; score=float(cost(b.exact([best]),y)[0]); assert lower[best]<=score+EPS<=upper+2*EPS
    eligible=np.flatnonzero((lower<=score+EPS)&(np.arange(n)!=best)); order=np.argsort(lower,kind='stable'); order=order[order!=best]; cols=np.argsort(-gap,kind='stable')
    order=order[lower[order]<=score+EPS]; assert len(order)==len(eligible)
    stages['first_screen']=rss(); first_ms=(time.perf_counter()-t)*1000-setup_ms
    if mode=='dense': fidx=np.arange(n,dtype=np.int64)
    else:
        fidx=np.full(n,-1,dtype=np.int64); fidx[eligible]=np.arange(len(eligible)); floors=np.empty((len(eligible),p))
        if mode=='recompute':
            for i in range(0,b.d,8):
                end=min(i+8,b.d); lo,hi=i*b.d,end*b.d
                a,z=np.searchsorted(eligible,(lo,hi)); keys=eligible[a:z]
                if not len(keys): continue
                pred,err=b.bound(i,end); l=np.maximum(0,abs(pred-y)-err)**2
                floors[a:z]=l[keys-lo]; recompute_rows+=hi-lo
            if recompute_rows: del pred,err,l
        else:
            written=np.zeros(len(eligible),bool)
            for k in range(len(provisional)):
                ids,rows=provisional[k]; j=fidx[ids]; keep=j>=0
                floors[j[keep]]=rows[keep]; written[j[keep]]=True
                provisional[k]=None
            assert np.all(written)
            del written,ids,rows,j,keep
        assert np.array_equal(floors.sum(1),lower[eligible])
    stages['floor_ready']=rss()
    hashes=dict(lower=digest(lower),order=digest(order),cols=digest(cols),eligible=digest(eligible),floors=digest(floors[fidx[eligible]])) if audit else None
    arr=[np.ascontiguousarray(a) for a in [b.w.T,b.u,b.first_z3,b.proj,b.bias,b.r,y,floors,fidx,lower,order,cols]]
    before=time.perf_counter(); best,score,calls,work,checks=finish(*arr,best,score,1,EPS); end=time.perf_counter(); stages['finish']=rss()
    return dict(best=int(best),score=float(score),incumbent=int(incumbent),calls=int(calls),pairprobes=int(work),checks=int(checks),eligible=int(len(eligible)),floor_bytes=floors.nbytes,index_bytes=fidx.nbytes,state_bytes=payload(b),provisional_rows=provisional_rows,recompute_rows=recompute_rows,setup_ms=setup_ms,first_screen_ms=first_ms,screen_ms=(before-t)*1000,finish_ms=(end-before)*1000,ms=(end-t)*1000,hashes=hashes,stages=stages)

def warmup():
    rng=np.random.default_rng(34); d=6
    w,o,u,v=[rng.standard_normal((d,d))*.1 for _ in range(4)]; x,c=rng.standard_normal((2,12,d)); y=rng.standard_normal(12)
    t=time.perf_counter(); solve((w,o,u,v,x,c),y,'dense'); return (time.perf_counter()-t)*1000

def audit(path):
    args,y,hidden,noise=load(path); b=BlockBank(*args,block=8); b.setup_rank(); checks=0; score=np.inf; best=-1
    for i in range(0,b.d,8):
        end=min(i+8,b.d); f,e=b.bound(i,end); ids=np.arange(i*b.d,end*b.d)
        for k in range(0,len(ids),64):
            keys=ids[k:k+64]; ex=b.exact(keys); assert np.all(abs(ex-f[k:k+64])<=e[k:k+64]+EPS); c=cost(ex,y); j=int(c.argmin())
            if c[j]<score: best,score=int(keys[j]),float(c[j])
        checks+=f.size
    del b
    rr={m:solve(args,y,m,True) for m in MODES}; ref=rr['dense']
    for m,r in rr.items():
        assert r['best']==best and abs(r['score']-score)<EPS
        assert r['hashes']==ref['hashes'] and r['pairprobes']==ref['pairprobes'] and r['incumbent']==ref['incumbent'],m
    return dict(path=path.name,best=best,score=score,hidden=hidden,noise=noise,scalar_checks=checks,hashes=ref['hashes'],work=ref['pairprobes'],eligible=ref['eligible'])

def bench(paths,mode,reps):
    warm=warmup(); baseline=rss()
    for path in paths:
        args,y,hidden,noise=load(path); rows=[solve(args,y,mode) for _ in range(reps)]; r=rows[0].copy()
        for x in rows:
            assert x['best']==r['best'] and abs(x['score']-r['score'])<EPS and x['pairprobes']==r['pairprobes']
        for k in ('ms','setup_ms','screen_ms','first_screen_ms','finish_ms'): r[k]=float(np.median([x[k] for x in rows]))
        print(json.dumps(dict(path=path.name,mode=mode,hidden=hidden,noise=noise,baseline=baseline,peak_kib=rss()['peak_kib'],compile_ms=warm,times_ms=[x['ms'] for x in rows],stage_repetitions=[x['stages'] for x in rows],**r)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('audit','bench')); p.add_argument('paths',type=Path,nargs='+'); p.add_argument('--mode',choices=MODES,default='dense'); p.add_argument('--reps',type=int,default=2); a=p.parse_args()
    if a.action=='audit':
        warmup()
        for path in a.paths: print(json.dumps(audit(path)),flush=True)
    else: bench(a.paths,a.mode,a.reps)
