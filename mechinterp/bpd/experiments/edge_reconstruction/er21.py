"""Demand-matched eager versus sorted bound refinement on 12 requested probes."""
import json,time
import numpy as np
import torch
from er20 import Bank,train,WIDTHS,ROOTS,P,CAL,B
from certified_crosspass import EPS


def sorted_refine(pred,err,y,query):
    d=abs(pred-y); lo=(np.maximum(0,d-err)**2).sum(1); hi=((d+err)**2).sum(1)
    incumbent=int(np.argmin(hi)); best=incumbent; score=float(np.sum((query(best)-y)**2)); calls=1
    assert lo[best]-EPS<=score<=hi[best]+EPS
    for k in np.argsort(lo,kind='stable'):
        if k==incumbent: continue
        if score+EPS<lo[k]: break
        v=float(np.sum((query(int(k))-y)**2)); calls+=1
        assert lo[k]-EPS<=v<=hi[k]+EPS
        if v<score or (v==score and k<best): best=int(k); score=v
    return best,calls


def decision(model,x,c,y,mode):
    t=time.perf_counter(); b=Bank(model,x,c)
    if mode=='eager': best=int(np.argmin(((b.eager()-y)**2).sum(1))); calls=b.d**2
    else:
        pred,err=b.bounds(); best,calls=sorted_refine(pred,err,y,b.exact)
    return best,calls,1000*(time.perf_counter()-t)


def run():
    torch.set_num_threads(1)
    for d in WIDTHS:
        for root in ROOTS:
            model,_=train(d,root); rng=np.random.default_rng(np.random.SeedSequence([261138,d,root])); x,c=rng.standard_normal((CAL+P,d)),rng.standard_normal((CAL+P,d))
            cal=Bank(model,x[:CAL],c[:CAL]); sigma=max(1e-5,.1*np.sqrt(np.mean((cal.eager()-cal.singles()[-1])**2))); x,c=x[CAL:],c[CAL:]
            bank=Bank(model,x,c); truth=bank.eager(); rows=[]
            for ep in range(8):
                k=int(rng.integers(d*d)); a=rng.permutation(P)[:B]; y=truth[k,a]+rng.normal(0,sigma,B); best=int(np.argmin(((truth[:,a]-y)**2).sum(1))); times={m:[] for m in ('eager','sorted')}
                for rep in range(3):
                    for mode in (('eager','sorted') if (rep+ep+root)%2 else ('sorted','eager')):
                        got,calls,ms=decision(model,x[a],c[a],y,mode); assert got==best; times[mode].append(ms)
                        if mode=='sorted': n=calls
                rows.append(dict(episode=ep,hit=int(k==best),calls=n,eager_ms=float(np.median(times['eager'])),sorted_ms=float(np.median(times['sorted']))))
            print(json.dumps(dict(d=d,root=root,rows=rows)),flush=True)
if __name__=='__main__': run()
