"""ER-30: compiled stage loop vs frozen NumPy controls."""
import argparse,json,time
import numpy as np
from numba import njit
from er23 import Bank,EPS,cost,decision
from er25 import world
from er29 import partial
WIDTHS=(32,64,128)
ROOTS=(60,61,62)
MODES=('eager','gap4','gap2','gap1','fused1','fused2')

@njit(fastmath=False)
def finish(wt,u,z3,proj,bias,r,y,floors,lower,order,cols,best,score,step,eps):
    d=len(u); p=len(y); work=p; calls=1; checks=0
    for start in range(0,len(order),64):
        if lower[order[start]]>score+eps: break
        ids=order[start:start+64].copy(); ids=ids[lower[ids]<=score+eps]
        terms=floors[ids].copy()
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

def fused(args,y,step):
    t=time.perf_counter(); b=Bank(*args); b.setup(); d=b.d; p=b.p
    floors=np.empty((d*d,p)); gap=np.zeros(p); upper=np.inf; best=-1
    for i in range(0,d,8):
        end=min(i+8,d); pred,err=b.bounds(i,end); delta=abs(pred-y)
        l=np.maximum(0,delta-err)**2; h=(delta+err)**2
        floors[i*d:end*d]=l; gap+=(h-l).sum(0); hi=h.sum(1); j=int(hi.argmin())
        if hi[j]<upper: upper=float(hi[j]); best=i*d+j
    lower=floors.sum(1); score=float(cost(b.exact([best]),y)[0]); assert lower[best]<=score+EPS<=upper+2*EPS
    order=np.argsort(lower,kind='stable'); order=order[order!=best]; cols=np.argsort(-gap,kind='stable')
    arr=[b.w.T,b.u,b.first_z3,b.proj,b.bias,b.r,y,floors,lower,order,cols]
    arr=[np.ascontiguousarray(a) for a in arr]; before=time.perf_counter()
    best,score,calls,work,checks=finish(*arr,best,score,step,EPS)
    end=time.perf_counter()
    return dict(best=int(best),score=float(score),calls=int(calls),pairprobes=int(work),checks=int(checks),screen_ms=(before-t)*1000,finish_ms=(end-before)*1000,ms=(end-t)*1000)

def warmup():
    rng=np.random.default_rng(30); d=6
    w,o,u,v=[rng.standard_normal((d,d))*.1 for _ in range(4)]; x,c=rng.standard_normal((2,12,d)); y=rng.standard_normal(12)
    start=time.perf_counter(); fused((w,o,u,v,x,c),y,1)
    return (time.perf_counter()-start)*1000

def run(width,roots,reps):
    print(json.dumps(dict(kind='warmup',width=width,compile_and_first_run_ms=warmup())),flush=True)
    for root in roots:
        for ep,(args,y,hidden,noise) in enumerate(world(width,root)):
            out={m:[] for m in MODES}
            for rep in range(reps):
                shift=(root+ep+rep)%len(MODES)
                for m in MODES[shift:]+MODES[:shift]:
                    if m=='eager':
                        t=time.perf_counter(); b=Bank(*args); best,score,calls,_=decision(b,y,'eager',batch=64)
                        r=dict(best=best,score=score,calls=calls,pairprobes=calls*len(y),ms=(time.perf_counter()-t)*1000)
                    elif m.startswith('fused'): r=fused(args,y,int(m[-1]))
                    else: r=partial(args,y,m,0)
                    out[m].append(r)
            ref=out['eager'][0]; results={}
            for m,rr in out.items():
                assert all(r['best']==ref['best'] and abs(r['score']-ref['score'])<EPS for r in rr)
                assert len({r['pairprobes'] for r in rr})==1
                r=rr[0].copy()
                for k in ('ms','screen_ms','finish_ms'):
                    if k in r:r[k]=float(np.median([z[k] for z in rr]))
                r['times_ms']=[z['ms'] for z in rr]; results[m]=r
            print(json.dumps(dict(kind='episode',width=width,root=root,episode=ep,noise=noise,hidden=hidden,hit=hidden==ref['best'],arms=results)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--width',type=int,choices=WIDTHS,required=True); p.add_argument('--roots',type=int,nargs='+',default=ROOTS); p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args(); run(a.width,a.roots,a.repeats)
