"""ER-29: probe-order nulls and interval-based exclusion utility."""
import argparse,json,time
import numpy as np
from er23 import Bank,EPS,cost,decision
from er25 import world
WIDTHS=(32,64,128)
ROOTS=(57,58,59)
MODES=('eager','gap4','fixed4','shuffle4','gap2','gap6','utility4')

def exact_cols(b,keys,cols):
    i,j=keys//b.d,keys%b.d
    z=b.first_z3[i[:,None],cols[None,:]]-b.proj[i[:,None],cols[None,:],j[:,None]][:,:,None]*b.u[j,None,:]
    return (np.tanh(np.tanh(z)@b.w.T+b.bias[cols])*b.r[cols]).sum(-1)

def choose(mode,floors,lower,score,eligible,pred,err,y,gap,seed):
    if mode.startswith('gap'): return np.argsort(-gap,kind='stable')
    if mode=='fixed4': return np.arange(len(y))
    if mode=='shuffle4': return np.random.default_rng(seed).permutation(len(y))
    assert mode=='utility4'
    if not len(eligible): return np.arange(len(y))
    # Uniform pseudo-model within the admissible response interval. Score the
    # probability that an exact residual on this probe alone excludes a candidate.
    radius=np.sqrt(np.maximum(0,score+EPS-lower[eligible,None]+floors[eligible]))
    left=pred[eligible]-err[eligible]; right=pred[eligible]+err[eligible]
    overlap=np.maximum(0,np.minimum(right,y+radius)-np.maximum(left,y-radius))
    prob=np.clip(1-overlap/np.maximum(right-left,np.finfo(float).tiny),0,1)
    return np.argsort(-prob.mean(axis=0),kind='stable')

def partial(args,y,mode,seed):
    t=time.perf_counter(); b=Bank(*args); n=b.d**2; p=b.p; b.setup()
    floors=np.empty((n,p)); gap=np.zeros(p); upper_best=np.inf; incumbent=-1
    pred_all=np.empty_like(floors) if mode=='utility4' else None
    err_all=np.empty_like(floors) if mode=='utility4' else None
    for i in range(0,b.d,8):
        end=min(i+8,b.d); pred,err=b.bounds(i,end); delta=abs(pred-y)
        l=np.maximum(0,delta-err)**2; u=(delta+err)**2; sl=slice(i*b.d,end*b.d)
        floors[sl]=l; gap+=(u-l).sum(0)
        if pred_all is not None: pred_all[sl]=pred; err_all[sl]=err
        hi=u.sum(1); j=int(hi.argmin())
        if hi[j]<upper_best: upper_best=float(hi[j]); incumbent=i*b.d+j
    lower=floors.sum(1); best=incumbent; score=float(cost(b.exact([best]),y)[0]); assert lower[best]<=score+EPS<=upper_best+2*EPS
    order=np.argsort(lower,kind='stable'); order=order[order!=best]; eligible=order[lower[order]<=score+EPS]
    cols=choose(mode,floors,lower,score,eligible,pred_all,err_all,y,gap,seed); del pred_all,err_all
    step=int(mode[-1]); pairprobes=p; calls=1; stages=[0]*((p+step-1)//step); screen_ms=(time.perf_counter()-t)*1000
    for a in range(0,len(order),64):
        ids=order[a:a+64]
        if lower[ids[0]]>score+EPS: break
        ids=ids[lower[ids]<=score+EPS]
        if not len(ids): continue
        terms=floors[ids].copy()
        for stage,z in enumerate(range(0,p,step)):
            c=cols[z:z+step]; v=(exact_cols(b,ids,c)-y[c])**2
            assert np.all(terms[:,c]<=v+EPS)
            terms[:,c]=v; pairprobes+=len(ids)*len(c); stages[stage]+=len(ids); energy=terms.sum(1)
            if z+step>=p:
                calls+=len(ids); val=float(energy.min()); key=int(ids[energy==val].min())
                if val<score or (val==score and key<best): score,best=val,key
                break
            keep=energy<=score+EPS; ids=ids[keep]; terms=terms[keep]
            if not len(ids): break
    return dict(best=best,score=score,calls=calls,pairprobes=pairprobes,stage_counts=stages,probe_order=cols.tolist(),initial_eligible=len(eligible),screen_ms=screen_ms,ms=(time.perf_counter()-t)*1000)

def run(width,roots,reps):
    for root in roots:
        for ep,(args,y,hidden,noise) in enumerate(world(width,root)):
            rows={m:[] for m in MODES}; seed=np.random.SeedSequence([261142,width,root,ep])
            for rep in range(reps):
                shift=(root+ep+rep)%len(MODES)
                for m in MODES[shift:]+MODES[:shift]:
                    if m=='eager':
                        t=time.perf_counter(); b=Bank(*args); best,score,calls,_=decision(b,y,'eager',batch=64)
                        v=dict(best=best,score=score,calls=calls,pairprobes=calls*len(y),ms=(time.perf_counter()-t)*1000)
                    else: v=partial(args,y,m,seed)
                    rows[m].append(v)
            ref=rows['eager'][0]; out={}
            for m,rr in rows.items():
                assert all(v['best']==ref['best'] and abs(v['score']-ref['score'])<EPS for v in rr)
                assert len({v['pairprobes'] for v in rr})==1
                v=rr[0].copy(); v['ms']=float(np.median([r['ms'] for r in rr])); v['times_ms']=[r['ms'] for r in rr]
                if 'screen_ms' in v: v['screen_ms']=float(np.median([r['screen_ms'] for r in rr]))
                out[m]=v
            print(json.dumps(dict(width=width,root=root,episode=ep,noise=noise,hidden=hidden,hit=hidden==ref['best'],arms=out)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--width',type=int,choices=WIDTHS,required=True); p.add_argument('--roots',type=int,nargs='+',default=ROOTS); p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args(); run(a.width,a.roots,a.repeats)
