"""ER-28: exact partial-probe energy screening with admissible additive floors."""
import argparse,json,time
import numpy as np
from er23 import Bank,EPS,cost
from er25 import world
from er27 import run_one
WIDTHS=(32,64,128)
ROOTS=(55,56)
MODES=('eager','base','partial4','partial6')


def exact_cols(b,keys,cols):
    i,j=keys//b.d,keys%b.d
    z=b.first_z3[i[:,None],cols[None,:]]-b.proj[i[:,None],cols[None,:],j[:,None]][:,:,None]*b.u[j,None,:]
    return (np.tanh(np.tanh(z)@b.w.T+b.bias[cols])*b.r[cols]).sum(-1)


def partial(args,y,step):
    t=time.perf_counter(); b=Bank(*args); n=b.d**2; p=b.p; b.setup()
    floors=np.empty((n,p)); gap=np.zeros(p); upper_best=np.inf; incumbent=-1
    for i in range(0,b.d,8):
        end=min(i+8,b.d); pred,err=b.bounds(i,end); delta=abs(pred-y)
        l=np.maximum(0,delta-err)**2; u=(delta+err)**2
        floors[i*b.d:end*b.d]=l; gap+=(u-l).sum(0)
        hi=u.sum(1); j=int(hi.argmin())
        if hi[j]<upper_best: upper_best=float(hi[j]); incumbent=i*b.d+j
    lower=floors.sum(1); best=incumbent; score=float(cost(b.exact([best]),y)[0]); assert lower[best]<=score+EPS<=upper_best+2*EPS
    order=np.argsort(lower,kind='stable'); order=order[order!=best]
    cols=np.argsort(-gap,kind='stable'); calls=1; pairprobes=p; stage_counts=[0]*((p+step-1)//step); before=time.perf_counter()
    for a in range(0,len(order),64):
        ids=order[a:a+64]
        if lower[ids[0]]>score+EPS: break
        ids=ids[lower[ids]<=score+EPS]
        if not len(ids): continue
        terms=floors[ids].copy(); full=False
        for stage,z in enumerate(range(0,p,step)):
            c=cols[z:z+step]; v=(exact_cols(b,ids,c)-y[c])**2
            assert np.all(terms[:,c]<=v+EPS)
            terms[:,c]=v; pairprobes+=len(ids)*len(c); stage_counts[stage]+=len(ids)
            energy=terms.sum(1)
            if z+step>=p:
                calls+=len(ids); j=int(np.argmin(energy)); val=float(energy[j]); key=int(ids[energy==val].min())
                if val<score or (val==score and key<best): score,best=val,key
                full=True; break
            keep=energy<=score+EPS; ids=ids[keep]; terms=terms[keep]
            if not len(ids): break
    return dict(best=best,score=score,calls=calls,pairprobes=pairprobes,stage_counts=stage_counts,probe_order=cols.tolist(),screen_ms=(before-t)*1000,ms=(time.perf_counter()-t)*1000)


def run(width,roots,reps):
    for root in roots:
        for episode,(args,y,hidden,noise) in enumerate(world(width,root)):
            rr={m:[] for m in MODES}
            for rep in range(reps):
                shift=(root+episode+rep)%len(MODES)
                for m in MODES[shift:]+MODES[:shift]:
                    if m.startswith('partial'): v=partial(args,y,int(m[7:]))
                    else:
                        v,_=run_one(args,y,m); v['pairprobes']=v['calls']*len(y)
                    rr[m].append(v)
            ref=rr['eager'][0]; out={}
            for m,rows in rr.items():
                assert all(v['best']==ref['best'] and abs(v['score']-ref['score'])<EPS for v in rows)
                v=rows[0].copy(); v['ms']=float(np.median([r['ms'] for r in rows])); out[m]=v
            print(json.dumps(dict(width=width,root=root,episode=episode,noise=noise,hidden=hidden,hit=hidden==ref['best'],arms=out)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--width',type=int,choices=WIDTHS,required=True); p.add_argument('--roots',type=int,nargs='+',default=ROOTS); p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args(); run(a.width,a.roots,a.repeats)
