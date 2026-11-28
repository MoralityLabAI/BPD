"""ER-19: end-to-end MAP workload with reusable lazy joint rows."""
import json,time
import numpy as np
import torch
from certified_crosspass import Model,ROOTS,CAL,P,EP,B,D,train,refine


def eager(model,x,c,cases):
    t=time.perf_counter(); exact=np.stack([model.pair(x,c,k) for k in range(D*D)])
    build=time.perf_counter()-t
    hits=[int(np.argmin(np.sum((exact[:,a]-y)**2,axis=1))) for a,y in cases]
    return dict(wall_ms=(time.perf_counter()-t)*1000,build_ms=build*1000,joint_rows=D*D,base_rollouts=0,predictions=hits)


def lazy(model,x,c,cases):
    t=time.perf_counter(); add,_,bound,_,_=model.envelopes(x,c); build=time.perf_counter()-t
    cache={}; hits=[]; calls=0; cache_counts=[]
    for a,y in cases:
        def query(k):
            if k not in cache: cache[k]=model.pair(x,c,k)
            return cache[k][a]
        best,requested=refine(add[:,a],bound[:,a],y,query); hits.append(best); calls+=len(requested); cache_counts.append(len(cache))
    return dict(wall_ms=(time.perf_counter()-t)*1000,build_ms=build*1000,joint_rows=len(cache),base_rollouts=1+2*D,predictions=hits,logical_calls=calls,cache_counts=cache_counts)


def run():
    torch.set_num_threads(1); rows=[]
    for root in ROOTS:
        trained,_=train(root); model=Model(trained)
        torch.manual_seed(261134+root)
        x,c=[v.numpy().astype(np.float64) for v in (torch.randn(CAL+P,D),torch.randn(CAL+P,D))]
        clean,_=model.rollout(x[:CAL],c[:CAL]); cal=np.stack([model.pair(x[:CAL],c[:CAL],k) for k in range(D*D)])
        sigma=max(1e-5,.1*float(np.sqrt(np.mean((cal-clean)**2))))
        x,c=x[CAL:],c[CAL:]; exact=np.stack([model.pair(x,c,k) for k in range(D*D)])
        rng=np.random.default_rng(np.random.SeedSequence([261135,root])); rng.permutation(D*D)
        cases=[]
        for _ in range(EP):
            h=int(rng.integers(D*D)); a=rng.permutation(P)[:B]; cases.append((a,exact[h,a]+rng.normal(0,sigma,B)))
        groups={}
        for count in (1,8,48):
            rr=[]
            for rep in range(3):
                runs={}
                for fn in ((eager,lazy) if (rep+root)%2==0 else (lazy,eager)):
                    runs[fn.__name__]=fn(model,x,c,cases[:count])
                assert runs['eager']['predictions']==runs['lazy']['predictions']
                rr.append(runs)
            groups[str(count)]={k:dict(wall_ms=float(np.median([r[k]['wall_ms'] for r in rr])),build_ms=float(np.median([r[k]['build_ms'] for r in rr])),joint_rows=rr[0][k]['joint_rows'],base_rollouts=rr[0][k]['base_rollouts']) for k in ('eager','lazy')}
            groups[str(count)]['lazy']['cache_counts']=rr[0]['lazy']['cache_counts']
        rows.append(dict(root=root,groups=groups))
    summary={}
    for count in ('1','8','48'):
        groups=[r['groups'][count] for r in rows]
        summary[count]={k:dict(median_ms=float(np.median([g[k]['wall_ms'] for g in groups])),median_build_ms=float(np.median([g[k]['build_ms'] for g in groups])),median_joint_rows=float(np.median([g[k]['joint_rows'] for g in groups])),joint_rows=[g[k]['joint_rows'] for g in groups]) for k in ('eager','lazy')}
        summary[count]['median_paired_ratio_eager_over_lazy']=float(np.median([g['eager']['wall_ms']/g['lazy']['wall_ms'] for g in groups]))
    return dict(roots=rows,summary=summary,scope='full 48-probe prediction rows retained; timings exclude training and environment generation, include bounds and decisions')

if __name__=='__main__': print(json.dumps(run(),indent=2))
