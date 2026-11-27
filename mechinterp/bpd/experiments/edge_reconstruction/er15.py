"""ER-15: fresh model-root replay of implicit cross-pass response banks."""
import json,time
import numpy as np
import torch
from sparse_crosspass import Bank,D,CAL,P,B,EP,train
from sparse_store import SparseBank


def posterior(query,cases,sigma):
    hits=[]; nll=[]; coverage=[]
    for h,probes,y in cases:
        logp=np.cumsum(-.5*((query(probes)-y)/sigma)**2,axis=1)
        p=np.exp(logp-logp.max(axis=0)); p/=p.sum(axis=0)
        hits.append(p.argmax(axis=0)==h)
        nll.append(-np.log2(np.maximum(p[h],1e-300)))
        s=np.sort(p,axis=0)[::-1]
        ix=(np.cumsum(s,axis=0)<.95).sum(axis=0)
        coverage.append(p[h]>=s[np.minimum(ix,len(s)-1),np.arange(B)])
    return dict(hits=np.sum(hits,axis=0).tolist(),nll=np.mean(nll,axis=0).tolist(),coverage=np.sum(coverage,axis=0).tolist())


def run():
    torch.set_num_threads(1); rows=[]; checks=0
    for root in range(16,24):
        model,_=train(root); torch.manual_seed(261134+root)
        x,c=torch.randn(CAL+P,D),torch.randn(CAL+P,D)
        cal=Bank(model); clean,_,scores=cal.base(x[:CAL],c[:CAL])
        cal_exact=np.stack([cal.pair(x[:CAL],c[:CAL],k) for k in range(D*D)])
        sigma=max(1e-5,.1*float(np.sqrt(np.mean((cal_exact-clean)**2))))
        oracle=Bank(model); exact=np.stack([oracle.pair(x[CAL:],c[CAL:],k) for k in range(D*D)])
        rng=np.random.default_rng(np.random.SeedSequence([261135,root])); order=rng.permutation(D*D)
        cases=[]
        for _ in range(EP):
            h=int(rng.integers(D*D)); probes=rng.permutation(P)[:B]
            cases.append((h,probes,exact[h,probes]+rng.normal(0,sigma,B)))
        out={}
        for name,ranking,k in [('random',order,16),('tangent',np.argsort(-scores['tangent'],kind='stable'),16),('full',order,36),('additive',order,0)]:
            bank=Bank(model); t=time.perf_counter()
            clean,_=bank.roll(x[CAL:],c[CAL:])
            first=np.stack([bank.roll(x[CAL:],c[CAL:],((1,i),))[0] for i in range(D)])
            second=np.stack([bank.roll(x[CAL:],c[CAL:],((2,j),))[0] for j in range(D)])
            ids=ranking[:k].astype(np.int64)
            overrides=np.stack([bank.pair(x[CAL:],c[CAL:],int(j)) for j in ids]) if k else np.empty((0,P),dtype=np.float32)
            store=SparseBank(clean,first,second,ids,overrides); build_ms=1000*(time.perf_counter()-t)
            assert bank.rollouts==13+k
            dense=store.dense()
            if k==36: assert np.array_equal(dense,exact)
            for _,probes,_ in cases:
                assert np.array_equal(store.query(probes),dense[:,probes]); checks+=1
            times={}; results={}
            for mode in (('sparse','dense') if root%2 else ('dense','sparse')):
                query=store.query if mode=='sparse' else lambda a: dense[:,a]
                t=time.perf_counter(); results[mode]=posterior(query,cases,sigma); times[mode]=1000*(time.perf_counter()-t)
            assert results['sparse']==results['dense']
            out[name]=dict(**results['sparse'],rollouts=bank.rollouts,payload=store.payload,dense_payload=dense.nbytes,build_ms=build_ms,posterior_ms=times)
        rows.append(dict(root=root,arms=out))
    summary={}
    for name in rows[0]['arms']:
        rr=[r['arms'][name] for r in rows]
        summary[name]=dict(hits=np.sum([r['hits'] for r in rr],axis=0).tolist(),coverage=np.sum([r['coverage'] for r in rr],axis=0).tolist(),nll=np.mean([r['nll'] for r in rr],axis=0).tolist(),payload=rr[0]['payload'],dense_payload=rr[0]['dense_payload'],median_posterior_ms={m:float(np.median([r['posterior_ms'][m] for r in rr])) for m in ('dense','sparse')})
    return dict(roots=rows,summary=summary,exact_query_checks=checks)

if __name__=='__main__': print(json.dumps(run(),indent=2))
