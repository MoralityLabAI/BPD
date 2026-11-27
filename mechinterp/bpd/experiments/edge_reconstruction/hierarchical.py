"""ER-12: joint support/rate posterior on fixed ER-11 probe histories."""
import json
import numpy as np
from noisy import world,SUP,P,B,PROBS
from robust import ROOTS,RATES,YMIN,NY,likelihood
GRID=np.array([.01,.05,.1,.2,.3,.5])
LEVELS=np.stack([(1-GRID)*PROBS[1]+GRID/NY,(1-GRID)*PROBS[0]+GRID/NY,GRID/NY],axis=1)
LOG=np.log(LEVELS)

def run():
    receipt=json.load(open('robust.json')); results={}; maxerr=0.; steps=0
    for rate in RATES:
        rows=[]
        for root,history in zip(ROOTS,receipt['results'][str(rate)]['robust_adaptive']['root_rows']):
            obs,h,noise,_=world(root); rng=np.random.default_rng(np.random.SeedSequence([261129,root])); mask=rng.random(P)<rate
            y=np.where(mask,rng.integers(YMIN,YMIN+NY,size=P),obs[h].astype(int)+noise)
            c0=np.zeros(len(SUP),dtype=np.uint8); c1=c0.copy(); w=np.full((len(GRID),len(SUP)),1/(len(GRID)*len(SUP))); rr=[]
            for t,old in enumerate(history,1):
                a=old['action']; d=np.abs(y[a]-obs[:,a].astype(int)); c0+=d==0; c1+=d==1
                w*=np.stack([likelihood(obs[:,a],int(y[a]),e) for e in GRID]); w/=w.sum()
                logw=LOG[:,0,None]*c0+LOG[:,1,None]*c1+LOG[:,2,None]*(t-c0.astype(int)-c1.astype(int)); v=np.exp(logw-logw.max()); v/=v.sum()
                err=float(np.max(np.abs(v-w))); assert err<1e-12; maxerr=max(maxerr,err); steps+=1
                p=v.sum(axis=0); rates=v.sum(axis=1); s=np.sort(p)[::-1]; cut=s[min(len(s)-1,np.searchsorted(np.cumsum(s),.95))]
                rr.append(dict(hit=int(np.argmax(p)==h),credible=int(p[h]>=cut),nll=float(-np.log2(p[h])),truth_mass=float(p[h]),rate_mean=float(rates@GRID),rate_posterior=rates.tolist()))
            rows.append(rr)
        results[str(rate)]={k:np.mean([[r[k] for r in row] for row in rows],axis=0).tolist() for k in ('hit','credible','nll','truth_mass','rate_mean','rate_posterior')}; results[str(rate)]['root_rows']=rows
    return dict(grid=GRID.tolist(),roots=[ROOTS.start,ROOTS.stop-1],steps=steps,max_abs_error=maxerr,count_bytes=2*len(SUP),joint_float64_bytes=8*len(GRID)*len(SUP),results=results)
if __name__=='__main__':print(json.dumps(run(),indent=2))
