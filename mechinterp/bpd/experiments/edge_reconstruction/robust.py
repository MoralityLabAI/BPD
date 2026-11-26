"""ER-11: contamination-aware sensing, paired replay and calibration."""
import json
import numpy as np
from noisy import world, SUP, P, B, PROBS
ROOTS=range(296,360)
RATES=(0.,.1,.3)
YMIN=-6; NY=31

def likelihood(obs,y,eps):
    d=y-obs.astype(int); valid=np.abs(d)<=1
    l=np.full(len(obs),eps/NY)
    l[valid]+=(1-eps)*PROBS[d[valid]+1]
    return l

def choose(obs,w,eps,used):
    q=np.array([np.bincount(obs[:,a],weights=w,minlength=19) for a in range(P)])
    py=np.full((P,NY),eps/NY)
    py[:,5:26]+=(1-eps)*np.stack([np.convolve(v,PROBS) for v in q])
    assert np.allclose(py.sum(axis=1),1)
    ent=-(py*np.log2(np.where(py>0,py,1))).sum(axis=1)
    return int(np.argmax(np.where(used,-np.inf,ent)))

def update(w,obs,y,eps,h):
    w=w*likelihood(obs,y,eps); z=w.sum()
    if z==0: return None,dict(hit=0,credible=0,alive=0,empty=1,truth_mass=0.,nll=None)
    w/=z; s=np.sort(w)[::-1]; ix=min(len(s)-1,np.searchsorted(np.cumsum(s),.95)); truth=w[h]
    return w,dict(hit=int(np.argmax(w)==h),credible=int(truth>0 and truth>=s[ix]),alive=int(truth>0),empty=0,truth_mass=float(truth),nll=float(-np.log2(truth)) if truth>0 else None)

def run_root(root,rate):
    obs,h,noise,order=world(root); rng=np.random.default_rng(np.random.SeedSequence([261129,root])); mask=rng.random(P)<rate
    y=np.where(mask,rng.integers(YMIN,YMIN+NY,size=P),obs[h].astype(int)+noise)
    arms={k:[] for k in ('robust_adaptive','clean_replay','matched_replay','robust_random')}
    w={k:np.full(len(SUP),1/len(SUP)) for k in arms}; used=np.zeros(P,bool)
    for step in range(B):
        a=choose(obs,w['robust_adaptive'],.1,used); used[a]=True
        for k,eps in [('robust_adaptive',.1),('clean_replay',0.),('matched_replay',rate),('robust_random',.1)]:
            aa=int(order[step]) if k=='robust_random' else a
            if w[k] is None: row=dict(hit=0,credible=0,alive=0,empty=1,truth_mass=0.,nll=None)
            else: w[k],row=update(w[k],obs[:,aa],int(y[aa]),eps,h)
            row['action']=aa; arms[k].append(row)
        assert w['robust_adaptive'][h]>0 and w['matched_replay'][h]>0
    return arms

def main():
    results={}
    for rate in RATES:
        rr=[run_root(r,rate) for r in ROOTS]; result={}
        for arm in rr[0]:
            rows=[r[arm] for r in rr]; stats={}
            for key in ('hit','credible','alive','empty','truth_mass'):
                stats[key]=np.mean([[v[key] for v in row] for row in rows],axis=0).tolist()
            stats['finite_nll_mean']=[float(np.mean(v)) if v else None for v in [[row[t]['nll'] for row in rows if row[t]['nll'] is not None] for t in range(B)]]
            stats['root_rows']=rows; result[arm]=stats
        results[str(rate)]=result
    print(json.dumps(dict(roots=[ROOTS.start,ROOTS.stop-1],n=len(ROOTS),rates=RATES,budget=B,results=results),indent=2))
if __name__=='__main__':main()
