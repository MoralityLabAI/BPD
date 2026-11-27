"""ER-08: discrete noisy sensing, exact Bayesian posterior, matched probes."""
import json
from itertools import combinations
import numpy as np
N=6; P=64; B=16; SEED=261128
EDGES=[(i,j) for i in range(N) for j in range(N) if i!=j]
SUP=np.array(list(combinations(range(len(EDGES)),3)))
NOISE=np.array([-1,0,1]); PROBS=np.array([.15,.7,.15]); OFFSET=9

def world(root):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,root]))
    x=rng.choice([-1,1],size=(P,N)); c=rng.choice([-1,1],size=(P,N))
    weights=rng.integers(1,4,size=len(EDGES)); f=np.stack([c[:,i]*x[:,j] for i,j in EDGES],axis=1)*weights
    obs=(f[:,SUP].sum(axis=2).T+OFFSET).astype(np.int8)
    h=int(rng.integers(len(SUP))); noise=rng.choice(NOISE,size=P,p=PROBS)
    return obs,h,noise,rng.permutation(P)

def scores(obs,w):
    p=np.array([np.bincount(obs[:,a],weights=w,minlength=19) for a in range(P)])
    py=np.stack([np.convolve(v,PROBS) for v in p]); mask=py>0
    return -(py*np.log2(np.where(mask,py,1))).sum(axis=1)

def run_root(root,policy):
    obs,h,noise,order=world(root); w=np.full(len(SUP),1/len(SUP)); used=np.zeros(P,bool); hard=np.ones(len(SUP),bool)
    static=scores(obs,w); rows=[]
    for step in range(B):
        if policy=='random': a=int(order[step])
        elif policy=='static': a=int(np.argmax(np.where(used,-np.inf,static)))
        else: a=int(np.argmax(np.where(used,-np.inf,scores(obs,w))))
        used[a]=True; y=int(obs[h,a]+noise[a]); delta=y-obs[:,a].astype(int)
        likelihood=np.zeros(len(SUP)); valid=np.abs(delta)<=1; likelihood[valid]=PROBS[delta[valid]+1]
        w*=likelihood; z=w.sum(); assert z>0 and w[h]>0; w/=z
        assert abs(w.sum()-1)<1e-12
        hard &= delta==0
        s=np.sort(w)[::-1]; threshold=s[np.searchsorted(np.cumsum(s),.95)]; credible=w>=threshold
        rows.append(dict(action=a,hit=int(np.argmax(w)==h),truth_mass=float(w[h]),nll_bits=float(-np.log2(w[h])),credible=int(credible[h]),credible_size=int(credible.sum()),hard_survives=int(hard[h]),hard_empty=int(not hard.any()),positive=int((w>0).sum())))
    return rows

def main():
    arms={}; roots=range(232,296)
    for p in ('random','static','adaptive'):
        rr=[run_root(root,p) for root in roots]; arms[p]={'root_rows':rr}
        for key in ('hit','truth_mass','nll_bits','credible','credible_size','hard_survives','hard_empty','positive'):
            arms[p][key+'_mean']=np.mean([[x[key] for x in row] for row in rr],axis=0).tolist()
    obs,h,noise,_=world(232); w=np.full(len(SUP),1/len(SUP)); a=0
    q=np.convolve(np.bincount(obs[:,a],weights=w,minlength=19),PROBS)
    hnoise=-sum(PROBS*np.log2(PROBS)); explicit=0.
    for d,p in zip(NOISE,PROBS): explicit+=np.sum(w*p*np.log2(p/q[obs[:,a]+d+1]))
    assert abs(scores(obs,w)[a]-hnoise-explicit)<1e-12
    print(json.dumps(dict(seed=SEED,roots=[232,295],hypotheses=len(SUP),probes=P,budget=B,noise=PROBS.tolist(),arms=arms,mi_check=explicit),indent=2))
if __name__=='__main__': main()
