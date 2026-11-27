"""Root-disjoint learned entropy scheduler and exact cache replay."""
import json,time
from itertools import combinations
import numpy as np
import torch
from torch import nn
N=6; E=30; P=64; B=12; SEED=261127
SUP=np.array(list(combinations(range(E),3)))
EDGES=[(i,j) for i in range(N) for j in range(N) if i!=j]
def world(root):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,root]))
    x=rng.choice([-1,1],size=(P,N)); c=rng.choice([-1,1],size=(P,N))
    f=np.stack([c[:,i]*x[:,j] for i,j in EDGES],axis=1).astype(np.int8)
    return f,int(rng.integers(len(SUP))),rng.permutation(P)
def predict(f,ids): return f[:,SUP[ids]].sum(axis=2).T.astype(np.int8)
def entropy(obs):
    p=np.stack([(obs==v).sum(axis=0) for v in (-3,-1,1,3)])/len(obs)
    return -(p*np.log2(np.maximum(p,1e-300))).sum(axis=0)
def features(f,ids):
    p=np.bincount(SUP[ids].ravel(),minlength=E)/len(ids); pos=f==1; v=p*(1-p)
    return np.column_stack([np.sort(pos*p,axis=1),np.sort((~pos)*p,axis=1),np.full(P,np.log2(len(ids))/12),pos.mean(axis=1),f@p,pos@v,(~pos)@v]).astype(np.float32)
class LoopEstimator(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=nn.Linear(65,32); self.step=nn.Linear(64,32); self.out=nn.Linear(32,1)
    def forward(self,x):
        z=torch.tanh(self.enc(x)); h=z
        for _ in range(3): h=h+0.5*torch.tanh(self.step(torch.cat([h,z],dim=-1)))
        return 2*torch.sigmoid(self.out(h).squeeze(-1))
def dataset(roots):
    xs=[]; ys=[]
    for root in roots:
        f,h,order=world(root); obs=predict(f,np.arange(len(SUP)))
        for policy in ('random','exact'):
            ids=np.arange(len(SUP)); used=np.zeros(P,bool)
            for step in range(B):
                scores=entropy(obs[ids]); xs.append(features(f,ids)[~used]); ys.append(scores[~used])
                a=order[step] if policy=='random' else int(np.argmax(np.where(used,-np.inf,scores)))
                used[a]=True; ids=ids[obs[ids,a]==obs[h,a]]
                if len(ids)==1: break
    return torch.from_numpy(np.concatenate(xs)),torch.from_numpy(np.concatenate(ys).astype(np.float32))
def train():
    torch.manual_seed(42); torch.set_num_threads(1)
    x,y=dataset(range(64)); xd,yd=dataset(range(64,80)); m=LoopEstimator(); opt=torch.optim.AdamW(m.parameters(),lr=.002)
    best=float('inf'); best_state=None
    for epoch in range(12):
        for ix in torch.randperm(len(x)).split(512):
            loss=(m(x[ix])-y[ix]).square().mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): val=(m(xd)-yd).square().mean().item()
        if val<best: best=val; best_state={k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    return m,dict(train_examples=len(x),dev_examples=len(xd),dev_mse=best,parameters=sum(p.numel() for p in m.parameters()))
def trajectory(root,policy,model=None,cache=True):
    f,h,order=world(root); ids=np.arange(len(SUP)); used=np.zeros(P,bool)
    t=time.perf_counter(); obs=predict(f,ids) if cache else None; cache_build=time.perf_counter()-t
    truth=predict(f,np.array([h]))[0]; sizes=[]; actions=[]; score_time=0.; filter_time=0.
    static=entropy(obs if cache else predict(f,ids)) if policy=='static' else None
    for step in range(B):
        t=time.perf_counter()
        if policy=='random': a=int(order[step])
        elif policy in ('exact','static'):
            score=entropy(obs[ids] if cache else predict(f,ids)) if policy=='exact' else static
            a=int(np.argmax(np.where(used,-np.inf,score)))
        else:
            z=features(f,ids)
            if policy=='learned':
                with torch.no_grad(): score=model(torch.from_numpy(z)).numpy()
            elif policy=='variance': score=z[:,-2:].min(axis=1)
            else: raise ValueError(policy)
            a=int(np.argmax(np.where(used,-np.inf,score)))
        score_time+=time.perf_counter()-t
        t=time.perf_counter(); used[a]=True
        vals=obs[ids,a] if cache else f[a,SUP[ids]].sum(axis=1)
        ids=ids[vals==truth[a]]; assert h in ids
        filter_time+=time.perf_counter()-t; actions.append(a); sizes.append(len(ids))
    return dict(sizes=sizes,actions=actions,cache_build=cache_build,score_time=score_time,filter_time=filter_time)
def run():
    m,training=train(); results={}; cache_results=[]; roots=range(80,144)
    for policy in ('random','static','variance','learned','exact'):
        rows=[trajectory(r,policy,m) for r in roots]; sizes=np.array([x['sizes'] for x in rows]); results[policy]=dict(unique=(sizes==1).sum(axis=0).tolist(),root_sizes=sizes.tolist(),median_remaining=np.median(sizes,axis=0).tolist(),median_score_ms=1000*np.median([r['score_time'] for r in rows]))
    for root in roots:
        if root%2==0: a,b=trajectory(root,'exact',cache=True),trajectory(root,'exact',cache=False)
        else: b,a=trajectory(root,'exact',cache=False),trajectory(root,'exact',cache=True)
        assert a['actions']==b['actions'] and a['sizes']==b['sizes']
        cache_results.append([1000*sum(a[k] for k in ('cache_build','score_time','filter_time')),1000*sum(b[k] for k in ('cache_build','score_time','filter_time'))])
    out=dict(seed=SEED,train_roots=[0,63],development_roots=[64,79],test_roots=[80,143],training=training,arms=results,cache=dict(paired_replays=len(cache_results),raw_bytes=len(SUP)*P,median_total_ms=np.median(cache_results,axis=0).tolist(),root_times_ms=cache_results))
    torch.save(m.state_dict(),'learned.pt'); print(json.dumps(out,indent=2))
if __name__=='__main__': run()
