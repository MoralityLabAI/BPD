"""ER-13: trained tied recurrence, injected component faults, native BPD bridge."""
import argparse,json
import numpy as np
import torch
from torch import nn
D,T,P,B=6,4,32,12

class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.rec=nn.Linear(D,D,bias=False); self.out=nn.Linear(D,D,bias=False)
    def forward(self,x):
        h=x
        for _ in range(T): h=torch.tanh(self.rec(h)+.25*x)
        return self.out(h)

def train(root):
    torch.manual_seed(261130+root); teacher=Model(); model=Model(); opt=torch.optim.Adam(model.parameters(),lr=.015)
    for _ in range(350):
        x=torch.randn(128,D)
        with torch.no_grad(): y=teacher(x)
        loss=(model(x)-y).square().mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x=torch.randn(256,D); mse=(model(x)-teacher(x)).square().mean().item()
    return model.eval(),mse

@torch.no_grad()
def bank(model,x,c,native=False):
    if native:
        from mechinterp.bpd.components import TiedComponentLinear,ComponentIntervention
        from mechinterp.bpd.config import AblationConfig
        layer=TiedComponentLinear(model.rec,D,max_passes=T,ablation=AblationConfig(mode='zero',sphere_rescale=False),init_svd=True)
        V,U=layer.V,layer.U
    else:
        u,s,vh=torch.linalg.svd(model.rec.weight.double(),full_matrices=False); V=(vh.T*s.sqrt()).float(); U=(s.sqrt()[:,None]*u.T).float()
    def rollout(faults):
        h=x
        if native: layer.reset_pass_state()
        for t in range(T):
            ids=tuple(i for tau,i in faults if tau==t)
            if native:
                layer.intervention=ComponentIntervention(ids,frozenset([t]),'ablate') if ids else None
                z=layer(h)
            else:
                z=model.rec(h)
                if ids: z=z-(h@V[:,list(ids)])@U[list(ids),:]
            h=torch.tanh(z+.25*x)
        return (model.out(h)*c).sum(dim=-1).numpy()
    clean=rollout([]); assert np.max(np.abs(clean-(model(x)*c).sum(-1).numpy()))<1e-6
    single={(t,i):rollout([(t,i)]) for t in (1,2) for i in range(D)}
    exact=[]; add=[]
    for i in range(D):
        for j in range(D):
            a,b=(1,i),(2,j); exact.append(rollout([a,b])); add.append(single[a]+single[b]-clean)
    return np.array(exact),np.array(add),clean

def evaluate(exact,add,sigma,root):
    rng=np.random.default_rng(np.random.SeedSequence([261131,root])); rows=[]
    for episode in range(24):
        h=int(rng.integers(len(exact))); probes=rng.permutation(P)[:B]; y=exact[h,probes]+rng.normal(0,sigma,B)
        result={}
        for name,pred in [('exact',exact),('additive',add)]:
            logp=np.zeros(len(pred)); hits=[]; nll=[]
            for a,obs in zip(probes,y):
                logp-=.5*((pred[:,a]-obs)/sigma)**2; p=np.exp(logp-logp.max()); p/=p.sum()
                hits.append(int(p.argmax()==h)); nll.append(float(-np.log2(max(p[h],1e-300))))
            result[name]=dict(hit=hits,nll=nll)
        rows.append(result)
    return rows

def main(native=False):
    torch.set_num_threads(1); rows=[]
    for root in range(8):
        model,mse=train(root); torch.manual_seed(261132+root); x=torch.randn(P+16,D); c=torch.randn(P+16,D)
        exact,add,clean=bank(model,x,c,native)
        if native:
            er,ar,cr=bank(model,x,c,False); assert np.max(abs(er-exact))<1e-6 and np.max(abs(ar-add))<1e-6
        sigma=max(1e-5,.1*float(np.sqrt(np.mean((exact[:,:16]-clean[:16])**2))))
        rows.append(dict(root=root,test_mse=mse,sigma=sigma,additive_rmse=float(np.sqrt(np.mean((exact[:,16:]-add[:,16:])**2))),episodes=evaluate(exact[:,16:],add[:,16:],sigma,root)))
    out=dict(native=native,roots=rows,summary={})
    for name in ('exact','additive'):
        out['summary'][name]={key:np.mean([e[name][key] for r in rows for e in r['episodes']],axis=0).tolist() for key in ('hit','nll')}
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--native',action='store_true'); main(p.parse_args().native)
