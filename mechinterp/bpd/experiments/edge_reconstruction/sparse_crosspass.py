"""ER-14: matched joint-rollout budget, fresh trained roots; native path optional."""
import argparse
import json
import time
import numpy as np
import torch
from neural_bridge import D, T, train

P, B, CAL, EP = 48, 12, 24, 48
ROOTS = range(8, 16)
BUDGETS = (0, 4, 8, 16, 36)

class Bank:
    def __init__(self, model, native=False):
        self.model = model
        self.native = native
        if native:
            from mechinterp.bpd.components import TiedComponentLinear, ComponentIntervention
            from mechinterp.bpd.config import AblationConfig
            self.layer = TiedComponentLinear(model.rec, D, max_passes=T,
                ablation=AblationConfig(mode='zero', sphere_rescale=False), init_svd=True)
            self.intervention = ComponentIntervention
            self.V, self.U = self.layer.V, self.layer.U
        else:
            u, s, vh = torch.linalg.svd(model.rec.weight.double(), full_matrices=False)
            self.V, self.U = (vh.T*s.sqrt()).float(), (s.sqrt()[:,None]*u.T).float()
        self.rollouts = 0

    @torch.no_grad()
    def roll(self, x, c, faults=()):
        self.rollouts += 1
        h = x
        states = [h]
        if self.native:
            self.layer.reset_pass_state()
        for t in range(T):
            ids = tuple(i for tau,i in faults if tau == t)
            if self.native:
                self.layer.intervention = self.intervention(ids, frozenset([t]), 'ablate') if ids else None
                z = self.layer(h)
            else:
                z = self.model.rec(h)
                if ids:
                    z = z - (h @ self.V[:,list(ids)]) @ self.U[list(ids),:]
            h = torch.tanh(z + .25*x)
            states.append(h)
        return (self.model.out(h)*c).sum(-1).numpy(), states

    @torch.no_grad()
    def base(self, x, c):
        clean, states = self.roll(x,c)
        first = [self.roll(x,c,((1,i),)) for i in range(D)]
        second = [self.roll(x,c,((2,j),)) for j in range(D)]
        a = np.stack([v[0] for v in first]); b = np.stack([v[0] for v in second])
        additive = (a[:,None,:]+b[None,:,:]-clean).reshape(D*D,-1)
        # Calibration scores use only the 13 base rollouts. No pair outcomes.
        delta = torch.stack([v[1][2]-states[2] for v in first])
        cross = -(delta @ self.V).permute(0,2,1)[:,:,:,None]*self.U[None,:,None,:]
        overlap = cross.square().mean((2,3)).sqrt().numpy().ravel()
        # Approximate cross write's downstream effect using clean tangent maps.
        g3 = (c @ self.model.out.weight)*(1-states[4].square()) @ self.model.rec.weight
        g2 = g3*(1-states[3].square())
        tangent = (cross*g2[None,None,:,:]).sum(-1).square().mean(-1).sqrt().numpy().ravel()
        marginal = np.outer(np.sqrt(np.mean((a-clean)**2,axis=1)),np.sqrt(np.mean((b-clean)**2,axis=1))).ravel()
        return clean,additive,dict(overlap=overlap,tangent=tangent,marginal=marginal)

    def pair(self,x,c,k):
        return self.roll(x,c,((1,k//D),(2,k%D)))[0]


def metrics(pred, exact, cases, sigma):
    hits=[]; nll=[]; coverage=[]
    for h,probes,y in cases:
        logp=-.5*((pred[:,probes]-y)/sigma)**2
        logp=np.cumsum(logp,axis=1); p=np.exp(logp-logp.max(axis=0)); p/=p.sum(axis=0)
        hits.append(p.argmax(axis=0)==h)
        nll.append(-np.log2(np.maximum(p[h],1e-300)))
        s=np.sort(p,axis=0)[::-1]; ix=(np.cumsum(s,axis=0)<.95).sum(axis=0); cutoff=s[np.minimum(ix,len(s)-1),np.arange(B)]
        coverage.append(p[h]>=cutoff)
    return dict(hits=np.sum(hits,axis=0).tolist(),nll=np.mean(nll,axis=0).tolist(),coverage=np.sum(coverage,axis=0).tolist(),rmse=float(np.sqrt(np.mean((pred-exact)**2))))


def main(native=False):
    torch.set_num_threads(1); rows=[]
    for root in ROOTS:
        model,mse=train(root); torch.manual_seed(261134+root)
        x,c=torch.randn(CAL+P,D),torch.randn(CAL+P,D)
        oracle=Bank(model,native); clean,add,scores=oracle.base(x[:CAL],c[:CAL]); cal_exact=np.stack([oracle.pair(x[:CAL],c[:CAL],k) for k in range(D*D)])
        sigma=max(1e-5,.1*float(np.sqrt(np.mean((cal_exact-clean)**2))))
        # The exact calibration rank is a diagnostic; it uses 36 extra joint rollouts.
        scores['cal_oracle']=np.sqrt(np.mean((cal_exact-add)**2,axis=1))
        _,_,_=oracle.base(x[CAL:],c[CAL:]); exact=np.stack([oracle.pair(x[CAL:],c[CAL:],k) for k in range(D*D)])
        rng=np.random.default_rng(np.random.SeedSequence([261135,root])); order=rng.permutation(D*D)
        cases=[]
        for e in range(EP):
            h=int(rng.integers(D*D)); probes=rng.permutation(P)[:B]; y=exact[h,probes]+rng.normal(0,sigma,B); cases.append((h,probes,y))
        rankings={k:np.argsort(-v,kind='stable') for k,v in scores.items()}; rankings['random']=order
        arm={}
        for name,ranking in rankings.items():
            for k in BUDGETS:
                bank=Bank(model,native); t0=time.perf_counter(); _,pred,_=bank.base(x[CAL:],c[CAL:]); selected=ranking[:k]
                for pair in selected: pred[pair]=bank.pair(x[CAL:],c[CAL:],int(pair))
                time_ms=(time.perf_counter()-t0)*1000
                assert bank.rollouts==13+k
                if k==36: assert np.allclose(pred,exact,rtol=0,atol=1e-6)
                arm[f'{name}/{k}']=dict(**metrics(pred,exact,cases,sigma),selected=selected.tolist(),test_rollouts=bank.rollouts,cal_rollouts=49 if name=='cal_oracle' else 13,build_ms=time_ms)
        rows.append(dict(root=root,teacher_mse=mse,sigma=sigma,arms=arm))
    summary={}
    for name in rows[0]['arms']:
        vals=[r['arms'][name] for r in rows]; summary[name]=dict(hits=np.sum([v['hits'] for v in vals],axis=0).tolist(),coverage=np.sum([v['coverage'] for v in vals],axis=0).tolist(),nll=np.mean([v['nll'] for v in vals],axis=0).tolist(),median_rmse=float(np.median([v['rmse'] for v in vals])),median_build_ms=float(np.median([v['build_ms'] for v in vals])),test_rollouts=vals[0]['test_rollouts'],cal_rollouts=vals[0]['cal_rollouts'])
    print(json.dumps(dict(native=native,roots=rows,summary=summary,episodes_per_root=EP,probes=B),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--native',action='store_true'); main(p.parse_args().native)
