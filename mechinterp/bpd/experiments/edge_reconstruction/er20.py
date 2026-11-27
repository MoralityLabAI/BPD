"""Prefix-reused exact rollout vs contracted global and vectorized local bounds."""
import json,time
import numpy as np
import torch
from torch import nn
from certified_crosspass import mixed_bound,BETA,EPS,refine
WIDTHS=(6,16,32,64)
ROOTS=range(32,36)
P,CAL,B,EP=48,24,12,24
class Net(nn.Module):
    def __init__(self,d):
        super().__init__(); self.rec=nn.Linear(d,d,bias=False); self.out=nn.Linear(d,d,bias=False)
    def forward(self,x):
        h=x
        for _ in range(4): h=torch.tanh(self.rec(h)+.25*x)
        return self.out(h)
def train(d,root):
    torch.manual_seed(261136+root+1000*d); teacher,model=Net(d),Net(d); opt=torch.optim.Adam(model.parameters(),lr=.015)
    for _ in range(350):
        x=torch.randn(128,d)
        with torch.no_grad(): y=teacher(x)
        loss=(model(x)-y).square().mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x=torch.randn(256,d); mse=(model(x)-teacher(x)).square().mean().item()
    return model,mse
class Bank:
    def __init__(self,model,x,c):
        self.W=model.rec.weight.detach().numpy().astype(np.float64); self.O=model.out.weight.detach().numpy().astype(np.float64)
        u,s,vh=np.linalg.svd(self.W,full_matrices=False); self.U=(u*s).T; self.V=vh
        self.r=c@self.O; self.d=len(s); self.bias=.25*x
        h1=np.tanh(x@self.W.T+self.bias); z2=h1@self.W.T+self.bias; self.h2=np.tanh(z2)
        self.first2=np.tanh(z2[None]-(h1@self.V.T).T[:,:,None]*self.U[:,None,:])
        self.z3=self.h2@self.W.T+self.bias; self.first_z3=self.first2@self.W.T+self.bias; self.proj=self.first2@self.V.T
    def exact(self,k):
        i,j=divmod(k,self.d); z=self.first_z3[i]-self.proj[i,:,j,None]*self.U[j]
        return (np.tanh(np.tanh(z)@self.W.T+self.bias)*self.r).sum(-1)
    def eager(self):
        rows=[]
        for i in range(self.d):
            z=self.first_z3[i,None]-self.proj[i].T[:,:,None]*self.U[:,None,:]
            rows.append((np.tanh(np.tanh(z)@self.W.T+self.bias)*self.r).sum(-1))
        return np.concatenate(rows)
    def singles(self):
        h3=np.tanh(self.z3); first3=np.tanh(self.first_z3); b=-(self.h2@self.V.T).T[:,:,None]*self.U[:,None,:]; second3=np.tanh(self.z3+b)
        z4=h3@self.W.T+self.bias; a4=(first3-h3)@self.W.T; b4=(second3-h3)@self.W.T
        clean=(np.tanh(z4)*self.r).sum(-1); first=(np.tanh(z4+a4)*self.r).sum(-1); second=(np.tanh(z4+b4)*self.r).sum(-1)
        add=(first[:,None]+second[None]-clean).reshape(self.d**2,-1)
        return add,self.first_z3-self.z3,b,a4,b4,z4,clean
    def bounds(self,local=False):
        add,a,b,a4,b4,z4,clean=self.singles(); q=abs(self.r)@abs(self.W); dp=abs((self.first2-self.h2)@self.V.T).transpose(0,2,1)
        if not local:
            def contract(x,y,w): return ((abs(x)*w).transpose(1,0,2)@abs(y).transpose(1,2,0)).transpose(1,2,0)
            g=dp*(q@abs(self.U).T).T[None]+BETA*contract(a,b,q)+BETA*contract(a4,b4,abs(self.r))
            return add,g.reshape(self.d**2,-1)+EPS
        rows=[]
        for i in range(self.d):
            cross=dp[i,:,:,None]*abs(self.U[:,None,:]); r3=mixed_bound(self.z3,a[i],b,cross,True); r4=mixed_bound(z4,a4[i],b4,r3@abs(self.W).T,True)
            rows.append((r4*abs(self.r)).sum(-1)+EPS)
        return add,np.concatenate(rows)
def workload(model,x,c,cases,mode):
    start=time.perf_counter(); bank=Bank(model,x,c); cache={}; calls=0
    if mode=='eager': pred=bank.eager()
    else: add,err=bank.bounds(local=mode=='local')
    build=(time.perf_counter()-start)*1000; decisions=[]
    for a,y in cases:
        if mode=='eager': best=int(np.argmin(((pred[:,a]-y)**2).sum(1)))
        else:
            def query(k):
                if k not in cache: cache[k]=bank.exact(k)
                return cache[k][a]
            best,ks=refine(add[:,a],err[:,a],y,query); calls+=len(ks)
        decisions.append(best)
    return dict(ms=(time.perf_counter()-start)*1000,build_ms=build,decisions=decisions,logical_calls=calls,rows=bank.d**2 if mode=='eager' else len(cache))
def run():
    torch.set_num_threads(1)
    for d in WIDTHS:
        for root in ROOTS:
            model,mse=train(d,root); rng=np.random.default_rng(np.random.SeedSequence([261137,d,root])); x,c=rng.standard_normal((CAL+P,d)),rng.standard_normal((CAL+P,d))
            cal=Bank(model,x[:CAL],c[:CAL]); sigma=max(1e-5,.1*np.sqrt(np.mean((cal.eager()-cal.singles()[-1])**2)))
            x,c=x[CAL:],c[CAL:]; bank=Bank(model,x,c); exact=bank.eager()
            for k in rng.choice(d*d,min(16,d*d),replace=False):
                i,j=divmod(int(k),d); h=x
                for t in range(4):
                    w=bank.W
                    if t==1: w=w-np.outer(bank.U[i],bank.V[i])
                    if t==2: w=w-np.outer(bank.U[j],bank.V[j])
                    h=np.tanh(h@w.T+.25*x)
                assert np.allclose((h@bank.O.T*c).sum(-1),exact[k],atol=1e-10,rtol=0)
            add,g=bank.bounds(); _,l=bank.bounds(local=True); error=abs(exact-add)
            assert np.all(error<=g) and np.all(error<=l) and np.all(l<=g+EPS)
            cases=[]; hidden=[]
            for _ in range(EP):
                k=int(rng.integers(d*d)); a=rng.permutation(P)[:B]; cases.append((a,exact[k,a]+rng.normal(0,sigma,B))); hidden.append(k)
            groups={}
            for n in (1,8,EP):
                runs=[]
                for rep in range(3):
                    modes=['eager','global','local']; shift=(rep+root)%3; rr={m:workload(model,x,c,cases[:n],m) for m in modes[shift:]+modes[:shift]}
                    assert rr['eager']['decisions']==rr['global']['decisions']==rr['local']['decisions']; runs.append(rr)
                groups[n]={m:dict(ms=float(np.median([r[m]['ms'] for r in runs])),build_ms=float(np.median([r[m]['build_ms'] for r in runs])),rows=runs[0][m]['rows'],logical_calls=runs[0][m]['logical_calls'],hits=sum(a==b for a,b in zip(hidden[:n],runs[0][m]['decisions']))) for m in runs[0]}
            print(json.dumps(dict(d=d,root=root,mse=mse,sigma=float(sigma),bound_checks=2*error.size,mean_error=float(error.mean()),mean_global=float(g.mean()),mean_local=float(l.mean()),groups=groups)),flush=True)
if __name__=='__main__': run()
