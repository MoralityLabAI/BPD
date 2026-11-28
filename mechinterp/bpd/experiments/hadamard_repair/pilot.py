"""HAD-03 development pilot. Odd sequence lengths; exact integer verifier."""
import argparse, copy, json, time
import numpy as np
import torch
from torch import nn

def mechanics(x):
    b,_,m=x.shape; t=np.arange(1,m//2+1); i=np.arange(m)
    p=x[:,:,(i[:,None]+t)%m]; q=x[:,:,(i[:,None]-t)%m]
    r=(x[...,None]*p).sum((1,2)); d=(-2*x[...,None]*(p+q)).reshape(b,4*m,-1)
    e=(r*r).sum(-1); nxt=((r[:,None,:]+d)**2).sum(-1)
    f=np.stack([x.reshape(b,-1),(nxt-e[:,None])/(16*m),(d*d).sum(-1)/(16*m),np.broadcast_to(e[:,None]/(4*m*m),nxt.shape),np.repeat(x.sum(-1),m,axis=1)/m],-1).astype('float32')
    return r,d,e,nxt,torch.from_numpy(f)

def flip(x,a):
    y=x.copy(); y.reshape(len(y),-1)[np.arange(len(y)),a]*=-1; return y

def oracle(x):
    b,_,m=x.shape; r,d,e,nxt,f=mechanics(x); n=4*m; a=np.arange(n); p=a%m
    diff=(p[:,None]-p[None,:])%m; dist=np.minimum(diff,m-diff)
    same=a[:,None]//m==a[None,:]//m; t=np.arange(1,m//2+1)
    c=4*(same[...,None] & (dist[...,None]==t))[None]*x.reshape(b,n,1,1)*x.reshape(b,1,n,1)
    rr=r[:,None,None,:]+d[:,:,None,:]+d[:,None,:,:]+c
    v=(rr*rr).sum(-1); v[:,a,a]=10**9
    best=np.minimum(nxt,v.min(-1)); return f,torch.from_numpy(best==best.min(-1,keepdims=True))

def certify(x):
    m=x.shape[-1]; i=np.arange(m); A,B,C,D=[s[(i[None,:]-i[:,None])%m] for s in x]; J=np.eye(m,dtype=np.int64)[:,::-1]
    H=np.block([[A,B@J,C@J,D@J],[-B@J,A,-D.T@J,C.T@J],[-C@J,D.T@J,A,-B.T@J],[-D@J,-C.T@J,B.T@J,A]])
    return bool(np.all(np.abs(H)==1) and np.array_equal(H@H.T,4*m*np.eye(4*m,dtype=np.int64)))

class Model(nn.Module):
    def __init__(self,w=24):
        super().__init__(); self.enc=nn.Linear(5,w); self.ctx=nn.Linear(w,w,bias=False); self.V=nn.Parameter(torch.randn(w,w)/w**.5); self.U=nn.Parameter(torch.randn(w,w)/w**.5); self.out=nn.Linear(w,1)
    def step(self,h,z): return torch.tanh(z+(h@self.V)@self.U.T+self.ctx(h.mean(1,keepdim=True)))
    def mid(self,f):
        z=self.enc(f); return z,self.step(z,z)
    def finish(self,z,h): return self.out(self.step(self.step(h,z),z)).squeeze(-1)
    def forward(self,f): return self.finish(*self.mid(f))

def loss(l,y): return (l.logsumexp(-1)-l.masked_fill(~y,-torch.inf).logsumexp(-1)).mean()
def projector(model,g):
    q,r=torch.linalg.qr(model.U[:,g].detach()); assert r.diag().abs().min()>1e-7; return q@q.T

def unit(v): return v/v.flatten(1).norm(dim=1).clamp_min(1e-9)[:,None,None]

def anchor(model,f,y):
    z,h=model.mid(f); l=loss(model.finish(z,h),y); d=-torch.autograd.grad(l,h,retain_graph=True)[0].detach(); return z,h,l,d

def choose(model,f,y):
    z,h,l,d=anchor(model,f,y); scores=[]
    with torch.no_grad():
        for g in range(0,24,4): scores.append(float(loss(model.finish(z,h+0.75*unit(d@projector(model,list(range(g,g+4))))),y)))
    g=int(np.argmin(scores))*4; return list(range(g,g+4)),scores,float(l.detach())

def update(model,opt,f,y,arm,P,g,rot):
    z,h,l,d=anchor(model,f,y); edit=0.75*unit(d@P); edits=[edit,edit@rot,0.75*unit(d@projector(model,g))]
    ps=list(model.parameters()); grads=[torch.autograd.grad(l,ps,retain_graph=True)]
    for ed in edits: grads.append(torch.autograd.grad(loss(model.finish(z,h+ed.detach()),y),ps,retain_graph=True))
    gn=lambda gs: torch.sqrt(sum((v*v).sum() for v in gs)); a=grads[0]; k=grads[arm]; scale=0.25*gn(a)/gn(k).clamp_min(1e-12) if arm else 0.
    opt.zero_grad()
    for p,v,w in zip(ps,a,k): p.grad=v+scale*w
    nn.utils.clip_grad_norm_(ps,1.0); opt.step()

def batch(rng,m,n): return oracle(rng.choice([-1,1],size=(n,4,m)))

def solve(model,x,kind,seed,steps):
    rng=np.random.default_rng(seed); b,_,m=x.shape; best=mechanics(x)[2]; solved=best==0
    for k in range(steps):
        _,_,e,nxt,f=mechanics(x)
        if kind=='greedy':
            a=(nxt+rng.random(nxt.shape)*0.01).argmin(-1); a=np.where(nxt[np.arange(b),a]>=e,rng.integers(4*m,size=b),a)
        elif kind=='anneal':
            a=rng.integers(4*m,size=b); delta=nxt[np.arange(b),a]-e; temp=max(4,4*m*(1-k/steps)); a=np.where((delta<=0)|(rng.random(b)<np.exp(-np.maximum(delta,0)/temp)),a,-1)
        else:
            with torch.no_grad(): logits=model(f).numpy()
            a=(logits+0.15*rng.gumbel(size=logits.shape)).argmax(-1)
        y=flip(x,np.maximum(a,0)); active=(~solved)&(a>=0); x[active]=y[active]
        en=mechanics(x)[2]; best=np.minimum(best,en); solved|=en==0
    assert all(certify(v) for v in x[solved]); return dict(solved=int(solved.sum()),n=b,best_energy_mean=float(best.mean()),candidate_scores=steps*b*4*m,bits=solved.astype(int).tolist())

def verify():
    rng=np.random.default_rng(10)
    for m in [3,5,7,11]:
        x=rng.choice([-1,1],size=(3,4,m)); r,d,e,nxt,f=mechanics(x); _,lab=oracle(x); costs=[]
        for a in range(4*m):
            y=flip(x,np.full(3,a)); r2,_,e2,n2,_=mechanics(y); np.testing.assert_array_equal(r2,r+d[:,a]); np.testing.assert_array_equal(e2,nxt[:,a]); n2[:,a]=10**9; costs.append(np.minimum(e2,n2.min(-1)))
        c=np.stack(costs,1); np.testing.assert_array_equal(lab.numpy(),c==c.min(1,keepdims=True))
    assert certify(np.array([[1,1,1],[1,1,-1],[1,1,-1],[1,1,-1]]))

def run(roots,warm,paired,n,steps):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); verify(); rows=[]
    for root in range(7301,7301+roots):
        torch.manual_seed(root); rng=np.random.default_rng(root); model=Model(); opt=torch.optim.AdamW(model.parameters(),lr=.003)
        for k in range(warm):
            f,y=batch(rng,[5,7,9][k%3],12); opt.zero_grad(); loss(model(f),y).backward(); opt.step()
        f,y=batch(rng,7,64); g,scores,base=choose(model,f,y); P=projector(model,g); rot=torch.eye(24)[:,torch.randperm(24)]*(2*torch.randint(2,(24,))-1)
        train=[batch(rng,[5,7,9][k%3],12) for k in range(paired)]; models=[]
        for arm in range(4):
            mdl=copy.deepcopy(model); opt=torch.optim.AdamW(mdl.parameters(),lr=.003)
            for f,y in train: update(mdl,opt,f,y,arm,P,g,rot)
            models.append(mdl)
        for m in (11,13):
            x=np.random.default_rng(root*100+m).choice([-1,1],size=(n,4,m)); f,y=oracle(x)
            for kind,mdl in zip(['task','fixed','sham','live','greedy','anneal'],models+[None,None]):
                rec=solve(mdl,x.copy(),kind,root*100+m+1,steps); rec.update(root=root,m=m,arm=kind,group=g,calibration_losses=scores,calibration_base=base)
                if mdl is not None:
                    with torch.no_grad(): rec['oracle_top1']=float(y.gather(1,mdl(f).argmax(-1,keepdim=True)).float().mean())
                rows.append(rec)
        print('finished root',root,flush=True)
    return dict(scope='DEVELOPMENT_ONLY',roots=roots,warmup=warm,paired=paired,cases_per_order=n,steps=steps,parameters=sum(p.numel() for p in Model().parameters()),rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--roots',type=int,default=4); p.add_argument('--warm',type=int,default=80); p.add_argument('--paired',type=int,default=40); p.add_argument('--cases',type=int,default=32); p.add_argument('--steps',type=int,default=64); p.add_argument('--out',default='results.json'); a=p.parse_args(); start=time.perf_counter(); result=run(a.roots,a.warm,a.paired,a.cases,a.steps); result['seconds']=time.perf_counter()-start
    with open(a.out,'w') as f: json.dump(result,f,indent=2)
