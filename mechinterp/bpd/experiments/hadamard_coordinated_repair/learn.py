"""HAD-04B: learned ordering at tabu-search stalls; development only."""
import argparse, copy, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from search import State, certify

class Model(nn.Module):
    def __init__(self, native=False, w=24):
        super().__init__(); self.native=native; self.w=w
        self.enc=nn.Linear(7,w); self.ctx=nn.Linear(w,w,bias=False); self.out=nn.Linear(w,1)
        if native:
            from mechinterp.bpd.components import TiedComponentLinear
            self.core=TiedComponentLinear(nn.Linear(w,w,bias=False),w,max_passes=3,init_svd=False); self.core.mode='component'
        else:
            self.V=nn.Parameter(torch.randn(w,w)/w**.5); self.U=nn.Parameter(torch.randn(w,w)/w**.5)
    def basis(self): return self.core.U.T if self.native else self.U
    def prepare(self,h,offset):
        if self.native:
            from mechinterp.bpd.components import PassMaskSchedule
            self.core.reset_pass_state(pass_offset=offset)
            self.core.schedule=PassMaskSchedule(torch.ones(1,3,1,self.w,device=h.device),torch.zeros(1,3,1,device=h.device))
    def step(self,h,z):
        a=self.core(h) if self.native else (h@self.V)@self.U.T
        return torch.tanh(z+a+self.ctx(h.mean(1,keepdim=True)))
    def mid(self,f):
        z=self.enc(f); self.prepare(z,0); return z,self.step(z,z)
    def finish(self,z,h):
        self.prepare(h,1); return self.out(self.step(self.step(h,z),z)).squeeze(-1)
    def forward(self,f): return self.finish(*self.mid(f))

def features(s,vals,remaining,allowed):
    m=s.m; return np.stack([s.x,(vals-s.energy)/(16*m),(s.d*s.d).sum(1)/(16*m),np.full(s.n,s.energy/(4*m*m)),np.repeat(s.s,m)/m,remaining/7,allowed],-1).astype('float32')

def teacher(s,vals,allowed):
    i,j=np.triu_indices(s.n,1); v=s.pairs(i,j,vals); ok=allowed[i]&allowed[j]; i,j,v=i[ok],j[ok],v[ok]
    best=np.where(allowed,vals,np.iinfo(np.int64).max).copy(); np.minimum.at(best,i,v); np.minimum.at(best,j,v)
    return best==best.min()

def batch_bank(rng,m,count):
    fs=[]; ys=[]; n=4*m
    while len(fs)<count:
        s=State(rng.choice([-1,1],size=(4,m))); tabu=np.zeros(n,dtype=int); best=s.energy
        for it in range(1,100):
            if s.energy==0: break
            v=s.singles(); allowed=(tabu<=it)|(v<best); rank=np.lexsort((rng.random(n),v)); eligible=rank[allowed[rank]]; a=int(eligible[0] if len(eligible) else rank[0])
            if v[a]>=s.energy:
                fs.append(features(s,v,np.maximum(tabu-it,0),allowed)); ys.append(teacher(s,v,allowed))
                if len(fs)==count: break
            s.flip(a); tabu[a]=it+7; best=min(best,s.energy)
    return torch.from_numpy(np.stack(fs)),torch.from_numpy(np.stack(ys))

def loss(l,y): return (l.logsumexp(-1)-l.masked_fill(~y,-torch.inf).logsumexp(-1)).mean()
def masked(l,f): return l.masked_fill(f[...,6]==0,-torch.inf)
def projector(model,g):
    q,r=torch.linalg.qr(model.basis()[:,g].detach()); assert r.diag().abs().min()>1e-7; return q@q.T

def unit(v): return v/v.flatten(1).norm(dim=1).clamp_min(1e-9)[:,None,None]
def anchor(model,f,y):
    z,h=model.mid(f); l=loss(masked(model.finish(z,h),f),y); d=-torch.autograd.grad(l,h,retain_graph=True)[0].detach(); return z,h,l,d

def choose(model,f,y):
    z,h,l,d=anchor(model,f,y); scores=[]
    with torch.no_grad():
        for g in range(0,24,4): scores.append(float(loss(masked(model.finish(z,h+.75*unit(d@projector(model,list(range(g,g+4))))),f),y)))
    g=int(np.argmin(scores))*4; return list(range(g,g+4)),scores,float(l.detach())

def update(model,opt,f,y,arm,P,g,rot):
    z,h,l,d=anchor(model,f,y); edit=.75*unit(d@P); edits=[edit,edit@rot,.75*unit(d@projector(model,g))]; ps=list(model.parameters()); grads=[torch.autograd.grad(l,ps,retain_graph=True)]
    for ed in edits: grads.append(torch.autograd.grad(loss(masked(model.finish(z,h+ed.detach()),f),y),ps,retain_graph=True))
    norm=lambda gs: torch.sqrt(sum((v*v).sum() for v in gs)); a=grads[0]; k=grads[arm]; scale=.25*norm(a)/norm(k).clamp_min(1e-12) if arm else 0.
    opt.zero_grad()
    for p,v,w in zip(ps,a,k): p.grad=v+scale*w
    nn.utils.clip_grad_norm_(ps,1.); opt.step()
    return float(norm(a)),float(norm(k)),float((edit.flatten(1).norm(dim=1)-.75).abs().max())

def evaluate(x,seed,budget,model=None):
    rng=np.random.default_rng(seed); s=State(x); n=s.n; best=s.energy; tabu=np.zeros(n,dtype=int); it=cost=stalls=0; start=time.perf_counter()
    while s.energy and cost+n<=budget:
        v=s.singles(); cost+=n; it+=1; allowed=(tabu<=it)|(v<best); tie=rng.random(n); rank=np.lexsort((tie,v)); eligible=rank[allowed[rank]]; a=int(eligible[0] if len(eligible) else rank[0])
        if v[a]>=s.energy:
            stalls+=1
            if model is not None:
                f=torch.from_numpy(features(s,v,np.maximum(tabu-it,0),allowed))[None]
                with torch.no_grad(): l=masked(model(f),f)[0].numpy()
                a=int(np.lexsort((tie,-l))[0])
        s.flip(a); tabu[a]=it+7; best=min(best,s.energy)
    c=s.certificate() if s.energy==0 else None
    if c is not None: assert certify(c)
    return dict(solved=c is not None,best=best,seconds=time.perf_counter()-start,candidate_scores=cost,stalls=stalls,certificate=c)

def run(a):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); rows=[]; roots=[]; start=time.perf_counter()
    for root in range(a.root,a.root+a.roots):
        torch.manual_seed(root); rng=np.random.default_rng(root); model=Model(a.native); opt=torch.optim.AdamW(model.parameters(),lr=.003)
        bank={m:batch_bank(rng,m,a.bank) for m in [11,13,15,17]}; bbytes=sum(t.nelement()*t.element_size() for ts in bank.values() for t in ts)
        assert bbytes<=a.bank_mib*(1<<20), 'training bank exceeded cap'
        for step in range(a.warm):
            f,y=bank[[11,13,15,17][step%4]]; i=rng.integers(len(f),size=16); f,y=f[i],y[i]; opt.zero_grad(); loss(masked(model(f),f),y).backward(); opt.step()
        f,y=batch_bank(rng,17,64); g,scores,base=choose(model,f,y); P=projector(model,g); rot=torch.eye(24)[:,torch.randperm(24)]*(2*torch.randint(2,(24,))-1)
        train=[]
        for step in range(a.paired):
            f,y=bank[[11,13,15,17][step%4]]; i=rng.integers(len(f),size=16); train.append((f[i],y[i]))
        models=[]; diagnostics=[]
        for arm in range(4):
            mdl=copy.deepcopy(model); opt=torch.optim.AdamW(mdl.parameters(),lr=.003); d=[]
            for f,y in train: d.append(update(mdl,opt,f,y,arm,P,g,rot))
            models.append(mdl); diagnostics.append(dict(arm=arm,drift=float((projector(mdl,g)-P).norm()/np.sqrt(8)),max_dose_error=max(z[2] for z in d),mean_task_grad=float(np.mean([z[0] for z in d])),mean_selected_grad=float(np.mean([z[1] for z in d]))))
        roots.append(dict(root=root,group=g,calibration_losses=scores,calibration_base=base,bank_bytes=bbytes,parameters=sum(p.numel() for p in model.parameters()),diagnostics=diagnostics))
        for m in a.lengths:
            for case in range(a.cases):
                seed=root*100000+m*100+case; x=np.random.default_rng(seed).choice([-1,1],size=(4,m)); h=hashlib.sha256(x.astype('int8').tobytes()).hexdigest()
                for name,mdl in zip(['tabu','task','fixed','sham','live'],[None]+models):
                    r=evaluate(x,seed+1,a.budget,mdl); r.update(root=root,m=m,case=case,arm=name,seed=seed,start_hash=h); rows.append(r)
        print('root',root,'complete',flush=True)
    result=dict(scope='DEVELOPMENT_ONLY',backend='native TiedComponentLinear' if a.native else 'shared-factor proxy',args=vars(a),roots=roots,rows=rows,seconds=time.perf_counter()-start)
    Path(a.out).write_text(json.dumps(result,indent=2))
    for m in a.lengths:
        for arm in ['tabu','task','fixed','sham','live']:
            q=[r for r in rows if r['m']==m and r['arm']==arm]; print(m,arm,sum(r['solved'] for r in q),len(q),round(np.mean([r['best'] for r in q]),3),round(sum(r['seconds'] for r in q),3),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--native',action='store_true'); p.add_argument('--root',type=int,default=84201); p.add_argument('--roots',type=int,default=4); p.add_argument('--cases',type=int,default=12); p.add_argument('--lengths',type=int,nargs='+',default=[19,23,29]); p.add_argument('--bank',type=int,default=192); p.add_argument('--bank-mib',type=int,default=4); p.add_argument('--warm',type=int,default=100); p.add_argument('--paired',type=int,default=60); p.add_argument('--budget',type=int,default=16384); p.add_argument('--out',default='learn_results.json'); run(p.parse_args())
