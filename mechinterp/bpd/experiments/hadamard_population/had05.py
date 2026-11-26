"""HAD-05 development: exact LDT bounds, population tabu and recurrent allocation."""
import argparse, copy, functools, hashlib, itertools, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

HISTORY=4
F=15
LDT=slice(8,15)

@functools.lru_cache(None)
def sectors(m):
    if m<3 or m%2==0: raise ValueError('odd sequence length >=3 required')
    odd=range(1,min(m,math.isqrt(4*m))+1,2)
    out=set()
    for a in itertools.combinations_with_replacement(odd,4):
        if sum(x*x for x in a)==4*m: out.update(itertools.permutations(a))
    return np.asarray(sorted(out),dtype=np.int64).reshape(-1,4)

def power3(x):
    if x.shape[-1]%3: return None
    a,b,c=x.reshape(4,-1,3).sum(axis=1).T
    return a*a+b*b+c*c-a*b-b*c-c*a

def spectral_lower(q,m):
    # Exact: q > (2 sqrt(m)+2k)^2 iff D>0 and D^2>64 k^2 m.
    if q is None: return np.zeros(4,dtype=np.int64)
    out=[]
    for v in q:
        k=0
        while True:
            d=int(v)-4*m-4*k*k
            if d<=0 or d*d<=64*k*k*m: break
            k+=1
        out.append(k)
    return np.array(out,dtype=np.int64)

def ldt(x,r,u,horizon):
    m=x.shape[1]; ss=sectors(m); q=power3(x); b=spectral_lower(q,m)
    d=(np.abs(ss-np.abs(u))//2)
    totals=np.maximum(d,b).sum(1)
    lb_sum=int(d.sum(1).min()); lb_spec=int(b.sum()); lb_joint=int(totals.min())
    lb_corr=(int(np.abs(r).max())+3)//4
    # Number of magnitude-assignment sectors not ruled out by the joint row/spectral bound.
    # Sign choices closest to u minimize per-row flips; this is a necessary filter only.
    possible=int(np.count_nonzero(totals<=horizon)) if lb_corr<=horizon else 0
    return dict(row=lb_sum,spectral=lb_spec,joint=lb_joint,corr=lb_corr,
                lower=max(lb_joint,lb_corr),sectors=possible,total_sectors=len(ss),
                order3_power=None if q is None else q.tolist())

class State:
    def __init__(self,x):
        self.x=np.asarray(x,dtype=np.int64).reshape(-1).copy(); self.m=len(self.x)//4
        assert self.m>=3 and self.m%2 and len(self.x)==4*self.m and np.all(abs(self.x)==1)
        self.n=len(self.x); self.t=np.arange(1,self.m//2+1); a=np.arange(self.n); p=a%self.m; b=a//self.m*self.m
        self.r=(self.x[:,None]*self.x[b[:,None]+(p[:,None]+self.t)%self.m]).sum(0)
        self.d=-2*self.x[:,None]*(self.x[b[:,None]+(p[:,None]+self.t)%self.m]+self.x[b[:,None]+(p[:,None]-self.t)%self.m])
        self.s=self.x.reshape(4,self.m).sum(1)
    @property
    def energy(self): return int(self.r@self.r)
    @property
    def nbytes(self): return sum(a.nbytes for a in (self.x,self.r,self.d,self.s,self.t))
    def singles(self): return self.energy+2*(self.d@self.r)+(self.d*self.d).sum(1)
    def flip(self,i):
        i=int(i); m=self.m; self.r+=self.d[i]; self.s[i//m]-=2*self.x[i]
        a=np.arange(i//m*m,(i//m+1)*m); a=a[a!=i]; t=np.abs(a-i); t=np.minimum(t,m-t)-1
        self.d[a,t]+=4*self.x[i]*self.x[a]; self.d[i]*=-1; self.x[i]*=-1
    def matrix(self): return self.x.reshape(4,self.m)

def direct(x):
    return np.array([(x*np.roll(x,-t,axis=1)).sum() for t in range(1,x.shape[1]//2+1)],dtype=np.int64)

def certify(x):
    x=np.asarray(x,dtype=np.int64); m=x.shape[1]
    if x.shape!=(4,m) or not np.all(abs(x)==1) or np.any(direct(x)): return False
    i=np.arange(m); A,B,C,D=[s[(i[None,:]-i[:,None])%m] for s in x]; J=np.eye(m,dtype=np.int64)[:,::-1]
    H=np.block([[A,B@J,C@J,D@J],[-B@J,A,-D.T@J,C.T@J],[-C@J,D.T@J,A,-B.T@J],[-D@J,-C.T@J,B.T@J,A]])
    return bool(np.array_equal(H@H.T,4*m*np.eye(4*m,dtype=np.int64)))

def payload_per(m):
    # int64 state + int64 tabu + HISTORY float32 feature vectors; no Python/temporary/RSS claim.
    return 8*(8*m+2*(m//2)+4*m*(m//2)+4)+4*HISTORY*F

def preflight(m,p,cache_mib):
    if p*payload_per(m)>cache_mib*(1<<20): raise ValueError('population persistent payload exceeds cap')

def starts(seed,m,p):
    rng=np.random.default_rng(seed); xs=rng.choice([-1,1],size=(p,4,m)); ss=rng.integers(0,2**62,size=p,dtype=np.int64)
    return xs,ss,hashlib.sha256(xs.astype('int8').tobytes()+ss.tobytes()).hexdigest()

class Trajectory:
    def __init__(self,x,seed,chunk,tenure=7):
        self.state=State(x); self.rng=np.random.default_rng(int(seed)); self.chunk=chunk; self.tenure=tenure
        self.tabu=np.zeros(self.state.n,dtype=np.int64); self.steps=0; self.blocks=0; self.since=0; self.best=self.state.energy; self.gain=0.; self.change=0.; self.prev=self.state.energy; self.history=[self.snapshot()]
    def snapshot(self):
        s=self.state; m=s.m; b=ldt(s.matrix(),s.r,s.s,self.chunk); e=s.energy
        # all quantities available before allocation. FFT PSD excess is a heuristic, separate from exact bounds.
        ps=np.abs(np.fft.fft(s.matrix(),axis=1))**2; ps_excess=float(np.maximum(ps-4*m,0).max()/(4*m))
        f=np.array([np.log1p(e)/np.log1p(16*m*m),np.log1p(self.best)/np.log1p(16*m*m),
            self.gain,self.change,min(self.since/64,4),self.blocks/24,self.steps/(24*self.chunk),float(self.best==e),
            b['row']/m,b['spectral']/m,b['joint']/m,b['corr']/m,b['sectors']/b['total_sectors'],float(m%3==0),ps_excess],dtype=np.float32)
        return dict(f=f,best=self.best,current=e,lower=b['lower'],bounds=b)
    def advance(self):
        s=self.state; before=self.best; e0=s.energy; work=0
        for _ in range(self.chunk):
            if s.energy==0: break
            self.steps+=1; work+=s.n; v=s.singles(); ok=(self.tabu<=self.steps)|(v<self.best)
            tie=self.rng.random(s.n); rank=np.lexsort((tie,v)); elig=rank[ok[rank]]; a=int(elig[0] if len(elig) else rank[0]); s.flip(a); self.tabu[a]=self.steps+self.tenure
            if s.energy<self.best: self.best=s.energy; self.since=0
            else: self.since+=1
        self.blocks+=1; self.gain=(before-self.best)/max(1,before); self.change=(e0-s.energy)/max(1,e0)
        self.history=(self.history+[self.snapshot()])[-HISTORY:]; return work
    def obs(self): return padded(self.history)

def padded(hist):
    h=hist[-HISTORY:]; return np.stack([h[0]['f']]*(HISTORY-len(h))+[s['f'] for s in h])

def teacher_value(cur,nxt,global_best):
    return (4.*(nxt['best']==0)+(global_best-min(global_best,nxt['best']))/max(1,global_best)+.15*(cur['best']-nxt['best'])/max(1,cur['best']))

def make_bank(seed,lengths,p,chunk,episodes,decisions,bank_mib):
    # Preallocate at the known upper bound before simulation; slices share backing arrays.
    cap=episodes*decisions; nbytes=4*cap*p*(HISTORY*F+1)
    if nbytes>bank_mib*(1<<20): raise ValueError('teacher bank would exceed configured array budget')
    X=np.empty((cap,p,HISTORY,F),dtype=np.float32); Y=np.empty((cap,p),dtype=np.float32); count=inform=work=0
    rng=np.random.default_rng(seed)
    for ep in range(episodes):
        m=lengths[ep%len(lengths)]; xs,ss,_=starts(int(rng.integers(2**60)),m,p)
        ts=[Trajectory(x,z,chunk) for x,z in zip(xs,ss)]; tape=[[t.history[-1]] for t in ts]; idx=np.zeros(p,dtype=int)
        if any(t.best==0 for t in ts): continue
        for i,t in enumerate(ts): work+=t.advance(); tape[i].append(t.history[-1])
        for j in range(decisions):
            cur=[tape[i][idx[i]] for i in range(p)]; nxt=[tape[i][idx[i]+1] for i in range(p)]; g=min(q['best'] for q in cur)
            x=np.stack([padded(tape[i][:idx[i]+1]) for i in range(p)]); y=np.array([teacher_value(c,n,g) for c,n in zip(cur,nxt)],dtype=np.float32)
            X[count]=x; Y[count]=y; count+=1; inform+=int(y.max()-y.min()>1e-8)
            z=rng.random(); i=int(np.argmax(y+rng.random(p)*1e-8)) if z<.33 else (int(np.argmin([c['best'] for c in cur])) if z<.66 else int(rng.integers(p)))
            idx[i]+=1
            if tape[i][idx[i]]['best']==0: break
            work+=ts[i].advance(); tape[i].append(ts[i].history[-1])
    if count==0: raise ValueError('empty teacher bank')
    return torch.from_numpy(X[:count]),torch.from_numpy(Y[:count]),dict(samples=count,informative=inform,allocated_bytes=nbytes,teacher_candidate_scores=work)

class Recurrent(nn.Module):
    def __init__(self,w=24,use_ldt=True):
        super().__init__(); self.use_ldt=use_ldt; self.w=w
        self.enc=nn.Linear(F,w); self.ctx=nn.Linear(w,w,bias=False); self.V=nn.Parameter(torch.randn(w,w)/w**.5); self.U=nn.Parameter(torch.randn(w,w)/w**.5); self.out=nn.Linear(w,1)
    def prepare(self,x):
        if not self.use_ldt: x=x.clone(); x[...,LDT]=0
        return x
    def step(self,h,f): return torch.tanh(self.enc(f)+(h@self.V)@self.U.T+self.ctx(h.mean(1,keepdim=True)))
    def mid(self,x):
        x=self.prepare(x); h=torch.zeros((*x.shape[:2],self.w),device=x.device)
        for k in range(x.shape[2]-1): h=self.step(h,x[:,:,k])
        return x[:,:,-1],h
    def finish(self,last,h): return self.out(self.step(h,last)).squeeze(-1)
    def forward(self,x): return self.finish(*self.mid(x))

class MLP(nn.Module):
    def __init__(self,w=48):
        super().__init__(); self.enc=nn.Linear(F,w); self.mid=nn.Linear(w*2,16); self.out=nn.Linear(16,1)
    def forward(self,x):
        h=torch.tanh(self.enc(x[:,:,-1])); g=h.mean(1,keepdim=True).expand_as(h)
        return self.out(torch.tanh(self.mid(torch.cat([h,g],-1)))).squeeze(-1)

def loss(l,y): return -(torch.softmax(3*y,-1)*torch.log_softmax(l,-1)).sum(-1).mean()
def quality(model,x,y):
    with torch.no_grad():
        l=model(x); z=l.argmax(-1); achieved=y.gather(1,z[:,None]).squeeze(1); mx=y.max(-1).values; informative=(mx-y.min(-1).values)>1e-8
        return dict(loss=float(loss(l,y)),regret=float((mx-achieved).mean()),informative=int(informative.sum()),hit=float(((mx-achieved).abs()[informative]<1e-7).float().mean()) if informative.any() else None)

def proj(model,g):
    q,r=torch.linalg.qr(model.U[:,g].detach()); assert r.diag().abs().min()>1e-7; return q@q.T

def unit(v): return v/v.flatten(1).norm(dim=1).clamp_min(1e-9)[:,None,None]
def anchor(model,x,y):
    last,h=model.mid(x); l=loss(model.finish(last,h),y); d=-torch.autograd.grad(l,h,retain_graph=True)[0].detach(); return last,h,l,d

def discover(model,x,y,dose=.75):
    last,h,l,d=anchor(model,x,y); scores=[]
    with torch.no_grad():
        for g in range(0,model.w,4): scores.append(float(loss(model.finish(last,h+dose*unit(d@proj(model,list(range(g,g+4))))),y)))
    g=int(np.argmin(scores))*4; return list(range(g,g+4)),scores,float(l.detach())

def update(model,opt,x,y,arm,P,g,rot,dose=.75):
    last,h,l,d=anchor(model,x,y); e=dose*unit(d@P); edits=[e,e@rot,dose*unit(d@proj(model,g))]
    ps=list(model.parameters()); gs=[torch.autograd.grad(l,ps,retain_graph=True)]
    for z in edits: gs.append(torch.autograd.grad(loss(model.finish(last,h+z.detach()),y),ps,retain_graph=True))
    norm=lambda v: torch.sqrt(sum((a*a).sum() for a in v)); base=gs[0]; aux=gs[arm]; scale=.25*norm(base)/norm(aux).clamp_min(1e-12) if arm else 0.
    opt.zero_grad()
    for p,a,b in zip(ps,base,aux): p.grad=a+scale*b
    nn.utils.clip_grad_norm_(ps,1.); opt.step()
    return [float(norm(base)),float(norm(aux)),float((e.flatten(1).norm(dim=1)-edits[1].flatten(1).norm(dim=1)).abs().max())]

def causal_audit(model,x,y,P,rot):
    # Label-free physical removal of warmup-subspace projection at the history boundary.
    with torch.no_grad():
        last,h=model.mid(x); e=-(h@P); base=model.finish(last,h); target=model.finish(last,h+e); sham=model.finish(last,h+e@rot)
        z=base.argmax(-1); t=target.argmax(-1); s=sham.argmax(-1)
        return dict(base_loss=float(loss(base,y)),target_loss=float(loss(target,y)),sham_loss=float(loss(sham,y)),target_flip=float((t!=z).float().mean()),sham_flip=float((s!=z).float().mean()),max_norm_mismatch=float((e.flatten(1).norm(dim=1)-(e@rot).flatten(1).norm(dim=1)).abs().max()))

def ordinary_train(model,X,Y,batches):
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    for i in batches: opt.zero_grad(); loss(model(X[i]),Y[i]).backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()

# Optional transfer: exact same signed row sum and, when available, exact same order-3 power.
# Score each admissible complete child using exact autocorrelation; charge n candidate units
# conservatively per child evaluation (different operations from cached single-flip scoring).
def transfer(ts,i,budget):
    t=ts[i]; s=t.state; m=s.m; x=s.matrix(); q=power3(x); base=s.energy; best=None; work=0; offers=0
    for j,u in enumerate(ts):
        if j==i: continue
        y=u.state.matrix(); qp=power3(y)
        for a in range(4):
            for b in range(4):
                if s.s[a]!=u.state.s[b] or (q is not None and q[a]!=qp[b]) or np.array_equal(x[a],y[b]): continue
                if work+s.n>budget: return best,work,offers
                child=x.copy(); child[a]=y[b]; r=direct(child); e=int(r@r); work+=s.n; offers+=1
                if e<base: best=(child,j,a,b,e); base=e
    return best,work,offers

def choose_policy(name,ts,obs,j,model):
    p=len(ts)
    if name=='round_robin': return j%p
    if name=='serial': return 0
    if name=='best': return min(range(p),key=lambda i:(ts[i].best,ts[i].blocks,i))
    if name=='ldt_best':
        viable=[i for i,t in enumerate(ts) if t.history[-1]['lower']<=t.chunk]
        return min(viable or range(p),key=lambda i:(ts[i].best,ts[i].blocks,i))
    if name in ('ucb','ucb_transfer'):
        # Prioritize low incumbent energy but retain explicit exploration of less-run trajectories.
        return max(range(p),key=lambda i:(-float(obs[i,-1,1])+0.22*math.sqrt(math.log(j+2)/(ts[i].blocks+1)), -i))
    with torch.no_grad(): return int(model(torch.from_numpy(obs)[None])[0].argmax())

def evaluate(seed,m,p,chunk,rounds,name,model,cache_mib):
    preflight(m,p,cache_mib); xs,ss,h=starts(seed,m,p); ts=[Trajectory(x,z,chunk) for x,z in zip(xs,ss)]
    if any(t.state.energy==0 for t in ts): rounds=0
    budget=rounds*chunk*4*m; work=0; history=[]; transfers=offers=0; ruled_out=sector_eliminations=0; begin=time.perf_counter(); calls=0
    for j in range(rounds):
        if work+4*m>budget or any(t.state.energy==0 for t in ts): break
        ruled_out+=sum(t.history[-1]['lower']>chunk for t in ts); sector_eliminations+=sum(t.history[-1]['bounds']['total_sectors']-t.history[-1]['bounds']['sectors'] for t in ts); obs=np.stack([t.obs() for t in ts]); i=choose_policy(name,ts,obs,j,model); calls+=int(model is not None); t=ts[i]
        # transfer is an optional common deterministic offer policy, only when stalled.
        if name.endswith('_transfer') and t.since>=chunk and work+(chunk+1)*4*m<=budget:
            child,w,n=transfer(ts,i,min(4*m*4,budget-work-chunk*4*m)); work+=w; offers+=n
            if child is not None:
                x,jj,a,b,e=child; old=t.state; new=State(x); assert new.energy==e and np.array_equal(new.s,old.s); q=power3(x); q0=power3(old.matrix()); assert q is None or np.array_equal(q,q0)
                t.state=new; t.tabu[:]=0; t.since=0; t.best=min(t.best,e); t.history=(t.history+[t.snapshot()])[-HISTORY:]; transfers+=1
        if t.state.energy!=0:
            old=t.chunk; t.chunk=min(chunk,(budget-work)//(4*m)); work+=t.advance(); t.chunk=old
        history.append([i,t.best,t.state.energy,work])
    solved=next((t for t in ts if t.state.energy==0),None); cert=solved.state.matrix().tolist() if solved else None
    if cert is not None: assert certify(cert)
    return dict(solved=solved is not None,best=min(t.best for t in ts),candidate_scores=work,rounds=len(history),model_calls=calls,allocation=[t.blocks for t in ts],trace=history,transfers=transfers,offers=offers,horizon_ruled_out=ruled_out,sector_eliminations=sector_eliminations,certificate=cert,start_hash=h,seconds=time.perf_counter()-begin,tracked_population_bytes=sum(t.state.nbytes+t.tabu.nbytes+4*HISTORY*F for t in ts))

def run(a):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); begin=time.perf_counter(); rows=[]; diag=[]
    for m in set(a.train_lengths+a.lengths): preflight(m,a.population,a.cache_mib); assert len(sectors(m))
    for root in range(a.root,a.root+a.roots):
        torch.manual_seed(root); rng=np.random.default_rng(root)
        X,Y,b=make_bank(root*100+1,a.train_lengths,a.population,a.chunk,a.train_episodes,a.decisions,a.bank_mib)
        C,D,c=make_bank(root*100+2,a.train_lengths,a.population,a.chunk,8,a.decisions,a.bank_mib)
        Z,W,z=make_bank(root*100+3,a.lengths,a.population,a.chunk,8,a.decisions,a.bank_mib)
        batches=[rng.integers(len(X),size=16) for _ in range(a.warm+a.paired)]
        model=Recurrent(); no=copy.deepcopy(model); no.use_ldt=False; ordinary_train(model,X,Y,batches[:a.warm]); g,scores,base=discover(model,C,D); P=proj(model,g); rot=torch.eye(24)[:,torch.randperm(24)]*(2*torch.randint(2,(24,))-1)
        models={}; info=[]
        for arm,name in enumerate(('trm','fixed','sham','live')):
            mdl=copy.deepcopy(model); opt=torch.optim.AdamW(mdl.parameters(),lr=.003); logs=[]
            for ids in batches[a.warm:]: logs.append(update(mdl,opt,X[ids],Y[ids],arm,P,g,rot))
            models[name]=mdl; info.append(dict(arm=name,drift=float((proj(mdl,g)-P).norm()/math.sqrt(8)),grad_receipt=np.mean(logs,0).tolist(),max_dose_mismatch=max(v[2] for v in logs),calibration=quality(mdl,C,D),ood_teacher=quality(mdl,Z,W),causal=causal_audit(mdl,Z,W,P,rot)))
        mlp=MLP(); ordinary_train(no,X,Y,batches[:a.warm]); ordinary_train(no,X,Y,batches[a.warm:]); ordinary_train(mlp,X,Y,batches[:a.warm]); ordinary_train(mlp,X,Y,batches[a.warm:]); models.update(no_ldt=no,mlp=mlp,trm_transfer=models['trm'])
        for name,mdl in [('no_ldt',no),('mlp',mlp)]: info.append(dict(arm=name,calibration=quality(mdl,C,D),ood_teacher=quality(mdl,Z,W)))
        diag.append(dict(root=root,bank=b,calibration=c,ood_diagnostic=z,group=g,discovery_losses=scores,discovery_base=base,parameters={n:sum(p.numel() for p in mdl.parameters()) for n,mdl in models.items()},arms=info))
        for m in a.lengths:
            for case in range(a.cases):
                seed=root*100000+m*100+case
                for name in ('serial','round_robin','best','ldt_best','ucb','ucb_transfer','mlp','no_ldt','trm','fixed','sham','live','trm_transfer'):
                    r=evaluate(seed,m,a.population,a.chunk,a.rounds,name,models.get(name),a.cache_mib); r.update(root=root,m=m,case=case,seed=seed,arm=name); rows.append(r)
        print('root',root,'done',len(X),'teacher snapshots',flush=True)
    out=dict(scope='DEVELOPMENT_ONLY',args=vars(a),roots=diag,rows=rows,seconds=time.perf_counter()-begin)
    Path(a.out).write_text(json.dumps(out,indent=2)); print('seconds',out['seconds'],flush=True)
    for name in ('serial','round_robin','best','ldt_best','ucb','ucb_transfer','mlp','no_ldt','trm','fixed','sham','live','trm_transfer'):
        r=[r for r in rows if r['arm']==name]; print(name,[(m,sum(z['solved'] for z in r if z['m']==m)) for m in a.lengths],sum(z['solved'] for z in r),round(float(np.mean([z['best'] for z in r])),2),round(sum(z['seconds'] for z in r),2),'transfers',sum(z['transfers'] for z in r),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=int,default=85101); p.add_argument('--roots',type=int,default=4); p.add_argument('--cases',type=int,default=6); p.add_argument('--population',type=int,default=8); p.add_argument('--chunk',type=int,default=16); p.add_argument('--rounds',type=int,default=24); p.add_argument('--train-lengths',nargs='+',type=int,default=[19,21,25]); p.add_argument('--lengths',nargs='+',type=int,default=[23,27,29]); p.add_argument('--train-episodes',type=int,default=32); p.add_argument('--decisions',type=int,default=12); p.add_argument('--warm',type=int,default=100); p.add_argument('--paired',type=int,default=60); p.add_argument('--cache-mib',type=int,default=8); p.add_argument('--bank-mib',type=int,default=4); p.add_argument('--out',default='results.json'); run(p.parse_args())
