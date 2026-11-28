"""ER-23: streamed bound scoring and batched exact MAP; float64, not interval-certified."""
import argparse,json,resource,time
from pathlib import Path
import numpy as np
BETA=4/(3*np.sqrt(3)); EPS=1e-10
class Bank:
    def __init__(self,w,o,u,v,x,c):
        self.w,self.u,self.v=w,u,v; self.d,self.p=len(u),len(x)
        self.r,self.bias=c@o,.25*x
        h1=np.tanh(x@w.T+self.bias); z2=h1@w.T+self.bias; self.h2=np.tanh(z2)
        self.first2=np.tanh(z2[None]-(h1@v.T).T[:,:,None]*u[:,None,:])
        self.z3=self.h2@w.T+self.bias; self.first_z3=self.first2@w.T+self.bias; self.proj=self.first2@v.T
    def exact(self,keys):
        keys=np.asarray(keys,dtype=np.int64); i,j=keys//self.d,keys%self.d
        z=self.first_z3[i]-self.proj[i,:,j][:,:,None]*self.u[j,None,:]
        return (np.tanh(np.tanh(z)@self.w.T+self.bias)*self.r).sum(-1)
    def setup(self):
        h3=np.tanh(self.z3); first3=np.tanh(self.first_z3)
        self.a=self.first_z3-self.z3; self.b=-(self.h2@self.v.T).T[:,:,None]*self.u[:,None,:]
        second3=np.tanh(self.z3+self.b); z4=h3@self.w.T+self.bias
        self.a4=(first3-h3)@self.w.T; self.b4=(second3-h3)@self.w.T
        self.clean=(np.tanh(z4)*self.r).sum(-1); self.first=(np.tanh(z4+self.a4)*self.r).sum(-1); self.second=(np.tanh(z4+self.b4)*self.r).sum(-1)
        self.q=abs(self.r)@abs(self.w); self.qu=(self.q@abs(self.u).T).T
        self.br=abs(self.b).transpose(1,2,0).copy(); self.b4r=abs(self.b4).transpose(1,2,0).copy()
    def bounds(self,i,end):
        dp=abs((self.first2[i:end]-self.h2)@self.v.T).transpose(0,2,1)
        def contract(a,bt,weight): return ((abs(a[i:end])*weight).transpose(1,0,2)@bt).transpose(1,2,0)
        e=dp*self.qu[None]+BETA*contract(self.a,self.br,self.q)+BETA*contract(self.a4,self.b4r,abs(self.r))
        pred=self.first[i:end,None]+self.second[None]-self.clean
        return pred.reshape(-1,self.p),e.reshape(-1,self.p)+EPS

def cost(rows,y): return ((rows-y)**2).sum(-1)
def decision(bank,y,mode,batch=64,block=8):
    n=bank.d**2
    if mode=='eager':
        best,score=-1,np.inf
        for i in range(0,n,batch):
            q=cost(bank.exact(np.arange(i,min(i+batch,n))),y); k=int(q.argmin())
            if q[k]<score: best,score=i+k,q[k]
        return best,float(score),n,0
    bank.setup(); lower=np.empty(n); upper_best,incumbent=np.inf,-1
    block=bank.d if mode=='dense' else block
    for i in range(0,bank.d,block):
        end=min(i+block,bank.d); pred,err=bank.bounds(i,end); delta=abs(pred-y); start,stop=i*bank.d,end*bank.d
        lower[start:stop]=(np.maximum(0,delta-err)**2).sum(-1); upper=((delta+err)**2).sum(-1); j=int(upper.argmin())
        if upper[j]<upper_best: upper_best,incumbent=upper[j],start+j
    del pred,err,delta,upper
    best=incumbent; score=float(cost(bank.exact([best]),y)[0]); assert lower[best]-EPS<=score<=upper_best+EPS
    order=np.argsort(lower,kind='stable'); order=order[order!=incumbent]; sorted_lo=lower[order]
    calls,rounds,pos=1,0,0
    while pos<len(order):
        stop=int(np.searchsorted(sorted_lo,score+EPS,side='right'))
        if stop<=pos: break
        ids=order[pos:min(pos+batch,stop)]; exact=cost(bank.exact(ids),y); assert np.all(lower[ids]<=exact+EPS)
        m=exact.min(); k=int(ids[exact==m].min())
        if m<score or (m==score and k<best): best,score=k,float(m)
        calls+=len(ids); pos+=len(ids); rounds+=1
    return best,score,calls,rounds

def prepare(directory,widths,roots,episodes):
    import torch
    from er20 import train
    torch.set_num_threads(1); directory.mkdir(parents=True,exist_ok=True)
    for d in widths:
        for root in roots:
            model,mse=train(d,root); w=model.rec.weight.detach().numpy().astype(np.float64); o=model.out.weight.detach().numpy().astype(np.float64)
            u,s,v=np.linalg.svd(w,full_matrices=False); u=(u*s).T.copy(); rng=np.random.default_rng(np.random.SeedSequence([261139,d,root]))
            cx,cc=rng.standard_normal((2,24,d)); cal=Bank(w,o,u,v,cx,cc); clean=cx
            for _ in range(4): clean=np.tanh(clean@w.T+.25*cx)
            clean=((clean@o.T)*cc).sum(-1); keys=rng.choice(d*d,min(d*d,256),replace=False)
            sigma=max(1e-5,.1*np.sqrt(np.mean((cal.exact(keys)-clean)**2))); xs,cs,ys,hidden=[],[],[],[]
            for _ in range(episodes):
                x,c=rng.standard_normal((2,12,d)); h=int(rng.integers(d*d)); z=x; i,j=divmod(h,d)
                for t in range(4):
                    m=w if t not in (1,2) else w-np.outer(u[i if t==1 else j],v[i if t==1 else j]); z=np.tanh(z@m.T+.25*x)
                truth=((z@o.T)*c).sum(-1); assert np.allclose(truth,Bank(w,o,u,v,x,c).exact([h])[0],atol=1e-10,rtol=0)
                y=truth+rng.normal(0,sigma,12); xs.append(x); cs.append(c); ys.append(y); hidden.append(h)
            np.savez(directory/f'{d}_{root}.npz',w=w,o=o,u=u,v=v,x=xs,c=cs,y=ys,hidden=hidden,sigma=sigma,mse=mse)
            print(json.dumps(dict(width=d,root=root,sigma=float(sigma),mse=mse)),flush=True)

def bench(path,mode,repeats,batch,block):
    with np.load(path) as f: w,o,u,v,xs,cs,ys,hidden=[f[k] for k in ('w','o','u','v','x','c','y','hidden')]
    baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; rows=[]
    for x,c,y,h in zip(xs,cs,ys,hidden):
        times=[]; results=[]
        for _ in range(repeats):
            t=time.perf_counter(); bank=Bank(w,o,u,v,x,c); result=decision(bank,y,mode,batch,block); times.append((time.perf_counter()-t)*1000); del bank; results.append(result)
        assert all(r==results[0] for r in results); best,score,calls,rounds=result
        rows.append(dict(best=best,score=score,hidden=int(h),calls=calls,rounds=rounds,ms=float(np.median(times)),times=times))
    return dict(path=path.name,width=len(u),mode=mode,batch=batch,block=block,baseline_kib=baseline,peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','bench']); p.add_argument('path',type=Path)
    p.add_argument('--widths',type=int,nargs='+',default=[128,256]); p.add_argument('--roots',type=int,nargs='+',default=[36,37]); p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--mode',choices=['eager','dense','stream'],default='stream'); p.add_argument('--repeats',type=int,default=2); p.add_argument('--batch',type=int,default=64); p.add_argument('--block',type=int,default=8)
    a=p.parse_args()
    if a.action=='prepare': prepare(a.path,a.widths,a.roots,a.episodes)
    else: print(json.dumps(bench(a.path,a.mode,a.repeats,a.batch,a.block)),flush=True)
