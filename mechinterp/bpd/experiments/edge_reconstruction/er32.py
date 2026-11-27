"""ER-32: rank-one curvature contraction (float64, not directed rounding)."""
import argparse,json,resource,time
from pathlib import Path
import numpy as np
from er23 import Bank,BETA,EPS,cost
from er31 import GroupBank
from er30 import finish,warmup
MODES=('reuse','global','rank')

class RankBank(Bank):
    def setup_rank(self):
        self.cp=self.h2@self.v.T; self.absuT=abs(self.u).T.copy()
        h3=np.tanh(self.z3); first3=np.tanh(self.first_z3)
        a=self.first_z3-self.z3
        second3=np.tanh(self.z3-self.cp.T[:,:,None]*self.u[:,None,:])
        z4=h3@self.w.T+self.bias
        a4=(first3-h3)@self.w.T; b4=(second3-h3)@self.w.T
        self.clean=(np.tanh(z4)*self.r).sum(-1)
        self.first=(np.tanh(z4+a4)*self.r).sum(-1)
        self.second=(np.tanh(z4+b4)*self.r).sum(-1)
        self.q=abs(self.r)@abs(self.w); self.qu=(self.q@self.absuT).T
        self.wa=abs(a)*self.q
        self.wa4=(abs(a4)*abs(self.r)).transpose(1,0,2).copy()
        self.b4r=abs(b4).transpose(1,2,0).copy()
    def bound(self,i,end):
        dp=abs(self.proj[i:end]-self.cp).transpose(0,2,1)
        first=(self.wa[i:end].reshape(-1,self.w.shape[0])@self.absuT).reshape(end-i,self.p,self.d)
        first=(first*abs(self.cp)[None]).transpose(0,2,1)
        last=(self.wa4[:,i:end]@self.b4r).transpose(1,2,0)
        e=dp*self.qu[None]+BETA*first+BETA*last
        pred=self.first[i:end,None]+self.second[None]-self.clean
        return pred.reshape(-1,self.p),e.reshape(-1,self.p)+EPS

def setup(args,mode):
    if mode in ('reuse','global'):
        b=GroupBank(*args); b.setup_group(1 if mode=='reuse' else b.w.shape[0])
    else:
        b=RankBank(*args); b.setup_rank()
    return b

def payload(b):
    # Named NumPy array payloads, including inputs; excludes object overhead.
    return sum(v.nbytes for v in vars(b).values() if isinstance(v,np.ndarray))

def solve(args,y,mode):
    t=time.perf_counter(); b=setup(args,mode); d,p=b.d,b.p
    state_bytes=payload(b); floors=np.empty((d*d,p)); gap=np.zeros(p); upper=np.inf; best=-1
    for i in range(0,d,8):
        end=min(i+8,d); pred,err=b.bound(i,end); delta=abs(pred-y)
        l=np.maximum(0,delta-err)**2; h=(delta+err)**2
        floors[i*d:end*d]=l; gap+=(h-l).sum(0); hi=h.sum(1); j=int(hi.argmin())
        if hi[j]<upper: upper=float(hi[j]); best=i*d+j
    lower=floors.sum(1); score=float(cost(b.exact([best]),y)[0]); assert lower[best]<=score+EPS<=upper+2*EPS
    order=np.argsort(lower,kind='stable'); order=order[order!=best]; cols=np.argsort(-gap,kind='stable')
    arr=[np.ascontiguousarray(a) for a in [b.w.T,b.u,b.first_z3,b.proj,b.bias,b.r,y,floors,lower,order,cols]]
    before=time.perf_counter(); best,score,calls,work,checks=finish(*arr,best,score,1,EPS); end=time.perf_counter()
    return dict(best=int(best),score=float(score),calls=int(calls),pairprobes=int(work),checks=int(checks),state_bytes=state_bytes,floor_bytes=floors.nbytes,screen_ms=(before-t)*1000,finish_ms=(end-before)*1000,ms=(end-t)*1000)

def prepare(path,width,roots):
    from er25 import world
    path.mkdir(parents=True,exist_ok=True)
    for root in roots:
        for ep,(args,y,hidden,noise) in enumerate(world(width,root)):
            np.savez(path/f'{width}_{root}_{ep}.npz',w=args[0],o=args[1],u=args[2],v=args[3],x=args[4],c=args[5],y=y,hidden=hidden,noise=noise)

def load(path):
    with np.load(path) as f:
        return tuple(f[k] for k in ('w','o','u','v','x','c')),f['y'],int(f['hidden']),float(f['noise'])

def audit(path):
    args,y,hidden,noise=load(path); b=setup(args,'reuse'); r=setup(args,'rank'); g=setup(args,'global'); n=b.d*b.d
    checks=0; maxdiff=0.; best=-1; score=np.inf
    for i in range(0,b.d,8):
        end=min(i+8,b.d); pred,err=b.bound(i,end); ids=np.arange(i*b.d,end*b.d)
        exact=np.concatenate([b.exact(ids[k:k+64]) for k in range(0,len(ids),64)])
        for mode in (r,g):
            pp,ee=mode.bound(i,end)
            assert np.allclose(pp,pred,atol=EPS,rtol=0)
            assert np.all(abs(exact-pp)<=ee+EPS)
            if mode is r:
                maxdiff=max(maxdiff,float(abs(ee-err).max())); assert np.allclose(ee,err,atol=EPS,rtol=0)
            else: assert np.all(ee>=err-EPS)
            checks+=ee.size
        v=cost(exact,y); j=int(v.argmin())
        if v[j]<score: best,score=int(ids[j]),float(v[j])
    return dict(path=path.name,best=best,score=score,hidden=hidden,noise=noise,checks=checks,max_envelope_difference=maxdiff)

def bench(paths,mode,reps):
    warm=warmup(); base=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for path in paths:
        args,y,hidden,noise=load(path); rows=[solve(args,y,mode) for _ in range(reps)]
        for r in rows: assert r['best']==rows[0]['best'] and abs(r['score']-rows[0]['score'])<EPS and r['pairprobes']==rows[0]['pairprobes']
        r=rows[0].copy()
        for k in ('ms','screen_ms','finish_ms'): r[k]=float(np.median([x[k] for x in rows]))
        print(json.dumps(dict(path=path.name,mode=mode,hidden=hidden,noise=noise,baseline_kib=base,peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,compile_ms=warm,times_ms=[x['ms'] for x in rows],**r)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('prepare','audit','bench')); p.add_argument('paths',type=Path,nargs='+')
    p.add_argument('--width',type=int,default=128); p.add_argument('--roots',type=int,nargs='+',default=[66,67]); p.add_argument('--mode',choices=MODES,default='rank'); p.add_argument('--reps',type=int,default=2)
    a=p.parse_args()
    if a.action=='prepare': prepare(a.paths[0],a.width,a.roots)
    elif a.action=='audit':
        for path in a.paths: print(json.dumps(audit(path)),flush=True)
    else: bench(a.paths,a.mode,a.reps)
