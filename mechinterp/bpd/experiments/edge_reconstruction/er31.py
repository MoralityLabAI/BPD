"""ER-31: grouped Cauchy-Schwarz bounds and prefix projection reuse.
Real-arithmetic bounds; float64 checks, not directed rounding.
"""
import argparse,json,time
import numpy as np
from er23 import Bank,BETA,EPS,cost,decision
from er25 import world
from er30 import finish,fused,warmup
MODES=('eager','fused1','reuse','group4','group16','global','cascade4')
class GroupBank(Bank):
    def setup_group(self,g):
        self.setup(); self.cp=self.h2@self.v.T; self.g=g
        if g>1:
            def norms(x,w):
                z=abs(x)*np.sqrt(w)[None]
                z=np.pad(z,((0,0),(0,0),(0,(-z.shape[-1])%g)))
                return np.sqrt((z.reshape(*z.shape[:-1],-1,g)**2).sum(-1))
            self.ag=norms(self.a,self.q); self.bg=norms(self.b,self.q).transpose(1,2,0).copy()
            self.a4g=norms(self.a4,abs(self.r)); self.b4g=norms(self.b4,abs(self.r)).transpose(1,2,0).copy()
    def bound(self,i,end):
        dp=abs(self.proj[i:end]-self.cp).transpose(0,2,1)
        if self.g==1:
            def gram(a,b,w): return ((abs(a[i:end])*w).transpose(1,0,2)@b).transpose(1,2,0)
            e=dp*self.qu[None]+BETA*gram(self.a,self.br,self.q)+BETA*gram(self.a4,self.b4r,abs(self.r))
        else:
            def gram(a,b): return (a[i:end].transpose(1,0,2)@b).transpose(1,2,0)
            e=dp*self.qu[None]+BETA*gram(self.ag,self.bg)+BETA*gram(self.a4g,self.b4g)
        pred=self.first[i:end,None]+self.second[None]-self.clean
        return pred.reshape(-1,self.p),e.reshape(-1,self.p)+EPS
    def fine_pairs(self,keys):
        i,j=keys//self.d,keys%self.d
        dp=abs(self.proj[i,:,j]-self.cp[:,j].T)
        e=dp*self.qu[j]+BETA*(abs(self.a[i]*self.b[j])*self.q).sum(-1)+BETA*(abs(self.a4[i]*self.b4[j])*abs(self.r)).sum(-1)
        return self.first[i]+self.second[j]-self.clean,e+EPS

def grouped(args,y,mode):
    t=time.perf_counter(); b=GroupBank(*args); d,p=b.d,b.p; n=d*d
    b.setup_group({'reuse':1,'group4':4,'group16':16,'global':d,'cascade4':4}[mode])
    floors=np.empty((n,p)); gap=np.zeros(p); upper=np.inf; best=-1
    for i in range(0,d,8):
        end=min(i+8,d); pred,err=b.bound(i,end); delta=abs(pred-y)
        l=np.maximum(0,delta-err)**2; h=(delta+err)**2
        floors[i*d:end*d]=l; gap+=(h-l).sum(0); hi=h.sum(1); j=int(hi.argmin())
        if hi[j]<upper: upper=float(hi[j]); best=i*d+j
    lower=floors.sum(1); score=float(cost(b.exact([best]),y)[0]); assert lower[best]<=score+EPS<=upper+2*EPS
    eligible=np.flatnonzero(lower<=score+EPS); coarse=time.perf_counter(); refined=0
    if mode=='cascade4':
        gap=np.zeros(p)
        for pos in range(0,len(eligible),128):
            ids=eligible[pos:pos+128]; pred,err=b.fine_pairs(ids); delta=abs(pred-y)
            l=np.maximum(0,delta-err)**2; h=(delta+err)**2
            floors[ids]=np.maximum(floors[ids],l); gap+=(h-floors[ids]).sum(0)
        refined=len(eligible); lower=floors.sum(1)
    order=np.argsort(lower,kind='stable'); order=order[order!=best]; cols=np.argsort(-gap,kind='stable')
    arr=[np.ascontiguousarray(a) for a in [b.w.T,b.u,b.first_z3,b.proj,b.bias,b.r,y,floors,lower,order,cols]]
    before=time.perf_counter(); best,score,calls,work,checks=finish(*arr,best,score,1,EPS); end=time.perf_counter()
    return dict(best=int(best),score=float(score),calls=int(calls),pairprobes=int(work),checks=int(checks),screen_ms=(before-t)*1000,coarse_ms=(coarse-t)*1000,finish_ms=(end-before)*1000,ms=(end-t)*1000,coarse_eligible=len(eligible),refined=refined,sstar=int((lower<=score+EPS).sum()))

def audit(args,y):
    b=GroupBank(*args); d=b.d; p=b.p; n=d*d; ref=Bank(*args); ref.setup()
    exact=np.concatenate([b.exact(np.arange(a,min(a+128,n))) for a in range(0,n,128)])
    old=[ref.bounds(i,min(i+8,d)) for i in range(0,d,8)]
    pred0=np.concatenate([r[0] for r in old]); err0=np.concatenate([r[1] for r in old]); scores=cost(exact,y); opt=int(scores.argmin()); star=float(scores[opt]); rows={}
    for g in (1,4,16,d):
        b.setup_group(g); parts=[b.bound(i,min(i+8,d)) for i in range(0,d,8)]; f=np.concatenate([r[0] for r in parts]); e=np.concatenate([r[1] for r in parts])
        assert np.allclose(f,pred0,atol=EPS,rtol=0) and np.all(abs(exact-f)<=e+EPS) and np.all(e>=err0-EPS)
        if g==1: assert np.allclose(e,err0,atol=EPS,rtol=0)
        l=(np.maximum(0,abs(f-y)-e)**2).sum(1); assert np.all(l<=scores+EPS)
        rows[str(g)]=dict(mean_envelope=float(e.mean()),sstar=int((l<=star+EPS).sum()))
    # Independent verification of the selective fine-pair evaluator.
    for a in range(0,n,128):
        ids=np.arange(a,min(a+128,n)); f,e=b.fine_pairs(ids); assert np.allclose(f,pred0[ids],atol=EPS,rtol=0) and np.allclose(e,err0[ids],atol=EPS,rtol=0)
    return dict(opt=opt,score=star,scalar_checks=4*n*p,fine_pair_checks=n*p,groups=rows)

def run(width,roots,reps):
    print(json.dumps(dict(kind='warmup',width=width,ms=warmup())),flush=True)
    for root in roots:
        for ep,(args,y,hidden,noise) in enumerate(world(width,root)):
            out={m:[] for m in MODES}
            for rep in range(reps):
                shift=(root+ep+rep)%len(MODES)
                for m in MODES[shift:]+MODES[:shift]:
                    if m=='eager':
                        t=time.perf_counter(); b=Bank(*args); k,s,n,_=decision(b,y,'eager',batch=64); r=dict(best=k,score=s,calls=n,pairprobes=n*len(y),ms=(time.perf_counter()-t)*1000)
                    elif m=='fused1': r=fused(args,y,1)
                    else:r=grouped(args,y,m)
                    out[m].append(r)
            v=audit(args,y); result={}
            for m,rr in out.items():
                assert all(r['best']==v['opt'] and abs(r['score']-v['score'])<EPS for r in rr),m
                assert len({r['pairprobes'] for r in rr})==1
                r=rr[0].copy()
                for k in ('ms','screen_ms','coarse_ms','finish_ms'):
                    if k in r:r[k]=float(np.median([z[k] for z in rr]))
                r['times_ms']=[z['ms'] for z in rr]; result[m]=r
            print(json.dumps(dict(kind='episode',width=width,root=root,episode=ep,noise=noise,hidden=hidden,hit=hidden==v['opt'],audit=v,arms=result)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--width',type=int,choices=(32,64,128),required=True); p.add_argument('--roots',type=int,nargs='+',default=[63,64,65]); p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args(); run(a.width,a.roots,a.repeats)
