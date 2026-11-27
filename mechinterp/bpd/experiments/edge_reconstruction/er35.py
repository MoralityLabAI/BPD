"""ER-35: streamed left curvature and optional prefix replay.
Real arithmetic equivalence; float64 numerical audit, not interval arithmetic.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from numba import njit
from er23 import BETA, EPS, cost
from er33 import BlockBank, rss
from er32 import load, payload

MODES=('dense','stream','project','replay')
BLOCK=8

class StreamBank:
    def __init__(self,w,o,u,v,x,c,mode):
        self.w,self.u,self.v=w,u,v
        self.d,self.p=len(u),len(x); self.mode=mode
        self.r,self.bias=c@o,.25*x
        h1=np.tanh(x@w.T+self.bias)
        self.z2=h1@w.T+self.bias; self.reads=(h1@v.T).T.copy()
        h2=np.tanh(self.z2); self.cp=h2@v.T
        self.z3=h2@w.T+self.bias; self.h3=np.tanh(self.z3)
        self.z4=self.h3@w.T+self.bias; self.ar=abs(self.r)
        self.clean=(np.tanh(self.z4)*self.r).sum(-1)
        self.absuT=abs(u).T.copy(); self.q=self.ar@abs(w)
        self.qu=(self.q@self.absuT).T
        self.second=np.empty((self.d,self.p))
        self.b4r=np.empty((self.p,w.shape[0],self.d))
        # The right curvature table is shared by every left block.
        for i in range(0,self.d,BLOCK):
            end=min(i+BLOCK,self.d); s=slice(i,end)
            h=self.cp.T[s,:,None]*u[s,None,:]
            np.subtract(self.z3,h,out=h); np.tanh(h,out=h); h-=self.h3
            b4=h@w.T; del h
            self.second[s]=(np.tanh(self.z4+b4)*self.r).sum(-1)
            np.abs(b4,out=b4); self.b4r[:,:,s]=b4.transpose(1,2,0)
        self.first_z3=np.empty((self.d,self.p,w.shape[0])) if mode in ('stream','project') else np.empty((0,0,0))
        self.proj=np.empty((self.d,self.p,self.d)) if mode=='stream' else np.empty((0,0,0))

    def prefix(self,i,end):
        h=self.reads[i:end,:,None]*self.u[i:end,None,:]
        np.subtract(self.z2,h,out=h); np.tanh(h,out=h)
        return h@self.w.T+self.bias,h@self.v.T

    def bound(self,i,end):
        z,proj=self.prefix(i,end)
        if self.first_z3.size: self.first_z3[i:end]=z
        if self.proj.size: self.proj[i:end]=proj
        wa=abs(z-self.z3)*self.q
        a4=(np.tanh(z)-self.h3)@self.w.T
        first_response=(np.tanh(self.z4+a4)*self.r).sum(-1)
        wa4=(abs(a4)*self.ar).transpose(1,0,2).copy()
        dp=abs(proj-self.cp).transpose(0,2,1)
        first=(wa.reshape(-1,self.w.shape[0])@self.absuT).reshape(end-i,self.p,self.d)
        first=(first*abs(self.cp)[None]).transpose(0,2,1)
        last=(wa4@self.b4r).transpose(1,2,0)
        e=dp*self.qu[None]+BETA*first+BETA*last
        pred=first_response[:,None]+self.second[None]-self.clean
        return pred.reshape(-1,self.p),e.reshape(-1,self.p)+EPS

    def exact_incumbent(self,k):
        i,j=divmod(k,self.d); start=i//BLOCK*BLOCK
        z,pr=self.prefix(start,min(start+BLOCK,self.d)); z=z[i-start]; pr=pr[i-start,:,j]
        h=np.tanh(z-pr[:,None]*self.u[j])
        return (np.tanh(h@self.w.T+self.bias)*self.r).sum(-1)

@njit(fastmath=False)
def finish(wt,u,v,z3,proj,z2,reads,bias,r,y,floors,lower,order,cols,best,score,mode,eps):
    d=len(u); width=u.shape[1]; p=len(y); work=p; calls=1; checks=0; prefix_terms=0
    for start in range(0,len(order),64):
        if lower[order[start]]>score+eps: break
        ids=order[start:start+64].copy(); ids=ids[lower[ids]<=score+eps]
        terms=floors[ids].copy()
        for a in range(p):
            if not len(ids): break
            k=cols[a]; m=len(ids)
            h=np.empty((m,width),np.float64)
            if mode==2:
                # Replay both prefix tensors for demanded pairs/probes only.
                h2=np.empty((m,width),np.float64); pr=np.empty(m,np.float64)
                for t in range(m):
                    i,j=ids[t]//d,ids[t]%d; s=0.
                    for l in range(width):
                        val=np.tanh(z2[k,l]-reads[i,k]*u[i,l]); h2[t,l]=val; s+=val*v[j,l]
                    pr[t]=s
                zz=h2@wt
                for t in range(m):
                    j=ids[t]%d
                    for l in range(width): h[t,l]=np.tanh(zz[t,l]+bias[k,l]-pr[t]*u[j,l])
                prefix_terms+=m
            else:
                for t in range(m):
                    i,j=ids[t]//d,ids[t]%d
                    if mode==1:
                        s=0.
                        for l in range(width): s+=np.tanh(z2[k,l]-reads[i,k]*u[i,l])*v[j,l]
                        prefix_terms+=1
                    else: s=proj[i,k,j]
                    for l in range(width): h[t,l]=np.tanh(z3[i,k,l]-s*u[j,l])
            zz=h@wt; energy=np.empty(m,np.float64)
            for t in range(m):
                f=0.
                for l in range(width): f+=np.tanh(zz[t,l]+bias[k,l])*r[k,l]
                value=(f-y[k])**2
                assert terms[t,k]<=value+eps
                terms[t,k]=value; checks+=1; energy[t]=terms[t].sum()
            work+=m
            if a+1>=p:
                calls+=m
                for t in range(m):
                    value=energy[t]; key=ids[t]
                    if value<score or (value==score and key<best): best,score=key,value
                break
            keep=energy<=score+eps; ids=ids[keep]; terms=terms[keep]
    return best,score,calls,work,checks,prefix_terms

def solve(args,y,mode,details=False):
    t=time.perf_counter(); stages={'start':rss()}
    if mode=='dense':
        b=BlockBank(*args,block=BLOCK); b.setup_rank()
        w,o,u,v,x,c=args; bias=.25*x; h1=np.tanh(x@w.T+bias); z2=h1@w.T+bias; reads=(h1@v.T).T.copy()
    else:
        b=StreamBank(*args,mode); z2,reads=b.z2,b.reads
    stages['setup']=rss(); d,p=b.d,b.p; floors=np.empty((d*d,p)); gap=np.zeros(p); upper=np.inf; best=-1
    for i in range(0,d,BLOCK):
        end=min(i+BLOCK,d); pred,err=b.bound(i,end); delta=abs(pred-y)
        l=np.maximum(0,delta-err)**2; h=(delta+err)**2
        floors[i*d:end*d]=l; gap+=(h-l).sum(0); hi=h.sum(1); j=int(hi.argmin())
        if hi[j]<upper: upper=float(hi[j]); best=i*d+j
    del pred,err,delta,l,h,hi
    lower=floors.sum(1); incumbent=best
    if mode=='dense': response=b.exact([best])[0]
    else: response=b.exact_incumbent(best)
    score=float(cost(response,y)); assert lower[best]<=score+EPS<=upper+2*EPS
    order=np.argsort(lower,kind='stable'); order=order[order!=best]; cols=np.argsort(-gap,kind='stable')
    state=payload(b); stages['screen']=rss()
    # Common policy: release all screen-only tensors before refinement.
    arr=[np.ascontiguousarray(a) for a in [b.w.T,b.u,b.v,b.first_z3,b.proj,z2,reads,b.bias,b.r,y,floors,lower,order,cols]]
    del b,z2,reads
    stages['before_finish']=rss(); before=time.perf_counter()
    best,score,calls,work,checks,prefix_terms=finish(*arr,best,score,{'dense':0,'stream':0,'project':1,'replay':2}[mode],EPS)
    end=time.perf_counter(); stages['finish']=rss()
    result=dict(best=int(best),score=float(score),incumbent=int(incumbent),calls=int(calls),pairprobes=int(work),checks=int(checks),replayed_prefix_terms=int(prefix_terms),state_bytes=state,floor_bytes=floors.nbytes,screen_ms=(before-t)*1000,finish_ms=(end-before)*1000,ms=(end-t)*1000,stages=stages)
    if details: return result,floors,order,cols
    return result

def warmup():
    rng=np.random.default_rng(35); d=6
    w,o,u,v=[rng.standard_normal((d,d))*.1 for _ in range(4)]
    x,c=rng.standard_normal((2,12,d)); y=rng.standard_normal(12)
    t=time.perf_counter(); solve((w,o,u,v,x,c),y,'dense'); return (time.perf_counter()-t)*1000

def audit(path):
    args,y,hidden,noise=load(path); ref=BlockBank(*args,block=BLOCK); ref.setup_rank()
    s=StreamBank(*args,'replay'); n=ref.d; checks=0; diff=0.; best=-1; score=np.inf
    for i in range(0,n,BLOCK):
        end=min(i+BLOCK,n); a,e=ref.bound(i,end); b,f=s.bound(i,end)
        diff=max(diff,float(abs(e-f).max()))
        assert np.allclose(a,b,rtol=0,atol=EPS) and np.allclose(e,f,rtol=0,atol=EPS)
        ids=np.arange(i*n,end*n)
        for pos in range(0,len(ids),64):
            keys=ids[pos:pos+64]; exact=ref.exact(keys)
            assert np.all(abs(exact-b[pos:pos+64])<=f[pos:pos+64]+EPS)
            values=cost(exact,y); j=int(values.argmin())
            if values[j]<score: best,score=int(keys[j]),float(values[j])
        checks+=e.size
    del ref,s
    r,fl,order,cols=solve(args,y,'dense',True); out={}
    assert r['best']==best and abs(r['score']-score)<EPS
    for m in MODES[1:]:
        v,ff,oo,cc=solve(args,y,m,True)
        assert v['best']==best and abs(v['score']-score)<EPS
        assert np.array_equal(ff,fl) and np.array_equal(oo,order) and np.array_equal(cc,cols)
        assert v['incumbent']==r['incumbent'] and v['pairprobes']==r['pairprobes'],(m,v,r)
        out[m]=v['replayed_prefix_terms']
    return dict(path=path.name,best=best,score=score,hidden=hidden,noise=noise,scalar_checks=checks,max_envelope_difference=diff,work=r['pairprobes'],prefix_work=out)

def bench(paths,mode,reps):
    warm=warmup(); baseline=rss()
    for path in paths:
        args,y,hidden,noise=load(path); rows=[solve(args,y,mode) for _ in range(reps)]; r=rows[0].copy()
        for x in rows: assert x['best']==r['best'] and abs(x['score']-r['score'])<EPS and x['pairprobes']==r['pairprobes']
        for k in ('ms','screen_ms','finish_ms'): r[k]=float(np.median([x[k] for x in rows]))
        print(json.dumps(dict(path=path.name,mode=mode,hidden=hidden,noise=noise,baseline=baseline,peak_kib=rss()['peak_kib'],compile_ms=warm,times_ms=[x['ms'] for x in rows],**r)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('audit','bench')); p.add_argument('paths',type=Path,nargs='+'); p.add_argument('--mode',choices=MODES,default='stream'); p.add_argument('--reps',type=int,default=2); a=p.parse_args()
    if a.action=='audit':
        warmup()
        for path in a.paths: print(json.dumps(audit(path)),flush=True)
    else: bench(a.paths,a.mode,a.reps)
