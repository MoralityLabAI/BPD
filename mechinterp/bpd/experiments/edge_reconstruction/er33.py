"""ER-33: bounded setup temporaries; release screen tensors before exact finish."""
import argparse,json,resource,time
from pathlib import Path
import numpy as np
from er23 import BETA,EPS,cost
from er30 import finish,warmup
from er32 import RankBank,load,payload

class BlockBank(RankBank):
    def __init__(self,w,o,u,v,x,c,block=8):
        self.w,self.u,self.v=w,u,v; self.d,self.p=len(u),len(x)
        self.block=block; self.r,self.bias=c@o,.25*x
        h1=np.tanh(x@w.T+self.bias); z2=h1@w.T+self.bias
        self.h2=np.tanh(z2); self.z3=self.h2@w.T+self.bias
        shape=(self.d,self.p,w.shape[0])
        self.first_z3=np.empty(shape); self.proj=np.empty((self.d,self.p,self.d))
        reads=(h1@v.T).T
        for i in range(0,self.d,block):
            j=min(i+block,self.d); h=reads[i:j,:,None]*u[i:j,None,:]
            np.subtract(z2,h,out=h); np.tanh(h,out=h)
            self.first_z3[i:j]=h@w.T+self.bias; self.proj[i:j]=h@v.T
    def setup_rank(self):
        self.cp=self.h2@self.v.T; self.absuT=abs(self.u).T.copy()
        h3=np.tanh(self.z3); z4=h3@self.w.T+self.bias; ar=abs(self.r)
        self.clean=(np.tanh(z4)*self.r).sum(-1)
        self.first=np.empty((self.d,self.p)); self.second=np.empty_like(self.first)
        self.q=ar@abs(self.w); self.qu=(self.q@self.absuT).T
        self.wa=np.empty_like(self.first_z3)
        self.wa4=np.empty((self.p,self.d,self.w.shape[0]))
        self.b4r=np.empty((self.p,self.w.shape[0],self.d))
        for i in range(0,self.d,self.block):
            j=min(i+self.block,self.d); s=slice(i,j)
            a=self.first_z3[s]-self.z3; np.abs(a,out=a); a*=self.q; self.wa[s]=a; del a
            h=np.tanh(self.first_z3[s]); h-=h3; a4=h@self.w.T; del h
            self.first[s]=(np.tanh(z4+a4)*self.r).sum(-1)
            np.abs(a4,out=a4); a4*=ar; self.wa4[:,s]=a4.transpose(1,0,2); del a4
            h=self.cp.T[s,:,None]*self.u[s,None,:]; np.subtract(self.z3,h,out=h); np.tanh(h,out=h); h-=h3
            b4=h@self.w.T; del h
            self.second[s]=(np.tanh(z4+b4)*self.r).sum(-1)
            np.abs(b4,out=b4); self.b4r[:,:,s]=b4.transpose(1,2,0)


def rss():
    with open('/proc/self/status') as f:
        current=next(int(l.split()[1]) for l in f if l.startswith('VmRSS:'))
    return dict(rss_kib=current,peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def solve(args,y,mode,block):
    t=time.perf_counter(); stages={'start':rss()}
    b=RankBank(*args) if mode=='rank' else BlockBank(*args,block=block)
    stages['prefix']=rss(); prefix_ms=(time.perf_counter()-t)*1000
    b.setup_rank(); stages['setup']=rss(); setup_ms=(time.perf_counter()-t)*1000-prefix_ms
    state=payload(b); d,p=b.d,b.p; floors=np.empty((d*d,p)); gap=np.zeros(p); upper=np.inf; best=-1
    for i in range(0,d,8):
        end=min(i+8,d); pred,err=b.bound(i,end); delta=abs(pred-y)
        l=np.maximum(0,delta-err)**2; h=(delta+err)**2
        floors[i*d:end*d]=l; gap+=(h-l).sum(0); hi=h.sum(1); j=int(hi.argmin())
        if hi[j]<upper: upper=float(hi[j]); best=i*d+j
    lower=floors.sum(1); score=float(cost(b.exact([best]),y)[0]); assert lower[best]<=score+EPS<=upper+2*EPS
    order=np.argsort(lower,kind='stable'); order=order[order!=best]; cols=np.argsort(-gap,kind='stable')
    stages['screen']=rss(); del pred,err,delta,l,h,hi,gap
    if mode=='block_release':
        for name in ('wa','wa4','b4r','cp','absuT','q','qu','first','second','clean','h2','z3','v'):
            delattr(b,name)
    retained=payload(b); stages['before_finish']=rss()
    arr=[np.ascontiguousarray(a) for a in [b.w.T,b.u,b.first_z3,b.proj,b.bias,b.r,y,floors,lower,order,cols]]
    before=time.perf_counter(); best,score,calls,work,checks=finish(*arr,best,score,1,EPS); end=time.perf_counter(); stages['finish']=rss()
    return dict(best=int(best),score=float(score),calls=int(calls),pairprobes=int(work),checks=int(checks),state_bytes=state,finish_state_bytes=retained,floor_bytes=floors.nbytes,prefix_ms=prefix_ms,setup_ms=setup_ms,screen_ms=(before-t)*1000,finish_ms=(end-before)*1000,ms=(end-t)*1000,stages=stages)


def audit(path,block):
    args,y,hidden,noise=load(path); a=RankBank(*args); a.setup_rank(); b=BlockBank(*args,block=block); b.setup_rank()
    d,p=b.d,b.p; checks=0; maxdiff=0.; best=-1; score=np.inf
    for i in range(0,d,8):
        end=min(i+8,d); pa,ea=a.bound(i,end); pb,eb=b.bound(i,end); ids=np.arange(i*d,end*d)
        assert np.allclose(pa,pb,atol=EPS,rtol=0) and np.allclose(ea,eb,atol=EPS,rtol=0)
        maxdiff=max(maxdiff,float(abs(ea-eb).max()))
        for j in range(0,len(ids),64):
            k=ids[j:j+64]; exact=a.exact(k); other=b.exact(k)
            assert np.allclose(exact,other,atol=EPS,rtol=0)
            assert np.all(abs(exact-pb[j:j+64])<=eb[j:j+64]+EPS)
            v=cost(exact,y); q=int(v.argmin())
            if v[q]<score: best,score=int(k[q]),float(v[q])
        checks+=len(ids)*p
    return dict(path=path.name,best=best,score=score,hidden=hidden,noise=noise,checks=checks,max_envelope_difference=maxdiff)


def bench(paths,mode,block,reps):
    warm=warmup(); base=rss()
    for path in paths:
        args,y,hidden,noise=load(path); rows=[solve(args,y,mode,block) for _ in range(reps)]
        for r in rows: assert r['best']==rows[0]['best'] and abs(r['score']-rows[0]['score'])<EPS and r['pairprobes']==rows[0]['pairprobes']
        r=rows[0].copy()
        for k in ('ms','prefix_ms','setup_ms','screen_ms','finish_ms'): r[k]=float(np.median([x[k] for x in rows]))
        print(json.dumps(dict(path=path.name,mode=mode,block=block,hidden=hidden,noise=noise,baseline=base,peak_kib=rss()['peak_kib'],compile_ms=warm,times_ms=[x['ms'] for x in rows],stage_repetitions=[x['stages'] for x in rows],**r)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('audit','bench')); p.add_argument('paths',type=Path,nargs='+')
    p.add_argument('--mode',choices=('rank','block','block_release'),default='block_release'); p.add_argument('--block',type=int,default=8); p.add_argument('--reps',type=int,default=2)
    a=p.parse_args()
    if a.action=='audit':
        for path in a.paths: print(json.dumps(audit(path,a.block)),flush=True)
    else: bench(a.paths,a.mode,a.block,a.reps)
