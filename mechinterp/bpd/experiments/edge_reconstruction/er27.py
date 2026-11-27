"""Signed mixed-state Taylor screening, float64 with numerical slack."""
import argparse,json,time
import numpy as np
from er23 import Bank,BETA,EPS,cost
from er25 import world,screen
WIDTHS=(32,64,128)
ROOTS=(52,53,54)
MODES=('eager','base','cascade','signed')

class Signed:
    def __init__(self,b):
        self.b=b; self.h3=np.tanh(b.z3); self.hi=np.tanh(b.first_z3)
        second_write=-(b.h2@b.v.T).T[:,:,None]*b.u[:,None,:]
        self.hj=np.tanh(b.z3+second_write)
        self.z4=self.h3@b.w.T+b.bias
        self.a=(self.hi-self.h3)@b.w.T; self.c=(self.hj-self.h3)@b.w.T
        r=b.r; self.absr=abs(r)
        self.gi=(r*(1-np.tanh(self.z4+self.a)**2))@b.w
        self.gj=(r*(1-np.tanh(self.z4+self.c)**2))@b.w
        self.na=np.sqrt((self.absr*self.a**2).sum(-1)); self.nc=np.sqrt((self.absr*self.c**2).sum(-1))
        gram=b.w.T@(self.absr[:,:,None]*b.w)
        self.diag=abs(gram).sum(axis=-1)
        self.lam=np.minimum(self.diag.max(-1),self.absr.max(-1)*np.linalg.norm(b.w,2)**2)

    def interval(self,keys):
        b=self.b; i,j=keys//b.d,keys%b.d
        z3=b.first_z3[i]-b.proj[i,:,j][:,:,None]*b.u[j,None,:]
        mix=np.tanh(z3)-self.hi[i]-self.hj[j]+self.h3
        base=(np.tanh(self.z4+self.a[i]+self.c[j])*b.r).sum(-1)
        sq=mix**2; q=np.minimum((sq*self.diag).sum(-1),sq.sum(-1)*self.lam)
        rootq=np.sqrt(q)
        mi=base+(self.gi[i]*mix).sum(-1); mj=base+(self.gj[j]*mix).sum(-1)
        ei=BETA*(rootq*self.nc[j]+.5*q)+EPS; ej=BETA*(rootq*self.na[i]+.5*q)+EPS
        e0=np.sqrt(self.absr.sum(-1)*q)+EPS
        lo=np.maximum(np.maximum(mi-ei,mj-ej),base-e0)
        hi=np.minimum(np.minimum(mi+ei,mj+ej),base+e0)
        assert np.all(lo<=hi+EPS)
        return lo,hi

def interval_cost(lo,hi,y):
    return (np.maximum(np.maximum(lo-y,y-hi),0)**2).sum(-1),(np.maximum(abs(lo-y),abs(hi-y))**2).sum(-1)

def finish(b,y,lower,incumbent,score,batch=64):
    ids=np.flatnonzero((lower<=score+EPS)&(np.arange(len(lower))!=incumbent))
    order=ids[np.argsort(lower[ids],kind='stable')]; calls=1; best=incumbent
    for a in range(0,len(order),batch):
        chunk=order[a:a+batch]
        if lower[chunk[0]]>score+EPS: break
        chunk=chunk[lower[chunk]<=score+EPS]; vals=cost(b.exact(chunk),y)
        assert np.all(lower[chunk]<=vals+EPS)
        m=float(vals.min()); k=int(chunk[vals==m].min())
        if m<score or (m==score and k<best): best,score=k,m
        calls+=len(chunk)
    return best,score,calls

def run_one(args,y,mode,batch=64):
    start=time.perf_counter(); b=Bank(*args); n=b.d**2
    if mode=='eager':
        best,score=-1,np.inf
        for a in range(0,n,batch):
            v=cost(b.exact(np.arange(a,min(n,a+batch))),y); j=int(v.argmin())
            if v[j]<score: best,score=a+j,float(v[j])
        return dict(best=best,score=score,calls=n,ms=(time.perf_counter()-start)*1000),None
    if mode in ('base','cascade'):
        lower,eligible,incumbent,score=screen(b,y); initial=len(eligible)+1
        if mode=="cascade": eligible=np.sort(np.append(eligible,incumbent))
    else:
        lower=np.full(n,-np.inf); eligible=np.arange(n); incumbent,score=-1,np.inf; initial=n
    before=time.perf_counter(); screened=0
    if mode in ('cascade','signed'):
        sb=Signed(b); best_upper,upper_id=np.inf,-1
        for a in range(0,len(eligible),256):
            ids=eligible[a:a+256]; l,h=sb.interval(ids); ll,uu=interval_cost(l,h,y)
            lower[ids]=np.maximum(lower[ids],ll); j=int(uu.argmin())
            if uu[j]<best_upper: best_upper,upper_id=float(uu[j]),int(ids[j])
        screened=len(eligible)
        if mode=='signed':
            incumbent=upper_id; score=float(cost(b.exact([incumbent]),y)[0])
            assert lower[incumbent]-EPS<=score<=best_upper+EPS
    after=time.perf_counter(); best,score,calls=finish(b,y,lower,incumbent,score,batch); end=time.perf_counter()
    return dict(best=best,score=score,calls=calls,initial=initial,screened=screened,prep_ms=(before-start)*1000,signed_ms=(after-before)*1000,finish_ms=(end-after)*1000,ms=(end-start)*1000),lower

def audit(args,y):
    b=Bank(*args); s=Signed(b); n=b.d**2; exact=[]; low=[]; high=[]; old=[]; widths=[]; b.setup()
    for a in range(0,b.d,8):
        end=min(a+8,b.d); ids=np.arange(a*b.d,end*b.d)
        for z in range(0,len(ids),128):
            k=ids[z:z+128]; f=b.exact(k); l,h=s.interval(k)
            assert np.all(f>=l-EPS) and np.all(f<=h+EPS)
            exact.append(cost(f,y)); ll,hh=interval_cost(l,h,y); low.append(ll); high.append(hh)
            assert np.all(ll<=exact[-1]+EPS) and np.all(exact[-1]<=hh+EPS)
            widths.append((h-l).sum())
        p,e=b.bounds(a,end); old.append((np.maximum(0,abs(p-y)-e)**2).sum(-1))
    E=np.concatenate(exact); L=np.concatenate(low); U=np.concatenate(high); A=np.concatenate(old)
    k=int(E.argmin()); star=E[k]; assert np.all(A<=E+EPS)
    return E,dict(opt=k,score=float(star),scalar_interval_checks=n*b.p,cost_checks=n,base_sstar=int((A<=star+EPS).sum()),signed_sstar=int((L<=star+EPS).sum()),intersection_sstar=int((np.maximum(A,L)<=star+EPS).sum()),mean_signed_interval_width=float(sum(widths)/(n*b.p)))

def run(width,roots,repeats):
    for root in roots:
        for episode,(args,y,hidden,noise) in enumerate(world(width,root)):
            rows={m:[] for m in MODES}; lower={}
            for rep in range(repeats):
                shift=(root+episode+rep)%len(MODES)
                for mode in MODES[shift:]+MODES[:shift]:
                    r,lo=run_one(args,y,mode); rows[mode].append(r); lower[mode]=lo
            exact,verify=audit(args,y); results={}
            for mode,rr in rows.items():
                assert all(r['best']==verify['opt'] and abs(r['score']-verify['score'])<EPS for r in rr)
                if lower[mode] is not None: assert np.all(lower[mode]<=exact+EPS)
                r=rr[0].copy()
                for k in ('ms','prep_ms','signed_ms','finish_ms'):
                    if k in r:r[k]=float(np.median([v[k] for v in rr]))
                if mode=='cascade': assert int((lower[mode]<=verify['score']+EPS).sum())==verify['intersection_sstar']
                results[mode]=r
            print(json.dumps(dict(width=width,root=root,episode=episode,noise=noise,hidden=hidden,hit=hidden==verify['opt'],audit=verify,arms=results)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--width',type=int,choices=WIDTHS,required=True)
    p.add_argument('--roots',type=int,nargs='+',default=ROOTS); p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args(); run(a.width,a.roots,a.repeats)
