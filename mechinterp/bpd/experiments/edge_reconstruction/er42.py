"""ER-42: compact exact history and lexicographic top-K selection."""
import argparse,json,time
from pathlib import Path
import numpy as np
from numba import njit
from er23 import Bank,EPS,cost,decision
from er35 import StreamBank,BLOCK
from er33 import rss
from er32 import load
from er38 import unlink,append
from er42_select import choose,reference

MODES=('lru_budget','freq_budget','learned','learned_sort')
CAP=128

# Fold the frozen ER-40 standardization into the linear coefficients.
def model(path):
    r=json.loads(Path(path).read_text()); assert r['capacity']==128 and r['alphas']['learned']==.25 and r['alphas']['freq']==0
    m=r['model']; w=np.array(m['coef'])/np.array(m['scale']); b=m['intercept']-w@np.array(m['mean'])
    return w,float(b)

def bytes_for(d,width,p,cap,policy):
    # Resident vectors, key/slot mapping, and policy-specific persistent arrays.
    common=8*(cap*(width+d+1)+d*p)
    if policy==0: return common+8*(2*cap+3)
    if policy==1: return common+8*d*p
    return common+28*d*p+8*(d+2*p+17)

def config(d,width,p,mode):
    policy=2 if mode.startswith('learned') else int(mode.startswith('freq'))
    budget=bytes_for(d,width,p,CAP,2); cap=CAP
    if mode.endswith('budget'):
        while bytes_for(d,width,p,cap+1,policy)<=budget: cap+=1
    return policy,cap,budget,bytes_for(d,width,p,cap,policy)

@njit(cache=True,fastmath=False)
def finish(wt,vt,u,z2,reads,bias,r,y,floors,lower,order,cols,best,score,cap,policy,w,b0,eps,partition):
    d,width=u.shape; p=len(y); n=d*p; work=p; calls=1
    loc=np.full(n,-1,np.int64); keys=np.full(cap,-1,np.int64)
    cz=np.empty((cap,width)); cp=np.empty((cap,d)); used=0
    prev=np.full(cap if policy==0 else 0,-1,np.int64); nxt=prev.copy()
    head=np.full(1 if policy==0 else 0,-1,np.int64); tail=head.copy()
    assert ((d*d+63)//64)*p+1 < 2147483647
    last=np.full(n if policy else 0,-1,np.int32); counts=np.zeros(n if policy else 0,np.int32)
    gap=np.zeros(n if policy==2 else 0,np.int32); ew=np.zeros(n if policy==2 else 0); base=ew.copy()
    comp=np.zeros(d if policy==2 else 0); probe=np.zeros(p if policy==2 else 0)
    rank=np.argsort(cols) if policy==2 else np.empty(0,np.int64)
    t=total=hits=misses=dedup=bypass=nonempty=0
    for start in range(0,len(order),64):
        if lower[order[start]]>score+eps: break
        ids=order[start:start+64].copy(); ids=ids[lower[ids]<=score+eps]; terms=floors[ids].copy()
        for a in range(p):
            if not len(ids): break
            k=cols[a]; m=len(ids)
            group=np.full(d,-1,np.int64); first=np.empty(m,np.int64); gi=np.empty(m,np.int64); ng=0
            for j0 in range(m):
                i=ids[j0]//d
                if group[i]<0: group[i]=ng; first[ng]=i; ng+=1
                gi[j0]=group[i]
            dedup+=m-ng
            z=np.empty((ng,width)); pr=np.empty((ng,d)); miss=np.empty(ng,np.int64); nm=0
            for g in range(ng):
                key=first[g]*p+k; slot=loc[key]
                if slot>=0:
                    z[g]=cz[slot]; pr[g]=cp[slot]; hits+=1
                    if policy==0:
                        unlink(slot,0,prev,nxt,head,tail); append(slot,0,prev,nxt,head,tail)
                else: miss[nm]=g; nm+=1; misses+=1
            if nm:
                nonempty+=1; hh=np.empty((nm,width))
                for j0 in range(nm):
                    i=first[miss[j0]]
                    for l in range(width): hh[j0,l]=np.tanh(z2[k,l]-reads[i,k]*u[i,l])
                zz=hh@wt; pp=hh@vt
                for j0 in range(nm):
                    g=miss[j0]; z[g]=zz[j0]+bias[k]; pr[g]=pp[j0]
            if policy==0:
                for j0 in range(nm):
                    g=miss[j0]; key=first[g]*p+k
                    if used<cap: slot=used; used+=1
                    else:
                        slot=head[0]; loc[keys[slot]]=-1; unlink(slot,0,prev,nxt,head,tail)
                    keys[slot]=key; loc[key]=slot; cz[slot]=z[g]; cp[slot]=pr[g]; append(slot,0,prev,nxt,head,tail)
            else:
                # Snapshot causal features before updating the batch history.
                for g in range(ng):
                    i=first[g]; key=i*p+k; c=float(counts[key]); dt=t-int(last[key])
                    if policy==2:
                        base[key]=(b0+w[0]*np.log2(d)+w[1]*i/max(1,d-1)+w[2]*rank[k]/max(1,p-1)
                            +w[3]*np.log1p(t)+w[4]*np.log1p(ng)+w[5]*np.log1p(c)+w[6]*np.log1p(dt)
                            +w[7]*np.log1p(float(gap[key]))+w[8]*np.log1p(ew[key])+w[9]*np.log1p(comp[i])
                            +w[10]*np.log1p(probe[k])+w[11]*np.log1p(c/(1+t))+w[12]*np.log1p(probe[k]/max(1,total))
                            +w[13]*g/max(1,ng-1)+w[14]*(c==0)+w[15]*np.log2(CAP/d))
                        ew[key]=0. if c==0 else (dt if c==1 else .8*ew[key]+.2*dt)
                        gap[key]=dt if c>0 else 0.
                    last[key]=t; counts[key]+=1
                if policy==2:
                    for g in range(ng): comp[first[g]]+=1
                    probe[k]+=ng
                total+=ng
                # Union of resident and newly requested absent entries, then top K.
                pool=np.empty(used+nm,np.int64); pool[:used]=keys[:used]
                for j0 in range(nm): pool[used+j0]=first[miss[j0]]*p+k
                if len(pool)>cap:
                    vals=np.empty(len(pool))
                    for j0 in range(len(pool)):
                        key=pool[j0]; vals[j0]=base[key]-.25*np.log1p(t-last[key]) if policy==2 else counts[key]
                    # Stable sorts reproduce descending score, recency, then ascending ID.
                    chosen=choose(vals,last[pool],pool,cap) if partition else reference(vals,last[pool],pool,cap)
                    free=np.empty(cap,np.int64); nf=0
                    for j0 in range(used):
                        if not chosen[j0]: loc[keys[j0]]=-1; free[nf]=j0; nf+=1
                    for j0 in range(used,cap): free[nf]=j0; nf+=1
                    f=0
                    for j0 in range(nm):
                        if chosen[used+j0]:
                            g=miss[j0]; key=first[g]*p+k; slot=free[f]; f+=1
                            keys[slot]=key; loc[key]=slot; cz[slot]=z[g]; cp[slot]=pr[g]
                        else: bypass+=1
                    assert f==nf; used=cap
                else:
                    for j0 in range(nm):
                        g=miss[j0]; key=first[g]*p+k; slot=used; used+=1
                        keys[slot]=key; loc[key]=slot; cz[slot]=z[g]; cp[slot]=pr[g]
            h=np.empty((m,width))
            for j0 in range(m):
                j=ids[j0]%d; g=gi[j0]; s=pr[g,j]
                for l in range(width): h[j0,l]=np.tanh(z[g,l]-s*u[j,l])
            zz=h@wt; energy=np.empty(m)
            for j0 in range(m):
                f=0.
                for l in range(width): f+=np.tanh(zz[j0,l]+bias[k,l])*r[k,l]
                val=(f-y[k])**2; assert terms[j0,k]<=val+eps; terms[j0,k]=val; energy[j0]=terms[j0].sum()
            work+=m; t+=1
            if a+1>=p:
                calls+=m
                for j0 in range(m):
                    val=energy[j0]; key=ids[j0]
                    if val<score or (val==score and key<best): best,score=key,val
                break
            keep=energy<=score+eps; ids=ids[keep]; terms=terms[keep]
    assert hits+misses+dedup==work-p
    for j0 in range(used): assert loc[keys[j0]]==j0
    return best,score,calls,work,hits,misses,dedup,bypass,nonempty

def solve(args,y,mode,w,b0):
    t=time.perf_counter(); b=StreamBank(*args,'replay'); d,p=b.d,b.p; width=b.u.shape[1]
    floors=np.empty((d*d,p)); gaps=np.zeros(p); upper=np.inf; best=-1
    for i in range(0,d,BLOCK):
        end=min(i+BLOCK,d); pred,err=b.bound(i,end); delta=abs(pred-y)
        l=np.maximum(0,delta-err)**2; h=(delta+err)**2; floors[i*d:end*d]=l; gaps+=(h-l).sum(0)
        hi=h.sum(1); j=int(hi.argmin())
        if hi[j]<upper: upper=float(hi[j]); best=i*d+j
    lower=floors.sum(1); score=float(cost(b.exact_incumbent(best),y)); assert lower[best]<=score+EPS<=upper+2*EPS
    order=np.argsort(lower,kind='stable'); order=order[order!=best]; cols=np.argsort(-gaps,kind='stable'); incumbent=best
    arr=[np.ascontiguousarray(a) for a in [b.w.T,b.v.T,b.u,b.z2,b.reads,b.bias,b.r,y,floors,lower,order,cols]]; del b,pred,err,delta,l,h,hi
    policy,cap,budget,payload=config(d,width,p,mode); before=time.perf_counter()
    v=finish(*arr,best,score,cap,policy,w,b0,EPS,mode!='learned_sort'); end=time.perf_counter()
    out=dict(zip(('best','score','calls','pairprobes','hits','misses','dedup','bypass','nonempty_miss_batches'),v))
    return dict(**out,incumbent=incumbent,capacity=cap,budget_bytes=budget,payload_bytes=payload,screen_ms=(before-t)*1000,finish_ms=(end-before)*1000,ms=(end-t)*1000)

def warm(w,b0):
    rng=np.random.default_rng(41); d=6; w0,o,u,v=[rng.standard_normal((d,d))*.1 for _ in range(4)]; x,c=rng.standard_normal((2,12,d)); y=rng.standard_normal(12)
    for mode in MODES: solve((w0,o,u,v,x,c),y,mode,w,b0)

def main():
    p=argparse.ArgumentParser(); p.add_argument('paths',type=Path,nargs='+'); p.add_argument('--model',required=True); p.add_argument('--mode',choices=MODES+('audit',),required=True); p.add_argument('--reps',type=int,default=2); a=p.parse_args(); w,b0=model(a.model)
    if a.mode!='audit': warm(w,b0)
    baseline=rss()
    for path in a.paths:
        args,y,hidden,noise=load(path)
        if a.mode=='audit':
            best,score,n,_=decision(Bank(*args),y,'eager',batch=64); print(json.dumps(dict(path=path.name,best=best,score=score,exact_pairs=n,hidden=hidden)),flush=True); continue
        rows=[solve(args,y,a.mode,w,b0) for _ in range(a.reps)]; r=rows[0].copy()
        for x in rows: assert all(x[k]==r[k] for k in ('best','pairprobes','misses','bypass','dedup')) and abs(x['score']-r['score'])<EPS
        for k in ('ms','screen_ms','finish_ms'): r[k]=float(np.median([x[k] for x in rows]))
        print(json.dumps(dict(path=path.name,width=args[0].shape[0],mode=a.mode,hidden=hidden,noise=noise,baseline=baseline,peak_kib=rss()['peak_kib'],times_ms=[x['ms'] for x in rows],**r)),flush=True)

if __name__=='__main__': main()
