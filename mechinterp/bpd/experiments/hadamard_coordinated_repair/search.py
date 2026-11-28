"""HAD-04: integer exact cached coordinated repair; development experiment."""
import argparse, hashlib, json, time, resource
from pathlib import Path
import numpy as np

class State:
    def __init__(self,x):
        self.x=np.asarray(x,dtype=np.int64).reshape(-1).copy(); self.m=len(self.x)//4
        assert self.m%2==1 and self.m>=3 and len(self.x)==4*self.m and np.all(abs(self.x)==1)
        self.n=len(self.x); self.t=np.arange(1,self.m//2+1); a=np.arange(self.n); p=a%self.m; b=(a//self.m)*self.m
        self.r=(self.x[:,None]*self.x[b[:,None]+(p[:,None]+self.t)%self.m]).sum(0)
        self.d=-2*self.x[:,None]*(self.x[b[:,None]+(p[:,None]+self.t)%self.m]+self.x[b[:,None]+(p[:,None]-self.t)%self.m])
        self.s=self.x.reshape(4,self.m).sum(1)
    @property
    def energy(self): return int(self.r@self.r)
    @property
    def nbytes(self): return sum(a.nbytes for a in (self.x,self.r,self.d,self.s,self.t))
    def singles(self): return self.energy+2*(self.d@self.r)+(self.d*self.d).sum(1)
    def pairs(self,i,j,single=None):
        i=np.asarray(i); j=np.asarray(j); assert np.all(i!=j)
        v=self.singles() if single is None else single
        out=v[i]+v[j]-self.energy+2*(self.d[i]*self.d[j]).sum(1)
        mask=i//self.m==j//self.m; a=i[mask]; b=j[mask]; t=np.abs(a-b); t=np.minimum(t,self.m-t)-1
        c=4*self.x[a]*self.x[b]; out[mask]+=2*c*(self.r[t]+self.d[a,t]+self.d[b,t])+c*c
        return out
    def flip(self,i):
        i=int(i); m=self.m; self.r+=self.d[i]; self.s[i//m]-=2*self.x[i]
        a=np.arange(i//m*m,(i//m+1)*m); a=a[a!=i]; t=np.abs(a-i); t=np.minimum(t,m-t)-1
        self.d[a,t]+=4*self.x[i]*self.x[a]; self.d[i]*=-1; self.x[i]*=-1
    def certificate(self): return self.x.reshape(4,self.m).tolist()

def direct(x):
    x=np.array(x,dtype=np.int64); m=x.shape[1]
    return np.array([(x*np.roll(x,-t,axis=1)).sum() for t in range(1,m//2+1)])

def certify(x):
    x=np.array(x,dtype=np.int64); m=x.shape[1]; assert x.shape==(4,m) and np.all(abs(x)==1)
    if np.any(direct(x)): return False
    i=np.arange(m); A,B,C,D=[s[(i[None,:]-i[:,None])%m] for s in x]; J=np.eye(m,dtype=np.int64)[:,::-1]
    H=np.block([[A,B@J,C@J,D@J],[-B@J,A,-D.T@J,C.T@J],[-C@J,D.T@J,A,-B.T@J],[-D@J,-C.T@J,B.T@J,A]])
    return bool(np.array_equal(H@H.T,4*m*np.eye(4*m,dtype=np.int64)))

def pairs_for(n,rank,k):
    a=np.repeat(rank[:k],n); b=np.tile(np.arange(n),k); priority=np.full(n,n); priority[rank[:k]]=np.arange(k)
    ok=(a!=b)&(priority[a]<priority[b]); return a[ok],b[ok]

def row_sums(m):
    import itertools
    odd=range(1,min(m,int((4*m)**.5))+1,2)
    return [s for s in itertools.combinations_with_replacement(odd,4) if sum(v*v for v in s)==4*m]

def verify():
    rng=np.random.default_rng(84900); checks=0
    for m in [3,5,7,11,17,29,59]:
        for _ in range(3):
            s=State(rng.choice([-1,1],size=(4,m))); n=s.n; i,j=np.triu_indices(n,1)
            for _ in range(12):
                single=s.singles(); ij=rng.choice(len(i),size=min(64,len(i)),replace=False); out=s.pairs(i[ij],j[ij],single)
                for a,b,v in zip(i[ij],j[ij],out):
                    x=s.x.copy(); x[[a,b]]*=-1; r=direct(x.reshape(4,m)); assert v==r@r; checks+=1
                a=int(rng.integers(n)); pred=single[a]; s.flip(a); ref=State(s.x); assert pred==s.energy and np.array_equal(ref.r,s.r) and np.array_equal(ref.d,s.d) and np.array_equal(ref.s,s.s); checks+=1
            for k in [1,4,n]:
                a,b=pairs_for(n,rng.permutation(n),min(k,n)); assert len(set(tuple(sorted(z)) for z in zip(a,b)))==len(a); checks+=1
    assert certify([[1,1,1],[1,1,-1],[1,1,-1],[1,1,-1]])
    return checks+1

def search(x,seed,arm,budget,k=4,cache_bytes=1<<20,tenure=7):
    rng=np.random.default_rng(seed); s=State(x); n=s.n; cost=0; it=0; flips=0; paircalls=0; best=s.energy; cert=None; tabu=np.zeros(n,dtype=int); peak=s.nbytes
    # Explicit memory bound for persistent map plus pair-scoring workspace estimate; no RSS claim.
    if cache_bytes<s.nbytes: raise ValueError('cache cap smaller than state map')
    start=time.perf_counter()
    while s.energy and cost+n<=budget:
        e=s.energy; vals=s.singles(); cost+=n; it+=1
        tie=rng.random(n); rank=np.lexsort((tie,vals)); use_tabu=arm in ('tabu','pair_tabu'); admiss=(tabu<=it)|(vals<best) if use_tabu else np.ones(n,dtype=bool)
        eligible=rank[admiss[rank]]; a=int(eligible[0] if len(eligible) else rank[0]); move=[a]; chosen=int(vals[a]);
        if arm in ('pair','pair_tabu','full_pair') and chosen>=e:
            # Rank first moves by single cost; evaluate distinct unordered pairs only.
            kk=n if arm=='full_pair' else min(k,n); ii,jj=pairs_for(n,rank,kk)
            count=min(len(ii),budget-cost)
            if count:
                # Do not bias an incomplete full scan towards early candidates.
                if count<len(ii): pick=rng.choice(len(ii),size=count,replace=False); ii,jj=ii[pick],jj[pick]
                # Actual arrays + conservative per-pair scratch estimate. Cap controls batch width.
                persistent=s.nbytes+ii.nbytes+jj.nbytes+vals.nbytes
                per=8*(4*len(s.r)+20); width=(cache_bytes-persistent)//per
                if width<1: raise ValueError('pair workspace exceeds cache cap; raise cap or lower shortlist')
                cost+=count; paircalls+=count
                for lo in range(0,count,width):
                    i,j=ii[lo:lo+width],jj[lo:lo+width]; v=s.pairs(i,j,vals); peak=max(peak,persistent+len(i)*per)
                    if use_tabu: v=np.where(((tabu[i]<=it)&(tabu[j]<=it))|(v<best),v,np.iinfo(np.int64).max)
                    z=int(v.argmin()); vv=int(v[z]);
                    if vv<chosen: move=[int(i[z]),int(j[z])]; chosen=vv
        if chosen>=e and not use_tabu and len(move)==1: move=[int(rng.integers(n))]; chosen=int(vals[move[0]])
        for a in move: s.flip(a); tabu[a]=it+tenure; flips+=1
        assert s.energy==chosen
        best=min(best,s.energy)
    elapsed=time.perf_counter()-start
    if s.energy==0:
        cert=s.certificate(); assert certify(cert)
        assert int(s.s@s.s)==4*s.m
        assert np.max(abs((np.abs(np.fft.fft(np.array(cert),axis=1))**2).sum(0)-4*s.m))<1e-6
    return dict(solved=cert is not None,best=best,final=s.energy,candidate_scores=cost,pair_scores=paircalls,iterations=it,flips=flips,seconds=elapsed,tracked_workspace_bytes=peak,state_bytes=s.nbytes,certificate=cert)

def run(args):
    checks=verify(); rows=[]; start=time.perf_counter(); arms=args.arms.split(',')
    for m in args.lengths:
        assert row_sums(m)
        for root in range(args.root,args.root+args.roots):
            for case in range(args.cases):
                seed=root*100000+m*100+case; x=np.random.default_rng(seed).choice([-1,1],size=(4,m))
                h=hashlib.sha256(x.astype('int8').tobytes()).hexdigest()
                for arm in arms:
                    r=search(x,seed+1,arm,args.budget,args.shortlist,args.cache_mib*(1<<20)); r.update(m=m,root=root,case=case,seed=seed,start_hash=h,arm=arm); rows.append(r)
            print('length',m,'root',root,'complete',flush=True)
    result=dict(scope='DEVELOPMENT_ONLY',args=vars(args),checks=checks,seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rows=rows)
    Path(args.out).write_text(json.dumps(result,indent=2))
    for m in args.lengths:
        for arm in arms:
            a=[r for r in rows if r['m']==m and r['arm']==arm]; print(m,arm,sum(r['solved'] for r in a),len(a),'mean_best',np.mean([r['best'] for r in a]),'seconds',round(sum(r['seconds'] for r in a),3),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--lengths',nargs='+',type=int,default=[17,23,29]); p.add_argument('--root',type=int,default=84101); p.add_argument('--roots',type=int,default=4); p.add_argument('--cases',type=int,default=8); p.add_argument('--budget',type=int,default=16384); p.add_argument('--shortlist',type=int,default=4); p.add_argument('--cache-mib',type=int,default=4); p.add_argument('--arms',default='greedy,tabu,pair,pair_tabu,full_pair'); p.add_argument('--out',default='results.json'); run(p.parse_args())
