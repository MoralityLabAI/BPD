"""HAD-05C: partial LDT pair joins and guarded population injection (development)."""
import argparse, json, time
from pathlib import Path
import numpy as np
from archive import Archive
from had05 import Trajectory, starts, preflight, certify

class NearArchive(Archive):
    def __init__(self,m,cap,mib,k=4):
        super().__init__(m,cap,mib); self.k=k; self.buckets={}; self.offer=None; self.used=set(); self.near_checks=0
        # Separate conservative encoded payload for bucket keys and at most k pair references.
        self.bucket_max=self.pair_cap*(32+k*16)
        if self.max_payload+self.bucket_max>mib*(1<<20): raise ValueError('near archive encoded cap exceeded')
    def reset_offer(self,limit): self.offer=None; self.limit=limit
    def bucket(self,r,dc,q): return (int(dc),int(q),int(r[0]),int(r[-1]))
    def add(self,row,source,budget):
        if self.cert is not None: return 0
        if len(self.rows)>=self.row_cap: self.cap_skips+=1; return 0
        if budget<self.m+(len(self.rows)+1)*(1+self.k): return 0
        m=self.m; self.offers+=1; work=m; row=np.asarray(row,dtype=np.int8).copy(); assert row.shape==(m,) and np.all(abs(row)==1)
        s=abs(int(row.sum(dtype=np.int64))); q=0
        if m%3==0:
            a,b,c=(int(row[k::3].sum(dtype=np.int64)) for k in range(3)); q=a*a+b*b+c*c-a*b-b*c-c*a
        if s not in self.allowed_s or q>4*m: self.row_rejected+=1; return work
        r=np.array([np.dot(row.astype(np.int64),np.roll(row,-t).astype(np.int64)) for t in range(1,self.h+1)],dtype='<i2'); key=r.tobytes()
        if key in self.row_keys: self.duplicates+=1; return work
        i=len(self.rows); self.rows.append((row,r,s*s,q,tuple(source))); self.row_keys.add(key)
        for j,(_,v,dc,qp,_) in enumerate(self.rows):
            work+=1; self.pair_checked+=1; dc+=s*s; qp+=q
            if dc not in self.allowed_pair_dc or qp>4*m: self.pair_rejected+=1; continue
            k=(r+v).astype('<i2',copy=False); kk=k.tobytes(); other=self.pairs.get((-k).astype('<i2',copy=False).tobytes())
            if other is not None or np.all(k==0):
                ids=(i,j)+((i,j) if other is None else other); self.cert=[self.rows[t][0].tolist() for t in ids]; self.sources=[self.rows[t][4] for t in ids]; assert certify(self.cert); return work
            if kk in self.pairs: continue
            # Exact commitments at DC, order 3 (when defined), and shifts 1 and floor(m/2).
            match=self.bucket(-k,4*m-dc,4*m-qp if m%3==0 else 0)
            for key0,ij in self.buckets.get(match,[]):
                self.near_checks+=1; work+=1; ids=tuple(sorted((i,j)+ij)); residual=k.astype(np.int64)+np.frombuffer(key0,dtype='<i2').astype(np.int64); e=int(residual@residual)
                if ids not in self.used and e<=self.limit and (self.offer is None or e<self.offer[0]): self.offer=(e,ids)
            self.pairs[kk]=(i,j); bk=self.bucket(k,dc,qp); b=self.buckets.setdefault(bk,[]); b.append((kk,(i,j)))
            if len(b)>self.k: b.pop(0)
        assert work<=budget; return work
    def observe(self,trajectory,slot,budget,rng=None):
        work=0
        for k,row in enumerate(trajectory.state.matrix()):
            if rng is not None: row=rng.permutation(row)
            work+=self.add(row,(slot,trajectory.blocks,k,getattr(trajectory,'generation',0)),budget-work)
            if self.cert is not None: break
        return work
    def near_receipt(self):
        return dict(near_checks=self.near_checks,buckets=len(self.buckets),bucket_refs=sum(map(len,self.buckets.values())),bucket_max_payload=self.bucket_max)

def canonical(x):
    # Row order and sign invariance; cyclic/reversal equivalence is not deduplicated here.
    return tuple(sorted(min(z.astype('int8').tobytes(),(-z).astype('int8').tobytes()) for z in x))

def evaluate(seed,m,p,chunk,rounds,arm,cap,mib,cache_mib):
    preflight(m,p,cache_mib); xs,ss,h=starts(seed,m,p); ts=[Trajectory(x,z,chunk) for x,z in zip(xs,ss)]; rng=np.random.default_rng(seed+1234567)
    budget=rounds*chunk*4*m; n=4*m; work=aw=0; trace=[]; injections=[]; cert=None; kind=None; solution_injection=None; begin=time.perf_counter()
    arch=None if arm=='best' else (Archive(m,cap,mib) if arm=='exact64' else NearArchive(m,cap,mib))
    def observe(t,i):
        nonlocal work,aw,cert,kind
        if arch is None: return
        if isinstance(arch,NearArchive): arch.reset_offer(min(t.best for t in ts)+32)
        w=arch.observe(t,i,budget-work,rng if arm=='random_near64' else None); work+=w; aw+=w
        if arch.cert is not None: cert=arch.cert; kind='archive'; return
        if isinstance(arch,NearArchive) and arch.offer is not None and arm!='near64_noinject' and budget-work>=n:
            e,ids=arch.offer; worst=max(range(p),key=lambda j:(ts[j].best,-ts[j].blocks,j)); child=np.stack([arch.rows[j][0] for j in ids]); key=canonical(child)
            if e<ts[worst].best and key not in [canonical(t.state.matrix()) for t in ts]:
                sources=[arch.rows[j][4] for j in ids]; old=ts[worst].best; tr=Trajectory(child,rng.integers(2**62),chunk); assert tr.best==e; assert int(tr.state.s@tr.state.s)==4*m
                from had05 import power3
                q=power3(child); assert q is None or int(q.sum())==4*m; assert tr.state.r[0]==tr.state.r[-1]==0
                tr.generation=len(injections)+1; ts[worst]=tr; arch.used.add(ids); work+=n; aw+=n; injections.append(dict(energy=e,replaced_best=old,slot=worst,generation=tr.generation,sources=sources,work=work,sequences=child.tolist()))
    if arch is not None:
        for i in range(p):
            observe(ts[i],i)
            if cert is not None: break
    for j in range(rounds):
        if cert is not None or work+n>budget: break
        solved=next((t for t in ts if t.state.energy==0),None)
        if solved: cert=solved.state.matrix().tolist(); kind='native'; solution_injection=getattr(solved,'generation',0); break
        i=min(range(p),key=lambda i:(ts[i].best,ts[i].blocks,i)); t=ts[i]; old=t.chunk; t.chunk=min(chunk,(budget-work)//n); w=t.advance(); t.chunk=old; work+=w
        if t.state.energy==0: cert=t.state.matrix().tolist(); kind='native'; solution_injection=getattr(t,'generation',0)
        else: observe(t,i)
        trace.append([i,t.best,t.state.energy,work,aw])
    if cert is not None: assert certify(cert)
    assert work<=budget
    out=dict(arm=arm,solved=cert is not None,solution_kind=kind,solution_injection=solution_injection,certificate=cert,best=0 if cert else min(t.best for t in ts),candidate_units=work,archive_candidate_units=aw,search_candidate_scores=work-aw,seconds=time.perf_counter()-begin,start_hash=h,trace=trace,injections=injections)
    if arch is not None: out.update(arch.receipt())
    if isinstance(arch,NearArchive): out.update(arch.near_receipt())
    return out

def run(a):
    begin=time.perf_counter(); rows=[]; arms=('best','exact64','near64_noinject','near64','random_near64')
    for root in range(a.root,a.root+a.roots):
        for m in a.lengths:
            for case in range(a.cases):
                seed=root*100000+m*100+case
                for arm in arms:
                    r=evaluate(seed,m,a.population,a.chunk,a.rounds,arm,a.row_cap,a.archive_mib,a.cache_mib); r.update(root=root,m=m,case=case,seed=seed); rows.append(r)
        print('root',root,'done',flush=True)
    out=dict(scope='DEVELOPMENT_ONLY',args=vars(a),rows=rows,seconds=time.perf_counter()-begin); Path(a.out).write_text(json.dumps(out,indent=2)); print('seconds',out['seconds'])
    for arm in arms:
        z=[r for r in rows if r['arm']==arm]; print(arm,[(m,sum(r['solved'] for r in z if r['m']==m)) for m in a.lengths],sum(r['solved'] for r in z),'archive',sum(r['solution_kind']=='archive' for r in z),'injections',sum(len(r['injections']) for r in z),'cross',sum(len(set(s[0] for s in t['sources']))>1 for r in z for t in r['injections']),'seconds',round(sum(r['seconds'] for r in z),3),'mean best',round(np.mean([r['best'] for r in z]),3))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=int,default=85401); p.add_argument('--roots',type=int,default=4); p.add_argument('--cases',type=int,default=6); p.add_argument('--population',type=int,default=8); p.add_argument('--chunk',type=int,default=16); p.add_argument('--rounds',type=int,default=24); p.add_argument('--lengths',type=int,nargs='+',default=[23,27,29]); p.add_argument('--row-cap',type=int,default=64); p.add_argument('--archive-mib',type=int,default=8); p.add_argument('--cache-mib',type=int,default=8); p.add_argument('--out',default='near_results.json'); run(p.parse_args())
