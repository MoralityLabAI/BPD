"""HAD-05B: LDT exact 2+2 residual-complement archive; RAM-scaled recombination."""
import argparse, itertools, json, math, sys, time
from pathlib import Path
import numpy as np
from had05 import Trajectory, starts, preflight, sectors, certify, power3

class Archive:
    def __init__(self,m,row_cap,archive_mib):
        if m>8191: raise ValueError('int16 pair residual encoding requires m<=8191')
        self.m=m; self.h=m//2; self.row_cap=row_cap; self.pair_cap=row_cap*(row_cap+1)//2
        # Exact encoded payload model, not a Python heap or RSS cap.
        self.max_payload=row_cap*(m+2*self.h)+self.pair_cap*(2*self.h+16)
        if self.max_payload>archive_mib*(1<<20): raise ValueError('archive encoded payload exceeds cap')
        ss=sectors(m); self.allowed_s=set(int(s) for s in ss.ravel()); self.allowed_pair_dc=set(int(a*a+b*b) for a,b in ss[:,:2])
        self.rows=[]; self.row_keys=set(); self.pairs={}; self.offers=0; self.duplicates=0; self.row_rejected=0; self.pair_checked=0; self.pair_rejected=0; self.cert=None; self.sources=None; self.cap_skips=0
    def payload(self): return len(self.rows)*(self.m+2*self.h)+len(self.pairs)*(2*self.h+16)
    def heap_estimate(self):
        # Approximate Python heap: object headers + arrays/bytes/tuples (shared references not charged twice).
        return sys.getsizeof(self.rows)+sys.getsizeof(self.row_keys)+sys.getsizeof(self.pairs)+sum(sys.getsizeof(z)+sys.getsizeof(z[0])+sys.getsizeof(z[1])+sys.getsizeof(z[4]) for z in self.rows)+sum(sys.getsizeof(k)+sys.getsizeof(v) for k,v in self.pairs.items())+sum(sys.getsizeof(k) for k in self.row_keys)
    def add(self,row,source,budget):
        if self.cert is not None: return 0
        if len(self.rows)>=self.row_cap: self.cap_skips+=1; return 0
        # Preflight enough work for direct row AC (m units) plus every potential new pair.
        if budget<self.m+len(self.rows)+1: return 0
        m=self.m; self.offers+=1; work=m; row=np.asarray(row,dtype=np.int8).copy(); assert row.shape==(m,) and np.all(abs(row)==1)
        s=abs(int(row.sum(dtype=np.int64))); q=0
        if m%3==0:
            a,b,c=(int(row[k::3].sum(dtype=np.int64)) for k in range(3)); q=a*a+b*b+c*c-a*b-b*c-c*a
        if s not in self.allowed_s or q>4*m: self.row_rejected+=1; return work
        r=np.array([np.dot(row.astype(np.int64),np.roll(row,-t).astype(np.int64)) for t in range(1,self.h+1)],dtype='<i2')
        key=r.tobytes()
        if key in self.row_keys: self.duplicates+=1; return work
        i=len(self.rows); self.rows.append((row,r,s*s,q,tuple(source))); self.row_keys.add(key)
        for j,(rr,v,dc,qp,src) in enumerate(self.rows):
            work+=1; self.pair_checked+=1
            if s*s+dc not in self.allowed_pair_dc or q+qp>4*m: self.pair_rejected+=1; continue
            k=(r+v).astype('<i2',copy=False); other=self.pairs.get((-k).astype('<i2',copy=False).tobytes()); kk=k.tobytes()
            if other is not None or np.all(k==0):
                ids=(i,j)+((i,j) if other is None else other); self.cert=[self.rows[t][0].tolist() for t in ids]; self.sources=[self.rows[t][4] for t in ids]; assert certify(self.cert); return work
            if kk not in self.pairs: self.pairs[kk]=(i,j)
        assert len(self.pairs)<=self.pair_cap and self.payload()<=self.max_payload
        return work
    def observe(self,trajectory,slot,budget,rng=None):
        work=0
        for k,row in enumerate(trajectory.state.matrix()):
            if rng is not None:
                # Preserve the observed signed sum distribution, randomize the arrangement.
                row=rng.permutation(row)
            work+=self.add(row,(slot,trajectory.blocks,k),budget-work)
            if self.cert is not None: break
        return work
    def receipt(self):
        return dict(archive_rows=len(self.rows),archive_pairs=len(self.pairs),archive_offers=self.offers,archive_duplicates=self.duplicates,archive_row_rejected=self.row_rejected,archive_pair_checked=self.pair_checked,archive_pair_rejected=self.pair_rejected,archive_cap_skips=self.cap_skips,archive_payload_bytes=self.payload(),archive_max_payload=self.max_payload,archive_heap_estimate=self.heap_estimate(),archive_sources=self.sources)

def evaluate(seed,m,p,chunk,rounds,arm,row_cap,archive_mib,cache_mib):
    preflight(m,p,cache_mib); xs,ss,h=starts(seed,m,p); ts=[Trajectory(x,z,chunk) for x,z in zip(xs,ss)]; budget=rounds*chunk*4*m; n=4*m; work=aw=0; trace=[]; begin=time.perf_counter(); rng=np.random.default_rng(seed+9876543); arch=None if arm=='best' else Archive(m,row_cap,archive_mib); cert=None; kind=None
    solved=next((t for t in ts if t.state.energy==0),None)
    if solved is not None: cert=solved.state.matrix().tolist(); kind='native'
    if arch is not None and cert is None:
        for i,t in enumerate(ts):
            w=arch.observe(t,i,budget-work,rng if arm=='random64' else None); aw+=w; work+=w
            if arch.cert is not None: cert=arch.cert; kind='archive'; break
    for j in range(rounds):
        if cert is not None: break
        solved=next((t for t in ts if t.state.energy==0),None)
        if solved is not None: cert=solved.state.matrix().tolist(); kind='native'; break
        if work+n>budget: break
        i=min(range(p),key=lambda i:(ts[i].best,ts[i].blocks,i)); t=ts[i]; old=t.chunk; t.chunk=min(chunk,(budget-work)//n); w=t.advance(); t.chunk=old; work+=w
        if t.state.energy==0: cert=t.state.matrix().tolist(); kind='native'
        elif arch is not None:
            w=arch.observe(t,i,budget-work,rng if arm=='random64' else None); work+=w; aw+=w
            if arch.cert is not None: cert=arch.cert; kind='archive'
        trace.append([i,t.best,t.state.energy,work,aw])
    if cert is not None: assert certify(cert)
    assert work<=budget
    out=dict(arm=arm,solved=cert is not None,solution_kind=kind,certificate=cert,best=0 if cert else min(t.best for t in ts),candidate_units=work,archive_candidate_units=aw,search_candidate_scores=work-aw,seconds=time.perf_counter()-begin,start_hash=h,trace=trace)
    if arch is not None: out.update(arch.receipt())
    else: out.update(archive_rows=0,archive_pairs=0,archive_payload_bytes=0,archive_heap_estimate=0,archive_sources=None)
    return out

def run(a):
    begin=time.perf_counter(); rows=[]
    for root in range(a.root,a.root+a.roots):
        for m in a.lengths:
            for case in range(a.cases):
                seed=root*100000+m*100+case
                for arm,cap in [('best',0),('archive16',16),('archive64',64),('archive128',128),('random64',64)]:
                    r=evaluate(seed,m,a.population,a.chunk,a.rounds,arm,cap,a.archive_mib,a.cache_mib); r.update(root=root,m=m,case=case,seed=seed); rows.append(r)
        print('root',root,'done',flush=True)
    out=dict(scope='DEVELOPMENT_ONLY',args=vars(a),rows=rows,seconds=time.perf_counter()-begin); Path(a.out).write_text(json.dumps(out,indent=2)); print('seconds',out['seconds'])
    for arm in ('best','archive16','archive64','archive128','random64'):
        z=[r for r in rows if r['arm']==arm]; print(arm,[(m,sum(r['solved'] for r in z if r['m']==m)) for m in a.lengths],sum(r['solved'] for r in z),'archive',sum(r['solution_kind']=='archive' for r in z),'mean payload',round(np.mean([r['archive_payload_bytes'] for r in z])),'seconds',round(sum(r['seconds'] for r in z),3),'mean best',round(np.mean([r['best'] for r in z]),3))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=int,default=85201); p.add_argument('--roots',type=int,default=4); p.add_argument('--cases',type=int,default=8); p.add_argument('--population',type=int,default=8); p.add_argument('--chunk',type=int,default=16); p.add_argument('--rounds',type=int,default=24); p.add_argument('--lengths',type=int,nargs='+',default=[23,27,29]); p.add_argument('--archive-mib',type=int,default=8); p.add_argument('--cache-mib',type=int,default=8); p.add_argument('--out',default='archive_results.json'); run(p.parse_args())
