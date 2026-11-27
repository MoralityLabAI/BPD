"""ER-39: batch access traces and offline optimal optional-admission caching."""
import argparse,json,itertools,heapq
from collections import OrderedDict,Counter
from pathlib import Path
import numpy as np
from er23 import EPS,cost,Bank,decision
from er35 import StreamBank,BLOCK
from er32 import load


def trace(path):
    args,y,hidden,noise=load(path); b=StreamBank(*args,'stream'); d,p=b.d,b.p
    floors=np.empty((d*d,p)); gap=np.zeros(p); upper=np.inf; best=-1
    for i in range(0,d,BLOCK):
        end=min(i+BLOCK,d); pred,err=b.bound(i,end); delta=abs(pred-y)
        l=np.maximum(0,delta-err)**2; h=(delta+err)**2; floors[i*d:end*d]=l; gap+=(h-l).sum(0)
        hi=h.sum(1); j=int(hi.argmin())
        if hi[j]<upper: upper=float(hi[j]); best=i*d+j
    lower=floors.sum(1); score=float(cost(b.exact_incumbent(best),y)); incumbent=best
    order=np.argsort(lower,kind='stable'); order=order[order!=best]; cols=np.argsort(-gap,kind='stable')
    batches=[]; work=p; dedup=0
    for start in range(0,len(order),64):
        if lower[order[start]]>score+EPS: break
        ids=order[start:start+64]; ids=ids[lower[ids]<=score+EPS]; terms=floors[ids].copy()
        for a,k in enumerate(cols):
            if not len(ids): break
            i,j=ids//d,ids%d; _,ix=np.unique(i,return_index=True); keys=(i[np.sort(ix)]*p+k).tolist()
            batches.append(keys); work+=len(ids); dedup+=len(ids)-len(keys)
            h=np.tanh(b.first_z3[i,k]-b.proj[i,k,j,None]*b.u[j]); f=(np.tanh(h@b.w.T+b.bias[k])*b.r[k]).sum(-1)
            v=(f-y[k])**2; assert np.all(terms[:,k]<=v+EPS); terms[:,k]=v; energy=terms.sum(1)
            if a+1>=p:
                val=float(energy.min()); key=int(ids[energy==val].min())
                if val<score or (val==score and key<best): best,score=key,val
                break
            keep=energy<=score+EPS; ids=ids[keep]; terms=terms[keep]
    ref,rs,n,_=decision(Bank(*args),y,'eager',batch=64)
    assert best==ref and abs(score-rs)<EPS
    assert sum(map(len,batches))+dedup==work-p
    return dict(path=path.name,width=d,p=p,hidden=hidden,noise=noise,best=best,score=score,incumbent=incumbent,exact_pairs=n,work=work,dedup=dedup,probe_order=cols.tolist(),batches=batches)


def online(batches,cap,p,mode,quotas=None):
    segmented=mode.startswith('seg'); touch=mode.endswith('lru')
    banks=[OrderedDict() for _ in range(p if segmented else 1)]
    limits=quotas if segmented else [cap]; misses=0; profile=[]
    for batch in batches:
        ix=batch[0]%p if segmented else 0; bank=banks[ix]; limit=limits[ix]
        absent=[]
        for k in batch:
            if k in bank:
                if touch: bank.move_to_end(k)
            else: absent.append(k)
        misses+=len(absent); profile.append(len(absent))
        if limit:
            for k in absent:
                if len(bank)==limit: bank.popitem(last=False)
                bank[k]=None
    return misses,profile


def oracle(batches,cap):
    """Optimal unit-miss count with free bypass and batch-start hit semantics.

    Each consecutive request pair defines one retention interval [t,next_t).
    Retain intervals greedily; discard the active interval ending latest on overflow.
    """
    nexts={}; ends=[]
    for t in range(len(batches)-1,-1,-1):
        row=[]
        for k in batches[t]:
            row.append(nexts.get(k,len(batches))); nexts[k]=t
        ends.append(row)
    ends.reverse(); active={}; heap=[]; total=0; profile=[]
    for t,(batch,after) in enumerate(zip(batches,ends)):
        m=sum(k not in active for k in batch); total+=m; profile.append(m)
        for k,end in zip(batch,after):
            active.pop(k,None)
            if end<len(batches):
                active[k]=end; heapq.heappush(heap,(-end,k))
        while len(active)>cap:
            neg,k=heapq.heappop(heap)
            if active.get(k)==-neg: del active[k]
    return total,profile


def exhaustive():
    # Independent dynamic program over every reachable post-batch cache subset.
    universe=range(3); choices=[tuple(k for k in universe if mask>>k&1) for mask in range(8)]
    checks=0
    for batches in itertools.product(choices,repeat=4):
        for cap in range(4):
            states={frozenset():0}
            for batch in batches:
                nxt={}; b=set(batch)
                for c,val in states.items():
                    cost0=val+len(b-c); pool=sorted(c|b)
                    for size in range(min(cap,len(pool))+1):
                        for subset in itertools.combinations(pool,size):
                            s=frozenset(subset); nxt[s]=min(nxt.get(s,10**9),cost0)
                states=nxt
            assert oracle(batches,cap)[0]==min(states.values()),(batches,cap)
            checks+=1
    return checks


def quota(cap,weights):
    # Largest remainder integer apportionment; reserve one entry per probe when possible.
    p=len(weights); base=int(cap>=p); left=cap-base*p
    raw=left*np.asarray(weights)/sum(weights); out=np.floor(raw).astype(int)+base
    for k in np.argsort(-(raw-np.floor(raw)),kind='stable')[:cap-out.sum()]: out[k]+=1
    assert out.sum()==cap
    return out.tolist()


def weights_from(rows):
    out={}
    for d in sorted({r['width'] for r in rows}):
        samples=[]
        for r in rows:
            if r['width']!=d: continue
            count=Counter(k%r['p'] for b in r['batches'] for k in b)
            total=sum(count.values()); samples.append([count[k]/max(1,total) for k in r['probe_order']])
        out[d]=(np.mean(samples,axis=0)+1e-9).tolist()
    return out


def replay(row,weights,cap):
    b=row['batches']; p=row['p']; n=sum(map(len,b)); q=quota(cap,[1]*p)
    rankq=quota(cap,weights[row['width']]); demand=[0]*p
    for k,c in zip(row['probe_order'],rankq): demand[k]=c
    policies={}
    for name in ('fifo','lru','segfifo','seglru','weighted_fifo','weighted_lru','oracle'):
        if name=='oracle': misses,profile=oracle(b,cap)
        else:
            mode={'weighted_fifo':'segfifo','weighted_lru':'seglru'}.get(name,name)
            misses,profile=online(b,cap,p,mode,demand if name.startswith('weighted') else q)
        policies[name]=dict(misses=misses,hits=n-misses,nonempty_miss_batches=sum(x>0 for x in profile),mean_positive_miss_batch=misses/max(1,sum(x>0 for x in profile)),miss_batch_hist=dict(Counter(profile)))
    assert all(v['misses']>=policies['oracle']['misses'] for v in policies.values())
    return dict(**{k:v for k,v in row.items() if k!='batches'},requests=n,distinct=len(set(k for batch in b for k in batch)),batch_count=len(b),capacity=cap,demand_quotas=demand,policies=policies)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('trace','replay','audit')); p.add_argument('paths',type=Path,nargs='*'); p.add_argument('--cal',type=Path); p.add_argument('--caps',type=int,nargs='+',default=[128,512,1024]); a=p.parse_args()
    if a.action=='audit': print(json.dumps(dict(exhaustive_checks=exhaustive())))
    elif a.action=='trace':
        for path in a.paths: print(json.dumps(trace(path),separators=(',',':')),flush=True)
    else:
        cal=[json.loads(l) for l in a.cal.read_text().splitlines()]; w=weights_from(cal)
        for path in a.paths:
            for line in path.read_text().splitlines():
                r=json.loads(line)
                for cap in a.caps: print(json.dumps(replay(r,w,cap),separators=(',',':')),flush=True)
