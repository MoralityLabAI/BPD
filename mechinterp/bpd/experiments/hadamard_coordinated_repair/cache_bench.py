"""Independent incremental/rebuild and blocked-pair microbenchmarks."""
import argparse, json, time
from pathlib import Path
import numpy as np
from search import State, pairs_for

def run():
    rng=np.random.default_rng(84401); rows=[]; blocks=[]
    for m in (19,29,167,599):
        x=rng.choice([-1,1],size=(4,m)); moves=rng.integers(4*m,size=128)
        s=State(x); t=time.perf_counter()
        for i in moves: s.flip(i)
        delta=time.perf_counter()-t
        x=x.reshape(-1).copy(); t=time.perf_counter()
        for i in moves: x[i]*=-1; r=State(x)
        rebuild=time.perf_counter()-t
        assert np.array_equal(s.r,r.r) and np.array_equal(s.d,r.d) and np.array_equal(s.s,r.s)
        rows.append(dict(m=m,flips=len(moves),state_bytes=s.nbytes,incremental_seconds=delta,rebuild_seconds=rebuild,speedup=rebuild/delta))
        print(rows[-1],flush=True)
    s=State(rng.choice([-1,1],size=(4,599))); v=s.singles(); i,j=pairs_for(s.n,np.argsort(v),8); persistent=s.nbytes+i.nbytes+j.nbytes+v.nbytes; per=8*(4*len(s.r)+20); ref=None
    for mib in (8,16,64):
        width=((mib<<20)-persistent)//per; assert width>0
        t=time.perf_counter(); score=0; best=10**15
        for lo in range(0,len(i),width):
            q=s.pairs(i[lo:lo+width],j[lo:lo+width],v); score+=int(q.sum()); best=min(best,int(q.min()))
        dt=time.perf_counter()-t; check=(score,best)
        if ref is None: ref=check
        assert check==ref
        blocks.append(dict(m=599,shortlist=8,pairs=len(i),cap_mib=mib,block_width=width,tracked_workspace_bytes=persistent+min(width,len(i))*per,seconds=dt,pairs_per_second=len(i)/dt,checksum=score,min_energy=best))
        print(blocks[-1],flush=True)
    return dict(incremental=rows,blocks=blocks)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='cache_results.json'); a=p.parse_args(); Path(a.out).write_text(json.dumps(run(),indent=2))
