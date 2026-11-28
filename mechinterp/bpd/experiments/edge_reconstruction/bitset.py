"""Exact compressed posterior masks: compare full-cache trajectories."""
import json,time
import numpy as np
from learned import SUP,P,B,world,predict,trajectory


def packed(root):
    f,h,_=world(root); t=time.perf_counter(); obs=predict(f,np.arange(len(SUP)))
    masks=[[int.from_bytes(np.packbits(obs[:,a]==v,bitorder='little').tobytes(),'little') for v in (-3,-1,1,3)] for a in range(P)]
    truth=obs[h].copy(); del obs
    alive=(1<<len(SUP))-1; used=np.zeros(P,bool); sizes=[]; actions=[]
    for _ in range(B):
        n=alive.bit_count()
        counts=np.array([[(alive&m).bit_count() for m in row] for row in masks]).T
        p=counts/n; scores=-(p*np.log2(np.maximum(p,1e-300))).sum(axis=0)
        a=int(np.argmax(np.where(used,-np.inf,scores))); used[a]=True
        alive &= masks[a][(int(truth[a])+3)//2]; assert alive & (1<<h)
        sizes.append(alive.bit_count()); actions.append(a)
    return actions,sizes,1000*(time.perf_counter()-t)


if __name__=='__main__':
    rows=[]
    for root in range(80,144):
        if root%2: b=packed(root); a=trajectory(root,'exact')
        else: a=trajectory(root,'exact'); b=packed(root)
        assert a['actions']==b[0] and a['sizes']==b[1],root
        rows.append([1000*sum(a[k] for k in ('cache_build','score_time','filter_time')),b[2]])
    print(json.dumps(dict(roots=len(rows),raw_mask_bytes=P*4*((len(SUP)+7)//8),raw_matrix_bytes=P*len(SUP),median_ms=np.median(rows,axis=0).tolist(),median_paired_speedup=float(np.median(np.array(rows)[:,0]/np.array(rows)[:,1])),times=rows),indent=2))
