"""ER-05: exact moment and packed-plane replay, pair-only max entropy."""
import json,time
from itertools import combinations,product
import numpy as np
P,B,SEED=64,12,261127

def entropy(q):
    q=np.maximum(q,0)
    return -(q*np.log2(np.maximum(q,1e-300))).sum(axis=0)
def dist(x,y,z): return np.stack([1-x+y-z,x-2*y+3*z,y-3*z,z])
def maxent(n,a,b):
    x,y=a/n,b/n
    lo=np.maximum(0,(2*y-x)/3); hi=np.maximum(lo,np.minimum(y/3,1-x+y))
    for _ in range(28):
        mid=(lo+hi)/2; q=np.maximum(dist(x,y,mid),1e-300)
        grad=np.log(q[0])+3*np.log(q[2])-3*np.log(q[1])-np.log(q[3])
        lo=np.where(grad>0,mid,lo); hi=np.where(grad>0,hi,mid)
    return entropy(dist(x,y,(lo+hi)/2))
def pack(x): return int.from_bytes(np.packbits(x,bitorder='little').tobytes(),'little')
def world(n,root,sup):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,root])); edges=[(i,j) for i in range(n) for j in range(n) if i!=j]
    x=rng.choice([-1,1],size=(P,n)); c=rng.choice([-1,1],size=(P,n))
    f=np.stack([c[:,i]*x[:,j] for i,j in edges],axis=1)>0
    return f,int(rng.integers(len(sup))),rng.permutation(P)
def replay(f,sup,h,order,mode):
    N=len(sup); E=f.shape[1]; t=time.perf_counter(); audit=0
    obs=f[:,sup].sum(axis=2).T.astype(np.uint8); truth=obs[h].copy(); used=np.zeros(P,bool); ids=np.arange(N); alive=(1<<N)-1
    if mode=='planes':
        masks=[(pack(obs[:,a]&1),pack((obs[:,a]&2)!=0)) for a in range(P)]; del obs
    if mode in ('moments','pair'):
        pp=np.array(list(combinations(range(E),2))); lookup=np.zeros((E,E),int); lookup[pp[:,0],pp[:,1]]=np.arange(len(pp)); sp=lookup[sup[:,[0,0,1]],sup[:,[1,2,2]]]
        pf=(f[:,pp[:,0]]&f[:,pp[:,1]]).astype(np.int64); ff=f.astype(np.int64)
        if mode=='moments': triple=[pack(obs[:,a]==3) for a in range(P)]; del obs
    lut=np.zeros(N+1); lut[1:]=np.arange(1,N+1)*np.log2(np.arange(1,N+1)); actions=[]; sizes=[]; checks=0
    for step in range(B):
        m=alive.bit_count() if mode=='planes' else len(ids)
        if mode=='planes':
            both=np.array([(alive&l&r).bit_count() for l,r in masks]); low=np.array([(alive&l).bit_count() for l,r in masks]); high=np.array([(alive&r).bit_count() for l,r in masks]); counts=np.stack([m-low-high+both,low-both,high-both,both])
        elif mode in ('moments','pair'):
            a=ff@np.bincount(sup[ids].ravel(),minlength=E); b=pf@np.bincount(sp[ids].ravel(),minlength=len(pp))
            if mode=='moments':
                cc=np.array([(alive&mask).bit_count() for mask in triple]); counts=np.stack([m-a+b-cc,a-2*b+3*cc,b-3*cc,cc]); ta=time.perf_counter(); direct=f[:,sup[ids]].sum(axis=2).T; expected=np.stack([(direct==k).sum(axis=0) for k in range(4)]); assert np.array_equal(counts,expected); audit+=time.perf_counter()-ta; checks+=P
            else: scores=maxent(m,a,b)
        elif mode=='dense': counts=np.stack([(obs[ids]==k).sum(axis=0) for k in range(4)])
        if mode=='random': action=int(order[step])
        else:
            if mode!='pair':
                assert np.all(counts>=0) and np.all(counts.sum(axis=0)==m); scores=np.log2(m)-lut[counts].sum(axis=0)/m
            z=np.where(used,-np.inf,scores); action=int(np.flatnonzero(z>=np.max(z)-1e-12)[0])
        used[action]=True; actions.append(action); value=truth[action]
        if mode=='planes':
            l,r=masks[action]; alive&=(l if value&1 else ~l)&(r if value&2 else ~r); assert alive&(1<<h); sizes.append(alive.bit_count())
        else:
            vals=f[action,sup[ids]].sum(axis=1) if mode=='moments' else obs[ids,action]; ids=ids[vals==value]; assert h in ids; sizes.append(len(ids))
            if mode=='moments': row=np.zeros(N,bool); row[ids]=True; alive=pack(row)
    return dict(actions=actions,sizes=sizes,ms=(time.perf_counter()-t-audit)*1000,checks=checks)
def fixture():
    bits=np.array(list(product((0,1),repeat=3))); masks=np.zeros((8,6),int); masks[np.arange(8)[:,None],2*np.arange(3)+bits]=1; p=masks[bits.sum(axis=1)%2==0]; q=masks
    assert np.array_equal(p.mean(axis=0),q.mean(axis=0)); assert np.array_equal(p.T@p/len(p),q.T@q/len(q))
    outcomes=[np.bincount(bank[:,::2].sum(axis=1),minlength=4)/len(bank) for bank in (p,q)]
    return dict(P=outcomes[0].tolist(),Q=outcomes[1].tolist(),entropies=[float(entropy(d)) for d in outcomes])
def equal_size_fixture():
    # Two disjoint parity cubes, each using three distinct +/- edge pairs.
    bits=np.array(list(product((0,1),repeat=3))); even=bits.sum(axis=1)%2==0
    rows=[]
    for offset in (0,6):
        m=np.zeros((8,12),int); m[np.arange(8)[:,None],offset+2*np.arange(3)+bits]=1; rows.append(m)
    p=np.concatenate([rows[0][even],rows[1][even]])
    q=np.concatenate([rows[0][even],rows[1][~even]])
    assert len(p)==len(q)==8 and np.array_equal(p.sum(axis=0),q.sum(axis=0))
    assert np.array_equal(p.T@p,q.T@q)
    outcomes=[np.bincount(bank[:,::2].sum(axis=1),minlength=4)/len(bank) for bank in (p,q)]
    # Embed in the actual n=6 directed bilinear family: c all +, x three +, three -.
    edges=[(i,j) for i in range(6) for j in range(6) if i!=j]
    pos=[(i,j) for i,j in edges if j<3][:6]; neg=[(i,j) for i,j in edges if j>=3][:6]
    edge_order=[e for pair in zip(pos,neg) for e in pair]; assert len(set(edge_order))==12
    return dict(P=outcomes[0].tolist(),Q=outcomes[1].tolist(),entropies=[float(entropy(d)) for d in outcomes],bank_size=len(p),edge_order=edge_order)

def main():
    results={}
    for n,roots in ((6,range(144,208)),(7,range(208,232))):
        sup=np.array(list(combinations(range(n*(n-1)),3))); modes=['dense','planes','moments','pair','random']; rows={m:[] for m in modes}
        for root in roots:
            f,h,order=world(n,root,sup)
            for mode in modes[root%len(modes):]+modes[:root%len(modes)]: rows[mode].append(replay(f,sup,h,order,mode))
            for mode in ('planes','moments'):
                assert rows[mode][-1]['actions']==rows['dense'][-1]['actions']; assert rows[mode][-1]['sizes']==rows['dense'][-1]['sizes']
        results[n]=dict(roots=[roots.start,roots.stop-1],hypotheses=len(sup),raw_dense_bytes=P*len(sup),raw_plane_bytes=P*2*((len(sup)+7)//8),raw_triple_bytes=P*((len(sup)+7)//8),arms={m:dict(unique=(np.array([x['sizes'] for x in vals])==1).sum(axis=0).tolist(),median_ms=float(np.median([x['ms'] for x in vals])),checks=sum(x['checks'] for x in vals),root_rows=vals) for m,vals in rows.items()},paired_speedup=float(np.median([a['ms']/b['ms'] for a,b in zip(rows['dense'],rows['planes'])])))
    print(json.dumps(dict(fixture=fixture(),equal_size_fixture=equal_size_fixture(),results=results),indent=2))
if __name__=='__main__': main()
