"""ER-40: causal oracle-supervised cache admission on batched access traces."""
import argparse, json, heapq, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from er39 import online, oracle

ALPHAS=(0., .25, .5, 1., 2., 4.)
CAP=512

def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]

def features(row,cap=CAP):
    """All features are computed at batch start from current/past requests only."""
    d,p=row['width'],row['p']; n=d*p; rank=np.argsort(row['probe_order'])
    last=np.full(n,-1); counts=np.zeros(n); ew=np.zeros(n); gap=np.zeros(n)
    comp=np.zeros(d); probe=np.zeros(p); total=0; X=[]; ids=[]; starts=[0]
    for t,b in enumerate(row['batches']):
        k=np.asarray(b,dtype=int); i=k//p; j=k%p; c=counts[k]; g=t-last[k]
        # All rows in a batch use the same pre-batch history snapshot.
        f=np.column_stack([np.full(len(k),np.log2(d)),i/max(1,d-1),rank[j]/max(1,p-1),
            np.full(len(k),np.log1p(t)),np.full(len(k),np.log1p(len(k))),
            np.log1p(c),np.log1p(g),np.log1p(gap[k]),np.log1p(ew[k]),
            np.log1p(comp[i]),np.log1p(probe[j]),np.log1p(c/(1+t)),
            np.log1p(probe[j]/max(1,total)),np.arange(len(k))/max(1,len(k)-1),
            (c==0).astype(float),np.full(len(k),np.log2(cap/d))])
        X.append(f); ids.extend(k); starts.append(len(ids))
        ew[k]=np.where(c==0,0,np.where(c==1,g,.8*ew[k]+.2*g)); gap[k]=np.where(c>0,g,0)
        last[k]=t; counts[k]+=1; np.add.at(comp,i,1); np.add.at(probe,j,1); total+=len(k)
    return np.concatenate(X),np.array(ids),np.array(starts)

def labels(row,cap,ids,starts):
    """Label intervals retained by the ER-39 optional-admission optimum."""
    T=len(starts)-1; end=np.full(len(ids),T); seen={}
    for t in range(T-1,-1,-1):
        for a in range(starts[t],starts[t+1]):
            k=int(ids[a]); end[a]=seen.get(k,T); seen[k]=t
    active={}; heap=[]; y=np.zeros(len(ids),dtype=np.uint8)
    for t in range(T):
        for a in range(starts[t],starts[t+1]):
            k=int(ids[a]); prev=active.pop(k,None)
            if prev is not None: y[prev]=1
            if end[a]<T: active[k]=a; heapq.heappush(heap,(-int(end[a]),k,a))
        while len(active)>cap:
            _,k,a=heapq.heappop(heap)
            if active.get(k)==a: del active[k]
    assert len(ids)-int(y.sum())==oracle(row['batches'],cap)[0]
    return y

def prepare(rows,cap):
    out=[]
    for row in rows:
        X,k,s=features(row,cap); y=labels(row,cap,k,s)
        out.append((row,X,k,s,y))
    return out

def replay(item,priority,alpha,cap):
    row,X,ids,starts,_=item; n=row['width']*row['p']; cache=np.zeros(n,bool)
    base=np.full(n,-np.inf); last=np.full(n,-1); misses=0; bypass=0
    profile=[]
    for t in range(len(starts)-1):
        a,b=starts[t:t+2]; k=ids[a:b]; absent=~cache[k]; m=int(absent.sum()); misses+=m; profile.append(m)
        base[k]=priority[a:b]; last[k]=t; cache[k]=True; pool=np.flatnonzero(cache)
        if len(pool)>cap:
            val=base[pool]-alpha*np.log1p(t-last[pool]); order=np.lexsort((pool,-last[pool],-val)); drop=pool[order[cap:]]; cache[drop]=False
        bypass+=int((absent&~cache[k]).sum())
    return dict(misses=misses,bypass=bypass,nonempty_miss_batches=sum(x>0 for x in profile))

def scores(item,scaler,models):
    row,X,*_=item; z=scaler.transform(X)
    return dict(learned=models[0].decision_function(z),shuffled=models[1].decision_function(z),
                freq=X[:,5],gap=-X[:,7]+np.where(X[:,14]>0,-2.,0.),recency=np.zeros(len(X)))

def run(train,dev,test,cap=CAP):
    A,B,C=[prepare(rows,cap) for rows in (train,dev,test)]
    def rootset(rows): return {(r['width'],r['path'].split('_')[1]) for r in rows}
    assert not (rootset(train)&rootset(dev) or rootset(train)&rootset(test) or rootset(dev)&rootset(test))
    X=np.concatenate([i[1] for i in A]); y=np.concatenate([i[4] for i in A]); scaler=StandardScaler().fit(X); Z=scaler.transform(X)
    rng=np.random.default_rng(261144); models=[LogisticRegression(max_iter=200,C=1).fit(Z,z) for z in (y,rng.permutation(y))]
    ds=[scores(i,scaler,models) for i in B]; alphas={}; calibration={}
    for name in ds[0]:
        totals={str(a):sum(replay(i,s[name],a,cap)['misses'] for i,s in zip(B,ds)) for a in ALPHAS}
        alphas[name]=min(ALPHAS,key=lambda a:totals[str(a)]); calibration[name]=totals
    rows=[]; prediction={}
    for split,items in [('development',B),('test',C)]:
        ys=np.concatenate([i[4] for i in items]); sc=[scores(i,scaler,models) for i in items]
        pred=np.concatenate([s['learned'] for s in sc]); sh=np.concatenate([s['shuffled'] for s in sc])
        prediction[split]=dict(examples=len(ys),positive_rate=float(ys.mean()),learned_auc=float(roc_auc_score(ys,pred)),shuffled_auc=float(roc_auc_score(ys,sh)),learned_logloss=float(log_loss(ys,1/(1+np.exp(-pred)))))
        for i,s in zip(items,sc):
            row=i[0]; out={}
            for m in ('fifo','lru'):
                n,p=online(row['batches'],cap,row['p'],m); out[m]=dict(misses=n,nonempty_miss_batches=sum(x>0 for x in p))
            n,p=oracle(row['batches'],cap); out['oracle']=dict(misses=n,nonempty_miss_batches=sum(x>0 for x in p))
            for name,alpha in alphas.items(): out[name]=replay(i,s[name],alpha,cap)
            assert all(v['misses']>=out['oracle']['misses'] for v in out.values())
            assert out['recency']['misses']==out['lru']['misses'] or alphas['recency']==0
            rows.append(dict(split=split,path=row['path'],width=row['width'],noise=row['noise'],requests=len(i[2]),policies=out))
    return dict(capacity=cap,train_examples=len(y),train_positive=float(y.mean()),feature_count=X.shape[1],parameters=X.shape[1]+1,alphas=alphas,calibration=calibration,prediction=prediction,model=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=models[0].coef_[0].tolist(),intercept=float(models[0].intercept_[0])),rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--train',required=True); p.add_argument('--dev',required=True); p.add_argument('--test',required=True); p.add_argument('--cap',type=int,default=CAP); a=p.parse_args()
    print(json.dumps(run(read(a.train),read(a.dev),read(a.test),a.cap),indent=2))
