"""SCCG-04 constructed development pilot; no native model claims."""
from fractions import Fraction as F
from itertools import combinations
import argparse
import json
import random
import time

import numpy as np
import torch
from torch import nn

N=8
SUPPORTS=list(combinations(range(N),3))
TRAIN=96
DEV=48
SEEDS=(41,43,47)
STEPS=160
BUDGET=4


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def sub(a,b):
    return [x-y for x,y in zip(a,b)]


def root(seed):
    rng=random.Random(seed)
    mode=seed%4
    r=1+mode%2
    C=[[rng.randint(-2,2) for _ in range(N)] for _ in range(r)]
    c=[3*rng.choice((-2,-1,0,1,2)) for _ in range(r)]
    a=[rng.randint(-3,3) for _ in range(N)]
    if mode>=2:
        i,j,k=rng.sample(range(N),3)
        for row in C:
            row[k]=-row[i]-row[j]
        for idx in (i,j,k):
            a[idx]=rng.choice((2,3))
    a0=3*rng.randint(-2,2)
    g=rng.choice((2,3,4))
    return dict(seed=seed,mode=mode,C=C,c=c,a=a,a0=a0,g=g)


def exact_geometry(w,S):
    # x=3 e_side, m=7, rho0=4, edit radius<=3.
    # Augmented y=(z,r), z=(rho0/rho)delta, r=4 rho0/rho.
    # ||y||²=16, r>=16/5, D y=4 c.
    rows=[[F(4*row[i]) for i in S]+[F(c)] for row,c in zip(w["C"],w["c"])]
    # Rational orthogonal row basis; at most 2 rows.
    basis=[]
    for row in rows:
        for q in basis:
            factor=dot(row,q)/dot(q,q)
            row=sub(row,[factor*x for x in q])
        if any(row):
            basis.append(row)
    def null(v):
        v=list(map(F,v))
        for q in basis:
            factor=dot(v,q)/dot(q,q)
            v=sub(v,[factor*x for x in q])
        return v
    clean=[F(0)]*3+[F(4)]
    center=sub(clean,null(clean))
    p=null([0,0,0,1])
    l=[F(4*w["a"][i]) for i in S]+[F(w["a0"])]
    b=null(l)
    return (16-dot(center,center),dot(p,p),dot(b,b),dot(p,b),
            F(16,5)-center[-1],4*w["a0"]-dot(l,center))


def threshold(geo,g):
    R,P,B,D,H,K0=geo
    K=K0+4*g
    assert R>=0
    if H<=0 and K<=0:
        return True
    if P>0 and D*H>=P*K and H*H<=P*R:
        return True
    if B>0 and D*K>=B*H and K*K<=B*R:
        return True
    det=P*B-D*D
    return det>0 and B*H*H-2*D*H*K+P*K*K<=det*R


def float_scores(w):
    # Independent vectorized pseudoinverse implementation of finite cap.
    C=np.array(w["C"],dtype=float)
    c=np.array(w["c"],dtype=float)
    a=np.array(w["a"],dtype=float)
    S=np.array(SUPPORTS)
    A=C[:,S].transpose(1,0,2)
    D=np.concatenate((4*A,np.broadcast_to(c[None,:,None],(len(S),len(c),1))),axis=-1)
    I=np.eye(4)[None]
    Q=I-np.linalg.pinv(D,rcond=1e-12)@D
    clean=np.array([0.,0.,0.,4.])
    center=clean-Q@clean
    p=Q@np.array([0.,0.,0.,1.])
    l=np.column_stack((4*a[S],np.full(len(S),w["a0"])))
    b=(Q@l[:,:,None])[:,:,0]
    R=np.maximum(0,16-np.sum(center*center,axis=1))
    P=np.sum(p*p,axis=1)
    B=np.sum(b*b,axis=1)
    cross=np.sum(p*b,axis=1)
    H=16/5-center[:,-1]
    gain=np.sqrt(R*B)
    for j in range(len(S)):
        if B[j]>1e-20 and math_sqrt(R[j]/B[j])*cross[j]<H[j]-1e-10:
            if P[j]<=1e-20:
                raise ArithmeticError("cap missing despite clean feasible point")
            gain[j]=cross[j]*H[j]/P[j]+math_sqrt(
                max(0.,(B[j]-cross[j]**2/P[j])*(R[j]-H[j]**2/P[j])))
    finite=(np.sum(l*center,axis=1)+gain)/4-w["a0"]
    tangent=a[S]-(np.linalg.pinv(A,rcond=1e-12)@A@a[S,:,None])[:,:,0]
    return np.linalg.norm(a[S],axis=1),3*np.linalg.norm(tangent,axis=1),finite


def math_sqrt(x):
    return float(np.sqrt(x))


def features(w):
    C=np.zeros((2,N),np.float32)
    C[:len(w["C"])]=w["C"]
    c=np.zeros(2,np.float32)
    c[:len(w["c"])]=w["c"]
    per=np.column_stack((np.array(w["a"],np.float32)/3,C.T/4))
    return per[np.array(SUPPORTS)],np.array([*list(c/6),w["a0"]/6,w["g"]/4,len(w["c"])/2],
                                          np.float32)


class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Linear(3,12)
        self.inp=nn.Linear(29,24)
        self.step=nn.Linear(24,24)
        self.out=nn.Linear(24,1)

    def forward(self,comp,context):
        h=torch.tanh(self.enc(comp))
        c=context[:,None,:].expand(-1,h.shape[1],-1)
        z=torch.tanh(self.inp(torch.cat((h.mean(-2),h.amax(-2),c),-1)))
        h=torch.zeros_like(z)
        for _ in range(3):
            h=torch.tanh(z+self.step(h))
        return self.out(h).squeeze(-1)


def make_set(start,count):
    worlds=[root(start+i) for i in range(count)]
    labels=np.array([[threshold(exact_geometry(w,S),w["g"]) for S in SUPPORTS]
                     for w in worlds],dtype=bool)
    features_=list(map(features,worlds))
    comp=torch.from_numpy(np.stack([x[0] for x in features_]))
    ctx=torch.from_numpy(np.stack([x[1] for x in features_]))
    return worlds,labels,comp,ctx


def score_order(scores):
    return np.argsort(-scores,axis=1,kind="stable")


def evaluate(order,labels,worlds):
    hits=[]
    calls=[]
    for rank,y,w in zip(order,labels,worlds):
        success=False
        count=0
        for i in rank[:BUDGET]:
            count+=1
            got=threshold(exact_geometry(w,SUPPORTS[int(i)]),w["g"])
            if got!=bool(y[i]):
                raise ArithmeticError("label replay mismatch")
            if got:
                success=True
                break
        hits.append(success)
        calls.append(count)
    return dict(top1=int(np.sum(labels[np.arange(len(labels)),order[:,0]])),
                hits=int(sum(hits)),queries=int(sum(calls)),
                per_root_hits=list(map(int,hits)),per_root_queries=calls)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output")
    args=ap.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0=time.perf_counter()
    tr,Y,X,G=make_set(11000,TRAIN)
    dv,V,U,H=make_set(21000,DEV)
    label_seconds=time.perf_counter()-t0
    assert not set(w["seed"] for w in tr)&set(w["seed"] for w in dv)
    t0=time.perf_counter()
    scores=np.array([float_scores(w) for w in dv])
    analytic_seconds=time.perf_counter()-t0
    # Do not silently ignore numerical disagreements.
    finite=scores[:,2,:]
    margins=finite-np.array([w["g"] for w in dv])[:,None]
    false_pos=int(np.sum((margins>1e-8)&~V))
    false_neg=int(np.sum((margins<-1e-8)&V))
    ambiguous=int(np.sum(np.abs(margins)<=1e-8))
    if false_pos or false_neg:
        raise ArithmeticError("independent exact/float mismatch")
    order_random=np.array([np.random.default_rng(93000+w["seed"]).permutation(len(SUPPORTS))
                           for w in dv])
    arms={"random":evaluate(order_random,V,dv)}
    for j,name in enumerate(("magnitude","tangent","analytic_finite")):
        arms[name]=evaluate(score_order(scores[:,j,:]),V,dv)
    for seed in SEEDS:
        torch.manual_seed(seed)
        model=Ranker()
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        generator=torch.Generator().manual_seed(seed+100)
        target=torch.tensor(Y,dtype=torch.float32)
        t0=time.perf_counter()
        for _ in range(STEPS):
            ix=torch.randint(TRAIN,(8,),generator=generator)
            loss=nn.functional.binary_cross_entropy_with_logits(model(X[ix],G[ix]),target[ix])
            opt.zero_grad();loss.backward();opt.step()
        train_seconds=time.perf_counter()-t0
        t0=time.perf_counter()
        with torch.no_grad():
            logits=model(U,H).numpy()
        rank_seconds=time.perf_counter()-t0
        row=evaluate(score_order(logits),V,dv)
        row.update(parameters=sum(p.numel() for p in model.parameters()),
                   train_seconds=train_seconds,rank_seconds=rank_seconds)
        arms[f"recurrent_{seed}"]=row
    result=dict(scope="CONSTRUCTED_DEVELOPMENT_ONLY",train_roots=TRAIN,dev_roots=DEV,
                supports=len(SUPPORTS),budget=BUDGET,steps=STEPS,seeds=list(SEEDS),
                train_positive=int(Y.sum()),dev_positive=int(V.sum()),
                train_solvable=int(Y.any(1).sum()),dev_solvable=int(V.any(1).sum()),
                dev_root_ids=[w["seed"] for w in dv],dev_solvable_bits=V.any(1).astype(int).tolist(),
                exact_labels=int(Y.size+V.size),label_seconds=label_seconds,
                analytic_rank_seconds=analytic_seconds,
                exact_float_comparison=dict(false_positive=false_pos,false_negative=false_neg,
                                            near_threshold=ambiguous,total=int(V.size)),
                arms=arms)
    payload=json.dumps(result,indent=2,sort_keys=True)
    if args.output:
        with open(args.output,"w") as f:
            f.write(payload+"\n")
    print(payload)


if __name__=="__main__":
    main()
