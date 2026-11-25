"""Exact finite-horizon BPD mask risk. Standard library only."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from fractions import Fraction as F
from pathlib import Path

SEED=20261125061
N_ROOTS=128
D=8
K=2
H=2


def dot(a,b):
    return sum((x*y for x,y in zip(a,b,strict=True)),F(0))


def norm2(a):
    return dot(a,a)


def add(vectors,dim):
    return tuple(sum((v[i] for v in vectors),F(0)) for i in range(dim))


def scale(v,s):
    return tuple(s*x for x in v)


def apply(columns,x):
    return add([scale(u,a) for u,a in zip(columns,x,strict=True)],len(columns[0]))


def signature(x,T,C,horizon=H):
    y=[]
    for _ in range(horizon+1):
        y.extend(dot(row,x) for row in C)
        x=apply(T,x)
    return tuple(y)


def signed_perm(rng,n):
    p=list(range(n))
    rng.shuffle(p)
    return [tuple((rng.choice((-1,1)) if j==i else 0) for j in range(n)) for i in p]


def rotation(n):
    # Orthogonal block diagonal Walsh H4/2.
    if n%4:
        raise ValueError("width must be divisible by four")
    return [tuple(F((-1)**((i%4 & j%4).bit_count()),2) if i//4==j//4 else F(0)
                  for j in range(n)) for i in range(n)]


def rotate(a,U,Q):
    return tuple(dot(q,a) for q in Q),[apply(U,q) for q in Q]


def risk_for(mask,vectors):
    keep=set(mask)
    return norm2(add([v for i,v in enumerate(vectors) if i not in keep],len(vectors[0])))


def gram(vectors):
    return [[dot(u,v) for v in vectors] for u in vectors]


def gram_risk(mask,G):
    keep=set(mask)
    missing=[i for i in range(len(G)) if i not in keep]
    return sum((G[i][j] for i in missing for j in missing),F(0))


def diagonal(vectors,k):
    return tuple(sorted(sorted(range(len(vectors)),key=lambda i:(-norm2(vectors[i]),i))[:k]))


def greedy(vectors,k):
    keep=()
    for _ in range(k):
        keep=min((tuple(sorted((*keep,i))) for i in range(len(vectors)) if i not in keep),
                 key=lambda m:(risk_for(m,vectors),m))
    return keep


def bounded(error,energy):
    return error/(error+energy) if error+energy else F(0)


def strf(x):
    return str(F(x))


def basis(a,U,T,C,k=K):
    n=len(a)
    writes=[scale(u,x) for x,u in zip(a,U,strict=True)]
    obs=[signature(v,T,C) for v in writes]
    present=[v[:len(C)] for v in obs]
    full=add(writes,len(U[0]))
    visible=add(obs,len(obs[0]))
    assert signature(full,T,C)==visible
    energies=[norm2(full),norm2(visible),norm2(visible[:len(C)])]
    Gp,Go=gram(writes),gram(obs)
    masks=list(itertools.combinations(range(n),k))
    table={}
    for mask in masks:
        p,o,c=[risk_for(mask,x) for x in (writes,obs,present)]
        assert p==gram_risk(mask,Gp) and o==gram_risk(mask,Go)
        table[mask]=(p,o,c)
    pbest=min(masks,key=lambda m:(table[m][0],m))
    obest=min(masks,key=lambda m:(table[m][1],m))
    choices={"physical_diagonal":diagonal(writes,k),
             "observable_diagonal":diagonal(obs,k),
             "observable_greedy":greedy(obs,k),
             "physical_oracle":pbest,"observable_oracle":obest}
    result={}
    for name,m in choices.items():
        v=table[m]
        assert table[pbest][0]<=v[0] and table[obest][1]<=v[1]
        result[name]={"mask":list(m),"risk":[strf(x) for x in v],
                      "bounded":[strf(bounded(x,e)) for x,e in zip(v,energies,strict=True)]}
    result["uniform"]={"mask":None,"risk":[strf(sum((table[m][j] for m in masks),F(0))/len(masks))
                                             for j in range(3)],
                       "bounded":[strf(sum((bounded(table[m][j],energies[j]) for m in masks),F(0))/len(masks))
                                  for j in range(3)]}
    # Keep every enumerated risk, not just the chosen masks.
    return {"full":[strf(x) for x in full],"observed_full":[strf(x) for x in visible],
            "energies":[strf(x) for x in energies],"arms":result,
            "enumerated":[{"mask":list(m),"risk":[strf(x) for x in table[m]]} for m in masks]}


def make_world(rng):
    U=signed_perm(rng,D)
    T=signed_perm(rng,D)
    high=set(rng.sample(range(D),2))
    a=tuple(rng.choice((-1,1))*(rng.randint(4,7) if i in high else rng.randint(0,2))
            for i in range(D))
    C=tuple(tuple(rng.randint(-3,3) for _ in range(D)) for _ in range(2))
    return a,U,T,C


def mean(seq):
    seq=list(seq)
    return sum(seq,F(0))/len(seq)


def summarize(rows):
    summary={}
    for b in ("native","rotated"):
        arms=rows[0][b]["arms"]
        summary[b]={}
        for arm in arms:
            summary[b][arm]={name:[float(mean(F(r[b]["arms"][arm][name][j]) for r in rows))
                                    for j in range(3)] for name in ("risk","bounded")}
        summary[b]["comparisons"]={}
        for x,y in (("observable_diagonal","physical_diagonal"),("observable_greedy","physical_diagonal"),
                    ("observable_greedy","observable_diagonal"),("observable_oracle","observable_greedy")):
            delta=[F(r[b]["arms"][y]["risk"][1])-F(r[b]["arms"][x]["risk"][1]) for r in rows]
            summary[b]["comparisons"][x+"_vs_"+y]={
                "wins":sum(v>0 for v in delta),"ties":sum(v==0 for v in delta),
                "losses":sum(v<0 for v in delta),"mean_observable_gain":float(mean(delta)),
                "different_masks":sum(r[b]["arms"][x]["mask"]!=r[b]["arms"][y]["mask"] for r in rows)}
    summary["native_vs_rotated"]={}
    for arm in rows[0]["native"]["arms"]:
        summary["native_vs_rotated"][arm]={}
        for j,metric in enumerate(("physical","observable","present")):
            gains=[F(r["rotated"]["arms"][arm]["risk"][j])-F(r["native"]["arms"][arm]["risk"][j])
                   for r in rows]
            summary["native_vs_rotated"][arm][metric]={"native_wins":sum(x>0 for x in gains),
                 "ties":sum(x==0 for x in gains),"native_losses":sum(x<0 for x in gains),
                 "mean_native_gain":float(mean(gains))}
    return summary


def self_test():
    I=lambda n:[tuple(int(i==j) for j in range(n)) for i in range(n)]
    # Physical selection loses a readout despite being optimal physically.
    a=(2,1);U=I(2);C=((0,1),)
    p=[scale(u,x) for u,x in zip(U,a)]
    o=[tuple(dot(c,v) for c in C) for v in p]
    assert diagonal(p,1)==(0,) and diagonal(o,1)==(1,)
    assert risk_for((0,),p)==1 and risk_for((0,),o)==1
    assert risk_for((1,),p)==4 and risk_for((1,),o)==0
    T=[(0,1),(1,0)]
    assert all(risk_for(m,p)==risk_for(m,[signature(v,T,C,1) for v in p]) for m in [(),(0,),(1,),(0,1)])
    # Signed cancellation and failure of diagonal selection.
    v=[(6,),(5,),(-6,)]
    assert diagonal(v,1)==(0,) and risk_for((0,),v)==1
    assert greedy(v,1)==(1,) and risk_for((1,),v)==0
    v=[(1,),(-1,)]
    assert [risk_for(m,v) for m in [(),(0,),(1,),(0,1)]]==[0,1,1,0]
    assert risk_for((),v)-risk_for((0,),v)<risk_for((1,),v)-risk_for((0,1),v)
    Q=rotation(8)
    assert gram(Q)==I(8)
    a,U,T,C=make_world(random.Random(101))
    b,V=rotate(a,U,Q)
    assert apply(U,a)==apply(V,b)
    assert norm2(a)==norm2(b)
    x,y=basis(a,U,T,C),basis(b,V,T,C)
    assert x["full"]==y["full"] and x["observed_full"]==y["observed_full"]
    print("BPD-NBR-01 exact witness and rotation tests PASS")


def run(out):
    rng=random.Random(SEED)
    Q=rotation(D)
    rows=[]
    for i in range(N_ROOTS):
        a,U,T,C=make_world(rng)
        b,V=rotate(a,U,Q)
        assert norm2(a)==norm2(b)
        x,y=basis(a,U,T,C),basis(b,V,T,C)
        assert x["full"]==y["full"] and x["observed_full"]==y["observed_full"]
        rows.append({"root":i,"a":a,"U":U,"T":T,"C":C,"native":x,"rotated":y})
    report={"schema":"BPD/NBR01/exact-risk-v1","synthetic_only":True,"seed":SEED,
            "roots":N_ROOTS,"dimension":D,"cardinality":K,"horizon":H,
            "metrics_order":["physical","observable","present"],
            "engineering_gate":"PASS","enumerated_masks":N_ROOTS*2*len(list(itertools.combinations(range(D),K))),
            "summary":summarize(rows),"worlds":rows}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as f:
        json.dump(report,f,sort_keys=True,separators=(",",":"),allow_nan=False);f.write("\n")
    print("BPD-NBR-01 result",json.dumps({"roots":N_ROOTS,"engineering_gate":"PASS",
        "enumerated_masks":report["enumerated_masks"],"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
        "summary":report["summary"]},sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.self_test:self_test()
    elif a.out:run(a.out)
    else:p.error("supply --self-test or --out")


if __name__=="__main__":
    main()
