"""Independent arithmetic audit of NBR-03 finite-risk development receipts."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOTS=tuple(range(202611250301,202611250309))
PATTERNS=[m for m in itertools.product((0,1),repeat=4) if sum(m)==2]
FIRST=("physical","local_joint","path_joint","finite_probe","distilled","random")
SECOND=("physical","local_joint","path_joint","finite_oracle","distilled","learned","random")

def mean(xs):
    return math.fsum(xs)/len(xs)

def check(a,n,width):
    assert len(a["mask"])==len(a["logits"])==n
    assert all(tuple(m) in PATTERNS for m in a["mask"])
    assert all(len(l)==width and all(math.isfinite(x) for x in l) for l in a["logits"])

def check_risks(risks,n):
    assert len(risks)==n
    assert all(len(v)==6 and all(math.isfinite(x) and x>=0 for x in v) for v in risks)

def near(a,b):
    return math.isclose(a,b,rel_tol=3e-5,abs_tol=2e-5)

def chosen(a,risk):
    return [v[PATTERNS.index(tuple(m))] for m,v in zip(a["mask"],risk,strict=True)]

def oracle(a,risk):
    assert all(near(x,min(v)) for x,v in zip(chosen(a,risk),risk,strict=True))

def metrics(a,y,full):
    return {"both_correct":sum(all(l*t>0 for l,t in zip(v,z,strict=True)) for v,z in zip(a,y,strict=True)),
            "first_correct":sum(v[0]*z[0]>0 for v,z in zip(a,y,strict=True)),
            "second_correct":sum(v[1]*z[1]>0 for v,z in zip(a,y,strict=True)),
            "first_mse":mean([(v[0]-z[0])**2 for v,z in zip(a,full,strict=True)]),
            "second_mse":mean([(v[1]-z[1])**2 for v,z in zip(a,full,strict=True)])}

def root(r):
    assert r["development_only"] and r["source_blob"]=="6f9e03268ea5c3d8a706ee547b8e76da296e86e7"
    assert r["native_max_error"]<=1e-5 and len(r["probes"])==8
    t=r["task"];y=t["targets"];full=t["full"];n=256
    assert len(y)==len(full)==len(t["teacher"])==n
    assert all(y[i][0]==y[i+1][0] and y[i][1]==-y[i+1][1] for i in range(0,n,2))
    out={"seed":r["seed"],"native_max_error":r["native_max_error"],
         "teacher":metrics(t["teacher"],y,full),"full":metrics(full,y,full),
         "first":{},"second":{},"donor":{},"distill":r["distill"]}
    out["competent"]=min(out["teacher"]["both_correct"],out["full"]["both_correct"])>=.9*n
    for phase,arms,key in (("first",FIRST,"finite_probe"),("second",SECOND,"finite_oracle")):
        risks=t[phase+"_risks"];check_risks(risks,n)
        assert set(t[phase])==set(arms)
        for name,a in t[phase].items():
            check(a,n,2)
            if phase=="first":
                assert all(a["mask"][i]==a["mask"][i+1] for i in range(0,n,2))
            values=chosen(a,risks)
            if name==key:oracle(a,risks)
            if phase=="second":
                assert all(near(z,(v[1]-f[1])**2) for z,v,f in zip(values,a["logits"],full,strict=True))
            out[phase][name]={**metrics(a["logits"],y,full),
                              "mean_oracle_regret":mean([x-min(v) for x,v in zip(values,risks,strict=True)])}
    d=r["donor"];n=128
    yr=d["recipient_targets"];yd=d["donor_targets"];f=d["donor_full"]
    assert len(yr)==len(yd)==len(f)==len(d["outside"])==n and not any(d["outside"])
    assert all(a[0]==b[0] and a[1]==-b[1] for a,b in zip(yr,yd,strict=True))
    risks=d["risks"];check_risks(risks,n)
    out["donor_full_hits"]=sum(l*y[1]>0 for l,y in zip(f,yd,strict=True))
    assert set(d["arms"])==set(SECOND)
    for name,pair in d["arms"].items():
        a,b=pair["feedback"],pair["frozen"]
        if name=="finite_oracle":oracle(a,risks)
        if name=="random":assert a==b
        rec={"switches":sum(x!=y for x,y in zip(a["mask"],b["mask"],strict=True))}
        for mode,v in pair.items():
            check(v,n,1)
            z=chosen(v,risks)
            assert all(near(x,(l[0]-j)**2) for x,l,j in zip(z,v["logits"],f,strict=True))
            rec[mode]={"hits":sum(l[0]*y[1]>0 for l,y in zip(v["logits"],yd,strict=True)),
                       "mse":mean(z),"mean_oracle_regret":mean([x-min(q) for x,q in zip(z,risks,strict=True)])}
        out["donor"][name]=rec
    return out

def contrast(rows,fn):
    xs=[fn(r) for r in rows]
    return {"mean":mean(xs),"wins":sum(x>0 for x in xs),"ties":sum(x==0 for x in xs),
            "losses":sum(x<0 for x in xs),"by_root":xs}

def summary(rows):
    res={"roots":len(rows),"competent_roots":sum(r["competent"] for r in rows),
         "teacher_both":sum(r["teacher"]["both_correct"] for r in rows),
         "full_both":sum(r["full"]["both_correct"] for r in rows),
         "donor_full_hits":sum(r["donor_full_hits"] for r in rows),
         "max_native_error":max(r["native_max_error"] for r in rows)}
    for phase,arms in (("first",FIRST),("second",SECOND)):
        res[phase]={a:{"both_correct":sum(r[phase][a]["both_correct"] for r in rows),
                       **{k:mean([r[phase][a][k] for r in rows])
                          for k in ("first_mse","second_mse","mean_oracle_regret")},
                       "second_mse_gain_vs_physical":contrast(rows,lambda r:r[phase]["physical"]["second_mse"]-r[phase][a]["second_mse"])}
                    for a in arms}
    res["distill"]={p:{k:mean([r["distill"][p][k] for r in rows])
                      for k in ("calibration_choice_agreement","calibration_mean_regret","parameters")}
                   for p in ("first","second")}
    res["donor"]={}
    for a in SECOND:
        def p(r):return r["donor"][a]
        res["donor"][a]={"feedback_hits":sum(p(r)["feedback"]["hits"] for r in rows),
                         "frozen_hits":sum(p(r)["frozen"]["hits"] for r in rows),
                         "feedback_mse":mean([p(r)["feedback"]["mse"] for r in rows]),
                         "frozen_mse":mean([p(r)["frozen"]["mse"] for r in rows]),
                         "switches":sum(p(r)["switches"] for r in rows),
                         "feedback_gain_hits_per_pair":contrast(rows,lambda r:(p(r)["feedback"]["hits"]-p(r)["frozen"]["hits"])/128)}
    return res

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();assert not a.out.exists()
    rows=[];hashes={}
    for seed in ROOTS:
        b=(a.input/f"{seed}.json").read_bytes();r=json.loads(b);assert r["seed"]==seed
        hashes[str(seed)]=hashlib.sha256(b).hexdigest();rows.append(root(r))
    assert len(list(a.input.glob("*.json")))==len(ROOTS)
    s=summary(rows)
    a.out.write_text(json.dumps({"schema":"BPD/NBR03/audit-v1","engineering_gate":"PASS",
        "hashes":hashes,"summary":s,"roots":rows,
        "scope":"independent arithmetic and masks; does not retrain or regenerate finite candidate risks"},
        indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("BPD-NBR-03 AUDIT",json.dumps(s,separators=(",",":"),allow_nan=False))
if __name__=="__main__":main()
