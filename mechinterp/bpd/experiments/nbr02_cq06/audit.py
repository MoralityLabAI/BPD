"""Independent standard-library audit of NBR-02 saved native results.

Recomputes mask risks and outcome summaries, not model training or derivatives.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOTS=tuple(range(202611250201,202611250209))
MASKS=list(itertools.combinations(range(4),2))
SOURCE="6f9e03268ea5c3d8a706ee547b8e76da296e86e7"
FIRST=("physical","current_diagonal","delayed_diagonal","delayed_greedy","delayed_joint","random")
SECOND=("physical","observable_diagonal","observable_greedy","observable_joint","learned","random","finite_oracle")
PATCHES=("partial_dark","random_dark","sham","full_donor")

def mean(v):
    return math.fsum(v)/len(v)

def sqrisk(v,ids):
    return math.fsum(math.fsum(v[i][j] for i in range(4) if i not in ids)**2
                     for j in range(len(v[0])))

def ids(m):
    assert len(m)==4 and all(x in (0,1) for x in m) and sum(m)==2
    return tuple(i for i,x in enumerate(m) if x)

def close(a,b):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=3e-5,abs_tol=2e-5),(a,b)

def verify_arm(a,sig,phys,n):
    assert all(len(a[k])==n for k in ("mask","logits","proxy_risk","physical_risk"))
    for m,v,w,p,q,lg in zip(a["mask"],sig,phys,a["proxy_risk"],a["physical_risk"],a["logits"],strict=True):
        ix=ids(m);close(p,sqrisk(v,ix));close(q,sqrisk(w,ix))
        assert all(math.isfinite(x) for x in lg)

def selection(a,signature,kind):
    """Check cost optimality to numerical tolerance; tie identity may differ
    in double arithmetic, so check the objective rather than a reselected ID."""
    for m,v in zip(a["mask"],signature,strict=True):
        selected=ids(m)
        if kind=="joint":
            close(sqrisk(v,selected),min(sqrisk(v,ix) for ix in MASKS))
        elif kind=="diagonal":
            scores=[math.fsum(x*x for x in q) for q in v]
            close(sum(scores[i] for i in selected),sum(sorted(scores,reverse=True)[:2]))
        elif kind=="greedy":
            cost=[sqrisk(v,(i,)) for i in range(4)]
            # Allow float32 roundoff ties in the first choice.
            best=min(cost)
            first=[i for i in selected if math.isclose(cost[i],best,rel_tol=3e-5,abs_tol=2e-5)]
            assert first
            assert any(math.isclose(sqrisk(v,selected),
                                    min(sqrisk(v,tuple(sorted((i,j)))) for j in range(4) if j!=i),
                                    rel_tol=3e-5,abs_tol=2e-5) for i in first)

def task_metrics(logits,y,full):
    assert len(logits)==len(y)==len(full)
    z=[[x*t>0 for x,t in zip(l,ys,strict=True)] for l,ys in zip(logits,y,strict=True)]
    err=[mean([(l[j]-f[j])**2 for l,f in zip(logits,full,strict=True)]) for j in (0,1)]
    return {"both_correct":sum(all(v) for v in z),"first_correct":sum(v[0] for v in z),
            "second_correct":sum(v[1] for v in z),"first_mse":err[0],"second_mse":err[1],
            "two_readout_mse":mean(err)}

def check_finite(arms,full):
    l=arms["finite_oracle"]["logits"]
    for a in arms.values():
        for x,y,f in zip(l,a["logits"],full,strict=True):
            # Squared errors audited in float64; allow a small roundoff
            # allowance in addition to the native 1e-5 engineering check.
            assert (x[-1]-f)**2-(y[-1]-f)**2 <= 2e-5

def audit_root(r):
    assert r["source_blob"]==SOURCE and r["engineering_only"]
    assert r["native_max_abs_error"]<=1e-5 and r["access"]["cardinality"]==2
    task=r["task"];y=task["targets"];full=task["full"];n=256
    assert len(y)==n and len(task["probes"])==8
    assert all(y[i][0]==y[i+1][0] and y[i][1]==-y[i+1][1] for i in range(0,n,2))
    out={"seed":r["seed"],"teacher":task_metrics(task["teacher"],y,full),"full":task_metrics(full,y,full),
         "first":{},"second":{},"donor":{},"native_max_abs_error":r["native_max_abs_error"]}
    out["source_competent"]=min(out["teacher"]["both_correct"],out["full"]["both_correct"])/n>=.9
    assert set(task["first"])==set(FIRST) and set(task["second"])==set(SECOND)
    sig=task["first_signature"]
    for name,a in task["first"].items():
        verify_arm(a,sig["delayed"],sig["physical"],n)
        assert all(a["mask"][i]==a["mask"][i+1] for i in range(0,n,2))
        if name=="physical":selection(a,sig["physical"],"diagonal")
        elif name=="current_diagonal":selection(a,sig["current"],"diagonal")
        elif name.startswith("delayed_"):selection(a,sig["delayed"],name.split("_")[1])
        out["first"][name]=task_metrics(a["logits"],y,full)
    sig=task["second_signature"]
    for name,a in task["second"].items():
        verify_arm(a,sig["observable"],sig["physical"],n)
        if name=="physical":selection(a,sig["physical"],"diagonal")
        elif name.startswith("observable_"):selection(a,sig["observable"],name.split("_")[1])
        out["second"][name]=task_metrics(a["logits"],y,full)
    check_finite(task["second"],[f[1] for f in full])
    d=r["donor"];yr=d["recipient_targets"];yd=d["donor_targets"];n=128
    assert len(yr)==len(yd)==n
    assert all(a[0]==b[0] and a[1]==-b[1] for a,b in zip(yr,yd,strict=True))
    assert set(d["patches"])==set(PATCHES)
    clean=d["patches"]["sham"]["arms"]
    for patch,p in d["patches"].items():
        sig=p["signature"];f=p["full"]
        assert all(len(p[k])==n for k in ("full","current","dose","dark_error","valid","outside"))
        assert all(math.isfinite(x) for k in ("full","current","dose","dark_error") for x in p[k])
        if patch in ("partial_dark","random_dark"):
            for dose,raw,err,valid in zip(p["dose"],d["raw_dose"],p["dark_error"],p["valid"],strict=True):
                if valid:
                    assert abs(dose-raw)<=1e-5 and err<=1e-5
                else:assert dose<=1e-5
        if patch=="sham":
            assert max(p["dose"])==0 and p["current"]==d["clean_current"]
        if patch=="full_donor":
            assert not any(p["outside"])
        rec={"outside":sum(p["outside"]),"degenerate":n-sum(p["valid"]),
             "mean_dose":mean(p["dose"]),"max_dark_error":max(p["dark_error"]),
             "mean_abs_current_drift":mean([abs(a-b) for a,b in zip(p["current"],d["clean_current"])]),
             "full_target_hits":sum(l*b[1]>0 for l,b in zip(f,yd)),
             "policies":{}}
        assert set(p["arms"])==set(SECOND)
        for name,pair in p["arms"].items():
            for mode in ("feedback","frozen"):
                a=pair[mode];verify_arm(a,sig["observable"],sig["physical"],n)
                if mode=="frozen":assert a["mask"]==clean[name]["frozen"]["mask"]
            a,b=pair["feedback"],pair["frozen"]
            if name=="physical":selection(a,sig["physical"],"diagonal")
            elif name.startswith("observable_"):selection(a,sig["observable"],name.split("_")[1])
            if patch=="sham" or name=="random":
                assert a["mask"]==b["mask"] and a["logits"]==b["logits"]
            assert pair["switches"]==sum(x!=y for x,y in zip(a["mask"],b["mask"]))
            modes={}
            for mode,row in (("feedback",a),("frozen",b)):
                ls=[v[0] for v in row["logits"]]
                modes[mode]={"donor_hits":sum(l*t[1]>0 for l,t in zip(ls,yd)),
                             "full_mse":mean([(l-z)**2 for l,z in zip(ls,f)])}
            rec["policies"][name]={**modes,"switches":pair["switches"]}
        check_finite({a:v["feedback"] for a,v in p["arms"].items()},f)
        out["donor"][patch]=rec
    return out

def contrast(rows,func):
    v=[func(r) for r in rows]
    return {"mean":mean(v),"wins":sum(x>0 for x in v),"ties":sum(x==0 for x in v),
            "losses":sum(x<0 for x in v),"by_root":v}

def summarize(rows):
    out={"roots":len(rows),"competent_roots":sum(r["source_competent"] for r in rows),
         "teacher_both":sum(r["teacher"]["both_correct"] for r in rows),
         "full_both":sum(r["full"]["both_correct"] for r in rows),
         "max_native_abs_error":max(r["native_max_abs_error"] for r in rows)}
    for stage,arms in (("first",FIRST),("second",SECOND)):
        out[stage]={a:{"both_correct":sum(r[stage][a]["both_correct"] for r in rows),
                         "mean_first_mse":mean([r[stage][a]["first_mse"] for r in rows]),
                         "mean_second_mse":mean([r[stage][a]["second_mse"] for r in rows]),
                         "second_mse_gain_vs_physical":contrast(rows,lambda r:r[stage]["physical"]["second_mse"]-r[stage][a]["second_mse"])}
                     for a in arms}
    out["donor"]={}
    for patch in PATCHES:
        out["donor"][patch]={"outside":sum(r["donor"][patch]["outside"] for r in rows),
             "degenerate":sum(r["donor"][patch]["degenerate"] for r in rows),
             "full_target_hits":sum(r["donor"][patch]["full_target_hits"] for r in rows),"policies":{}}
        for a in SECOND:
            def p(r):return r["donor"][patch]["policies"][a]
            out["donor"][patch]["policies"][a]={
              "feedback_hits":sum(p(r)["feedback"]["donor_hits"] for r in rows),
              "frozen_hits":sum(p(r)["frozen"]["donor_hits"] for r in rows),
              "feedback_mse":mean([p(r)["feedback"]["full_mse"] for r in rows]),
              "frozen_mse":mean([p(r)["frozen"]["full_mse"] for r in rows]),
              "feedback_gain_hits_per_pair":contrast(rows,lambda r:(p(r)["feedback"]["donor_hits"]-p(r)["frozen"]["donor_hits"])/128),
              "switches":sum(p(r)["switches"] for r in rows)}
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();assert not a.out.exists()
    rows=[];hashes={}
    for seed in ROOTS:
        q=a.input/f"{seed}.json";b=q.read_bytes();r=json.loads(b)
        assert r["seed"]==seed
        hashes[str(seed)]=hashlib.sha256(b).hexdigest()
        rows.append(audit_root(r))
    assert len(list(a.input.glob("*.json")))==len(ROOTS)
    summary=summarize(rows)
    report={"schema":"BPD/NBR02/audit-v1","engineering_gate":"PASS","hashes":hashes,
            "summary":summary,"roots":rows,"scope":"receipt arithmetic, not independent model or derivative replay"}
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("BPD-NBR-02 AUDIT",json.dumps(summary,separators=(",",":"),allow_nan=False))

if __name__=="__main__":main()
