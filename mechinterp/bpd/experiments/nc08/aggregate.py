"""NC-08 frozen eight-model-root engineering aggregator.

One independent root counts once. Any nonadmitted root blocks all preregistered
root-bootstrap PASS claims. The original receipts remain immutable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mechinterp.bpd.experiments.nc08.experiment import ROOTS, PROTO_HASH

BOOT=2000
BOOT_SEED=808008


def root_bootstrap(values:list[tuple[float,float,float]],*,n:int=BOOT,seed:int=BOOT_SEED):
    a=np.asarray(values,dtype=float)
    if a.shape!=(8,3) or not np.isfinite(a).all():
        raise ValueError("NC08 requires exactly 8 finite 3-metric root records")
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,8,size=(n,8))
    means=a[ix].mean(axis=1)
    names=("keep_cost","sham_margin","selected_minus_flat4")
    return {
        name:{"mean":float(a[:,k].mean()),
              "lower95":float(np.percentile(means[:,k],2.5)),
              "upper95":float(np.percentile(means[:,k],97.5))}
        for k,name in enumerate(names)
    }|{"seed":seed,"repetitions":n}


def aggregate(records:list[dict]):
    by={r["root"]:r for r in records}
    if len(records)!=8 or len(by)!=8 or set(by)!=set(ROOTS):
        raise ValueError("Exactly independent model roots 800..807 required")
    per=[]
    for seed in ROOTS:
        rec=by[seed]
        if rec["study"]!="BPD-NC-08" or rec["phase"]!="DEVELOPMENT":
            raise AssertionError("Wrong study phase")
        if rec["protocol_sha256"]!=PROTO_HASH:
            raise AssertionError("Wrong source-only protocol hash")
        selected=rec["selected_total_rank"]
        cards=rec["candidates_by_budget"]
        chosen=next((budget for budget in (8,10,12,14)
                     if cards[str(budget)]["eligible"]),None)
        if selected!=chosen:
            raise AssertionError("Selected budget violates smallest source-only rule")
        tuple_r=rec["selected_rank_tuple"]
        if tuple_r is not None and (sum(tuple_r)!=selected or
                len(tuple_r)!=4 or any(r not in (2,3,4,5,6) for r in tuple_r)):
            raise AssertionError("Invalid selected passwise rank tuple")
        scores=rec["heldout"] or {}
        clean=rec.get("clean_final_nll_nats")
        selected_nll=(scores["selected"]["nll_nats"]
                      if scores.get("selected") is not None else None)
        baseline3=(scores["flat_rank3"]["nll_nats"]
                   if scores.get("flat_rank3") is not None else None)
        baseline4=(scores["flat_rank4"]["nll_nats"]
                   if scores.get("flat_rank4") is not None else None)
        dp12=(scores["dp_budget12"]["nll_nats"]
              if scores.get("dp_budget12") is not None else None)
        sham=rec.get("median_sham_minus_selected_nll_nats")
        row={
            "root":seed,"eligible":bool(rec["eligible"]),
            "selected_total_rank":selected,
            "selected_rank_tuple":tuple_r,
            "selected_shams":rec["sham_admission"]["selected"]
                if rec.get("sham_admission") is not None else 0,
            "clean_nll":clean,"selected_nll":selected_nll,
            "flat_rank3_nll":baseline3,"flat_rank4_nll":baseline4,
            "dp_budget12_nll":dp12,"matched_sham_advantage":sham,
        }
        if all(isinstance(row[k],(int,float)) for k in (
            "clean_nll","selected_nll","flat_rank4_nll","matched_sham_advantage"
        )):
            row.update({
                "keep_cost":selected_nll-clean,
                "selected_minus_flat4":selected_nll-baseline4,
            })
        if isinstance(dp12,(int,float)) and isinstance(baseline3,(int,float)):
            row["dp_budget12_minus_flat3"]=dp12-baseline3
        per.append(row)
    admitted=all(x["eligible"] and x["selected_total_rank"] in (8,10,12,14)
                 and x["selected_shams"]==8 and
                 all(isinstance(x.get(k),(int,float)) for k in (
                    "keep_cost","selected_minus_flat4","matched_sham_advantage"))
                 for x in per)
    bootstrap=None
    mean_budget=None
    count_le12=0
    if admitted:
        bootstrap=root_bootstrap([
            (x["keep_cost"],x["matched_sham_advantage"],x["selected_minus_flat4"])
            for x in per
        ])
        budgets=[x["selected_total_rank"] for x in per]
        mean_budget=sum(budgets)/len(budgets)
        count_le12=sum(r<=12 for r in budgets)
    gates={
        "all_eight_roots_admitted":bool(admitted),
        "strict_total_rank_reduction":bool(admitted and mean_budget<=12.5
                                           and count_le12>=5),
        "absolute_external_task_fidelity":bool(
            admitted and bootstrap["keep_cost"]["mean"]<=.05
            and bootstrap["keep_cost"]["upper95"]<=.05
        ),
        "source_matched_sham_specificity":bool(
            admitted and bootstrap["sham_margin"]["mean"]>=.02
            and bootstrap["sham_margin"]["lower95"]>0
        ),
        "bounded_extra_task_cost_vs_flat_rank4":bool(
            admitted and bootstrap["selected_minus_flat4"]["upper95"]<=.02
        )
    }
    return {
        "study":"BPD-NC-08","phase":"DEVELOPMENT","evidence_class":"ENGINEERING_ONLY",
        "protocol_sha256":PROTO_HASH,
        "independent_model_roots":8,
        "admitted_roots":sum(x["eligible"] for x in per),
        "budget_counts":{str(b):sum(x["selected_total_rank"]==b for x in per)
                         for b in (8,10,12,14)},
        "mean_selected_budget":mean_budget,
        "selected_budget_le12_roots":count_le12,
        "abstained_roots":[x["root"] for x in per if x["selected_total_rank"] is None],
        "bootstrap":bootstrap,"gates":gates,
        "original_development_verdict":"ENGINEERING_POSITIVE"
            if all(gates.values()) else "ENGINEERING_NULL_OR_INELIGIBLE",
        "scientific_confirmation_performed":False,
        "independent_sham_calibration_roots_used":0,
        "independent_confirmation_roots_used":0,
        "per_root":per,
        "secondary_mean_dp12_minus_flat3":(
            sum(x["dp_budget12_minus_flat3"] for x in per)/8
            if all("dp_budget12_minus_flat3" in x for x in per) else None
        ),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    docs=[
        json.loads((args.root_dir/("nc08_root_%d.json"%root)).read_text())
        for root in ROOTS
    ]
    rec=aggregate(docs)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in rec.items()
                      if k not in ("per_root","bootstrap")},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
