"""BPD-NC-09 fixed-12 first-run 16-model-root engineering aggregator.

This reads frozen completed model receipts. Every model root is one sampling
unit. An ineligible root leaves the joint development score scientifically
unresolved/failed, while preserving its data for descriptive audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mechinterp.bpd.experiments.nc08.experiment import candidate_budget_knapsack
from mechinterp.bpd.experiments.nc09_fixed12.experiment import ROOTS, PROTO_SHA

BOOTSTRAP_DRAWS=4096
BOOTSTRAP_SEED=909009
ARMS=("dp12","flat3","flat3_dose","flat4","reverse")
METRICS=("flat3_minus_dp12","doseflat3_minus_dp12",
         "dp12_minus_clean","dp12_minus_flat4","matched_sham_minus_dp12")


def rooted_bootstrap(records:list[tuple[float,...]], *,
                     seed:int=BOOTSTRAP_SEED,
                     n:int=BOOTSTRAP_DRAWS) -> dict:
    a=np.asarray(records,dtype=float)
    if a.shape!=(16,len(METRICS)) or not np.isfinite(a).all():
        raise ValueError("Require 16 complete independent model roots, five paired metrics")
    rng=np.random.default_rng(seed)
    sampled=rng.integers(0,len(ROOTS),size=(n,len(ROOTS)))
    means=a[sampled].mean(axis=1)
    return {
        "seed":seed,"draws":n,
        "metrics":{
            name:{
                "mean":float(a[:,i].mean()),
                "lower95":float(np.percentile(means[:,i],2.5)),
                "upper95":float(np.percentile(means[:,i],97.5)),
            } for i,name in enumerate(METRICS)
        }
    }


def source_dp12_check(root_record:dict):
    raw=root_record["passwise_source_candidates"]
    candidates={int(t):{int(rank):obj for rank,obj in per.items()}
                for t,per in raw.items()}
    dp=candidate_budget_knapsack(candidates)[12]
    registered=root_record["fixed_budget12_allocation"]
    if dp is None:
        if registered is not None:
            raise AssertionError("NC09 source optimizer found no rank12 tuple")
        return
    if registered is None:
        raise AssertionError("NC09 source optimizer omitted available rank12 tuple")
    if list(dp["ranks"])!=list(registered["ranks"]):
        raise AssertionError("NC09 fixed rank12 tuple differs from exact source DP")
    if abs(float(dp["proxy_loss"])-float(registered["source_proxy_loss"]))>1e-10:
        raise AssertionError("NC09 source proxy changed post-registration")


def aggregate(rows:list[dict]) -> dict:
    by={r["root"]:r for r in rows}
    if len(rows)!=16 or len(by)!=16 or set(by)!=set(ROOTS):
        raise ValueError("NC09 requires exactly 16 unique independent roots 900..915")
    per=[]
    for root in ROOTS:
        r=by[root]
        if (r["study"]!="BPD-NC-09-fixed12" or
                r["phase"]!="DEVELOPMENT" or r["protocol_sha256"]!=PROTO_SHA):
            raise AssertionError("Wrong developmental source or protocol")
        source_dp12_check(r)
        allocation=r["fixed_budget12_allocation"]
        if allocation is not None:
            if len(allocation["ranks"])!=4 or sum(allocation["ranks"])!=12:
                raise AssertionError("NC09 physical rank budget changed")
            if r["arms"]["dp12"]["total_rank"]!=12 or \
                    r["arms"]["flat3"]["total_rank"]!=12 or \
                    r["arms"]["flat3_dose"]["total_rank"]!=12 or \
                    r["arms"]["flat4"]["total_rank"]!=16:
                raise AssertionError("NC09 same-rank and fidelity controls mismatched")
        scores=r["heldout"]
        nll={key:scores[key]["nll_nats"] if scores.get(key) else None
             for key in ARMS}
        clean=r.get("clean_final_nll_nats")
        sham=r.get("median_sham_minus_dp12_nll")
        row={
            "root":root,
            "admitted":bool(r["eligible"]),
            "ranks":allocation["ranks"] if allocation else None,
            "source_proxy_loss":allocation["source_proxy_loss"] if allocation else None,
            "dose_ratio_flat3_to_dp12":r.get("dose_matched_flat3_gain"),
            "source_dp12_write_rms":r.get("dp12_source_write_rms"),
            "source_flat3_write_rms":r.get("flat3_source_write_rms"),
            "qualified_sham_proposals":r["sham_admission"]["qualified"]
                                      if r.get("sham_admission") else 0,
            "selected_shams":r["sham_admission"]["selected"]
                               if r.get("sham_admission") else 0,
            "clean_nll":clean,
            **{k+"_nll":nll[k] for k in ARMS},
            "matched_sham_minus_dp12":sham,
        }
        if all(isinstance(x,(float,int)) for x in (
            clean,nll["dp12"],nll["flat3"],nll["flat3_dose"],
            nll["flat4"],sham
        )):
            row.update({
                "flat3_minus_dp12":nll["flat3"]-nll["dp12"],
                "doseflat3_minus_dp12":nll["flat3_dose"]-nll["dp12"],
                "dp12_minus_clean":nll["dp12"]-clean,
                "dp12_minus_flat4":nll["dp12"]-nll["flat4"],
            })
        if isinstance(nll["reverse"],(float,int)) and \
                isinstance(nll["dp12"],(float,int)):
            row["reverse_minus_dp12"]=nll["reverse"]-nll["dp12"]
        per.append(row)
    admitted=all(
        x["admitted"] and x["selected_shams"]==8 and
        all(isinstance(x.get(key),(float,int)) for key in METRICS)
        for x in per
    )
    bootstrap=None
    if admitted:
        bootstrap=rooted_bootstrap([
            tuple(float(r[key]) for key in METRICS) for r in per
        ])
    metrics=bootstrap["metrics"] if bootstrap is not None else None
    gates={
        "all_sixteen_native_roots_and_eight_shams":admitted,
        "fixed12_beats_equal12_flat3":bool(
            admitted and metrics["flat3_minus_dp12"]["mean"]>=.008 and
            metrics["flat3_minus_dp12"]["lower95"]>0
        ),
        "fixed12_beats_dose_matched_flat3":bool(
            admitted and metrics["doseflat3_minus_dp12"]["mean"]>=.006 and
            metrics["doseflat3_minus_dp12"]["lower95"]>0
        ),
        "absolute_task_fidelity":bool(
            admitted and metrics["dp12_minus_clean"]["mean"]<=.05 and
            metrics["dp12_minus_clean"]["upper95"]<=.05
        ),
        "near_rank4_task_fidelity":bool(
            admitted and metrics["dp12_minus_flat4"]["mean"]<=.020 and
            metrics["dp12_minus_flat4"]["upper95"]<=.020
        ),
        "source_dose_matched_sham_specificity":bool(
            admitted and metrics["matched_sham_minus_dp12"]["mean"]>=.020 and
            metrics["matched_sham_minus_dp12"]["lower95"]>0
        ),
    }
    return {
        "study":"BPD-NC-09-fixed12",
        "phase":"DEVELOPMENT","evidence_class":"ENGINEERING_ONLY",
        "protocol_sha256":PROTO_SHA,
        "independent_model_roots":16,
        "admitted_roots":sum(bool(x["admitted"]) for x in per),
        "model_budget":12,"physical_reference_budget":16,
        "budget_saving_percent":25.0,
        "nonflat_allocations":sum(bool(x["ranks"] is not None and
                                   x["ranks"]!=[3,3,3,3]) for x in per),
        "bootstrap":bootstrap,
        "gates":gates,
        "registered_verdict":"DEVELOPMENT_POSITIVE" if all(gates.values())
                             else "DEVELOPMENT_NULL_OR_INELIGIBLE",
        "confirmed_scientific_result":False,
        "calibration_roots_evaluated":0,
        "confirmation_roots_evaluated":0,
        "secondary_mean_reverse_minus_dp12":(
            sum(float(x["reverse_minus_dp12"]) for x in per)/len(per)
            if all("reverse_minus_dp12" in x for x in per) else None
        ),
        "per_root":per,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    recs=[
        json.loads((args.root_dir/("nc09_fixed12_root_%d.json"%seed)).read_text())
        for seed in ROOTS
    ]
    summary=aggregate(recs)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in summary.items()
                      if k not in ("per_root","bootstrap")},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
