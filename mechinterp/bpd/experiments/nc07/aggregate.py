"""NC-07 frozen eight-model-root DEVELOPMENT decision, root-cluster bootstrap only.

The assessor reads original root receipts. It never trains models, selects a
different rank, or evaluates task labels not already inside the frozen record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mechinterp.bpd.experiments.nc07.experiment import DEVELOPMENT, PROTO_HASH

ROOT=Path(__file__).resolve().parent
BOOT_REPEATS=2000
BOOT_SEED=707007


def paired_root_bootstrap(matrix:list[tuple[float,float,float]],
                          *,n:int=BOOT_REPEATS,seed:int=BOOT_SEED):
    """Sample each of eight complete trained-model roots once per draw."""
    values=np.asarray(matrix,dtype=float)
    if values.shape!=(8,3) or not np.isfinite(values).all():
        raise ValueError("Need eight independent roots and three paired metrics")
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,8,size=(n,8))
    means=values[idx].mean(1)
    return {
        "seed":seed,"repeats":n,
        "selected_cost":{"mean":float(values[:,0].mean()),
                         "lower95":float(np.percentile(means[:,0],2.5)),
                         "upper95":float(np.percentile(means[:,0],97.5))},
        "sham_advantage":{"mean":float(values[:,1].mean()),
                          "lower95":float(np.percentile(means[:,1],2.5)),
                          "upper95":float(np.percentile(means[:,1],97.5))},
        "selected_minus_fixed4":{"mean":float(values[:,2].mean()),
                                 "lower95":float(np.percentile(means[:,2],2.5)),
                                 "upper95":float(np.percentile(means[:,2],97.5))},
    }


def aggregate(records:list[dict]):
    by={r["root"]:r for r in records}
    if len(records)!=8 or len(by)!=8 or set(by)!=set(DEVELOPMENT):
        raise ValueError("Require precisely roots 700..707")
    roots=[]
    for seed in DEVELOPMENT:
        r=by[seed]
        if r["study"]!="BPD-NC-07" or r["phase"]!="DEVELOPMENT" or \
                r["protocol_sha256"]!=PROTO_HASH:
            raise AssertionError("Wrong NC07 source or phase")
        selected=r["selected_rank"]
        scores=r["heldout"] or {}
        score=scores.get("selected")
        fixed3=scores.get("fixed_rank3")
        fixed4=scores.get("fixed_rank4")
        sham=r.get("median_sham_minus_selected_nll_nats")
        row={
            "root":seed,"eligible":r["eligible"],"selected_rank":selected,
            "clean_nll":r.get("clean_final_nll_nats"),
            "selector":r["selector_by_rank"],
            "selected_nll":score.get("nll_nats") if score else None,
            "fixed3_nll":fixed3.get("nll_nats") if fixed3 else None,
            "fixed4_nll":fixed4.get("nll_nats") if fixed4 else None,
            "matched_sham_advantage":sham,
        }
        if all(isinstance(row[x],(int,float)) for x in
               ("clean_nll","selected_nll","fixed4_nll","matched_sham_advantage")):
            row["selected_keep_cost"]=row["selected_nll"]-row["clean_nll"]
            row["selected_minus_fixed4"]=row["selected_nll"]-row["fixed4_nll"]
        roots.append(row)
    eligible=all(r["eligible"] and r["selected_rank"] in (2,3,4,6)
                 for r in records)
    if eligible:
        matrix=[(x["selected_keep_cost"],x["matched_sham_advantage"],
                 x["selected_minus_fixed4"]) for x in roots]
        stats=paired_root_bootstrap(matrix)
        rank_values=[x["selected_rank"] for x in roots]
        count_under4=sum(r<=3 for r in rank_values)
        mean_rank=sum(rank_values)/8
    else:
        stats=None;count_under4=0;mean_rank=None
    gates={
        "eight_full_model_roots_and_shams":eligible,
        "compress_dimension_vs_rank4":bool(
            eligible and mean_rank<=3.5 and count_under4>=4
        ),
        "absolute_true_label_fidelity":bool(
            eligible and stats["selected_cost"]["mean"]<=.05
            and stats["selected_cost"]["upper95"]<=.05
        ),
        "physical_dose_sham_specificity":bool(
            eligible and stats["sham_advantage"]["mean"]>=.02
            and stats["sham_advantage"]["lower95"]>0
        ),
        "bounded_task_cost_vs_rank4":bool(
            eligible and stats["selected_minus_fixed4"]["upper95"]<=.02
        ),
    }
    passed=all(gates.values())
    return {
        "study":"BPD-NC-07","phase":"DEVELOPMENT",
        "evidence_class":"ENGINEERING_ONLY",
        "protocol_sha256":PROTO_HASH,
        "independent_model_roots":8,
        "eligible_roots":sum(bool(r["eligible"]) for r in records),
        "selected_rank_counts":{
            str(rank):sum(x["selected_rank"]==rank for x in roots)
            for rank in (2,3,4,6)
        },
        "abstained_roots":[x["root"] for x in roots if x["selected_rank"] is None],
        "mean_selected_rank":mean_rank,
        "selected_rank_le3_roots":count_under4,
        "bootstrap":stats,
        "gates":gates,
        "registered_verdict":"ENGINEERING_POSITIVE" if passed else
            "ENGINEERING_NULL_OR_INELIGIBLE",
        "confirmed_scientific_result":False,
        "calibration_roots_evaluated":0,
        "confirmation_roots_evaluated":0,
        "per_root":roots,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    records=[
        json.loads((args.root_dir/f"nc07_root_{seed}.json").read_text())
        for seed in DEVELOPMENT
    ]
    summary=aggregate(records)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in summary.items()
                      if k not in ("per_root","bootstrap")},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
