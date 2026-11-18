"""Root-level NC-06 engineering decision and immutable raw development receipt audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mechinterp.bpd.experiments.nc06.experiment import DEV_ROOTS, PROTOCOL_HASH

ROOT=Path(__file__).resolve().parent
ARMS=("norm_r3","uniform_r3","shuffle_r3","uniform_r4_reference")


def boot_ci(values:list[float],seed:int=606006,n:int=2000):
    a=np.asarray(values,dtype=float)
    if len(a)!=8 or not np.isfinite(a).all():
        raise ValueError("Need exactly eight finite independent root values")
    gen=np.random.default_rng(seed)
    draws=gen.integers(0,8,size=(n,8))
    means=a[draws].mean(axis=1)
    return {"mean":float(a.mean()),"lower95":float(np.percentile(means,2.5)),
            "upper95":float(np.percentile(means,97.5)),
            "repeats":n,"seed":seed}


def aggregate(receipts:list[dict]):
    by={r["root"]:r for r in receipts}
    if len(by)!=len(receipts) or set(by)!=set(DEV_ROOTS):
        raise ValueError("Need eight distinct NC06 development roots")
    frozen=all(
        r["study"]=="BPD-NC-06" and r["phase"]=="ENGINEERING" and
        r["protocol_sha256"]==PROTOCOL_HASH for r in receipts
    )
    eligible=all(r["eligible"] and
                 all(r["methods"][arm]["status"]=="ELIGIBLE" for arm in ARMS)
                 for r in receipts)
    per=[]
    for root in DEV_ROOTS:
        r=by[root]
        method=r["methods"]
        clean=r["clean_nll"]
        vals={
            arm:method[arm].get("scored",{}).get("nll_nats",None)
            for arm in ARMS
        }
        row={"root":root,"eligible":r["eligible"],
             "clean_nll":clean,
             **{arm+"_nll":vals[arm] for arm in ARMS},
             "norm_r3_sham_advantage":
                 method["norm_r3"].get("median_sham_minus_kept_nll",None)}
        if all(isinstance(x,(int,float)) for x in vals.values()) and \
                isinstance(row["norm_r3_sham_advantage"],(int,float)):
            row.update({
                "norm_keep_cost":vals["norm_r3"]-clean,
                "unif_minus_norm":vals["uniform_r3"]-vals["norm_r3"],
                "shuffle_minus_norm":vals["shuffle_r3"]-vals["norm_r3"],
                "rank4_keep_cost":vals["uniform_r4_reference"]-clean,
            })
        per.append(row)
    stats={}
    for key in ("norm_keep_cost","unif_minus_norm","shuffle_minus_norm",
                "norm_r3_sham_advantage","rank4_keep_cost"):
        values=[row.get(key) for row in per]
        stats[key]=boot_ci(values) if all(isinstance(x,(int,float)) for x in values) else None
    valid=bool(frozen and eligible and all(stats[x] is not None
        for x in ("norm_keep_cost","unif_minus_norm","shuffle_minus_norm",
                  "norm_r3_sham_advantage")))
    gates={
        "all_roots_eligible":bool(eligible),
        "norm_rank3_absolute_fidelity":bool(valid and stats["norm_keep_cost"]["mean"]<=.05
                                            and stats["norm_keep_cost"]["upper95"]<=.05),
        "norm_beats_uniform_rank3":bool(valid and stats["unif_minus_norm"]["mean"]>=.01
                                         and stats["unif_minus_norm"]["lower95"]>0),
        "norm_beats_shuffled_rank3":bool(valid and stats["shuffle_minus_norm"]["mean"]>=.01
                                          and stats["shuffle_minus_norm"]["lower95"]>0),
        "norm_beats_matched_sham":bool(valid and stats["norm_r3_sham_advantage"]["mean"]>=.02
                                        and stats["norm_r3_sham_advantage"]["lower95"]>0)
    }
    passed=bool(frozen and all(gates.values()))
    return {"study":"BPD-NC-06","phase":"ENGINEERING",
            "evidence_class":"DEVELOPMENT_ONLY",
            "protocol_sha256":PROTOCOL_HASH,
            "model_roots":len(per),
            "source_frozen":frozen,
            "gates":gates,
            "frozen_development_positive":passed,
            "registered_verdict":"ENGINEERING_POSITIVE" if passed else "ENGINEERING_NULL_OR_INELIGIBLE",
            "statistics":stats,
            "per_root":per,
            "confirmation_roots_evaluated":0,
            "sham_calibration_roots_evaluated":0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    docs=[json.loads((args.root_dir/f"nc06_root_{seed}.json").read_text())
          for seed in DEV_ROOTS]
    result=aggregate(docs)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items()
                     if k not in ("statistics","per_root")},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
