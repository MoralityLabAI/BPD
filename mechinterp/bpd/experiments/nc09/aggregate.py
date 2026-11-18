"""NC-09 exact engineering-root aggregation; no confirmatory release path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import ROOTS, PROTOCOL_COMMIT, SHAMS

BOOTSTRAPS = 4096
COMPARISONS = 3
FAMILY_ALPHA = .05/COMPARISONS


def interval_root(effects: list[float], seed: int) -> dict:
    if len(effects)==0:
        return {"n_independent_roots":0,"mean":None,
                "familywise_interval":None,"status":"NO_ELIGIBLE_ROOTS"}
    arr=np.asarray(effects,dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("nonfinite root effect")
    rng=np.random.default_rng(seed)
    ids=rng.integers(0,len(arr),size=(BOOTSTRAPS,len(arr)))
    samples=arr[ids].mean(1)
    lo,hi=np.quantile(samples,[FAMILY_ALPHA/2,1-FAMILY_ALPHA/2])
    return {
        "n_independent_roots":len(arr),
        "mean":float(arr.mean()),
        "familywise_interval":[float(lo),float(hi)],
        "coverage":1-FAMILY_ALPHA,
        "paired_root_effects":[float(a) for a in arr],
        "scientific_status":"ENGINEERING_ONLY",
    }


def aggregate(receipts: list[dict]) -> dict:
    if len(receipts)!=len(ROOTS) or sorted(r["seed"] for r in receipts)!=list(ROOTS):
        raise ValueError("need exactly 8 distinct registered model roots 900..907")
    for r in receipts:
        if (r["study"]!="BPD-NC-09" or r["phase"]!="ENGINEERING"
                or r["protocol_commit"]!=PROTOCOL_COMMIT
                or r["data"]!={"train_base":2048,"train_task_marked":4096,
                               "discovery_base":1024,"evaluation_base":1024}
                or r["training"]["steps"]!=640):
            raise ValueError("root receipt differs from registered protocol")
    for key in ("source_sha256","core_sha256","protocol_sha256"):
        if len({r[key] for r in receipts})!=1:
            raise ValueError(f"mixed receipt SHA for {key}")
    rows=sorted(receipts,key=lambda r:r["seed"])
    eligible=[r for r in rows if r["eligible"] and r["status"]=="ENGINEERING_ELIGIBLE"]
    root_rows=[]
    for r in rows:
        root_rows.append({
            "seed":r["seed"], "eligible":bool(r["eligible"]),
            "admission":r["admission"],
            "task_accuracy":{
                task:{k:v["accuracy"] for k,v in r["heldout"][f"clean_{task}"]["by_context"].items()}
                for task in ("A","B")
            },
            "source_dose":r["dose_matching"],
            "sham_candidates":r["sham_matching"].get("qualifying",0),
            "projector_overlap_by_pass":r["discovery"].get("subspace_overlap_by_pass"),
            "shared_transfer_excess":r["primary"].get("shared_excess_nll_nats"),
            "private_transfer_excess":r["primary"].get("private_excess_nll_nats"),
            "specificity":r["primary"].get("specificity_difference_in_differences_nats"),
            "shared_sham_advantage":r["primary"].get("shared_sham_advantage_nats"),
        })
    end={
        "study":"BPD-NC-09","phase":"ENGINEERING",
        "scientific_status":"NOT_CONFIRMATORY",
        "evidence_class":"REGISTERED_SYNTHETIC_WITHIN_MODEL_CROSS_TASK_TRANSFER",
        "protocol_commit":PROTOCOL_COMMIT,
        "source_sha256":rows[0]["source_sha256"],
        "core_sha256":rows[0]["core_sha256"],
        "protocol_sha256":rows[0]["protocol_sha256"],
        "independent_model_roots":len(ROOTS),
        "eligible_seeds":[r["seed"] for r in eligible],
        "excluded_seeds":[r["seed"] for r in rows if r not in eligible],
        "model_root_rows":root_rows,
        "multiplicity_family":COMPARISONS,
        "bootstrap_draws":BOOTSTRAPS,
        "familywise_interval_coverage":1-FAMILY_ALPHA,
    }
    effects={
        "shared_transfer_excess":("shared_excess_nll_nats",909009),
        "specificity_private_minus_shared":("specificity_difference_in_differences_nats",909010),
        "shared_matched_sham_advantage":("shared_sham_advantage_nats",909011),
    }
    end["causal_endpoints"]={}
    for key,(field,seed) in effects.items():
        sample=[r["primary"][field] for r in eligible if r["primary"].get(field) is not None]
        end["causal_endpoints"][key]=interval_root(sample,seed)
    c=end["causal_endpoints"]
    all_roots=len(eligible)==len(ROOTS)
    gates={
        "all_eight_roots_admitted":all_roots,
        "shared_transfer_excess_upper_le_0_05":(
            all_roots and c["shared_transfer_excess"]["familywise_interval"] is not None
            and c["shared_transfer_excess"]["familywise_interval"][1]<=.05
        ),
        "specificity_lower_gt_0_02":(
            all_roots and c["specificity_private_minus_shared"]["familywise_interval"] is not None
            and c["specificity_private_minus_shared"]["familywise_interval"][0]>.02
        ),
        "shared_sham_advantage_lower_gt_0_02":(
            all_roots and c["shared_matched_sham_advantage"]["familywise_interval"] is not None
            and c["shared_matched_sham_advantage"]["familywise_interval"][0]>.02
        ),
    }
    end["preregistered_gates"]=gates
    end["development_resolution"]=(
        "ENGINEERING_CROSS_TASK_TRANSFER" if all(gates.values())
        else "ENGINEERING_NULL_OR_INELIGIBLE"
    )
    end["confirmation"]="SEALED; requires independent sham calibration, preflight and new roots"
    return end


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    docs=[json.loads((args.directory/f"nc09_root_{n}.json").read_text())
          for n in ROOTS]
    result=aggregate(docs)
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("NC09",result["development_resolution"],
          "eligible",len(result["eligible_seeds"]),"/8")
    for name,score in result["causal_endpoints"].items():
        print(name,score)
    print("admission",result["preregistered_gates"])


if __name__=="__main__":
    main()
