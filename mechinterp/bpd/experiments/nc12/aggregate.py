"""NC-12: 16-family low-target-gradient source-transfer paired-root audit.

The 8 model pairs and all 4 nested target-gradient budgets are development
only. A scientifically negative or ineligible result must serialize cleanly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .experiment import (
    ROOTS, BUDGETS, PROTOCOL_COMMIT, TRAIN_STEPS,
    DISCOVERY, CAL_POOL, ACTIVATION_CAL, MAX_B_GRAD,
    DOSE_SIZE, EVAL_SIZE, PROPOSALS, SHAMS,
)

ENDPOINTS=(
    "target_only_minus_transfer",
    "activation_minus_transfer",
    "median_sham_minus_transfer",
    "transfer_minus_clean_B",
)
MULTIPLICITY=len(BUDGETS)*len(ENDPOINTS)
BOOTSTRAPS=4096
ALPHA=.05/MULTIPLICITY


def paired_root_interval(values:list[float],seed:int)->dict:
    if not values:
        return {"n_independent_model_pairs":0, "mean_nats":None,
                "familywise_ci":None, "status":"UNSCORED"}
    data=np.asarray(values,dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("nonfinite root-level transfer results")
    g=np.random.default_rng(seed)
    indexes=g.integers(0,len(data),size=(BOOTSTRAPS,len(data)))
    means=data[indexes].mean(axis=1)
    lo,hi=np.quantile(means,[ALPHA/2,1-ALPHA/2])
    return {
        "n_independent_model_pairs":len(data),
        "mean_nats":float(data.mean()),
        "familywise_ci":[float(lo),float(hi)],
        "confidence":1-ALPHA,
        "root_effects":[float(z) for z in data],
        "scientific_status":"ENGINEERING_ONLY",
    }


def aggregate(receipts:list[dict])->dict:
    if len(receipts)!=8 or sorted(r["root"] for r in receipts)!=list(ROOTS):
        raise ValueError("NC-12 requires exactly independent model-pair roots 1200..1207")
    expected={
        "steps":TRAIN_STEPS,"train_A":2048,"train_B":2048,
        "discovery":DISCOVERY,"calibration_pool":CAL_POOL,
        "activation_calibration":ACTIVATION_CAL,
        "target_gradient_union":MAX_B_GRAD,
        "dose":DOSE_SIZE,"evaluation":EVAL_SIZE,
        "sham_proposals":PROPOSALS,
    }
    for r in receipts:
        if (r.get("study")!="BPD-NC-12" or r.get("phase")!="ENGINEERING"
            or r.get("protocol_commit")!=PROTOCOL_COMMIT
            or r.get("budgets")!=expected):
            raise ValueError("source-registered dataset or model contract mismatch")
    if len({r["source_sha256"] for r in receipts})!=1:
        raise ValueError("mixed NC12 experiment source SHA")
    if len({r["protocol_sha256"] for r in receipts})!=1:
        raise ValueError("mixed registered protocol SHA")
    ordered=sorted(receipts,key=lambda r:r["root"])
    result={
        "study":"BPD-NC-12","phase":"ENGINEERING",
        "scientific_status":"NOT_CONFIRMATORY",
        "evidence_class":"SYNTHETIC_NESTED_LOW_QUERY_ENGINEERING",
        "protocol_commit":PROTOCOL_COMMIT,
        "source_sha256":ordered[0]["source_sha256"],
        "protocol_sha256":ordered[0]["protocol_sha256"],
        "independent_training_pairs":len(ordered),
        "model_initializations":2*len(ordered),
        "roots":list(ROOTS),
        "nested_target_gradient_budgets":list(BUDGETS),
        "maximum_primary_B_gradient_rows":MAX_B_GRAD,
        "paired_activation_information_rows":ACTIVATION_CAL,
        "family_directional_comparisons":MULTIPLICITY,
        "bonferroni_cell_confidence":1-ALPHA,
        "bootstrap_draws":BOOTSTRAPS,
        "arms":{},
        "comparison_rules":{},
    }
    for q in BUDGETS:
        records=[(r,r.get("q_arms",{}).get(str(q),{})) for r in ordered]
        eligible=[(r,a) for r,a in records if a.get("technically_admitted") is True]
        missing=[
            {"root":r["root"],
             "reasons":sorted(set(
                 r.get("excluded_global",[])+a.get("excluded_reasons",[])
                 +([] if a.get("technically_admitted") is False else ["incomplete_record"])
             ))}
            for r,a in records if a.get("technically_admitted") is not True
        ]
        node={
            "q":q,
            "admitted_roots":[r["root"] for r,_ in eligible],
            "excluded_roots":missing,
            "root_sham_candidate_count":{
                str(r["root"]):a.get("sham_matching",{}).get("qualified")
                for r,a in records
            },
            "root_model_context_accuracies":{
                str(r["root"]):r.get("heldout",{}).get("clean_context_accuracies")
                for r,_ in records
            },
            "contrasts":{},
        }
        for index,name in enumerate(ENDPOINTS):
            vals=[a["comparisons"][name] for _,a in eligible]
            node["contrasts"][name]=paired_root_interval(
                vals,seed=1_212_012+q*31+index*101
            )
        # Conditioned-on-eligible diagnostics are descriptive even when root
        # eligibility fails. The actual gate below is all-eight fail-closed.
        node["mean_shared_nll"]=({
            arm:float(np.mean([
                a["observed"][arm]["by_context"]["shared"]["final_nll_nats"]
                for _,a in eligible
            ])) if eligible else None
            for arm in ("transfer","target_only","activation","permuted")
        })
        if eligible:
            node["mean_shared_clean_B_nll"]=float(np.mean([
                r["heldout"]["B_clean"]["by_context"]["shared"]["final_nll_nats"]
                for r,_ in eligible
            ]))
        else:
            node["mean_shared_clean_B_nll"]=None
        fully=len(eligible)==len(ROOTS)
        target_ci=node["contrasts"]["target_only_minus_transfer"]["familywise_ci"]
        act_ci=node["contrasts"]["activation_minus_transfer"]["familywise_ci"]
        sham_ci=node["contrasts"]["median_sham_minus_transfer"]["familywise_ci"]
        fid_ci=node["contrasts"]["transfer_minus_clean_B"]["familywise_ci"]
        gates={
            "all_eight_pairs_eligible":fully,
            "source_beats_equal_gradient_budget_B_only":fully and target_ci is not None and target_ci[0]>.01,
            "source_Jacobian_beats_activation_only":fully and act_ci is not None and act_ci[0]>.01,
            "physical_sham_specificity":fully and sham_ci is not None and sham_ci[0]>.02,
            "absolute_B_true_label_fidelity":fully and fid_ci is not None and fid_ci[1]<=.05,
        }
        node["gates"]=gates
        node["eligible_for_scientific_development_claim"]=bool(all(gates.values()))
        result["arms"][str(q)]=node
    lowest=next((
        q for q in (4,8)
        if result["arms"][str(q)]["eligible_for_scientific_development_claim"]
    ),None)
    result["lowest_qualifying_low_budget"]=lowest
    result["original_development_verdict"]=(
        "ENGINEERING_LOW_BUDGET_TRANSFER" if lowest is not None
        else "ENGINEERING_NULL_OR_INELIGIBLE"
    )
    result["confirmation"]="SEALED; fresh >=59 sham-calibration roots and independent EvalMath preflight required"
    return result


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    receipts=[json.loads((args.directory/f"nc12_root_{root}.json").read_text())
              for root in ROOTS]
    result=aggregate(receipts)
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("NC12 original",result["original_development_verdict"],
          "smallest low-q",result["lowest_qualifying_low_budget"])
    for q,arm in result["arms"].items():
        print("q",q,"eligible",len(arm["admitted_roots"]),
              "gates",arm["gates"])
        for name,contrast in arm["contrasts"].items():
            print("  ",name,contrast["mean_nats"],contrast["familywise_ci"])


if __name__=="__main__":
    main()
