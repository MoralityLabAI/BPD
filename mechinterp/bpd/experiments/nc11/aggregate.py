"""NC-11 original paired-model root-level scientific decision. No confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .experiment import (
    DEVELOPMENT, PROTOCOL_COMMIT, STEPS, DISCOVERY, CALIBRATION_POOL,
    TARGET_CALIBRATION, DOSE_POOL, EVALUATION, PROPOSALS, SHAMS
)

PRIMARY = (
    "jacobian_vs_activation_shared",
    "jacobian_vs_permuted_shared",
    "jacobian_minus_target_cal_shared",
    "matched_sham_advantage_shared",
)
BOOTSTRAPS=4096
FAMILY=len(PRIMARY)
ALPHA=.05/FAMILY


def root_bootstrap(effects:list[float],seed:int)->dict:
    if not effects:
        return {"n_independent_root_pairs":0,"mean_nats":None,
                "familywise_9875pct_ci":None,"status":"UNSCORED"}
    x=np.asarray(effects,dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite paired-root results")
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(x),size=(BOOTSTRAPS,len(x)))
    means=x[indices].mean(axis=1)
    lo,hi=np.quantile(means,[ALPHA/2,1.-ALPHA/2])
    return {"n_independent_root_pairs":len(x),
            "mean_nats":float(x.mean()),
            "familywise_9875pct_ci":[float(lo),float(hi)],
            "effects_by_root":[float(v) for v in x],
            "family_coverage":1-ALPHA,
            "evidence_class":"EIGHT_ROOT_ENGINEERING_ONLY"}


def aggregate(receipts:list[dict])->dict:
    if len(receipts)!=8 or sorted(r["root"] for r in receipts)!=list(DEVELOPMENT):
        raise ValueError("NC-11 requires precisely independent pair roots 1100..1107")
    budget={"steps":STEPS,"train_A":2048,"train_B":2048,
            "discovery":DISCOVERY,"calibration_pool":CALIBRATION_POOL,
            "target_gradient_calibration":TARGET_CALIBRATION,
            "dose":DOSE_POOL,"evaluation":EVALUATION,
            "sham_proposals":PROPOSALS}
    for r in receipts:
        if (r.get("study")!="BPD-NC-11" or r.get("phase")!="ENGINEERING"
            or r.get("protocol_commit")!=PROTOCOL_COMMIT
            or r.get("budgets")!=budget):
            raise ValueError("source-registered population or split mismatch")
    if len({r["source_hash_sha256"] for r in receipts})!=1:
        raise ValueError("mixed experiment source SHA-256")
    if len({r["protocol_sha256"] for r in receipts})!=1:
        raise ValueError("mixed frozen protocol SHA-256")
    ordered=sorted(receipts,key=lambda r:r["root"])
    admitted=[r for r in ordered if r.get("eligible")]
    report={
        "study":"BPD-NC-11","phase":"ENGINEERING",
        "evidence_class":"REGISTERED_SYNTHETIC_PAIRED_MODEL_DEVELOPMENT_ONLY",
        "scientific_status":"NOT_CONFIRMATORY",
        "protocol_commit":PROTOCOL_COMMIT,
        "experiment_source_sha256":ordered[0]["source_hash_sha256"],
        "protocol_sha256":ordered[0]["protocol_sha256"],
        "n_independent_model_pairs":8,"n_independently_trained_models":16,
        "roots":list(DEVELOPMENT),
        "technical_admission_roots":[r["root"] for r in admitted],
        "excluded_roots":[{"root":r["root"],"reasons":r["excluded_reasons"]}
                          for r in ordered if not r.get("eligible")],
        "calibration_target_gradient_queries_per_root":TARGET_CALIBRATION,
        "planned_directional_comparisons":FAMILY,
        "familywise_cell_confidence":1-ALPHA,
        "bootstrap_root_draws":BOOTSTRAPS,
        "root_diagnostics":[],
        "contrasts":{},"gates":{},
    }
    for r in ordered:
        ev=r.get("evaluation",{})
        shared={}
        for name in ("B_clean","jacobian","activation","permuted_jacobian",
                     "target_calibration","target_full_gradient_reference","identity"):
            if name in ev:
                shared[name]=ev[name]["by_context"]["shared"]["final_nll_nats"]
        report["root_diagnostics"].append({
            "root":r["root"],"eligible":bool(r.get("eligible")),
            "exclusion_reasons":r["excluded_reasons"],
            "positive_calibration_count":r.get("calibration",{}).get("positive_context_examples_in_pool"),
            "selected_target_calibration":r.get("calibration",{}).get("calibration_used"),
            "sham_candidates":r.get("shams",{}).get("qualified"),
            "sham_count":r.get("shams",{}).get("selected"),
            "jacobian_fit_residual":r.get("calibration",{}).get("jacobian_procrustes",{}).get("relative_alignment_residual_by_pass"),
            "activation_fit_residual":r.get("calibration",{}).get("activation_procrustes",{}).get("relative_alignment_residual_by_pass"),
            "within_B_gauge":r.get("gauge"),
            "external_shared_nll":shared,
            "contrasts":r.get("comparisons",{}),
        })
    for index,name in enumerate(PRIMARY):
        values=[r["comparisons"][name] for r in admitted]
        v=root_bootstrap(values,seed=1_110_011+index*101)
        v["fully_admitted"]=len(admitted)==8
        report["contrasts"][name]=v
    good=len(admitted)==8
    ci={k:report["contrasts"][k]["familywise_9875pct_ci"] for k in PRIMARY}
    report["gates"]={
        "all_eight_root_pairs_technically_admitted":good,
        "jacobian_better_than_activation":good and ci[PRIMARY[0]] is not None and ci[PRIMARY[0]][0]>.01,
        "jacobian_better_than_permuted_jacobian":good and ci[PRIMARY[1]] is not None and ci[PRIMARY[1]][0]>.01,
        "jacobian_noninferior_to_same_128_B_gradient_baseline":good and ci[PRIMARY[2]] is not None and ci[PRIMARY[2]][1]<=.05,
        "jacobian_better_than_local_physical_shams":good and ci[PRIMARY[3]] is not None and ci[PRIMARY[3]][0]>.02,
    }
    report["development_verdict"]=(
        "ENGINEERING_JACOBIAN_TRANSFER" if all(report["gates"].values())
        else "ENGINEERING_NULL_OR_INELIGIBLE"
    )
    report["confirmation"]="SEALED; new sham-root calibration + independent EvalMath preflight required"
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    docs=[json.loads((args.directory/f"nc11_root_{root}.json").read_text())
          for root in DEVELOPMENT]
    result=aggregate(docs)
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("NC11",result["development_verdict"],
          "technical roots",len(result["technical_admission_roots"]))
    for k,v in result["contrasts"].items():
        print(k,v["mean_nats"],v["familywise_9875pct_ci"])
    print("gates",result["gates"])


if __name__=="__main__":
    main()
