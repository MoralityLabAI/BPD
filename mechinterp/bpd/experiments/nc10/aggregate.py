"""Independent paired-model NC-10 audit; cluster-bootstrap whole paired roots only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .experiment import DEVELOPMENT, PROTOCOL_COMMIT, STEPS, DISCOVERY, ALIGNMENT, EVALUATION, SHAMS, PROPOSALS

COMPARISON_NAMES = (
    "shared_transfer_deficit_vs_B_owned",
    "branch_specificity",
    "paired_alignment_gain_shared",
    "matched_sham_advantage_shared",
)
BOOTS = 4096
ALPHA = .05 / len(COMPARISON_NAMES)


def bootstrap(effects: list[float], seed: int) -> dict:
    if not effects:
        return {"n_independent_pairs":0, "mean_nats":None, "familywise_ci":None,
                "status":"UNSCORED_MISSING_ROOTS"}
    values = np.array(effects,dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite pair effect")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0,len(values),size=(BOOTS,len(values)))
    means = values[draws].mean(1)
    ci = np.quantile(means,[ALPHA/2,1-ALPHA/2])
    return {"n_independent_pairs":len(values),
            "mean_nats":float(values.mean()),
            "familywise_ci":[float(x) for x in ci],
            "individual_root_effects":[float(v) for v in values],
            "familywise_coverage":1-ALPHA,
            "evidence_class":"ENGINEERING_ONLY"}


def aggregate(receipts: list[dict]) -> dict:
    if len(receipts) != 8 or sorted(r["root"] for r in receipts) != list(DEVELOPMENT):
        raise ValueError("NC-10 aggregate requires exactly eight distinct roots 1000..1007")
    expected={"train_A":2048,"train_B":2048,"steps":STEPS,
              "discovery":DISCOVERY,"alignment":ALIGNMENT,
              "evaluation":EVALUATION,"sham_proposals":PROPOSALS}
    for r in receipts:
        if (r.get("study")!="BPD-NC-10" or r.get("phase")!="ENGINEERING"
            or r.get("protocol_commit")!=PROTOCOL_COMMIT
            or r.get("budgets")!=expected
            or r.get("splits")!={"A_train":1,"B_train":11,"discovery":2,"alignment":4,"evaluation":3}):
            raise ValueError("root does not match frozen source/data contract")
    if len({r["source_sha256"] for r in receipts})!=1:
        raise ValueError("mixed experiment source hashes")
    if len({r["protocol_sha256"] for r in receipts})!=1:
        raise ValueError("mixed frozen protocol hashes")
    ordered=sorted(receipts,key=lambda r:r["root"])
    eligible=[r for r in ordered if r["eligible"]]
    summary={
        "study":"BPD-NC-10","phase":"ENGINEERING",
        "scientific_status":"NOT_CONFIRMATORY",
        "evidence_class":"PROSPECTIVELY_REGISTERED_SYNTHETIC_CROSS_MODEL_DEVELOPMENT",
        "n_independent_paired_training_roots":8,
        "root_ids":list(DEVELOPMENT),
        "training_models":16,
        "protocol_commit":PROTOCOL_COMMIT,
        "source_sha256":ordered[0]["source_sha256"],
        "protocol_sha256":ordered[0]["protocol_sha256"],
        "admitted_root_ids":[r["root"] for r in eligible],
        "rejected_roots":[{"root":r["root"],"reasons":r["eligibility_reasons"]}
                          for r in ordered if not r["eligible"]],
        "planned_directional_comparisons":len(COMPARISON_NAMES),
        "familywise_cell_confidence":1-ALPHA,
        "bootstrap_resamples":BOOTS,
        "rootwise_diagnostics":[],
        "contrasts":{},
    }
    for r in ordered:
        e=r.get("evaluations",{})
        summary["rootwise_diagnostics"].append({
            "root":r["root"],"eligible":r["eligible"],
            "alignment_positive_cases":r.get("alignment",{}).get("positive_context_count"),
            "sham_candidates":r.get("sham_matching",{}).get("qualifying_candidates"),
            "alignment_residual_by_pass":r.get("alignment",{}).get("paired_procrustes",{}).get("relative_alignment_residual_by_pass"),
            "A_own_final_kl":e.get("A_owned",{}).get("kl_by_pass_nats",[None])[-1],
            "B_own_final_kl":e.get("B_owned",{}).get("kl_by_pass_nats",[None])[-1],
            "B_transfer_final_kl":e.get("A_transferred_to_B",{}).get("kl_by_pass_nats",[None])[-1],
            "contrasts":r["contrasts"],
        })
    for i,name in enumerate(COMPARISON_NAMES):
        effects=[r["contrasts"][name] for r in eligible]
        # Comparisons are rootwise only. A missing root blocks scientific gates,
        # and descriptive effects use the actually admitted root count.
        summary["contrasts"][name]=bootstrap(effects,seed=909_009+i*101)
        summary["contrasts"][name]["all_registered_roots_admitted"]=len(eligible)==8
    all_eight=len(eligible)==8
    shared=summary["contrasts"][COMPARISON_NAMES[0]]["familywise_ci"]
    specificity=summary["contrasts"][COMPARISON_NAMES[1]]["familywise_ci"]
    perm=summary["contrasts"][COMPARISON_NAMES[2]]["familywise_ci"]
    sham=summary["contrasts"][COMPARISON_NAMES[3]]["familywise_ci"]
    gates={
        "all_eight_technical_roots":all_eight,
        "shared_branch_noninferiority":all_eight and shared is not None and shared[1]<=.05,
        "task_specific_branch_difference":all_eight and specificity is not None and specificity[0]>.01,
        "paired_alignment_vs_permuted":all_eight and perm is not None and perm[0]>.01,
        "matched_physical_sham_advantage":all_eight and sham is not None and sham[0]>.02,
    }
    summary["gates"]=gates
    summary["development_verdict"]=(
        "ENGINEERING_CROSS_TASK_TRANSFER" if all(gates.values())
        else "ENGINEERING_NULL_OR_INELIGIBLE"
    )
    summary["confirmation"]="SEALED; independent 59-root sham calibration and EvalMath power study required"
    return summary


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    original=[json.loads((args.directory/f"nc10_root_{root}.json").read_text())
              for root in DEVELOPMENT]
    receipt=aggregate(original)
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(receipt,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("NC-10",receipt["development_verdict"],"technical roots",len(receipt["admitted_root_ids"]))
    for n,d in receipt["contrasts"].items():
        print(n,d["mean_nats"],d["familywise_ci"])
    print("gates",receipt["gates"])

if __name__=="__main__":
    main()
