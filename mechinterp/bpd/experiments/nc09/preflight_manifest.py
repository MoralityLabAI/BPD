"""Build strict EvalMath preflight input from all eight NC-09 engineering roots.

Intentionally cannot access any confirmation population. The independent
unit is one trained model root and sham scores are root-specific contrasts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import ROOTS, SHAMS

EVALMATH_COMMIT = "6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a"


def build_inputs(receipts: list[dict]) -> dict:
    if len(receipts)!=8 or sorted(x["seed"] for x in receipts)!=list(ROOTS):
        raise ValueError("strict preflight needs source-registered model roots 900..907")
    accepted=[r for r in receipts if r["eligible"]]
    output={
        "study":"BPD-NC-09",
        "evalmath_pinned_commit":EVALMATH_COMMIT,
        "n_independent_development_roots":len(receipts),
        "n_eligible_roots":len(accepted),
        "eligible_root_ids":[r["seed"] for r in accepted],
        "confirmation":"SEALED",
        "strict_preflight":"NOT_RUN",
        "scientific_status":"ENGINEERING_ONLY",
    }
    if len(accepted)<8:
        output["status"]="HOLD_INSUFFICIENT_ELIGIBLE_DEVELOPMENT_ROOTS"
        output["reasons"]=["all_eight_source_admission_required",
                           "independent_95pct_sham_tolerance_uncalibrated"]
        return output
    def rational(v):
        if abs(float(v)) > 1:
            raise ValueError("EvalMath preflight finite-ratio input outside [-1,1]")
        return format(float(v),".12f")
    pilot=[]
    try:
        for root in sorted(accepted,key=lambda x:x["seed"]):
            s=root["heldout"]["B_to_B"]["by_context"]
            effects=[]
            for sham in root["heldout"]["A_shams_to_B"]:
                v=sham["by_context"]
                null_shared=v["shared"]["nll_nats"]-s["shared"]["nll_nats"]
                null_private=v["private"]["nll_nats"]-s["private"]["nll_nats"]
                effects.append(rational(null_private-null_shared))
            if len(effects)!=SHAMS:
                raise ValueError("one eligible model root lacks eight original shams")
            pilot.append({
                "unit_id":f"trained_model_root_{root['seed']}",
                "paired_effect":rational(root["primary"]["specificity_difference_in_differences_nats"]),
                "sham_effects":effects,
            })
    except ValueError as ex:
        output["status"]="HOLD_EVALMATH_INPUT_RANGE_OR_SHAMS"
        output["reasons"]=[str(ex),"independent_95pct_sham_tolerance_uncalibrated"]
        return output
    output["status"]="PILOT_INPUT_AVAILABLE_STRICT_PREFLIGHT_REQUIRED"
    output["spec"]={
        "schema_version":1,
        "experiment_id":"BPD-NC-09_cross_task_specificity_followup_planning",
        "independent_unit":"fresh independently trained multitask recurrent model root",
        "alpha":"0.05",
        "scientific_min_effect":"0.02",
        "target_effect":"0.08",
        "measurement_resolution":"0.005",
        "sham_quantile":"0.95",
        "sham_confidence":"0.95",
        "planned_comparisons":3,
        "confirm_units_planned":64,
        "pilot_from_development_only":True,
        "heldout_sealed":True,
        "units_independent":True,
        "pilot":pilot,
    }
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    receipts=[json.loads((a.directory/f"nc09_root_{seed}.json").read_text())
              for seed in ROOTS]
    data=build_inputs(receipts)
    with a.out.open("x",encoding="utf-8") as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("EvalMath strict planning input",data["status"],
          "admitted",data["n_eligible_roots"],"/8")
    print("External confirmation remains",data["confirmation"])


if __name__=="__main__":
    main()
