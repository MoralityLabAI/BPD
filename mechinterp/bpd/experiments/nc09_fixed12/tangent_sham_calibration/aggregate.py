"""FC02: root-mean sham-only calibration on 59 independent model roots.

The 256x4096 nested bootstrap estimates a 32-root mean null planning
threshold; it is NOT a distribution-free 95/95 theorem or an exact p-value.
The old rootwise 59-maximum 95/95 bound remains a separate diagnostic.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from mechinterp.bpd.experiments.nc09_fixed12.tangent_sham_calibration.experiment import (
    ROOT, ROOTS, PROTO_SHA,
)

BOOT_SEED=13_131_313
BOOT_DRAWS=4096
AGGREGATE_SEED=18_180_018
OUTER_DRAWS=256
INNER_DRAWS=4096
FUTURE_MODEL_ROOTS=32
SHAMS=8
METRICS=(
    "A_flat3_minus_dp12","D_dose_flat3_minus_dp12",
    "S_mean_sham_minus_dp12","C_dp12_minus_clean",
    "F_dp12_minus_flat4",
)
EXACT_ROOTWISE_95_95=1-.95**59


def root_mean_bootstrap(matrix:np.ndarray):
    a=np.asarray(matrix,dtype=np.float64)
    if a.shape!=(59,5) or not np.isfinite(a).all():
        raise ValueError("Need 59 independent roots with five finite effects")
    rng=np.random.default_rng(BOOT_SEED)
    ids=rng.integers(0,59,size=(BOOT_DRAWS,59))
    means=a[ids].mean(axis=1)
    return {
        key:{
            "mean":float(a[:,i].mean()),
            "lower95":float(np.percentile(means[:,i],2.5)),
            "upper95":float(np.percentile(means[:,i],97.5)),
            "root_sd":float(a[:,i].std(ddof=1))
        } for i,key in enumerate(METRICS)
    }


def aggregate_root_mean_null(pseudo:np.ndarray):
    """Frozen nested bootstrap of *mean across 32 roots*, with root uncertainty."""
    x=np.asarray(pseudo,dtype=np.float64)
    if x.shape!=(59,8) or not np.isfinite(x).all():
        raise ValueError("Need 59 complete independent sham-outcome vectors")
    if np.max(np.abs(x.mean(axis=1)))>1e-10:
        raise AssertionError("Pseudo-selected null conditional expectation not zero")
    rng=np.random.default_rng(AGGREGATE_SEED)
    outer=[]
    for _ in range(OUTER_DRAWS):
        outer_roots=rng.integers(0,59,size=59)
        indices=rng.integers(0,59,size=(INNER_DRAWS,FUTURE_MODEL_ROOTS))
        sham_index=rng.integers(0,SHAMS,size=(INNER_DRAWS,FUTURE_MODEL_ROOTS))
        simulated=x[outer_roots[indices],sham_index].mean(axis=1)
        outer.append(float(np.percentile(simulated,95)))
    floor=float(np.percentile(outer,95))
    return {
        "target_estimand":"mean_pseudo_selected_sham_NLL_effect_across_32_independent_models",
        "method":"nested_model_root_pseudo_target_bootstrap_planning_only",
        "seed":AGGREGATE_SEED,
        "outer_resamples":OUTER_DRAWS,
        "inner_simulations_per_outer":INNER_DRAWS,
        "future_model_roots_per_simulated_mean":FUTURE_MODEL_ROOTS,
        "outer_95pct_tail_estimates":outer,
        "upper95_across_outer_95pct_estimates":floor,
        "mathematical_exact_coverage_claim":False,
    }


def approximate_power(effect:float,root_sd:float,mean_null_floor:float):
    delta=effect-mean_null_floor-.020
    from math import ceil,log
    sign_floor=max(2,math.ceil(math.log(3/.05,2)))
    z=NormalDist().inv_cdf(1-.05/3)+NormalDist().inv_cdf(.80)
    required=(max(sign_floor,math.ceil((z*root_sd/delta)**2))
              if delta>0 and root_sd>0 else None)
    return {
        "source_mean_sham_advantage":effect,
        "aggregate_mean_null_planning_threshold":mean_null_floor,
        "mandatory_incremental_mechanistic_margin":.020,
        "remaining_signal_above_floor":delta,
        "root_sample_sd":root_sd,
        "target_familywise_alpha":.05,
        "planned_comparisons":3,
        "exact_minimum_all_positive_root_sign_units":sign_floor,
        "normal_approx_80pct_power_roots_required":required,
        "planned_future_confirmation_roots":32,
        "engineering_power_gate_passed":bool(required is not None and required<=32),
        "limits":"Normal approximation and nested bootstrap are planning only; not a distribution-free scientific threshold."
    }


def aggregate(records:list[dict]):
    by={x["root"]:x for x in records}
    if len(records)!=59 or len(by)!=59 or set(by)!=set(ROOTS):
        raise ValueError("Original FC02 cohort must contain 59 unique registered model roots")
    per=[]
    for root in ROOTS:
        d=by[root]
        if (d["study"]!="BPD-NC09-FC02" or
            d["phase"]!="AGGREGATE_TANGENT_SHAM_CALIBRATION" or
            d["protocol_sha256"]!=PROTO_SHA):
            raise AssertionError("Source-registered FC02 study/phase mismatch")
        e=d.get("effects")
        s=d.get("sham_nlls",[])
        admit=d.get("sham_admission") or {}
        valid=bool(
            d["eligible"] and e is not None and
            len(s)==8 and admit.get("selected")==8 and
            admit.get("proposals")==4096 and
            all(isinstance(e.get(k),(int,float)) and math.isfinite(e[k])
                for k in METRICS) and
            len(e.get("Z_pseudo_null_by_sham",[]))==8 and
            all(math.isfinite(v) for v in e["Z_pseudo_null_by_sham"]) and
            all(math.isfinite(float(v)) for v in s)
        )
        if valid:
            center=float(np.median(s))
            pseudo=[float((sum(s)-v)/7-v) for v in s]
            if max(abs(a-b) for a,b in zip(pseudo,e["Z_pseudo_null_by_sham"]))>1e-10:
                raise AssertionError("Sham-only pseudo-target null construction drift")
            if abs(sum(pseudo))>1e-10:
                raise AssertionError("Pseudo-selected null did not center exactly")
            if abs(max(abs(v-center) for v in s)-e["T_null_worst_abs"])>1e-10:
                raise AssertionError("Rootwise worst sham orientation statistic changed")
            original_index=int(np.random.default_rng(18_180_000+root).integers(0,8))
            if e["pseudo_selected_sham_index"]!=original_index:
                raise AssertionError("Prespecified independently selected sham index changed")
            if abs(e["Z_preselected_pseudo_sham"]-pseudo[original_index])>1e-10:
                raise AssertionError("Selected pseudo-sham control differs")
            if abs(e["S_mean_sham_minus_dp12"]-
                   (float(np.mean(s))-d["task_nll_by_arm"]["dp12"]["nll_nats"]))>1e-10:
                raise AssertionError("Aggregate DP-versus-sham effect changed")
            if (not admit.get("uses_external_labels",True) is False or
                any(t>.15+1e-12 for t in admit["selected_dose_relative_errors"]) or
                any(t>admit["signed_score_abs_tolerance"]+1e-12
                    for t in admit["selected_score_mean_abs_errors"]) or
                any(t>admit["rms_score_abs_tolerance"]+1e-12
                    for t in admit["selected_score_rms_abs_errors"])):
                raise AssertionError("Original source-only three-way matching failed")
        per.append({
            "root":root,"eligible":valid,
            "rank_tuple":d["source_rank12_optimum"]["ranks"] if d.get("source_rank12_optimum") else None,
            "qualified_source_sham_proposals":admit.get("qualified",0),
            "selected_shams":len(s),
            "source_score_target_mean":admit.get("target_signed_score_mean"),
            "source_score_target_rms":admit.get("target_first_order_score_rms"),
            "effects":e,
            "original_source_sha256":d["source_sha256"],
        })
    eligible=all(x["eligible"] for x in per)
    result={
        "study":"BPD-NC09-FC02",
        "phase":"AGGREGATE_TANGENT_SHAM_CALIBRATION",
        "evidence_class":"CALIBRATION_DEVELOPMENT_ONLY",
        "protocol_sha256":PROTO_SHA,
        "independent_model_roots":59,
        "eligible_roots":sum(x["eligible"] for x in per),
        "rootwise_95_95_confidence":EXACT_ROOTWISE_95_95,
        "rootwise_sham_max_95_95_diagnostic":None,
        "root_mean_bootstrap":None,
        "aggregate_32_root_pseudo_null":None,
        "approximate_power":None,
        "gates":None,
        "registered_development_verdict":"CALIBRATION_INCOMPLETE",
        "original_FC01_verdict_unchanged":"CALIBRATION_REDESIGN_OR_PILOT_ONLY",
        "calibration_scientific_confirmation_ready":False,
        "scientific_confirmation_roots_evaluated":0,
        "other_protected_model_roots_evaluated":0,
        "per_root":per,
    }
    if not eligible:
        return result
    mat=np.array([[float(x["effects"][key]) for key in METRICS]
                  for x in per],dtype=float)
    z=np.array([x["effects"]["Z_pseudo_null_by_sham"] for x in per],dtype=float)
    boot=root_mean_bootstrap(mat)
    null=aggregate_root_mean_null(z)
    rootmax=max(float(x["effects"]["T_null_worst_abs"]) for x in per)
    p=approximate_power(
        boot["S_mean_sham_minus_dp12"]["mean"],
        boot["S_mean_sham_minus_dp12"]["root_sd"],
        null["upper95_across_outer_95pct_estimates"]
    )
    bmean=null["upper95_across_outer_95pct_estimates"]
    gates={
        "all_59_source_eligible_and_three_way_matched":True,
        "DP12_benefit_over_flat3_and_source_dose_match":bool(
            boot["A_flat3_minus_dp12"]["mean"]>=.008 and
            boot["A_flat3_minus_dp12"]["lower95"]>0 and
            boot["D_dose_flat3_minus_dp12"]["mean"]>=.006 and
            boot["D_dose_flat3_minus_dp12"]["lower95"]>0
        ),
        "absolute_and_rank4_task_fidelity_and_local_shams":bool(
            boot["C_dp12_minus_clean"]["mean"]<=.05 and
            boot["C_dp12_minus_clean"]["upper95"]<=.05 and
            boot["F_dp12_minus_flat4"]["mean"]<=.020 and
            boot["F_dp12_minus_flat4"]["upper95"]<=.020 and
            boot["S_mean_sham_minus_dp12"]["mean"]>=.020 and
            boot["S_mean_sham_minus_dp12"]["lower95"]>0
        ),
        "aggregate_null_exceeded_with_incremental_margin":bool(
            np.isfinite(bmean) and
            boot["S_mean_sham_minus_dp12"]["lower95"]>bmean+.020
        ),
        "future_32_root_normal_power_planning":p["engineering_power_gate_passed"],
    }
    result.update({
        "rootwise_sham_max_95_95_diagnostic":rootmax,
        "root_mean_bootstrap":boot,
        "aggregate_32_root_pseudo_null":null,
        "approximate_power":p,
        "gates":gates,
        "registered_development_verdict":(
            "AGGREGATE_TANGENT_NULL_DEVELOPMENT_POSITIVE"
            if all(gates.values()) else "AGGREGATE_TANGENT_NULL_PILOT_ONLY"
        )
    })
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    roots=[
        json.loads((args.root_dir/("nc09_fc02_root_%d.json"%root)).read_text())
        for root in ROOTS
    ]
    result=aggregate(roots)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "calibration_verdict":result["registered_development_verdict"],
        "eligible":result["eligible_roots"],
        "source_only_three_way_sham_null":result["rootwise_sham_max_95_95_diagnostic"],
        "aggregate_32_root_pseudo_sham_threshold":(
            result["aggregate_32_root_pseudo_null"] or {}
        ).get("upper95_across_outer_95pct_estimates"),
        "mean_root_effects":(
            {k:v["mean"] for k,v in result["root_mean_bootstrap"].items()}
            if result["root_mean_bootstrap"] else None),
        "gates":result["gates"],
        "confirmation_remains_sealed":True
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
