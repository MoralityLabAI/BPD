"""Independent original 16-root FC04 source geometry and task receipt aggregator.

The model-root cohort is the statistical unit. Full tangent chart starts,
recurrent passes, local shams and task examples remain nested observations.
Scientific confirmation and 59-root recalibration remain sealed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_fc04.experiment import (
    ROOTS,ROOT,PROTO_SHA,PARENT_SOURCE_SHA,
)
from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_fc04.solver import (
    CHART_STARTS,MAX_UPDATES,ANGLE,DP12_MIN_CHORDAL,
    PAIRWISE_MIN_CHORDAL,
)

BOOTSTRAP_SEED=15_041_504
BOOTSTRAP_DRAWS=4096
REQUIRED_SHAMS=8


def decoded_projector(receipt:dict):
    binary=base64.b64decode(receipt["projector_b64_le_float64"],validate=True)
    if hashlib.sha256(binary).hexdigest()!=receipt["projector_sha256"]:
        raise AssertionError("Original physical projector SHA256 digest mismatch")
    if len(binary)!=4*20*20*8:
        raise AssertionError("Expected little-endian float64 4x20x20 matrices")
    p=np.frombuffer(binary,dtype="<f8").reshape((4,20,20))
    if not np.isfinite(p).all():
        raise AssertionError("Nonfinite physical projector in root archive")
    return p


def distance(a,b):
    return float(np.sqrt(np.square(a-b).sum()/24.))


def independently_audit_full_tangent_projectors(root:dict):
    adm=root.get("sham_admission")
    if adm is None:
        return {"matched":0,"complete":False,"geometry":[]}
    if adm["charts_configured"]!=CHART_STARTS or adm["max_corrections_per_chart"]!=MAX_UPDATES:
        raise AssertionError("Prospective full-tangent source compute budget changed")
    if len(adm["charts"])!=CHART_STARTS:
        raise AssertionError("First source chart census truncated")
    if abs(adm["fixed_angle_radians"]-ANGLE)>1e-12 or (
        adm["minimum_DP12_chordal"]!=DP12_MIN_CHORDAL or
        adm["minimum_pairwise_chordal"]!=PAIRWISE_MIN_CHORDAL
    ):
        raise AssertionError("Source registered geometric minimum changed")
    if adm["uses_external_task_labels"]:
        raise AssertionError("External task leakage into source solver")
    if len(adm["projectors"])!=adm["selected"] or (
        len(root["sham_nlls"])!=adm["selected"]
    ):
        raise AssertionError("Source sham count differs from heldout scored outcomes")
    parent=root.get("source_dp12_projector_receipt")
    if parent is None:
        raise AssertionError("Parent original DP12 physical projector missing")
    P0=decoded_projector(parent)
    tuple_r=root["source_rank12_optimum"]["ranks"]
    if sum(tuple_r)!=12 or len(tuple_r)!=4:
        raise AssertionError("Registered source DP rank changed")
    accepted=[]
    diag=[]
    for sh in adm["projectors"]:
        p=decoded_projector(sh)
        symmetry=float(np.max(np.abs(p-p.transpose(0,2,1))))
        idempotence=float(np.max(np.abs(p@p-p)))
        if symmetry>1e-6 or idempotence>1e-6:
            raise AssertionError("Grassmann sham projector lost orthogonality")
        if any(abs(float(np.trace(p[t]))-tuple_r[t])>1e-6 for t in range(4)):
            raise AssertionError("Source physical passwise rank changed")
        dpdist=distance(p,P0)
        pair=min((distance(p,q) for q in accepted),default=None)
        if dpdist<DP12_MIN_CHORDAL-1e-8 or (
            pair is not None and pair<PAIRWISE_MIN_CHORDAL-1e-8
        ):
            raise AssertionError("Source nontrivial physical projector separation failed")
        if abs(dpdist-sh["distance_from_dp12"])>1e-8 or (
            pair is not None and abs(pair-sh["minimum_pairwise_distance"])>1e-8
        ):
            raise AssertionError("Original source projector separation receipt disagrees")
        if sh["physical_rms_relative_error"]>.15+1e-8 or (
            sh["signed_score_abs_error"]>adm["source_signed_score_abs_tolerance"]+1e-8
        ) or (
            sh["score_rms_abs_error"]>adm["source_score_rms_abs_tolerance"]+1e-8
        ):
            raise AssertionError("Sham violates original frozen FC02 source metric")
        accepted.append(p)
        diag.append({
            "chart_index":sh["chart_index"],
            "correction_updates":sh["correction_updates"],
            "dp12_distance":dpdist,
            "minimum_previous_distance":pair,
            "projector_sha256":sh["projector_sha256"],
            "max_symmetry_error":symmetry,
            "max_idempotence_error":idempotence
        })
    return {
        "matched":len(accepted),
        "complete":adm["eligible"] and len(accepted)==REQUIRED_SHAMS,
        "geometry":diag,
    }


def root_bootstrap(values):
    x=np.asarray(values,dtype=float)
    if x.shape!=(16,) or not np.isfinite(x).all():
        raise ValueError("16 independent finite model-root effects required")
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    resampled=rng.integers(0,16,size=(BOOTSTRAP_DRAWS,16))
    empirical=x[resampled].mean(axis=1)
    return {
        "mean":float(x.mean()),
        "lower95":float(np.percentile(empirical,2.5)),
        "upper95":float(np.percentile(empirical,97.5)),
        "root_sd":float(x.std(ddof=1)),
        "seed":BOOTSTRAP_SEED,
        "draws":BOOTSTRAP_DRAWS
    }


def aggregate(records:list[dict]):
    by={r["root"]:r for r in records}
    if len(records)!=16 or len(by)!=16 or set(by)!=set(ROOTS):
        raise ValueError("Exactly sixteen distinct prospective fresh FC04 roots required")
    per=[]
    for seed in ROOTS:
        d=by[seed]
        if d["study"]!="BPD-NC09-FC04" or (
            d["phase"]!="FULL_TANGENT_JACOBIAN_ENGINEERING_DEVELOPMENT"
        ) or d["protocol_sha256"]!=PROTO_SHA:
            raise AssertionError("Unexpected first-run source identity or protocol")
        if d["parent_integrity"]["nc09_original_experiment_sha256"]!=PARENT_SOURCE_SHA:
            raise AssertionError("Original first NC09 physical source mutated")
        a=independently_audit_full_tangent_projectors(d)
        fc03=d.get("fc03_original_eightdim_admission") or {}
        fc02=d.get("fc02_original_rejection_admission") or {}
        if fc03.get("charts_configured")!=32 or (
            fc03.get("max_adam_steps_per_chart")!=64
        ) or fc02.get("proposals")!=4096:
            raise AssertionError("Frozen source competitor search budgets mismatch")
        if len(d["fc03_original_eightdim_sham_nlls"])!=fc03["selected"] or (
            len(d["fc02_original_rejection_sham_nlls"])!=fc02["selected"]
        ):
            raise AssertionError("Original source comparator task outputs incomplete")
        mandatory=("dp12","flat3","flat3_dose","flat4")
        clean=bool(d["full_mask_closure"]["passed"] and
                   d["source_gain_eligible"] and
                   all(d["task_nll_by_arm"][k] is not None for k in mandatory) and
                   all(d["native_rotated_gauge"][k]["physical_map_relative_error"]<=1e-4
                       and d["native_rotated_gauge"][k]["max_complete_rollout_logit_error"]<=5e-5
                       for k in mandatory))
        full=bool(clean and d["eligible"] and a["complete"])
        dp=float(d["task_nll_by_arm"]["dp12"]["nll_nats"]) if clean else None
        row={
            "root":seed,
            "clean_native_eligible":clean,
            "full_registered_eligible":full,
            "solver_shams":a["matched"],
            "FC03_eight_coordinate_shams":fc03["selected"],
            "FC02_random_shams":fc02["selected"],
            "FC03_full_admission":fc03["selected"]==8,
            "FC02_full_admission":fc02["selected"]==8,
            "rank_tuple":d["source_rank12_optimum"]["ranks"],
            "full_tangent_chart_count":d["sham_admission"]["charts_actually_started"],
            "full_jacobian_corrections":d["sham_admission"]["total_jacobian_corrections"],
            "DP12_nll":dp,
            "A_flat3_minus_DP12":float(d["task_nll_by_arm"]["flat3"]["nll_nats"]-dp)
                if clean else None,
            "D_doseflat3_minus_DP12":float(d["task_nll_by_arm"]["flat3_dose"]["nll_nats"]-dp)
                if clean else None,
            "full_tangent_mean_sham_minus_DP12":float(np.mean(d["sham_nlls"])-dp)
                if full else None,
            "FC03_mean_sham_minus_DP12":float(np.mean(d["fc03_original_eightdim_sham_nlls"])-dp)
                if clean and fc03["selected"]==8 else None,
            "FC02_mean_sham_minus_DP12":float(np.mean(d["fc02_original_rejection_sham_nlls"])-dp)
                if clean and fc02["selected"]==8 else None,
            "geometry":a["geometry"],
            "original_source_sha256":d["source_sha256"],
        }
        per.append(row)
    solver=sum(x["full_registered_eligible"] for x in per)
    fc03=sum(x["FC03_full_admission"] for x in per)
    fc02=sum(x["FC02_full_admission"] for x in per)
    geometry=all(len(x["geometry"])==8 for x in per)
    alltask=all(x["full_registered_eligible"] and
                x["full_tangent_mean_sham_minus_DP12"] is not None for x in per)
    gates={
        "sixteen_independent_full_source_and_eight_tangent_shams":solver==16,
        "two_more_full_model_root_sham_admissions_than_FC03":solver-fc03>=2 and fc03<=14,
        "independent_numerical_projector_geometry_and_source_constraints":geometry,
        "all_sixteen_native_four_pass_future_task_outcomes":alltask,
    }
    bootstrap=(root_bootstrap([x["full_tangent_mean_sham_minus_DP12"] for x in per])
               if solver==16 else None)
    return {
        "study":"BPD-NC09-FC04",
        "phase":"FULL_TANGENT_JACOBIAN_ENGINEERING_DEVELOPMENT",
        "evidence_class":"ENGINEERING_ONLY",
        "protocol_sha256":PROTO_SHA,
        "model_roots":16,
        "full_tangent_admitted_roots":solver,
        "FC03_old_eightdim_solver_admitted_roots":fc03,
        "FC02_original_random_sampler_admitted_roots":fc02,
        "gates":gates,
        "first_source_locked_development_verdict":(
            "FULL_TANGENT_ENGINEERING_POSITIVE" if all(gates.values())
            else "ENGINEERING_NULL_OR_INELIGIBLE"
        ),
        "solver_sham_effect_descriptive_bootstrap":bootstrap,
        "protected_59_root_recalibration_evaluated":0,
        "protected_32_root_confirmation_evaluated":0,
        "original_FC01_hold":"CALIBRATION_REDESIGN_OR_PILOT_ONLY",
        "original_FC02_hold":"CALIBRATION_INCOMPLETE",
        "original_FC03_verdict":"ENGINEERING_NULL_OR_INELIGIBLE",
        "per_root":per,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    docs=[
        json.loads((args.root_dir/f"nc09_fc04_root_{root}.json").read_text())
        for root in ROOTS
    ]
    summary=aggregate(docs)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "first_registered_verdict":summary["first_source_locked_development_verdict"],
        "solver_admitted_roots":summary["full_tangent_admitted_roots"],
        "FC03_admitted_roots":summary["FC03_old_eightdim_solver_admitted_roots"],
        "FC02_admitted_roots":summary["FC02_original_random_sampler_admitted_roots"],
        "gates":summary["gates"],
        "scientific_confirmation_sealed":True,
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
