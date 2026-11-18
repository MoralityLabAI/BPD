"""First source-locked sixteen-root FC03 constrained Grassmann engineering audit.

Root-level inference is descriptive engineering only. Sham charts, optimizer
steps, 1024 task cases and individual sham NLLs are never independent roots.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from mechinterp.bpd.experiments.nc09_fixed12.grassmann_fc03.experiment import (
    ROOT, ROOTS, PROTO_SHA, PARENT_SOURCE_SHA
)
from mechinterp.bpd.experiments.nc09_fixed12.grassmann_fc03.solver import (
    ANGLE,CHART_STARTS,MAX_ADAM_STEPS,DP12_MIN_CHORDAL,
    PAIRWISE_MIN_CHORDAL
)

BOOTSTRAP_SEED=14_031_403
BOOTSTRAP_DRAWS=4096
REQUIRED_SHAMS=8


def matrix_from_receipt(raw:dict):
    data=base64.b64decode(raw["projector_b64_le_float64"],validate=True)
    assert hashlib.sha256(data).hexdigest()==raw["projector_sha256"]
    x=np.frombuffer(data,dtype="<f8").reshape((4,20,20))
    assert np.isfinite(x).all()
    return x


def chordal(a:np.ndarray,b:np.ndarray):
    return float(np.sqrt(np.square(a-b).sum()/24.))


def independent_source_audit(root:dict):
    adm=root["sham_admission"]
    baseline=root["fc02_original_rejection_admission"]
    if adm is None:
        return {"source_eligible":False,"reason":"missing solver receipt"}
    if adm["charts_configured"]!=CHART_STARTS or adm["max_adam_steps_per_chart"]!=MAX_ADAM_STEPS:
        raise AssertionError("Original source search budget changed")
    if abs(adm["chart_angle_radians"]-ANGLE)>1e-12:
        raise AssertionError("Different physical Grassmann angle")
    if not adm["uses_external_labels"]:
        pass
    else:
        raise AssertionError("Sham selector accessed external labels")
    if baseline is None or baseline["proposals"]!=4096:
        raise AssertionError("Original FC02 rejection baseline absent")
    if len(adm["charts"])!=CHART_STARTS:
        raise AssertionError("Original chart audit incomplete")
    if adm["charts_actually_attempted"]+adm["charts_skipped_after_full"]!=CHART_STARTS:
        raise AssertionError("Original chart enumeration incomplete")
    if adm["selected"]!=len(adm["accepted"]):
        raise AssertionError("Original source selected count mismatch")
    if adm["selected"]!=len(root.get("sham_nlls",[])):
        raise AssertionError("Selected projectors lack heldout task scores")
    dp_receipt=root.get("source_dp12_projector_receipt")
    if dp_receipt is None:
        raise AssertionError("DP12 projector audit receipt unavailable")
    dp=matrix_from_receipt(dp_receipt)
    ranks=root["source_rank12_optimum"]["ranks"] if root["source_rank12_optimum"] else None
    if ranks is None or len(ranks)!=4 or sum(ranks)!=12:
        raise AssertionError("Original DP12 rank tuple missing")
    accepted=[]
    geometry=[]
    for a in adm["accepted"]:
        P=matrix_from_receipt(a)
        if max(np.max(np.abs(P@P-P)),np.max(np.abs(P-P.transpose(0,2,1))))>1e-6:
            raise AssertionError("Sham projector idempotence or symmetry failed")
        if max(abs(np.trace(P[t])-ranks[t]) for t in range(4))>1e-5:
            raise AssertionError("Sham projector changed pass-specific rank")
        sep=chordal(P,dp)
        pair=min((chordal(P,q) for q in accepted),default=None)
        if sep<DP12_MIN_CHORDAL-1e-8:
            raise AssertionError("Original DP12 separation floor failed")
        if pair is not None and pair<PAIRWISE_MIN_CHORDAL-1e-8:
            raise AssertionError("Independent same-root sham separation failed")
        if abs(sep-a["distance_from_dp12"])>1e-8:
            raise AssertionError("Recorded DP12 separation disagrees")
        if pair is not None and abs(pair-a["minimum_pairwise_distance"])>1e-8:
            raise AssertionError("Recorded sham-pair distance disagrees")
        if a["physical_rms_relative_error"]>.15+1e-8:
            raise AssertionError("Original FC02 physical source RMS tolerance exceeded")
        if a["signed_score_abs_error"]>adm["source_signed_score_abs_tolerance"]+1e-8:
            raise AssertionError("Original FC02 signed derivative mean tolerance exceeded")
        if a["score_rms_abs_error"]>adm["source_score_rms_abs_tolerance"]+1e-8:
            raise AssertionError("Original FC02 model-score RMS tolerance exceeded")
        accepted.append(P)
        geometry.append({
            "chart_index":a["chart_index"],"optimization_step":a["optimization_step"],
            "distance_to_dp12":sep,"minimum_previous_sham_distance":pair,
            "symmetry_max":float(np.max(np.abs(P-P.transpose(0,2,1)))),
            "idempotence_max":float(np.max(np.abs(P@P-P))),
        })
    return {
        "source_eligible":bool(adm["admitted"] and len(accepted)==REQUIRED_SHAMS),
        "geometry":geometry,
        "source_qualified":adm["qualified"],
        "fc02_baseline_qualified":baseline["qualified"],
        "fc02_full_original_admission":bool(baseline["selected"]==REQUIRED_SHAMS),
    }


def root_bootstrap(values:list[float]):
    v=np.asarray(values,dtype=float)
    if len(v)!=16 or not np.isfinite(v).all():
        raise ValueError("Exactly sixteen fresh independent paired model roots")
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    sampled=rng.integers(0,16,size=(BOOTSTRAP_DRAWS,16))
    scores=v[sampled].mean(1)
    return {
        "mean":float(v.mean()),
        "lower95":float(np.percentile(scores,2.5)),
        "upper95":float(np.percentile(scores,97.5)),
        "root_sd":float(v.std(ddof=1)),
        "seed":BOOTSTRAP_SEED,
        "draws":BOOTSTRAP_DRAWS,
    }


def aggregate(records:list[dict]):
    by={r["root"]:r for r in records}
    if len(records)!=16 or len(by)!=16 or set(by)!=set(ROOTS):
        raise ValueError("Require exactly 16 new FC03 engineering model roots")
    per=[]
    for seed in ROOTS:
        r=by[seed]
        if r["study"]!="BPD-NC09-FC03" or r["phase"]!="CONSTRAINED_GRASSMANN_ENGINEERING_DEVELOPMENT" or r["protocol_sha256"]!=PROTO_SHA:
            raise AssertionError("FC03 registered source/phase changed")
        if r["parent_integrity"]["nc09_original_experiment_sha256"]!=PARENT_SOURCE_SHA:
            raise AssertionError("Frozen original NC09 source changed")
        audited=independent_source_audit(r)
        primary=("dp12","flat3","flat3_dose","flat4")
        endpoints=all((r["task_nll_by_arm"].get(k) is not None for k in primary))
        solver_nll=r["sham_nlls"]
        baseline_nll=r["fc02_original_rejection_sham_nlls"]
        primary_dp=(r["task_nll_by_arm"]["dp12"]["nll_nats"]
                    if endpoints else None)
        own_sham_avg=(float(np.mean(solver_nll)) if solver_nll else None)
        row={
            "root":seed,
            "full_model_eligible":bool(r["full_mask_closure"]["passed"] and endpoints
                                       and r["source_gain_eligible"]),
            "solver_source_eligible":audited["source_eligible"],
            "full_registered_eligible":bool(r["eligible"] and audited["source_eligible"]),
            "rank_tuple":r["source_rank12_optimum"]["ranks"] if r["source_rank12_optimum"] else None,
            "solver_matching_charts":r["sham_admission"]["charts_actually_attempted"]
                if r.get("sham_admission") else 0,
            "solver_accepted":len(solver_nll),
            "solver_qualified":audited.get("source_qualified",0),
            "baseline_qualified":audited.get("fc02_baseline_qualified",0),
            "baseline_accepted":len(baseline_nll),
            "baseline_full_source_admission":audited.get("fc02_full_original_admission",False),
            "DP12_NLL":primary_dp,
            "solver_sham_NLL_mean":own_sham_avg,
            "solver_sham_minus_DP12":own_sham_avg-primary_dp
                if primary_dp is not None and len(solver_nll)==REQUIRED_SHAMS else None,
            "original_FC02_rejection_sham_minus_DP12":(
                float(np.mean(baseline_nll))-primary_dp
                if primary_dp is not None and len(baseline_nll)==REQUIRED_SHAMS else None
            ),
            "A_flat3_minus_DP12":(
                r["task_nll_by_arm"]["flat3"]["nll_nats"]-primary_dp if endpoints else None
            ),
            "D_doseflat3_minus_DP12":(
                r["task_nll_by_arm"]["flat3_dose"]["nll_nats"]-primary_dp if endpoints else None
            ),
            "original_geometry_audit":audited.get("geometry",[]),
        }
        per.append(row)
    solver_admitted=sum(bool(x["full_registered_eligible"]) for x in per)
    baseline_admitted=sum(bool(x["baseline_full_source_admission"]) for x in per)
    full_geometry=all(
        len(x["original_geometry_audit"])==REQUIRED_SHAMS and
        all(v["symmetry_max"]<=1e-6 and v["idempotence_max"]<=1e-6
            for v in x["original_geometry_audit"])
        for x in per
    )
    full_behavior=all(
        x["full_model_eligible"] and x["DP12_NLL"] is not None
        and x["solver_sham_NLL_mean"] is not None
        and x["solver_accepted"]==REQUIRED_SHAMS
        for x in per
    )
    gates={
        "sixteen_full_native_models_and_eight_constrained_shams":solver_admitted==16,
        "two_more_root_admissions_than_original_fc02_random":baseline_admitted<=14
            and solver_admitted-baseline_admitted>=2,
        "source_constraint_and_orthogonal_rank_geometry":full_geometry,
        "all_sixteen_native_task_and_sham_outcomes":full_behavior,
    }
    bootstrap=(
        root_bootstrap([x["solver_sham_minus_DP12"] for x in per])
        if solver_admitted==16 else None
    )
    return {
        "study":"BPD-NC09-FC03",
        "phase":"CONSTRAINED_GRASSMANN_ENGINEERING_DEVELOPMENT",
        "evidence_class":"ENGINEERING_ONLY",
        "protocol_sha256":PROTO_SHA,
        "source_roots":16,
        "solver_fully_admitted_roots":solver_admitted,
        "FC02_frozen_sampler_fully_admitted_roots":baseline_admitted,
        "first_original_development_verdict":(
            "CONSTRAINED_GRASSMANN_ENGINEERING_POSITIVE"
            if all(gates.values()) else "ENGINEERING_NULL_OR_INELIGIBLE"
        ),
        "gates":gates,
        "solver_sham_mean_effect_descriptive_bootstrap":bootstrap,
        "scientific_confirmation_roots_evaluated":0,
        "new_independent_59_root_recalibration_evaluated":0,
        "original_FC01_hold":"CALIBRATION_REDESIGN_OR_PILOT_ONLY",
        "original_FC02_hold":"CALIBRATION_INCOMPLETE",
        "per_root":per,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    docs=[json.loads((args.root_dir/("nc09_fc03_root_%d.json"%seed)).read_text())
          for seed in ROOTS]
    result=aggregate(docs)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "study":result["study"],
        "original_verdict":result["first_original_development_verdict"],
        "solver_admitted":result["solver_fully_admitted_roots"],
        "original_random_sampler_admitted":result["FC02_frozen_sampler_fully_admitted_roots"],
        "gates":result["gates"],
        "descriptive_sham_effect":result["solver_sham_mean_effect_descriptive_bootstrap"],
        "confirmation_stays_sealed":True,
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
