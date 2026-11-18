"""FC01: source-locked 59-independent-root sham calibration and exact EvalMath.

All source formulas, calibration rules and schema were prospectively frozen.
No independent 32-root scientific confirmation is trained, generated or read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from mechinterp.bpd.experiments.nc09_fixed12.calibration import evalmath_preflight
from mechinterp.bpd.experiments.nc09_fixed12.calibration.experiment import (
    ROOT, ROOTS, PROTO_SHA, PARENT_SOURCE_SHA,
)

EVALMATH_BLOB_SHA="0ff062f76f5534d9f18425f986ae3e750397f50c"
CALIBRATION_CONFIDENCE=1-.95**59
NULL_FIELDS=(
    "A_flat3_minus_dp12","D_dose_flat3_minus_dp12",
    "S_median_sham_minus_dp12","C_dp12_minus_clean",
    "F_dp12_minus_flat4",
)


def git_blob_sha(content:bytes) -> str:
    return hashlib.sha1(b"blob "+str(len(content)).encode()+b"\x00"+content).hexdigest()


def verify_pinned_evalmath():
    actual=git_blob_sha((ROOT/"evalmath_preflight.py").read_bytes())
    if actual!=EVALMATH_BLOB_SHA:
        raise AssertionError("EvalMath script not byte-identical to pinned upstream")
    return actual


def make_evalmath_pilot(by_root:list[dict]):
    return {
        "schema_version":1,
        "experiment_id":"BPD-NC09-FC01",
        "independent_unit":"independently trained dense 20D four-pass model root",
        "alpha":"1/20",
        "scientific_min_effect":"1/125",
        "target_effect":"27/1000",
        "measurement_resolution":"1/1000",
        "sham_quantile":"19/20",
        "sham_confidence":"19/20",
        "planned_comparisons":3,
        "confirm_units_planned":32,
        "pilot_from_development_only":True,
        "heldout_sealed":True,
        "units_independent":True,
        "pilot":[{
            "unit_id":"NC09-FC01-root-%d"%r["root"],
            "paired_effect":str(r["effects"]["A_flat3_minus_dp12"]),
            "sham_effects":[str(v) for v in r["effects"]["null_sham_signed_effects"]],
        } for r in by_root],
    }


def aggregate(records:list[dict]) -> tuple[dict,dict|None,dict|None]:
    by={r["root"]:r for r in records}
    if len(records)!=59 or len(by)!=59 or set(by)!=set(ROOTS):
        raise ValueError("First calibration requires exactly the registered 59 roots")
    per=[]
    all_admitted=True
    for seed in ROOTS:
        r=by[seed]
        if (r["study"]!="BPD-NC09-FC01" or
            r["phase"]!="INDEPENDENT_SHAM_CALIBRATION" or
            r["protocol_sha256"]!=PROTO_SHA):
            raise AssertionError("Incompatible NC09 original calibration source")
        if r["parent_integrity"]["nc09_original_experiment_sha256"]!=PARENT_SOURCE_SHA:
            raise AssertionError("NC09 original physical source drift")
        eff=r.get("effects")
        shams=r.get("sham_nlls",[])
        admitted=bool(r["eligible"] and eff is not None and
                      len(shams)==8 and
                      r["sham_admission"] is not None and
                      r["sham_admission"]["selected"]==8 and
                      all(isinstance(eff[k],(int,float)) and math.isfinite(eff[k])
                          for k in NULL_FIELDS) and
                      len(eff["null_sham_signed_effects"])==8 and
                      all(math.isfinite(float(z)) for z in eff["null_sham_signed_effects"]))
        all_admitted &= admitted
        row={
            "root":seed,"eligible":admitted,
            "rank_tuple":r["source_rank12_optimum"]["ranks"]
                        if r.get("source_rank12_optimum") else None,
            "source_dose_ratio":r.get("physical_source_gain"),
            "qualified_shams":r["sham_admission"]["qualified"]
                              if r.get("sham_admission") else 0,
            "selected_shams":len(shams),
            "effects":eff,
            "original_root_source_sha256":r["source_sha256"],
        }
        if eff is not None:
            null=np.asarray(eff["null_sham_signed_effects"],dtype=float)
            if abs(np.median(shams)-np.median(shams))>1e-12:
                raise AssertionError("unexpected median arithmetic")
            assert abs(float(null.max(initial=0)) - max([0]+list(null)))<1e-12
            assert abs(eff["T_null_worst_abs"]-float(abs(null).max()))<1e-12
        per.append(row)
    summary={
        "study":"BPD-NC09-FC01",
        "phase":"INDEPENDENT_SHAM_CALIBRATION",
        "evidence_class":"CALIBRATION_ONLY",
        "protocol_sha256":PROTO_SHA,
        "root_count":59,
        "eligible_root_count":sum(x["eligible"] for x in per),
        "full_59_root_admission":all_admitted,
        "original_95_95_root_tolerance_coverage":CALIBRATION_CONFIDENCE,
        "95_95_sham_floor":None,
        "order_statistic_rank":59 if all_admitted else None,
        "sample_mean_effects":None,
        "sample_sd_effects":None,
        "calibration_verdict":"CALIBRATION_INCOMPLETE",
        "evalmath_script_blob_sha256":None,
        "evalmath_decision":None,
        "evalmath_reasons":None,
        "independent_confirmation_roots_evaluated":0,
        "other_protected_roots_evaluated":0,
        "per_root":per,
    }
    if not all_admitted:
        return summary,None,None
    source_stat=np.asarray([[float(r["effects"][k]) for k in NULL_FIELDS]
                            for r in per],dtype=float)
    floor=max(float(r["effects"]["T_null_worst_abs"]) for r in per)
    summary["95_95_sham_floor"]=floor
    summary["sample_mean_effects"]={
        key:float(source_stat[:,i].mean()) for i,key in enumerate(NULL_FIELDS)
    }
    summary["sample_sd_effects"]={
        key:float(source_stat[:,i].std(ddof=1)) for i,key in enumerate(NULL_FIELDS)
    }
    pilot=make_evalmath_pilot([by[s] for s in ROOTS])
    try:
        source_blob=verify_pinned_evalmath()
        result=evalmath_preflight.preflight(pilot)
        if result["order_statistic_rank"]!=59:
            raise AssertionError("No 59th-order original 95/95 sham floor")
        if abs(result["upper_sham_quantile"]-floor)>1e-12:
            raise AssertionError("EvalMath floor disagrees with independent 59-root max")
        summary["evalmath_script_blob_sha256"]=source_blob
        summary["evalmath_decision"]=result["decision"]
        summary["evalmath_reasons"]=result["reasons"]
        summary["calibration_verdict"]=(
            "CALIBRATION_READY_FOR_SEPARATE_CONFIRMATION_PROTOCOL"
            if result["decision"]=="READY_FOR_CONFIRMATION"
            else "CALIBRATION_REDESIGN_OR_PILOT_ONLY"
        )
    except Exception as error:
        summary["calibration_verdict"]="EVALMATH_NOT_RUN"
        summary["evalmath_reasons"]=[type(error).__name__+": "+str(error)]
        return summary,pilot,None
    return summary,pilot,result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    docs=[json.loads((args.root_dir/("nc09_cal_root_%d.json"%root)).read_text())
          for root in ROOTS]
    summary,pilot,evalmath=aggregate(docs)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/"CAL01_ORIGINAL_59_ROOT_SUMMARY.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n"
    )
    if pilot is not None:
        (args.out_dir/"EVALMATH_ORIGINAL_59_ROOT_PILOT.json").write_text(
            json.dumps(pilot,indent=2,sort_keys=True)+"\n"
        )
    if evalmath is not None:
        (args.out_dir/"EVALMATH_PINNED_FIRST_PREFLIGHT.json").write_text(
            json.dumps(evalmath,indent=2,sort_keys=True)+"\n"
        )
    print(json.dumps({
        "study":summary["study"],"original_verdict":summary["calibration_verdict"],
        "admitted":summary["eligible_root_count"],
        "sham_floor":summary["95_95_sham_floor"],
        "mean_effects":summary["sample_mean_effects"],
        "EvalMath":summary["evalmath_decision"],
        "reasons":summary["evalmath_reasons"],
        "confirmation_models_untouched":True
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
