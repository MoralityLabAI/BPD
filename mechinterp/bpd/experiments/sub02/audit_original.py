"""Independent read-only auditor for original SUB-02 Actions development artifact.

Recomputes root admission, primary contrasts and whole-root bootstrap from
archived JSON. This module does not import/train the source model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ORIGINAL_RUN = 44269044080
ORIGINAL_ARTIFACT = 14411157818
ORIGINAL_JSON_SHA256 = "3f736bcc90d331a00ae8927857d3ee2454029a653e729538cbe1538bbd2e2c12"
ORIGINAL_SOURCE_SHA256 = "8b9f50e5a01d685fc48396199dd0abcff7541fc75f69a61685aca5ecd10ed037"
ORIGINAL_PROTOCOL_SHA256 = "b0380669185b774d8131cc509364a62fdaebd467039dd852a76b5b31d7a56a4a"
ORIGINAL_SUB01_SHA256 = "fd0bb330562b9fb7c61d0d126005b257d0567f2aaf92d7bba2e15848a19d9a2e"
ROOTS = tuple(range(1300, 1308))
RANKS = (8,10)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(source_json: bytes):
    assert sha(source_json) == ORIGINAL_JSON_SHA256, "Input differs from original artifact"
    j = json.loads(source_json)
    a=j["summary"]
    roots=j["roots"]
    assert [x["root"] for x in roots] == list(ROOTS)
    assert a["root_ids"] == list(ROOTS)
    assert a["independent_roots"] == len(ROOTS)
    assert a["primary_rank"] == 8
    assert a["registered_minimum_mean_effect"] == 0.15
    assert a["independent_sham_tail_calibrated"] is False
    assert a["strict_evalmath_preflight"] == "PENDING_INDEPENDENT_SHAM_CALIBRATION"
    assert a["study"] == "BPD-SUB-02"
    assert a["phase"] == "ENGINEERING"
    assert all(x["protocol_sha256"] == ORIGINAL_PROTOCOL_SHA256 for x in roots)
    assert all(x["source_sha256"] == ORIGINAL_SOURCE_SHA256 for x in roots)
    assert all(x["dependency_sub01_sha256"] == ORIGINAL_SUB01_SHA256 for x in roots)
    assert sha((ROOT/"PROTOCOL.md").read_bytes()) == ORIGINAL_PROTOCOL_SHA256
    assert sha((ROOT/"experiment.py").read_bytes()) == ORIGINAL_SOURCE_SHA256

    root_effects=[]
    per_root=[]
    for r in roots:
        assert r["phase"] == "ENGINEERING"
        assert r["training"]["training_steps"] == 256
        assert r["data_sizes"] == {
            "train":2048, "discovery":512, "calibration":512, "heldout":1024,
        }
        assert r["competent"] == (min(r["contexts"].values()) >= .80)
        g=r["gauge"]
        gauge=(g["max_clean_logit_error"]<=5e-5
            and g["max_transported_impulse_logit_error"]<=5e-5
            and g["max_clean_state_error"]<=5e-5
            and g["max_intervened_state_error"]<=5e-5
            and g["untransported_impulse_logit_rms"]>1e-5)
        assert gauge == r["gauge_ok"]
        geo=(r["pca_geometry"]["nested_max_abs"] <= 1e-5
             and r["obs_geometry"]["finite_difference_max_abs"] < .02
             and all(
                v["symmetry_max_abs"] <= 1e-5
                and v["idempotence_max_abs"] <= 1e-5
                and v["trace_error"] <= 1e-4
                for v in r["obs_geometry"]["rank_geometry"].values()))
        assert geo == r["geometry_ok"]
        dose=True
        assert tuple(sorted(int(k) for k in r["arms"]))==RANKS
        for rank in RANKS:
            k=str(rank)
            pair=r["arms"][k]
            assert len(pair["eight_same_dose_random_shams"])==8
            assert all(math.isfinite(z) for z in pair["eight_same_dose_random_shams"])
            for method in ("pca","observability"):
                f=pair[method]["fiber"]
                assert len(f["future_logit_rms_pass2_3_4"]) == 3
                assert f["physical_dose"] == r["physical_impulse_alpha"]
                if f["max_dose_deviation"]>5e-6 or f["max_null_projection"]>5e-6:
                    dose=False
            assert abs(pair["mean_primary_comparator_pca_minus_obs_logit_rms"]-
                       (pair["pca"]["fiber"]["terminal_logit_rms"]
                       -pair["observability"]["fiber"]["terminal_logit_rms"]))<1e-10
            assert abs(pair["median_sham_minus_obs_logit_rms"]
                       -(statistics.median(pair["eight_same_dose_random_shams"])
                       -pair["observability"]["fiber"]["terminal_logit_rms"]))<1e-10
            assert all(v <= 5e-6 for v in r["calibration_dose_checks"][k].values())
        assert dose == r["dose_ok"]
        admitted=bool(r["competent"] and gauge and geo and dose)
        assert admitted==r["root_eligible"]
        if admitted:
            root_effects.append(r["arms"]["8"]
                                ["mean_primary_comparator_pca_minus_obs_logit_rms"])
        per_root.append({
            "root":r["root"], "admitted":admitted,
            "rank8_pca_terminal_rms":r["arms"]["8"]["pca"]["fiber"]["terminal_logit_rms"],
            "rank8_observability_terminal_rms":
                r["arms"]["8"]["observability"]["fiber"]["terminal_logit_rms"],
            "rank8_primary_effect":
                r["arms"]["8"]["mean_primary_comparator_pca_minus_obs_logit_rms"],
            "rank8_observability_next_hidden_commutator":
                r["arms"]["8"]["observability"]["fiber"]
                  ["projected_next_hidden_norm_rms"],
        })
    assert len(root_effects)==a["eligible"]
    assert len(root_effects)==len(a["root_primary_effects"])
    assert all(abs(x-y)<1e-12 for x,y in zip(root_effects,a["root_primary_effects"]))
    ci=None
    if len(root_effects)==len(ROOTS):
        arr=np.array(root_effects)
        rng=np.random.default_rng(130202)
        samples=rng.choice(arr,size=(2000,len(arr)),replace=True).mean(1)
        ci=[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]
        assert all(abs(x-y)<1e-12 for x,y in zip(ci,a["bootstrap95_whole_roots"]))
    else:
        assert a["bootstrap95_whole_roots"] is None
    positive=sum(v>0 for v in root_effects)
    assert positive==a["positive_roots"]
    mean=float(statistics.mean(root_effects)) if root_effects else None
    assert mean is None or abs(mean-a["mean_primary_pca_minus_observability_logit_rms"])<1e-12
    expected=(len(root_effects)==8 and mean>=.15 and positive>=7 and ci[0]>0)
    assert a["verdict"] == (
        "ENGINEERING_SIGNAL_ONLY" if expected
        else "ENGINEERING_NULL_OR_INELIGIBLE"
    )
    return {
        "audit_status":"PASS_INDEPENDENT_ORIGINAL_RECEIPT",
        "original_run":ORIGINAL_RUN, "original_artifact":ORIGINAL_ARTIFACT,
        "source_json_sha256":ORIGINAL_JSON_SHA256,
        "units":8,"eligible":len(root_effects),
        "mean_primary_effect":mean,
        "primary_bootstrap95":ci,
        "registered_verdict":a["verdict"],
        "independent_sham_tail_calibrated":False,
        "per_root":per_root,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.receipt.read_bytes())
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
