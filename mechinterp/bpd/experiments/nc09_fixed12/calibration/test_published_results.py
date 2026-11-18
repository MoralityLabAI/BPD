"""Independent post-result NC09-FC01 first 59-root calibration audit.

The source/protocol/59 model-root roster and all original evaluation outcomes
are frozen. This module does no training, model selection or new root scoring.
Use --archive to verify every original JSON receipt and replay exact pinned
EvalMath power output from the original raw input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from mechinterp.bpd.experiments.nc09_fixed12.calibration import evalmath_preflight

ROOT=Path(__file__).resolve().parent
RUN_ID=44276266076
ARTIFACT_ID=14415331598
ZIP_SIZE=183939
ZIP_SHA="2c70ad34874ed5761199ab8f9795741377d7898f2dae4f5e960d1997f99926e6"
SUMMARY_SHA="e14f10a24fd9b846a68407b27f07eb7046965c198493c57ad9f43f6d1c23f9e6"
EVALMATH_INPUT_SHA="8aebc5b8238906b00abfb717bbcf099a225d55a583dfd63c582914c79109d6f5"
EVALMATH_OUTPUT_SHA="53ef0c0343347ee956afd77a903c9bd67ddcd2dd84ce468d3137a67c81c32b96"
PROTOCOL_SHA="273ba502894747ed48a8e953a6030ff96a57ea7b99df84de350e9860340b90c7"
EXPERIMENT_SHA="7e3adfdaea84a5c2bae1cb105c9046dc2feb87a0bc77fb1f315557917c6bd24c"
VENDORED_UPSTREAM_BLOB="0ff062f76f5534d9f18425f986ae3e750397f50c"
PARENT_ORIGINAL_SHA="8d1b88eea8518a83669f03f9126136ca3d56ac5bac5fe041e2e03fb12922d380"
FIELDS=("root","eligible","rank_tuple","physical_gain","qualified_shams",
        "A_flat3_minus_DP12","D_doseflat3_minus_DP12","S_median_sham_minus_DP12",
        "C_DP12_minus_clean","F_DP12_minus_flat4","T_worst_sham_variation")
FROZEN_MEANS={
    "A_flat3_minus_DP12":.032141546324148014,
    "D_doseflat3_minus_DP12":.031963123987286776,
    "S_median_sham_minus_DP12":.06243262698842307,
    "C_DP12_minus_clean":.014535354229353242,
    "F_DP12_minus_flat4":.0023490211973756048,
}
CAL_FLOOR=.29417911171913147


def sha256(raw:bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha(raw:bytes) -> str:
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()


def published_csv():
    with (ROOT/"PER_ROOT_20261118.csv").open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f)
        assert tuple(r.fieldnames)==FIELDS
        rows=list(r)
    assert len(rows)==59
    assert [int(r["root"]) for r in rows]==list(range(12000,12059))
    assert all(int(r["eligible"])==1 for r in rows)
    assert all(int(r["qualified_shams"])>=8 for r in rows)
    assert all(sum(int(v) for v in r["rank_tuple"].split("-"))==12 for r in rows)
    assert all(.5<=float(r["physical_gain"])<=1.5 for r in rows)
    for key,expected in FROZEN_MEANS.items():
        mean=sum(float(r[key]) for r in rows)/59
        assert abs(mean-expected)<1e-12,(key,mean,expected)
    nulls=np.asarray([float(r["T_worst_sham_variation"]) for r in rows])
    assert abs(float(nulls.max())-CAL_FLOOR)<1e-12
    assert int(rows[int(nulls.argmax())]["root"])==12047
    assert sum(float(r["A_flat3_minus_DP12"])>0 for r in rows)==50
    assert sum(float(r["D_doseflat3_minus_DP12"])>0 for r in rows)==51
    assert sum(float(r["S_median_sham_minus_DP12"])>0 for r in rows)==57
    assert abs(1-.95**59-.9515054747505769)<1e-15
    return rows


def original_manifest():
    with (ROOT/"ORIGINAL_59_ROOT_SHA256.csv").open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==["model_root","original_json_sha256"]
        rows=list(reader)
    assert len(rows)==59
    result={int(r["model_root"]):r["original_json_sha256"] for r in rows}
    assert set(result)==set(range(12000,12059))
    assert all(len(v)==64 for v in result.values())
    return result


def test_original_source_sham_floor_and_exact_calibration_publication():
    assert sha256((ROOT/"PROTOCOL.md").read_bytes())==PROTOCOL_SHA
    assert sha256((ROOT/"experiment.py").read_bytes())==EXPERIMENT_SHA
    assert git_blob_sha((ROOT/"evalmath_preflight.py").read_bytes())==VENDORED_UPSTREAM_BLOB
    published_csv()
    original_manifest()


def audit_first_run_archive(path:Path):
    raw=path.read_bytes()
    assert len(raw)==ZIP_SIZE and sha256(raw)==ZIP_SHA
    manifest=original_manifest()
    with zipfile.ZipFile(path) as archive:
        expected={
            "CAL01_ORIGINAL_59_ROOT_SUMMARY.json",
            "EVALMATH_ORIGINAL_59_ROOT_PILOT.json",
            "EVALMATH_PINNED_FIRST_PREFLIGHT.json",
        }|{"nc09_cal_root_%d.json"%root for root in manifest}
        assert set(archive.namelist())==expected
        data_summary=archive.read("CAL01_ORIGINAL_59_ROOT_SUMMARY.json")
        data_input=archive.read("EVALMATH_ORIGINAL_59_ROOT_PILOT.json")
        data_power=archive.read("EVALMATH_PINNED_FIRST_PREFLIGHT.json")
        assert sha256(data_summary)==SUMMARY_SHA
        assert sha256(data_input)==EVALMATH_INPUT_SHA
        assert sha256(data_power)==EVALMATH_OUTPUT_SHA
        summary=json.loads(data_summary)
        pilot=json.loads(data_input)
        power=json.loads(data_power)
        assert summary["study"]=="BPD-NC09-FC01"
        assert summary["calibration_verdict"]=="CALIBRATION_REDESIGN_OR_PILOT_ONLY"
        assert summary["full_59_root_admission"]
        assert summary["eligible_root_count"]==summary["root_count"]==59
        assert summary["order_statistic_rank"]==59
        assert abs(summary["95_95_sham_floor"]-CAL_FLOOR)<1e-12
        assert summary["independent_confirmation_roots_evaluated"]==0
        assert power["decision"]=="REDESIGN_OR_PILOT_ONLY"
        assert power["order_statistic_rank"]==59
        assert set(power["reasons"])=={
            "target_below_mechanistic_variation_floor",
            "pilot_effect_not_above_mechanistic_variation_floor",
            "confirmation_underpowered",
        }
        assert power["planned_confirmation_worlds"]==32
        assert power["pilot_worlds"]==59
        assert power["approx_worlds_for_80pct_power"] is None
        assert abs(power["approx_resolvable_effect"]-.3139165964704633)<1e-12
        # Recompute the unchanged EvalMath evaluator directly on original
        # first-run JSON input, not on manually chosen reported statistics.
        independent=evalmath_preflight.preflight(pilot)
        assert independent==power
        assert len(pilot["pilot"])==59
        by={r["root"]:r for r in summary["per_root"]}
        assert len(by)==59
        published={int(r["root"]):r for r in published_csv()}
        nulls=[]
        for root in range(12000,12059):
            member=archive.read("nc09_cal_root_%d.json"%root)
            assert sha256(member)==manifest[root]
            rec=json.loads(member)
            assert rec["root"]==root and rec["eligible"]
            assert rec["study"]=="BPD-NC09-FC01"
            assert rec["phase"]=="INDEPENDENT_SHAM_CALIBRATION"
            assert rec["protocol_sha256"]==PROTOCOL_SHA
            assert rec["source_sha256"]==EXPERIMENT_SHA
            assert rec["parent_integrity"]["nc09_original_experiment_sha256"]==PARENT_ORIGINAL_SHA
            assert rec["training"]["steps"]==384
            assert rec["splits"]=={
                "train":2048,"derivative_discovery":1024,
                "model_validation":512,"labeled_calibration":1024
            }
            assert rec["full_mask_closure"]["passed"]
            assert rec["sham_admission"]["proposals"]==4096
            assert rec["sham_admission"]["selected"]==8
            assert len(rec["sham_nlls"])==8
            assert sum(rec["source_rank12_optimum"]["ranks"])==12
            obs=np.asarray(rec["sham_nlls"],dtype=float)
            median=float(np.median(obs))
            t=float(np.max(np.abs(obs-median)))
            assert abs(t-rec["effects"]["T_null_worst_abs"])<1e-12
            nulls.append(t)
            row=published[root]
            assert abs(t-float(row["T_worst_sham_variation"]))<1e-12
            assert abs(rec["physical_source_gain"]-float(row["physical_gain"]))<1e-12
            assert rec["source_rank12_optimum"]["ranks"]==[
                int(v) for v in row["rank_tuple"].split("-")
            ]
            assert by[root]["eligible"] and by[root]["selected_shams"]==8
        assert abs(max(nulls)-CAL_FLOOR)<1e-12
    print("PASS: original 59-root full ZIP, independent model receipts and pinned EvalMath redesign verdict")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path)
    args=p.parse_args()
    test_original_source_sham_floor_and_exact_calibration_publication()
    if args.archive:
        audit_first_run_archive(args.archive)
    print("PASS: post-result FC01 immutable source and 59-root calibration hold")


if __name__=="__main__":
    main()
