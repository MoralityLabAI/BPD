"""Original NC09-FC03 sixteen-root evidence publication and independent ZIP audit.

This file is post-result analysis. It NEVER trains a model or regenerates
source/test observations. Optional --archive checks the exact immutable first
Actions ZIP and every original per-root model+physical projector receipt.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
RUN_ID=44293437804
ORIGINAL_ARTIFACT_ID=14421474360
ORIGINAL_ZIP_BYTES=1552861
ORIGINAL_ZIP_SHA="1c9e687ae4f00ac9e123e774fcf0f285dc6c6594bddfdddc43f6d7dd98e0de03"
ORIGINAL_SUMMARY_SHA="4a9b19253cff8036d5eb42a6fa298b38bee879717b0448250671d6b51919bbfe"
ORIGINAL_PROTOCOL_SHA="06871fba7528aaa687e564331d1b7b24391fb702ab3f1b0f9cf995edd27d1ffc"
ORIGINAL_EXPERIMENT_SHA="f6d6ee956c47d08eb601358943645370100d7bc3832784a2675b41c25cc660bd"
ORIGINAL_SOLVER_GIT_BLOB_SHA="0fc8c6989e16d6deeb8e642029c63414458a43bf"
EXPECTED_ROOTS=tuple(range(14000,14016))
EXPECTED_SOLVER_MATCHES=(8,0,8,8,1,8,1,8,8,8,8,8,8,8,8,8)
EXPECTED_BASELINE_MATCHES=(8,0,1,8,0,8,0,8,8,8,8,8,8,8,8,7)
FIELDS=(
 "root","rank_tuple","model_eligible","solver_selected",
 "original_fc02_selected","original_fc02_qualified","solver_chart_starts_used",
 "source_signed_score_mean","source_score_rms","DP12_nll","flat3_nll",
 "dose_flat3_nll","flat4_nll","A_flat3_minus_DP12",
 "D_doseflat3_minus_DP12","C_DP12_minus_clean","F_DP12_minus_flat4",
 "solver_mean_sham_minus_DP12_if_complete",
 "original_fc02_mean_sham_minus_DP12_if_complete","source_sha256"
)


def sha256(data:bytes):
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data:bytes):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def read_original_published_csv():
    with (ROOT/"PER_ROOT_20261118.csv").open(newline="",encoding="utf-8") as f:
        d=csv.DictReader(f)
        assert tuple(d.fieldnames)==FIELDS
        rows=list(d)
    with (ROOT/"ORIGINAL_16_ROOT_SHA256.csv").open(newline="",encoding="utf-8") as f:
        d=csv.DictReader(f)
        assert d.fieldnames==["root","original_json_sha256"]
        digests=list(d)
    assert len(rows)==len(digests)==16
    assert tuple(int(r["root"]) for r in rows)==EXPECTED_ROOTS
    assert [int(r["solver_selected"]) for r in rows]==list(EXPECTED_SOLVER_MATCHES)
    assert [int(r["original_fc02_selected"]) for r in rows]==list(EXPECTED_BASELINE_MATCHES)
    assert sum(int(r["model_eligible"]) for r in rows)==13
    assert sum(int(r["original_fc02_selected"])==8 for r in rows)==11
    assert tuple(int(r["root"]) for r in digests)==EXPECTED_ROOTS
    assert all(len(r["original_json_sha256"])==64 for r in digests)
    assert all(r["source_sha256"]==ORIGINAL_EXPERIMENT_SHA for r in rows)
    A=[float(r["A_flat3_minus_DP12"]) for r in rows]
    D=[float(r["D_doseflat3_minus_DP12"]) for r in rows]
    assert abs(float(np.mean(A))-.02704052859917283)<1e-12
    assert abs(float(np.mean(D))-.02656273078173399)<1e-12
    assert sum(z>0 for z in A)==15
    return rows,{int(x["root"]):x["original_json_sha256"] for x in digests}


def test_source_bytes_and_original_independent_root_publication():
    assert sha256((ROOT/"PROTOCOL.md").read_bytes())==ORIGINAL_PROTOCOL_SHA
    assert sha256((ROOT/"experiment.py").read_bytes())==ORIGINAL_EXPERIMENT_SHA
    assert git_blob_sha((ROOT/"solver.py").read_bytes())==ORIGINAL_SOLVER_GIT_BLOB_SHA
    read_original_published_csv()


def decode_projector(a:dict):
    raw=base64.b64decode(a["projector_b64_le_float64"],validate=True)
    assert hashlib.sha256(raw).hexdigest()==a["projector_sha256"]
    assert len(raw)==4*20*20*8
    p=np.frombuffer(raw,dtype="<f8").reshape(4,20,20)
    assert np.isfinite(p).all()
    return p


def audit_original_zip(zip_path:Path):
    original=zip_path.read_bytes()
    assert len(original)==ORIGINAL_ZIP_BYTES
    assert sha256(original)==ORIGINAL_ZIP_SHA
    rows,original_hashes=read_original_published_csv()
    by={int(r["root"]):r for r in rows}
    with zipfile.ZipFile(zip_path) as z:
        expected={"FC03_ORIGINAL_16_ROOT_SUMMARY.json"}|{
            f"nc09_fc03_root_{root}.json" for root in EXPECTED_ROOTS
        }
        assert set(z.namelist())==expected
        summary_b=z.read("FC03_ORIGINAL_16_ROOT_SUMMARY.json")
        assert sha256(summary_b)==ORIGINAL_SUMMARY_SHA
        summary=json.loads(summary_b)
        assert summary["study"]=="BPD-NC09-FC03"
        assert summary["phase"]=="CONSTRAINED_GRASSMANN_ENGINEERING_DEVELOPMENT"
        assert summary["evidence_class"]=="ENGINEERING_ONLY"
        assert summary["source_roots"]==16
        assert summary["solver_fully_admitted_roots"]==13
        assert summary["FC02_frozen_sampler_fully_admitted_roots"]==11
        assert summary["first_original_development_verdict"]=="ENGINEERING_NULL_OR_INELIGIBLE"
        assert summary["gates"]=={
            "all_sixteen_native_task_and_sham_outcomes":False,
            "sixteen_full_native_models_and_eight_constrained_shams":False,
            "source_constraint_and_orthogonal_rank_geometry":False,
            "two_more_root_admissions_than_original_fc02_random":True,
        }
        assert summary["solver_sham_mean_effect_descriptive_bootstrap"] is None
        assert summary["scientific_confirmation_roots_evaluated"]==0
        assert summary["new_independent_59_root_recalibration_evaluated"]==0
        assert summary["original_FC01_hold"]=="CALIBRATION_REDESIGN_OR_PILOT_ONLY"
        assert summary["original_FC02_hold"]=="CALIBRATION_INCOMPLETE"
        for seed in EXPECTED_ROOTS:
            raw=z.read(f"nc09_fc03_root_{seed}.json")
            assert sha256(raw)==original_hashes[seed]
            d=json.loads(raw)
            row=by[seed]
            assert d["root"]==seed and d["study"]=="BPD-NC09-FC03"
            assert d["phase"]=="CONSTRAINED_GRASSMANN_ENGINEERING_DEVELOPMENT"
            assert d["training"]["steps"]==384
            assert d["splits"]=={
                "train":2048,"derivative_discovery":1024,
                "model_validation":512,"labeled_calibration":1024,
            }
            assert d["protocol_sha256"]==ORIGINAL_PROTOCOL_SHA
            assert d["source_sha256"]==ORIGINAL_EXPERIMENT_SHA
            assert d["full_mask_closure"]["passed"]
            assert sum(d["source_rank12_optimum"]["ranks"])==12
            assert int(d["eligible"])==int(row["model_eligible"])
            assert int(row["solver_selected"])==len(d["sham_nlls"])
            assert int(row["original_fc02_selected"])==len(d["fc02_original_rejection_sham_nlls"])
            assert d["fc02_original_rejection_admission"]["proposals"]==4096
            adm=d["sham_admission"]
            assert adm["charts_configured"]==32 and adm["max_adam_steps_per_chart"]==64
            assert len(adm["charts"])==32 and adm["source_observation_count"]==1024
            assert adm["uses_external_labels"] is False
            assert abs(adm["target_source_signed_score_mean"]-
                       float(row["source_signed_score_mean"]))<1e-10
            assert abs(adm["target_source_score_rms"]-
                       float(row["source_score_rms"]))<1e-10
            dp=decode_projector(d["source_dp12_projector_receipt"])
            P=[]
            ranks=d["source_rank12_optimum"]["ranks"]
            for sham in adm["accepted"]:
                q=decode_projector(sham)
                assert float(np.max(np.abs(q@q-q)))<1e-6
                assert float(np.max(np.abs(q-q.transpose(0,2,1))))<1e-6
                assert all(abs(float(np.trace(q[t]))-ranks[t])<1e-5 for t in range(4))
                sep=float(np.sqrt(np.square(q-dp).sum()/24.))
                assert sep>=.20 and abs(sep-sham["distance_from_dp12"])<1e-8
                for previous in P:
                    assert float(np.sqrt(np.square(q-previous).sum()/24.))>=.18
                assert sham["physical_rms_relative_error"]<=.15+1e-9
                assert sham["signed_score_abs_error"]<=adm["source_signed_score_abs_tolerance"]+1e-9
                assert sham["score_rms_abs_error"]<=adm["source_score_rms_abs_tolerance"]+1e-9
                P.append(q)
            assert len(P)==adm["selected"]
            nll=d["task_nll_by_arm"]
            assert abs(nll["dp12"]["nll_nats"]-float(row["DP12_nll"]))<1e-10
            assert abs(nll["flat3"]["nll_nats"]-float(row["flat3_nll"]))<1e-10
            assert abs(nll["flat3_dose"]["nll_nats"]-float(row["dose_flat3_nll"]))<1e-10
            assert abs(nll["flat4"]["nll_nats"]-float(row["flat4_nll"]))<1e-10
            if len(P)==8:
                assert abs(np.mean(d["sham_nlls"])-nll["dp12"]["nll_nats"]-
                           float(row["solver_mean_sham_minus_DP12_if_complete"]))<1e-10
        print("PASS: immutable original sixteen-root FC03 source/shams and registered engineering FAIL")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path)
    args=p.parse_args()
    test_source_bytes_and_original_independent_root_publication()
    if args.archive is not None:
        audit_original_zip(args.archive)
    print("PASS: NC09-FC03 original source SHA and 16-root publication")


if __name__=="__main__":
    main()
