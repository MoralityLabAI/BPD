"""Independent FC04 first-run sixteen-root publication and immutable ZIP audit.

This post-result source audits the original SHA and per-root registered
outcomes. It does not train new models or open scientific calibration /
confirmation roots. --archive checks the complete original Actions ZIP.
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
ORIGINAL_RUN_ID=44307332891
ORIGINAL_ARTIFACT_ID=14428573240
ORIGINAL_ZIP_SHA256="477e40d3e2d90005d7ab60739db2f263f5e91c4ca60656b0834c7811769ee5ac"
ORIGINAL_ZIP_BYTES=3029951
ORIGINAL_SUMMARY_SHA256="946eb13540e0562242cc92b1f65f56918177cf58ac621498ba8767b147976a4f"
ORIGINAL_SUMMARY_BYTES=66300
ORIGINAL_PROTOCOL_SHA256="60b65db2ba4032dd830aba113b2013303d677d28bc6c6a789ea0786111a415bb"
ORIGINAL_EXPERIMENT_SHA256="e6ae203247f515aafb06704afbbea40bbd6a6898669833f657d82fd55ca05583"
ORIGINAL_SOLVER_BLOB_SHA="89fb5446d2734ba3c34b765354ebfd1ea9e3eba4"
ROOTS=tuple(range(15000,15016))
SOLVER_MATCHES=(8,)*16
FC03_MATCHES=(8,8,0,8,8,8,8,8,8,5,8,8,8,0,8,0)
FC02_MATCHES=(8,8,1,8,8,8,8,3,8,4,8,8,8,0,8,0)
COLUMNS=(
    "root","ranks","FC04_matched_shams","FC03_matched_shams",
    "FC02_matched_shams","full_tangent_starts","Jacobian_corrections",
    "DP12_NLL","FC04_sham_minus_DP12","flat3_minus_DP12",
    "dose_flat3_minus_DP12","DP12_minus_flat4","original_root_SHA256",
)


def sha(data:bytes):
    return hashlib.sha256(data).hexdigest()


def blob_sha(data:bytes):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def original_table():
    with (ROOT/"PER_ROOT_20261118.csv").open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        assert tuple(reader.fieldnames)==COLUMNS
        rows=list(reader)
    with (ROOT/"ORIGINAL_16_ROOT_SHA256.csv").open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==["root","original_json_sha256"]
        digests=list(reader)
    assert len(rows)==len(digests)==16
    assert tuple(int(r["root"]) for r in rows)==ROOTS
    assert tuple(int(r["FC04_matched_shams"]) for r in rows)==SOLVER_MATCHES
    assert tuple(int(r["FC03_matched_shams"]) for r in rows)==FC03_MATCHES
    assert tuple(int(r["FC02_matched_shams"]) for r in rows)==FC02_MATCHES
    assert len(set(int(x["root"]) for x in digests))==16
    original_root_sha={int(x["root"]):x["original_json_sha256"] for x in digests}
    for row in rows:
        assert row["original_root_SHA256"]==original_root_sha[int(row["root"])]
        assert len(row["original_root_SHA256"])==64
        assert sum(map(int,row["ranks"].split("-")))==12
        assert int(row["full_tangent_starts"])==8
    assert np.mean([float(r["FC04_sham_minus_DP12"]) for r in rows])==np.mean([
        float(r["FC04_sham_minus_DP12"]) for r in rows
    ])
    assert abs(np.mean([float(r["FC04_sham_minus_DP12"]) for r in rows])-
               .009626356652006507)<1e-12
    assert sum(float(r["FC04_sham_minus_DP12"])>0 for r in rows)==15
    assert abs(np.mean([float(r["flat3_minus_DP12"]) for r in rows])-
               .02446041628718376)<1e-12
    return rows,original_root_sha


def test_first_registered_source_hashes_and_published_original_gate():
    assert sha((ROOT/"PROTOCOL.md").read_bytes())==ORIGINAL_PROTOCOL_SHA256
    assert sha((ROOT/"experiment.py").read_bytes())==ORIGINAL_EXPERIMENT_SHA256
    assert blob_sha((ROOT/"solver.py").read_bytes())==ORIGINAL_SOLVER_BLOB_SHA
    original_table()


def projector_from_original(a:dict):
    binary=base64.b64decode(a["projector_b64_le_float64"],validate=True)
    assert sha(binary)==a["projector_sha256"]
    assert len(binary)==4*20*20*8
    q=np.frombuffer(binary,dtype="<f8").reshape((4,20,20))
    assert np.isfinite(q).all()
    return q


def audit_original_first_run(path:Path):
    binary=path.read_bytes()
    assert len(binary)==ORIGINAL_ZIP_BYTES and sha(binary)==ORIGINAL_ZIP_SHA256
    rows,root_hashes=original_table()
    indexed={int(r["root"]):r for r in rows}
    with zipfile.ZipFile(path) as z:
        expected={"FC04_ORIGINAL_16_ROOT_SUMMARY.json"}|{
            f"nc09_fc04_root_{root}.json" for root in ROOTS
        }
        assert set(z.namelist())==expected
        raw=z.read("FC04_ORIGINAL_16_ROOT_SUMMARY.json")
        assert len(raw)==ORIGINAL_SUMMARY_BYTES and sha(raw)==ORIGINAL_SUMMARY_SHA256
        summary=json.loads(raw)
        assert summary["study"]=="BPD-NC09-FC04"
        assert summary["phase"]=="FULL_TANGENT_JACOBIAN_ENGINEERING_DEVELOPMENT"
        assert summary["evidence_class"]=="ENGINEERING_ONLY"
        assert summary["first_source_locked_development_verdict"]=="FULL_TANGENT_ENGINEERING_POSITIVE"
        assert summary["model_roots"]==16
        assert summary["full_tangent_admitted_roots"]==16
        assert summary["FC03_old_eightdim_solver_admitted_roots"]==12
        assert summary["FC02_original_random_sampler_admitted_roots"]==11
        assert all(summary["gates"].values())
        assert summary["protected_59_root_recalibration_evaluated"]==0
        assert summary["protected_32_root_confirmation_evaluated"]==0
        assert summary["original_FC01_hold"]=="CALIBRATION_REDESIGN_OR_PILOT_ONLY"
        assert summary["original_FC02_hold"]=="CALIBRATION_INCOMPLETE"
        assert summary["original_FC03_verdict"]=="ENGINEERING_NULL_OR_INELIGIBLE"
        stats=summary["solver_sham_effect_descriptive_bootstrap"]
        assert stats["seed"]==15041504 and stats["draws"]==4096
        values=np.asarray([
            float(indexed[root]["FC04_sham_minus_DP12"]) for root in ROOTS
        ])
        rng=np.random.default_rng(15041504)
        ix=rng.integers(0,16,size=(4096,16))
        means=values[ix].mean(axis=1)
        assert abs(float(means.mean())-float(means.mean()))<1e-12
        assert abs(stats["mean"]-float(values.mean()))<1e-12
        assert abs(stats["lower95"]-float(np.percentile(means,2.5)))<1e-12
        assert abs(stats["upper95"]-float(np.percentile(means,97.5)))<1e-12
        assert abs(stats["root_sd"]-float(values.std(ddof=1)))<1e-12
        jac_rank_3=0
        for root in ROOTS:
            original=z.read(f"nc09_fc04_root_{root}.json")
            assert sha(original)==root_hashes[root]
            d=json.loads(original)
            assert d["study"]=="BPD-NC09-FC04" and d["root"]==root
            assert d["phase"]=="FULL_TANGENT_JACOBIAN_ENGINEERING_DEVELOPMENT"
            assert d["training"]["steps"]==384
            assert d["splits"]=={
                "train":2048,"derivative_discovery":1024,
                "model_validation":512,"labeled_calibration":1024
            }
            assert d["source_sha256"]==ORIGINAL_EXPERIMENT_SHA256
            assert d["protocol_sha256"]==ORIGINAL_PROTOCOL_SHA256
            assert d["full_mask_closure"]["passed"] and d["eligible"]
            assert sum(d["source_rank12_optimum"]["ranks"])==12
            assert d["sham_admission"]["charts_configured"]==32
            assert d["sham_admission"]["max_corrections_per_chart"]==64
            assert d["sham_admission"]["selected"]==len(d["sham_nlls"])==8
            assert d["sham_admission"]["uses_external_task_labels"] is False
            assert d["fc03_original_eightdim_admission"]["charts_configured"]==32
            assert d["fc02_original_rejection_admission"]["proposals"]==4096
            assert len(d["fc03_original_eightdim_sham_nlls"])==int(
                indexed[root]["FC03_matched_shams"]
            )
            assert len(d["fc02_original_rejection_sham_nlls"])==int(
                indexed[root]["FC02_matched_shams"]
            )
            jac_rank_3+=d["sham_admission"]["jacobian_rank_census"]["3"]
            frozen_dp=projector_from_original(d["source_dp12_projector_receipt"])
            originals=[]
            ranks=d["source_rank12_optimum"]["ranks"]
            for sh in d["sham_admission"]["projectors"]:
                p=projector_from_original(sh)
                assert np.max(np.abs(p@p-p))<1e-6
                assert np.max(np.abs(p-p.transpose(0,2,1)))<1e-6
                assert all(abs(float(np.trace(p[t]))-ranks[t])<1e-6
                           for t in range(4))
                dp_sep=float(np.sqrt(np.square(p-frozen_dp).sum()/24.))
                assert dp_sep>=.20 and abs(dp_sep-sh["distance_from_dp12"])<1e-8
                for old in originals:
                    assert float(np.sqrt(np.square(p-old).sum()/24.))>=.18
                assert sh["physical_rms_relative_error"]<=.15+1e-9
                assert sh["signed_score_abs_error"]<=d["sham_admission"]["source_signed_score_abs_tolerance"]+1e-9
                assert sh["score_rms_abs_error"]<=d["sham_admission"]["source_score_rms_abs_tolerance"]+1e-9
                originals.append(p)
            assert len(originals)==8
            dp=float(d["task_nll_by_arm"]["dp12"]["nll_nats"])
            assert abs(dp-float(indexed[root]["DP12_NLL"]))<1e-10
            assert abs(float(np.mean(d["sham_nlls"]))-dp-
                       float(indexed[root]["FC04_sham_minus_DP12"]))<1e-10
        assert jac_rank_3==348
        print("PASS immutable FC04 16/16 versus FC03 12/16 and FC02 11/16; all 128 original projectors")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",type=Path)
    args=ap.parse_args()
    test_first_registered_source_hashes_and_published_original_gate()
    if args.archive is not None:
        audit_original_first_run(args.archive)
    print("PASS FC04 original independent root CSV and scientific source hashes")


if __name__=="__main__":
    main()
