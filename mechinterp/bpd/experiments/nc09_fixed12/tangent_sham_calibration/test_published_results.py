"""Independent post-first-run NC09-FC02 publication and raw ZIP receipt auditor.

Reads original frozen source and published rows only. Never trains new models,
fits sham projectors or generates calibration/confirmation root data.
Run with --archive to also verify original 60-member GitHub Actions ZIP.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from statistics import mean,median

ROOT=Path(__file__).resolve().parent
ORIGINAL_RUN=44281207145
ORIGINAL_ARTIFACT=14417745672
ZIP_SHA="982b0b5ae0fb231187aa8cb05e2437f64c6f4da951e09f7254ee4d1952956db3"
SUMMARY_SHA="c4c29944424ff34ed2d507cc909b83349879a02b18c03569e6fdd2cfbce7624a"
PROTOCOL_SHA="3341d78083059acb284628b2c744c6c17bdc5d4ebfef99dabe51e73f95e1d4db"
EXPERIMENT_SHA="0921757e48dafbd0592b8cc7467869d50a6e0f8b47f00c07bf5a219ad5998883"
ORIGINAL_PARENT_SHA="8d1b88eea8518a83669f03f9126136ca3d56ac5bac5fe041e2e03fb12922d380"
FAILED_ROOTS={
    13008,13010,13012,13014,13015,13016,13018,13021,13022,13029,
    13030,13034,13035,13037,13047,13049,13053,13054,13056
}
HEADER=(
    "root","eligible","passwise_rank_tuple","sham_proposals",
    "qualified_shams","selected_shams",
    "signed_score_target","score_rms_target",
    "signed_score_tolerance","score_rms_tolerance",
    "A_flat3_minus_DP12","D_dose_flat3_minus_DP12",
    "C_DP12_minus_clean","F_DP12_minus_flat4",
    "S_mean_sham_minus_DP12_if_8_shams",
    "T_worst_sham_if_8_shams","DP12_NLL","flat3_NLL",
    "dose_flat3_NLL","flat4_NLL","clean_NLL","source_sha256",
)


def digest(data:bytes):
    return hashlib.sha256(data).hexdigest()


def published_data():
    with (ROOT/"PER_ROOT_20261118.csv").open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        assert tuple(reader.fieldnames)==HEADER
        rows=list(reader)
    assert len(rows)==59
    assert {int(r["root"]) for r in rows}==set(range(13000,13059))
    eligible={int(r["root"]) for r in rows if int(r["eligible"])==1}
    assert len(eligible)==40
    assert set(range(13000,13059))-eligible==FAILED_ROOTS
    q=[int(r["qualified_shams"]) for r in rows]
    assert min(q)==0 and max(q)==990 and median(q)==49
    assert sum(x==0 for x in q)==8
    assert all(int(r["sham_proposals"])==4096 for r in rows)
    assert all(int(r["selected_shams"])==min(8,int(r["qualified_shams"])) for r in rows)
    assert all(int(r["sham_proposals"])==4096 for r in rows)
    assert all(r["source_sha256"]==EXPERIMENT_SHA for r in rows)
    for r in rows:
        assert sum(int(v) for v in r["passwise_rank_tuple"].split("-"))==12
        for value,a,b in (
            ("A_flat3_minus_DP12","flat3_NLL","DP12_NLL"),
            ("D_dose_flat3_minus_DP12","dose_flat3_NLL","DP12_NLL"),
            ("C_DP12_minus_clean","DP12_NLL","clean_NLL"),
            ("F_DP12_minus_flat4","DP12_NLL","flat4_NLL"),
        ):
            assert abs(float(r[value])-(float(r[a])-float(r[b])))<1e-10
    def avg(key,selection=rows):
        return mean(float(r[key]) for r in selection)
    assert abs(avg("A_flat3_minus_DP12")-.019324820551831844)<1e-12
    assert abs(avg("D_dose_flat3_minus_DP12")-.01937711188348673)<1e-12
    assert abs(avg("C_DP12_minus_clean")-.011810004142886502)<1e-12
    assert abs(avg("F_DP12_minus_flat4")-.002790774834358086)<1e-12
    yes=[r for r in rows if int(r["eligible"])]
    no=[r for r in rows if not int(r["eligible"])]
    assert abs(avg("S_mean_sham_minus_DP12_if_8_shams",yes)-.010241092252545058)<1e-12
    assert abs(max(float(r["T_worst_sham_if_8_shams"]) for r in yes)-
               .0540989488363266)<1e-12
    assert abs(avg("A_flat3_minus_DP12",yes)-.02403775043785572)<1e-12
    assert abs(avg("A_flat3_minus_DP12",no)-.009402862897044733)<1e-12
    assert all(r["S_mean_sham_minus_DP12_if_8_shams"]=="" for r in no)
    return rows


def root_manifest():
    with (ROOT/"ORIGINAL_ROOT_SHA256.csv").open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f)
        assert tuple(r.fieldnames)==("root","original_json_sha256")
        records=list(r)
    assert len(records)==59
    result={int(x["root"]):x["original_json_sha256"] for x in records}
    assert set(result)==set(range(13000,13059))
    assert all(len(x)==64 and all(c in "0123456789abcdef" for c in x)
               for x in result.values())
    return result


def test_original_protected_source_and_all_59_published_root_outcomes():
    assert digest((ROOT/"PROTOCOL.md").read_bytes())==PROTOCOL_SHA
    assert digest((ROOT/"experiment.py").read_bytes())==EXPERIMENT_SHA
    published_data()
    root_manifest()


def audit_original_zip(path:Path):
    raw=path.read_bytes()
    assert len(raw)==203951 and digest(raw)==ZIP_SHA
    with zipfile.ZipFile(path) as zipdata:
        expected={"FC02_FIRST_59_ROOT_AGGREGATE_RESULT.json"}
        expected.update(f"nc09_fc02_root_{root}.json" for root in range(13000,13059))
        assert set(zipdata.namelist())==expected
        summary_bytes=zipdata.read("FC02_FIRST_59_ROOT_AGGREGATE_RESULT.json")
        assert len(summary_bytes)==70271 and digest(summary_bytes)==SUMMARY_SHA
        s=json.loads(summary_bytes)
        assert s["study"]=="BPD-NC09-FC02"
        assert s["phase"]=="AGGREGATE_TANGENT_SHAM_CALIBRATION"
        assert s["evidence_class"]=="CALIBRATION_DEVELOPMENT_ONLY"
        assert s["registered_development_verdict"]=="CALIBRATION_INCOMPLETE"
        assert s["independent_model_roots"]==59 and s["eligible_roots"]==40
        assert s["aggregate_32_root_pseudo_null"] is None
        assert s["root_mean_bootstrap"] is None
        assert s["approximate_power"] is None
        assert s["rootwise_sham_max_95_95_diagnostic"] is None
        assert s["gates"] is None
        assert s["scientific_confirmation_roots_evaluated"]==0
        assert s["calibration_scientific_confirmation_ready"] is False
        assert s["original_FC01_verdict_unchanged"]=="CALIBRATION_REDESIGN_OR_PILOT_ONLY"
        roots={int(r["root"]):r for r in published_data()}
        shas=root_manifest()
        for root in range(13000,13059):
            b=zipdata.read(f"nc09_fc02_root_{root}.json")
            assert digest(b)==shas[root]
            r=json.loads(b)
            row=roots[root]
            assert r["root"]==root and r["study"]=="BPD-NC09-FC02"
            assert r["training"]["steps"]==384 and r["full_mask_closure"]["passed"]
            assert r["splits"]=={
                "train":2048,"derivative_discovery":1024,
                "model_validation":512,"labeled_calibration":1024
            }
            assert r["protocol_sha256"]==PROTOCOL_SHA
            assert r["source_sha256"]==EXPERIMENT_SHA
            assert r["parent_integrity"]["nc09_original_experiment_sha256"]==ORIGINAL_PARENT_SHA
            assert int(r["eligible"])==int(row["eligible"])
            assert r["sham_admission"]["proposals"]==4096
            assert r["sham_admission"]["qualified"]==int(row["qualified_shams"])
            assert r["sham_admission"]["selected"]==int(row["selected_shams"])
            assert r["selected_sham_score_source_only"] is True
            assert r["sham_admission"]["uses_external_labels"] is False
            if r["eligible"]:
                assert len(r["sham_nlls"])==8 and r["effects"] is not None
                assert abs(sum(r["effects"]["Z_pseudo_null_by_sham"]))<1e-10
                assert all(x<=.15+1e-12 for x in
                           r["sham_admission"]["selected_dose_relative_errors"])
                assert all(x<=r["sham_admission"]["signed_score_abs_tolerance"]+1e-12
                           for x in r["sham_admission"]["selected_score_mean_abs_errors"])
                assert all(x<=r["sham_admission"]["rms_score_abs_tolerance"]+1e-12
                           for x in r["sham_admission"]["selected_score_rms_abs_errors"])
            else:
                assert root in FAILED_ROOTS
                assert r["sham_admission"]["qualified"]<8
    print("PASS: protected first registered NC09-FC02 59-root admission FAIL, all complete raw receipts")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path)
    args=p.parse_args()
    test_original_protected_source_and_all_59_published_root_outcomes()
    if args.archive is not None:
        audit_original_zip(args.archive)
    print("PASS: original FC02 source and 59-root publication audit")


if __name__=="__main__":
    main()
