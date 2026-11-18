"""NC-08 independent original first-run publication and immutable ZIP auditor.

This module was added after the original eight-root source-locked run. It
never fits a projector, trains a model, or opens newly sampled model roots.
Provide --archive to verify full GitHub Actions original nine-member ZIP.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUN_ID=44265232267
ARTIFACT_ID=14409767606
ZIP_SHA256="533b150c6021093c1bf6d91d1152cce690eb3b09d357d641ec9de350aa9ca061"
SUMMARY_SHA256="daaf9d400147f862dd468e956d2950e794a4c350b6600e4139b289883cf21937"
PROTOCOL_SHA256="6289aa6914820c5d66fac999354f693fb15a94246cab56e7d3648985f5376d33"
EXPERIMENT_SHA256="6706366bcef169cc72a7961f1fba8a20fdd73a6e590f1ca93510b4aee7754e6c"
ROOT_SHA256={
    800:"62655de37228a43faae96471c319fbf4ea0a0b10b29ac2b27d834806aa9c6a0f",
    801:"c6b35a9e22aa1a75b2f7b96b2089ce8fc9b187e286237f00d55527751c99d47c",
    802:"2977eab81bd624237715e062a7deceb7d304f100e9bf925655abe80471e83e3b",
    803:"ba1d53ff7338b673fd0186b1a3c82ed7804102d31d8260d35849d74abd41a2de",
    804:"5a0a8d6b0b4833d1a1da86b7712e734fdb4bf6ea204251809af302ea08978a15",
    805:"daca7c61c60a7256cd049ada6bfe778cc902c79b63089d163d67a27ab18adb6c",
    806:"a0344e785fe750243817aeadee7ea4a5fb02a9e164e71b1d626b1004c272ff70",
    807:"4f4398d479121049fed4ce7414ac1538f10a8c9c9f5b5db1410f1206130ce2f6",
}
COLUMNS=(
    "root","selected_total_rank","pass0_rank","pass1_rank",
    "pass2_rank","pass3_rank","full_admission","selected_shams",
    "eligible_sham_proposals","source_only_selected_final_kl",
    "selected_keep_nll_cost","selected_minus_flat4_task_nll",
    "selected_median_sham_advantage","dp_budget12_minus_flat_rank3_task_nll"
)


def sha256(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def published_root_table():
    with (ROOT/"PER_ROOT_20261118.csv").open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        assert tuple(reader.fieldnames)==COLUMNS
        rows=list(reader)
    assert len(rows)==8
    assert {int(r["root"]) for r in rows}==set(ROOT_SHA256)
    assert all(int(r["full_admission"])==1 and int(r["selected_shams"])==8
               for r in rows)
    selected=[int(r["selected_total_rank"]) for r in rows]
    assert selected==[10,14,8,12,8,12,12,10]
    assert sum(selected)/8==10.75
    assert sum(b<=12 for b in selected)==7
    for r in rows:
        assert sum(int(r[f"pass{i}_rank"]) for i in range(4))==int(r["selected_total_rank"])
        assert int(r["eligible_sham_proposals"])>=8
    return rows


def test_frozen_first_run_source_and_original_rank_choices():
    assert sha256((ROOT/"PROTOCOL.md").read_bytes())==PROTOCOL_SHA256
    assert sha256((ROOT/"experiment.py").read_bytes())==EXPERIMENT_SHA256
    published_root_table()


def audit_original_zip(path:Path):
    assert sha256(path.read_bytes())==ZIP_SHA256
    with zipfile.ZipFile(path) as z:
        expect={"NC08_ORIGINAL_DEVELOPMENT_SUMMARY.json"}|{
            "nc08_root_%d.json"%r for r in ROOT_SHA256
        }
        assert set(z.namelist())==expect
        source=z.read("NC08_ORIGINAL_DEVELOPMENT_SUMMARY.json")
        assert sha256(source)==SUMMARY_SHA256
        s=json.loads(source)
        assert s["study"]=="BPD-NC-08"
        assert s["phase"]=="DEVELOPMENT"
        assert s["evidence_class"]=="ENGINEERING_ONLY"
        assert s["independent_model_roots"]==8 and s["admitted_roots"]==8
        assert s["original_development_verdict"]=="ENGINEERING_NULL_OR_INELIGIBLE"
        assert s["gates"]=={
            "absolute_external_task_fidelity":True,
            "all_eight_roots_admitted":True,
            "bounded_extra_task_cost_vs_flat_rank4":False,
            "source_matched_sham_specificity":True,
            "strict_total_rank_reduction":True,
        }
        assert s["budget_counts"]=={"8":2,"10":2,"12":3,"14":1}
        assert s["mean_selected_budget"]==10.75
        assert s["selected_budget_le12_roots"]==7
        assert s["independent_sham_calibration_roots_used"]==0
        assert s["independent_confirmation_roots_used"]==0
        assert not s["scientific_confirmation_performed"]
        assert abs(s["bootstrap"]["selected_minus_flat4"]["upper95"]-
                   .024799252278171462)<1e-12
        assert abs(s["bootstrap"]["keep_cost"]["mean"]-
                   .022712692618370056)<1e-12
        by={r["root"]:r for r in s["per_root"]}
        assert len(by)==8
        for row in published_root_table():
            root=int(row["root"])
            data=z.read("nc08_root_%d.json"%root)
            assert sha256(data)==ROOT_SHA256[root]
            rec=json.loads(data)
            assert rec["study"]=="BPD-NC-08"
            assert rec["root"]==root and rec["phase"]=="DEVELOPMENT"
            assert rec["protocol_sha256"]==PROTOCOL_SHA256
            assert rec["source_sha256"]==EXPERIMENT_SHA256
            assert rec["training"]["steps"]==384
            assert rec["splits"]=={
                "train":2048,"discovery":1024,"selection":512,"heldout":1024
            }
            assert rec["clean_closure"]["passed"]
            assert rec["eligible"]
            assert rec["selected_total_rank"]==int(row["selected_total_rank"])
            assert rec["selected_rank_tuple"]==[
                int(row[f"pass{i}_rank"]) for i in range(4)
            ]
            assert rec["sham_admission"]["selected"]==int(row["selected_shams"])==8
            assert rec["sham_admission"]["qualified"]==int(row["eligible_sham_proposals"])
            assert rec["sham_admission"]["proposals"]==2048
            assert len(rec["sham_heldout_nlls"])==8
            assert abs(rec["candidates_by_budget"][str(rec["selected_total_rank"])]
                       ["mean_final_kl_nats"]-
                       float(row["source_only_selected_final_kl"]))<1e-10
            for key,col in (
                ("keep_cost","selected_keep_nll_cost"),
                ("selected_minus_flat4","selected_minus_flat4_task_nll"),
                ("matched_sham_advantage","selected_median_sham_advantage"),
                ("dp_budget12_minus_flat3","dp_budget12_minus_flat_rank3_task_nll"),
            ):
                assert abs(by[root][key]-float(row[col]))<1e-10
            assert rec["gauge"]["relative_physical_operator_error"]<1e-4
            assert rec["gauge"]["max_full_rollout_logit_error"]<5e-5
            assert rec["heldout"]["flat_rank3"] is not None
            assert rec["heldout"]["flat_rank4"] is not None
        print("PASS: immutable first NC08 eight-root archive, source hashes, and original FAIL")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path)
    args=p.parse_args()
    test_frozen_first_run_source_and_original_rank_choices()
    if args.archive is not None:
        audit_original_zip(args.archive)
    print("PASS: original NC08 published source-only rank and per-root evidence")


if __name__=="__main__":
    main()
