"""Independent NC06 first-eight-root publication and immutable ZIP receipt audit.

This file was added after first development. It never trains a model or
regenerates selected heldout examples. Pass --archive to inspect original
immutable GitHub Actions artifact #14405931779.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ZIP_SHA="a5042b13d6aab84bf990baf1ed411e457f5a6795457f5f64a3deea7a988fd95c"
PROTO_SHA="8f4ab148445d225914a44d8b79db22dfd8b47f1d46bcdc0041ffb708e075d7de"
SOURCE_SHA="7b5f9946b83aa8e837c2ade2e744aaaaa544b2ee100495e1b7285c503462d6bd"
SUMMARY_SHA="7d58798e80b7d837d7a18b1158b4d1876e821ee27f667b55a591de236ae6e1c2"
ROOT_SHAS={
  600:"44fc25a142bcbd2f812e2a53ffe5d8b1ec8ad93c4f3ecb2548b2bfddd6e70055",
  601:"c43e683498fb5e7c3b764ed066e01a3f598e64e0f0cede483e8f8e5695eb016e",
  602:"297fdac089a34a587cd5f03102262a180f5e5d6337bb65b152421b84076896ed",
  603:"40712b482966978bb57b22a6e41c26e7adea080b9d63d4d5c9f9567c07b0d9cf",
  604:"3ef092bd956b220352fa94f554c505ec21bb1d73fe7e584cdaff9bce6e759931",
  605:"bd1ac327d74e82ba1798593e038143091a0fe509f37b0fe157a00cf31ca876cf",
  606:"b9082cfd4f1ad892e239ea86dc811c6da0d0da40be12c6f4c4d1c58bfa7fcdc3",
  607:"81581ee95f9dfa51cb3eb0d73ed5b2bedeffc10f25c79af03172939cbf6ba0a9",
}
HEADER=("root","clean_nll","norm_r3_minus_clean_nll",
        "uniform_r3_minus_norm_r3_nll","shuffled_r3_minus_norm_r3_nll",
        "norm_r3_sham_advantage","uniform_r4_minus_clean_nll",
        "all_primary_arms_eligible")


def per_root():
    with (ROOT/"PER_ROOT_20261118.csv").open(newline="") as f:
        read=csv.DictReader(f)
        assert tuple(read.fieldnames)==HEADER
        rows=list(read)
    assert len(rows)==8 and {int(r["root"]) for r in rows}==set(ROOT_SHAS)
    assert all(int(r["all_primary_arms_eligible"])==1 for r in rows)
    assert abs(sum(float(r["norm_r3_minus_clean_nll"]) for r in rows)/8
               -.02115786913782358)<1e-9
    assert abs(sum(float(r["uniform_r3_minus_norm_r3_nll"]) for r in rows)/8
               -.0003573615103960037)<1e-9
    assert abs(sum(float(r["shuffled_r3_minus_norm_r3_nll"]) for r in rows)/8
               -.0003364579752087593)<1e-9
    return rows


def test_original_source_hashes_and_published_root_null():
    assert hashlib.sha256((ROOT/"PROTOCOL.md").read_bytes()).hexdigest()==PROTO_SHA
    assert hashlib.sha256((ROOT/"experiment.py").read_bytes()).hexdigest()==SOURCE_SHA
    per_root()


def audit_original_zip(path:Path):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==ZIP_SHA
    with zipfile.ZipFile(path) as f:
        assert set(f.namelist())=={
            "NC06_DEVELOPMENT_SUMMARY.json",
            *(f"nc06_root_{i}.json" for i in ROOT_SHAS)
        }
        summary_raw=f.read("NC06_DEVELOPMENT_SUMMARY.json")
        assert hashlib.sha256(summary_raw).hexdigest()==SUMMARY_SHA
        summary=json.loads(summary_raw)
        assert summary["study"]=="BPD-NC-06"
        assert summary["phase"]=="ENGINEERING"
        assert summary["source_frozen"]
        assert summary["model_roots"]==8
        assert summary["registered_verdict"]=="ENGINEERING_NULL_OR_INELIGIBLE"
        assert not summary["frozen_development_positive"]
        assert summary["gates"]=={
            "all_roots_eligible":True,
            "norm_rank3_absolute_fidelity":True,
            "norm_beats_uniform_rank3":False,
            "norm_beats_shuffled_rank3":False,
            "norm_beats_matched_sham":True
        }
        assert summary["confirmation_roots_evaluated"]==0
        assert summary["sham_calibration_roots_evaluated"]==0
        by={r["root"]:r for r in summary["per_root"]}
        for row in per_root():
            root=int(row["root"])
            raw=f.read(f"nc06_root_{root}.json")
            assert hashlib.sha256(raw).hexdigest()==ROOT_SHAS[root]
            doc=json.loads(raw)
            assert doc["root"]==root and doc["eligible"]
            assert doc["training"]["steps"]==384
            assert doc["discovery_examples"]==doc["evaluation_examples"]==1024
            assert doc["protocol_sha256"]==PROTO_SHA
            assert doc["source_sha256"]==SOURCE_SHA
            for arm in ("norm_r3","uniform_r3","shuffle_r3"):
                assert doc["methods"][arm]["status"]=="ELIGIBLE"
                assert doc["methods"][arm]["sham_admission"]["selected"]==8
                assert doc["methods"][arm]["gauge"]["max_logit_error"]<=5e-5
                assert doc["methods"][arm]["gauge"]["relative_physical_operator_error"]<=1e-4
            assert abs(by[root]["norm_keep_cost"]-
                       float(row["norm_r3_minus_clean_nll"]))<=1e-9
        print("PASS independent original NC06 eight-root frozen evidence and NULL")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path)
    args=p.parse_args()
    test_original_source_hashes_and_published_root_null()
    if args.archive:audit_original_zip(args.archive)
    print("PASS original NC06 eight-root publication without retraining")


if __name__=="__main__":
    main()
