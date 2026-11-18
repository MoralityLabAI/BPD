"""NC07 independent original eight-root publication/immutable receipt auditor.

This post-result module reads frozen CSV and historical source files only.
Optional --archive checks the exact original GitHub Actions ZIP and its nine
uncompressed JSON members without running any model or choosing a new rank.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUN_ID=44260710669
ARTIFACT_ID=14408012541
ORIGINAL_ZIP_SHA256="55f831d8c7f37b2db8e7e0e6e79824b8f99a26d189f452a2dd642e3383d4dbce"
ORIGINAL_SUMMARY_SHA256="9c00149d8f6b12042d6e648eb56fc2d125e5d75358b6241864d3b4c982bbb2ab"
ORIGINAL_PROTOCOL_SHA256="dfd071c31c1e9bfcdcfc8cf5db3d82ae26948dcb8bf31d3d5b9059b9851ee603"
ORIGINAL_EXPERIMENT_SHA256="701cf5947fb3e75008b49d9b242cace43bc6b465570edddc103c1c8bf3a38115"
ROOT_HASHES={
    700:"122cfc244ade41a693029631ebcd86e42969c2369180e1fad5ab07fca1ec19d3",
    701:"21b391053cd5ac23a15b1505c858b2fcbf3f8c6852b62a2ada3259a322e3f917",
    702:"b2312d2a7f3d9ad817d5a6bacbe272ae19fad6b67e79469e63e9fd376dc362aa",
    703:"c7c2334ca2826e1bfd56edeb71a26ee59295b96",
    704:"9620ea6aac957bac221899668b1e41f57e68654a3c4b69c252c8a9b9a57b5e37",
    705:"2083cee2c5d721921d1503d3f84d6bf906eb1c88ebfd9387157d2e39e24ef1b6",
    706:"1a05659860ea8eb395fb59192c360678127b62a9397835c27565f40c3f0aa96e",
    707:"68722c8d288b67248fb6d889d6046b1f51021549af002e4f8fd95f48e5864784",
}
HEADER=(
    "root","selected_rank","fully_admitted","selected_shams",
    "qualified_sham_proposals","selector_final_kl","selector_p95_final_kl",
    "clean_nll","selected_nll","fixed_rank4_nll",
    "selected_keep_cost","selected_minus_fixed4","median_sham_advantage"
)


def _sha(data:bytes):
    return hashlib.sha256(data).hexdigest()


def original_table():
    with (ROOT/"PER_ROOT_20261118.csv").open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        assert tuple(reader.fieldnames)==HEADER
        rows=list(reader)
    assert len(rows)==8 and {int(r["root"]) for r in rows}==set(ROOT_HASHES)
    assert [int(r["selected_rank"]) for r in rows]==[3,4,3,4,6,3,6,3]
    assert sum(int(r["fully_admitted"]) for r in rows)==7
    assert sum(int(r["selected_rank"]) for r in rows)/8==4.
    assert sum(int(r["selected_rank"])<=3 for r in rows)==4
    assert [int(r["selected_shams"]) for r in rows]==[8,8,8,8,8,8,7,8]
    assert int(rows[6]["qualified_sham_proposals"])==7
    assert 0<float(rows[6]["selector_final_kl"])<.027
    for r in rows:
        assert abs((float(r["selected_nll"])-float(r["clean_nll"]))-
                   float(r["selected_keep_cost"]))<=1e-10
        assert abs((float(r["selected_nll"])-float(r["fixed_rank4_nll"]))-
                   float(r["selected_minus_fixed4"]))<=1e-10
    return rows


def test_publication_and_unchanged_first_source_hashes():
    assert _sha((ROOT/"PROTOCOL.md").read_bytes())==ORIGINAL_PROTOCOL_SHA256
    assert _sha((ROOT/"experiment.py").read_bytes())==ORIGINAL_EXPERIMENT_SHA256
    original_table()


def audit_full_original_archive(path:Path):
    raw=path.read_bytes()
    assert _sha(raw)==ORIGINAL_ZIP_SHA256
    with zipfile.ZipFile(path) as archive:
        expected={"NC07_ORIGINAL_DEVELOPMENT_SUMMARY.json"}
        expected.update("nc07_root_%d.json"%s for s in ROOT_HASHES)
        assert set(archive.namelist())==expected
        summary_bytes=archive.read("NC07_ORIGINAL_DEVELOPMENT_SUMMARY.json")
        assert _sha(summary_bytes)==ORIGINAL_SUMMARY_SHA256
        s=json.loads(summary_bytes)
        assert s["study"]=="BPD-NC-07" and s["phase"]=="DEVELOPMENT"
        assert s["evidence_class"]=="ENGINEERING_ONLY"
        assert s["registered_verdict"]=="ENGINEERING_NULL_OR_INELIGIBLE"
        assert s["eligible_roots"]==7 and s["independent_model_roots"]==8
        assert s["bootstrap"] is None and s["mean_selected_rank"] is None
        assert not any(s["gates"].values())
        assert s["selected_rank_counts"]=={"2":0,"3":4,"4":2,"6":2}
        assert s["calibration_roots_evaluated"]==0
        assert s["confirmation_roots_evaluated"]==0
        by={x["root"]:x for x in s["per_root"]}
        assert len(by)==8
        for row in original_table():
            root=int(row["root"])
            member=archive.read("nc07_root_%d.json"%root)
            assert _sha(member)==ROOT_HASHES[root]
            r=json.loads(member)
            assert r["root"]==root
            assert r["phase"]=="DEVELOPMENT" and r["study"]=="BPD-NC-07"
            assert r["protocol_sha256"]==ORIGINAL_PROTOCOL_SHA256
            assert r["source_sha256"]==ORIGINAL_EXPERIMENT_SHA256
            assert r["training"]["steps"]==384
            assert r["split_sizes"]=={
                "train":2048,"derivative_discovery":1024,
                "rank_selection":512,"final_heldout":1024,
            }
            assert r["full_mask_closure"]["passed"]
            assert r["selected_rank"]==int(row["selected_rank"])
            assert int(r["eligible"])==int(row["fully_admitted"])
            candidate=next((rank for rank in (2,3,4,6)
                if r["selector_by_rank"][str(rank)]["eligible"]),None)
            assert candidate==r["selected_rank"]
            assert r["selected_sham_admission"]["selected"]==int(row["selected_shams"])
            assert r["selected_sham_admission"]["qualified"]==int(row["qualified_sham_proposals"])
            assert abs(r["selector_by_rank"][str(candidate)]["mean_final_kl_nats"]-
                       float(row["selector_final_kl"]))<1e-10
            assert abs(r["heldout"]["selected"]["nll_nats"]-
                       float(row["selected_nll"]))<1e-10
            assert abs(r["median_sham_minus_selected_nll_nats"]-
                       float(row["median_sham_advantage"]))<1e-10
            assert by[root]["eligible"]==r["eligible"]
        print("PASS: full original NC07 eight-root NULL and all source-locked receipts")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path)
    args=p.parse_args()
    test_publication_and_unchanged_first_source_hashes()
    if args.archive:
        audit_full_original_archive(args.archive)
    print("PASS: NC07 published source hashes, selected rank and sham admissions")


if __name__=="__main__":
    main()
