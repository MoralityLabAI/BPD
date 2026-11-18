"""Independent publication auditor for PD16 first original 128-arm cohort.

CI audits frozen source hashes, all published per-root outcomes and paired
McNemar. Supply --receipt with original GitHub Actions uncompressed JSON to
check all 128 native physical receipts without running new model seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from mechinterp.bpd.theory.pd02.experiment import exact_mcnemar_p

ROOT=Path(__file__).resolve().parent
EXPECTED={"norm_aligned":32,"norm_shuffled":0,"uniform_control":0,
          "planted_Q_oracle":32}
ARMS=tuple(EXPECTED)
SOURCES={
    "CONFIG.json":"c5aa3f24176552676ca2f9c6475a3ff866a03d21031dfdc22eb872bb114c703e",
    "PROTOCOL.md":"f36e3977deaf32f2a19808e1e0d5d1c767f7c5638d0ada7fcc6a7eef17904a1d",
    "experiment.py":"ea0a15b0d9602810e20bc354b6c8431234322ff1edf88ed2a3e39bbd747433ce",
}
ORIGINAL_SHA="fc8d6ccdc12dad07fe05723e5de1ee23da7c5576fef8fa23a4e10021763839e4"
HEADER=["seed",*[a+"_success" for a in ARMS],
        *[a+"_cos" for a in ARMS],*[a+"_k4_unseen" for a in ARMS],
        "diagnostic_passed"]


def table():
    with (ROOT/"PER_SEED_20261118.csv").open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==HEADER
        rows=list(reader)
    seeds=set(json.loads((ROOT/"CONFIG.json").read_text())["seed_cell"]["seeds"])
    assert len(rows)==32 and {int(r["seed"]) for r in rows}==seeds
    for arm,count in EXPECTED.items():
        assert sum(int(r[arm+"_success"]) for r in rows)==count
    assert all(int(r["diagnostic_passed"])==1 for r in rows)
    assert all(float(r["norm_aligned_k4_unseen"])==1. for r in rows)
    assert all(float(r["norm_shuffled_k4_unseen"])==0. for r in rows)
    for r in rows:
        for arm in ARMS:
            assert int(r[arm+"_success"]) in (0,1)
            assert 0<=float(r[arm+"_cos"])<=1.000001
            assert 0<=float(r[arm+"_k4_unseen"])<=1
    assert exact_mcnemar_p(32,0)==2**-31
    return rows


def test_original_source_hashes_and_all_32_paired_seed_outcomes():
    for filename,expected in SOURCES.items():
        actual=hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
        assert actual==expected,(filename,actual)
    table()


def original_receipt(path:Path):
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA
    r=json.loads(raw)
    assert r["study_id"]=="BPD-PD-16"
    assert r["source_hashes"]==SOURCES
    assert len(r["arms"])==128 and len(r["diagnostics"])==32
    s=r["summary"]
    assert s["registered"] and s["registered_scientific_passed"]
    assert s["primary_mechanism_gate_passed"]
    assert s["norm_cardinality_diagnostic_gate_passed"]
    assert s["norm_cardinality_diagnostic_passes"]==32
    assert s["success_by_arm"]==EXPECTED
    assert s["primary_only"]==32 and s["control_only"]==0
    assert s["McNemar_exact_two_sided_p"]==2**-31
    by={(x["seed"],x["arm"]):x for x in r["arms"]}
    assert len(by)==128
    for row in table():
        seed=int(row["seed"])
        for arm in ARMS:
            x=by[(seed,arm)]
            assert int(x["success"])==int(row[arm+"_success"])
            assert abs(x["heldout"]["alignment"]["mean_cosine"]-
                       float(row[arm+"_cos"]))<=1e-9
            assert abs(x["heldout"]["per_family"]["k4_unseen"]["support_fraction"]-
                       float(row[arm+"_k4_unseen"]))<=1e-9
            assert all(v["binary_mask"] for v in x["heldout"]["per_family"].values())
            if arm in ("norm_aligned","planted_Q_oracle"):
                assert all(v["support_fraction"]==1. for v in x["heldout"]["per_family"].values())
                assert all(v["false_off"]==v["false_on"]==0 for v in x["heldout"]["per_family"].values())
            elif arm=="norm_shuffled":
                assert x["training"]["candidate_angle_scores"]==39960
                assert x["training"]["permutation_sha256"]
    print("PD16 original 128-arm scientific PASS and full raw source archive PASS")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--receipt",type=Path)
    args=ap.parse_args()
    test_original_source_hashes_and_all_32_paired_seed_outcomes()
    if args.receipt:
        original_receipt(args.receipt)
    print("PASS PD16 independently published 32 source roots")


if __name__=="__main__":
    main()
