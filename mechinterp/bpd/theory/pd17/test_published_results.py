"""Independent original BPD-PD-17 publication/first-256-arm receipt audit.

This file was added after first-run registration. It does not sample,
train or evaluate any registered scientific seed. Download the original
Actions artifact and use --receipt to audit raw original JSON as well.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SHA256="c75c16e4f901df4b6c0d6c97f92711fe7763555c8fe77402e146d5e6f918c321"
SOURCES={
    "CONFIG.json":"d2f13d622036aedf81170b8b6384841b02df8a39183402ce16a898f3416b3b9a",
    "PROTOCOL.md":"2254b6c2d34ea898ae90cab8b07faa17a6a12a9a2eea7a688fa5c739427ed6e8",
    "angular.py":"5b2734d80b89067faa1ee2a734c666aac6e015fdf79a8e92df1ca47ff0ddcbf1",
    "experiment.py":"b96bc09d0b248bf87dffbaaf7d0db46f32fa6eb364cfd42a0a867d378bef0e5b",
    "source.py":"2fc826448c8e9a558ce07c47ad2ce35a667c2ef4c12034e0a284e5e2304ab6b2"
}
LAWS=("original","radial_camouflage")
ARMS=("norm_givens","uniform_givens","angular_atoms","planted_Q_oracle")
EXPECTED={
    "original":{"norm_givens":32,"uniform_givens":0,"angular_atoms":32,"planted_Q_oracle":32},
    "radial_camouflage":{"norm_givens":0,"uniform_givens":0,"angular_atoms":32,"planted_Q_oracle":32}
}
HEADER=["seed","interaction","original_corr_K_norm2","camouflaged_corr_K_norm2",
        "source_independence_gate",*[law+"_"+a+"_success" for law in LAWS for a in ARMS]]


def rows():
    with (ROOT/"PER_ROOT_20261118.csv").open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f)
        assert r.fieldnames==HEADER
        data=list(r)
    frozen=json.loads((ROOT/"CONFIG.json").read_text(encoding="utf-8"))
    assert {int(x["seed"]) for x in data}==set(frozen["seeds"])
    assert len(data)==32 and all(float(r["interaction"])==1 for r in data)
    assert all(int(x["source_independence_gate"])==1 for x in data)
    for law in LAWS:
        for a in ARMS:
            key=law+"_"+a+"_success"
            assert sum(int(r[key]) for r in data)==EXPECTED[law][a]
    for r in data:
        assert float(r["original_corr_K_norm2"])>=.7
        assert abs(float(r["camouflaged_corr_K_norm2"]))<=.05
    return data


def test_original_source_hashes_and_paired_root_publication():
    for name,digest in SOURCES.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    rows()


def raw_archive(path:Path):
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SHA256
    d=json.loads(raw)
    assert d["source_hashes"]==SOURCES
    assert d["study_id"]=="BPD-PD-17" and len(d["arms"])==256
    assert len(d["diagnostics"])==32
    s=d["summary"]
    assert s["registered"] and s["registered_scientific_passed"]
    assert s["mean_root_interaction"]==1.
    assert s["root_bootstrap"]["lower95"]==s["root_bootstrap"]["upper95"]==1.
    assert s["independent_radial_source_gate_passes"]==32
    assert s["physical_oracles_admitted"]
    assert s["success_by_law_and_arm"]==EXPECTED
    by={(r["seed"],r["law"],r["arm"]):r for r in d["arms"]}
    assert len(by)==256
    for row in rows():
        seed=int(row["seed"])
        for law in LAWS:
            for arm in ARMS:
                r=by[(seed,law,arm)]
                assert int(r["success"])==int(row[law+"_"+arm+"_success"])
                assert set(r["heldout"]["per_family"])==set(json.loads(
                    (ROOT/"CONFIG.json").read_text())["families"])
                assert all(f["binary_mask"] for f in r["heldout"]["per_family"].values())
                if arm in ("angular_atoms","planted_Q_oracle") or (
                    law=="original" and arm=="norm_givens"
                ):
                    assert all(f["support_fraction"]==1.0 for f in
                               r["heldout"]["per_family"].values())
                    assert all(f["false_off"]==f["false_on"]==0 for f in
                               r["heldout"]["per_family"].values())
                if arm.endswith("_givens"):
                    t=r["training"]
                    assert t["candidate_angle_scores"]==39960
                    assert t["candidate_scalar_projection_evaluations"]==327352320
                    assert t["training_board_count"]==2048
                    assert t["pair_sweeps"]==9
                if arm=="angular_atoms":
                    assert r["training"]["angular_scalar_multiplications"]==1024*8192*16
    print("PASS original full PD17 256-arm source-locked receipt")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args()
    test_original_source_hashes_and_paired_root_publication()
    if args.receipt:raw_archive(args.receipt)
    print("PASS original PD17 source SHA and paired root publication")


if __name__=="__main__":
    main()
