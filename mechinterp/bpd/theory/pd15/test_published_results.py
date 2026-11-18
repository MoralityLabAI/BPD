"""Independent publication audit for BPD-PD-15's frozen first cohort.

Test the published CSV and unchanged original source SHA256 in standard CI.
For complete 160-arm raw JSON replay, pass --receipt <downloaded artifact>.
This module performs no optimization or evaluation on any scientific seed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from mechinterp.bpd.theory.pd02.experiment import exact_mcnemar_p

ROOT = Path(__file__).resolve().parent
ORIGINAL_RUN = 44233123380
ARTIFACT = 14396987094
ORIGINAL_JSON_SHA256 = (
    "94e47c61b9020a092cb2062e40681ef2e7b6693904ae42bdb5bdd196163015ee"
)
FROZEN_SOURCE_HASHES = {
    "CONFIG.json": "c8bdacee0ea1d6c48d5fd4c9de67a760f06048440095902d5d4bb342a957a0cd",
    "PROTOCOL.md": "445afce4039530d748b03e80ce7cba0acddd59578137ce98596c8ab7e7b17f8f",
    "experiment.py": "0f0dfc351e74510fe5195ba7f5c0f1694560a1d26d16248e4c4579e50363f10c",
}
ARMS = (
    "uniform_wide", "weighted_wide", "uniform_narrow",
    "weighted_narrow", "planted_Q_oracle"
)
EXPECTED = {
    "uniform_wide": 0, "weighted_wide": 32,
    "uniform_narrow": 10, "weighted_narrow": 27,
    "planted_Q_oracle": 32,
}
SECONDARY = {
    "weighted_narrow": {3037, 3121, 3229, 3251, 3253}
}
FAMILIES = (
    "k1_seen", "k2_seen", "k3_seen", "k4_seen",
    "k2_unseen", "k3_unseen", "k4_unseen", "transition_unseen"
)
HEADER = ["seed", *[a+"_success" for a in ARMS],
          *[a+"_cosine" for a in ARMS],
          *[a+"_k4_unseen" for a in ARMS], "objective_dual_sign"]


def published_csv() -> list[dict]:
    with (ROOT / "PER_SEED_20261118.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == HEADER
        rows = list(reader)
    cfg = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
    assert len(rows) == 32
    assert {int(x["seed"]) for x in rows} == set(cfg["seed_cell"]["seeds"])
    for arm, target in EXPECTED.items():
        assert sum(int(r[arm+"_success"]) for r in rows) == target
    for r in rows:
        assert int(r["objective_dual_sign"]) == 1
        for arm in ARMS:
            success = int(r[arm+"_success"])
            cos = float(r[arm+"_cosine"])
            k4 = float(r[arm+"_k4_unseen"])
            assert success in (0, 1)
            assert 0 <= cos <= 1.00001
            assert 0 <= k4 <= 1.00001
            if success:
                assert cos >= .9 and k4 >= .7
        assert int(r["planted_Q_oracle_success"]) == 1
        assert int(r["weighted_wide_success"]) == 1
        assert int(r["uniform_wide_success"]) == 0
        assert abs(float(r["weighted_wide_k4_unseen"]) - 1) <= 1e-12
    weighted_fails = {
        int(r["seed"]) for r in rows if not int(r["weighted_narrow_success"])
    }
    assert weighted_fails == SECONDARY["weighted_narrow"]
    return rows


def test_source_hashes_unchanged():
    for f, expected in FROZEN_SOURCE_HASHES.items():
        actual = hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
        assert actual == expected, (f, actual, expected)


def test_frozen_scientific_publication_and_pairing():
    rows = published_csv()
    aonly = sum(int(r["weighted_wide_success"]) == 1 and
                int(r["uniform_wide_success"]) == 0 for r in rows)
    bonly = sum(int(r["uniform_wide_success"]) == 1 and
                int(r["weighted_wide_success"]) == 0 for r in rows)
    assert aonly == 32 and bonly == 0
    assert abs(exact_mcnemar_p(aonly, bonly) - 2**-31) < 1e-15
    assert sum(int(r["objective_dual_sign"]) for r in rows) == 32


def audit_full_raw_receipt(path: Path) -> None:
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL_JSON_SHA256
    data = json.loads(raw)
    assert data["source_hashes"] == FROZEN_SOURCE_HASHES
    assert data["study_id"] == "BPD-PD-15"
    assert len(data["arms"]) == 160 and len(data["diagnostics"]) == 32
    summary = data["summary"]
    assert summary["registered"] and summary["units"] == 32 and summary["arms"] == 160
    assert summary["success_by_arm"] == EXPECTED
    assert summary["primary_only"] == 32 and summary["control_only"] == 0
    assert summary["net_primary_wins"] == 32
    assert summary["McNemar_exact_two_sided_p"] == 2**-31
    assert summary["source_objective_dual_sign_count"] == 32
    assert summary["objective_consistency_gate_passed"]
    assert summary["primary_mechanism_gate_passed"]
    assert summary["registered_scientific_passed"]
    assert len({(r["seed"], r["arm"]) for r in data["arms"]}) == 160
    by = {(r["seed"], r["arm"]): r for r in data["arms"]}
    dby = {d["seed"]: d for d in data["diagnostics"]}
    assert len(dby) == 32
    for row in published_csv():
        seed = int(row["seed"])
        diagnostic = dby[seed]
        assert diagnostic["dual_sign"]
        assert diagnostic["weights"]["none"]["remix_minus_planted"] > 0
        assert diagnostic["weights"]["norm1"]["remix_minus_planted"] < 0
        for arm in ARMS:
            r = by[(seed, arm)]
            assert r["study_id"] == "BPD-PD-15"
            assert int(r["success"]) == int(row[arm+"_success"])
            assert abs(r["heldout"]["alignment"]["mean_cosine"] -
                       float(row[arm+"_cosine"])) <= 1e-9
            assert abs(r["heldout"]["per_family"]["k4_unseen"]["support_fraction"] -
                       float(row[arm+"_k4_unseen"])) <= 1e-9
            held = r["heldout"]
            assert set(held["per_family"]) == set(FAMILIES)
            assert all(f["binary_mask"] for f in held["per_family"].values())
            assert held["overall_mse"] <= .035
            assert all(f["mse"] <= .05 for f in held["per_family"].values())
            if arm == "planted_Q_oracle":
                assert r["uses_true_Q_for_optimizer"]
                assert r["training"]["candidate_angle_scores"] == 0
            else:
                assert not r["uses_true_Q_for_optimizer"]
                t = r["training"]
                assert t["candidate_angle_scores"] == 39960
                assert t["candidate_scalar_projection_evaluations"] == 327352320
                assert t["training_board_count"] == 2048
                assert t["training_observed_vectors"] == 8192
                assert t["pair_sweeps"] == 9
                assert t["orthogonality_max_error"] < 1e-4
            if arm in ("weighted_wide", "planted_Q_oracle"):
                assert all(f["support_fraction"] == 1.0 for f in held["per_family"].values())
                assert all(f["false_off"] == f["false_on"] == 0
                           for f in held["per_family"].values())
    print("PASS: original first 160-arm source-locked JSON, SHA256, masks and outcomes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", type=Path)
    args = ap.parse_args()
    test_source_hashes_unchanged()
    test_frozen_scientific_publication_and_pairing()
    if args.receipt is not None:
        audit_full_raw_receipt(args.receipt)
    print("PASS: published PD15 first-run scientific PASS and immutable provenance")


if __name__ == "__main__":
    main()
