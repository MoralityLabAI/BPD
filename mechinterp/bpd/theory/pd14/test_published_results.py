"""Independent audit of the frozen BPD-PD-14 first-run receipt.

This file was added after the registered first run. It reads published
artifacts and never fits any of the registered sources.

For full raw-JSON verification, download original artifact #14392996069:
python -m mechinterp.bpd.theory.pd14.test_published_results --receipt first.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from mechinterp.bpd.theory.pd02.experiment import exact_mcnemar_p

ROOT = Path(__file__).resolve().parent
RECEIPT_SHA256 = "605f0433e58c2d6b77952d553c007ff3a623429a636a34624852ad0fff7b4a22"
SOURCE_SHA256 = {
    "CONFIG.json": "5610ba9604bfe8ae7c56561a47903f9f96e6b4e2cc7b350751e5d42f5a553dc3",
    "PROTOCOL.md": "1798858070b9ed1f5599f3affeeeee3b354e3a3fadb6441fbe2d8ccc8f8611aa",
    "experiment.py": "1975554b15c69970ff64bee1139e881471338e3eec1869055a950fc801ecdfec",
}
ARMS = ("wide_single", "narrow_single", "wide_restart",
        "narrow_restart", "planted_Q_oracle")
EXPECTED = {
    "clustered16d_strong": {
        "wide_single": 0, "narrow_single": 3, "wide_restart": 0,
        "narrow_restart": 0, "planted_Q_oracle": 16,
    },
    "clustered16d_weak": {
        "wide_single": 16, "narrow_single": 1, "wide_restart": 0,
        "narrow_restart": 0, "planted_Q_oracle": 16,
    },
}
SUCCESS_SEEDS = {"narrow_single": {2797, 2801, 2803, 2939}}
FIELDS = ["seed", "cell", *[a + "_success" for a in ARMS],
          *[a + "_cos" for a in ARMS],
          *[a + "_k4_unseen" for a in ARMS]]


def read_csv() -> list[dict]:
    with (ROOT / "PER_SEED_20261118.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDS
        rows = list(reader)
    assert len(rows) == 32
    assert len({int(x["seed"]) for x in rows}) == 32
    assert {x["cell"] for x in rows} == set(EXPECTED)
    assert Counter(x["cell"] for x in rows) == {
        "clustered16d_strong": 16, "clustered16d_weak": 16
    }
    return rows


def test_frozen_publication_and_paired_scientific_result():
    rows = read_csv()
    for cell, target in EXPECTED.items():
        subset = [r for r in rows if r["cell"] == cell]
        for arm, count in target.items():
            assert sum(int(r[arm + "_success"]) for r in subset) == count
        for row in subset:
            for arm in ARMS:
                success = int(row[arm + "_success"])
                cosine = float(row[arm + "_cos"])
                k4 = float(row[arm + "_k4_unseen"])
                assert success in (0, 1)
                assert 0 <= cosine <= 1.00001
                assert 0 <= k4 <= 1.00001
                if success:
                    assert cosine >= .9 and k4 >= .7
            assert int(row["planted_Q_oracle_success"]) == 1

    strong = [r for r in rows if r["cell"] == "clustered16d_strong"]
    n_only = sum(int(r["narrow_single_success"]) == 1 and
                 int(r["wide_single_success"]) == 0 for r in strong)
    w_only = sum(int(r["wide_single_success"]) == 1 and
                 int(r["narrow_single_success"]) == 0 for r in strong)
    assert n_only == 3 and w_only == 0
    assert abs(exact_mcnemar_p(n_only, w_only) - .25) < 1e-14
    assert {
        int(r["seed"]) for r in rows if int(r["narrow_single_success"])
    } == SUCCESS_SEEDS["narrow_single"]
    assert all(int(r["wide_restart_success"]) == 0 and
               int(r["narrow_restart_success"]) == 0 for r in rows)
    assert sum(int(r["wide_single_success"]) for r in rows) == 16


def test_first_run_source_files_remain_locked():
    for path, expected in SOURCE_SHA256.items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert actual == expected, (path, actual)


def audit_raw_receipt(path: Path) -> None:
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == RECEIPT_SHA256
    r = json.loads(data)
    assert r["source_hashes"] == SOURCE_SHA256
    assert r["study_id"] == "BPD-PD-14"
    assert len(r["arms"]) == 160 and len(r["diagnostics"]) == 32
    s = r["summary"]
    assert s["registered"] and s["units"] == 32 and s["arms"] == 160
    assert not s["registered_scientific_passed"]
    assert s["objective_consistency_gate_passed"]
    assert not s["primary_mechanism_gate_passed"]
    assert s["strong_objective_dual_sign_count"] == 16
    assert s["strong_narrow_only"] == 3 and s["strong_wide_only"] == 0
    assert s["strong_narrow_single_successes"] == 3
    assert abs(s["strong_McNemar_exact_two_sided_p"] - .25) < 1e-14
    assert s["success_by_cell"] == EXPECTED
    by = {(row["seed"], row["arm"]): row for row in r["arms"]}
    assert len(by) == 160
    for row in read_csv():
        seed = int(row["seed"])
        for arm in ARMS:
            x = by[(seed, arm)]
            assert x["cell"] == row["cell"]
            assert int(x["success"]) == int(row[arm + "_success"])
            assert abs(x["heldout"]["alignment"]["mean_cosine"] -
                       float(row[arm + "_cos"])) <= 1e-9
            assert abs(x["heldout"]["per_family"]["k4_unseen"]["support_fraction"] -
                       float(row[arm + "_k4_unseen"])) <= 1e-9
            assert set(x["heldout"]["per_family"]) == {
                "k1_seen", "k2_seen", "k3_seen", "k4_seen",
                "k2_unseen", "k3_unseen", "k4_unseen", "transition_unseen",
            }
            assert all(v["binary_mask"] for v in x["heldout"]["per_family"].values())
            assert x["heldout"]["overall_mse"] <= .035
            assert all(v["mse"] <= .05 for v in x["heldout"]["per_family"].values())
            if arm == "planted_Q_oracle":
                assert x["uses_true_Q_for_optimizer"]
                assert x["training"]["candidate_angle_scores"] == 0
            else:
                assert not x["uses_true_Q_for_optimizer"]
                assert x["training"]["candidate_angle_scores"] == 39960
                assert x["training"]["candidate_scalar_projection_evaluations"] == 327352320
                assert x["training"]["training_board_count"] == 2048
                assert x["training"]["candidate_sweeps_total"] == 9
                assert x["training"]["orthogonality_max_error"] < 1e-4
    print("PASS: original 160-arm registered JSON, SHA256, strict per-seed masks and PASS/FAIL")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    test_frozen_publication_and_paired_scientific_result()
    test_first_run_source_files_remain_locked()
    if args.receipt is not None:
        audit_raw_receipt(args.receipt)
    print("PASS: first-run published PD14 scientific FAIL and provenance")


if __name__ == "__main__":
    main()
