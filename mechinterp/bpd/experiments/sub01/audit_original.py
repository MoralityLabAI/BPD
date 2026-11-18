"""Read-only audit of the original BPD-SUB-01 eight-root artifact.

This script does not train or score models. It checks all raw per-root records
against the preregistration and returns a report including all ineligible roots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RANKS = (2, 4, 6, 8, 10)
ROOTS = tuple(range(1200, 1208))
ORIGINAL_RUN_ID = 44266927855
ORIGINAL_ARTIFACT_ID = 14409764910


def audit(receipt: dict) -> dict:
    roots = receipt["roots"]
    summary = receipt["summary"]
    assert summary["study"] == "BPD-SUB-01"
    assert summary["phase"] == "ENGINEERING"
    assert summary["units"] == 8
    assert summary["root_ids"] == list(ROOTS)
    assert [row["root"] for row in roots] == list(ROOTS)
    assert len({r["root"] for r in roots}) == len(ROOTS)
    for key in ("source_sha256", "protocol_sha256"):
        assert len({r[key] for r in roots}) == 1
        assert roots[0][key] == summary[key]
    assert hashlib.sha256((ROOT / "PROTOCOL.md").read_bytes()).hexdigest() == summary["protocol_sha256"]
    assert hashlib.sha256((ROOT / "experiment.py").read_bytes()).hexdigest() == summary["source_sha256"]
    detail = []
    valid_effects = []
    for row in roots:
        assert row["phase"] == "ENGINEERING"
        assert row["training"]["training_steps"] == 256
        assert row["data_sizes"] == {"train": 2048, "discovery": 512,
                                     "selection": 512, "heldout": 1024}
        assert tuple(int(x) for x in row["selection"]) == RANKS
        assert tuple(int(x) for x in row["all_ranks"]) == RANKS
        chosen = [
            rank for rank in RANKS
            if row["selection"][str(rank)]["clean_to_projected_kl_nats"] <= .025
            and row["selection"][str(rank)]["fiber"]["terminal_logit_rms"] <= .050
        ]
        expected_rank = min(chosen) if chosen else None
        assert row["selected_rank"] == expected_rank
        assert row["geometry"]["nested_max_abs"] <= 1e-5
        g = row["gauge"]
        gauge_ok = (
            g["max_clean_logit_error"] <= 5e-5
            and g["max_transported_impulse_logit_error"] <= 5e-5
            and g["max_clean_state_error"] <= 5e-5
            and g["max_intervened_state_error"] <= 5e-5
            and g["untransported_impulse_logit_rms"] > 1e-5
        )
        assert row["gauge_ok"] == gauge_ok
        dose_ok = True
        for rank in RANKS:
            arm = row["all_ranks"][str(rank)]
            assert len(arm["eight_same_dose_sham_logit_rms"]) == 8
            assert all(math.isfinite(y) for y in arm["eight_same_dose_sham_logit_rms"])
            f = arm["fiber"]
            if f["max_dose_deviation"] > 5e-6 or f["max_null_projection"] > 5e-6:
                dose_ok = False
            assert f["physical_dose"] == row["impulse_norm"]
            original_effect = (statistics.median(arm["eight_same_dose_sham_logit_rms"])
                               - f["terminal_logit_rms"])
            assert abs(original_effect - arm["median_sham_minus_pca_logit_rms"]) < 1e-9
        assert row["dose_ok"] == dose_ok
        competent = min(row["contexts"].values()) >= .80
        assert row["competent"] == competent
        eligible = bool(expected_rank is not None and competent and gauge_ok
                        and dose_ok and row["geometry"]["nested_max_abs"] <= 1e-5)
        assert row["root_eligible"] == eligible
        effect = None
        fiber = None
        clean_kl = None
        if expected_rank is not None:
            selected = row["all_ranks"][str(expected_rank)]
            assert selected == row["selected_heldout"]
            effect = selected["median_sham_minus_pca_logit_rms"]
            fiber = selected["fiber"]["terminal_logit_rms"]
            clean_kl = selected["clean_to_projected_kl_nats"]
        else:
            assert row["selected_heldout"] is None
        if eligible:
            valid_effects.append(effect)
        detail.append({
            "root": row["root"], "training_accuracy": row["training"]["train_accuracy"],
            "context_accuracies": row["contexts"], "competent": competent,
            "selected_rank": expected_rank, "eligible": eligible,
            "gauge_ok": gauge_ok, "dose_ok": dose_ok,
            "clean_gauge_max_error": g["max_clean_logit_error"],
            "counterfactual_gauge_max_error": g["max_transported_impulse_logit_error"],
            "mismatched_gauge_rms": g["untransported_impulse_logit_rms"],
            "selected_heldout_clean_kl_nats": clean_kl,
            "selected_heldout_fiber_rms": fiber, "selected_sham_gain_rms": effect,
        })
    assert len(valid_effects) == summary["eligible_units"]
    assert all(abs(a-b) < 1e-9 for a, b in zip(valid_effects, summary["root_effects"]))
    if valid_effects:
        assert abs(statistics.mean(valid_effects)
                   - summary["mean_sham_minus_pca_logit_rms"]) < 1e-9
    else:
        assert summary["mean_sham_minus_pca_logit_rms"] is None
    assert summary["verdict"] in (
        "ENGINEERING_SIGNAL_ONLY", "ENGINEERING_NULL_OR_INELIGIBLE")
    assert summary["scientific_sham_tail_calibrated"] is False
    return {
        "study": "BPD-SUB-01",
        "source_run": ORIGINAL_RUN_ID,
        "source_artifact": ORIGINAL_ARTIFACT_ID,
        "source_receipt_generated_before_cli_print_failure": True,
        "source_sha256": summary["source_sha256"],
        "protocol_sha256": summary["protocol_sha256"],
        "units": 8,
        "eligible": len(valid_effects),
        "selected_rank_distribution": {
            str(rank): sum(r["selected_rank"] == rank for r in detail)
            for rank in RANKS
        },
        "none_selected": sum(r["selected_rank"] is None for r in detail),
        "mean_eligible_effect": summary["mean_sham_minus_pca_logit_rms"],
        "whole_root_bootstrap95": summary["whole_root_bootstrap95"],
        "registered_verdict": summary["verdict"],
        "scientific_status": "ENGINEERING_ONLY_NO_INDEPENDENT_SHAM_TAIL",
        "root_detail": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    report = audit(json.loads(raw))
    report["raw_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
