"""Frozen BPD-SF04 exploratory gates. Failures are recorded, never hidden.

This script checks a JSON receipt without retraining or modifying the teacher.
An Actions job may succeed while science gates fail, provided results persist.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

FRESH = (1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367)


def decide(obj):
    rows = obj["rows"]
    if tuple(r["seed"] for r in rows) != FRESH or len(rows) != 8:
        raise ValueError("frozen eight-root roster mismatch")
    detail = []
    for r in rows:
        fields = (
            "teacher_mae", "source_blind_oracle_mae", "physical_factor_mse",
            "partial_recovery", "full_factor_recovery", "full_dark_recovery",
            "signed_tangent_sham_mean", "public_partial_abs_change",
            "gauge_full_max_error", "immediate_readout_error",
            "max_sham_norm_error", "max_sham_tangent_error",
        )
        if any(not math.isfinite(r[k]) for k in fields):
            raise ValueError(f"nonfinite scientific receipt at seed {r['seed']}")
        if r["generated"] != 512:
            raise ValueError(f"wrong source sample count at seed {r['seed']}")
        engineering = (
            r["teacher_mae"] <= .08
            and r["physical_factor_mse"] <= 1e-4
            and r["gauge_full_max_error"] <= 2e-5
            and r["immediate_readout_error"] <= 1e-6
            and r["max_sham_norm_error"] <= 1e-4
            and r["max_sham_tangent_error"] <= 1e-4
        )
        provenance = (
            r["authorized_eligible"] >= 100
            and r["teacher_mae"] <= .08
            and r["teacher_mae"] <= .6 * r["source_blind_oracle_mae"]
            and r["full_factor_recovery"] >= .90
            and r["full_dark_recovery"] >= .90
        )
        partial = (
            r["authorized_eligible"] >= 100
            and r["partial_recovery"] >= .60
            and (r["partial_recovery"] - r["signed_tangent_sham_mean"]) >= .20
            and r["public_partial_abs_change"] <= .08
        )
        detail.append({
            "seed": r["seed"], "authorized_eligible": r["authorized_eligible"],
            "engineering_pass": bool(engineering),
            "provenance_pass": bool(provenance),
            "partial_routing_pass": bool(partial),
            "partial_recovery": r["partial_recovery"],
            "signed_sham_mean": r["signed_tangent_sham_mean"],
            "public_partial_abs_change": r["public_partial_abs_change"],
        })
    counts = {
        "engineering": sum(d["engineering_pass"] for d in detail),
        "provenance": sum(d["provenance_pass"] for d in detail),
        "partial_routing": sum(d["partial_routing_pass"] for d in detail),
    }
    return {
        "evidence_class": "synthetic eight-root exploratory",
        "root_count": 8,
        "engineering_gate_passed": counts["engineering"] == 8,
        "provenance_learning_gate_passed": counts["provenance"] >= 6,
        "partial_routing_gate_passed": counts["partial_routing"] >= 6,
        "root_pass_counts": counts,
        "roots": detail,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("receipt", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    obj = json.loads(args.receipt.read_text())
    verdict = decide(obj)
    result = json.dumps(verdict, indent=2)
    print(result)
    if args.out:
        args.out.write_text(result + "\n")
