"""Recompute every AMM-01 case and verify a small committed outcome lock.

Full raw per-case traces are deterministic and can be emitted by experiment.py.
The lock pins their exact canonical JSON digest, all-arm totals, and analysis.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .experiment import bundle, digest, ID


def lock_for(run: dict, source_commit: str) -> dict:
    out = {"schema": "BPD-AMM-01/locked-full-cohort-digest/v1",
           "experiment": ID,
           "phase": run["phase"],
           "source_commit": source_commit,
           "source_code_sha256": run["source_code_sha256"],
           "training_memory_sha256": run["training_memory"]["sha256"],
           "independent_units": run["independent_units"],
           "scored_cases": run["total_scored_cases"],
           "all_arm_hits": run["totals"],
           "full_raw_canonical_sha256": digest(run)}
    if "analysis" in run:
        out["analysis"] = run["analysis"]
        out["evalmath_preflight_sha256"] = run["evalmath_preflight_sha256"]
    else:
        out["pilot_input_sha256"] = digest(run["evalmath_preflight_input"])
    return out


def verify(lock: dict, preflight: dict | None) -> dict:
    phase = lock["phase"]
    if phase not in ("development", "confirm_fresh"):
        raise AssertionError("unrecognized phase")
    run = bundle(phase, preflight)
    if lock != lock_for(run, lock["source_commit"]):
        raise AssertionError("AMM-01 locked source/results mismatch")
    return run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--preflight", type=Path)
    args = p.parse_args()
    lock = json.loads(args.lock.read_text())
    pf = json.loads(args.preflight.read_text()) if args.preflight else None
    run = verify(lock, pf)
    print(f"{ID} {run['phase']} full raw reconstruction PASS: {lock['full_raw_canonical_sha256']}")


if __name__ == "__main__":
    main()
