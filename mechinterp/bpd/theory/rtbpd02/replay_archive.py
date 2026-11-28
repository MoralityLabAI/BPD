"""Read-only NumPy replay of the registration's local Nanbeige CAL archive."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from archive_lease import lease


HERE = Path(__file__).resolve().parent


def replay(repo, snapshot, registration, check):
    import numpy as np

    from archive_analysis import CompactWrite, archive_evalmath, calibrate, compact_closure, contrast, transfer
    from archive_checkpoint import checkpoint_replay
    from archive_inputs import load_inputs

    spec = registration["archive"]
    states, banks, custody, checkpoint, tokenizer_sha = load_inputs(repo, spec, check)
    affine, fixed_k, weights = checkpoint_replay(
        snapshot, spec["snapshot_revision"], checkpoint, tokenizer_sha, states["bf16"], check,
    )
    comparison, closures = {}, {}
    recipes = spec["passes"]
    for backend in spec["backends"]:
        comparison[backend] = {}
        target = CompactWrite(banks[backend]["t2"])
        closures[backend] = compact_closure(states[backend], banks[backend]["t2"], target)
        for p in recipes:
            is_fixed = (backend, p) == ("bf16", 1)
            result, roots = calibrate(
                states[backend][p], banks[backend], target, fixed_k if is_fixed else None, spec["ranks"], spec, check,
            )
            comparison[backend][str(p)] = result
            if is_fixed:
                planning = archive_evalmath(roots[2], registration["evalmath"])
    transfers = {}
    rtol = spec["support_relative_cutoff"]
    for p in recipes:
        transfers[f"pass_{p}_bf16_to_nf4_t2"] = transfer(
            banks["bf16"]["t2"], banks["nf4"]["t2"],
            contrast(states["bf16"][p].x), contrast(states["nf4"][p].x), rtol,
        )
    for backend in spec["backends"]:
        t1, t2 = banks[backend]["t1"], banks[backend]["t2"]
        a, b = [contrast(states[backend][p].x) for p in recipes]
        transfers[f"{backend}_pass_1_to_2_tied_t2"] = transfer(t2, t2, a, b, rtol)
        transfers[f"{backend}_pass_1_to_2_t1_then_t2"] = transfer(t1, t2, a, b, rtol)
    return {
        "schema": "rtbpd02-nanbeige-cal-archive/v1",
        "status": "completed_exposed_CAL_affine_and_fixed_reader_replay",
        "numpy_version": np.__version__,
        "root_count": spec["expected_roots"],
        "split": spec["fit_split"],
        "arm": spec["arm"],
        "denominators": "one complete A/B root per fold; no additional native trials",
        "pretrained_decoder_forwards": 0,
        "gpu_jobs": 0,
        "protected_ARG03_access": False,
        "archive_custody": custody,
        "checkpoint_custody": weights,
        "bf16_native_affine": affine,
        "fixed_readers": {"bf16/pass1": spec["fixed_reader"], "nf4": "exact native K projection and arithmetic receipt unavailable"},
        "t2_target_closure": closures,
        "leave_one_root_out": comparison,
        "paired_operator_state_transport": transfers,
    }, planning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="LoopedTransformers local archive root")
    parser.add_argument("--snapshot", type=Path, required=True, help="Pinned existing checkpoint snapshot")
    parser.add_argument("--out", type=Path, required=True, help="New output directory outside the two input trees")
    args = parser.parse_args()
    repo, snapshot, output = args.archive.resolve(), args.snapshot.resolve(), args.out.resolve()
    if output.exists() or output.is_relative_to(repo) or output.is_relative_to(snapshot):
        parser.error("use a new output directory outside the immutable archive and checkpoint")
    registration = (HERE / "EXTENSION_REGISTRATION.json").read_bytes()
    cfg = json.loads(registration)
    with lease(cfg["archive"]) as (check, samples):
        receipt, planner = replay(repo, snapshot, cfg, check)
        check()
    receipt["registration_sha256"] = sha256(registration).hexdigest()
    receipt["replay_source_sha256"] = {
        name: sha256((HERE / name).read_bytes()).hexdigest() for name in (
            "replay_archive.py", "archive_lease.py", "archive_inputs.py", "archive_checkpoint.py", "archive_analysis.py",
            "controls.py", "transport.py", "weighted_rank.py",
        )
    }
    receipt["resource_receipt"] = {
        "single_thread": True,
        "min_sampled_available_physical_bytes": min(a for a, _, _ in samples),
        "max_sampled_process_private_bytes": max(p for _, p, _ in samples),
        "elapsed_seconds": round(max(s for _, _, s in samples), 3),
        "sample_count": len(samples),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, content in (("archive_aggregate.json", receipt), ("archive_evalmath_input.json", planner)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as f:
            json.dump(content, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
    print(json.dumps({
        "status": receipt["status"], "roots": receipt["root_count"],
        "affine_case_writes": receipt["bf16_native_affine"]["case_writes"],
        "affine_bit_exact_case_writes": receipt["bf16_native_affine"]["bit_exact_case_writes"],
        "peak_sampled_private_bytes": receipt["resource_receipt"]["max_sampled_process_private_bytes"],
        "elapsed_seconds": receipt["resource_receipt"]["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
