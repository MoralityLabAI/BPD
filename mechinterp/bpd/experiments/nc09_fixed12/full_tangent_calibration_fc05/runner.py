"""Execute the original excluded smoke or registered calibration exactly once."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import traceback
from pathlib import Path

import numpy as np
import scipy
import torch

from .experiment import (
    ROOT, ROOTS, SMOKE, PHASE, PROTO_SHA, evaluate_calibration_root,
    protect_source_manifest,
)

PROTOCOL_COMMIT = "8eb34f3533591513b46f14b902721d7f78bc0d43"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_lock(phase: str) -> dict:
    filename = "FULL_SMOKE_LOCK.json" if phase == "smoke" else "EXECUTE_LOCK.json"
    lock = json.loads((ROOT / filename).read_text())
    expected_phase = (
        "FIRST_EXCLUDED_FULL_SMOKE" if phase == "smoke"
        else "FIRST_ORIGINAL_59_ROOT_CALIBRATION"
    )
    assert lock["study"] == "BPD-NC09-FC05"
    assert lock["phase"] == expected_phase
    assert lock["protocol_commit_sha"] == PROTOCOL_COMMIT
    assert lock["excluded_smoke_root"] == SMOKE == 15999
    assert lock["model_roots"] == ([] if phase == "smoke" else list(ROOTS))
    assert lock["scientific_confirmation_models_evaluated"] == 0
    assert lock["source_manifest_sha256"] == sha_file(ROOT / "SOURCE_MANIFEST.json")
    assert lock["budgets"] == {
        "steps": 384, "discovery_n": 1024, "selector_n": 512,
        "heldout_n": 1024, "shams": 8, "solver_starts": 32,
        "solver_steps": 64,
    }
    repo = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip())
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for name, digest in manifest["files"].items():
        assert sha_file(repo / name) == digest, name
    protected = list(manifest["files"]) + [
        str((ROOT / "SOURCE_MANIFEST.json").relative_to(repo))
    ]
    for ancestor in (PROTOCOL_COMMIT, lock["source_commit_sha"]):
        subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, "HEAD"], check=True)
    subprocess.run([
        "git", "diff", "--exit-code", lock["source_commit_sha"], "HEAD", "--", *protected
    ], check=True)
    return lock


def compare_frozen_sampler(record: dict, source_file: Path) -> dict:
    from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_fc04.solver import (
        fit_full_tangent_shams,
    )
    with np.load(source_file, allow_pickle=False) as arrays:
        accepted, receipt = fit_full_tangent_shams(
            torch.from_numpy(arrays["a"].copy()),
            torch.from_numpy(arrays["U"].copy()),
            torch.from_numpy(arrays["z"].copy()),
            torch.from_numpy(arrays["g"].copy()),
            # Frozen passwise_source_candidates slices column-major eigh
            # eigenvectors. NPZ stores logical values; restore the live stride.
            [torch.from_numpy(np.asfortranarray(arrays[f"E_{t}"])) for t in range(4)],
            SMOKE,
        )
        instrumented = arrays["sham_projectors"].copy()
    expected = record["sham_admission"]
    selected_hashes = [r["projector_sha256"] for r in receipt["projectors"]]
    recorded_hashes = [r["projector_sha256"] for r in expected["projectors"]]
    assert selected_hashes == recorded_hashes
    assert len(accepted) == len(instrumented) == 8
    for p, q in zip(accepted, instrumented):
        assert np.array_equal(p.detach().cpu().numpy(), q)
    counters = (
        "selected", "charts_actually_started", "total_jacobian_corrections",
        "jacobian_rank_census", "root_seed", "full_tangent_parameter_count",
    )
    assert all(receipt[key] == expected[key] for key in counters)
    return {
        "passed": True, "exact_selected_projector_bytes_equal": True,
        "sampler": "frozen_FC04_89fb5446d2734ba3c34b765354ebfd1ea9e3eba4",
        "projector_sha256_sequence": selected_hashes,
        "compared_counters": {key: receipt[key] for key in counters},
        "new_model_trainings_in_comparison": 0,
    }


def execute(phase: str, out_dir: Path, run_url: str) -> int:
    lock = verify_lock(phase)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError("Original run directory must be empty; refusing a replay")
    manifest_sha = sha_file(ROOT / "SOURCE_MANIFEST.json")
    source_integrity = protect_source_manifest()
    original_environment = {
        "study": "BPD-NC09-FC05", "phase": phase, "run_url": run_url,
        "execution_commit_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_commit_sha": lock["source_commit_sha"],
        "protocol_commit_sha": PROTOCOL_COMMIT,
        "protocol_sha256": PROTO_SHA,
        "source_manifest_sha256": manifest_sha,
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "scientific_confirmation_models_evaluated": 0,
    }
    write_json(out_dir / "ORIGINAL_ENVIRONMENT.json", original_environment)
    write_json(out_dir / "SOURCE_MANIFEST.json", json.loads((ROOT / "SOURCE_MANIFEST.json").read_text()))
    write_json(out_dir / "EXECUTION_LOCK.json", lock)
    roots = (SMOKE,) if phase == "smoke" else ROOTS
    failures = []
    for seed in roots:
        path = out_dir / f"nc09_fc05_root_{seed}.json"
        record = None
        try:
            record = evaluate_calibration_root(seed, out_dir=out_dir)
            json.dumps(record, allow_nan=False)
        except Exception as error:
            failures.append(seed)
            diagnostic = None
            if record is not None:
                diagnostic = f"nc09_fc05_root_{seed}_serialization_failure.txt"
                (out_dir / diagnostic).write_text(
                    json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"
                )
            record = {
                "study": "BPD-NC09-FC05", "phase": PHASE,
                "root": seed, "eligible": False, "runtime_failure": True,
                "failure_type": type(error).__name__, "failure_message": str(error),
                "traceback": traceback.format_exc(), "protocol_sha256": PROTO_SHA,
                "source_sha256": sha_file(ROOT / "experiment.py"),
                "source_manifest_sha256": manifest_sha,
                "source_integrity": source_integrity,
                "parent_integrity": source_integrity,
                "failure": {"type": type(error).__name__, "message": str(error),
                            "traceback": traceback.format_exc(),
                            "returned_record_diagnostic": diagnostic},
                "scientific_confirmation_models_evaluated": 0,
            }
        record["original_run_url"] = run_url
        write_json(path, record)
        # The aggregate audits the persisted JSON schema, including rank lists.
        record = json.loads(path.read_text())
        print(json.dumps({
            "root": seed, "eligible": record.get("eligible", False),
            "runtime_failure": record.get("runtime_failure", False),
            "selected": (record.get("sham_admission") or {}).get("selected", 0),
            "source_audit": (record.get("independent_source_audit") or {}).get("passed"),
            "task_audit": (record.get("independent_task_audit") or {}).get("passed"),
            "root_json_sha256": sha_file(path),
        }), flush=True)
        if phase == "smoke" and not record.get("runtime_failure"):
            from .aggregate import technical_admission_errors
            technical_errors = technical_admission_errors(record)
            try:
                comparison = compare_frozen_sampler(
                    record, out_dir / record["source_archive"]["file"]
                )
            except Exception as error:
                comparison = {
                    "passed": False, "error_type": type(error).__name__,
                    "error": str(error), "traceback": traceback.format_exc(),
                }
            passed = bool(
                record.get("eligible") and comparison["passed"]
                and record["independent_source_audit"]["passed"]
                and record["independent_task_audit"]["passed"]
                and not technical_errors
            )
            write_json(out_dir / "FULL_SMOKE_RESULT.json", {
                "study": "BPD-NC09-FC05", "root": SMOKE, "passed": passed,
                "frozen_sampler_agreement": comparison,
                "technical_admission_errors": technical_errors,
                "root_json_sha256": sha_file(path),
                "source_manifest_sha256": manifest_sha,
                "scientific_confirmation_models_evaluated": 0,
            })
            if not passed:
                failures.append(seed)
        elif phase == "smoke":
            write_json(out_dir / "FULL_SMOKE_RESULT.json", {
                "study": "BPD-NC09-FC05", "root": SMOKE, "passed": False,
                "failure": record["failure"], "root_json_sha256": sha_file(path),
                "source_manifest_sha256": manifest_sha,
                "scientific_confirmation_models_evaluated": 0,
            })
    write_json(out_dir / "ORIGINAL_RUN_COMPLETION.json", {
        "study": "BPD-NC09-FC05", "phase": phase,
        "attempted_roots": list(roots), "runtime_or_smoke_failures": failures,
        "scientific_confirmation_models_evaluated": 0,
    })
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "calibration"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-url", required=True)
    args = parser.parse_args()
    return execute(args.phase, args.out_dir, args.run_url)


if __name__ == "__main__":
    raise SystemExit(main())
