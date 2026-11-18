"""Validate the BPD-DT04 execution lock and preserve immutable historical locks.

Verifies:
1. Historical DT01 `LOCK_STAGE0.json` and DT02-A `LOCK_CI.json` git blob hashes.
2. Archived DT02-A first-run JSON SHA-256 (`9d4f9463...`).
3. BPD-DT04 `EXECUTE_LOCK.json` protocol commit, registered root roster, and
   exact SHA-256 digests of every DT04 source file before any model initialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

from . import DEV_ROOTS, SMOKE_ROOT, STUDY_ID

HERE = Path(__file__).resolve().parent
GYM = HERE.parent
REPO = HERE.parents[4]
EXPECTED_DT02_SHA256 = (
    "9d4f9463c0853c6cc2491711ebf3cb6c37021a61861f3947962dff67fa76b30b"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def verify_historical_locks() -> dict:
    """Confirm no historical DT01, DT02-A, DT03 or UI artifact was altered."""
    checked = 0
    for lock_name in ("LOCK_STAGE0.json", "LOCK_CI.json"):
        lock_path = GYM / lock_name
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for rel, expected_blob in lock["files"].items():
            actual_blob = git_blob_sha1(GYM / rel)
            if actual_blob != expected_blob:
                raise ValueError(
                    f"Historical lock {lock_name} mismatch on {rel}: {actual_blob} != {expected_blob}"
                )
            checked += 1
    dt02_archive = GYM / "results" / "dt02a_native_excluded.json"
    if sha256_file(dt02_archive) != EXPECTED_DT02_SHA256:
        raise ValueError("Historical DT02-A archive SHA-256 mismatch")
    return {"historical_files_checked": checked, "dt02_archive_verified": True}


def verify_source(*, require_runtime: bool = True) -> dict:
    """Validate DT04 execution lock and all historical gym locks."""
    hist = verify_historical_locks()
    lock_path = HERE / "EXECUTE_LOCK.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Missing BPD-DT04 EXECUTE_LOCK.json")
    raw = lock_path.read_bytes()
    lock = json.loads(raw)
    if lock.get("schema") != "BPD/DT04-execution-lock/v1":
        raise ValueError("Invalid DT04 execution lock schema")
    if lock.get("study") != STUDY_ID:
        raise ValueError("Mismatched DT04 study identifier")
    if lock.get("smoke_root") != SMOKE_ROOT:
        raise ValueError("Registered smoke root mismatch")
    if lock.get("development_roots") != list(DEV_ROOTS):
        raise ValueError("Registered 59-root development roster mismatch")
    if lock.get("arg03_n_sealed") is not True:
        raise ValueError("ARG-03-N seal flag missing")

    files_sha = lock.get("files_sha256")
    if not isinstance(files_sha, dict) or not files_sha:
        raise ValueError("EXECUTE_LOCK.json has no frozen source files")
    for rel_name, expected in files_sha.items():
        parts = PurePosixPath(rel_name)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError(f"Unsafe path in execution lock: {rel_name}")
        target = (REPO / rel_name).resolve()
        if not target.is_relative_to(REPO) or not target.is_file():
            raise ValueError(f"Missing locked source file: {rel_name}")
        actual = sha256_file(target)
        if actual != expected:
            raise ValueError(f"DT04 source SHA-256 mismatch on {rel_name}: {actual} != {expected}")

    if require_runtime:
        import numpy
        import torch

        expected_rt = lock["runtime_recipe"]
        actual_rt = {
            "python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch": torch.__version__,
            "numpy": numpy.__version__,
        }
        for key, val in actual_rt.items():
            if val != expected_rt.get(key):
                raise ValueError(f"Runtime mismatch on {key}: {val} != {expected_rt.get(key)}")

    return {
        "source_check": "PASS",
        "execute_lock_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_commit": lock.get("protocol_commit"),
        "dt04_files_checked": len(files_sha),
        **hist,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-runtime", action="store_true")
    args = parser.parse_args()
    out = verify_source(require_runtime=not args.skip_runtime)
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
