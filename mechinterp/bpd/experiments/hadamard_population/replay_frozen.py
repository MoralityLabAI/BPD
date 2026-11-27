"""Replay the pinned, already exposed HAD-05 development pilots and audit receipts.

Supply a directory exported from ORIGINAL_COMMIT. This command rejects modified
source, records the current runtime, and labels every output as development.
Wall times are descriptive and have no cross-host or matched-compute guarantee.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ORIGINAL_COMMIT = "3b7ee1775b1e7a709c536641105ee4fc3195a5a8"
GIT_BLOBS = {
    "had05.py": "3a06be89b3a502b1de02a4a34d8182c97377be6f",
    "archive.py": "563cdb30ffcc4f0facd341c72bc413d36baf8632",
    "near.py": "b74568f2a18f4cdcc13c3e3bb167d46842156d27",
    "audit.py": "e726e437cf6dc917878c54761abbd504dd235564",
    "archive_audit.py": "5aa264f8b61350668582562204d7af867681f2e5",
}
REPORTED_TOTALS = {
    "allocation": {"serial": 22, "round_robin": 19, "best": 36,
        "ldt_best": 36, "ucb": 36, "ucb_transfer": 33, "mlp": 24,
        "no_ldt": 19, "trm": 19, "fixed": 18, "sham": 22,
        "live": 19, "trm_transfer": 18},
    "exact": {"best": 39, "archive16": 39, "archive64": 36,
        "archive128": 34, "random64": 36},
    "near": {"best": 24, "exact64": 23, "near64_noinject": 22,
        "near64": 28, "random_near64": 28},
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def summarize(path, family):
    data = json.loads(path.read_text())
    groups = collections.defaultdict(list)
    for row in data["rows"]:
        groups[row["arm"]].append(row)
    arms = {}
    for arm, rows in sorted(groups.items()):
        roots = sorted({row["root"] for row in rows})
        solved = sum(row["solved"] for row in rows)
        arms[arm] = {
            "cases": len(rows), "solved": solved,
            "root_solves": {str(root): sum(row["solved"] for row in rows
                if row["root"] == root) for root in roots},
            "reported_original_solves": REPORTED_TOTALS[family][arm],
            "aggregate_matches_report": solved == REPORTED_TOTALS[family][arm],
            "summed_search_seconds": sum(row["seconds"] for row in rows),
            "injections": sum(len(row.get("injections", [])) for row in rows),
            "injected_descendant_solves": sum(bool(row.get("solution_injection"))
                for row in rows),
        }
    return {"rows": len(data["rows"]), "arms": arms,
        "scope": "DEVELOPMENT_REPLAY",
        "comparison_limit": "Only totals reported in the old README are available. "
            "Matching totals do not establish casewise or numeric replay parity."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    src, out = args.source_dir.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "MANIFEST.json").exists():
        raise SystemExit("Use a fresh output directory to preserve earlier receipts")
    sources = {}
    for name, expected in GIT_BLOBS.items():
        path = src / name
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != expected:
            raise SystemExit(f"Source mismatch: {name}: {blob}")
        sources[name] = {"git_blob": blob, "sha256": sha256(path)}
    import numpy as np
    import torch
    manifest = {
        "scope": "DEVELOPMENT_REPLAY", "source_commit": ORIGINAL_COMMIT,
        "source_files": sources,
        "runtime": {"python": sys.version, "numpy": np.__version__,
            "torch": torch.__version__, "platform": platform.platform(),
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        "historical_runtime": {"numpy": "2.3.5", "torch": "2.10.0+cpu"},
        "selection_exposure": "All roots and choices repeat exposed development pilots.",
        "original_raw_receipts_available": False,
        "jobs": [], "complete": False,
    }
    write(out / "MANIFEST.json", manifest)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    jobs = [
        ("allocation", "had05.py", ["--out", str(out / "results.json")]),
        ("exact", "archive.py", ["--out", str(out / "archive_results.json")]),
        ("near", "near.py", ["--out", str(out / "near_results.json")]),
        ("audit", "audit.py", ["--results", str(out / "results.json"),
            "--out", str(out / "audit.json")]),
        ("archive_audit", "archive_audit.py", ["--dir", str(out)]),
    ]
    for name, script, flags in jobs:
        command = [sys.executable, str(src / script), *flags]
        start = time.perf_counter()
        print(f"Starting {name}", flush=True)
        with (out / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=src, env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False)
        manifest["jobs"].append({"name": name, "command": command,
            "returncode": result.returncode, "seconds": time.perf_counter()-start})
        write(out / "MANIFEST.json", manifest)
        print(f"Finished {name}: returncode={result.returncode}", flush=True)
        if result.returncode:
            return result.returncode
    summary = {family: summarize(out / filename, family) for family, filename in (
        ("allocation", "results.json"), ("exact", "archive_results.json"),
        ("near", "near_results.json"))}
    write(out / "SUMMARY.json", summary)
    manifest["complete"] = True
    manifest["output_sha256"] = {p.name: sha256(p) for p in sorted(out.iterdir())
        if p.is_file() and p.name != "MANIFEST.json"}
    write(out / "MANIFEST.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
