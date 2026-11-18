"""One-shot source-competence runner; explicit design/prospective role required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

import torch

from .oracle import exact_certificate
from .source import (DESIGN_ROOTS, PROSPECTIVE_ROOTS, CANDIDATES, setup_runtime,
                     sha256, train_trajectory, evaluate, summarize, implementation_checks)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PREFIX = "mechinterp/bpd/experiments/sub08/"
SOURCE_FILES = (
    "DESIGN_PROTOCOL.md", "oracle.py", "source.py", "run.py", "audit.py",
    "test_oracle.py", "test_source.py", "test_audit.py", "EXACT_ORACLE.json",
    "evalmath_preflight_pinned.py", "INHERITED_SUB07_PILOT.json", "INHERITED_SUB07_PREFLIGHT.json",
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(data)
    temp.replace(path)


def source_hashes():
    return {PREFIX + name: sha256(HERE / name) for name in SOURCE_FILES}


def design_selection(rows):
    candidates = []
    selected = None
    for width, updates in CANDIDATES:
        subset = [r for r in rows if r["width"] == width and r["updates"] == updates]
        gates = []
        for row in subset:
            ok = False
            if row.get("status") == "complete":
                m = summarize(row["splits"]["validation"])
                ok = 20 * m["correct"] >= 19 * m["n"] and all(
                    10 * m["confusion"][c][c] >= 9 * m["class_counts"][c] for c in range(4)
                )
            gates.append({"seed": row["seed"], "passed": bool(ok)})
        passed = len(subset) == 3 and sorted(r["seed"] for r in subset) == list(DESIGN_ROOTS) and all(g["passed"] for g in gates)
        candidates.append({"width": width, "updates": updates, "passed": passed, "roots": gates})
        if passed and selected is None:
            selected = {"width": width, "updates": updates}
    return {"verdict": "DESIGN_PASS" if selected else "DESIGN_HOLD", "selected_config": selected, "candidates": candidates}


def run(role, output, lock_path=None, source_commit=None, design_directory=None):
    if role not in {"design", "prospective"}:
        raise ValueError("Runner role must be design or prospective")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("One-shot output directory already exists")
    runtime = setup_runtime()
    hashes = source_hashes()
    certificate = exact_certificate()
    inherited = json.loads((HERE / "INHERITED_SUB07_PREFLIGHT.json").read_text())
    if inherited["decision"] != "REDESIGN_OR_PILOT_ONLY":
        raise ValueError("Inherited scientific boundary changed")
    lock = None
    if role == "prospective":
        if lock_path is None:
            raise ValueError("Prospective run requires committed COMPETENCE_LOCK")
        lock = json.loads(Path(lock_path).read_text())
        if lock.get("schema") != 1 or source_commit != lock.get("source_commit"):
            raise ValueError("Prospective source commit differs from locked source")
        if lock["source_hashes"] != hashes or lock["runtime"] != runtime:
            raise ValueError("Source/runtime differs from prospective lock")
        if lock["roots"] != list(PROSPECTIVE_ROOTS):
            raise ValueError("Prospective root roster differs from lock")
        if (lock["width"], lock["updates"]) not in CANDIDATES:
            raise ValueError("Unregistered candidate")
        if lock["partition_hashes"] != {k: v["ids_sha256"] for k, v in certificate["partitions"].items()}:
            raise ValueError("Partition hashes differ from lock")
        if design_directory is None:
            raise ValueError("Prospective run requires original excluded design receipts")
        design_directory = Path(design_directory).resolve()
        for name, key in (("manifest.json", "design_manifest_sha256"),
                          ("selection.json", "design_selection_sha256"),
                          ("AUDIT.json", "design_audit_sha256")):
            if sha256(design_directory / name) != lock[key]:
                raise ValueError("Design receipt differs from locked original: " + name)
        from .audit import audit_role
        audited = audit_role(design_directory, repo_root=REPO)
        if not audited.get("audit_valid") or audited.get("verdict") != "DESIGN_PASS":
            raise ValueError("Original excluded design failed independent audit")
        design_manifest = json.loads((design_directory / "manifest.json").read_text())
        if design_manifest["source_commit"] != source_commit or design_manifest["source_hashes"] != hashes:
            raise ValueError("Design source differs from prospective source")
        design_rows = [json.loads((design_directory / p).read_text()) for p in design_manifest["roots"]]
        selection = design_selection(design_rows)
        if selection != json.loads((design_directory / "selection.json").read_text()):
            raise ValueError("Archived design selection cannot be reproduced")
        if selection["selected_config"] != {"width": lock["width"], "updates": lock["updates"]}:
            raise ValueError("Prospective configuration differs from selected design")
        jobs = [(seed, lock["width"], lock["updates"]) for seed in PROSPECTIVE_ROOTS]
    else:
        if lock_path is not None:
            raise ValueError("Design run does not accept a prospective lock")
        jobs = [(seed, width, 3600) for width in (48, 64) for seed in DESIGN_ROOTS]
    output.mkdir(parents=True)
    if lock is not None:
        shutil.copyfile(lock_path, output / "lock.json")
    rows = []
    manifest = {
        "schema": 1, "role": role, "runtime": runtime, "source_hashes": hashes,
        "source_commit": source_commit, "roots": [], "started_unix": time.time(),
        "partition_hashes": {k: v["ids_sha256"] for k, v in certificate["partitions"].items()},
        "causal_scope": {"n_causal_units": 0, "physical_dose": None, "effect_floor": None,
                         "sham_floor": None, "causal_power": None, "status": "CAUSAL_STUDY_NOT_RUN"},
    }
    write_json(output / "manifest.json", manifest)
    for seed, width, updates in jobs:
        completed_checkpoints = set()

        def record(model, step, training):
            checkpoint_rel = f"checkpoints/seed{seed}_w{width}_u{step}.pt"
            checkpoint = output / checkpoint_rel
            checkpoint.parent.mkdir(exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "seed": seed, "width": width,
                        "updates": step, "role": role}, checkpoint)
            split_names = ("train", "validation", "heldout") if role == "prospective" else ("train", "validation")
            splits = {name: evaluate(model, name, role) for name in split_names}
            row = {
                "schema": 1, "role": role, "status": "complete", "seed": seed,
                "width": width, "updates": step, **training, "source_hashes": hashes,
                "checkpoint": {"path": checkpoint_rel, "sha256": sha256(checkpoint)},
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "training_ids_sha256": splits["train"]["ids_sha256"],
                "splits": splits, "metrics": {name: summarize(value) for name, value in splits.items()},
                "checks": implementation_checks(model, role),
            }
            rel = f"roots/seed{seed}_w{width}_u{step}.json"
            write_json(output / rel, row)
            rows.append(row)
            manifest["roots"].append(rel)
            write_json(output / "manifest.json", manifest)
            completed_checkpoints.add(step)
            report_split = "heldout" if role == "prospective" else "validation"
            metrics = row["metrics"][report_split]
            print(json.dumps({"role": role, "seed": seed, "width": width, "updates": step,
                              "accuracy": metrics["accuracy"], "class_recall": metrics["class_recall"]}), flush=True)

        try:
            train_trajectory(seed, width, updates, role, record)
        except Exception as error:
            for step in ((2400, 3600) if role == "design" else (updates,)):
                if step in completed_checkpoints:
                    continue
                row = {"schema": 1, "role": role, "status": "runtime_failure", "seed": seed,
                       "width": width, "updates": step, "runtime": runtime, "source_hashes": hashes,
                       "completed_updates": getattr(error, "completed_updates", 0),
                       "initial_state_sha256": getattr(error, "initial_state_sha256", None),
                       "trace": getattr(error, "trace", []), "exception": repr(error),
                       "traceback": traceback.format_exc()}
                failed_checkpoint = output / f"checkpoints/seed{seed}_w{width}_u{step}.pt"
                if failed_checkpoint.exists():
                    row["checkpoint"] = {"path": str(failed_checkpoint.relative_to(output)),
                                         "sha256": sha256(failed_checkpoint)}
                rel = f"roots/seed{seed}_w{width}_u{step}.json"
                write_json(output / rel, row)
                rows.append(row)
                manifest["roots"].append(rel)
            write_json(output / "manifest.json", manifest)
            print(json.dumps({"seed": seed, "width": width, "error": repr(error)}), flush=True)
    if role == "design":
        manifest["selection"] = design_selection(rows)
        write_json(output / "selection.json", manifest["selection"])
    manifest["finished_unix"] = time.time()
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("design", "prospective"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--design-directory", type=Path)
    args = parser.parse_args()
    run(args.role, args.output, args.lock, args.source_commit, args.design_directory)


if __name__ == "__main__":
    main()

