"""Audit the first original smoke from its saved arrays, without model training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .audit import source_audit, task_audit
from .aggregate import technical_admission_errors
from .experiment import ROOT, protect_source_manifest
from .runner import compare_frozen_sampler
from .solver import fit_full_tangent_shams

ORIGINAL_RUN = 44317319540
ORIGINAL_ARTIFACT = 14433326335
ORIGINAL_ROOT_SHA = "e7c00303308e44606cc05b64ad31057dbbea5339d6b48ccff37f47ff31a825bc"


def audit_original_smoke(root_dir: Path) -> dict:
    protect_source_manifest()
    path = root_dir / "nc09_fc05_root_15999.json"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != ORIGINAL_ROOT_SHA:
        raise ValueError("The original smoke root bytes changed")
    record = json.loads(path.read_text())
    assert record["root"] == 15999 and record["training"]["steps"] == 384
    assert record["splits"] == {
        "train": 2048, "derivative_discovery": 1024,
        "model_validation": 512, "labeled_calibration": 1024,
    }
    source_path = root_dir / record["source_archive"]["file"]
    task_path = root_dir / record["task_archive"]["file"]
    source = source_audit(record, source_path)
    task = task_audit(record, task_path)
    technical = technical_admission_errors(record)
    frozen = compare_frozen_sampler(record, source_path)
    with np.load(source_path, allow_pickle=False) as arrays:
        _, replay = fit_full_tangent_shams(
            torch.from_numpy(arrays["a"].copy()),
            torch.from_numpy(arrays["U"].copy()),
            torch.from_numpy(arrays["z"].copy()),
            torch.from_numpy(arrays["g"].copy()),
            [torch.from_numpy(np.asfortranarray(arrays[f"E_{t}"])) for t in range(4)],
            15999,
        )
    original = record["sham_admission"]
    replay_hashes = [x["projector_sha256"] for x in replay["projectors"]]
    expected_hashes = [x["projector_sha256"] for x in original["projectors"]]
    assert replay_hashes == expected_hashes == frozen["projector_sha256_sequence"]
    states = lambda admission: [
        state["parameters"]["sha256"] for chart in admission["charts"]
        for state in chart["iterations"]
    ]
    assert states(replay) == states(original)
    for name in ("target_source_write_rms", "target_source_signed_score_mean",
                 "target_source_score_rms", "total_jacobian_corrections"):
        assert replay[name] == original[name]
    return {
        "study": "BPD-NC09-FC05", "phase": "ORIGINAL_SMOKE_REPLAY_AUDIT",
        "original_run_id": ORIGINAL_RUN, "original_artifact_id": ORIGINAL_ARTIFACT,
        "original_root_json_sha256": actual,
        "original_workflow_conclusion": "failure",
        "original_source_archive_sha256": record["source_archive"]["sha256"],
        "original_task_archive_sha256": record["task_archive"]["sha256"],
        "passed": bool(source["passed"] and task["passed"] and frozen["passed"] and not technical),
        "source_audit_passed": source["passed"], "task_audit_passed": task["passed"],
        "technical_admission_errors": technical,
        "frozen_sampler_agreement": frozen,
        "instrumented_projector_sha256_sequence": replay_hashes,
        "exact_original_parameter_states_reproduced": len(states(replay)),
        "original_jacobian_corrections": original["total_jacobian_corrections"],
        "replay_basis_stride_elements": [1, 20],
        "layout_source": "Frozen nc08.passwise_source_candidates slices torch.linalg.eigh eigenvectors",
        "new_models_trained": 0, "new_task_evaluations": 0,
        "scientific_confirmation_models_evaluated": 0,
        "audit_source_manifest_sha256": hashlib.sha256((ROOT / "SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_original_smoke(args.root_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
