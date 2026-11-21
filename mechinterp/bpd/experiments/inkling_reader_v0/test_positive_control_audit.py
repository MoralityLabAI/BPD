"""Tamper and replay tests for independent synthetic recurrence receipts."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    from mechinterp.bpd.experiments.inkling_reader_v0.audit_positive_control import (
        ARCH, SCHEMA, VERIFIER, _argmax, _digest, _walk_state, audit,
    )
except ModuleNotFoundError:
    from audit_positive_control import ARCH, SCHEMA, VERIFIER, _argmax, _digest, _walk_state, audit


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_fixture(folder: Path) -> tuple[list[dict], list[dict]]:
    theta = 5.0
    alpha = 1 / (1 + math.exp(-theta))
    sha = _digest({"theta": theta, "arch": ARCH})
    checkpoint = {
        "theta": theta, "alpha": alpha, "sha256": sha,
        "architecture": "explicit_permutation_one_parameter_recurrent_gate",
    }
    (folder / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    depth_rows = []
    intervention_rows = []
    for index in range(2):
        root_id = f"development-10-{index:04d}"
        case_id = f"{root_id}-h4"
        perm = tuple((i + 1) % 17 for i in range(17))
        start = index
        hops = 4
        target = (start + hops) % 17
        world = {"start": start, "hops": hops, "permutation": list(perm)}
        root = {
            "root_id": root_id, "case_id": case_id,
            "model_sha": sha, "task_world": world,
            "task_digest": _digest({"root_id": root_id, "world": world}),
            "verifier_id": VERIFIER, "family": "explicit_single_cycle_pointer_chase",
            "split": "development", "hops": hops, "target": target,
        }
        for depth in (1, 2, 5):
            state = _walk_state(perm, start, min(hops, depth), alpha)
            prediction = _argmax(state)
            depth_rows.append({
                **root, "depth": depth, "verified": prediction == target,
                "predicted": prediction,
            })
        before = _walk_state(perm, start, 2, alpha)
        peak = _argmax(before)
        dose = 0.85
        for arm in ("clean", "zero", "active_coordinate", "random_direction", "random_coordinate"):
            post = before[:]
            if arm == "active_coordinate":
                post[peak] -= dose / math.sqrt(2)
                post[(peak + 1) % 17] += dose / math.sqrt(2)
            elif arm == "random_direction":
                post[8] += dose
            elif arm == "random_coordinate":
                post[9] -= dose / math.sqrt(2)
                post[10] += dose / math.sqrt(2)
            if arm == "clean":
                final = _walk_state(perm, start, 4, alpha)
                patch = None
                displacement = 0.0
            else:
                final = _walk_state(perm, start, 2, alpha, state=post)
                patch = before
                displacement = math.sqrt(sum((a - b) ** 2 for a, b in zip(before, post, strict=True)))
            prediction = _argmax(final)
            intervention_rows.append({
                **root, "arm": arm, "depth": 5, "patch_visit": 2,
                "predicted": prediction, "verified": prediction == target,
                "nominal_l2_dose": 0.0 if arm in {"clean", "zero"} else dose,
                "realized_raw_l2_displacement": displacement,
                "pre_patch_raw_state": patch,
                "post_patch_raw_state": None if patch is None else post,
                "pre_patch_argmax": None if patch is None else peak,
            })
    _write_jsonl(folder / "depth_matrix.jsonl", depth_rows)
    _write_jsonl(folder / "interventions.jsonl", intervention_rows)
    summary = {
        "schema_version": SCHEMA,
        "evidence_class": "synthetic_trained_gate_development_positive_control",
        "checkpoint_sha256": sha, "source_roots": 2, "cases": 2,
        "nominal_dose": 0.85, "depth_accuracy": {}, "intervention_accuracy": {},
    }
    for depth in (1, 2, 5):
        subset = [row for row in depth_rows if row["depth"] == depth]
        summary["depth_accuracy"][str(depth)] = {
            "correct": sum(row["verified"] for row in subset), "total": len(subset),
        }
    for arm in ("clean", "zero", "active_coordinate", "random_direction", "random_coordinate"):
        subset = [row for row in intervention_rows if row["arm"] == arm]
        summary["intervention_accuracy"][arm] = {
            "correct": sum(row["verified"] for row in subset), "total": len(subset),
        }
    summary["files_sha256"] = {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
        for name in ("checkpoint.json", "depth_matrix.jsonl", "interventions.jsonl")
    }
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return depth_rows, intervention_rows


def reseal_test_fixture(folder: Path) -> None:
    """Test-only reseal ensures auditor catches semantics, not just raw hash tampering."""
    summary = json.loads((folder / "summary.json").read_text())
    for name in ("checkpoint.json", "depth_matrix.jsonl", "interventions.jsonl"):
        summary["files_sha256"][name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


class IndependentAuditTests(unittest.TestCase):
    def test_fresh_matched_source_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            build_fixture(folder)
            result = audit(folder)
            self.assertTrue(result["receipt_replay_passed"])
            self.assertEqual(result["roots"], 2)
            self.assertEqual(result["intervention_cases"], 2)
            self.assertEqual(result["per_depth_correct"][5], 2)

    def test_independent_oracle_rejects_resealed_target_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows, _ = build_fixture(folder)
            rows[0]["target"] = (rows[0]["target"] + 1) % 17
            _write_jsonl(folder / "depth_matrix.jsonl", rows)
            reseal_test_fixture(folder)
            with self.assertRaisesRegex(ValueError, "target differs"):
                audit(folder)

    def test_physical_patch_rejects_resealed_dose_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _, rows = build_fixture(folder)
            patched = next(row for row in rows if row["arm"] == "active_coordinate")
            patched["post_patch_raw_state"][0] += 0.1
            _write_jsonl(folder / "interventions.jsonl", rows)
            reseal_test_fixture(folder)
            with self.assertRaises(ValueError):
                audit(folder)

    def test_incomplete_pair_matrix_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _, rows = build_fixture(folder)
            _write_jsonl(folder / "interventions.jsonl", rows[:-1])
            reseal_test_fixture(folder)
            with self.assertRaisesRegex(ValueError, "incomplete intervention"):
                audit(folder)


if __name__ == "__main__":
    unittest.main()
