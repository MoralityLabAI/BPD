"""Read-only audit fault injection, exact thresholds, and fixture-only rosters.

No registered model root is initialized or trained.  Checkpoint fixtures use
literal zero tensors; the replay module's temporary initialization uses1889.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import audit


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")


def literal_state(width=48):
    return {
        "cell.weight_ih": torch.zeros(3 * width, 4, dtype=torch.float32, device="cpu"),
        "cell.weight_hh": torch.zeros(3 * width, width, dtype=torch.float32, device="cpu"),
        "cell.bias_ih": torch.zeros(3 * width, dtype=torch.float32, device="cpu"),
        "cell.bias_hh": torch.zeros(3 * width, dtype=torch.float32, device="cpu"),
        "head.weight": torch.zeros(4, width, dtype=torch.float32, device="cpu"),
        "head.bias": torch.zeros(4, dtype=torch.float32, device="cpu"),
    }


def fixture(directory, role="design"):
    """Synthetic runtime failures exercise roster retention without training."""
    directory = Path(directory)
    repository = directory / "repo"
    output = directory / "receipt"
    output.mkdir(parents=True)
    hashes = {}
    for name in audit.REQUIRED_SOURCES:
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source fixture: " + name)
        hashes[name] = audit.sha256_file(path)
    runtime = {
        "python": "3.12.8", "python_minor": "3.12", "torch": "2.5.1+cpu",
        "numpy": "2.3.5", "platform": "synthetic-test", "device": "cpu",
        "dtype": "float32", "torch_threads": 1, "deterministic_algorithms": True,
    }
    jobs = ([(seed, width, updates) for seed in audit.DESIGN_SEEDS for width, updates in audit.CONFIGS]
            if role == "design" else [(seed, 48, 2400) for seed in audit.PROSPECTIVE_SEEDS])
    paths = []
    for seed, width, updates in jobs:
        relative = f"roots/{seed}_{width}_{updates}.json"
        write_json(output / relative, {
            "schema": 1, "role": role, "status": "runtime_failure", "seed": seed,
            "width": width, "updates": updates, "completed_updates": 0,
            "runtime": runtime, "source_hashes": hashes, "exception": "synthetic failure fixture",
        })
        paths.append(relative)
    manifest = {
        "schema": 1, "role": role, "runtime": runtime, "source_hashes": hashes,
        "source_commit": "a" * 40, "roots": paths,
        "partition_hashes": {name: audit.ids_sha256(ids) for name, ids in audit.exact_partitions().items()},
    }
    write_json(output / "manifest.json", manifest)
    if role == "prospective":
        lock = {
            "schema": 1, "width": 48, "updates": 2400, "roots": list(audit.PROSPECTIVE_SEEDS),
            "runtime": runtime, "source_hashes": hashes, "partition_hashes": manifest["partition_hashes"],
            "protocol_commit": "b" * 40, "source_commit": "a" * 40,
            "design_manifest_sha256": "c" * 64, "design_audit_sha256": "d" * 64,
            "design_selection_sha256": "e" * 64,
        }
        write_json(output / "lock.json", lock)
        write_json(repository / audit.SOURCE_PREFIX / "COMPETENCE_LOCK.json", lock)
    return repository, output, manifest


class AuditTests(unittest.TestCase):
    def test_literal_oracle_and_population_vs_subset_certificates(self):
        certificate = audit.exact_certificate()
        self.assertEqual(certificate["class_counts"], [781, 1146, 1024, 1145])
        self.assertEqual(certificate["population_correct_ceilings"]["counts"], 1603)
        self.assertEqual(certificate["heldout_correct_ceilings"]["counts"], 449)
        self.assertGreater(449 / 1024, 1603 / 4096)
        all_ids = np.concatenate(list(audit.exact_partitions().values()))
        self.assertEqual(len(set(all_ids.tolist())), 4096)

    def test_exact_threshold_boundaries_and_small_class_failure(self):
        result = {"n": 80, "correct": 72, "class_support": [20] * 4, "class_correct": [17, 17, 19, 19]}
        self.assertTrue(all(audit.accuracy_recall_gates(result, "prospective").values()))
        result["correct"] = 71
        self.assertFalse(audit.accuracy_recall_gates(result, "prospective")["heldout_accuracy"])
        result.update(correct=72, class_correct=[16, 18, 19, 19])
        self.assertFalse(audit.accuracy_recall_gates(result, "prospective")["heldout_class_recall"])
        result.update(correct=76, class_correct=[18, 18, 20, 20])
        self.assertTrue(all(audit.accuracy_recall_gates(result, "design").values()))
        result["class_correct"] = [17, 19, 20, 20]
        self.assertFalse(audit.accuracy_recall_gates(result, "design")["validation_class_recall"])

    def test_metrics_reject_label_corruption_nonfinite_logits_and_duplicate_ids(self):
        _, truth = audit.exact_universe()
        ids = audit.exact_partitions()["validation"][:32].tolist()
        labels = truth[ids].tolist()
        logits = np.zeros((len(ids), 4))
        baseline = audit.metrics_from_logits(ids, labels, logits)
        self.assertEqual(baseline["correct"], labels.count(0))
        corrupted = labels.copy()
        corrupted[0] = (corrupted[0] + 1) % 4
        with self.assertRaisesRegex(audit.AuditError, "literal oracle"):
            audit.metrics_from_logits(ids, corrupted, logits)
        logits[0, 0] = np.inf
        with self.assertRaisesRegex(audit.AuditError, "nonfinite logits"):
            audit.metrics_from_logits(ids, labels, logits)
        with self.assertRaisesRegex(audit.AuditError, "duplicate"):
            audit.metrics_from_logits([ids[0], ids[0]], [labels[0], labels[0]], np.zeros((2, 4)))

    def test_exact_binomial_interval_boundaries_and_symmetry(self):
        low, high = audit.clopper_pearson(8, 8)
        self.assertAlmostEqual(low, .025 ** (1 / 8), places=14)
        self.assertEqual(high, 1)
        zero = audit.clopper_pearson(0, 8)
        self.assertEqual(zero[0], 0)
        self.assertAlmostEqual(zero[1], 1 - low, places=14)
        seven, one = audit.clopper_pearson(7, 8), audit.clopper_pearson(1, 8)
        self.assertAlmostEqual(seven[0], 1 - one[1], places=14)
        self.assertAlmostEqual(seven[1], 1 - one[0], places=14)
        self.assertLess(seven[0], 7 / 8)

    def test_all_failed_design_and_prospective_roots_are_retained(self):
        for role, expected_n, expected_records in (("design", 6, 12), ("prospective", 8, 8)):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                repository, output, _ = fixture(directory, role)
                result = audit.audit_role(output, repository)
                self.assertTrue(result["audit_valid"], result["errors"])
                self.assertEqual(result["n_roots"], expected_n)
                self.assertEqual(result["n_records"], expected_records)
                self.assertTrue(result["verdict"].endswith("HOLD"))
                self.assertTrue(all(not row["passed"] for row in result["root_results"]))

    def test_source_tamper_and_copied_lock_tamper_force_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, output, _ = fixture(directory)
            (repository / audit.REQUIRED_SOURCES[0]).write_text("tampered source")
            result = audit.audit_role(output, repository)
            self.assertFalse(result["audit_valid"])
            self.assertTrue(any("source hash mismatch" in error for error in result["errors"]))
        with tempfile.TemporaryDirectory() as directory:
            repository, output, _ = fixture(directory, "prospective")
            lock = json.loads((output / "lock.json").read_text())
            lock["updates"] = 3600
            write_json(output / "lock.json", lock)
            result = audit.audit_role(output, repository)
            self.assertFalse(result["audit_valid"])
            self.assertEqual(result["verdict"], "SOURCE_COMPETENCE_HOLD")
            self.assertTrue(any("repository lock" in error for error in result["errors"]))

    def test_missing_root_stays_in_audit_and_complete_roster_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, output, manifest = fixture(directory)
            (output / manifest["roots"][0]).unlink()
            result = audit.audit_role(output, repository)
            self.assertFalse(result["audit_valid"])
            self.assertEqual(len(result["root_results"]), 12)
            self.assertEqual(sum(row["status"] == "runtime_failure" for row in result["root_results"]), 11)
            self.assertEqual(result["verdict"], "DESIGN_HOLD")

    def test_forbidden_reserved_split_and_unsafe_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, output, manifest = fixture(directory)
            path = output / manifest["roots"][0]
            row = json.loads(path.read_text())
            row["splits"] = {"reserved_dose": {}}
            write_json(path, row)
            result = audit.audit_role(output, repository)
            self.assertFalse(result["audit_valid"])
            self.assertTrue(any("unexpectedly contains scored splits" in error for error in result["errors"]))
            with self.assertRaisesRegex(audit.AuditError, "unsafe path"):
                audit._safe_file(output, "../repo/outside")

    def test_deterministic_candidate_choice_uses_all_three_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, output, _ = fixture(directory)

            def synthetic_gate(row, *args):
                # Fixture-only aggregate testing; no checkpoint is represented as audited.
                passing = (row["width"], row["updates"]) in ((48, 3600), (64, 2400), (64, 3600))
                digest = hashlib.sha256(f'{row["seed"]}/{row["width"]}'.encode()).hexdigest()
                return {"seed": row["seed"], "width": row["width"], "updates": row["updates"],
                        "status": "complete", "passed": passing, "initial_state_sha256": digest}

            with patch.object(audit, "audit_root", side_effect=synthetic_gate):
                result = audit.audit_role(output, repository)
            self.assertTrue(result["audit_valid"], result["errors"])
            self.assertEqual(result["selected_config"], {"width": 48, "updates": 3600})
            self.assertEqual(result["n_roots"], 6)

    def test_checkpoint_hash_finite_weights_and_saved_logit_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            state = literal_state()
            payload = {"state_dict": state, "seed": 1890, "width": 48, "updates": 2400, "role": "design"}
            path = directory / "literal.pt"
            torch.save(payload, path)
            row = {key: payload[key] for key in ("seed", "width", "updates", "role")}
            row.update(parameter_count=7972, checkpoint={"path": "literal.pt", "sha256": audit.sha256_file(path)})
            verified, _ = audit._checkpoint(row, directory)
            ids = audit.exact_partitions()["validation"][:16].tolist()
            saved = {"validation": {"ids": ids, "logits": np.zeros((16, 4)).tolist()}}
            self.assertEqual(audit._replay(verified, 48, saved)["validation.logits"], 0)
            saved["validation"]["logits"][0][0] = .01
            with self.assertRaisesRegex(audit.AuditError, "checkpoint replay mismatch"):
                audit._replay(verified, 48, saved)
            with path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(audit.AuditError, "checkpoint hash mismatch"):
                audit._checkpoint(row, directory)
            state["head.bias"][0] = float("nan")
            torch.save(payload, path)
            row["checkpoint"]["sha256"] = audit.sha256_file(path)
            with self.assertRaisesRegex(audit.AuditError, "nonfinite"):
                audit._checkpoint(row, directory)


if __name__ == "__main__":
    unittest.main()
