"""Acceptance-gate tests; no model fitting or torch import is required."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("amm09r_compare", Path(__file__).with_name("compare.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def pack(shape, number=0.):
    count = 1
    for dimension in shape:
        count *= dimension
    values = number
    for dimension in reversed(shape):
        values = [values] * dimension
    return {"dtype": "float32", "shape": shape,
            "sha256": hashlib.sha256(struct.pack("<" + str(count) + "f", *([number] * count))).hexdigest(),
            "values": values}


def fixture_receipts():
    assays = {}
    for split, n in (("training_pairs", 128), ("engineering", 512)):
        assays[split] = {
            "candidate_bits": {f"mask_{action}": "0" * n for action in range(6)},
            "features": pack([n, 6, 18]), "probes": pack([n, 6, 2]),
            "patches": pack([n, 6, 6]), "margins": pack([n, 6], -.25),
        }
    rows = [{
        "root": f"engineering/{index:04d}",
        "initial": {"I": pack([1], 1.)},
        "train_input_sha256": {"teacher_train": "a" * 64, "factor_train": "b" * 64},
        "teacher": {"weight": pack([1], 1.)},
        "factor_before_svd": {"weight": pack([1], 1.)},
        "factor": {"weight": pack([1], 1.)}, "fit": {"finite": True},
        "assays": assays,
        "physical": {"passed": True, "gates": {key: True for key in c.PHYSICAL_GATES}, "masks": c.MASKS},
    } for index in range(8)]
    engineering = {"phase": "engineering", "n_roots": 8,
                   "namespace": "AMM09R-ENGINEERING-V1/", "rows": rows,
                   "fixture": {"coefficients": [0.] * 18, "pooled_action": 0}}
    historical = {"rows": [{"root": f"training/{index:04d}"} for index in range(64)]}
    receipts = []
    for mode, configuration in c.GROUPS:
        payload = engineering if mode == "engineering" else historical
        for worker in c.WORKERS:
            for ordinal in c.ORDINALS:
                receipts.append((f"{mode}_{configuration}_{worker}_{ordinal}.json", {
                    "schema": c.SCHEMA, "mode": mode, "configuration": configuration,
                    "execution_status": "completed", "confirmation_accessed": False,
                    "source_sha256": {"run.py": "c" * 64},
                    "payload": payload, "payload_sha256": c.digest(payload),
                    "provenance": {
                        "worker": worker, "process_ordinal": ordinal, "host_id": worker + "-host",
                        "cpu_model": "same test CPU", "registered_software": True,
                        "environment": c.PINNED_ENV, "deterministic_algorithms": True,
                        "num_threads": 1, "num_interop_threads": 1, "cpu_capability": "NO AVX",
                        **c.VERSIONS, "torch_git": "test-commit",
                        "torch_binary_sha256": {"libtorch_cpu.so": "d" * 64, "_C.so": "e" * 64},
                    },
                }))
    return receipts


class ComparatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipts = fixture_receipts()

    def replaced(self, mutate):
        receipts = list(self.receipts)
        index = next(i for i, (_, r) in enumerate(receipts)
                     if r["mode"] == "engineering" and r["configuration"] == "pinned32")
        name, receipt = receipts[index]
        replacement = copy.deepcopy(receipt)
        mutate(replacement)
        receipts[index] = (name, replacement)
        return receipts

    def test_same_cpu_passes_with_narrow_scope(self):
        report = c.compare_receipts(self.receipts)
        self.assertTrue(report["registered_gate_passed"], report["receipt_errors"])
        self.assertFalse(report["distinct_cpu_models_observed"])
        self.assertFalse(report["confirmation_authorized"])
        self.assertEqual(report["cpu_models"], ["same test CPU"])

    def test_tampered_payload_digest_fails(self):
        report = c.compare_receipts(self.replaced(lambda r: r.update(payload_sha256="0" * 64)))
        self.assertFalse(report["registered_gate_passed"])
        self.assertFalse(report["gates"]["all_receipts_valid"])
        self.assertTrue(any("payload digest mismatch" in e["errors"] for e in report["receipt_errors"]))

    def test_missing_worker_fails(self):
        receipts = [(name, r) for name, r in self.receipts if not (
            r["mode"] == "engineering" and r["configuration"] == "pinned32"
            and r["provenance"]["worker"] == c.WORKERS[1])]
        report = c.compare_receipts(receipts)
        self.assertFalse(report["registered_gate_passed"])
        self.assertFalse(report["gates"]["all_registered_repeats_present"])
        self.assertFalse(report["gates"]["two_distinct_hosts"])

    def test_changed_weights_with_same_bits_fails(self):
        def change(receipt):
            receipt["payload"]["rows"][0]["teacher"]["weight"] = pack([1], 1.125)
            receipt["payload_sha256"] = c.digest(receipt["payload"])
        report = c.compare_receipts(self.replaced(change))
        self.assertFalse(report["registered_gate_passed"])
        self.assertTrue(report["gates"]["all_receipts_valid"])
        comparisons = report["groups"]["engineering/pinned32"]["comparisons"]
        self.assertTrue(all(row["candidate_bit_flips"] == 0 for row in comparisons))
        self.assertTrue(all(row["max_absolute_numeric_difference"] == .125 for row in comparisons))
        self.assertTrue(all(row["first_differing_recorded_stage"] == "/rows/0/teacher" for row in comparisons))

    def test_valid_json_digest_cannot_hide_bad_tensor_digest(self):
        def change(receipt):
            receipt["payload"]["rows"][0]["teacher"]["weight"]["values"][0] = 2.
            receipt["payload_sha256"] = c.digest(receipt["payload"])
        report = c.compare_receipts(self.replaced(change))
        self.assertFalse(report["registered_gate_passed"])
        self.assertTrue(any("tensor byte hash mismatch" in error
                            for row in report["receipt_errors"] for error in row["errors"]))

    def test_alias_worker_cannot_count_as_independent_host(self):
        receipts = []
        for name, original in self.receipts:
            receipt = dict(original)
            receipt["provenance"] = dict(original["provenance"], host_id="one-host")
            receipts.append((name, receipt))
        report = c.compare_receipts(receipts)
        self.assertFalse(report["registered_gate_passed"])
        self.assertFalse(report["gates"]["two_distinct_hosts"])

    def test_baseline_mislabeled_pinned_fails(self):
        def change(receipt):
            receipt["provenance"]["environment"]["ATEN_CPU_CAPABILITY"] = None
            receipt["provenance"]["cpu_capability"] = "AVX512"
            receipt["provenance"]["deterministic_algorithms"] = False
        report = c.compare_receipts(self.replaced(change))
        self.assertFalse(report["registered_gate_passed"])
        self.assertTrue(report["gates"]["pinned_payloads_bit_exact"])
        self.assertFalse(report["gates"]["pinned_runtime_settings"])

    def test_bit_difference_includes_paired_margins(self):
        left = {"rows": [{"assays": {"engineering": {
            "candidate_bits": {"mask_0": "01"}, "margins": {"values": [[-.01], [.02]]}}}}]}
        right = {"rows": [{"assays": {"engineering": {
            "candidate_bits": {"mask_0": "11"}, "margins": {"values": [[.005], [.02]]}}}}]}
        report = c.differences(left, right)
        self.assertEqual(report["candidate_bit_flips"], 1)
        self.assertEqual(report["candidate_bit_examples"][0]["left_margin"], -.01)
        self.assertEqual(report["candidate_bit_examples"][0]["right_margin"], .005)

    def test_incomplete_cli_never_sets_passing_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "comparison.json"
            args = ["--directory", temporary, "--out", str(output)]
            self.assertEqual(c.main(args), 1)
            self.assertEqual(c.main([*args, "--allow-incomplete"]), 0)
            self.assertFalse(json.loads(output.read_text())["registered_gate_passed"])


if __name__ == "__main__":
    unittest.main()
