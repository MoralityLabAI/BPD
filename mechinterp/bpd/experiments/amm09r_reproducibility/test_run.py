"""No-training checks for the AMM-09R execution boundary and instrumentation."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run = load("_amm09r_runner_tests", HERE / "run.py")
physical = load("_amm09r_physical_tests", HERE / "physical_audit.py")


class FixtureFactor(torch.nn.Module):
    """Fixed arrays with distinct singular values; no initialization RNG or fit."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.V = torch.nn.Parameter(torch.tensor([
            [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
            [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=dtype))
        self.U = torch.nn.Parameter(torch.tensor([
            [1., 0., 0., 1., 0., 0.],
            [0., 2., 0., 0., 1., 0.],
            [0., 0., 3., 0., 0., 1.]], dtype=dtype))

    def forward(self, value):
        return (value @ self.V) @ self.U

    def weight(self):
        return (self.V @ self.U).T


def fixed_pairs(n=128, dtype=torch.float32):
    recipient = torch.zeros((n, 6), dtype=dtype)
    recipient[:, 3:] = torch.linspace(-.3, .3, n * 3, dtype=dtype).reshape(n, 3)
    donor = recipient.clone()
    donor[:, 3:] += torch.tensor([.31, -.29, .33], dtype=dtype)
    return recipient, donor, donor[:, [4, 5]]


class ExecutionBoundaryTests(unittest.TestCase):
    def test_corrupt_frozen_source_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory)
            for filename in run.ORIGINAL_HASHES:
                (original / filename).write_text("raise AssertionError('must never execute')\n")
            with patch.object(run, "ORIGINAL", original), patch.object(run, "load_module") as loader:
                with self.assertRaisesRegex(ValueError, "frozen original source mismatch"):
                    run.load_original(True)
                loader.assert_not_called()

    def test_unverified_archive_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "training.json"
            original.write_text('{"rows": []}\n')
            with patch.object(run.json, "loads") as parser:
                with self.assertRaisesRegex(ValueError, "original TRAIN archive hash mismatch"):
                    run.compare_original({}, original)
                parser.assert_not_called()

    def test_cli_has_no_scientific_phase_or_release_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "receipt.json"
            attempts = [(mode, []) for mode in ("training", "development", "confirmation")]
            attempts.append(("engineering", ["--release", "unused.json"]))
            for mode, extra in attempts:
                argv = [str(HERE / "run.py"), "--mode", mode,
                        "--configuration", "baseline32", "--worker", "test",
                        "--process-ordinal", "1", "--out", str(out), *extra]
                with self.subTest(mode=mode), patch.object(sys, "argv", argv), \
                        patch.object(run, "load_original") as loader, \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as stopped:
                        run.main()
                    self.assertEqual(stopped.exception.code, 2)
                    loader.assert_not_called()
                    self.assertFalse(out.exists())

    def test_physical_audit_rejects_phases_before_any_model_access(self):
        model = Mock()
        for phase in ("training", "development", "confirmation"):
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, "engineering"):
                physical.audit(model, Mock(), Mock(), Mock(), phase, 0)
        self.assertEqual(model.mock_calls, [])

    def test_tensor_capture_preserves_bytes_and_parameters(self):
        value = torch.nn.Parameter(torch.tensor([[0., -0., .125]], dtype=torch.float32))
        before = value.detach().clone()
        captured = run.tensor(value)
        expected = struct.pack("<3f", 0., -0., .125)
        self.assertEqual(captured["dtype"], "float32")
        self.assertEqual(captured["shape"], [1, 3])
        self.assertEqual(captured["sha256"], hashlib.sha256(expected).hexdigest())
        self.assertTrue(torch.equal(value.detach(), before))
        self.assertIsNone(value.grad)
        self.assertTrue(value.requires_grad)


class FrozenModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.case, cls.experiment = run.load_original(True)

    def test_engineering_namespace_cannot_generate_scientific_roots(self):
        self.assertEqual(self.m.PHASE_N, {"engineering": 8})
        for phase in ("training", "development", "confirmation"):
            with self.subTest(phase=phase), patch.object(self.m, "gen") as generator:
                with self.assertRaisesRegex(ValueError, "unregistered model root"):
                    self.m.donor_pairs(phase, 0, 128, "training_pairs")
                generator.assert_not_called()
        for index in (-1, 8):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.m.root_key("engineering", index)
        text = run.NAMESPACE + "engineering/0000/teacher_train"
        expected = int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)
        self.assertEqual(self.m.seed("engineering/0000", "teacher_train"), expected)
        old = int(hashlib.sha256(b"AMM09/engineering/0000/teacher_train").hexdigest()[:15], 16)
        self.assertNotEqual(expected, old)

    def test_source_snapshot_and_math_remain_frozen(self):
        for filename, expected in run.ORIGINAL_HASHES.items():
            self.assertEqual(run.file_sha(run.ORIGINAL / filename), expected)
        self.assertEqual(self.m.TEACHER_STEPS, 400)
        self.assertEqual(self.m.FACTOR_STEPS, 240)
        self.assertEqual(self.m.MASKS, physical.MASKS)
        self.assertEqual(self.m.TOL, .18)

    def test_assay_capture_agrees_with_frozen_score_and_preserves_state(self):
        factor = FixtureFactor()
        recipient, donor, target = fixed_pairs()
        before = {name: value.detach().clone() for name, value in factor.named_parameters()}
        rng_before = torch.random.get_rng_state().clone()
        # Fixed synthetic fixtures avoid spending any registered root stream.
        with patch.object(self.m, "donor_pairs", return_value=(recipient, donor, target)), \
                patch.object(self.m, "root_key", return_value="synthetic_fixture/0000"), \
                patch.object(self.m, "gen", side_effect=lambda *args: torch.Generator().manual_seed(17)):
            reference = self.case.score("engineering", 0, factor, "engineering", 128)
            instrumented = run.assay(self.m, self.case, factor, 0, "engineering", 128, torch)
        self.assertEqual(reference["candidate_bits"], instrumented["candidate_bits"])
        self.assertEqual(reference["candidate_hits"], instrumented["candidate_hits"])
        self.assertEqual(reference["features"], instrumented["features"]["values"])
        self.assertEqual(reference["diagnostics"], instrumented["diagnostics"])
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        for name, value in factor.named_parameters():
            self.assertTrue(torch.equal(before[name], value.detach()))
            self.assertIsNone(value.grad)
        json.dumps(instrumented, allow_nan=False)

    def test_input_hash_capture_uses_private_generators(self):
        model = SimpleNamespace(
            TEACHER_STEPS=2, FACTOR_STEPS=3, R=3,
            root_key=lambda phase, index: f"synthetic_fixture/{index}",
            gen=lambda root, stream: torch.Generator().manual_seed(
                int(hashlib.sha256((root + stream).encode()).hexdigest()[:8], 16)),
        )
        rng_before = torch.random.get_rng_state().clone()
        first = run.input_hashes(model, 0, torch)
        second = run.input_hashes(model, 0, torch)
        changed = run.input_hashes(model, 1, torch)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(set(first), {"teacher_train", "factor_train"})
        self.assertTrue(all(len(value) == 64 for value in first.values()))
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

    def test_zero_dose_and_degenerate_projection_fallback(self):
        raw = torch.tensor([[3., 0., 0., 0., 0., 0.],
                            [4., 0., 0., 0., 0., 0.],
                            [9., 0., 2., 0., 0., 0.]])
        dose = torch.tensor([[0.], [3.], [4.]])
        expected = torch.tensor([[0., 0., 0., 0., 0., 0.],
                                 [0., 3., 0., 0., 0., 0.],
                                 [0., 0., 4., 0., 0., 0.]])
        actual, count = self.m.normalize_dark(raw, dose)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(physical._dark(raw, dose), expected))
        self.assertEqual(count, 1)
        self.assertTrue(torch.equal(actual[:, 0], torch.zeros(3)))
        self.assertTrue(torch.equal(actual.norm(dim=1), dose[:, 0]))

    def test_native_physical_audit_on_fixed_arrays(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                recipient, donor, target = fixed_pairs(dtype=dtype)
                model = SimpleNamespace(
                    PHASE_N={"engineering": 1}, MASKS=physical.MASKS, D=6, R=3,
                    TRAIN_PAIRS=128, root_key=lambda *args: "synthetic_fixture/0000",
                    family_for=lambda index: "wren",
                    donor_pairs=lambda *args: (recipient, donor, target),
                )
                factor = FixtureFactor(dtype)
                rng_before = torch.random.get_rng_state().clone()
                receipt = physical.audit(model, self.case, factor, factor, "engineering", 0)
                self.assertTrue(receipt["passed"], receipt["gates"])
                self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
                self.assertEqual(receipt["full_donor_control"]["max_dose_error"], 0.)
                self.assertGreater(receipt["full_donor_control"]["max_immediate_readout_displacement"], 0.)
                self.assertEqual(receipt["masks"], [list(mask) for mask in physical.MASKS])
                self.assertEqual(len(receipt["native_subset_audits"]), 6)
                self.assertTrue(all(row["calls"] == row["expected"]
                                    for row in receipt["native_call_records"]))
                json.dumps(receipt, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
