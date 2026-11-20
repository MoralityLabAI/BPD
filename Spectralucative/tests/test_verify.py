"""Adversarial receipt checks using independent small engineering fixtures."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from Spectralucative import math as instrument
from Spectralucative import verify


REPO = Path(__file__).resolve().parents[2]


class TraceReplayTests(unittest.TestCase):
    def test_nonnormal_norm_domain_is_not_spectral_radius(self):
        matrix = np.array([[0.25, 2.0], [0.0, 0.25]])
        raw = instrument.trace_logdet_diagnostic(matrix, z=1.0)
        self.assertEqual(raw["status"], "outside_norm_domain")
        verify.verify_trace(matrix, raw)
        damped = instrument.trace_logdet_diagnostic(matrix, z=0.2)
        self.assertAlmostEqual(damped["negative_logdet"], -2 * np.log(0.95))
        verify.verify_trace(matrix, damped)

    def test_strict_boundary_refuses_certificate(self):
        matrix = [[1.0]]
        report = instrument.trace_logdet_diagnostic(matrix, z=1.0)
        verify.verify_trace(matrix, report)
        report["analytic_tail_bound"] = 0.0
        with self.assertRaisesRegex(verify.VerificationError, "outside norm domain"):
            verify.verify_trace(matrix, report)

    def test_forged_small_tail_rejected_below_observation_tolerance(self):
        matrix = [[0.5]]
        report = instrument.trace_logdet_diagnostic(matrix, z=1, terms=32)
        self.assertLess(report["analytic_tail_bound"], verify.ATOL)
        report["analytic_tail_bound"] = 0.0
        with self.assertRaisesRegex(verify.VerificationError, "analytic_tail_bound"):
            verify.verify_trace(matrix, report)

    def test_forged_precision_claim_rejected(self):
        matrix = [[0.5]]
        report = instrument.trace_logdet_diagnostic(matrix, z=1, terms=64)
        report["absolute_log_error"] = 1e-10
        with self.assertRaisesRegex(verify.VerificationError, "declared criterion"):
            verify.verify_trace(matrix, report)

    def test_inflated_slack_and_roundoff_claim_rejected(self):
        matrix = [[0.25, 0.0], [0.0, -0.25]]
        original = instrument.trace_logdet_diagnostic(matrix, z=1)
        for key, value in (("floating_point_slack", 1.0), ("roundoff_certified", True)):
            report = deepcopy(original)
            report[key] = value
            with self.subTest(key=key), self.assertRaises(verify.VerificationError):
                verify.verify_trace(matrix, report)


class TransportReplayTests(unittest.TestCase):
    def fixture(self):
        A = [np.array([[0.5, 1.0], [0.0, 0.25]]), np.array([[0.2, 0.0], [0.8, 0.3]])]
        B = [np.array([[1.0, 0.4, 0.0], [0.0, 0.5, 0.7], [0.2, 0.0, 0.3]]),
             np.array([[0.4, 0.0, 0.6], [0.2, 0.8, 0.0], [0.0, 0.5, 0.9]])]
        L = [np.array([[1.0, 0.0], [0.0, 1.0], [0.2, -0.1]])] * 3
        return A, B, L, instrument.transport_diagnostic(A, B, L)

    def test_noncommuting_rectangular_transport(self):
        A, B, L, report = self.fixture()
        self.assertFalse(np.allclose(B[1] @ B[0], B[0] @ B[1]))
        expected = B[1] @ B[0] @ L[0] - L[2] @ A[1] @ A[0]
        np.testing.assert_allclose(report["final_difference"], expected)
        verify.verify_transport(A, B, L, report)

    def test_varying_dimensions(self):
        rng = np.random.default_rng(7)
        L = [rng.normal(size=(3, 2)), rng.normal(size=(4, 3)), rng.normal(size=(2, 1))]
        A = [rng.normal(size=(3, 2)), rng.normal(size=(1, 3))]
        B = [rng.normal(size=(4, 3)), rng.normal(size=(2, 4))]
        verify.verify_transport(A, B, L, instrument.transport_diagnostic(A, B, L))

    def test_reversed_product_rejected(self):
        A, B, L, report = self.fixture()
        report["final_difference"] = (B[0] @ B[1] @ L[0] - L[2] @ A[0] @ A[1]).tolist()
        with self.assertRaisesRegex(verify.VerificationError, "final_difference"):
            verify.verify_transport(A, B, L, report)

    def test_forged_residual_below_replay_tolerance_rejected(self):
        A, B, L, report = self.fixture()
        report["reconstruction_residual_norm"] = 1e-10
        with self.assertRaisesRegex(verify.VerificationError, "declared criterion"):
            verify.verify_transport(A, B, L, report)

    def test_misleading_envelope_rejected(self):
        A, B, L, report = self.fixture()
        report["norm_envelope"][-1] = 0
        with self.assertRaisesRegex(verify.VerificationError, "norm_envelope"):
            verify.verify_transport(A, B, L, report)


class WalshReplayTests(unittest.TestCase):
    def fixture(self):
        ids = [2, 0, 3, 1]
        values = np.array([[1 + 0.2 * (-1) ** (x & 1),
                            -0.7 * (-1) ** ((x >> 1) & 1) + 0.3 * (-1) ** x.bit_count()]
                           for x in ids])
        return ids, values, instrument.walsh_diagnostic(ids, values, max_degree=1)

    def test_known_vector_coefficients_and_residual(self):
        ids, values, report = self.fixture()
        np.testing.assert_allclose(report["coefficients"], [[1, 0], [0.2, 0], [0, -0.7], [0, 0.3]], atol=1e-15)
        self.assertAlmostEqual(report["discarded_coefficient_energy"], 0.09)
        verify.verify_walsh(ids, values, report)

    def test_duplicate_and_missing_masks_rejected(self):
        ids, values, report = self.fixture()
        for bad in ([2, 0, 3, 3], [2, 0, 3]):
            with self.subTest(ids=bad), self.assertRaises(verify.VerificationError):
                verify.verify_walsh(bad, values, report)

    def test_coefficient_tampering_rejected(self):
        ids, values, report = self.fixture()
        report["coefficients"][3][1] = 0
        with self.assertRaisesRegex(verify.VerificationError, "coefficients"):
            verify.verify_walsh(ids, values, report)

    def test_nonuniform_measure_rejected(self):
        ids, values, report = self.fixture()
        report["measure"] = "selected_masks"
        with self.assertRaisesRegex(verify.VerificationError, "measure"):
            verify.verify_walsh(ids, values, report)

    def test_forged_identity_error_rejected_below_replay_tolerance(self):
        ids, values, report = self.fixture()
        report["residual_identity_absolute_error"] = 1e-10
        with self.assertRaisesRegex(verify.VerificationError, "declared criterion"):
            verify.verify_walsh(ids, values, report)

    def test_scalar_constant_has_zero_variance(self):
        values = [3.0] * 8
        report = instrument.walsh_diagnostic(list(range(8)), values)
        self.assertEqual(report["high_degree_fraction_of_variance"], 0)
        verify.verify_walsh(list(range(8)), values, report)


class ReceiptBoundaryTests(unittest.TestCase):
    def test_digest_tracks_numeric_payload(self):
        payload = {"answer": [0.0, 2.0]}
        payload["payload_sha256"] = verify.payload_digest(payload)
        verify._check_digest(payload)
        payload["answer"][1] = 3.0
        with self.assertRaisesRegex(verify.VerificationError, "digest"):
            verify._check_digest(payload)

    def test_rehashed_scope_mutations_still_fail(self):
        original = dict(schema="BPD/Spectralucative/native-unit/v1", evidence_class="engineering_implementation_only",
                        engineering_initialization_id="spectralucative_cpu_init_2026112100", seed=2026112100,
                        model_initialization="random", **dict.fromkeys(verify.ZERO_COUNTERS, 0))
        for key, value in (("scientific_roots", 4), ("confirmation_roots", 1), ("training_updates", 1),
                           ("checkpoint_downloads", 1), ("gpu_runs", 1), ("reserved_development_roots", 1),
                           ("model_initialization", "trained"), ("evidence_class", "causal_ready"),
                           ("scientific_readiness", "ready")):
            payload = dict(original, **{key: value})
            payload["payload_sha256"] = verify.payload_digest(payload)
            with self.subTest(key=key), self.assertRaises(verify.VerificationError):
                verify.verify_unit(payload, REPO)

    def test_engineering_initializations_are_separate_from_scientific_roots(self):
        payload = dict(evidence_class="engineering_implementation_only", model_initialization="random",
                       engineering_root_count=4, **dict.fromkeys(verify.ZERO_COUNTERS, 0))
        verify._engineering_scope(payload)
        self.assertEqual(payload["scientific_roots"], 0)

    def test_nan_and_duplicate_json_fields_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            for encoded in ('{"a":NaN}', '{"a":1,"a":2}'):
                path.write_text(encoded)
                with self.subTest(encoded=encoded), self.assertRaises(verify.VerificationError):
                    verify._load_json(path)

    def test_source_hash_and_required_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.py").write_text("x = 1\n")
            digest = hashlib.sha256((root / "source.py").read_bytes()).hexdigest()
            manifest = {"source_files_sha256": {"source.py": digest}}
            self.assertEqual(verify.verify_sources(manifest, root), 1)
            with self.assertRaisesRegex(verify.VerificationError, "Required source"):
                verify.verify_sources(manifest, root, required={"missing.py"})
            (root / "source.py").write_text("x = 2\n")
            with self.assertRaisesRegex(verify.VerificationError, "hash mismatch"):
                verify.verify_sources(manifest, root)

    def test_source_paths_cannot_open_data_or_escape_repository(self):
        for relative in ("../outside.py", "/tmp/outside.py", "data/confirmation.json", "nested\\outside.py"):
            with self.subTest(path=relative), self.assertRaises(verify.VerificationError):
                verify._source_path(REPO, relative)

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            outside = Path(directory) / "outside.py"
            outside.write_text("x = 1\n")
            (root / "source.py").symlink_to(outside)
            with self.assertRaisesRegex(verify.VerificationError, "symlink"):
                verify._source_path(root, "source.py")


class NativeArithmeticReplayTests(unittest.TestCase):
    def test_rectangular_factor_parity_and_residual_retention(self):
        V = np.array([[1, 0.5], [0, 1], [0.2, -0.3]])
        U = np.array([[0.3, -0.1], [0.2, 0.4]])
        reconstruction = U.T @ V.T
        target = reconstruction + 0.1
        factor = dict(V=V.tolist(), U=U.tolist(), target=target.tolist(), mask=[1, 1],
                      reconstruction=reconstruction.tolist(), residual=(target - reconstruction).tolist(),
                      all_ones_tolerance=2e-7, all_ones_max_abs_weight_error=0.0,
                      residual_retained=True, module="layers.0.down_proj")
        self.assertEqual(verify.verify_factor(factor), (2, 2))
        factor["residual"] = np.zeros_like(target).tolist()
        with self.assertRaisesRegex(verify.VerificationError, "residual"):
            verify.verify_factor(factor)

    def test_finite_displacement_replays_ordered_products_and_remainder(self):
        B = [np.array([[0.5, 0.2], [0, 0.6]]), np.array([[0.8, 0], [0.4, 0.3]])]
        Q = np.eye(2)
        baseline = np.array([[1.0, 2.0], [0.9, 1.2], [0.72, 0.72]])
        records = []
        for dose in (1e-4, 1e-3, 1e-2):
            requested = dose * np.linalg.norm(baseline[0]) * Q[:, 0]
            perturbed = [baseline[0] + requested]
            for J in B:
                state = J @ perturbed[-1]
                state[1] += dose ** 2
                perturbed.append(state)
            actual = np.asarray(perturbed) - baseline
            linear = [actual[0]]
            for J in B:
                linear.append(J @ linear[-1])
            errors = np.linalg.norm(actual - linear, axis=1)
            records.append(dict(relative_dose=dose, requested_initial_delta=requested.tolist(),
                                actual_initial_delta=actual[0].tolist(), perturbed_states=np.asarray(perturbed).tolist(),
                                actual_deltas=actual.tolist(), linearized_deltas=np.asarray(linear).tolist(),
                                remainder_l2=errors.tolist(),
                                relative_remainder=(errors / np.maximum(1e-15, np.linalg.norm(actual, axis=1))).tolist()))
        finite = dict(direction=Q[:, 0].tolist(), relative_doses=[1e-4, 1e-3, 1e-2],
                      state_norm_l2=float(np.linalg.norm(baseline[0])), baseline_states=baseline.tolist(), records=records)
        verify.verify_finite(finite, B, Q)
        finite["records"][2]["remainder_l2"][2] = 0
        with self.assertRaisesRegex(verify.VerificationError, "remainder"):
            verify.verify_finite(finite, B, Q)

    def test_frozen_and_ordered_products_are_distinct(self):
        A, B, Q = [np.array([[0.5]]), np.array([[0.3]])], [np.diag([0.5, 1]), np.diag([0.3, 2])], np.array([[1], [0]])
        comparison = dict(depths=[1, 2], reduced_ordered_products=[[[0.5]], [[0.15]]],
                          reduced_frozen_powers=[[[0.5]], [[0.25]]], reduced_gap_inf=[0, 0.1],
                          full_ordered_pushes=[[[0.5], [0]], [[0.15], [0]]],
                          full_frozen_pushes=[[[0.5], [0]], [[0.25], [0]]], full_gap_inf=[0, 0.1])
        verify.verify_frozen(comparison, A, B, Q)
        comparison["reduced_gap_inf"][1] = 0
        with self.assertRaisesRegex(verify.VerificationError, "reduced_gap_inf"):
            verify.verify_frozen(comparison, A, B, Q)


class HistoricalGateReplayTests(unittest.TestCase):
    def test_strict_cli_retains_exit_two_and_original_verdict(self):
        result = verify.verify_historical(REPO / "Spectralucative")
        self.assertEqual(result, dict(strict_exit_status=2, decision="REDESIGN_OR_PILOT_ONLY", exact_reproduction=True))

    def test_historical_mutation_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in ("vendor", "evidence"):
                shutil.copytree(REPO / "Spectralucative" / name, target / name)
            path = target / "evidence/historical_dt05_input.json"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(verify.VerificationError, "Historical file changed"):
                verify.verify_historical(target)

    def test_cli_writes_a_concrete_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "verification.json"
            process = subprocess.run([sys.executable, "-m", "Spectralucative.verify", "--historical-only",
                                      "--out", str(output)], cwd=REPO, capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            receipt = json.loads(output.read_text())
            self.assertIs(receipt["passed"], True)
            self.assertEqual(receipt["historical_gate"]["strict_exit_status"], 2)
            verify._check_digest(receipt)


if __name__ == "__main__":
    unittest.main()
