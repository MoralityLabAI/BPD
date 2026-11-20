"""Native CPU checks run in CI with the required Torch dependency, without skips."""

import unittest
from pathlib import Path

import numpy as np

from Spectralucative.native import SEED_BASE, run_initialization, validate_options
from Spectralucative.verify import verify_unit


class ResourceGuardTests(unittest.TestCase):
    def test_invalid_work_is_rejected_before_model_creation(self):
        for options in ({"roots": 0}, {"roots": 9}, {"steps": 9},
                        {"components": 5}, {"rank": 5}, {"cpu_threads": 8},
                        {"roots": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_options(**options)


class NativeInstrumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = run_initialization(seed=SEED_BASE, steps=2, rank=3, components=4,
                                         cpu_threads=1, repo_root=Path(__file__).resolve().parents[2])

    def test_independent_receipt_replay(self):
        result = verify_unit(self.receipt, Path(__file__).resolve().parents[2])
        self.assertTrue(result["transport_checked"])
        self.assertTrue(result["complete_walsh_cube_checked"])

    def test_real_native_full_state_derivatives(self):
        receipt = self.receipt
        self.assertEqual(receipt["state"]["dimension"], 80)
        self.assertEqual(np.shape(receipt["transport"]["B_steps"]), (2, 80, 80))
        self.assertEqual(receipt["runtime"]["native_dtype"], "float32")
        self.assertEqual(receipt["runtime"]["analysis_dtype"], "float64")
        self.assertTrue(receipt["checks"]["autodiff_jvp_parity"])
        self.assertTrue(receipt["checks"]["central_difference_jvp_sanity"])
        self.assertTrue(any(not step["raw"]["convergence_condition"]
                            for step in receipt["full_state_spectral"]))
        self.assertTrue(all(step["damped"]["convergence_condition"]
                            for step in receipt["full_state_spectral"]))

    def test_complete_native_masks_and_full_head_target_parity(self):
        cube = self.receipt["mask_cube"]
        self.assertEqual(cube["mask_ids"], list(range(16)))
        self.assertEqual(np.shape(cube["full_head_logits"]), (16, 4, 11))
        self.assertEqual(np.shape(self.receipt["walsh"]["values"]), (16, 44))
        self.assertTrue(cube["residual_retained"])
        self.assertEqual(cube["patch_invocation"], 0)
        self.assertLessEqual(cube["all_ones_max_abs_logit_error"], cube["all_ones_tolerance"])
        self.assertGreater(cube["physical_write_displacement_l2"][0], 1e-6)
        self.assertLessEqual(cube["physical_write_displacement_l2"][-1], 1e-5)
        self.assertTrue(self.receipt["checks"]["delivered_write_formula"])

    def test_recorded_nonlinear_remainder_and_execution_boundary(self):
        receipt = self.receipt
        for record in receipt["finite_displacement"]["records"]:
            self.assertEqual(np.shape(record["actual_deltas"]), (3, 80))
            self.assertGreater(np.linalg.norm(record["actual_initial_delta"]), 0)
            self.assertEqual(record["remainder_l2"][0], 0)
        self.assertEqual(receipt["evidence_class"], "engineering_implementation_only")
        self.assertEqual(receipt["model_initialization"], "random")
        self.assertIsNone(receipt["engineering_fixture"]["task_performance_endpoint"])
        for name in ("training_updates", "checkpoint_downloads", "gpu_runs",
                     "scientific_roots", "confirmation_roots", "reserved_development_roots"):
            self.assertEqual(receipt[name], 0)
        self.assertTrue(receipt["all_checks_passed"], receipt["checks"])


if __name__ == "__main__":
    unittest.main()
