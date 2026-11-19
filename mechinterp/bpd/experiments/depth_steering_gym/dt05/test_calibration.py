"""Exercise native calibration on tiny random inputs, never scientific roots."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from .calibration import run_calibration, run_full_step_benchmark
from .resources import MemoryBudget


class NativeCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.budget = MemoryBudget(max_host_rss_gib=12, min_host_available_gib=0)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cpu_native_inference_restores_rng_and_weights(self):
        before = torch.random.get_rng_state().clone()
        report = run_calibration("mini", "inference", 2, [1, 2], "cpu", self.budget,
                                 warmups=0, repeats=1)
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        self.assertEqual(report["parameter_count"], 14082)
        self.assertEqual(report["source_weights_sha256"], report["final_weights_sha256"])
        self.assertTrue(report["parameters_unchanged"])
        self.assertEqual(report["optimizer_updates"], 0)
        self.assertEqual(report["persistent_adam_moment_bytes"], 0)
        self.assertEqual(report["inference_act_steps"], 8)
        self.assertTrue(all(t["status"] == "OK" for t in report["calibration"]["trials"]))
        self.assertFalse(report["scientific_cohorts_accessed"])

    def test_cpu_training_reserves_adam_and_ema_without_updates(self):
        before = torch.random.get_rng_state().clone()
        report = run_calibration("mini", "training", 1, [1], "cpu", self.budget,
                                 warmups=0, repeats=1)
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        self.assertEqual(report["training_fit_depths"], [2, 4, 6])
        self.assertEqual(report["persistent_adam_moment_bytes"], 2 * report["parameter_bytes"])
        self.assertGreaterEqual(report["persistent_ema_shadow_bytes"], report["parameter_bytes"])
        self.assertEqual(report["source_weights_sha256"], report["final_weights_sha256"])
        self.assertEqual(report["selected_microbatch"], 1)

    def test_native_calls_inside_inference_context_are_safe(self):
        with torch.inference_mode():
            inference = run_calibration("mini", "inference", 1, [1], "cpu", self.budget,
                                        warmups=0, repeats=1)
            training = run_calibration("mini", "training", 1, [1], "cpu", self.budget,
                                       warmups=0, repeats=1)
        self.assertEqual(inference["selected_microbatch"], 1)
        self.assertEqual(training["selected_microbatch"], 1)

    def test_invalid_profiles_precision_and_device_rejected(self):
        for kwargs in ({"profile": "unknown"}, {"stage": "unknown"}, {"precision": "fp16"},
                       {"precision": "bf16"}, {"candidates": [3]}):
            args = dict(profile="mini", stage="inference", logical_batch=2, candidates=[1],
                        device="cpu", budget=self.budget)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_calibration(**args)
        with patch("torch.cuda.is_available", return_value=False), self.assertRaisesRegex(ValueError, "unavailable"):
            run_calibration("mini", "inference", 1, [1], "cuda:0", self.budget)

    def test_exception_restores_rng_and_tf32_settings(self):
        before = torch.random.get_rng_state().clone()
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        with patch("mechinterp.bpd.experiments.depth_steering_gym.dt05.calibration.calibrate_microbatch", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                run_calibration("mini", "inference", 1, [1], "cpu", self.budget)
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, old_matmul)
        self.assertEqual(torch.backends.cudnn.allow_tf32, old_cudnn)

    def test_full_step_benchmark_measures_all_stages_and_restores_rng(self):
        before = torch.random.get_rng_state().clone()
        for mode in ("reference", "optimized"):
            report = run_full_step_benchmark(
                "mini", 4, [2, 4], "cpu", self.budget,
                warmups=0, repeats=1, execution_mode=mode, checkpoint_act=(mode == "optimized"),
            )
            self.assertEqual(report["status"], "BENCHMARKED")
            self.assertIn(report["selected_microbatch"], (2, 4))
            self.assertFalse(report["scientific_cohorts_accessed"])
            ok = [t for t in report["trials"] if t["status"] == "OK"]
            self.assertEqual(len(ok), 2)
            stages = ok[0]["stage_median_seconds"]
            for key in ("data_preparation", "h2d_transfer", "forward_backward_accumulated",
                        "grad_clip_and_optimizer_step", "ema_update"):
                self.assertGreaterEqual(stages[key], 0.0)
            self.assertEqual(ok[0]["ema_updates_completed"], 1)
            self.assertNotEqual(ok[0]["post_trial_weights_sha256"], report["initial_weights_sha256"])
        torch.testing.assert_close(torch.random.get_rng_state(), before)


if __name__ == "__main__":
    unittest.main()
