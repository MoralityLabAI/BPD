"""Admission and statistical-contract tests using manufactured receipts only."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_calibration_fc05 import aggregate as mod


def _admitted_numeric_receipt():
    """Manufactured metrics for testing gate independence from eligible=True."""
    metric = {"nll_nats": .1, "accuracy": .95, "correct_count": 973,
              "nll_delta_from_clean_nats": .01, "final_kl_nats": .001,
              "closed_loop_write_displacement_rms": .1, "kl_by_pass_nats": [.001] * 4}
    gauge = {"physical_map_relative_error": 1e-8, "max_complete_rollout_logit_error": 1e-7}
    ranks = {"dp12": [4, 3, 3, 2], "flat3": [3] * 4,
             "flat3_dose": [3] * 4, "flat4": [4] * 4}
    return {
        "root": 16000, "eligible": True,
        "training": {"root": 16000, "steps": 384, "train_size": 2048, "mix_density": 1.},
        "splits": {"train": 2048, "derivative_discovery": 1024,
                   "model_validation": 512, "labeled_calibration": 1024},
        "clean_task_context": {"context_positive": .95, "context_nonpositive": .96},
        "full_mask_closure": {"passed": True, "max_abs_logit": 1e-7, "max_kl_nats": 1e-9},
        "source_rank12_optimum": {"ranks": [4, 3, 3, 2], "total_rank": 12},
        "physical_source_gain": 1.,
        "source_arms": {k: {"eligible": True, "ranks": v, "total_rank": sum(v)} for k, v in ranks.items()},
        "task_nll_by_arm": {k: dict(metric) for k in ranks},
        "native_rotated_gauge": {k: dict(gauge) for k in ranks},
        "sham_nlls": [.11] * 8, "sham_task_metrics": [dict(metric) for _ in range(8)],
        "native_rotated_sham_gauge": [dict(gauge) for _ in range(8)],
        "selected_sham_score_source_only": True,
        "sham_admission": {"root": 16000, "root_seed": 15056000,
            "charts_configured": 32, "max_corrections_per_chart": 64,
            "source_observation_count": 1024, "fixed_angle_radians": .25,
            "minimum_DP12_chordal": .20, "minimum_pairwise_chordal": .18,
            "source_write_relative_tolerance": .15, "jacobian_damping": .001,
            "minimum_norm_step_limit": .5, "line_search_scales": [1., .5, .25, .125],
            "uses_external_task_labels": False, "charts": [{} for _ in range(32)],
            "selected": 8, "projectors": [{} for _ in range(8)]},
    }


class TestIncompleteOriginalCohort(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.integrity = {"passed": True, "manifest_sha256": "a" * 64,
                          "checked_files": 1, "errors": []}

    def tearDown(self):
        self.temp.cleanup()

    def run_aggregate(self, records):
        for record in records:
            (self.directory / f"nc09_fc05_root_{record['root']}.json").write_text(json.dumps(record))
        with patch.object(mod, "verify_source_manifest", return_value=self.integrity), \
             patch.object(mod, "aggregate_root_mean_null") as null:
            result = mod.aggregate(records, self.directory, archive_source="manufactured unit test")
            null.assert_not_called()
        self.assertEqual(result["registered_development_verdict"], "CALIBRATION_INCOMPLETE")
        self.assertEqual(len(result["per_root"]), 59)
        self.assertIsNone(result["aggregate_32_root_pseudo_null"])
        self.assertIsNone(result["post_calibration_evalmath"])
        return result

    def test_missing_roots_remain_in_denominator(self):
        result = self.run_aggregate([])
        self.assertEqual(result["eligible_roots"], 0)
        self.assertEqual([r["root"] for r in result["per_root"]], list(mod.ROOTS))
        self.assertTrue(all(r["errors"] == ["missing_root"] for r in result["per_root"]))

    def test_missing_optimum_is_a_receipted_failure(self):
        result = self.run_aggregate([{"root": 16000, "eligible": False,
                                      "source_rank12_optimum": None,
                                      "failure": {"type": "source_ineligible"}}])
        first = result["per_root"][0]
        self.assertIsNone(first["rank_tuple"])
        self.assertIn("source_rank12_optimum_missing_or_invalid", first["errors"])
        self.assertEqual(first["original_failure"], {"type": "source_ineligible"})
        self.assertIsNotNone(first["original_json_sha256"])

    def test_duplicate_and_unregistered_ids_are_explicit_errors(self):
        result = self.run_aggregate([{"root": 16000}, {"root": 16000}, {"root": 17000}])
        self.assertIn("duplicate_root:16000", result["evidence_errors"])
        self.assertIn("unregistered_root:17000", result["evidence_errors"])
        self.assertEqual(result["per_root"][0]["errors"], ["duplicate_root"])

    def test_source_hash_mismatch_cannot_pass_on_eligible_flag(self):
        result = self.run_aggregate([{"root": 16000, "eligible": True,
                                      "source_sha256": "wrong source"}])
        self.assertIn("root_16000:source_identity:source_sha256", result["evidence_errors"])
        self.assertFalse(result["per_root"][0]["eligible"])


class TestNumericAdmission(unittest.TestCase):
    def test_boolean_claim_does_not_override_clean_accuracy_or_sham_gauge(self):
        record = _admitted_numeric_receipt()
        self.assertEqual(mod.technical_admission_errors(record), [])
        record["clean_task_context"]["context_positive"] = .849
        record["native_rotated_sham_gauge"][3]["physical_map_relative_error"] = 1.01e-4
        errors = mod.technical_admission_errors(record)
        self.assertIn("clean_context_accuracy", errors)
        self.assertIn("sham_native_rotated_gauge", errors)

    def test_missing_optimum_and_nonfinite_closure_fail_without_exception(self):
        record = _admitted_numeric_receipt()
        record["source_rank12_optimum"] = None
        record["full_mask_closure"]["max_abs_logit"] = float("nan")
        errors = mod.technical_admission_errors(record)
        self.assertIn("source_rank12_optimum_missing_or_invalid", errors)
        self.assertIn("native_full_mask_closure", errors)


class TestStatisticalContract(unittest.TestCase):
    def test_pseudo_null_center_and_variance_identity(self):
        record = {"root": 16000, "clean_nll": 3., "sham_nlls": list(range(1, 9)),
                  "task_nll_by_arm": {"dp12": {"nll_nats": 4.},
                                      "flat3": {"nll_nats": 5.},
                                      "flat3_dose": {"nll_nats": 6.},
                                      "flat4": {"nll_nats": 2.}}}
        result = mod.recompute_effects(record)
        self.assertAlmostEqual(sum(result["Z_pseudo_null_by_sham"]), 0.)
        self.assertAlmostEqual(result["conditional_pseudo_target_variance"], 48 / 7)
        self.assertEqual(result["T_null_worst_abs"], 3.5)
        self.assertEqual([result[k] for k in mod.METRICS], [1., 2., .5, 1., 2.])
        record["sham_nlls"] = record["sham_nlls"][:7]
        with self.assertRaisesRegex(ValueError, "Eight finite"):
            mod.recompute_effects(record)

    def test_incremental_margin_is_strict_and_additive(self):
        boot = {
            "A_flat3_minus_dp12": {"mean": .010, "lower95": .002},
            "D_dose_flat3_minus_dp12": {"mean": .009, "lower95": .001},
            "S_mean_sham_minus_dp12": {"mean": .05, "lower95": .03},
            "C_dp12_minus_clean": {"mean": .01, "upper95": .02},
            "F_dp12_minus_flat4": {"mean": .001, "upper95": .01},
        }
        null = {"upper95_across_outer_95pct_estimates": .01}
        power = {"engineering_power_gate_passed": True}
        gates = mod.development_gates(boot, null, power)
        self.assertFalse(gates["aggregate_null_exceeded_with_incremental_margin"])
        boot["S_mean_sham_minus_dp12"]["lower95"] = float(np.nextafter(.03, np.inf))
        self.assertTrue(all(mod.development_gates(boot, null, power).values()))

    def test_evalmath_preserves_original_development_rows(self):
        original = json.loads((mod.ROOT / "prereview/FC04_HISTORICAL_TARGET_PREFLIGHT_INPUT.json").read_text())
        result = mod.evalmath_review(.14, "original FC05 manufactured archive reference")
        self.assertEqual(result["input"]["pilot"], original["pilot"])
        self.assertEqual(result["input"]["external_sham_floor"], "0.14")
        self.assertEqual(result["input"]["target_effect"], "27/1000")
        self.assertEqual(result["output"]["pilot_worlds"], 16)
        self.assertEqual(result["decision"], "REDESIGN_OR_PILOT_ONLY")
        self.assertEqual(result["strict_exit_code"], 2)
        unsupported = mod.evalmath_review(1.01, "original FC05 manufactured archive reference")
        self.assertEqual(unsupported["decision"], "EVALMATH_UNSUPPORTED_INPUT_HOLD")


if __name__ == "__main__":
    unittest.main()
