from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cases
import scoring


def distribution(label, value):
    return {label: value, "B" if label == "A" else "A": 1 - value}


def record_for(case, effect_probability=0.75, control_probability=0.12):
    target = case["effect_donor"]["label"]
    clean = {role: {"probabilities": distribution(case[role]["label"], 0.9)}
             for role in scoring.CLEAN_ROLES}
    probabilities = {"identity": 0.1, "effect": effect_probability,
                     "sham": control_probability, "random": control_probability,
                     "full_donor_prefix": 0.9}
    arms = {name: {"probabilities": distribution(target, p),
                   "physical_dose": 0.0 if name == "identity" else 1.0,
                   "protected_readout_max_abs": 0.0, "admission": True}
            for name, p in probabilities.items()}
    arms["full_donor_prefix"]["physical_dose"] = 10.0
    return {"case_id": case["case_id"], "root_id": case["root_id"],
            "status": "completed", "clean": clean, "arms": arms}


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.manifest = cases.generate_manifest()
        self.records = [record_for(case) for case in self.manifest["cases"]]

    def test_paired_control_scoring_and_pilot_scope(self):
        result = scoring.summarize(self.records, self.manifest)
        self.assertEqual(result["verdict"], "PILOT_ONLY")
        self.assertEqual(result["n_eligible_cases"], 4)
        self.assertEqual(result["n_eligible_units"], 4)
        self.assertAlmostEqual(result["primary_root_mean"], 0.63)
        self.assertIsNone(result["matched_sham_floor"])
        self.assertEqual(result["score_semantics"],
                         "conditional next-token probability over the two declared answers A/B")
        self.assertEqual(result["competence_coverage"]["fraction_of_completed_cases"], 1)
        for entry in result["per_case"]:
            self.assertTrue(entry["donor_recovery"]["effect"])
            self.assertFalse(entry["recipient_task_correct"]["effect"])
        # Full donor is a distinct positive control and can have a larger dose.
        self.assertEqual(result["instrument"]["max_matched_dose_spread"], 0)

    def test_strongest_absolute_control_can_refute_targeted_advantage(self):
        case = self.manifest["cases"][0]
        record = record_for(case, effect_probability=0.12, control_probability=0.0)
        result = scoring.summarize([record], [case])
        self.assertAlmostEqual(result["primary_root_mean"], -0.08)
        self.assertEqual(result["n_eligible_units"], 1)

    def test_failed_competence_and_instrument_cases_remain_accounted(self):
        bad = deepcopy(self.records)
        first = self.manifest["cases"][0]
        bad[0]["clean"]["recipient"]["probabilities"] = distribution(first["recipient"]["label"], 0.3)
        bad[1]["arms"]["sham"]["physical_dose"] = 2.0
        bad[2]["arms"]["effect"]["protected_readout_max_abs"] = 0.001
        bad[3] = {"case_id": bad[3]["case_id"], "root_id": bad[3]["root_id"],
                  "status": "error", "reason": "memory budget exceeded"}
        result = scoring.summarize(bad, self.manifest)
        self.assertEqual(result["n_attempted_cases"], 4)
        self.assertEqual(result["n_eligible_units"], 0)
        self.assertIsNone(result["primary_root_mean"])
        self.assertEqual(result["case_status_counts"],
                         {"ERROR": 1, "INELIGIBLE_BASE_TASK": 1, "INELIGIBLE_INSTRUMENT": 2})
        self.assertEqual(result["ineligibility_reason_counts"]["base_task_incompetence"], 1)
        self.assertEqual(result["n_completed_cases"], 3)
        self.assertEqual(result["competence_coverage"]["all_clean_correct_cases"], 2)
        self.assertAlmostEqual(result["competence_coverage"]["fraction_of_completed_cases"], 2 / 3)
        self.assertEqual(result["competence_coverage"]["fraction_of_planned_cases"], 0.5)

    def test_cross_basin_calibration_has_separate_root_summary(self):
        records = deepcopy(self.records)
        calibration = self.manifest["cases"][0]
        self.assertEqual(calibration["d"], 13)
        records[0]["arms"]["effect"]["probabilities"] = distribution(
            calibration["effect_donor"]["label"], 0.35)
        result = scoring.summarize(records, self.manifest)
        same, cross = (result["basin_strata"][name] for name in ("same_basin", "cross_basin"))
        self.assertEqual((same["n_units"], cross["n_units"]), (3, 1))
        self.assertAlmostEqual(same["primary_root_mean"], 0.63)
        self.assertAlmostEqual(cross["primary_root_mean"], 0.23)
        self.assertAlmostEqual(result["primary_root_mean"], 0.53)
        missing = scoring.summarize(records[1:], self.manifest)
        self.assertEqual(missing["basin_strata"]["cross_basin"]["n_units"], 1)
        self.assertEqual(missing["basin_strata"]["cross_basin"]["n_eligible_units"], 0)
        self.assertIsNone(missing["basin_strata"]["cross_basin"]["primary_root_mean"])

    def test_root_averaging_is_equal_weight_and_requires_complete_root(self):
        manifest = cases.generate_manifest(2, (1, 2), cases_per_root=2)
        records = [record_for(case) for case in manifest["cases"]]
        root0 = manifest["root_ids"][0]
        for case, record in zip(manifest["cases"], records):
            if case["root_id"] == root0:
                record["arms"]["effect"]["probabilities"] = distribution(case["effect_donor"]["label"], 0.35)
        result = scoring.summarize(records, manifest)
        self.assertAlmostEqual(result["primary_root_mean"], (0.23 + 0.63) / 2)
        missing = scoring.summarize(records[1:], manifest)
        self.assertEqual(missing["n_units"], 2)
        self.assertEqual(missing["n_eligible_units"], 1)
        self.assertEqual(missing["case_status_counts"]["NOT_ATTEMPTED"], 1)
        self.assertAlmostEqual(missing["primary_root_mean"], 0.63)

    def test_nonfinite_probabilities_duplicate_ids_and_bool_roots_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), True, -0.1, 1.1):
            records = deepcopy(self.records)
            records[0]["arms"]["effect"]["probabilities"]["A"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                scoring.summarize(records, self.manifest)
        with self.assertRaises(ValueError):
            scoring.summarize(self.records + [self.records[0]], self.manifest)
        record = deepcopy(self.records[0])
        record["root_id"] = True
        with self.assertRaises(ValueError):
            scoring.summarize([record], self.manifest)
        record = deepcopy(self.records[0])
        record["case_id"] = "unknown"
        with self.assertRaises(ValueError):
            scoring.summarize([record], self.manifest)
        record = deepcopy(self.records[0])
        del record["arms"]["sham"]
        with self.assertRaises(ValueError):
            scoring.summarize([record], self.manifest)

    def test_identity_and_admission_controls(self):
        record = deepcopy(self.records[0])
        record["arms"]["identity"]["physical_dose"] = 1.0
        record["arms"]["random"]["admission"] = False
        result = scoring.summarize([record], [self.manifest["cases"][0]])
        self.assertEqual(result["n_eligible_units"], 0)
        reasons = result["per_case"][0]["reasons"]
        self.assertIn("identity_nonzero_dose", reasons)
        self.assertIn("rejected_random", reasons)
        self.assertEqual(result["arm_admission"]["random"]["rejected"], 1)

    def test_prospective_bf16_delivered_dose_allowance(self):
        case = self.manifest["cases"][0]
        record = deepcopy(self.records[0])
        record["arms"]["effect"]["physical_dose"] = 1.0
        record["arms"]["sham"]["physical_dose"] = 0.98
        record["arms"]["random"]["physical_dose"] = 0.99
        accepted = scoring.summarize([record], [case])
        self.assertEqual(accepted["n_eligible_units"], 1)
        self.assertEqual(accepted["instrument"]["dose_relative_tolerance"], 0.02)
        self.assertEqual(accepted["instrument"]["dose_absolute_tolerance"], 1e-6)
        self.assertEqual(accepted["scientific_variation_floor"], 0.05)
        record["arms"]["sham"]["physical_dose"] = 0.979
        rejected = scoring.summarize([record], [case])
        self.assertEqual(rejected["n_eligible_units"], 0)
        self.assertIn("unmatched_physical_doses", rejected["per_case"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
