"""Recompute and audit BPD public small-sample evidence without opening test."""
from __future__ import annotations

import unittest
from fractions import Fraction as Q
from pathlib import Path

from mechinterp.fsi01_bpd_audit import ROOT, SOURCES, audit
from spiral_ln.evaluation_finite_sample import binomial_two_sided, holm_adjust


class RetrospectiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt=audit()

    def test_difficulty_strata_and_unique_roots(self):
        ls=self.receipt["loopscope02"]
        self.assertEqual(ls["n_cases"],256)
        self.assertEqual(ls["board_origins"],256)
        self.assertEqual(sorted(ls["strata"].values()),[64]*4)

    def test_exact_paired_counts_match_original_report(self):
        cases=self.receipt["loopscope02"]["contrasts"]
        self.assertEqual(
            (cases["trm_target_step16_vs_step1"]["candidate_only"],
             cases["trm_target_step16_vs_step1"]["baseline_only"]), (23,4))
        self.assertEqual(
            (cases["trm_target_step16_vs_donor_zH"]["candidate_only"],
             cases["trm_target_step16_vs_donor_zH"]["baseline_only"]), (32,4))
        self.assertEqual(
            (cases["trm_board_clean16_vs_fiber_erased"]["candidate_only"],
             cases["trm_board_clean16_vs_fiber_erased"]["baseline_only"]), (209,0))
        self.assertEqual(
            (cases["trm_target_step16_vs_qwen4b"]["candidate_only"],
             cases["trm_target_step16_vs_qwen4b"]["baseline_only"]), (189,3))

    def test_preexisting_scores_cannot_be_promoted_to_confirmation(self):
        for result in self.receipt["loopscope02"]["contrasts"].values():
            self.assertEqual(result["hypothesis_preregistered"],False)
            self.assertEqual(result["sealed_readout"],False)
            self.assertNotIn(result["verdict"],("confirmed","exact_finite_result"))

    def test_arg03_comparison_family_is_exploratory(self):
        x=self.receipt["arg03_dev96"]
        self.assertEqual((x["n_cases"],x["n_arms"],x["n_tests_reported"]),(96,40,103))
        self.assertEqual(x["nominal_p_below_005"],19)
        self.assertEqual(x["holm_p_below_005"],0)
        self.assertEqual(
            x["smallest_tests"][0]["p_exact"],str(binomial_two_sided(22,27)))
        self.assertAlmostEqual(x["smallest_tests"][0]["holm_all_reported"],0.1559131145477295,places=9)

    def test_source_scope_does_not_include_sealed_test(self):
        for path in SOURCES.values():
            self.assertNotIn("test.jsonl",path)
            self.assertTrue((ROOT/path).exists())
        self.assertIn("held-out ARG-03 test unaccessed",self.receipt["status"])


if __name__=="__main__":
    unittest.main()
