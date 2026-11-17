"""BPD-OUT05 exact sign/MAE divergence and exploratory receipt tests."""
from __future__ import annotations

from fractions import Fraction as F
import unittest

from mechinterp.dark_fiber_gyms.smc25_sign_vs_mae import (
    sign,weighted_median_exact,sign_stability_certificate,
    exact_sign_mae_counterexample,replay_exploratory_sign_effects
)


class TestsSMC25SignVsMae(unittest.TestCase):
    def test_exact_sign_better_mae_worse_witness(self):
        x=exact_sign_mae_counterexample()
        self.assertEqual(x["calibrated_median"],"10")
        self.assertEqual(x["uniform_median"],"-1")
        self.assertEqual(x["calibrated_sign_accuracy"],"7/10")
        self.assertEqual(x["uniform_sign_accuracy"],"3/10")
        self.assertEqual(x["calibrated_expected_MAE"],"123/10")
        self.assertEqual(x["uniform_expected_MAE"],"41/10")

    def test_sign_stability_under_small_total_variation(self):
        result=sign_stability_certificate(
            [-1,1],[F(2,5),F(3,5)],[F(9,20),F(11,20)])
        self.assertTrue(result["guaranteed_same_sign"])
        self.assertTrue(result["actual_same_sign"])
        self.assertEqual(result["total_variation"],"1/20")

    def test_sign_flip_is_possible_above_tv_margin(self):
        result=sign_stability_certificate(
            [-1,1],[F(2,5),F(3,5)],[F(7,10),F(3,10)])
        self.assertFalse(result["guaranteed_same_sign"])
        self.assertFalse(result["actual_same_sign"])

    def test_two_class_tie_and_zero_mass_admission(self):
        self.assertEqual(weighted_median_exact([-1,1],[F(1,2),F(1,2)]),-1)
        with self.assertRaisesRegex(ValueError,"nonzero"):
            sign_stability_certificate([-1,0,1],
                                      [F(1,3)]*3,[F(1,3)]*3)

    def test_exact_median_rejects_bad_probabilities(self):
        with self.assertRaisesRegex(ValueError,"probability"):
            weighted_median_exact([-1,1],[F(1,3),F(1,3)])
        with self.assertRaisesRegex(ValueError,"probability"):
            weighted_median_exact([-1,1],[-1,2])

    @staticmethod
    def fixture_scored():
        pairs=("newB_501_SAE_71","newB_501_SAE_89",
               "newB_701_SAE_71","newB_701_SAE_89")
        families=("smc25_novel30","smc25_novel34")
        fake={}
        for family in families:
            fake[family]={}
            for component in pairs:
                puzzle=[]
                for i in range(64):
                    goal={"status":"eligible",
                          "predictions":{"calibrated_median":1.,
                                         "uniform_median":-1.},
                          "observed":{"true_D":1.}}
                    puzzle.append({"puzzle_index":i,
                                   "goals":[goal]+[{"status":"dose_cap"}]*7})
                fake[family][component]={
                    "per_puzzle":puzzle,
                    "summary":{"eligible_goals":64,
                               "calibrated_median_sign_correct":64,
                               "uniform_median_sign_correct":0},
                }
        return {"schema":"SMC-25/finite-native-all-candidate-scored-20261117-v1",
                "n_unique_puzzles":128,"n_goal_slots":4096,
                "test_data":fake}

    def test_receipt_replay_preserves_whole_puzzle_clusters(self):
        receipt=replay_exploratory_sign_effects(self.fixture_scored(),draws=200)
        self.assertEqual(len(receipt["exploratory_sign_by_cell"]),8)
        self.assertEqual(receipt["n_eligible_goal_SAE_measurements"],512)
        for item in receipt["exploratory_sign_by_cell"]:
            self.assertEqual(item["eligible_goals"],64)
            self.assertEqual(item["calibrated_minus_uniform_goal_count"],64)
            self.assertEqual(item["puzzle_mean_sign_gain"],1.)
            self.assertEqual(item["posthoc_puzzle_bootstrap95"],[1.,1.])
            self.assertEqual(item["posthoc_status"],"EXPLORATORY_AFTER_SMC25_RESULT")

    def test_ineligible_goal_denominators_remain_in_receipt(self):
        source=self.fixture_scored()
        item=source["test_data"]["smc25_novel30"]["newB_501_SAE_71"]
        item["per_puzzle"][0]["goals"][0]["status"]="dose_cap"
        item["summary"]["eligible_goals"]-=1
        item["summary"]["calibrated_median_sign_correct"]-=1
        result=replay_exploratory_sign_effects(source,draws=200)
        first=result["exploratory_sign_by_cell"][0]
        self.assertEqual(first["eligible_goals"],63)
        self.assertEqual(first["eligible_puzzles"],63)
        self.assertGreater(first["puzzle_mean_sign_gain"],.99)


if __name__=="__main__":
    unittest.main()
