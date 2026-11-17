"""BPD-OUT06 exact Bayes sign/abstention decisions and robust regret tests."""
from __future__ import annotations

from fractions import Fraction as Q
import unittest

from mechinterp.dark_fiber_gyms.smc27_bayes_abstention import (
    utility,effect_truth,posterior_classes,expected_actions_from_mass,
    bayes_candidate_action,weighted_median_effect,regret_certificate,
    exact_median_abstention_counterexample,
)


class TestCostAwareNativeAbstention(unittest.TestCase):
    def test_registered_ternary_utility_is_exact(self):
        self.assertEqual(utility(1,1),Q(9,10))
        self.assertEqual(utility(-1,1),Q(-11,10))
        self.assertEqual(utility(0,1),0)
        self.assertEqual(utility(0,0),1)
        self.assertEqual(utility(1,0),Q(-11,10))
        self.assertEqual(effect_truth(Q(3,20)),0)
        self.assertEqual(effect_truth(-Q(3,20)),0)
        self.assertEqual(effect_truth(Q(3,10)),1)
        with self.assertRaisesRegex(ValueError,"ternary"):
            utility(2,1)

    def test_cost_and_neutral_mass_shift_decision_boundary(self):
        r=expected_actions_from_mass(Q(7,10),Q(1,10),Q(1,5))
        self.assertEqual(r["optimal_action"],1)
        self.assertEqual(r["expected_utility"][1],Q(3,10))
        self.assertEqual(r["expected_utility"][0],Q(1,5))
        self.assertEqual(r["positive_mass_threshold"],Q(13,20))
        more_expensive=expected_actions_from_mass(Q(7,10),Q(1,10),Q(1,5),Q(3,10))
        self.assertEqual(more_expensive["optimal_action"],0)

    def test_abstain_on_exact_tie(self):
        r=expected_actions_from_mass(Q(3,5),Q(3,10),Q(1,10))
        self.assertEqual(r["expected_utility"][1],Q(1,10))
        self.assertEqual(r["expected_utility"][0],Q(1,10))
        self.assertEqual(r["optimal_action"],0)

    def test_median_sign_fails_optimal_abstention(self):
        r=exact_median_abstention_counterexample()
        self.assertEqual(r["median_action"],1)
        self.assertEqual(r["median_expected_utility"],Q(1,10))
        self.assertEqual(r["Bayes_action"],0)
        self.assertEqual(r["Bayes_expected_utility"],Q(3,10))
        self.assertEqual(r["improvement"],Q(1,5))

    def test_label_weighted_median_and_bayes_mass_are_consistent(self):
        effects=(-2,-1,0,0,1,1,2,2,2)
        probs=(Q(1,10),Q(1,10),Q(1,5),Q(0),Q(1,5),Q(1,5),Q(1,5),Q(0),Q(0))
        med=weighted_median_effect(effects,probs)
        r=bayes_candidate_action(effects,probs)
        self.assertEqual(med,1)
        self.assertEqual(r["p_plus"],Q(3,5))
        self.assertEqual(r["p_minus"],Q(1,5))
        self.assertEqual(r["p_neutral"],Q(1,5))
        self.assertEqual(r["optimal_action"],0)

    def test_prior_mismatch_regret_is_bounded_by_four_times_tv(self):
        effects=(1,-1)
        actual=(1,-1)
        p=(Q(7,10),Q(3,10))
        q=(Q(2,5),Q(3,5))
        r=regret_certificate(effects,actual,p,q)
        self.assertEqual(r["posterior_total_variation"],Q(3,10))
        self.assertEqual(r["proxy_finite_class_mismatch"],0)
        self.assertLessEqual(r["true_conditional_regret"],r["certified_regret_upper"])
        self.assertGreater(r["true_conditional_regret"],0)

    def test_finite_tangent_sign_mismatch_receives_explicit_budget(self):
        effects=(1,-1)
        actual=(-1,-1)
        q=(Q(3,4),Q(1,4))
        r=regret_certificate(effects,actual,q,q)
        self.assertEqual(r["posterior_total_variation"],0)
        self.assertEqual(r["proxy_finite_class_mismatch"],Q(3,4))
        self.assertGreater(r["true_conditional_regret"],0)
        self.assertLessEqual(r["true_conditional_regret"],r["certified_regret_upper"])

    def test_invalid_probability_and_negative_action_cost(self):
        with self.assertRaisesRegex(ValueError,"normalized"):
            expected_actions_from_mass(Q(1,2),Q(1,4),Q(1,4),Q(-1))
        with self.assertRaisesRegex(ValueError,"normalized"):
            posterior_classes((1,-1),(Q(1,3),Q(1,3)))
        with self.assertRaisesRegex(ValueError,"dimensions"):
            bayes_candidate_action((1,-1),(Q(1,2),Q(1,4),Q(1,4)))
        with self.assertRaisesRegex(ValueError,"ternary"):
            regret_certificate((1,-1),(2,-1),(Q(1,2),Q(1,2)),(Q(1,2),Q(1,2)))


if __name__=="__main__":
    unittest.main()
