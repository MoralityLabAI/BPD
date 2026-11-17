"""BPD-OUT04 standard-library/NumPy CPU regression of labelwise risk bounds."""
from __future__ import annotations

from fractions import Fraction
import unittest

import numpy as np

from mechinterp.dark_fiber_gyms.smc24_label_uncertainty import (
    class_margin_directions, squared_risk_certificate,
    distribution_mismatch_certificate, optimal_absolute_prediction,
    exact_two_label_absolute_floor,
)


class TestLabelConditionedFunctional(unittest.TestCase):
    def test_native_linear_tangent_matches_finite_margin(self):
        logits=np.array([3.0,2.0,1.0])
        J=np.array([[1.0,0.0],[0.0,1.0],[-1.0,1.0]])
        kill=np.array([1.0,0.0])
        sham=np.array([0.0,1.0])
        effects,rivals=class_margin_directions(logits,J,kill,sham)
        self.assertEqual(rivals.tolist(),[1,0,0])
        for y in range(3):
            c=int(rivals[y])
            lkill=logits-J@kill
            lsham=logits-J@sham
            actual=(lsham[y]-lsham[c])-(lkill[y]-lkill[c])
            self.assertAlmostEqual(effects[y],actual,places=12)

    def test_bayes_squared_risk_orthogonal_decomposition(self):
        rng=np.random.default_rng(202624)
        d=rng.normal(size=9)
        p=rng.dirichlet(np.ones(9))
        for candidate in (-2.0,0.0,1.5):
            r=squared_risk_certificate(d,p,candidate)
            self.assertAlmostEqual(
                r["submitted_squared_risk"],
                r["irreducible_squared_risk"]+r["prediction_excess_squared_risk"],
                places=12)
        mu=squared_risk_certificate(d,p,0)["bayes_mean_prediction"]
        opt=squared_risk_certificate(d,p,mu)
        self.assertAlmostEqual(opt["prediction_excess_squared_risk"],0.0,places=12)

    def test_sharp_tv_bound_and_mixed_prior(self):
        d=np.array([-2.0,3.0])
        extreme=distribution_mismatch_certificate(d,[1,0],[0,1])
        self.assertAlmostEqual(extreme["certified_probability_bias_upper"],5.0)
        self.assertAlmostEqual(extreme["absolute_prediction_bias"],5.0)
        mixed=distribution_mismatch_certificate(d,[.25,.75],[.4,.6])
        self.assertLessEqual(mixed["absolute_prediction_bias"],
                             mixed["certified_probability_bias_upper"]+1e-12)

    def test_exact_fraction_two_target_mae_floor(self):
        self.assertEqual(exact_two_label_absolute_floor(
            Fraction(1,4),Fraction(-3),Fraction(2)),Fraction(5,4))
        self.assertEqual(exact_two_label_absolute_floor(
            Fraction(1,2),Fraction(-2),Fraction(2)),Fraction(2))
        self.assertEqual(exact_two_label_absolute_floor(0,-3,2),0)
        with self.assertRaises(ValueError):
            exact_two_label_absolute_floor(Fraction(3,2),0,1)

    def test_conditioned_median_is_minimal_for_l1(self):
        d=np.array([-3.,2.]); p=np.array([.25,.75])
        answer=optimal_absolute_prediction(d,p)
        self.assertAlmostEqual(answer["optimal_median_prediction"],2.)
        self.assertAlmostEqual(answer["irreducible_mean_absolute_error"],1.25)
        candidates=np.linspace(-5,5,101)
        self.assertLessEqual(answer["irreducible_mean_absolute_error"],
                             min(float(p@np.abs(d-c)) for c in candidates)+1e-12)

    def test_equal_labelwise_effect_has_zero_bayes_floor(self):
        d=np.array([.3,.3,.3,.3])
        p=np.array([.2,.3,.4,.1])
        m=squared_risk_certificate(d,p,.3)
        a=optimal_absolute_prediction(d,p)
        self.assertLess(m["irreducible_squared_risk"],1e-27)
        self.assertAlmostEqual(a["irreducible_mean_absolute_error"],0.0,places=12)

    def test_physical_gauge_covariance(self):
        rng=np.random.default_rng(103)
        logits=rng.normal(size=9)
        J=rng.normal(size=(9,48))
        kill=rng.normal(size=48)
        sham=rng.normal(size=48)
        q,_=np.linalg.qr(rng.normal(size=(48,48)))
        a,_=class_margin_directions(logits,J,kill,sham)
        b,_=class_margin_directions(logits,J@q.T,q@kill,q@sham)
        self.assertTrue(np.allclose(a,b,atol=1e-11))

    def test_invalid_distributions_and_dimensions(self):
        with self.assertRaisesRegex(ValueError,"sum to one"):
            squared_risk_certificate([1.,2.],[.1,.1],0.)
        with self.assertRaisesRegex(ValueError,"invalid"):
            optimal_absolute_prediction([1.,2.],[1.1,-.1])
        with self.assertRaisesRegex(ValueError,"required"):
            class_margin_directions([1.,2.],np.zeros((2,3)),[1.,2.],[1.,2.])


if __name__=="__main__":
    unittest.main()
