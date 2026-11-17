"""BPD-OUT03 CPU verification for exact and numerical gradient geometry."""
from __future__ import annotations

import math
import unittest
import numpy as np

from mechinterp.dark_fiber_gyms.smc23_gradient_split import (
    exact_rank2_counterexample, certificate, sample_tangent_sham,
)


class TestSMC23GradientDecomposition(unittest.TestCase):
    def test_exact_same_feature_opposite_task_effect(self):
        a=exact_rank2_counterexample(2)
        b=exact_rank2_counterexample(-2)
        self.assertEqual(a["axial_term"],b["axial_term"])
        self.assertEqual(a["kill_norm_squared"],b["sham_norm_squared"])
        self.assertEqual(a["paired_margin_damage"],-1)
        self.assertEqual(b["paired_margin_damage"],3)

    def test_physical_selective_kill_does_not_fix_task_sign(self):
        A=np.array([[1.,0.,0.],[0.,1.,0.]])
        s=np.array([0.,0.,1.])
        bad=certificate(A,[1,0,2],0,1,s)
        good=certificate(A,[3,0,.2],0,1,s)
        self.assertAlmostEqual(bad["paired_linear_margin_damage"],-1.)
        self.assertAlmostEqual(good["paired_linear_margin_damage"],2.8)
        self.assertFalse(bad["any_sham_positive_sufficient"])
        self.assertTrue(good["any_sham_positive_sufficient"])
        self.assertAlmostEqual(bad["dual_gradient_coeff"],1.)
        self.assertAlmostEqual(bad["task_gradient_tangent_norm"],2.)

    def test_full_rank_randomized_48d_certificates(self):
        rng=np.random.default_rng(20262317)
        for idx in range(64):
            A=rng.normal(size=(4,48))
            gradient=rng.normal(size=48)
            amplitude=.2+float(rng.random())
            b=np.zeros(4);b[idx%4]=amplitude
            delta=A.T@np.linalg.solve(A@A.T,b)
            sham=sample_tangent_sham(A,float(np.linalg.norm(delta)),seed=idx+100)
            c=certificate(A,gradient,idx%4,amplitude,sham,curvature_bound=1.5)
            self.assertLessEqual(c["all_tangent_sham_linear_damage_range"][0]-1e-10,
                                 c["paired_linear_margin_damage"])
            self.assertLessEqual(c["paired_linear_margin_damage"],
                                 c["all_tangent_sham_linear_damage_range"][1]+1e-10)
            self.assertAlmostEqual(c["paired_linear_margin_damage"],
                                   c["axial_term"]-c["tangent_sham_term"],places=10)
            self.assertEqual(c["physical_rank"],4)

    def test_curvature_remainder_for_exact_quadratic_margin(self):
        rng=np.random.default_rng(103)
        A=rng.normal(size=(3,12))
        grad=rng.normal(size=12)
        h=rng.normal(size=(12,12))
        H=(h+h.T)/2
        H=H/max(np.linalg.svd(H,compute_uv=False))*1.7
        amp=.4
        b=np.array([amp,0.,0.])
        delta=A.T@np.linalg.solve(A@A.T,b)
        sham=sample_tangent_sham(A,np.linalg.norm(delta),seed=19)
        rec=certificate(A,grad,0,amp,sham,curvature_bound=1.7)
        f=lambda x: float(grad@x + .5*x@H@x)
        true=f(-sham)-f(-delta)
        lower,upper=rec["certified_nonlinear_interval_for_this_sham"]
        self.assertLessEqual(lower-1e-10,true)
        self.assertLessEqual(true,upper+1e-10)

    def test_frame_covariance(self):
        rng=np.random.default_rng(3307)
        A=rng.normal(size=(4,10))
        grad=rng.normal(size=10)
        b=np.array([0,.6,0,0])
        delta=A.T@np.linalg.solve(A@A.T,b)
        sham=sample_tangent_sham(A,np.linalg.norm(delta),seed=101)
        Q,_=np.linalg.qr(rng.normal(size=(10,10)))
        a=certificate(A,grad,1,.6,sham)
        b=certificate(A@Q.T,Q@grad,1,.6,Q@sham)
        for field in ("axial_term","tangent_sham_term",
                      "paired_linear_margin_damage","dose",
                      "task_gradient_tangent_norm"):
            self.assertAlmostEqual(a[field],b[field],places=10)

    def test_invalid_geometry_rejected(self):
        A=np.array([[1.,0.,0.],[0.,1.,0.]])
        with self.assertRaisesRegex(ValueError,"tangent"):
            certificate(A,[1,0,2],0,1,[1,0,0])
        with self.assertRaisesRegex(ValueError,"rank deficient"):
            certificate(np.array([[1.,0.,0.],[1.,0.,0.]]),[1,0,2],0,1,[0,0,1])
        with self.assertRaisesRegex(ValueError,"positive"):
            certificate(A,[1,0,2],0,0,[0,0,1])


if __name__=="__main__":
    unittest.main()
