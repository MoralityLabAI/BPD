from fractions import Fraction as F
import unittest
import sympy as s
from mechinterp.bpd.experiments.sccg01.exact_radius import (
    t,signs_feasible,ReaderProblem,
)


class ExactRadiusTests(unittest.TestCase):
    def test_sign_cells(self):
        self.assertIsNotNone(signs_feasible([t-1,1-t]))
        self.assertIsNone(signs_feasible([t-1,-t]))
        root=signs_feasible([t*t-2,2-t*t,t])
        self.assertEqual(root["kind"],"algebraic_root")
        lo,hi=map(s.Rational,root["interval"])
        self.assertGreater(lo,0)
        self.assertLess(lo*lo,2)
        self.assertGreater(hi*hi,2)
        self.assertIsNotNone(signs_feasible([(t-1)*(t-3),2-t,t]))
        self.assertIsNone(signs_feasible([t*t-2,2-t*t,t-2]))
        self.assertIsNotNone(signs_feasible([t*t+1]))
        self.assertIsNone(signs_feasible([-t*t-1]))

    def test_variable_radius_witness(self):
        p=ReaderProblem([1,1],[[1,0],[0,1]],[[0,1]],[-1,0],1,1)
        self.assertTrue(p.decide(F(4,5))["feasible"])
        self.assertFalse(p.decide(2)["feasible"])
        # Independent optimization on the exact protected ellipse:
        # y2=t, y1²=2t²-1, dose<=1. Verify a feasible analytic
        # threshold and reject a larger one through exact sign cells.
        r=p.bracket(12)
        self.assertGreaterEqual(s.Rational(r["lower"]),s.Rational(799,1000))
        self.assertLessEqual(s.Rational(r["width"]),s.Rational(6,4096))

    def test_independent_quartic_optimum(self):
        # For x=(1,1), C=(0,1), a=(-1,0), e=m=1,
        # protected states obey y2=tau, y1^2=2*tau^2-1.
        # The dose condition requires y1 >= tau*(3*tau-2)/2 > 0.
        # The objective 1-y1/tau decreases with tau.
        # Thus the smallest feasible tau is the first positive root
        # of 9*tau^4-12*tau^3-4*tau^2+4.
        quartic=s.Poly(9*t**4-12*t**3-4*t**2+4,t)
        l=s.Rational(5161852,7293017)
        h=s.Rational(5403205,7634017)
        self.assertGreater(l*l,s.Rational(1,2))
        self.assertLess(h,s.Rational(3,4))
        self.assertGreater(quartic.eval(l),0)
        self.assertLess(quartic.eval(h),0)
        self.assertEqual(quartic.count_roots(0,l),0)
        self.assertEqual(quartic.count_roots(l,h),1)
        lo,hi=2-3*h/2,2-3*l/2
        p=ReaderProblem([1,1],[[1,0],[0,1]],[[0,1]],[-1,0],1,1)
        self.assertTrue(p.decide(lo)["feasible"])
        self.assertFalse(p.decide(hi)["feasible"])

    def test_zero_dim_and_fixed_isolated(self):
        p=ReaderProblem([1,1],[[1],[0]],[[0,1]],[-1,0],1,1)
        self.assertTrue(p.fixed)
        self.assertTrue(p.decide(0)["feasible"])
        self.assertFalse(p.decide(F(1,100))["feasible"])
        p=ReaderProblem([1,1],[[1],[0]],[[0,1]],[-1,0],2,1)
        self.assertTrue(p.decide(2)["feasible"])
        self.assertFalse(p.decide(F(201,100))["feasible"])
        p=ReaderProblem([1,1],[[0],[1]],[[0,1]],[1,0],0,1)
        self.assertEqual(p.U.cols,0)
        self.assertTrue(p.decide(0)["feasible"])
        self.assertFalse(p.decide(F(1,100))["feasible"])

    def test_unprotected_known_optimum(self):
        # x=0, eps_mass=1, dose=1; optimal scaled gain 1/sqrt(2).
        for d in (1,2,3):
            x=[0]*d
            a=[1]+[0]*(d-1)
            W=[[int(i==j) for j in range(d)] for i in range(d)]
            p=ReaderProblem(x,W,[],a,1,1)
            self.assertTrue(p.decide(F(7,10))["feasible"])
            self.assertFalse(p.decide(F(71,100))["feasible"])
            r=p.bracket(10)
            lo,hi=map(s.Rational,(r["lower"],r["upper"]))
            self.assertLessEqual(lo*lo,s.Rational(1,2))
            self.assertGreater(hi*hi,s.Rational(1,2))

    def test_dependent_writes_and_protection(self):
        p=ReaderProblem([1,1],[[1,2,0],[0,0,1]],
                        [[0,1],[0,2]],[-1,0],1,1)
        q=ReaderProblem([1,1],[[1,0],[0,1]],[[0,1]],[-1,0],1,1)
        for g in (0,F(1,2),F(4,5),1,2):
            self.assertEqual(p.decide(g)["feasible"],q.decide(g)["feasible"])

    def test_rank_zero_and_zero_target(self):
        p=ReaderProblem([1,2],[[],[]],[],[1,0],1,1)
        self.assertTrue(p.decide(0)["feasible"])
        self.assertFalse(p.decide(F(1,1000))["feasible"])
        p=ReaderProblem([1,2],[[1,0],[0,1]],[[1,0]],[0,0],1,1)
        self.assertTrue(p.decide(0)["feasible"])
        self.assertFalse(p.decide(F(1,1000))["feasible"])

    def test_input_contract(self):
        with self.assertRaises(ValueError):
            ReaderProblem([1.0],[[1]],[],[1],1,1)
        with self.assertRaises(ValueError):
            ReaderProblem([1],[[1]],[],[1],1,0)


if __name__=="__main__":
    unittest.main()
