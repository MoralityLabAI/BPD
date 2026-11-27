import json
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import minimize

from mechinterp.bpd.experiments.sccg01.frontier import (
    frontier, transport, reader_boundary_problem, protected_correction,
)
from mechinterp.bpd.experiments.sccg01.exact_witnesses import check


class Tests(unittest.TestCase):
    def test_exact(self):
        check()

    def test_closed_forms(self):
        r=frontier([1.,1.],[[1.,-1.]],1.,0.)
        self.assertAlmostEqual(r["lower"],2**.5)
        self.assertLess(r["collateral"],1e-10)
        r=frontier([1.,0.],[[1.,0.]],1.,.2)
        self.assertAlmostEqual(r["lower"],.2,places=9)
        self.assertLess(abs(r["gap"]),1e-9)
        r=frontier([1.,1.],[[1.,0.]],1.,.3)
        self.assertAlmostEqual(r["lower"],.3+(1-.09)**.5,places=8)
        self.assertLess(abs(r["gap"]),1e-8)
        r=frontier([1.,2.],np.zeros((0,2)),1.,.2)
        self.assertAlmostEqual(r["lower"],5**.5)
        self.assertEqual(frontier([1.],[[1.]],0.,0.)["lower"],0.)
        self.assertEqual(frontier([0.],[[1.]],1.,1.)["lower"],0.)

    def test_scipy_independent_comparison(self):
        rng=np.random.default_rng(727)
        for i in range(36):
            n=2+i%5
            a=rng.normal(size=n)
            C=rng.normal(size=(1+i%4,n))
            eps=.3+rng.random()
            eta=.05+2*rng.random()
            r=frontier(a,C,eps,eta)
            constraints=[
                dict(type="ineq",fun=lambda u:eps**2-u@u,jac=lambda u:-2*u),
                dict(type="ineq",fun=lambda u:eta**2-(C@u)@(C@u),
                     jac=lambda u:-2*C.T@(C@u))]
            oracle=minimize(lambda u:-a@u,np.zeros(n),jac=lambda u:-a,
                            constraints=constraints,method="SLSQP",
                            options={"ftol":1e-12,"maxiter":500})
            # SLSQP may emit a numerical status at an optimum. Require
            # primal feasibility and objective agreement independently.
            self.assertGreaterEqual(min(c["fun"](oracle.x) for c in constraints),-1e-7)
            self.assertLessEqual(r["dose"],eps+1e-10)
            self.assertLessEqual(r["collateral"],eta+1e-10)
            self.assertLess(abs(r["gap"]),1e-8)
            self.assertLess(abs(r["lower"]+oracle.fun),1e-6)

    def test_transport(self):
        J=[np.array([[1.,1.],[0.,1.]]),np.array([[1.,0.],[2.,1.]]),
           np.array([[2.,0.],[0.,3.]])]
        B=[np.array([[1.],[2.]]),np.eye(2),np.array([[0.],[1.]])]
        H=transport(J,B)
        u=np.array([.1,-.2,.3,.4])
        h=np.zeros(2)
        for j,b,v in zip(J,B,[u[:1],u[1:3],u[3:]]):
            h=j@h+b@v
        np.testing.assert_allclose(H@u,h)

    def test_native_reader_fixture(self):
        source=Path(__file__).resolve().parents[1]/"inkling_component_flow/evidence/source_fixture.json"
        raw=json.loads(source.read_text())
        for case in raw["cases"]:
            flow=case["flow"]
            p=reader_boundary_problem(flow,["component:0","component:1"],["Q"],["K","V"])
            self.assertEqual(p["scope"],"LOCAL_BOUNDARY_GAP_RESPONSE")
            r=frontier(p["A"][0],p["C"],.125,.01)
            edit=p["physical_basis"]@r["u"]
            self.assertLessEqual(np.linalg.norm(edit),.125+1e-12)
            W=np.array(flow["physical_writes"])[:2].T
            np.testing.assert_allclose(W@p["coefficient_map"]@r["u"],edit,atol=1e-12)
            gradients=np.array(flow["reader"]["gap_gradient_block3"])
            np.testing.assert_allclose(gradients[0]@edit,r["lower"],atol=1e-12)

    def test_redundant_writes(self):
        f=dict(physical_writes=[[1,0],[2,0],[0,0]],
               term_labels=["component:0","component:1","bias"],reader_ids=["Q","K"],
               reader=dict(gap_gradient_block3=[[1,0],[0,1]]))
        p=reader_boundary_problem(f,["component:0","component:1"],["Q"],["K"])
        self.assertEqual(p["rank"],1)
        with self.assertRaises(ValueError):
            reader_boundary_problem(f,["bias"],["Q"],["K"])

    def test_correction(self):
        g=lambda u:np.array([u[1]+u[0]**2])
        r=protected_correction(g,[[0.,1.]],[.1,0.],2.,.2)
        np.testing.assert_allclose(r["u"],[.1,-.01],atol=1e-12)
        self.assertLess(r["protected_residual"],1e-12)
        self.assertLessEqual(r["dose"],r["dose_bound"])
        with self.assertRaises(ValueError):
            protected_correction(lambda u:u*u,[[0.]],[.1],2.,1.)
        with self.assertRaises(ValueError):
            protected_correction(g,[[0.,1.]],[.3,0.],2.,1.)

    def test_validation(self):
        with self.assertRaises(ValueError):
            frontier([1.,np.nan],[[1.,0.]],1.,1.)
        with self.assertRaises(ValueError):
            frontier([1.,0.],[[1.]],1.,1.)
        with self.assertRaises(ValueError):
            frontier([1.],[[1.]],1.,-1.)


if __name__=="__main__":
    unittest.main()
