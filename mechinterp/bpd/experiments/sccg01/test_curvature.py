import math
from fractions import Fraction as F
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from mechinterp.bpd.experiments.sccg01.curvature import (
    rms_bounds, reader_bounds, recurrence_tube, fixed_radius_control,
)


class CurvatureTests(unittest.TestCase):
    def test_exact_squared_bound(self):
        # Rational grid independently verifies the polynomial expression,
        # the universal 4/3 factor, and global positive-epsilon constant².
        for i in range(101):
            s=F(i,100)
            for j in range(101):
                z=F(j,100)
                direct=s*((1+(2-3*s)*z)**2+4*z*(1-z))
                polynomial=s+2*s*(4-3*s)*z-3*s*s*(4-3*s)*z*z
                self.assertEqual(direct,polynomial)
                self.assertLessEqual(direct,F(4,3))
                self.assertLessEqual(direct*(1-s)**2,F(2304,3125))
        self.assertEqual(9*F(1,5)*(1-F(1,5))**4,F(2304,3125))

    def test_radial_sharp_and_singular(self):
        for d in (1,2,7):
            eps=.125
            x=torch.zeros(d,dtype=torch.float64)
            x[0]=math.sqrt(d*eps/4)
            h=torch.autograd.functional.hessian(
                lambda v:(v/(v.square().mean()+eps).sqrt())[0], x)
            b=rms_bounds(x.numpy(),0.,eps)
            self.assertAlmostEqual(abs(float(h[0,0])),b["curvature"],places=11)
            self.assertAlmostEqual(b["curvature"],48/(25*math.sqrt(5*d)*eps),places=11)
        with self.assertRaises(ValueError):
            rms_bounds([1.,0.],1.,0.)

    def test_autograd_reader_and_finite_remainder(self):
        rng=np.random.default_rng(42)
        for d in (2,4,8):
            for _ in range(8):
                x=rng.normal(size=d)
                a=rng.normal(size=d)
                q=rng.normal(size=d)
                g=rng.uniform(.25,1.5,size=d)
                eps=.05
                radius=.2
                t=lambda z:torch.tensor(z,dtype=torch.float64)
                ta,tq,tg=t(a),t(q),t(g)
                read=reader_bounds(a,x,q,radius,eps,g)
                norm=lambda v:tg*v/(v.square().mean()+eps).sqrt()
                n2=norm(ta)
                def gap(v):
                    return tq@(norm(v)-n2)/math.sqrt(d)
                def msg(v):
                    n=norm(v)
                    return n2+torch.sigmoid(gap(v))*(n-n2)
                delta=rng.normal(size=d)
                delta*=radius/np.linalg.norm(delta)
                for frac in (0.,.5,1.):
                    v=t(x+frac*delta)
                    H=torch.autograd.functional.hessian(gap,v).numpy()
                    self.assertLessEqual(np.linalg.norm(H,2),read["gap_curvature"]+1e-10)
                    q2=t(rng.normal(size=d))
                    q2=q2/q2.norm()
                    Hm=torch.autograd.functional.hessian(lambda z:q2@msg(z),v).numpy()
                    self.assertLessEqual(np.linalg.norm(Hm,2),read["message_curvature"]+1e-10)
                v=t(x)
                J=torch.autograd.functional.jacobian(gap,v)
                err=abs(float(gap(v+t(delta))-gap(v)-J@t(delta)))
                self.assertLessEqual(err,.5*read["gap_curvature"]*radius**2+1e-10)

    def test_nonlinear_tube(self):
        rng=np.random.default_rng(7)
        # F_t(x) = M_t x + c_t x² coordinatewise; clean zero.
        M=[np.array([[.8,.2],[-.1,.7]]),np.array([[.6,-.3],[.2,.9]]),
           np.array([[.9,.1],[.4,.6]])]
        cs=[.2,.1,.3]
        js=[float(np.linalg.norm(m,2)) for m in M]
        K=[2*c for c in cs]
        R=[1.]*3
        G=[j+k*r for j,k,r in zip(js,K,R)]
        e=[.08,.12,.1]
        tube=recurrence_tube(js,G,K,[1.]*3,e,R)["rows"]
        for _ in range(64):
            x=np.zeros(2); z=x.copy()
            for i,(m,c,dose) in enumerate(zip(M,cs,e)):
                u=rng.normal(size=2)
                u*=dose*rng.random()/np.linalg.norm(u)
                x=m@x+c*x*x+u
                z=m@z+u
                self.assertLessEqual(np.linalg.norm(x),tube[i+1]["radius_bound"]+1e-10)
                self.assertLessEqual(np.linalg.norm(x-z),tube[i+1]["remainder_bound"]+1e-10)
        with self.assertRaises(ValueError):
            recurrence_tube(js,G,K,[1.]*3,e,[0.]*3)

    def test_cap_optimum(self):
        x=[1.,0.,1.]; W=np.eye(3); C=[[0.,0.,1.]]
        a=[0.,1.,0.]
        result=fixed_radius_control(x,W,C,a,1.,.25)
        self.assertAlmostEqual(result["target_gain"],math.sqrt(3)/2/math.sqrt(2.75))
        self.assertLess(result["protected_gap_residual"],1e-12)
        self.assertLess(result["radius_residual"],1e-12)
        self.assertAlmostEqual(result["dose"],1.)
        # Exhaustive fine angular samples as independent feasible lower bound.
        theta=np.linspace(-math.pi,math.pi,20001)
        for a in ([0.,1.,0.],[-1.,0.,0.],[1.,.3,0.]):
            for dose in (.2,.8,2.):
                ans=fixed_radius_control(x,W,C,a,dose,.25)
                Ds=np.column_stack([np.cos(theta)-1,np.sin(theta),np.zeros_like(theta)])
                feasible=Ds[np.linalg.norm(Ds,axis=1)<=dose]
                oracle=np.max(feasible@np.array(a))/math.sqrt(2.75)
                self.assertGreaterEqual(ans["target_gain"]+1e-10,oracle)
                self.assertLess(ans["target_gain"]-oracle,5e-4)
                self.assertLessEqual(ans["dose"],dose+1e-12)
                self.assertLess(ans["radius_residual"],1e-12)
                self.assertLess(ans["protected_gap_residual"],1e-12)

    def test_one_dimensional_and_zero_radius(self):
        x=[1.,0.]
        W=np.array([[1.],[0.]])
        C=np.zeros((0,2))
        self.assertEqual(fixed_radius_control(x,W,C,[-1.,0.],1.99,.1)["dose"],0)
        self.assertAlmostEqual(fixed_radius_control(x,W,C,[-1.,0.],2.,.1)["target_gain"],
                               2/math.sqrt(1.2))
        self.assertEqual(fixed_radius_control([0.,1.],W,C,[1.,0.],2.,.1)["dose"],0)
        self.assertEqual(fixed_radius_control(x,np.zeros((2,0)),C,[1.,0.],1.,.1)["dose"],0)

    def test_native_constructed_fixture(self):
        path=Path(__file__).resolve().parents[1]/"inkling_component_flow/evidence/source_fixture.json"
        for case in json.loads(path.read_text())["cases"]:
            f=case["flow"]; spec=f["reader_spec"]
            W=np.array(f["physical_writes"])[:2].T
            rows=np.array(spec["queries"])*np.array(spec["rms_scale"])
            x=f["block3"]
            out=fixed_radius_control(x,W,rows[1:2],rows[0],.125,spec["eps"])
            self.assertLessEqual(out["dose"],.125+1e-10)
            self.assertLess(out["protected_gap_residual"],1e-10)
            self.assertLess(out["radius_residual"],1e-10)


if __name__=="__main__":
    unittest.main()
