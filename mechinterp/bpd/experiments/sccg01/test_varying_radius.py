"""Exact strict separation fixture for SCCG-02B."""
from fractions import Fraction as F
import unittest


class VaryingRadiusTests(unittest.TestCase):
    def test_strict_separation(self):
        x=(F(1),F(1))
        y=(F(1,7),F(5,7))
        tau=F(5,7)
        rho2=lambda z:sum(t*t for t in z)+1
        self.assertEqual(rho2(y),tau*tau*rho2(x))
        self.assertEqual(y[1],tau*x[1])
        self.assertEqual(sum((v-u)**2 for u,v in zip(x,y)),F(40,49))
        self.assertLess(F(40,49),1)
        # Protected raw coordinate y2=x2 at fixed radius, hence y1=±1.
        self.assertGreater((F(-1)-x[0])**2,1)
        # Target score difference times rho0 (strict positive).
        self.assertEqual(-y[0]/tau+x[0],F(4,5))

if __name__=="__main__":
    unittest.main()
