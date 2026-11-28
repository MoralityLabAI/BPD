from fractions import Fraction as F
import unittest
from admission import assess


def case(G,a,C,u,**kw):
    params=dict(dose=1,collateral=F(1,4),alpha=0,gamma=0,
                target_remainder=0,protected_remainder=0,min_gain=F(1,4))
    params.update(kw)
    return assess(G,a,C,u,**params)


class AdmissionTests(unittest.TestCase):
    def test_exact_selective(self):
        r=case([[1,0],[0,1]],[1,0],[[0,1]],[1,0])
        self.assertEqual(r["verdict"],"CONDITIONAL_REUSE_CANDIDATE")
        self.assertEqual(F(r["target_lower"]),1)
        self.assertEqual(F(r["regret_upper"]),0)

    def test_stale_protection_refuses(self):
        r=case([[1]],[1],[[0]],[1],gamma=F(1,2))
        self.assertFalse(r["safe"])
        self.assertEqual(r["verdict"],"REFRESH_OR_ORDINARY")

    def test_no_headroom(self):
        r=case([[4]],[F(1,8)],[[0]],[0])
        self.assertEqual(r["verdict"],"NO_HEADROOM_IN_DECLARED_SPAN")
        self.assertEqual(F(r["target_upper"]),F(1,16))

    def test_nonlinear_remainder(self):
        r=case([[1]],[1],[[0]],[1],protected_remainder=F(1,2))
        self.assertFalse(r["safe"])

    def test_relaxed_budget_needed_for_upper(self):
        # Cached C=1, actual C=0 is within gamma=1. True optimum is 1.
        r=case([[1]],[1],[[1]],[1],collateral=0,gamma=1)
        self.assertGreaterEqual(F(r["target_upper"]),1)
        self.assertFalse(r["safe"])

    def test_nonorthogonal_metric_and_exact_boundary(self):
        r=case([[2,1],[1,2]],[1,-1],[[1,1]],[F(1,2),F(-1,2)],
               gamma=F(1,8),protected_remainder=F(1,8))
        self.assertTrue(r["safe"])
        self.assertEqual(F(r["physical_dose_squared"]),F(1,2))
        self.assertEqual(F(r["collateral_upper"]),F(1,4))

    def test_small_exact_grid(self):
        # Explicit quadratic true responses with stated uniform remainders.
        # true f=(3/2)x+(1/8)x², true g=(1/2)x+(1/8)x².
        r=case([[1]],[1],[[F(1,4)]],[F(1,4)],collateral=F(1,2),
               alpha=F(1,2),gamma=F(1,4),target_remainder=F(1,8),
               protected_remainder=F(1,8),min_gain=F(1,8))
        lower,upper=F(r["target_lower"]),F(r["target_upper"])
        x=F(1,4)
        self.assertLessEqual(lower,F(3,2)*x+x*x/8)
        for k in range(-64,65):
            x=F(k,64)
            f=F(3,2)*x+x*x/8
            g=F(1,2)*x+x*x/8
            if abs(g)<=F(1,2):
                self.assertLessEqual(f,upper)

    def test_invalid(self):
        for G,u in (([[0]],[0]),([[1]],[2])):
            with self.assertRaises(ValueError):
                case(G,[1],[[0]],u)
        with self.assertRaises(ValueError):
            case([[1]],[1],[[0]],[1],dual_multipliers=(-1,))
        with self.assertRaises(ValueError):
            case([[1]],[1],[[0]],[1],alpha=.1)


if __name__=="__main__":
    unittest.main()
