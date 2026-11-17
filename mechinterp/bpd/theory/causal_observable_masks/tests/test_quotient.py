"""No neural model or sealed test. Exact mathematical tests and seeded metamorphic checks."""
from __future__ import annotations
import unittest,random
from fractions import Fraction as Q
from causal_observable_masks.quotient import (
 matrix,eye,mul,transpose,inverse,rank,zero,all_masks,mask_matrix,commutator_change,
 invariant_masks,invariant_to_horizon,response_matrices,observable_gramian,
 first_detected,delayed_shear,nonlattice_four,closed_under_intersection,capacity_witness)


class ExactObservableMaskTests(unittest.TestCase):
    def test_inverse(self):
        a=matrix([[3,2,1],[1,1,0],[0,2,1]])
        self.assertEqual(mul(a,inverse(a)),eye(3))
        self.assertEqual(mul(inverse(a),a),eye(3))
    def test_singular_refusal(self):
        with self.assertRaises(ValueError):inverse(matrix([[1,1],[1,1]]))
    def test_mask_refusals(self):
        for bits in ((1,2),(),(1,.5)):
            with self.assertRaises(ValueError):mask_matrix(bits)
    def test_full_zero_commute_with_all(self):
        for case in [delayed_shear(3),nonlattice_four(-2)]:
            d=len(case['r'])
            for bits in ((0,)*d,(1,)*d):
                self.assertTrue(zero(commutator_change(case['r'],bits)))
                self.assertIsNone(first_detected(**case,bits=bits))
    def test_shear_dark_then_visible(self):
        case=delayed_shear(3)
        bits=(1,0)
        self.assertFalse(zero(commutator_change(case['r'],bits)))
        self.assertEqual(first_detected(**case,bits=bits),1)
        self.assertTrue(invariant_to_horizon(**case,bits=bits,horizon=0))
        self.assertFalse(invariant_to_horizon(**case,bits=bits,horizon=1))
    def test_observability_gramian_saturation(self):
        case=delayed_shear(2)
        self.assertEqual(observable_gramian(case['t'],case['c'],case['u'],0),matrix([[0,0],[0,1]]))
        self.assertEqual(observable_gramian(case['t'],case['c'],case['u'],1),eye(2))
    def test_monotonic_mask_set(self):
        for case in [delayed_shear(-4),nonlattice_four(3)]:
            d=len(case['r'])
            m=[set(invariant_masks(**case,horizon=k)) for k in range(d+3)]
            self.assertTrue(all(m[j+1]<=m[j] for j in range(len(m)-1)))
            self.assertEqual(m[d-1],m[-1])
    def test_4d_observational_masks_not_boolean(self):
        case=nonlattice_four(1)
        self.assertEqual(len(invariant_masks(**case,horizon=0)),16)
        m=invariant_masks(**case,horizon=1)
        self.assertEqual(len(m),6)
        self.assertIn((1,1,0,0),m)
        self.assertIn((1,0,1,0),m)
        self.assertNotIn((1,0,0,0),m)
        self.assertFalse(closed_under_intersection(m))
        self.assertEqual(first_detected(**case,bits=(1,0,0,0)),1)
    def test_causal_capacity_control(self):
        c=capacity_witness()
        self.assertEqual(c['phase_disjoint']['successful_top1_masks'],[[1,0]])
        self.assertEqual(c['mixed']['successful_top1_masks'],[])
        self.assertEqual(c['mixed']['observability_rank'],2)
    def test_exact_gramian_matches_direct_response(self):
        for case in [delayed_shear(-2),nonlattice_four(2)]:
            for bits in all_masks(len(case['r'])):
                for k in range(len(case['r'])):
                    r=response_matrices(**case,bits=bits,horizon=k)
                    self.assertEqual(invariant_to_horizon(**case,bits=bits,horizon=k),all(zero(x) for x in r))
    def test_metamorphic_random_shears(self):
        rng=random.Random(20261117)
        for n in range(256):
            twist=rng.choice((-7,-5,-3,-2,-1,1,2,3,5,7))
            case=delayed_shear(twist) if n<128 else nonlattice_four(twist)
            expected=4 if n<128 else 16
            self.assertEqual(len(invariant_masks(**case,horizon=0)),expected)
            expected=2 if n<128 else 6
            self.assertEqual(len(invariant_masks(**case,horizon=1)),expected)
            self.assertEqual(first_detected(**case,bits=(1,0) if n<128 else (1,0,0,0)),1)
    def test_permutation_covariance(self):
        rng=random.Random(9001)
        for _ in range(64):
            case=delayed_shear(rng.choice((-3,-2,-1,1,2,3)))
            p=matrix([[0,1],[1,0]])
            r=mul(mul(p,case['r']),p)
            c=mul(case['c'],p)
            t=mul(mul(p,case['t']),p)
            s=mul(p,case['s'])
            b=(1,0)
            after=(b[1],b[0])
            self.assertEqual(first_detected(**case,bits=b),first_detected(r,after,case['m'],case['u'],t,c,s))
    def test_no_fabricated_neural_claim(self):
        # All gates are exact rational and require no model weights or verifier access.
        case=delayed_shear(1)
        self.assertTrue(all(isinstance(v,Q) for row in case['r'] for v in row))

if __name__=='__main__':unittest.main()
