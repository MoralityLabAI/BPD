"""CQ-07 excluded-root reachability and native BPD engineering tests."""
import unittest
import torch
from mechinterp.bpd.theory.cq07_reachability import geometry as G
from mechinterp.bpd.theory.cq06_nonlinear_feedback import experiment as Q
from mechinterp.bpd.theory.cq06_nonlinear_feedback import evalmath_preflight as EM

class CQ07GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Q.configure()
        cls.seed=G.DEVELOPMENT[0]
        cls.teacher=Q.train_teacher(cls.seed)
        cls.f=Q.train_factors(cls.seed,cls.teacher)
        Q.canonicalize(cls.f)
        cls.geom=G.legal_preactivation_geometry(cls.seed,cls.teacher,cls.f)
        cls.r=G.evaluate_root(cls.seed,n_pairs=32,bank_n=1024,opt_steps=100,gate_updates=200)

    def test_development_seeds_only(self):
        self.assertEqual(len(G.DEVELOPMENT),8)
        self.assertNotIn(6001,G.DEVELOPMENT)
        with self.assertRaises(ValueError):
            G.evaluate_root(6001,n_pairs=4,bank_n=4,opt_steps=1,gate_updates=1)

    def test_exact_input_span_rank_obstruction(self):
        self.assertEqual(self.geom['combined_rank'],4)
        self.assertGreater(self.geom['min_singular_value'],1e-3)
        self.assertTrue(self.geom['exact_nonzero_reachability_obstructed'])
        A=self.geom['input_basis'];B=self.geom['bpd_basis']
        self.assertEqual(tuple(A.shape),(2,Q.D))
        self.assertEqual(tuple(B.shape),(2,Q.D))
        # For a stack of full row rank 4, no nonzero vector belongs to both spans.
        v=torch.tensor([[1.,-0.8]],dtype=torch.float64)@B
        d,_=G.affine_residual(v,A)
        self.assertGreater(float(d.min()),1e-3)

    def test_clean_cue_preimage_and_tanh_injectivity(self):
        u0,_,y=Q.source(self.seed,32,945700)
        with torch.no_grad():
            s=Q.independent_step(self.teacher,self.f,torch.zeros(32,Q.D),u0)
            pre=torch.cat((torch.zeros(32,Q.D),u0),1)@self.f.weight()+self.teacher.bias
        self.assertLess(float((s-torch.tanh(pre)).abs().max()),1e-6)
        self.assertEqual(int((s.abs()>1.0000001).any(1).sum()),0)
        base=self.geom['base'];a=self.geom['input_basis'];r=Q.rotation(self.seed)
        unrot=u0@r.T
        signed=(base+unrot[:,:2].double()@a)
        self.assertLess(float((pre.double()-signed).abs().max()),3e-6)

    def test_source_labels_same_first_different_delayed(self):
        u0,ud,u1,yr,yd=Q.donor_inputs(self.seed,128)
        self.assertTrue(bool((yr[:,0]==yd[:,0]).all()))
        self.assertTrue(bool((yr[:,1]==-yd[:,1]).all()))

    def test_constrained_tanh_and_excluded_root_receipt(self):
        f=self.r['feasibility'];n=self.r['n_pairs']
        self.assertEqual(n,32)
        self.assertEqual(f['range_violations_bpd'],0)
        self.assertEqual(f['range_violations_clean_null1'],0)
        self.assertEqual(f['range_violations_full_donor'],0)
        self.assertEqual(f['exactly_reachable_nonzero_rank2_patches'],0)
        self.assertEqual(f['nonzero_rank2_writes'],n)
        self.assertLessEqual(f['joint_patch_null1'],f['patch_solver_feasible'])
        self.assertLessEqual(f['joint_patch_null1'],f['natural_null_1_matched'])
        self.assertEqual(len(self.r['pair_outcomes']['bpd']),n)

    def test_null_reachable_from_actual_legal_source(self):
        t,f=self.teacher,self.f
        ur,ud,u1,yr,yd=Q.donor_inputs(self.seed,32)
        with torch.no_grad():
            _,sr,_=Q.both(t,f,ur,u1)
            _,sd,_=Q.both(t,f,ud,u1)
        patch=G.constrained_rank2_state(t,f,ur,ud,sr,sd,n_steps=100)
        match=G.match_natural_states(self.seed,t,f,sr,yr[:,0],patch['dose'],bank_n=1024)
        bank,head,parity,x=G.clean_bank(self.seed,t,f,951000,n=1024)
        self.assertTrue(bool(torch.equal(match['state'],bank[match['chosen_index']])))
        valid=match['available']
        self.assertTrue(bool((parity[match['chosen_index'][valid]]==yr[:,0][valid]).all()))
        self.assertTrue(bool((match['head_error'][valid]<=G.HEAD_TOL+1e-5).all()))
        self.assertTrue(bool((match['dose_error'][valid]<=G.DOSE_BAND+1e-5).all()))

    def test_frozen_source_intention_to_treat(self):
        n=self.r['n_pairs'];counts=self.r['counts'];f=self.r['feasibility']
        self.assertEqual(counts['full_donor_oracle'],n)
        self.assertEqual(counts['sham'],0)
        self.assertLessEqual(counts['bpd_joint'],f['joint_patch_null1'])
        self.assertLessEqual(counts['reachable_null_1_joint'],f['joint_patch_null1'])
        self.assertEqual(counts['bpd_joint'],counts['reachable_null_1_joint'])

    def test_evalmath_root_calibration_restriction(self):
        from fractions import Fraction
        self.assertIsNone(EM.upper_quantile([Fraction(0)]*8)[0])
        self.assertEqual(EM.upper_quantile([Fraction(0)]*59)[0],Fraction(0))

    def test_native_second_pass_parity_if_available(self):
        try:from mechinterp.bpd.components import BPDModel
        except ImportError:self.skipTest('native repository BPD package unavailable in standalone sandbox')
        native=G.evaluate_root(self.seed,'native',n_pairs=32,bank_n=1024,opt_steps=100,gate_updates=200)
        self.assertLess(native['max_native_state_parity_error'],1e-5)
        self.assertEqual(native['counts'],self.r['counts'])
        self.assertEqual(native['feasibility'],self.r['feasibility'])

if __name__=='__main__':unittest.main()
