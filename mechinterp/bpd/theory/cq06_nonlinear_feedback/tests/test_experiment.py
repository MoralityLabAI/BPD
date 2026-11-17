"""Only excluded-root development smoke; never reads confirmation receipts."""
import unittest
import torch
from mechinterp.bpd.theory.cq06_nonlinear_feedback import experiment as E
from mechinterp.bpd.theory.cq06_nonlinear_feedback import evalmath_preflight as EM

class CQ06Engineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        E.configure()
        cls.seed=E.DEVELOPMENT[0]
        cls.teacher=E.train_teacher(cls.seed)
        cls.factor=E.train_factors(cls.seed,cls.teacher)
        cls.geom=E.canonicalize(cls.factor)
        cls.gate,cls.planner,cls.diag=E.train_policies(cls.seed,cls.teacher,cls.factor)

    def test_disjoint_independent_roots(self):
        self.assertEqual(len(E.SHAM),59)
        self.assertEqual(len(E.CONFIRMATION),26)
        self.assertTrue(set(E.DEVELOPMENT).isdisjoint(E.SHAM))
        self.assertTrue(set(E.SHAM).isdisjoint(E.CONFIRMATION))

    def test_late_cue_exact_information_ceiling(self):
        u0,u1,y=E.source(self.seed,256,810000,paired_z=True)
        self.assertTrue(bool(torch.equal(u0[::2],u0[1::2])))
        self.assertTrue(bool((y[::2,0]==y[1::2,0]).all()))
        self.assertTrue(bool((y[::2,1]==-y[1::2,1]).all()))
        with torch.no_grad():
            _,s1,_=E.both(self.teacher,self.factor,u0,u1)
            one_pass=E.logits(self.teacher,s1)
            self.assertEqual(int(((one_pass*y[:,1])>0).sum()),128)

    def test_learned_nonlinear_teacher_succeeds(self):
        u0,u1,y=E.source(self.seed,512,920000)
        with torch.no_grad():
            out,_,_=self.teacher(u0,u1)
            self.assertGreaterEqual(int(((out*y)>0).all(1).sum()),.95*len(y))

    def test_factor_geometry_and_output(self):
        self.assertLess(self.geom['initial_full_matrix_reconstruction_error'],1e-4)
        u0,u1,y=E.source(self.seed,256,821000)
        with torch.no_grad():
            out,_,_=E.both(self.teacher,self.factor,u0,u1)
            self.assertGreaterEqual(int(((out*y)>0).all(1).sum()),.95*len(y))

    def test_gate_and_task_controls(self):
        cal=E.calibrate(self.seed,self.teacher,self.factor,self.gate,self.planner,n=256)
        scores=E.scores_on_task(self.seed,self.teacher,self.factor,self.gate,self.planner,cal,n=256)
        self.assertEqual(scores['myopic_same_first_cue_pairs']['correct'],128)
        self.assertEqual(scores['algebraic_oracle']['both_correct'],256)
        self.assertEqual(scores['full']['both_correct'],256)
        self.assertTrue(4<=scores['feedback']['mean_active_slots']<=8)
        self.assertTrue(4<=scores['input_planner']['mean_active_slots']<=8)

    def test_all_donor_pairs_and_equal_state_doses(self):
        d=E.donor_assay(self.seed,self.teacher,self.factor,self.gate,n=32)
        self.assertEqual(d['n_pairs'],32)
        self.assertTrue(d['first_target_retained'] and d['per_root_target_flipped'])
        self.assertLess(d['max_equal_dose_error'],1e-5)
        self.assertEqual(set(d['out_of_tanh_range_by_arm']),set(d['counts']['feedback']))
        self.assertLess(d['max_immediate_readout_dark_effect'],1e-5)
        for arm in ('dark_bpd','manifold_dark_1','manifold_dark_2','random_dark_1','random_dark_2','sham'):
            self.assertEqual(len(d['pair_outcomes'][arm]['feedback']),32)
            self.assertLessEqual(d['counts']['feedback'][arm],32)
        self.assertEqual(d['switches']['sham'],0)

    def test_canonical_gauge_and_raw_instability(self):
        g=E.gauge_assay(self.seed,self.teacher,self.factor,self.gate,n=64)
        self.assertLess(g['max_full_operator_shear_error'],2e-6)
        self.assertLess(g['max_recanonicalized_logit_drift'],1e-5)
        self.assertEqual(g['recanonicalized_mask_switches'],0)
        self.assertGreater(g['max_raw_shear_masked_logit_drift'],.01)

    def test_evalmath_59_root_sham_tolerance(self):
        from fractions import Fraction
        self.assertEqual(EM.upper_quantile([Fraction('0.04')]*58)[0],None)
        self.assertEqual(EM.upper_quantile([Fraction('0.04')]*59)[0],Fraction('0.04'))

    def test_native_parity_when_repository_available(self):
        try:
            from mechinterp.bpd.components import PassMaskSchedule
        except ImportError:
            self.skipTest('actual repository BPDModel unavailable in standalone CPU sandbox')
        native=E.NativeModel(self.teacher,self.factor)
        self.assertLess(E.native_parity(self.seed,self.teacher,self.factor,native),1e-5)
        d=E.donor_assay(self.seed,self.teacher,self.factor,self.gate,'native',native,n=32)
        self.assertLess(d['max_equal_dose_error'],1e-5)
        self.assertEqual(set(d['out_of_tanh_range_by_arm']),set(d['counts']['feedback']))

if __name__=='__main__':unittest.main()
