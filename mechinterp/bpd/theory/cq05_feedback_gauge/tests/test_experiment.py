"""CQ-05 excluded engineering smoke and native masked execution checks."""
import unittest
import torch
from mechinterp.bpd.theory.cq05_feedback_gauge import experiment as E

class CQ05Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.seed=E.DEV_SEEDS[0]
        cls.model=E.physical_fit(cls.seed)
        cls.f=E.factor_fit(cls.seed,cls.model)
        cls.geometry=E.canonicalize(cls.f)
        cls.g=E.train_feedback(cls.seed,cls.f,n=1536,updates=240)

    def test_seed_split(self):
        self.assertTrue(set(E.DEV_SEEDS).isdisjoint(E.PILOT_SEEDS))
        self.assertEqual(len(E.PILOT_SEEDS),8)

    def test_labels_are_independent_input_coordinates(self):
        h,t,mode=E.make(self.seed,72,534000)
        self.assertTrue(bool((t==h[:,2:]).all()))
        self.assertEqual(set(mode.tolist()),{0,1,2})

    def test_frozen_model_parity_at_full_mask(self):
        h,t,_=E.make(self.seed,48,533000)
        with torch.no_grad():
            a=E.two_step(self.f,h,E.PATTERNS[-1].expand(len(h),-1))[0]
            x1=(h@self.f.V)@self.f.U
            x2=(x1@self.f.V)@self.f.U
        self.assertLess(float((a-torch.stack((x1[:,0],x2[:,0]),1)).abs().max()),1e-6)

    def test_decisions_binary_and_second_state_only(self):
        h,t,_=E.make(self.seed,48,534000)
        m1=self.g.first(h);one=E.step(self.f,h,m1)
        m2=self.g.second(one)
        self.assertEqual(tuple(m1.shape),(48,2))
        self.assertEqual(tuple(m2.shape),(48,2))
        self.assertTrue(bool(((m1==0)|(m1==1)).all()))
        self.assertTrue(bool(((m2==0)|(m2==1)).all()))
        # The second network has only a reached-state 8-vector input.
        self.assertEqual(self.g.second_net[0].in_features,8)

    def test_equal_dose_all_pairs(self):
        r=E.donor_assay(self.seed,self.f,self.g,n=32)
        self.assertEqual(r['n'],32)
        self.assertLess(r['max_equal_dose_error'],1e-5)
        for name in r['success_all_pairs']['feedback']:
            self.assertLessEqual(r['success_all_pairs']['feedback'][name],32)
        self.assertEqual(r['pass_two_switches']['sham'],0)

    def test_svd_canonicalization_recovers_sheared_trajectory(self):
        h,t,_=E.make(self.seed,48,534000)
        with torch.no_grad():
            p,bits,_=E.feedback_rollout(self.f,h,self.g)
            x=E.gauge_assay(self.seed,self.f,self.g,h,bits)
        self.assertLess(x['max_full_operator_delta'],2e-6)
        self.assertLess(x['max_canonical_masked_readout_delta'],1e-5)
        self.assertEqual(x['max_feedback_mask_switches_after_canonicalization'],0)
        self.assertGreater(x['max_raw_masked_readout_delta'],1e-3)

    def test_calibration_cost_budget_and_baseline(self):
        pol=E.calibrate_rules(self.seed,self.f,self.g,n=512)
        self.assertLessEqual(pol['support_cal_cost'],pol['budget']+1e-8)
        self.assertEqual(len(pol['support_patterns']),3)

    def test_native_parity_if_available(self):
        try:from mechinterp.bpd.components import PassMaskSchedule
        except ImportError:self.skipTest('native BPD unavailable')
        native=E.factor_fit(self.seed,self.model,'native')
        E.canonicalize(native)
        h,t,_=E.make(self.seed,48,534000)
        with torch.no_grad():
            m=torch.ones(len(h),2)
            x=E.step(self.f,h,m,0)
            y=E.step(native,h,m,0,'native')
            self.assertLess(float((x-y).abs().max()),1e-5)
            probe=E.donor_assay(self.seed,native,self.g,'native',n=32)
        self.assertLess(probe['max_equal_dose_error'],1e-5)

if __name__=='__main__':unittest.main()
