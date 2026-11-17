import importlib.util
import unittest
from pathlib import Path
import torch
PATH=Path(__file__).resolve().parents[1]/'experiment.py'
spec=importlib.util.spec_from_file_location('cq03_standalone',PATH)
X=importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)

class CQ03Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.teacher=X.train_teacher(X.ENGINEERING_SEED)
        cls.factors=X.train_factors(X.ENGINEERING_SEED,cls.teacher)
        cls.reference=X.assay(X.ENGINEERING_SEED,cls.teacher,cls.factors,32)
    def test_teacher_behaviors(self):
        x,ab=X.draw(10040,128)
        with torch.no_grad():
            a=self.teacher(x);b=self.teacher(a)
            self.assertLess(float((a[:,0]-ab[:,0]).square().mean()),.002)
            self.assertLess(float((b[:,0]-ab[:,1]).square().mean()),.002)
    def test_student_matches_learned_physical_W(self):
        self.assertLess(float((self.teacher.weight().detach()-self.factors.weight().detach()).abs().max()),.005)
    def test_standalone_factors_train(self):
        self.assertEqual(self.factors.V.shape,(4,2));self.assertEqual(self.factors.U.shape,(2,4))
    def test_independent_counterfactual_target_and_full_state(self):
        r=self.reference;self.assertEqual(r['clean_n'],32)
        self.assertEqual(r['transfer']['full_state'],32)
        self.assertEqual(r['transfer']['sham'],0)
    def test_matched_norms(self):self.assertLess(self.reference['max_norm_error'],1e-5)
    def test_delayed_observable_rank(self):self.assertEqual(self.reference['delayed_observable_rank'],2)
    def test_full_gauge_invariance(self):self.assertLess(self.reference['gauge_full_write_error'],2e-6)
    def test_singleton_mask_gauge_dependence(self):self.assertGreater(self.reference['gauge_singleton_drift'],1e-3)
    def test_fixed_registration_seed_exclusion(self):self.assertNotIn(X.ENGINEERING_SEED,X.SEEDS)
    def test_native_parity_if_installed(self):
        try:import mechinterp.bpd.components
        except ImportError:self.skipTest('native BPD package unavailable in standalone sandbox')
        f=X.train_factors(X.ENGINEERING_SEED,self.teacher,'native',80)
        a=X.assay(X.ENGINEERING_SEED,self.teacher,f,32)
        self.assertIsNotNone(a['native_parity_max']);self.assertLess(a['native_parity_max'],1e-5)

if __name__=='__main__':unittest.main()
