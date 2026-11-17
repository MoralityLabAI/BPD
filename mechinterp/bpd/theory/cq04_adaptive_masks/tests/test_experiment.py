import unittest
import torch
from mechinterp.bpd.theory.cq04_adaptive_masks import experiment as e

class CQ04Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1)
  cls.seed=e.ENGINEERING_SEEDS[0]
  cls.t=e.train_teacher(cls.seed)
  cls.f=e.train_factors(cls.seed,cls.t)
  cls.c=e.train_controller(cls.seed,cls.f)
 def test_split(self):
  self.assertTrue(set(e.SEEDS).isdisjoint(e.ENGINEERING_SEEDS))
 def test_gate_binary_and_shape(self):
  h,t,mode=e.source(self.seed,64,444000)
  m=self.c.masks(h)
  self.assertEqual(tuple(m.shape),(64,4));self.assertTrue(bool(((m==0)|(m==1)).all()))
 def test_all_masks_equal_full_factor(self):
  h,t,mode=e.source(self.seed,64,444000)
  with torch.no_grad():
   pred,a,b=e.eval_mask(self.f,h,e.PATTERNS[-1].expand(len(h),-1))
   full=(h@self.f.V)@self.f.U;full2=(full@self.f.V)@self.f.U
   self.assertLess(float((a-full).abs().max()),1e-6)
   self.assertLess(float((b-full2).abs().max()),1e-6)
 def test_eligible_independent_labels(self):
  h,t,mode=e.source(self.seed,128,444000)
  self.assertTrue(bool((h[:,2:]==t).all()))
 def test_calibration_static_budget(self):
  pol=e.static_budget_mix(self.seed,self.f,self.c)
  self.assertLessEqual(pol['calibration_static_expected_cost'],pol['calibration_gate_cost']+1e-6)
 def test_donor_norm_and_sham(self):
  r=e.donor_assay(self.seed,self.f,self.c)
  self.assertLess(r['max_equal_dose_error'],1e-5)
  self.assertEqual(r['success']['sham'],0)
 def test_native_parity_if_available(self):
  try:import mechinterp.bpd.components
  except ImportError:self.skipTest('native module unavailable')
  native=e.train_factors(self.seed,self.t,'native')
  h,t,mode=e.source(self.seed,64,444000)
  m=self.c.masks(h)
  a=e.eval_mask(self.f,h,m)[0];b=e.eval_mask(native,h,m,'native')[0]
  self.assertLess(float((a-b).abs().max()),1e-5)

if __name__=='__main__': unittest.main()
