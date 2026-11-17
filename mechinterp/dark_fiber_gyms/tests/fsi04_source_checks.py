"""FSI04 CPU controls: geometry, causal scoring, source seal and risk preflight."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import torch

from mechinterp.dark_fiber_gyms.core import trm_manifold_swap
from mechinterp.dark_fiber_gyms.fsi04.geometry import matched_random_joint
from mechinterp.dark_fiber_gyms.fsi04.native_pilot import _bad_path, native_case
from mechinterp.dark_fiber_gyms.fsi04.historical_preflight import audit as audit_historical
from mechinterp.dark_fiber_gyms.fsi04.roster import REG

class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        self.head=torch.randn(11,32)
        self.rec=torch.randn(1,97,32)
        self.donor=torch.randn(1,97,32)
        self.zl=torch.randn(1,97,32)
        self.donor_zl=torch.randn(1,97,32)
        self.b=16

    def patch(self):
        transplanted,cert=trm_manifold_swap(self.rec,self.donor,self.head,board_start=self.b)
        dl=self.zl.clone();dl[:,self.b:,:]=self.donor_zl[:,self.b:,:]
        return transplanted,dl,cert

    def test_random_sham_matches_donor_physical_dose_and_full_head(self):
        transplanted,dl,_=self.patch()
        h,l,cert=matched_random_joint(self.rec,transplanted,self.zl,dl,self.head,self.b,218)
        self.assertEqual(h.dtype,torch.float32)
        self.assertTrue(torch.equal(h[:,:self.b],self.rec[:,:self.b]))
        self.assertTrue(torch.equal(l[:,:self.b],self.zl[:,:self.b]))
        e0=(transplanted[:,self.b:]-self.rec[:,self.b:]).norm(dim=-1)
        e1=(h[:,self.b:]-self.rec[:,self.b:]).norm(dim=-1)
        self.assertLess((e0-e1).abs().max().item(),2e-3)
        d0=(dl[:,self.b:]-self.zl[:,self.b:]).norm(dim=-1)
        d1=(l[:,self.b:]-self.zl[:,self.b:]).norm(dim=-1)
        self.assertLess((d0-d1).abs().max().item(),2e-3)
        self.assertLess(cert["head_pre_readout_max_abs"],1e-3)
        self.assertLess(cert["head_pre_radius_max_abs"],2e-3)
        self.assertGreater((h-transplanted).abs().max().item(),0.01)

    def test_random_sham_deterministic_given_frozen_board_seed(self):
        transplanted,dl,_=self.patch()
        aa=matched_random_joint(self.rec,transplanted,self.zl,dl,self.head,self.b,500)
        bb=matched_random_joint(self.rec,transplanted,self.zl,dl,self.head,self.b,500)
        cc=matched_random_joint(self.rec,transplanted,self.zl,dl,self.head,self.b,501)
        self.assertTrue(torch.equal(aa[0],bb[0]))
        self.assertTrue(torch.equal(aa[1],bb[1]))
        self.assertFalse(torch.equal(aa[0],cc[0]))

    def test_invalid_visible_donor_is_refused(self):
        with self.assertRaisesRegex(ValueError,"audited"):
            matched_random_joint(self.rec,self.donor,self.zl,self.donor_zl,self.head,self.b,44)

    def test_zero_dose_behaves_as_noop(self):
        h,l,cert=matched_random_joint(self.rec,self.rec,self.zl,self.zl,self.head,self.b,55)
        self.assertLess((h-self.rec).abs().max().item(),2e-5)
        self.assertTrue(torch.equal(l,self.zl))
        self.assertLess(cert["zH_dose_mismatch_max_abs"],2e-4)

    def test_rejects_arg03_or_jsonl_readouts(self):
        for path in ("runs/arg03/test.jsonl","runs/arg03/development.jsonl","test.jsonl"):
            self.assertTrue(_bad_path(Path(path)))
        self.assertFalse(_bad_path(Path("runs/fsi04/development.csv")))

class _ToyTRM:
    puzzle_emb_len=16
    def __init__(self):
        torch.manual_seed(2026)
        self.H_init=torch.ones(1,97,32)
        self.lm_head=torch.nn.Linear(32,11,bias=False)
        self.head0=torch.randn(1,97,32)
        self.z0=torch.randn(1,97,32)
    def encode_inputs(self,x):return x
    def init_states(self,n):return self.head0.expand(n,-1,-1).clone(),self.z0.expand(n,-1,-1).clone()
    def act_step(self,h,l,x):
        return h+.02*torch.tanh(l),l+.025*torch.tanh(h)
    def digit_logits(self,h):return self.lm_head(h)[:,self.puzzle_emb_len:,2:11]

class _ToyTS:
    @staticmethod
    def encode_questions(questions,device):return torch.zeros(len(questions),97,32,device=device)

class PilotTests(unittest.TestCase):
    def test_native_pilot_preserves_self_and_exact_causal_corrections(self):
        model=_ToyTRM()
        row={"question":"."*81,"answer":"1"*81,"stratum":"S4_stalled"}
        out=native_case(_ToyTS,model,row,patch_step=2,horizon=5,sigma=.15,seed=19)
        a=out["arms"]
        self.assertEqual(a["clean"]["correct_cells"],a["self_control"]["correct_cells"])
        for arm in ("donor_joint","random_joint_dose_matched"):
            self.assertEqual(a[arm]["net_cells_vs_clean"],
                             a[arm]["gained_cells_vs_clean"]-a[arm]["lost_cells_vs_clean"])
        self.assertLess(out["certificate"]["pre_patch_full_head_max_abs"],1e-3)
        self.assertLess(out["certificate"]["dose_matched_sham"]["zH_dose_mismatch_max_abs"],2e-3)

    def test_registry_pinned_and_dev_scope(self):
        reg=json.loads(REG.read_text())
        self.assertEqual(reg["sampling"]["n_stalled_pilot"],24)
        self.assertEqual(reg["analysis"]["science_min_effect"],"0.10")
        self.assertIn("168-case",reg["test_seal"])

    def test_historical_planning_red_and_sham_distinct(self):
        if not os.environ.get("EVALMATH_PREFLIGHT_FILE"):
            self.skipTest("EvalMath source unavailable outside pinned CI")
        r=audit_historical()
        self.assertEqual(r["minimum_independent_roots_for_95th_quantile_95pct_confidence"],59)
        self.assertEqual(r["pilot_preflight"]["decision"],"REDESIGN_OR_PILOT_ONLY")
        self.assertEqual(r["historical"]["yagizdevre"]["n_root_boards"],40)
        self.assertLess(r["historical"]["yagizdevre"]["mean_donor_minus_clean"],0)
        self.assertLess(r["historical"]["yagizdevre"]["mean_random_minus_clean"],0)
        self.assertEqual(r["decision"],"DOSE_MATCHED_FRESH_PILOT_ONLY")

if __name__=="__main__":unittest.main()
