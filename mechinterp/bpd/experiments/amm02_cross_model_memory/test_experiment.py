"""Only excluded engineering roots are executed in source tests."""
from __future__ import annotations
import hashlib
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from . import experiment as e

class AMM02EngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher,cls.factor,cls.diagnostics=e.fit("engineering",0)
        cls.raw=e.candidate_assay("engineering",0,cls.factor,cls.teacher,cls.diagnostics)

    def test_externally_defined_labels(self):
        ab=torch.tensor([[.14,.49],[-.28,.3]])
        a,b=e.labels(ab,"wren")
        self.assertTrue(torch.equal(a,ab[:,0]) and torch.equal(b,ab[:,1]))
        b,a=e.labels(ab,"basalt")
        self.assertTrue(torch.equal(a,ab[:,0]) and torch.equal(b,ab[:,1]))
        for i in range(8):
            recipient,donor,goal=e.donors("engineering",i)
            self.assertEqual(recipient.shape,(e.PAIRS,4))
            if i%2==0:
                self.assertTrue(torch.equal(donor[:,2],recipient[:,2]))
                self.assertTrue(torch.equal(donor[:,3],goal))
            else:
                self.assertTrue(torch.equal(donor[:,3],recipient[:,3]))
                self.assertTrue(torch.equal(donor[:,2],goal))

    def test_independent_root_partitions(self):
        groups=[{e.root_key(p,i) for i in range(n)} for p,n in e.PHASE_N.items()]
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                self.assertFalse(groups[i]&groups[j])
        self.assertEqual(len(groups[-1]),128)

    def test_trained_teacher_factors_and_task_fit(self):
        d=self.diagnostics
        self.assertTrue(all(x>.05 for x in d["teacher_update_norms"]))
        self.assertTrue(all(x>.05 for x in d["factor_update_norms"]))
        self.assertLess(d["teacher_behavioral_mse"],.08)
        self.assertLess(d["factor_behavioral_mse"],.08)
        self.assertLess(d["weight_reconstruction_relative"],.08)
        self.assertLessEqual(int(torch.linalg.matrix_rank(self.teacher.weight().detach())),2)

    def test_equal_dose_and_dark_invariance(self):
        stats=self.raw["pair_diagnostics"]
        self.assertLess(stats["max_active_dose_error"],1e-5)
        self.assertEqual(stats["max_active_current_readout_error"],0.)
        self.assertEqual(self.raw["n"],e.PAIRS)
        for key,bits in self.raw["candidate_bits"].items():
            self.assertEqual(len(bits),e.PAIRS)
            self.assertEqual(bits.count("1"),self.raw["candidate_hits"][key])
        self.assertIn("random_dark_b",self.raw["candidate_bits"])

    def test_gauge_stable_canonical_rank_one_products(self):
        v=self.factor.V.detach().double()
        u=self.factor.U.detach().double()
        R=torch.tensor([[1.,.6],[0.,1.]],dtype=torch.float64)
        p=v@u
        self.assertLess(float((p-(v@R)@(torch.linalg.inv(R)@u)).abs().max()),1e-10)
        qa,sa,vha=torch.linalg.svd(p)
        qb,sb,vhb=torch.linalg.svd((v@R)@(torch.linalg.inv(R)@u))
        aa=qa[:,:2]*sa[:2].sqrt();au=sa[:2].sqrt()[:,None]*vha[:2]
        ba=qb[:,:2]*sb[:2].sqrt();bu=sb[:2].sqrt()[:,None]*vhb[:2]
        for c in (0,1):
            self.assertLess(float(((aa[:,c:c+1]@au[c:c+1])-(ba[:,c:c+1]@bu[c:c+1])).abs().max()),1e-10)

    def test_policy_control_accounting(self):
        memory={"family_action":{"wren":0,"basalt":1},"pooled_action":0}
        row=e.score_row(self.raw,memory)
        self.assertEqual(row["policy_hits"]["memory"],self.raw["candidate_hits"]["component_0"])
        self.assertEqual(row["policy_hits"]["swapped"],self.raw["candidate_hits"]["component_1"])
        self.assertEqual(len(row["policy_bits"]["random"]),e.PAIRS)
        self.assertEqual(set(row["policy_hits"]),set(e.ARMS))

    @staticmethod
    def placeholder_memory():
        m={"schema":e.SCHEMA+"/training-memory","root_count":64,
           "family_action":{"wren":0,"basalt":1},"pooled_action":0,
           "rows":[{"root":e.root_key("training",i)} for i in range(64)],
           "experiment_source_sha256":hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest()}
        m["sha256"]=e.digest(m)
        return m

    def test_confirmation_guard_before_root_access(self):
        memory=self.placeholder_memory()
        with patch.object(e,"fit_one_root",side_effect=AssertionError("SEALED DATA ACCESSED")) as spy:
            for preflight in (None,{},{"decision":"REDESIGN_OR_PILOT_ONLY"},
                             {"decision":"READY_FOR_CONFIRMATION"}):
                with self.assertRaises(ValueError):
                    e.evaluate("confirmation",memory,preflight)
            with self.assertRaises(ValueError):e.evaluate("confirmation",None)
            spy.assert_not_called()

    def test_root_level_preflight_input(self):
        rows=[{"root":e.root_key("development",i),
               "policy_hits":{"memory":31,"pooled":21,"random_dark":11,
                              "random_dark_b":12}} for i in range(64)]
        pf=e.preflight_input({"phase":"development","rows":rows})
        self.assertEqual(pf["confirm_units_planned"],128)
        self.assertEqual(pf["planned_comparisons"],3)
        self.assertEqual(pf["pilot"][0]["sham_effects"],["-1/48"])
        self.assertEqual(len({r["unit_id"] for r in pf["pilot"]}),64)

    def test_exact_sign_and_holm(self):
        self.assertAlmostEqual(e.root_sign_p([1,1,-1,0]),.5)
        self.assertEqual(e.root_sign_p([0,0]),1.)
        self.assertEqual(e.holm({"a":.01,"b":.02,"c":.5}),{"a":.03,"b":.04,"c":.5})

    def test_native_bpdmodel_pass_mask_parity(self):
        if importlib.util.find_spec("mechinterp") is None:
            self.skipTest("native package available only in BPD checkout")
        from mechinterp.bpd.components import BPDModel,PassMaskSchedule
        from mechinterp.bpd.config import AblationConfig
        class Base(torch.nn.Module):
            def __init__(self,w):
                super().__init__()
                self.core=torch.nn.Linear(4,4,bias=False)
                with torch.no_grad():self.core.weight.copy_(w)
            def forward(self,h):return self.core(h)
        bpd=BPDModel(Base(self.teacher.weight().detach()),{"core":2},max_passes=2,
                     ablation=AblationConfig(mode="zero",sphere_rescale=False),init_svd=False)
        with torch.no_grad():
            bpd.wrappers["core"].V.copy_(self.factor.V)
            bpd.wrappers["core"].U.copy_(self.factor.U)
            h,_=e.data("engineering",0,16,"native_parity")
            schedule=PassMaskSchedule(torch.ones(16,2,1,2),torch.zeros(16,2,1))
            with bpd.masked_components({"core":schedule}):
                y1=bpd.target_model(h[:,None,:])[:,0]
                y2=bpd.target_model(y1[:,None,:])[:,0]
            x1=self.factor(h);x2=self.factor(x1)
            self.assertLess(float((y1-x1).abs().max()),1e-5)
            self.assertLess(float((y2-x2).abs().max()),1e-5)

if __name__=="__main__":
    unittest.main()
