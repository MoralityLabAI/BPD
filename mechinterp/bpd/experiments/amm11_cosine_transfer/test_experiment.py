"""AMM11 excluded-engineering tests: model causality and fair cosine comparison."""
from __future__ import annotations
import math, unittest
from unittest.mock import patch
import torch
from . import models as m,case,experiment as e

class AMM11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher,cls.f,cls.fit=m.fit("engineering",0)
        cls.assay=case.score("engineering",0,cls.f,"engineering",m.TRAIN_PAIRS)
        cls.calibration=case.score("engineering",0,cls.f,"calibration",m.CAL_PAIRS)
        cls.fake_memory={
            "full_coefficients":[.2]+[0.]*17,
            "compact_coefficients":[.2]+[0.]*9,
            "pooled_action":4}
        cls.feedback=e.calibration_feedback("engineering",0,cls.calibration)
        cls.policy=e.fit_local(cls.calibration["features"],cls.feedback,cls.fake_memory)

    def test_phase_root_confirmation_sealed(self):
        self.assertEqual(m.NAME,"BPD-AMM-11")
        with self.assertRaises(ValueError):m.root_key("confirmation",0)
        with self.assertRaises(ValueError):e.score_root("confirmation",0,self.fake_memory)
        with self.assertRaises(ValueError):e.development({})
        self.assertEqual(m.PHASE_N["confirmation"],128)
        self.assertNotEqual(m.seed("training/0000"),m.seed("development/0000"))

    def test_exact_mask_budget_and_feature_budget(self):
        self.assertEqual(m.MASKS,((0,),(1,),(2,),(0,1),(0,2),(1,2)))
        self.assertEqual(tuple(e.COMPACT),(0,1,5,6,7,8,9,10,11,14))
        self.assertEqual(len(e.COMPARISONS),5)
        x=torch.tensor(self.assay["features"])
        self.assertEqual(x.shape,(m.TRAIN_PAIRS,6,18))

    def test_external_three_pass_labels_by_family(self):
        x=torch.tensor([[.1,.2,.3]])
        self.assertTrue(torch.allclose(torch.stack(m.labels(x,"wren"),dim=1),x))
        self.assertAlmostEqual(float(m.labels(x,"basalt")[0][0]),.3,places=6)

    def test_nuisance_and_future_input_deltas(self):
        for i in range(8):
            recipient,donor,target=m.donor_pairs("engineering",i,m.TRAIN_PAIRS,"engineering")
            order=m.ORDER[m.family_for(i)]
            self.assertTrue(torch.equal(target[:,0],donor[:,3+order[1]]))
            self.assertTrue(torch.equal(target[:,1],donor[:,3+order[2]]))
            move=(donor-recipient)[:,3:]
            self.assertGreaterEqual(float(move[:,order[0]].abs().min()),.149999)
            self.assertGreaterEqual(float(move[:,order[1:]].abs().min()),.249999)

    def test_physical_teacher_and_separate_factor(self):
        self.assertTrue(math.isfinite(self.fit["teacher_task_mse"]))
        self.assertLess(self.fit["teacher_task_mse"],.01)
        self.assertGreater(min(self.fit["teacher_update_norms"]),.05)
        self.assertGreater(min(self.fit["factor_update_norms"]),.05)
        self.assertEqual(int(torch.linalg.matrix_rank(self.teacher.weight().detach())),3)
        self.assertEqual(int(torch.linalg.matrix_rank(self.f.weight().detach())),3)

    def test_gauge_svd_component_parity(self):
        result=m.gauge_check(self.f)
        self.assertLess(result["max_full_gl3_error"],1e-9)
        if result["min_singular_gap"]>.01:
            self.assertLess(result["max_canonical_rankone_error"],1e-8)

    def test_physical_patch_dose_and_dark_readout(self):
        d=self.assay["diagnostics"]
        self.assertLess(d["max_equal_dose_error"],1e-5)
        self.assertEqual(d["max_immediate_readout_displacement"],0)
        self.assertEqual(len(d["degenerate_mask_projection_pairs"]),6)
        for key,bits in self.assay["candidate_bits"].items():
            self.assertEqual(len(bits),128)
            self.assertEqual(self.assay["candidate_hits"][key],bits.count("1"))
        self.assertNotIn("target",self.assay)

    def test_source_safe_cosine_decision(self):
        p=dict(self.policy)
        x=torch.tensor(self.assay["features"],dtype=torch.float64)
        a=e.choose(self.assay["features"],p,"engineering/0000")
        self.assertEqual(a["cosine"],torch.argmax(x[:,:,14],dim=1).tolist())
        self.assertEqual(a["max_cardinality"],[3]*m.TRAIN_PAIRS)
        self.assertEqual(set(a),set(e.POLICY_ARMS))
        self.assertTrue(all(0<=j<=5 for values in a.values() for j in values))

    def test_calibration_budget_and_blinding(self):
        f=self.feedback
        self.assertEqual(f["noisy_labels"],96)
        self.assertEqual(f["n_cases"],16)
        self.assertEqual(set(f["reported_bits"]),set(e.MAIN_KEYS))
        self.assertEqual(f,e.calibration_feedback("engineering",0,self.calibration))
        self.assertNotIn("clean",f)
        self.assertNotIn("target",f)
        with self.assertRaises(ValueError):
            e.fit_local(self.calibration["features"],{"n_cases":16,"noisy_labels":95},self.fake_memory)

    def test_ridge_full_and_compact_dimensions(self):
        g=m.gen("engineering","ridge_test")
        x=torch.randn(96,18,dtype=torch.float64,generator=g)
        y=torch.rand(96,dtype=torch.float64,generator=g)
        self.assertEqual(len(e.ridge(x,y,tuple(range(18)),10.)),18)
        self.assertEqual(len(e.ridge(x,y,e.COMPACT,10.)),10)
        self.assertEqual(len(e.ridge(x,y,tuple(range(18)),20.,True)),18)

    def test_source_runtime_binary_pin(self):
        signature=m.runtime_signature()
        self.assertEqual(signature["versions"]["torch"],"2.10.0+cpu")
        self.assertTrue(signature["deterministic"])
        self.assertIn(signature["cpu_capability"],("NO AVX","DEFAULT"))
        self.assertEqual(len(e.source_sha256()["RUNTIME.json"]),64)

    def test_repeated_donor_streams_disjoint(self):
        a=m.donor_pairs("engineering",0,m.PAIRS,"heldout")
        b=m.donor_pairs("engineering",0,m.PAIRS,"repeatability")
        self.assertFalse(torch.equal(a[2],b[2]))
        c=m.donor_pairs("engineering",0,m.CAL_PAIRS,"calibration")
        self.assertFalse(torch.equal(a[2][:m.CAL_PAIRS],c[2]))

    def test_root_sign_bootstrap_and_holm(self):
        self.assertEqual(e.exact_sign([1,1,1,-1,0]),5/16)
        self.assertEqual(e.exact_sign([0,0]),1.)
        self.assertEqual(e.holm({"a":.01,"b":.02,"c":.5}),{"a":.03,"b":.04,"c":.5})
        vals=[1]*64
        self.assertEqual(e.bootstrap(vals,"test"),[1/512,1/512])

    def test_primary_preflight_uses_correct_arms(self):
        rows=[{"root":m.root_key("development",i),
               "policy_hits":{"full_memory":270,"cosine":230},
               "repeatability":{"cosine_hits":220}} for i in range(64)]
        pf=e.preflight_input({"phase":"development","rows":rows})
        self.assertEqual(pf["pilot"][0]["paired_effect"],"40/512")
        self.assertEqual(pf["pilot"][0]["sham_effects"],["10/512"])
        self.assertEqual(pf["planned_comparisons"],5)
        self.assertEqual(pf["target_effect"],"0.12")

    def test_no_confirmation_in_cli_or_model(self):
        from inspect import getsource
        self.assertIn("confirmation namespace sealed",getsource(m.root_key))
        self.assertNotIn('"confirmation"),required=True',getsource(e.main))
        with patch.object(m,"fit",side_effect=AssertionError("sealed model accessed")) as spy:
            with self.assertRaises(ValueError):e.score_root("confirmation",0,{})
            spy.assert_not_called()

    def test_native_bpd_tied_three_pass_mask_parity(self):
        from mechinterp.bpd.components import BPDModel,PassMaskSchedule
        from mechinterp.bpd.config import AblationConfig
        class Base(torch.nn.Module):
            def __init__(self,w):
                super().__init__();self.core=torch.nn.Linear(6,6,bias=False)
                with torch.no_grad():self.core.weight.copy_(w)
            def forward(self,x):return self.core(x)
        native=BPDModel(Base(self.teacher.weight().detach()),{"core":3},max_passes=3,
                        ablation=AblationConfig(mode="zero",sphere_rescale=False),init_svd=False)
        with torch.no_grad():
            native.wrappers["core"].V.copy_(self.f.V)
            native.wrappers["core"].U.copy_(self.f.U)
            h,_=m.data("engineering",0,16,"native_parity")
            schedule=PassMaskSchedule(torch.ones(16,3,1,3),torch.zeros(16,3,1))
            with native.masked_components({"core":schedule}):
                a=native.target_model(h[:,None,:])[:,0]
                b=native.target_model(a[:,None,:])[:,0]
                c=native.target_model(b[:,None,:])[:,0]
            x,y,z=m.rollout(self.f,h)
            self.assertLess(max(float((a-x).abs().max()),
                                float((b-y).abs().max()),
                                float((c-z).abs().max())),1e-5)

if __name__=="__main__":unittest.main()
