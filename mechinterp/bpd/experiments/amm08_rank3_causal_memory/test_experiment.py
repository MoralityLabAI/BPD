"""AMM-08 excluded engineering-root tests. Never score development/confirmation."""
from __future__ import annotations
import importlib.util
import math
import unittest
from unittest.mock import patch
import torch
from . import models as m, case, experiment as e

class AMM08EngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher,cls.f,cls.metrics=m.fit("engineering",0)
        cls.sample=case.score("engineering",0,cls.f,"engineering",m.TRAIN_PAIRS)

    def test_root_namespaces_separated(self):
        groups=[{m.root_key(p,i) for i in range(n)} for p,n in m.PHASE_N.items()]
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                self.assertFalse(groups[i] & groups[j])
        self.assertEqual(m.PHASE_N["confirmation"],128)
        self.assertNotEqual(m.seed("training/0000"),m.seed("development/0000"))
        self.assertEqual(m.NAME,"BPD-AMM-08")

    def test_seven_frozen_nonempty_component_masks(self):
        self.assertEqual(m.MASKS,((0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)))
        self.assertEqual(len(m.MASKS),7)
        self.assertEqual(m.FEATURES,18)

    def test_ternary_task_labels_independent_of_model(self):
        abc=torch.tensor([[.12,.34,.56],[-.11,.22,-.33]])
        self.assertTrue(torch.equal(m.labels(abc,"wren")[1],abc[:,1]))
        self.assertTrue(torch.equal(m.labels(abc,"wren")[2],abc[:,2]))
        self.assertTrue(torch.equal(m.labels(abc,"basalt")[0],abc[:,2]))
        self.assertTrue(torch.equal(m.labels(abc,"basalt")[1],abc[:,0]))
        self.assertTrue(torch.equal(m.labels(abc,"basalt")[2],abc[:,1]))

    def test_two_future_goals_and_nuisance_displacement(self):
        for i in range(m.PHASE_N["engineering"]):
            rec,don,target=m.donor_pairs("engineering",i,m.TRAIN_PAIRS,"engineering")
            order=m.ORDER[m.family_for(i)]
            self.assertEqual(tuple(target.shape),(m.TRAIN_PAIRS,2))
            self.assertTrue(torch.equal(target[:,0],don[:,3+order[1]]))
            self.assertTrue(torch.equal(target[:,1],don[:,3+order[2]]))
            diff=(don-rec)[:,3:]
            self.assertTrue(bool((diff[:,order[0]].abs()>=.15-1e-6).all()))
            for idx in order[1:]:
                self.assertTrue(bool((diff[:,idx].abs()>=.25-1e-6).all()))
            self.assertTrue(bool((diff.abs()>0).all()))

    def test_genuine_teacher_and_separate_factor_training(self):
        d=self.metrics
        self.assertEqual(tuple(self.teacher.I.shape),(6,3))
        self.assertEqual(tuple(self.f.V.shape),(6,3))
        self.assertGreater(min(d["teacher_update_norms"]),.05)
        self.assertGreater(min(d["factor_update_norms"]),.05)
        self.assertTrue(math.isfinite(d["teacher_task_mse"]))
        self.assertTrue(math.isfinite(d["factor_task_mse"]))
        self.assertLess(d["teacher_task_mse"],.01)
        self.assertEqual(int(torch.linalg.matrix_rank(self.teacher.weight().detach())),3)
        self.assertEqual(int(torch.linalg.matrix_rank(self.f.weight().detach())),3)

    def test_gauge_invariance_after_rank3_svd(self):
        d=m.gauge_check(self.f)
        self.assertLess(d["max_full_gl3_error"],1e-9)
        if d["min_singular_gap"]>.01:
            self.assertLess(d["max_canonical_rankone_error"],1e-8)

    def test_physical_equal_norm_and_current_dark(self):
        d=self.sample["diagnostics"]
        self.assertLess(d["max_equal_dose_error"],1e-5)
        self.assertEqual(d["max_immediate_readout_displacement"],0.)
        self.assertEqual(self.sample["n"],m.TRAIN_PAIRS)
        self.assertEqual(len(d["degenerate_mask_projection_pairs"]),7)
        for key,bitstring in self.sample["candidate_bits"].items():
            self.assertEqual(len(bitstring),m.TRAIN_PAIRS)
            self.assertEqual(self.sample["candidate_hits"][key],bitstring.count("1"))

    def test_18_features_only_from_unlabeled_causal_probes(self):
        f=torch.tensor(self.sample["features"])
        self.assertEqual(tuple(f.shape),(m.TRAIN_PAIRS,7,18))
        self.assertTrue(bool(torch.isfinite(f).all()))
        self.assertNotIn("target",self.sample)
        self.assertNotIn("donor_future",self.sample)
        self.assertNotIn("donor_input",self.sample)

    def test_oracle_ceiling_and_distinct_dose_full_donor(self):
        bits=self.sample["candidate_bits"]
        best=max(bits[f"mask_{a}"].count("1") for a in range(7))
        oracle=self.sample["diagnostics"]["casewise_oracle_hits_audit_only"]
        self.assertGreaterEqual(oracle,best)
        self.assertLessEqual(oracle,m.TRAIN_PAIRS)
        self.assertGreaterEqual(bits["full_donor"].count("1"),oracle)
        # The full donor state has a different immediate-visible write.

    def test_calibration_noisy_budget_and_policy_same_inputs(self):
        cal=case.score("engineering",0,self.f,"calibration",m.CAL_PAIRS)
        noisy=e.corrupted_feedback("engineering",0,cal)
        self.assertEqual(noisy["noisy_labels"],112)
        self.assertEqual(len(noisy["reported_bits"]),7)
        self.assertEqual(noisy,e.corrupted_feedback("engineering",0,cal))
        self.assertNotIn("clean",noisy)
        self.assertNotIn("target",noisy)
        for bits in noisy["reported_bits"].values():
            self.assertEqual(len(bits),16)
        memory={"global_coefficients":[.1]+[0.]*17,"pooled_action":0}
        policy=e.policy_fit(cal["features"],noisy,memory)
        choices=e.choose(self.sample["features"],policy,"engineering/0000")
        self.assertEqual(set(choices),{"memory_plus_local","local_ridge","memory_only",
                                       "local_count","pooled","random_mask"})
        for name,values in choices.items():
            self.assertEqual(len(values),m.TRAIN_PAIRS)
            self.assertTrue(all(0<=v<7 for v in values),name)

    def test_training_ridge_fit_shapes_and_penalties(self):
        g=m.gen("engineering","ridge_test")
        X=torch.randn((70,18),generator=g,dtype=torch.float64)
        y=torch.rand(70,generator=g,dtype=torch.float64)
        a=e.ridge(X,y,10.,False)
        b=e.ridge(X,y,20.,True)
        self.assertEqual(len(a),18)
        self.assertEqual(len(b),18)
        self.assertTrue(all(math.isfinite(v) for v in a+b))

    @staticmethod
    def sample_memory():
        rows=[]
        for i in range(64):
            rows.append({"root":m.root_key("training",i),
                         "assay":{"candidate_bits":{f"mask_{a}":"0"*m.TRAIN_PAIRS
                                for a in range(7)}}})
        training_bits=m.digest([{"root":r["root"],"candidate_bits":r["assay"]["candidate_bits"]}
                                for r in rows])
        memory={"schema":m.SCHEMA+"/training_memory","phase":"training",
                "experiment":m.NAME,"source_sha256":e.source_sha256(),
                "n_independent_training_roots":64,
                "training_cases_per_root":m.TRAIN_PAIRS,
                "candidate_hits":{f"mask_{a}":0 for a in range(7)},
                "pooled_action":0,"global_coefficients":[.1]+[0.]*17,
                "ridge_lambda":e.GLOBAL_RIDGE,"train_bits_sha256":training_bits}
        memory["sha256"]=m.digest(memory)
        memory["rows"]=rows
        return memory

    def test_training_memory_source_and_4decimal_guards(self):
        mem=self.sample_memory()
        self.assertTrue(e.validate_memory(mem))
        mem["global_coefficients"][0]=.100001
        with self.assertRaises(ValueError):e.validate_memory(mem)
        mem=self.sample_memory()
        mem["rows"][0]["assay"]["candidate_bits"]["mask_0"]="1"*m.TRAIN_PAIRS
        with self.assertRaises(ValueError):e.validate_memory(mem)

    def test_invalid_preflight_does_not_touch_confirmation(self):
        mem=self.sample_memory()
        pf={"schema":"EvalMath/mechinterp-power-preflight/v1",
            "experiment_id":m.NAME,"decision":"READY_FOR_CONFIRMATION",
            "planned_confirmation_worlds":128,"pilot_worlds":64,
            "planned_comparisons":3,"mean_pilot_effect":.2,
            "mechanistic_variation_floor":.08,"reasons":[]}
        with patch.object(e,"score_root",side_effect=AssertionError("SEALED ROOT ACCESSED")) as spy:
            for p,release in ((None,None),({},None),
                              ({"decision":"REDESIGN_OR_PILOT_ONLY"},None),
                              (pf,None),(pf,{"human_approved":True})):
                with self.assertRaises(ValueError):
                    e.evaluate("confirmation",mem,p,release)
            spy.assert_not_called()

    def test_exact_independent_root_preflight_shape(self):
        rows=[{"root":m.root_key("development",i),
               "policy_hits":{"memory_plus_local":270,"local_ridge":230},
               "repeatability":{"local_ridge_hits":220}} for i in range(64)]
        pf=e.preflight_input({"phase":"development","n_independent_roots":64,"rows":rows})
        self.assertEqual(len(pf["pilot"]),64)
        self.assertEqual(pf["pilot"][0]["paired_effect"],"40/512")
        self.assertEqual(pf["pilot"][0]["sham_effects"],["10/512"])
        self.assertEqual(pf["target_effect"],"0.18")
        self.assertEqual(pf["planned_comparisons"],3)

    def test_sign_holm_exact_and_zero_effect(self):
        self.assertEqual(e.exact_root_sign_p([0,0]),1.)
        self.assertEqual(e.exact_root_sign_p([1,1,1,-1,0]),5/16)
        self.assertEqual(e.holm({"a":.01,"b":.02,"c":.5}),{"a":.03,"b":.04,"c":.5})

    def test_native_bpdmodel_three_pass_mask_execution_parity(self):
        if importlib.util.find_spec("mechinterp") is None:
            self.skipTest("native BPD requires repository checkout")
        from mechinterp.bpd.components import BPDModel,PassMaskSchedule
        from mechinterp.bpd.config import AblationConfig
        class Base(torch.nn.Module):
            def __init__(self,weight):
                super().__init__()
                self.core=torch.nn.Linear(6,6,bias=False)
                with torch.no_grad():self.core.weight.copy_(weight)
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
