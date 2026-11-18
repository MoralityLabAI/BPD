"""AMM-05 source-only tests: only excluded engineering model roots are fitted."""
from __future__ import annotations
import importlib.util
import math
from unittest.mock import patch
import unittest
import torch
from . import models as m,case,experiment as e


class AMM05Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher,cls.f,cls.metrics=m.fit('engineering',0)
        cls.cal=case.assay('engineering',0,cls.f,'local_calibration',4)
        cls.test=case.assay('engineering',0,cls.f,'engineering_test',128)
        cls.geo=m.geometry(cls.f,'wren')
        cls.feedback=e.judge_report('engineering',0,cls.cal)

    def test_new_source_namespace(self):
        self.assertEqual(m.NAME,'BPD-AMM-05')
        self.assertEqual(m.PHASE_N['confirmation'],128)
        ids=[{m.root_key(phase,i) for i in range(n)} for phase,n in m.PHASE_N.items()]
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                self.assertFalse(ids[i] & ids[j])
        self.assertNotEqual(m.seed('engineering/0000','teacher_init'),m.seed('training/0000','teacher_init'))

    def test_mixed_target_nuisance_edits_have_independent_roles(self):
        for i in range(8):
            rec,don,t=m.donors('engineering',i,128,'engineering_test')
            self.assertTrue(((don[:,2]-rec[:,2]).abs()>=.15-1e-5).all())
            self.assertTrue(((don[:,3]-rec[:,3]).abs()>=.15-1e-5).all())
            target=3 if i%2==0 else 2
            other=2 if target==3 else 3
            self.assertTrue(torch.equal(t,don[:,target]))
            self.assertGreater(float((don[:,other]-rec[:,other]).abs().min()),.14)

    def test_target_source_is_external_to_model_fit(self):
        ab=torch.tensor([[.1,.5],[-.3,.6]])
        a,b=m.labels(ab,'wren')
        self.assertTrue(torch.equal(a,ab[:,0]) and torch.equal(b,ab[:,1]))
        b,a=m.labels(ab,'basalt')
        self.assertTrue(torch.equal(a,ab[:,0]) and torch.equal(b,ab[:,1]))
        self.assertGreater(min(self.metrics['teacher_update_norms']),.05)
        self.assertGreater(min(self.metrics['factor_update_norms']),.05)
        self.assertLess(self.metrics['teacher_behavioral_mse'],.08)
        self.assertLessEqual(int(torch.linalg.matrix_rank(self.teacher.weight().detach())),2)

    def test_three_unlabelled_probe_features(self):
        rec,don,target=m.donors('engineering',0,128,'engineering_test')
        raw=case.candidate_features(self.f,rec,don,'wren')
        self.assertEqual(tuple(raw['features'].shape),(128,2,11))
        self.assertTrue(torch.isfinite(raw['features']).all())
        self.assertEqual(self.test['features'],raw['features'].tolist())
        self.assertNotIn('target',raw)
        self.assertNotIn('donor_future',raw)
        self.assertNotIn('candidate_bits',raw)

    def test_physical_matching_and_bit_denominators(self):
        a=self.test['diagnostics']
        self.assertLess(a['max_equal_dose_error'],1e-5)
        self.assertEqual(a['max_immediate_readout_displacement'],0.)
        self.assertEqual(self.test['n'],128)
        for k,bits in self.test['candidate_bits'].items():
            self.assertEqual(len(bits),128)
            self.assertEqual(self.test['candidate_hits'][k],bits.count('1'))
        self.assertLessEqual(a['casewise_oracle_hits_audit_only'],128)

    def test_noisy_labels_are_report_only_and_seed_fixed(self):
        v=self.feedback
        self.assertEqual(v['n_feedback_labels'],8)
        self.assertEqual(v['flip_probability'],.35)
        self.assertEqual(v,e.judge_report('engineering',0,self.cal))
        self.assertNotIn('candidate_bits',v)
        self.assertNotIn('calibration_clean_audit',v)
        self.assertNotIn('reported_flip_counts_audit_only',v)
        for k,bits in v['reported_bits'].items():
            self.assertEqual(len(bits),4)
            self.assertEqual(v['reported_hits'][k],bits.count('1'))

    def test_equal_budget_local_ridge_and_memory_policy(self):
        mem={'coefficients':[.5]+[0.]*10,'pooled_action':0}
        local,res=e.calibration_weights(self.cal['features'],self.feedback,mem)
        self.assertEqual(len(local),11)
        self.assertEqual(len(res),11)
        decisions=e.choose(self.test['features'],self.geo,self.feedback,mem,local,res,'engineering/0000')
        self.assertEqual(set(decisions),{'memory_calibrated','local_ridge','local_count',
                                         'memory_only','geometry_rule','pooled','random_component'})
        for choices in decisions.values():
            self.assertEqual(len(choices),128)
            self.assertTrue(all(c in (0,1) for c in choices))

    def test_gl2_gauge_recanonicalization(self):
        audit=m.gauge_check(self.f,'wren')
        self.assertLess(audit['full_gl2_max_abs_error'],1e-9)
        if audit['gap']>.01:self.assertLess(audit['recanonicalized_rank1_max_abs_error'],1e-8)

    def test_calibration_and_repeatability_streams_disjoint(self):
        cal=m.donors('engineering',0,4,'local_calibration')
        rep=m.donors('engineering',0,128,'repeatability')
        held=m.donors('engineering',0,128,'heldout_test')
        self.assertFalse(torch.equal(cal[2],rep[2][:4]))
        self.assertFalse(torch.equal(held[2],rep[2]))

    @staticmethod
    def fake_memory():
        memory={'schema':m.SCHEMA+'/training_memory','source_sha256':e.source_sha256(),
                'experiment':e.NAME,'phase':'training','independent_training_roots':64,
                'rows':[{'root':m.root_key('training',i)} for i in range(64)],
                'coefficients':[.5]+[0.]*10,'pooled_action':0}
        memory['sha256']=m.digest(memory)
        return memory

    def test_confirmation_sealed_before_any_root_access(self):
        mem=self.fake_memory()
        pf={'schema':'EvalMath/mechinterp-power-preflight/v1',
            'decision':'READY_FOR_CONFIRMATION','experiment_id':e.NAME,
            'pilot_worlds':64,'planned_confirmation_worlds':128,
            'planned_comparisons':3,'mean_pilot_effect':.2,
            'mechanistic_variation_floor':.08,'reasons':[]}
        with patch.object(e,'score_one',side_effect=AssertionError('SEALED ROOT REACHED')) as spy:
            for gate,release in ((None,None),({'decision':'REDESIGN_OR_PILOT_ONLY'},None),
                                 (pf,None),(pf,{'release_authorized':True})):
                with self.assertRaises(ValueError):e.evaluate('confirmation',mem,gate,release)
            spy.assert_not_called()

    def test_preflight_primary_root_units_and_sham(self):
        fake={'phase':'development','n_independent_roots':64,
              'rows':[{'root':m.root_key('development',i),
                       'policy_hits':{'memory_calibrated':60,'local_ridge':50},
                       'repeatability':{'local_ridge_hits':45}} for i in range(64)]}
        pf=e.preflight_input(fake)
        self.assertEqual(pf['planned_comparisons'],3)
        self.assertEqual(pf['pilot'][0]['paired_effect'],'10/128')
        self.assertEqual(pf['pilot'][0]['sham_effects'],['5/128'])
        self.assertEqual(len({r['unit_id'] for r in pf['pilot']}),64)

    def test_exact_root_sign_and_holm(self):
        self.assertEqual(e.root_sign_p([0,0]),1.)
        self.assertEqual(e.root_sign_p([1,1,1,-1,0]),5/16)
        self.assertEqual(e.holm({'a':.01,'b':.02,'c':.5}),{'a':.03,'b':.04,'c':.5})

    def test_native_bpd_tied_component_schedule_parity(self):
        if importlib.util.find_spec('mechinterp') is None:
            self.skipTest('native parity requires full BPD checkout')
        from mechinterp.bpd.components import BPDModel,PassMaskSchedule
        from mechinterp.bpd.config import AblationConfig
        class Base(torch.nn.Module):
            def __init__(self,weight):
                super().__init__();self.core=torch.nn.Linear(4,4,bias=False)
                with torch.no_grad():self.core.weight.copy_(weight)
            def forward(self,x):return self.core(x)
        net=BPDModel(Base(self.teacher.weight().detach()),{'core':2},max_passes=2,
                     ablation=AblationConfig(mode='zero',sphere_rescale=False),init_svd=False)
        with torch.no_grad():
            net.wrappers['core'].V.copy_(self.f.V)
            net.wrappers['core'].U.copy_(self.f.U)
            h,_=m.data('engineering',0,16,'native_parity')
            schedule=PassMaskSchedule(torch.ones(16,2,1,2),torch.zeros(16,2,1))
            with net.masked_components({'core':schedule}):
                y1=net.target_model(h[:,None,:])[:,0]
                y2=net.target_model(y1[:,None,:])[:,0]
            x1=self.f(h);x2=self.f(x1)
            self.assertLess(float((y1-x1).abs().max()),1e-5)
            self.assertLess(float((y2-x2).abs().max()),1e-5)


if __name__=='__main__':unittest.main()
