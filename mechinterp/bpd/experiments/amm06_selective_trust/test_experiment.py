"""AMM-06 tests run exclusively on excluded engineering roots."""
from __future__ import annotations
import importlib.util
import math
import unittest
from unittest.mock import patch
import torch
from . import models as m,case,gate,experiment as e


class AMM06Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher,cls.f,cls.metrics=m.fit('engineering',0)
        cls.cal=case.assay('engineering',0,cls.f,'local_calibration',m.CAL_PAIRS)
        cls.test=case.assay('engineering',0,cls.f,'engineering_test',m.PAIRS)
        cls.feedback=gate.reported_calibration('engineering',0,cls.cal)
        cls.prior=[.5]+[0.]*10
        cls.decision=gate.local_decision(cls.cal['features'],cls.feedback,cls.prior)

    def test_phases_and_balanced_hidden_quality(self):
        self.assertEqual(m.NAME,'BPD-AMM-06')
        self.assertEqual(m.PAIRS,256)
        self.assertEqual(m.CAL_PAIRS,8)
        self.assertEqual(m.PHASE_N['confirmation'],128)
        ids=[{m.root_key(phase,i) for i in range(n)} for phase,n in m.PHASE_N.items()]
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                self.assertFalse(ids[i]&ids[j])
        for phase in ('training','development','confirmation'):
            n=m.PHASE_N[phase]
            counts={(fam,q):sum(
                m.family_for(i)==fam and gate.hidden_quality(i)==q for i in range(n))
                for fam in m.FAMILIES for q in ('low_noise','high_noise')}
            self.assertEqual(set(counts.values()),{n//4})
        self.assertNotEqual(m.seed('training/0000'),m.seed('development/0000'))

    def test_recurrent_teacher_and_factor_training(self):
        d=self.metrics
        self.assertGreater(min(d['teacher_update_norms']),.05)
        self.assertGreater(min(d['factor_update_norms']),.05)
        self.assertLess(d['teacher_behavioral_mse'],.08)
        self.assertEqual(int(torch.linalg.matrix_rank(self.teacher.weight().detach())),2)

    def test_independent_target_and_mixed_nuisance(self):
        for i in range(8):
            recipient,donor,target=m.donors('engineering',i,256,'engineering_test')
            relevant=3 if i%2==0 else 2
            nuisance=2 if relevant==3 else 3
            self.assertTrue(torch.equal(donor[:,relevant],target))
            self.assertGreater(float((donor[:,nuisance]-recipient[:,nuisance]).abs().min()),.149)
            self.assertGreater(float((donor[:,relevant]-recipient[:,relevant]).abs().min()),.299)

    def test_casewise_features_and_audit_isolation(self):
        rec,donor,_=m.donors('engineering',0,m.PAIRS,'engineering_test')
        features=case.candidate_features(self.f,rec,donor,'wren')
        self.assertEqual(tuple(features['features'].shape),(m.PAIRS,2,11))
        self.assertTrue(torch.isfinite(features['features']).all())
        self.assertNotIn('target',features)
        self.assertNotIn('donor_future',features)
        self.assertEqual(self.test['features'],features['features'].tolist())

    def test_equal_dose_and_case_bits(self):
        d=self.test['diagnostics']
        self.assertLess(d['max_equal_dose_error'],1e-5)
        self.assertEqual(d['max_immediate_readout_displacement'],0.)
        self.assertEqual(self.test['n'],256)
        for name,bits in self.test['candidate_bits'].items():
            self.assertEqual(len(bits),256)
            self.assertEqual(self.test['candidate_hits'][name],bits.count('1'))

    def test_noisy_feedback_omits_hidden_regime(self):
        self.assertEqual(self.feedback['n_pairs'],8)
        self.assertEqual(self.feedback['n_feedback_labels'],16)
        self.assertEqual(self.feedback['known_noise_population'],[.10,.45])
        self.assertEqual(self.feedback,gate.reported_calibration('engineering',0,self.cal))
        self.assertNotIn('judge_quality',self.feedback)
        self.assertNotIn('clean_bits',self.feedback)
        self.assertNotIn('source_labels',self.feedback)
        for key,bits in self.feedback['reported_bits'].items():
            self.assertEqual(len(bits),8)
            self.assertEqual(self.feedback['reported_hits'][key],bits.count('1'))

    def test_likelihood_marginalizes_shared_root_regime(self):
        predicted=torch.full((8,),.5,dtype=torch.float64)
        observed=torch.ones(8,dtype=torch.float64)
        self.assertAlmostEqual(gate.log_mixture_likelihood(predicted,observed),-8*math.log(2.),12)
        self.assertAlmostEqual(gate.ETA_MARGINAL,.275)
        self.assertAlmostEqual(gate.THRESHOLD_LOG_BF,math.log(2.))

    def test_split_evidence_frozen_and_no_label_leak(self):
        d=self.decision
        self.assertEqual(d['fit_calibration_pairs'],4)
        self.assertEqual(d['validation_calibration_pairs'],4)
        self.assertEqual(d['n_noisy_labels'],16)
        self.assertEqual(len(d['full_residual_weights']),11)
        self.assertEqual(len(d['local_ridge_weights']),11)
        self.assertTrue(math.isfinite(d['log_bayes_factor']))
        self.assertEqual(d,gate.local_decision(self.cal['features'],self.feedback,self.prior))
        self.assertNotIn('true_noise_regime',d)
        self.assertNotIn('calibration_clean_bits',d)

    def test_switch_selectivity_is_only_the_frozen_gate(self):
        decision=dict(self.decision);decision['prior_weights']=self.prior
        decision['full_residual_weights']=[0.]*11
        decision['local_ridge_weights']=[0.]*11
        for accept in (True,False):
            decision['accepted']=accept
            picks=gate.choose_actions(self.test['features'],decision,'engineering/0000',0)
            self.assertEqual(picks['selective_trust'],
                             picks['always_update'] if accept else picks['memory_only'])
            self.assertTrue(all(len(actions)==256 for actions in picks.values()))

    def test_invalid_feedback_fails_closed(self):
        feedback=dict(self.feedback)
        feedback['reported_bits']={'component_0':'1','component_1':'0'}
        with self.assertRaises(ValueError):
            gate.local_decision(self.cal['features'],feedback,self.prior)

    def test_gauge_recanonicalization(self):
        result=m.gauge_check(self.f,'wren')
        self.assertLess(result['full_gl2_max_abs_error'],1e-9)
        if result['gap']>.01:
            self.assertLess(result['recanonicalized_rank1_max_abs_error'],1e-8)

    def test_calibration_test_repeatability_disjoint(self):
        a=m.donors('engineering',0,8,'local_calibration')
        b=m.donors('engineering',0,256,'heldout_test')
        c=m.donors('engineering',0,256,'repeatability')
        self.assertFalse(torch.equal(a[2],b[2][:8]))
        self.assertFalse(torch.equal(b[2],c[2]))

    @staticmethod
    def fake_memory():
        out={'schema':m.SCHEMA+'/training_memory','source_sha256':e.source_sha256(),
             'experiment':e.NAME,'phase':'training','independent_training_roots':64,
             'rows':[{'root':m.root_key('training',i)} for i in range(64)],
             'coefficients':[.5]+[0.]*10,'pooled_action':0}
        out['sha256']=m.digest(out)
        return out

    def test_confirmation_seal_prevents_even_first_root(self):
        mem=self.fake_memory()
        pf={'schema':'EvalMath/mechinterp-power-preflight/v1','decision':'READY_FOR_CONFIRMATION',
            'experiment_id':e.NAME,'pilot_worlds':64,'planned_confirmation_worlds':128,
            'planned_comparisons':3,'mean_pilot_effect':.2,
            'mechanistic_variation_floor':.08,'reasons':[]}
        with patch.object(e,'score_one',side_effect=AssertionError('SEALED ROOT OPENED')) as spy:
            for pre,release in ((None,None),({},None),
                                ({'decision':'REDESIGN_OR_PILOT_ONLY'},None),
                                (pf,None),(pf,{'release_authorized':True})):
                with self.assertRaises(ValueError):
                    e.evaluate('confirmation',mem,pre,release)
            spy.assert_not_called()

    def test_root_preflight_shape_and_sign_holm(self):
        rows=[{'root':m.root_key('development',i),
               'policy_hits':{'selective_trust':150,'always_update':130},
               'repeatability':{'always_update_hits':125}} for i in range(64)]
        pf=e.preflight_input({'phase':'development','n_independent_roots':64,'rows':rows})
        self.assertEqual(pf['pilot'][0]['paired_effect'],'20/256')
        self.assertEqual(pf['pilot'][0]['sham_effects'],['5/256'])
        self.assertEqual(pf['planned_comparisons'],3)
        self.assertEqual(pf['target_effect'],'0.15')
        self.assertEqual(e.root_sign_p([0,0]),1.)
        self.assertEqual(e.root_sign_p([1,1,1,-1,0]),5/16)
        self.assertEqual(e.holm({'a':.01,'b':.02,'c':.5}),{'a':.03,'b':.04,'c':.5})

    def test_native_tied_component_mask_parity(self):
        if importlib.util.find_spec('mechinterp') is None:
            self.skipTest('native BPDModel requires repo checkout')
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
            sched=PassMaskSchedule(torch.ones(16,2,1,2),torch.zeros(16,2,1))
            with net.masked_components({'core':sched}):
                y1=net.target_model(h[:,None,:])[:,0]
                y2=net.target_model(y1[:,None,:])[:,0]
            x1=self.f(h);x2=self.f(x1)
            self.assertLess(float((y1-x1).abs().max()),1e-5)
            self.assertLess(float((y2-x2).abs().max()),1e-5)


if __name__=='__main__':unittest.main()
