"""AMM-04 source-only engineering tests. No development or confirmation roots."""
from __future__ import annotations
import math
import importlib.util
import unittest
from unittest.mock import patch
import torch
from . import models as m
from . import experiment as e


class AMM04EngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher,cls.factor,cls.metrics,cls.geometry=e.trained_model_root('engineering',0)
        cls.cal=m.assay('engineering',0,cls.factor,'local_calibration',m.CAL_PAIRS)
        cls.test=m.assay('engineering',0,cls.factor,'engineering_test',m.PAIRS)

    def test_labels_defined_without_factors(self):
        ab=torch.tensor([[.1,.5],[-.3,.6]])
        a,b=m.labels(ab,'wren')
        self.assertTrue(torch.equal(a,ab[:,0]) and torch.equal(b,ab[:,1]))
        b,a=m.labels(ab,'basalt')
        self.assertTrue(torch.equal(a,ab[:,0]) and torch.equal(b,ab[:,1]))
        for i in range(8):
            x,y,z=m.donors('engineering',i,m.CAL_PAIRS,'local_calibration')
            self.assertEqual(z.shape,(m.CAL_PAIRS,))
            if i%2==0:
                self.assertTrue(torch.equal(y[:,3],z))
                self.assertTrue(torch.equal(y[:,2],x[:,2]))
            else:
                self.assertTrue(torch.equal(y[:,2],z))
                self.assertTrue(torch.equal(y[:,3],x[:,3]))

    def test_disjoint_root_seed_namespaces(self):
        sets=[{m.root_key(phase,i) for i in range(n)} for phase,n in m.PHASE_N.items()]
        for i in range(len(sets)):
            for j in range(i+1,len(sets)):
                self.assertFalse(sets[i]&sets[j])
        self.assertEqual(len(sets[-1]),128)
        self.assertNotEqual(m.seed('training/0001'),int('1'))

    def test_heldout_and_calibration_separate(self):
        cal=m.donors('engineering',0,m.CAL_PAIRS,'local_calibration')
        tst=m.donors('engineering',0,128,'heldout_test')
        self.assertFalse(torch.equal(cal[2],tst[2][:m.CAL_PAIRS]))
        for name,assay in [('cal',self.cal),('test',self.test)]:
            self.assertEqual(len(assay['candidate_bits']['component_0']),m.CAL_PAIRS if name=='cal' else 128)
            self.assertEqual(assay['candidate_hits']['component_1'],assay['candidate_bits']['component_1'].count('1'))

    def test_teacher_and_factors_learned(self):
        d=self.metrics
        self.assertGreater(min(d['teacher_update_norms']),.05)
        self.assertGreater(min(d['factor_update_norms']),.05)
        self.assertLess(d['teacher_behavioral_mse'],.08)
        self.assertTrue(math.isfinite(d['factor_behavioral_mse']))
        self.assertLess(d['factor_behavioral_mse'],.25)
        self.assertLess(float(torch.linalg.matrix_rank(self.teacher.weight().detach())),3)

    def test_equal_dose_dark_probes(self):
        d=self.test
        self.assertLess(d['dose_max_abs_error'],1e-5)
        self.assertEqual(d['current_readout_max_abs_error'],0.)
        self.assertEqual(d['n'],128)
        for key in ('component_0','component_1','random_dark','random_dark_b','sham','full_donor'):
            self.assertEqual(d['candidate_hits'][key],d['candidate_bits'][key].count('1'))

    def test_causal_geometry_and_gauge_invariant_signature(self):
        g=self.geometry
        self.assertEqual(len(g['features']),2)
        self.assertEqual(len(g['features'][0]),7)
        self.assertTrue(all(math.isfinite(z) for row in g['features'] for z in row))
        audit=m.gauge_check(self.factor,'wren')
        self.assertLess(audit['full_gl2_max_abs_error'],1e-9)
        if audit['gap']>.01:
            self.assertLess(audit['recanonicalized_rank1_max_abs_error'],1e-8)
        # A full-product preserving shear, followed by canonical SVD, should
        # produce the same model-conditioned features at a well-separated gap.
        with torch.no_grad():
            v=self.factor.V.detach().double();u=self.factor.U.detach().double()
            R=torch.tensor([[1.,.6],[0.,1.]],dtype=torch.float64)
            prod=(v@R)@(torch.linalg.inv(R)@u)
            q,s,vt=torch.linalg.svd(prod)
            f=m.Factors('engineering',0)
            f.V.copy_((q[:,:2]*s[:2].sqrt()).float())
            f.U.copy_((s[:2].sqrt()[:,None]*vt[:2]).float())
            other=m.geometry(f,'wren')
            for a,b in zip(g['features'],other['features']):
                self.assertLess(max(abs(x-y) for x,y in zip(a,b)),1e-5)

    @staticmethod
    def placeholder_memory():
        r={'schema':m.SCHEMA+'/training_memory','experiment':m.NAME,
           'source_sha256':e.source_sha256(),'independent_roots':64,
           'ridge_lambda':e.RIDGE_LAMBDA,
           'ridge_coefficients':[.5,0.,0.,0.,0.,0.,0.],
           'pooled_action':0,
           'rows':[{'root':m.root_key('training',i)} for i in range(64)]}
        r['sha256']=m.digest(r)
        return r

    def test_noisy_judge_is_independently_corrupted(self):
        feedback=e.report_calibration('engineering',0,self.cal)
        self.assertEqual(feedback['declared_flip_probability'],0.35)
        for c in (0,1):
            key=f'component_{c}'
            self.assertEqual(len(feedback['reported_bits'][key]),m.CAL_PAIRS)
            self.assertEqual(feedback['reported_hits'][key],feedback['reported_bits'][key].count('1'))
        self.assertNotIn('candidate_bits',feedback)
        self.assertEqual(e.JUDGE_FLIP_PROB,.35)

    def test_policy_equal_local_budget(self):
        memory=self.placeholder_memory()
        feedback=e.report_calibration('engineering',0,self.cal)
        a=e.choose_actions('engineering/0000',self.geometry,feedback,memory)
        self.assertEqual(set(a),{'memory_calibrated','local_only','memory_only','pooled','geometry_rule','random_component'})
        self.assertEqual(self.cal['n'],m.CAL_PAIRS)
        self.assertEqual(feedback['n_labeled_queries'],2*m.CAL_PAIRS)
        self.assertNotIn('candidate_hits',feedback)
        self.assertEqual(feedback,e.report_calibration('engineering',0,self.cal))
        self.assertTrue(all(x in (0,1) for x in a.values()))
        self.assertEqual(a['memory_only'],0)

    def test_preflight_rejects_before_any_model_fit(self):
        memory=self.placeholder_memory()
        with patch.object(e,'score_model',side_effect=AssertionError('SEALED ROOT REACHED')) as spy:
            for pf in (None,{}, {'decision':'REDESIGN_OR_PILOT_ONLY'},
                       {'decision':'READY_FOR_CONFIRMATION','experiment_id':m.NAME}):
                with self.assertRaises(ValueError):
                    e.evaluate('confirmation',memory,pf)
            with self.assertRaises(ValueError):
                e.evaluate('confirmation',None,None)
            spy.assert_not_called()

    def test_memory_validation_detects_mutation(self):
        memory=self.placeholder_memory()
        self.assertTrue(e.validate_memory(memory))
        memory['ridge_coefficients'][0]=.7
        with self.assertRaises(ValueError):e.validate_memory(memory)

    def test_root_level_preflight_shape(self):
        rows=[]
        for i in range(64):
            rows.append({'root':m.root_key('development',i),
                         'policy_hits':{'memory_calibrated':61,'local_only':51,
                                        'random_dark':40,'random_dark_b':42},
                         'repeatability':{'local_only_hits':52}})
        pf=e.preflight_input({'phase':'development','n_independent_roots':64,'rows':rows})
        self.assertEqual(len(pf['pilot']),64)
        self.assertEqual(pf['pilot'][0]['paired_effect'],'10/128')
        self.assertEqual(pf['pilot'][0]['sham_effects'],['-1/128'])
        self.assertEqual(pf['scientific_min_effect'],'0.08')
        self.assertEqual(pf['planned_comparisons'],3)
        self.assertEqual(pf['confirm_units_planned'],128)

    def test_exact_sign_holm(self):
        self.assertAlmostEqual(e.root_sign_p([1,1,1,-1,0]),5/16)
        self.assertEqual(e.root_sign_p([0,0]),1.)
        self.assertEqual(e.holm({'a':.01,'b':.02,'c':.5}),{'a':.03,'b':.04,'c':.5})

    def test_native_bpd_schedule_parity(self):
        if importlib.util.find_spec('mechinterp.bpd.components') is None:
            self.skipTest('native test executes in BPD repository checkout')
        from mechinterp.bpd.components import BPDModel,PassMaskSchedule
        from mechinterp.bpd.config import AblationConfig
        class Base(torch.nn.Module):
            def __init__(self,w):
                super().__init__();self.core=torch.nn.Linear(4,4,bias=False)
                with torch.no_grad():self.core.weight.copy_(w)
            def forward(self,h):return self.core(h)
        net=BPDModel(Base(self.teacher.weight().detach()),{'core':2},max_passes=2,
                     ablation=AblationConfig(mode='zero',sphere_rescale=False),init_svd=False)
        with torch.no_grad():
            net.wrappers['core'].V.copy_(self.factor.V)
            net.wrappers['core'].U.copy_(self.factor.U)
            h,_=m.data('engineering',0,16,'native_parity')
            sched=PassMaskSchedule(torch.ones(16,2,1,2),torch.zeros(16,2,1))
            with net.masked_components({'core':sched}):
                y1=net.target_model(h[:,None,:])[:,0]
                y2=net.target_model(y1[:,None,:])[:,0]
            direct1=self.factor(h);direct2=self.factor(direct1)
            self.assertLess(float((y1-direct1).abs().max()),1e-5)
            self.assertLess(float((y2-direct2).abs().max()),1e-5)


if __name__=='__main__':unittest.main()
