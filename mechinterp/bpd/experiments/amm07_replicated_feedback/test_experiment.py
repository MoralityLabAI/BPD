"""AMM-07 source-only tests. All trained roots come from excluded engineering."""
from __future__ import annotations
import importlib.util
import math
import unittest
from unittest.mock import patch
import torch
from . import models as m,case,noise,experiment as e

class AMM07Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher,cls.factor,cls.metrics=m.fit('engineering',0)
        cls.cal=case.assay('engineering',0,cls.factor,'local_calibration',m.CAL_PAIRS)
        cls.feedback=noise.reported_calibration('engineering',0,cls.cal)
        cls.decision=noise.local_decision(cls.cal['features'],cls.feedback,[.5]+[0.]*10)
        cls.test=case.assay('engineering',0,cls.factor,'engineering_test',m.PAIRS)

    def test_new_phase_namespaces_and_balanced_quality(self):
        self.assertEqual(m.NAME,'BPD-AMM-07')
        self.assertEqual(m.PAIRS,1024)
        self.assertEqual(m.CAL_PAIRS,16)
        self.assertEqual(m.PHASE_N['confirmation'],128)
        groups=[{m.root_key(p,i) for i in range(n)} for p,n in m.PHASE_N.items()]
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):self.assertFalse(groups[i]&groups[j])
        for p in ('training','development','confirmation'):
            n=m.PHASE_N[p]
            counts={(family,q):sum(m.family_for(i)==family and noise.hidden_quality(i)==q
                                    for i in range(n)) for family in m.FAMILIES
                                    for q in ('low_noise','high_noise')}
            self.assertEqual(set(counts.values()),{n//4})

    def test_independent_behavioral_target_and_mixed_nuisance(self):
        for i in range(8):
            recipient,donor,target=m.donors('engineering',i,m.PAIRS,'engineering_test')
            col=3 if i%2==0 else 2
            self.assertTrue(torch.equal(donor[:,col],target))
            self.assertGreater(float((donor[:,2]-recipient[:,2]).abs().min()),.149)
            self.assertGreater(float((donor[:,3]-recipient[:,3]).abs().min()),.149)
        self.assertLess(self.metrics['teacher_behavioral_mse'],.08)
        self.assertGreater(min(self.metrics['teacher_update_norms']),.05)
        self.assertGreater(min(self.metrics['factor_update_norms']),.05)
        self.assertEqual(int(torch.linalg.matrix_rank(self.teacher.weight().detach())),2)

    def test_physical_dose_and_readout_zero(self):
        d=self.test['diagnostics']
        self.assertLess(d['max_equal_dose_error'],1e-5)
        self.assertEqual(d['max_immediate_readout_displacement'],0.)
        self.assertEqual(self.test['n'],m.PAIRS)
        for key,bits in self.test['candidate_bits'].items():
            self.assertEqual(len(bits),m.PAIRS)
            self.assertEqual(self.test['candidate_hits'][key],bits.count('1'))

    def test_three_unlabeled_probes_have_no_donor_goal(self):
        rec,don,_=m.donors('engineering',0,m.PAIRS,'engineering_test')
        raw=case.candidate_features(self.factor,rec,don,'wren')
        self.assertEqual(tuple(raw['features'].shape),(m.PAIRS,2,11))
        self.assertTrue(bool(torch.isfinite(raw['features']).all()))
        self.assertNotIn('target',raw)
        self.assertNotIn('donor_future',raw)
        self.assertNotIn('candidate_bits',raw)

    def test_duplicate_reports_independent_deterministic_and_blinded(self):
        d=self.feedback
        self.assertEqual(d['n_noisy_reports'],64)
        self.assertEqual(d['n_calibration_pairs'],16)
        self.assertEqual(d['candidate_actions'],2)
        self.assertEqual(d['judge_repeats'],2)
        self.assertEqual(d,noise.reported_calibration('engineering',0,self.cal))
        self.assertNotIn('hidden_quality',d)
        self.assertNotIn('clean_label',d)
        for pair in d['reported_replicates'].values():
            self.assertEqual(len(pair),2)
            self.assertTrue(all(len(bits)==16 for bits in pair))
        y=noise.observations(d)
        self.assertEqual(tuple(y.shape),(16,2,2))
        self.assertTrue(bool(((y==0)|(y==1)).all()))

    def test_disagreement_likelihood_identifies_quality(self):
        y=torch.zeros((16,2,2),dtype=torch.float64)
        low=noise.quality_posterior(y)
        self.assertEqual(low['disagreements_first_half'],0)
        self.assertGreater(low['posterior_low'],.9)
        y[:8,:,1]=1
        high=noise.quality_posterior(y)
        self.assertEqual(high['disagreements_first_half'],16)
        self.assertLess(high['posterior_low'],.01)
        before=noise.quality_posterior(y)
        y[8:,:,1]=1
        self.assertEqual(before,noise.quality_posterior(y))

    def test_bayesian_truth_posterior_tracks_reliability(self):
        p=torch.full((1,2),.5,dtype=torch.float64)
        agreeing=torch.ones((1,2,2),dtype=torch.float64)
        low=noise.posterior_clean_truth(p,agreeing,1.0)
        high=noise.posterior_clean_truth(p,agreeing,0.0)
        self.assertTrue(bool((low>high).all()))
        self.assertGreater(float(low.min()),.9)
        mixed=torch.tensor([[[1.,0.],[0.,1.]]],dtype=torch.float64)
        result=noise.posterior_clean_truth(p,mixed,.5)
        self.assertTrue(bool(torch.allclose(result,p,atol=1e-12)))

    def test_frozen_equal_budget_policy_fitting(self):
        d=self.decision
        self.assertEqual(d['n_noisy_reports'],64)
        self.assertEqual(d['n_quality_cases'],8)
        self.assertEqual(d['n_shared_fit_cases'],8)
        for name in ('aware_residual','blind_residual','blind_all_residual','local_ridge'):
            self.assertEqual(len(d[name]),11)
        actions=noise.choose_actions(self.test['features'],d,'engineering/0000',0)
        self.assertEqual(set(actions),{'quality_aware','quality_blind','blind_all',
                                      'memory_only','local_ridge','pooled','random_component'})
        for choice in actions.values():
            self.assertEqual(len(choice),m.PAIRS)
            self.assertTrue(all(c in (0,1) for c in choice))
        self.assertEqual(d,noise.local_decision(self.cal['features'],self.feedback,[.5]+[0.]*10))

    def test_invalid_replicates_refused(self):
        broken=dict(self.feedback)
        broken['reported_replicates']={'component_0':['1'],'component_1':['0']}
        with self.assertRaises(ValueError):
            noise.local_decision(self.cal['features'],broken,[.5]+[0.]*10)
        with self.assertRaises(ValueError):
            noise.posterior_clean_truth(torch.zeros(1,2),torch.zeros(1,2,1),.5)

    def test_fidelity_flag_never_excludes_root(self):
        d=self.metrics
        self.assertTrue(math.isfinite(d['factor_behavioral_mse']))
        fit=e.model_fidelity(d)
        broken=dict(d);broken['factor_behavioral_mse']=.5
        self.assertFalse(e.model_fidelity(broken))
        self.assertIn(fit,(True,False))

    def test_gauge_invariant_rank_two_basis(self):
        geom=m.gauge_check(self.factor,'wren')
        self.assertLess(geom['full_gl2_max_abs_error'],1e-9)
        if geom['gap']>.01:
            self.assertLess(geom['recanonicalized_rank1_max_abs_error'],1e-8)

    def test_calibration_heldout_and_repeatability_separated(self):
        cal=m.donors('engineering',0,16,'local_calibration')
        a=m.donors('engineering',0,1024,'heldout_test')
        b=m.donors('engineering',0,1024,'repeatability')
        self.assertFalse(torch.equal(cal[2],a[2][:16]))
        self.assertFalse(torch.equal(a[2],b[2]))

    @staticmethod
    def fake_memory():
        rows=[{'root':m.root_key('training',i),'assay':
               {'candidate_bits':{'component_0':'0','component_1':'1'}}} for i in range(64)]
        digest=m.digest([{'root':r['root'],'candidate_bits':r['assay']['candidate_bits']} for r in rows])
        d={'schema':m.SCHEMA+'/training_memory','source_sha256':e.source_sha256(),
           'experiment':e.NAME,'phase':'training','independent_training_roots':64,
           'rows':rows,'train_rows_sha256':digest,
           'coefficients':[.5]+[0.]*10,'pooled_action':0}
        d['sha256']=m.digest({k:v for k,v in d.items() if k!='rows'})
        return d

    def test_compact_policy_memory_guards_numeric_source(self):
        d=self.fake_memory()
        self.assertTrue(e.validate_memory(d))
        d['coefficients'][0]=.50001
        with self.assertRaises(ValueError):e.validate_memory(d)
        d=self.fake_memory()
        d['rows'][0]['assay']['candidate_bits']['component_0']='1'
        with self.assertRaises(ValueError):e.validate_memory(d)

    def test_confirmation_hard_guard_before_root_access(self):
        memory=self.fake_memory()
        ready={'schema':'EvalMath/mechinterp-power-preflight/v1','decision':'READY_FOR_CONFIRMATION',
               'experiment_id':e.NAME,'pilot_worlds':64,'planned_confirmation_worlds':128,
               'planned_comparisons':3,'mean_pilot_effect':.2,
               'mechanistic_variation_floor':.08,'reasons':[]}
        with patch.object(e,'score_one',side_effect=AssertionError('SEALED ROOT OPENED')) as spy:
            for pf,release in ((None,None),({},None),({'decision':'REDESIGN_OR_PILOT_ONLY'},None),
                               (ready,None),(ready,{'release_authorized':True})):
                with self.assertRaises(ValueError):
                    e.evaluate('confirmation',memory,pf,release)
            spy.assert_not_called()

    def test_rootwise_preflight_and_exact_sign_family(self):
        rows=[{'root':m.root_key('development',i),
               'policy_hits':{'quality_aware':500,'quality_blind':480},
               'repeatability':{'blind_hits':470}} for i in range(64)]
        pf=e.preflight_input({'phase':'development','n_independent_roots':64,'rows':rows})
        self.assertEqual(pf['planned_comparisons'],3)
        self.assertEqual(pf['pilot'][0]['paired_effect'],'20/1024')
        self.assertEqual(pf['pilot'][0]['sham_effects'],['10/1024'])
        self.assertEqual(e.root_sign_p([0,0]),1.)
        self.assertEqual(e.root_sign_p([1,1,1,-1,0]),5/16)
        self.assertEqual(e.holm({'a':.01,'b':.02,'c':.5}),{'a':.03,'b':.04,'c':.5})

    def test_native_tied_bpd_pass_mask_parity(self):
        if importlib.util.find_spec('mechinterp') is None:
            self.skipTest('full BPD checkout required for native parity')
        from mechinterp.bpd.components import BPDModel,PassMaskSchedule
        from mechinterp.bpd.config import AblationConfig
        class Base(torch.nn.Module):
            def __init__(self,weight):
                super().__init__();self.core=torch.nn.Linear(4,4,bias=False)
                with torch.no_grad():self.core.weight.copy_(weight)
            def forward(self,x):return self.core(x)
        model=BPDModel(Base(self.teacher.weight().detach()),{'core':2},max_passes=2,
                       ablation=AblationConfig(mode='zero',sphere_rescale=False),init_svd=False)
        with torch.no_grad():
            model.wrappers['core'].V.copy_(self.factor.V)
            model.wrappers['core'].U.copy_(self.factor.U)
            h,_=m.data('engineering',0,16,'native_parity')
            schedule=PassMaskSchedule(torch.ones(16,2,1,2),torch.zeros(16,2,1))
            with model.masked_components({'core':schedule}):
                y1=model.target_model(h[:,None,:])[:,0]
                y2=model.target_model(y1[:,None,:])[:,0]
            a=self.factor(h);b=self.factor(a)
            self.assertLess(float((a-y1).abs().max()),1e-5)
            self.assertLess(float((b-y2).abs().max()),1e-5)

if __name__=='__main__':unittest.main()
