import copy
import unittest

from .gym02_export import stage_rows


def fixture():
    return {'window':0,'parent_model_sha256':'parent','parent_custody_sha256':'custody','feedback_cohort_id':'panel',
            'feedback_age':0,'feedback_loss0':2.,'charged_cpu_before':0.,
            'acquisition_start_cpu':.01,'scores_prefix':[.1]*16,'update_norms_prefix':[.2]*16,
            'stages':[{'stage':k,'cpu_to_stage':k/10} for k in (4,8,12,16)],
            'decision':'ACQUIRE_16','selected_indices':[0,1,2,3],'outcome_status':'committed',
            'actual_gain':.25,'cal_before':2.,'cal_after':1.75,'predicted_joint_native':.2,
            'committed':True,'cpu_to_proposal':1.7,'window_cpu_seconds':2.}


class TestGym02Export(unittest.TestCase):
    def test_stage_local_canary(self):
        w=fixture()
        a=stage_rows(w,task='E31',root=1,arm='C_analytic_full16')
        z=copy.deepcopy(w);z['scores_prefix'][12:]=[999.]*4;z['actual_gain']=.5;z['cal_after']=1.5
        b=stage_rows(z,task='E31',root=1,arm='C_analytic_full16')
        self.assertEqual(a[:4],b[:4])
        self.assertEqual([len(r['state']['scores']) for r in a],[0,4,8,12,16])
        self.assertTrue(all(r['outcome'] is None and not r['training_eligible'] for r in a[:-1]))
        self.assertEqual(a[-1]['preference']['chosen'],'SELECT_BATCH')
    def test_sham_and_unaudited(self):
        w=fixture()
        self.assertTrue(all(not r['training_eligible'] for r in stage_rows(w,task='E31',root=1,arm='F_shuffled_k4')))
        w.update(decision='ABSTAIN',outcome_status='missing_unaudited',actual_gain=None,
                 cal_before=None,cal_after=None,committed=False)
        r=stage_rows(w,task='E31',root=1,arm='D_joint_early12')[-1]
        self.assertIsNone(r['preference']);self.assertIsNone(r['outcome']['actual_gain'])
        self.assertNotIn('selected_indices',r['state'])


if __name__=='__main__':
    unittest.main()
