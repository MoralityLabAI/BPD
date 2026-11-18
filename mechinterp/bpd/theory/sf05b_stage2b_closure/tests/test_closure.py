"""SF05B Stage2-B engineering tests, training-family only.

No new scientific roots or heldout teacher-model scores are evaluated here.
"""
import unittest

import torch

from mechinterp.bpd.theory.sf05b_source_families.source_families import (
    change_last_source, make_batch,
)
from mechinterp.bpd.theory.sf05b_stage1.teacher import train_teacher, public_query
from mechinterp.bpd.theory.sf05b_stage2b_closure.experiment import (
    ARM_WEIGHTS, run, train_teacher_arm, training_only_closure_diagnostics,
)


class SourceClosureEngineering(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_frozen_four_arms_and_coefficients(self):
        self.assertEqual(ARM_WEIGHTS,{
            "baseline":(0.,0.),
            "current_only":(1.,0.),
            "public_only":(0.,1.),
            "joint":(1.,1.),
        })

    def test_baseline_is_exact_original_teacher_training(self):
        x=train_teacher_arm(901,"baseline",updates=5,batch_size=16)
        y=train_teacher(901,updates=5,batch_size=16)
        for key in x.state_dict():
            self.assertTrue(torch.equal(x.state_dict()[key],y.state_dict()[key]),key)

    def test_regularizers_are_active_for_train_only_histories(self):
        baseline=train_teacher_arm(901,"baseline",updates=5,batch_size=24)
        current=train_teacher_arm(901,"current_only",updates=5,batch_size=24)
        public=train_teacher_arm(901,"public_only",updates=5,batch_size=24)
        joint=train_teacher_arm(901,"joint",updates=5,batch_size=24)
        for teacher in (current,public,joint):
            self.assertTrue(torch.isfinite(teacher.C).all().item())
            self.assertGreater(float((teacher.C-baseline.C).abs().max()),1e-7)
        self.assertGreater(float((joint.C-current.C).abs().max()),1e-7)
        self.assertGreater(float((joint.C-public.C).abs().max()),1e-7)

    def test_source_pair_has_one_tag_difference_and_same_public_scalar(self):
        batch=make_batch(901105,32,"train")
        donor,_,_=change_last_source(batch,901201)
        changed=(batch.events-donor).abs().sum(dim=-1)>0
        self.assertTrue((changed.sum(1)==1).all().item())
        self.assertTrue(torch.equal(batch.events[:,:,3:],donor[:,:,3:]))
        public=public_query(batch)
        self.assertTrue(torch.equal(public[:,9],batch.public_value))

    def test_train_only_closure_diagnostics_are_finite(self):
        t=train_teacher_arm(901,"joint",updates=5,batch_size=24)
        x=training_only_closure_diagnostics(901,"joint",t,n=40)
        self.assertEqual(x["family"],"train")
        self.assertEqual(x["arm"],"joint")
        self.assertEqual(x["n"],40)
        for k in ("current_prequery_abs_difference_mean","public_clean_tag_only_abs_change_mean"):
            self.assertGreaterEqual(x[k],0.)
            self.assertTrue(torch.isfinite(torch.tensor(x[k])).item())

    def test_scientific_root_ids_and_unknown_arms_refused_before_training(self):
        with self.assertRaises(ValueError):
            train_teacher_arm(6101,"joint",updates=1,batch_size=16)
        with self.assertRaises(ValueError):
            train_teacher_arm(7101,"joint",updates=1,batch_size=16)
        with self.assertRaises(ValueError):
            train_teacher_arm(901,"unknown",updates=1,batch_size=16)
        with self.assertRaises(ValueError):
            run((901,6101),n=16,shams=1,updates=1,factor_steps=1,policy_steps=1)


if __name__=="__main__":
    unittest.main()
