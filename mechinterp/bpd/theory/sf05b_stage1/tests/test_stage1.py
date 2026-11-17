"""Stage 1 engineering checks use only F_train and excluded seeds."""
import inspect
import unittest

import torch

from mechinterp.bpd.theory.sf05b_source_families.source_families import (
    make_batch, change_last_source,
)
from mechinterp.bpd.theory.sf05b_stage1.teacher import (
    HIDDEN, INNER, Teacher, Factors, train_teacher, train_factors,
    recipient_donor_edit, public_query,
)
from mechinterp.bpd.theory.sf05b_stage1.policy import (
    NonprivilegedPolicy, predict_from_envelope, train_policy,
    CORRECTION_FRACTION, CORRECTION_MAX_NORM,
)
from mechinterp.bpd.theory.sf05b_stage1.evaluate import run_root, EXCLUDED


class StageOneEngineering(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_shared_teacher_valid_write_and_skipped_padding(self):
        teacher=Teacher(901)
        b=make_batch(90311,40,"train")
        t=teacher.rollout(b)
        h=torch.zeros(40,HIDDEN)
        for j in range(6):
            candidate=teacher.step(h,b.events[:,j])
            h=torch.where(b.event_mask[:,j,None],candidate,h)
        self.assertTrue(torch.allclose(t.prequery,h,atol=1e-7))
        self.assertTrue(torch.allclose(t.final_output,
                                       teacher.step(h,b.query)[:,0],atol=1e-7))

    def test_only_donor_final_source_label_is_changed(self):
        teacher=Teacher(901)
        factors=Factors(901)
        b=make_batch(90313,80,"train")
        donor,target,tags=change_last_source(b,91903)
        t=teacher.rollout(b)
        donor_state,delta,phi=recipient_donor_edit(teacher,factors,b,donor,t)
        self.assertTrue(torch.equal((t.prequery+delta)[:,0],t.prequery[:,0]))
        self.assertTrue(torch.allclose(
            donor_state,
            teacher.rollout(b,donor).prequery,atol=1e-6))
        self.assertTrue(torch.equal(donor[:,:,3:],b.events[:,:,3:]))
        self.assertEqual(tuple(phi.shape),(80,INNER))

    def test_policy_inputs_have_a_restricted_signature(self):
        parameters=list(inspect.signature(NonprivilegedPolicy.forward).parameters)
        self.assertEqual(parameters,[
            "self","state","candidate_delta","public_query","current_readout"])
        self.assertNotIn("teacher",parameters)
        self.assertNotIn("future_output",parameters)

    def test_policy_ignores_poisoned_future_metadata(self):
        policy=NonprivilegedPolicy(901)
        b=make_batch(90203,20,"train")
        state=torch.randn(20,HIDDEN,generator=torch.Generator().manual_seed(1))
        delta=torch.randn(20,HIDDEN,generator=torch.Generator().manual_seed(2))
        delta[:,0]=0
        base={
            "state":state,"candidate_delta":delta,
            "public_query":public_query(b),"current_readout":state[:,0:1],
        }
        a=predict_from_envelope(policy,base)
        poison={
            **base,
            "future_public_output":object(),
            "future_public_jacobian":object(),
            "donor_gold_label":object(),
            "teacher":object(),
            "oracle_correction":object(),
        }
        self.assertTrue(torch.equal(a,predict_from_envelope(policy,poison)))

    def test_correction_is_bounded_and_readout_preserving(self):
        policy=NonprivilegedPolicy(901)
        b=make_batch(90211,80,"train")
        g=torch.Generator().manual_seed(9187)
        state=torch.randn(80,HIDDEN,generator=g)
        delta=torch.randn(80,HIDDEN,generator=g)
        delta[:,0]=0
        corrected=policy.corrected_edit(state,delta,public_query(b),state[:,0:1])
        change=corrected-delta
        budget=torch.minimum(CORRECTION_FRACTION*delta.norm(dim=1),
                             torch.full((80,),CORRECTION_MAX_NORM))
        self.assertTrue(((change.norm(dim=1)-budget)<=1e-6).all().item())
        self.assertTrue(torch.equal((state+corrected)[:,0],state[:,0]))

    def test_gauge_invariant_full_physical_operator(self):
        f=Factors(901)
        shear=torch.eye(HIDDEN)
        shear[0,1]=.7
        a=f.weight()
        b=(f.V@shear)@torch.linalg.solve(shear,f.U)
        self.assertTrue(torch.allclose(a,b,atol=1e-6))

    def test_tiny_train_only_teacher_and_factors_run(self):
        teacher=train_teacher(901,updates=8,batch_size=32)
        factors=train_factors(901,teacher,updates=8,batch_size=32)
        self.assertTrue(torch.isfinite(teacher.C).all().item())
        self.assertTrue(torch.isfinite(factors.weight()).all().item())

    def test_tiny_train_only_policy_label_generation(self):
        teacher=train_teacher(901,updates=8,batch_size=16)
        factors=train_factors(901,teacher,updates=8,batch_size=16)
        policy,meta=train_policy(901,teacher,factors,updates=3,batch_size=16)
        self.assertEqual(meta["train_oracle_labels"],48)
        self.assertEqual(meta["training_family"],"train")
        self.assertFalse(meta["inference_uses_oracle"])
        self.assertTrue(all(torch.isfinite(p).all() for p in policy.parameters()))

    def test_reserved_fresh_roots_blocked(self):
        for root in (6101,6159,7101,7132):
            with self.assertRaises(ValueError):
                run_root(root,train_steps=1,factor_steps=1,policy_steps=1,n=8,shams=1)


if __name__=="__main__":
    unittest.main()
