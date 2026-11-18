"""Source-conditioned model admission checks: no fresh model roots."""
import unittest
from types import SimpleNamespace

import torch

from mechinterp.bpd.theory.sf05b_stage2_donors.donor_cohorts import make_cohort
from mechinterp.bpd.theory.sf05b_stage1.teacher import (
    HIDDEN, Teacher, Factors, recipient_donor_edit, public_query,
)
from mechinterp.bpd.theory.sf05b_stage1.policy import (
    NonprivilegedPolicy, predict_from_envelope,
)
from mechinterp.bpd.theory.sf05b_stage2_admission.evaluate import (
    EXCLUDED_ROOTS, COHORT_SEED_OFFSETS, eligibility_ladder, run,
)


class Stage2AdmissionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_source_strata_have_fixed_512_denominators(self):
        root=901
        for name,offset in COHORT_SEED_OFFSETS.items():
            c=make_cohort(root*430000+offset,512,name)
            self.assertEqual(c.accepted,512)
            self.assertEqual(c.recipient.family,"heldout")
            self.assertEqual(len(c.donor_target),512)
            if name=="authorized_high_contrast":
                self.assertEqual(c.attempted_candidates,2048)
                self.assertTrue(((c.recipient.target-c.donor_target).abs()>=.30).all())
            elif name=="public_U":
                self.assertTrue((c.recipient.query_role==3).all())
                self.assertTrue(torch.equal(c.recipient.target,c.donor_target))
                self.assertEqual(c.attempted_candidates,512)
            else:
                self.assertTrue((c.recipient.query_role!=3).all())

    def test_ladder_keeps_independent_and_sequential_denominators(self):
        batch=make_cohort(901*430000+530000,16,"authorized_unconditional").recipient
        n=len(batch.target)
        h=torch.zeros(n,HIDDEN)
        rec=SimpleNamespace(final_output=batch.target.clone(),prequery=h.clone())
        donor=SimpleNamespace(final_output=batch.target.clone(),prequery=h.clone())
        donor_target=batch.target.clone()
        # Correct all, same prequery, but future donor-recipient output gap zero.
        eligible,report=eligibility_ladder(batch,donor_target,rec,donor)
        self.assertEqual(report["generated"],16)
        self.assertEqual(report["both_clean_teacher_correct_count"],16)
        self.assertEqual(report["after_clean_then_boundary_count"],16)
        self.assertEqual(report["all_model_filters_count"],0)
        self.assertTrue(report["zero_eligible"])
        # Construct a separate fake model-visible donor difference on half the rows,
        # independently of source target, retaining all generated denominators.
        donor.final_output[:8]=rec.final_output[:8]+.3
        allowed,report=eligibility_ladder(batch,donor.final_output.clone(),rec,donor)
        self.assertEqual(report["all_model_filters_count"],8)
        self.assertEqual(report["generated"],16)
        self.assertEqual(int(allowed.sum()),8)

    def test_current_readout_zero_for_frozen_full_physical_write(self):
        c=make_cohort(901*430000+550000,48,"public_U")
        teacher=Teacher(901)
        factors=Factors(901)
        with torch.no_grad():
            rec=teacher.rollout(c.recipient)
            donorstate,edit,_=recipient_donor_edit(
                teacher,factors,c.recipient,c.donor_events,rec)
        self.assertTrue(torch.equal((rec.prequery+edit)[:,0],rec.prequery[:,0]))
        self.assertEqual(edit.shape,(48,HIDDEN))
        self.assertTrue(torch.isfinite(donorstate).all().item())

    def test_public_role_uses_same_independent_scalar(self):
        c=make_cohort(901*430000+550000,64,"public_U")
        q=public_query(c.recipient)
        self.assertTrue(torch.equal(q,c.recipient.query))
        self.assertTrue(torch.equal(c.blind_recipient,c.blind_donor))
        self.assertTrue(torch.equal(c.blind_recipient,c.recipient.public_value))

    def test_correction_policy_ignores_poisoned_inference_fields(self):
        policy=NonprivilegedPolicy(901)
        torch.manual_seed(7)
        x=torch.randn(16,HIDDEN)
        d=torch.randn(16,HIDDEN)
        d[:,0]=0
        p=public_query(make_cohort(901*430000+550000,16,"public_U").recipient)
        allowed={"state":x,"candidate_delta":d,"public_query":p,"current_readout":x[:,0:1]}
        baseline=predict_from_envelope(policy,allowed)
        altered=predict_from_envelope(policy,{**allowed,
            "teacher_forward":object(),"future_public_jacobian":object(),
            "future_public_logit":object(),"donor_gold_target":object(),
            "oracle_repair":object()})
        self.assertTrue(torch.equal(baseline,altered))
        self.assertTrue(torch.equal((x+baseline)[:,0],x[:,0]))

    def test_no_new_scientific_roots_admitted(self):
        with self.assertRaises(ValueError):
            run((6101,6159),n=16,shams=1)
        with self.assertRaises(ValueError):
            run((7101,7132),n=16,shams=1)
        with self.assertRaises(ValueError):
            run((901,),n=16,shams=1)

    def test_source_blind_recipient_donor_mse_are_distinct(self):
        c=make_cohort(901*430000+530000,128,"authorized_unconditional")
        self.assertTrue(torch.isfinite(c.blind_recipient).all())
        self.assertTrue(torch.isfinite(c.blind_donor).all())
        self.assertTrue(((c.blind_recipient-c.blind_donor).abs()>1e-5).any())


if __name__=="__main__":
    unittest.main()
