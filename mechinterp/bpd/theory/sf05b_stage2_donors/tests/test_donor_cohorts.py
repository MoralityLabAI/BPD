"""SF05B Stage 2 donor generator tests: source-only, no trained model."""
import unittest

import torch

from mechinterp.bpd.theory.sf05b_source_families.source_families import (
    HELDOUT_TEMPLATES, make_batch, target_from_records,
)
from mechinterp.bpd.theory.sf05b_stage2_donors.donor_cohorts import (
    HIGH_CONTRAST_GAP, DonorCohort, make_cohort, source_aware_blind_means,
)


class StageTwoDonorSourceChecks(unittest.TestCase):
    def test_change_is_only_last_source_name(self):
        for stratum in ("authorized_unconditional","authorized_high_contrast","public_U"):
            c=make_cohort(20401,128,stratum)
            b=c.recipient
            last=b.lengths-1
            row=torch.arange(len(b.target))
            changed=(b.events-c.donor_events).abs().sum(dim=-1)>0
            self.assertTrue(torch.equal(changed.sum(dim=1),torch.ones(128,dtype=torch.long)))
            self.assertTrue(changed[row,last].all())
            self.assertTrue(torch.equal(b.events[:,:,3:],c.donor_events[:,:,3:]))
            self.assertTrue(torch.equal(b.event_mask,c.donor_events[:,:,4].bool()))
            self.assertTrue(torch.equal(b.values,b.values.clone()))
            self.assertTrue((b.source_ids[row,last]!=c.donor_source_ids[row,last]).all())
            self.assertTrue(torch.equal(
                b.source_ids[:, :1], c.donor_source_ids[:, :1]))

    def test_authorized_query_role_matches_old_or_new_source(self):
        for stratum in ("authorized_unconditional","authorized_high_contrast"):
            c=make_cohort(20903,256,stratum)
            b=c.recipient
            row=torch.arange(256)
            old=b.source_ids[row,b.lengths-1]
            new=c.donor_source_ids[row,b.lengths-1]
            self.assertTrue(((b.query_role==old)|(b.query_role==new)).all())
            self.assertTrue((b.query_role!=3).all())
            self.assertTrue(torch.equal(b.query[:,5:9].argmax(dim=1),b.query_role))
            self.assertTrue(torch.equal(b.query[:,9],b.public_value))

    def test_high_contrast_accepts_first_n_with_explicit_denominator(self):
        c=make_cohort(20503,256,"authorized_high_contrast")
        self.assertEqual(c.attempted_candidates,1024)
        self.assertEqual(c.accepted,256)
        self.assertGreaterEqual(c.qualified_candidates,256)
        self.assertTrue(((c.recipient.target-c.donor_target).abs()>=HIGH_CONTRAST_GAP).all())
        self.assertEqual(c.label_change_fraction,1.)

    def test_unconditional_retains_unchanged_labels(self):
        c=make_cohort(20709,256,"authorized_unconditional")
        self.assertEqual(c.attempted_candidates,256)
        self.assertEqual(c.qualified_candidates,256)
        self.assertGreater(c.label_change_fraction,0.)
        self.assertLessEqual(c.label_change_fraction,1.)

    def test_public_cohort_is_separate_and_public_target_unchanged(self):
        c=make_cohort(20551,128,"public_U")
        self.assertTrue((c.recipient.query_role==3).all())
        self.assertTrue(torch.equal(c.recipient.target,c.recipient.public_value))
        self.assertTrue(torch.equal(c.donor_target,c.recipient.public_value))
        self.assertTrue(torch.equal(c.blind_recipient,c.recipient.public_value))
        self.assertTrue(torch.equal(c.blind_donor,c.recipient.public_value))
        self.assertEqual(c.label_change_fraction,0.)

    def test_donor_and_recipient_targets_recompute_exactly(self):
        c=make_cohort(20041,128,"authorized_high_contrast")
        b=c.recipient
        self.assertTrue(torch.equal(b.target,target_from_records(
            b.source_ids,b.values,b.lengths,b.query_role,b.public_value)))
        self.assertTrue(torch.equal(c.donor_target,target_from_records(
            c.donor_source_ids,b.values,b.lengths,b.query_role,b.public_value)))

    def test_exact_unconditional_blind_mean_closed_form(self):
        c=make_cohort(20661,160,"authorized_unconditional")
        b=c.recipient
        expected_rec,expected_don=[],[]
        for i in range(160):
            L=int(b.lengths[i])
            template=HELDOUT_TEMPLATES[L]
            old=template[-1]
            j_old=max(j for j,t in enumerate(template[:-1]) if t==old)
            others=[k for k in range(3) if k!=old]
            pos=[max(j for j,t in enumerate(template) if t==k) for k in others]
            v_last=b.values[i,L-1]
            expected_rec.append(.5*v_last+.25*(b.values[i,pos[0]]+b.values[i,pos[1]]))
            expected_don.append(.5*v_last+.5*b.values[i,j_old])
        self.assertTrue(torch.allclose(torch.stack(expected_rec),c.blind_recipient,atol=2e-6))
        self.assertTrue(torch.allclose(torch.stack(expected_don),c.blind_donor,atol=2e-6))
        self.assertTrue(((c.blind_recipient-c.blind_donor).abs()>1e-5).any())

    def test_high_contrast_conditional_blind_mean_closed_form(self):
        c=make_cohort(20819,160,"authorized_high_contrast")
        b=c.recipient
        expected_rec,expected_don=[],[]
        for i in range(160):
            L=int(b.lengths[i])
            template=HELDOUT_TEMPLATES[L]
            old=template[-1]
            j_old=max(j for j,t in enumerate(template[:-1]) if t==old)
            others=[k for k in range(3) if k!=old]
            pos=[max(j for j,t in enumerate(template) if t==k) for k in others]
            last=b.values[i,L-1]
            prev_old=b.values[i,j_old]
            other_a,other_b=b.values[i,pos[0]],b.values[i,pos[1]]
            wo=float((last-prev_old).abs()>=HIGH_CONTRAST_GAP)
            wa=float((last-other_a).abs()>=HIGH_CONTRAST_GAP)
            wb=float((last-other_b).abs()>=HIGH_CONTRAST_GAP)
            Z=2*wo+wa+wb
            self.assertGreater(Z,0)
            expected_rec.append((2*wo*last+wa*other_a+wb*other_b)/Z)
            expected_don.append((2*wo*prev_old+(wa+wb)*last)/Z)
        self.assertTrue(torch.allclose(torch.stack(expected_rec),c.blind_recipient,atol=2e-6))
        self.assertTrue(torch.allclose(torch.stack(expected_don),c.blind_donor,atol=2e-6))

    def test_reproducible_and_distinct_seeds(self):
        a=make_cohort(90103,80,"authorized_high_contrast")
        b=make_cohort(90103,80,"authorized_high_contrast")
        c=make_cohort(90104,80,"authorized_high_contrast")
        self.assertTrue(torch.equal(a.recipient.events,b.recipient.events))
        self.assertTrue(torch.equal(a.donor_events,b.donor_events))
        self.assertFalse(torch.equal(a.recipient.events,c.recipient.events))

    def test_stage_zero_heldout_length_semantics_preserved(self):
        c=make_cohort(20919,200,"authorized_unconditional")
        self.assertTrue(((c.recipient.lengths==5)|(c.recipient.lengths==6)).all())
        self.assertEqual(c.recipient.family,"heldout")
        for i in range(200):
            seq=c.recipient.source_ids[i,:c.recipient.lengths[i]]
            self.assertEqual(len(set(map(int,seq))),3)

    def test_source_only_generator_rejects_bad_stratum(self):
        with self.assertRaises(ValueError):
            make_cohort(901,16,"bogus")
        with self.assertRaises(ValueError):
            make_cohort(901,16,"public_U",family="train")


if __name__=="__main__":
    unittest.main()
