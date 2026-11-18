"""Excluded development engineering tests for the CQ-08 natural-state causal gym."""
import importlib.util
import unittest
from pathlib import Path
import torch

P = Path(__file__).resolve().parents[1] / 'experiment.py'
spec = importlib.util.spec_from_file_location('cq08_standalone', P)
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)


class CQ08Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        E.configure(); cls.seed = E.DEVELOPMENT[0]
        cls.teacher = E.train_teacher(cls.seed)
        cls.factors = E.train_factors(cls.seed, cls.teacher)
        cls.geometry = E.canonicalize(cls.factors)
        cls.selected, cls.losses = E.calibration_block(cls.seed, cls.teacher, cls.factors)
        with torch.no_grad():
            cls.matches = E.matched_nuisance(cls.seed, cls.teacher)
            cls.donor = E.score_donors(cls.seed, cls.teacher, cls.factors, cls.selected)

    def test_seed_populations_are_disjoint(self):
        self.assertEqual(len(E.DEVELOPMENT), 8); self.assertEqual(len(E.SHAM), 59)
        self.assertEqual(len(E.CONFIRMATION), 32)
        self.assertFalse(set(E.DEVELOPMENT)&set(E.SHAM))
        self.assertFalse(set(E.SHAM)&set(E.CONFIRMATION))

    def test_cues_and_independent_labels(self):
        a, b, lab, nuis = E.source(self.seed, 128, 430000)
        self.assertTrue(bool((lab[:,0] == torch.sign(a[:,0]*a[:,1])).all()))
        self.assertTrue(bool((lab[:,1] == torch.sign(a[:,0]*b[:,4])).all()))
        self.assertTrue(bool((b[:,:4] == 0).all()))
        self.assertTrue(bool((a[:,4] == 0).all()))

    def test_heldout_paired_z_proves_myopic_ceiling(self):
        with torch.no_grad():
            result = E.paired_z_task(self.seed, self.teacher, self.factors)
        self.assertTrue(result['first_cue_identical_z_pairs'])
        self.assertEqual(result['myopic_second_correct'], 128)
        self.assertEqual(result['symbolic_exact_ceiling'], 256)
        self.assertGreaterEqual(result['teacher_both_correct'], 245)

    def test_block_diagonal_learned_recurrence_and_rank(self):
        w = self.teacher.weight().detach()
        self.assertEqual(tuple(w.shape), (42, 36))
        self.assertLess(float(w[list(E.CORE_INPUTS),:][:,E.CORE:].abs().max()),1e-12)
        self.assertLess(float(w[list(E.NUISANCE_INPUTS),:][:,:E.CORE].abs().max()),1e-12)
        s = torch.linalg.svdvals(w)
        self.assertGreater(float(s[7]), 1e-4)
        self.assertLess(float(s[8]), 1e-4)

    def test_trained_full_bpd_reproduces_teacher_behavior(self):
        h0, h1, _, _ = E.source(self.seed, 96, 700000)
        with torch.no_grad():
            a, s1, s2 = self.teacher(h0,h1)
            a1 = E.factor_step(self.teacher,self.factors,torch.zeros(96,E.D),h0)
            a2 = E.factor_step(self.teacher,self.factors,a1,h1)
        self.assertLess(float((a1-s1).square().mean()),.008)
        self.assertLess(float((a2-s2).square().mean()),.008)
        self.assertLess(self.geometry['svd_full_reconstruction_error'],1e-4)

    def test_donor_flips_delayed_target_with_head_preserved(self):
        m = self.matches
        self.assertTrue(bool((m['x']*m['y'] != 0).all()))
        self.assertTrue(bool((m['head_donor']-m['head_rec']).abs().le(E.HEAD_TOL).all()))
        self.assertTrue(bool((m['donor_cue'][:,:2] == -m['rec_cue'][:,:2]).all()))
        self.assertTrue(bool((m['nuisance_cue'][:,:2] == m['rec_cue'][:,:2]).all()))

    def test_source_control_is_reachable_and_dose_matched(self):
        m = self.matches
        self.assertGreaterEqual(int(m['match'].sum()),103)
        with torch.no_grad():
            source_replay = self.teacher.step(torch.zeros(E.DONOR_CASES,E.D),m['nuisance_cue'])
        self.assertLess(float((source_replay-m['nuisance_state']).abs().max()),1e-6)
        self.assertEqual(m['max_state_box_violation'],0.)
        self.assertLessEqual(float((m['dose_donor'][m['match']]-m['dose_nuisance'][m['match']]).abs().max()),E.DOSE_TOL+1e-6)
        self.assertTrue(bool((m['nuisance_cue'][:,0:2]==m['rec_cue'][:,0:2]).all()))

    def test_frozen_block_selection_uses_only_calibration(self):
        self.assertEqual(len(self.selected),2)
        self.assertEqual(len(set(self.selected)),2)
        self.assertEqual(self.selected,sorted(range(E.COMPONENTS),key=lambda j:(-self.losses[j],j))[:2])

    def test_donor_intention_to_treat_and_controls(self):
        o=self.donor
        self.assertEqual(o['pairs'],128)
        self.assertEqual(len(o['pair_data']['match']),128)
        self.assertEqual(len(o['pair_data']['damage_selected']),128)
        self.assertLessEqual(o['damage_counts_itt']['selected_off'],o['eligible_match'])
        self.assertLessEqual(o['max_source_replay_error'],1e-6)
        self.assertEqual(o['max_state_box_violation'],0.)
        self.assertEqual(o['recipient_sham_donor_label_correct'],0)
        self.assertEqual(o['matched_nuisance_donor_label_correct'],0)

    def test_two_random_arms_have_exact_rank_cost(self):
        m1=E.random_masks(self.seed,64,850000)
        m2=E.random_masks(self.seed,64,850001)
        self.assertTrue(bool((m1.sum(1)==6).all()))
        self.assertTrue(bool((m2.sum(1)==6).all()))
        self.assertFalse(torch.equal(m1,m2))
        self.assertTrue(torch.equal(m1,E.random_masks(self.seed,64,850000)))

    def test_basis_gauge_parity_after_recanonicalization(self):
        with torch.no_grad():g=E.gauge_audit(self.seed,self.teacher,self.factors,self.selected)
        self.assertLess(g['max_full_operator_shear_error'],2e-6)
        if g['min_singular_gap']>=.001:
            self.assertLess(g['max_recanonical_masked_logit_drift'],1e-5)
        self.assertGreater(g['max_raw_masked_logit_drift'],1e-3)

    def test_confirmation_seal(self):
        self.assertFalse((E.ROOT/'CONFIRM_UNLOCK.json').exists())
        with self.assertRaises(RuntimeError):
            E.cohort('confirmation',seeds=[12001])

    def test_native_parity_if_repository_installed(self):
        try:from mechinterp.bpd.components import BPDModel
        except ImportError:self.skipTest('standalone CPU environment does not contain native BPDModel')
        n=E.NativeModel(self.teacher,self.factors)
        self.assertLess(E.native_parity(self.seed,self.teacher,self.factors,n),1e-5)

if __name__=='__main__':unittest.main()
