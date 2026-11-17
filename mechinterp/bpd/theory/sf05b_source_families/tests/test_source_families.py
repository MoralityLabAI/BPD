"""Exact source-family and counterfactual tests, with no trained model."""
import itertools
import unittest
import torch

from mechinterp.bpd.theory.sf05b_source_families.source_families import (
    HELDOUT_TEMPLATES, _blind_mean, change_last_source, make_batch, target_from_records,
)


class SourceFamilies(unittest.TestCase):
    def test_train_heldout_history_lengths_are_disjoint(self):
        a = make_batch(901, 400, "train")
        b = make_batch(901, 400, "heldout")
        self.assertTrue(((a.lengths >= 2) & (a.lengths <= 4)).all())
        self.assertTrue(((b.lengths >= 5) & (b.lengths <= 6)).all())

    def test_last_source_flip_preserves_every_payload_and_query(self):
        a = make_batch(509, 40, "heldout")
        ev, y, tags = change_last_source(a, 909)
        self.assertTrue(torch.equal(a.event_mask, (ev[:, :, 4] > 0)))
        self.assertTrue(torch.equal(ev[:, :, 3:], a.events[:, :, 3:]))
        changed = (ev-a.events).abs().sum(2) > 1e-8
        self.assertTrue((changed.sum(1) == 1).all())
        self.assertTrue(torch.equal(
            y, target_from_records(tags, a.values, a.lengths,
                                   a.query_role, a.public_value)))

    def test_unauthorized_role_always_returns_public(self):
        a = make_batch(901, 400, "heldout")
        ev, y, _ = change_last_source(a, 991)
        mask = a.query_role == 3
        self.assertTrue(mask.any().item())
        self.assertTrue(torch.equal(y[mask], a.public_value[mask]))

    def test_train_blind_conditional_mean_matches_exact_tag_enumeration(self):
        a = make_batch(901, 50, "train")
        i = int((a.query_role != 3).nonzero(as_tuple=True)[0][0])
        L = int(a.lengths[i])
        outcomes = []
        for pattern in itertools.product(range(3), repeat=L):
            t = torch.tensor([list(pattern)])
            v = a.values[i:i+1, :L]
            outcomes.append(float(target_from_records(
                t, v, torch.tensor([L]), a.query_role[i:i+1],
                a.public_value[i:i+1])[0]))
        self.assertAlmostEqual(float(a.source_blind_conditional_mean[i]),
                               sum(outcomes)/len(outcomes), places=6)

    def test_heldout_blind_mean_matches_source_permutation_enumeration(self):
        a = make_batch(991, 50, "heldout")
        i = int((a.query_role != 3).nonzero(as_tuple=True)[0][0])
        L = int(a.lengths[i])
        outcomes = []
        for perm in itertools.permutations(range(3)):
            t = torch.tensor([[perm[j] for j in HELDOUT_TEMPLATES[L]]])
            outcomes.append(float(target_from_records(
                t, a.values[i:i+1, :L], torch.tensor([L]),
                a.query_role[i:i+1], a.public_value[i:i+1])[0]))
        self.assertAlmostEqual(float(a.source_blind_conditional_mean[i]),
                               sum(outcomes)/len(outcomes), places=6)

    def test_seeds_are_reproducible_and_independent(self):
        a = make_batch(150, 200, "heldout")
        b = make_batch(150, 200, "heldout")
        c = make_batch(151, 200, "heldout")
        self.assertTrue(torch.equal(a.events, b.events))
        self.assertFalse(torch.equal(a.events, c.events))

    def test_heldout_templates_repeat_all_three_sources(self):
        a = make_batch(1234, 300, "heldout")
        for i in range(300):
            ids = a.source_ids[i, :a.lengths[i]]
            self.assertEqual(len(set(map(int, ids))), 3)

    def test_latest_write_wins_exactly(self):
        a = make_batch(507, 1200, "train")
        for i in range(1200):
            role = int(a.query_role[i])
            seen = [j for j in range(int(a.lengths[i]))
                    if int(a.source_ids[i, j]) == role]
            expected = float(a.public_value[i]) if role == 3 else (
                float(a.values[i, seen[-1]]) if seen else 0.)
            self.assertAlmostEqual(float(a.target[i]), expected, places=6)

    def test_source_blind_prediction_cannot_see_changed_tag(self):
        a = make_batch(1509, 300, "heldout")
        donor_events, new_target, _ = change_last_source(a, 2309)
        # The source-blind estimate receives the values, length, role and
        # public scalar, all byte-for-byte identical in this counterfactual.
        self.assertTrue(torch.isfinite(a.source_blind_conditional_mean).all())
        self.assertTrue((a.target != new_target).any().item())
        blind_from_donor = _blind_mean(donor_events[:, :, 3], a.lengths,
                                       a.query_role, a.public_value, 'heldout')
        self.assertTrue(torch.equal(a.source_blind_conditional_mean, blind_from_donor))


if __name__ == "__main__":
    unittest.main()
