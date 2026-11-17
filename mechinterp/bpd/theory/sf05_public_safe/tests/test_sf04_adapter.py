"""SF05 Stage-A adapter tests using an untrained SF04 teacher only."""
import unittest
import torch
from mechinterp.bpd.theory.sf04_source_binding.experiment import (
    H, Teacher, paired_source_swap,
)
from mechinterp.bpd.theory.sf05_public_safe.stage_a import (
    authorized_and_public_gradients, evaluate_root, test_arm,
)


class SF04AdapterTests(unittest.TestCase):
    def test_same_hidden_state_is_probed_with_two_query_roles(self):
        teacher = Teacher(901)
        (first, second, query, _, metadata), donor, _ = paired_source_swap(70111, 48)
        h0 = torch.zeros(48, H)
        with torch.no_grad():
            h1 = teacher.step(h0, first)
            h2 = teacher.step(h1, second)
        public, gp, ga = authorized_and_public_gradients(teacher, h2, query)
        self.assertTrue(torch.equal(public[:, 4], torch.zeros(48)))
        self.assertTrue(torch.equal(public[:, 5], torch.zeros(48)))
        self.assertTrue(torch.equal(public[:, 6], torch.ones(48)))
        self.assertTrue(torch.equal(public[:, 7:9], query[:, 7:9]))
        self.assertEqual(tuple(gp.shape), (48, H))
        self.assertTrue(torch.isfinite(gp).all())
        self.assertTrue(torch.isfinite(ga).all())
        self.assertGreater(float((gp-ga).norm(dim=-1).max()), 0.0)

    def test_tangent_safe_intervention_and_finite_public_observation(self):
        teacher = Teacher(901)
        (first, second, query, _, meta), donor, _ = paired_source_swap(70137, 42)
        with torch.no_grad():
            h1 = teacher.step(torch.zeros(42,H), first)
            h2 = teacher.step(h1, second)
            donor_visible = teacher.step(teacher.step(h1,donor),query)[:, 0]
        public, gp, ga = authorized_and_public_gradients(teacher,h2,query)
        g = torch.Generator().manual_seed(70231)
        delta = torch.randn(42,H,generator=g)*.1
        eligible = meta["role"] != 2
        record, safe = test_arm(teacher,h2,query,public,delta,gp,ga,donor_visible,eligible)
        self.assertLess(record["max_current_readout_error"], 1e-6)
        self.assertLess(record["max_public_tangent_error"], 1e-6)
        self.assertTrue(torch.isfinite(safe).all())

    def test_fresh_roots_are_refused_before_training(self):
        with self.assertRaises(ValueError):
            evaluate_root(1301,sham_arms=1,n=8)


if __name__ == "__main__":
    unittest.main()
