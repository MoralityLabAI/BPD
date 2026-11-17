"""Engineering-only SF02 checks using excluded seed 901."""
import unittest
import torch

from mechinterp.bpd.theory.sf02_secret_circuit.experiment import (
    D, C, Teacher, Factors, sample, donor_pairs, factor_writes, patch,
    train_teacher, train_factors,
)


class SF02Engineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.teacher = train_teacher(901, steps=60)
        cls.f = train_factors(901, cls.teacher, steps=60)

    def test_teacher_is_weight_tied_and_differentiable(self):
        h, z, role, target = sample(90105, 24)
        h1, h2 = self.teacher.rollout(h)
        self.assertEqual(tuple(h1.shape), (24, D))
        self.assertEqual(tuple(h2.shape), (24, D))
        self.assertTrue(torch.isfinite(h1).all().item())
        self.assertTrue(torch.isfinite(h2).all().item())

    def test_pair_changes_only_one_secret(self):
        h, d, z, role, target = donor_pairs(90205, 120)
        self.assertTrue(torch.equal(h[:, 0], d[:, 0]))
        self.assertTrue(torch.equal(h[:, 3:], d[:, 3:]))
        different = (h[:, 1:3] - d[:, 1:3]).abs() > 0
        self.assertTrue((different.sum(dim=1) == 1).all().item())
        self.assertTrue((d[:, 1:3].abs() <= .8).all().item())

    def test_full_factor_sum_and_current_readout_preservation(self):
        h, d, *_ = donor_pairs(90211, 100)
        with torch.no_grad():
            h1 = self.teacher(h)
            writes = factor_writes(self.teacher, self.f, h, d)
            direct = (self.teacher.features(d) - self.teacher.features(h)) @ self.f.weight()
            self.assertTrue(torch.allclose(writes.sum(1), direct, atol=1e-6))
            q, delta = patch(self.teacher, h1, writes, [0, 1, 2])
            self.assertTrue(torch.equal((h1 + delta)[:, 0], h1[:, 0]))
            self.assertTrue(torch.isfinite(q).all().item())

    def test_full_operator_invariant_under_invertible_shear(self):
        R = torch.eye(C)
        R[0, 1] = .7
        W = self.f.weight().detach()
        U = torch.linalg.solve(R, self.f.U.detach())
        V = self.f.V.detach() @ R
        self.assertTrue(torch.allclose(V @ U, W, atol=1e-6))

    def test_secret_role_target_is_independently_defined(self):
        h, z, role, target = sample(90227, 90)
        for r, col in [(0, 0), (1, 1), (2, 2)]:
            self.assertTrue(torch.equal(target[role == r], z[role == r, col]))


if __name__ == "__main__":
    unittest.main()
