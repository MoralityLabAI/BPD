"""Pre-data geometry checks on synthetic vectors. No scientific root scored."""
import unittest
import torch
from mechinterp.bpd.theory.sf03_tangent_null.experiment import draw_grad_matched
from mechinterp.bpd.theory.sf02_secret_circuit.experiment import D, init_rng


class TangentMatchedGeometry(unittest.TestCase):
    def setUp(self):
        self.rng = init_rng(1999)
        self.delta = torch.randn(64, D, generator=self.rng)
        self.delta[:, 0] = 0
        self.grad = torch.randn(64, D, generator=self.rng)

    def test_signed_sham_has_same_norm_and_signed_gradient(self):
        u, norm_error, abs_dot_error, signed_error = draw_grad_matched(
            self.delta, self.grad, self.rng)
        self.assertLess(norm_error, 1e-5)
        self.assertLess(abs_dot_error, 1e-5)
        self.assertLess(signed_error, 1e-5)
        self.assertTrue(torch.equal(u[:, 0], torch.zeros(64)))

    def test_unsigned_sham_preserves_only_absolute_gradient(self):
        u, norm_error, abs_error, signed_error = draw_grad_matched(
            self.delta, self.grad, self.rng, random_sign=True)
        self.assertLess(norm_error, 1e-5)
        self.assertLess(abs_error, 1e-5)
        self.assertIsNone(signed_error)
        self.assertTrue(torch.equal(u[:, 0], torch.zeros(64)))

    def test_zero_gradient_keeps_norm_and_darkness(self):
        g = torch.zeros_like(self.grad)
        u, ne, ae, se = draw_grad_matched(self.delta, g, self.rng)
        self.assertLess(ne, 1e-5)
        self.assertEqual(ae, 0.0)
        self.assertEqual(se, 0.0)
        self.assertTrue(torch.equal(u[:, 0], torch.zeros(64)))

    def test_constant_direction_zero_dose(self):
        z = torch.zeros_like(self.delta)
        u, ne, ae, se = draw_grad_matched(z, self.grad, self.rng)
        self.assertLess(ne, 1e-6)
        self.assertLess(ae, 1e-6)
        self.assertLess(se, 1e-6)


if __name__ == "__main__":
    unittest.main()
