"""Post-pilot development-only tests for privileged finite public repair."""
import unittest
import torch

from mechinterp.bpd.theory.sf05_public_safe.finite_correct import (
    finite_public_repair,
)


class FiniteRepairTests(unittest.TestCase):
    def test_linear_public_output_is_corrected(self):
        h = torch.zeros(12, 3)
        delta = torch.tensor([[0., .1, .3]]).expand(12, -1).clone()
        repaired = finite_public_repair(h, delta, lambda x: x[:, 0]+x[:, 1])
        self.assertTrue(repaired.converged.all().item())
        self.assertLess(float(repaired.public_error_after.abs().max()), 1e-4)
        self.assertTrue(torch.equal((h+repaired.delta)[:, 0], h[:, 0]))

    def test_quadratic_finite_public_output_is_corrected(self):
        h = torch.zeros(12, 3)
        delta = torch.tensor([[0., .2, .3]]).expand(12, -1).clone()
        repaired = finite_public_repair(h, delta, lambda x: x[:, 0]+x[:, 1]**2)
        self.assertTrue(repaired.converged.all().item())
        self.assertLess(float(repaired.public_error_after.abs().max()), 1e-4)

    def test_flat_gradient_with_nonzero_residual_is_reported(self):
        # At edited x1=1, G=h0+(x1^2-1)^2 has derivative zero, but
        # the baseline at x1=0 is different. Newton cannot correct it.
        h = torch.zeros(6, 3)
        delta = torch.tensor([[0., 1., 0.]]).expand(6, -1).clone()
        repaired = finite_public_repair(
            h, delta, lambda x: x[:, 0]+(x[:, 1]**2-1)**2)
        self.assertFalse(repaired.converged.any().item())
        self.assertTrue(repaired.failed_zero_gradient.all().item())

    def test_zero_dose_and_zero_public_change(self):
        h = torch.zeros(4, 3)
        out = finite_public_repair(h, torch.zeros_like(h), lambda x: x[:, 0]+x[:, 1])
        self.assertTrue(out.converged.all().item())
        self.assertFalse(out.failed_zero_gradient.any().item())

    def test_mismatched_shapes_rejected(self):
        h = torch.zeros(2, 3)
        with self.assertRaises(ValueError):
            finite_public_repair(h, torch.zeros(2, 4), lambda x: x[:, 0])


if __name__ == "__main__":
    unittest.main()
