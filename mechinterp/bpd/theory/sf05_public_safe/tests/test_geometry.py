"""Exact SF05 geometry controls (no trained model or private data)."""
import unittest
import torch

from mechinterp.bpd.theory.sf05_public_safe.geometry import (
    public_safe_projection, random_public_safe_sham,
    signed_authorized_matched_sham, physical_output_subspace_write,
    linearized_authorized_capacity,
)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(601)
        self.delta = torch.randn(128, 20, generator=self.g)
        self.public = torch.randn(128, 20, generator=self.g)
        self.auth = torch.randn(128, 20, generator=self.g)

    def test_projection_two_constraints(self):
        z = public_safe_projection(self.delta, self.public)
        self.assertTrue(torch.equal(z.value[:, 0], torch.zeros(128)))
        self.assertLess(float(z.public_tangent.abs().max()), 1e-5)
        g = self.public.clone()
        g[:, 0] = 0
        self.assertLess(float((z.value*g).sum(1).abs().max()), 1e-5)

    def test_analytic_minimum_distance(self):
        d = torch.tensor([[9., 2., 3.]])
        g = torch.tensor([[7., 1., 0.]])
        out = public_safe_projection(d, g).value
        self.assertTrue(torch.allclose(out, torch.tensor([[0., 0., 3.]]), atol=1e-6))
        self.assertLess(float((out-d).norm()), float((torch.tensor([[0., 0., 4.]])-d).norm()))

    def test_zero_gradient_retains_dark_edit(self):
        d = torch.tensor([[2., 3., 4.]])
        r = public_safe_projection(d, torch.zeros_like(d))
        self.assertTrue(torch.equal(r.value, torch.tensor([[0., 3., 4.]])))
        self.assertTrue(r.zero_public_gradient.all().item())

    def test_nonlinear_public_remainder(self):
        # G(h)=h0+h1^2 near h=(0,0,0) has zero dark gradient.
        # A tangent-safe finite edit h1 += 0.2 still changes G by 0.04.
        d = torch.tensor([[0., .2, 0.]])
        g = torch.tensor([[1., 0., 0.]])
        v = public_safe_projection(d, g).value
        self.assertAlmostEqual(float(v[0, 1]**2), .04, places=6)

    def test_hidden_rotation_covariance(self):
        random = torch.randn(19, 19, generator=self.g)
        q, _ = torch.linalg.qr(random)
        basis = torch.eye(20)
        basis[1:, 1:] = q
        v = public_safe_projection(self.delta, self.public).value
        rotated = public_safe_projection(self.delta @ basis, self.public @ basis).value
        self.assertTrue(torch.allclose(rotated, v @ basis, atol=1e-5))

    def test_random_intersection_sham_matches_norm(self):
        d = public_safe_projection(self.delta, self.public).value
        z = random_public_safe_sham(d, self.public, self.g)
        self.assertLess(float((z.norm(dim=1)-d.norm(dim=1)).abs().max()), 1e-5)
        self.assertLess(float(public_safe_projection(z, self.public).removed.abs().max()), 1e-5)

    def test_signed_authorized_tangent_sham(self):
        d = public_safe_projection(self.delta, self.public).value
        z = signed_authorized_matched_sham(d, self.public, self.auth, self.g)
        self.assertLess(float((z.norm(dim=1)-d.norm(dim=1)).abs().max()), 1e-5)
        self.assertLess(float(public_safe_projection(z, self.public).removed.abs().max()), 1e-5)
        self.assertLess(float(((z-d)*self.auth).sum(1).abs().max()), 1e-5)

    def test_physical_svd_full_gauge_invariance(self):
        f = torch.randn(64, 40, generator=self.g)
        V = torch.randn(40, 20, generator=self.g)
        U = torch.randn(20, 20, generator=self.g)
        R = torch.eye(20)
        R[0, 1] = .3
        for rank in (8, 12, 16, 20):
            a, _ = physical_output_subspace_write(f, V@U, rank)
            b, _ = physical_output_subspace_write(f, (V@R)@torch.linalg.solve(R, U), rank)
            self.assertTrue(torch.allclose(a, b, atol=2e-3, rtol=1e-5))

    def test_authorized_linear_capacity_inequality(self):
        d = public_safe_projection(self.delta, self.public).value
        bound = linearized_authorized_capacity(self.auth, self.public, d)
        direct = (d*self.auth).sum(1).abs()
        self.assertTrue((direct <= bound+1e-5).all().item())


if __name__ == "__main__":
    unittest.main()
