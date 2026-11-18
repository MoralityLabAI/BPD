"""Source-only tests for EFD-01; never fetch teacher weights or reserved samples."""
import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from energy_fibers import (energy_fiber, delayed_rank, split_native_update,
                           skew_representer, tangent_shams, row_nullspace)


class EnergyFiberTests(unittest.TestCase):
    def test_exact_three_constraint_kernel_and_delayed_capacity(self):
        a = np.eye(5)[:2]
        g = np.array([0., 0., 1., 0., 0.])
        f = energy_fiber(a, g)
        self.assertEqual((f.current_rank, f.joint_rank, f.dimension), (2, 3, 2))
        self.assertEqual(f.lower_bound, 2)
        self.assertLess(f.constraint_residual, 1e-11)
        self.assertLess(f.projector_error, 1e-11)
        b = np.array([[0., 0., 0., 1., 0.],
                      [0., 0., 0., 2., 0.]])
        self.assertEqual(delayed_rank(b, f)["rank"], 1)
        self.assertEqual(delayed_rank(a, f)["rank"], 0)

    def test_orthogonal_coordinate_transport(self):
        a = np.array([[1., 2., 0., 0., 0.], [0., 1., 1., 0., 0.]])
        g = np.array([0., 0., 1., 1., 0.])
        b = np.array([[0., 0., 0., 0., 1.], [1., 0., 1., 0., 0.]])
        f = energy_fiber(a, g)
        q, _ = np.linalg.qr(np.random.default_rng(123).normal(size=(5, 5)))
        transported = energy_fiber(a @ q, q.T @ g)
        self.assertEqual(f.dimension, transported.dimension)
        self.assertEqual(delayed_rank(b, f)["rank"],
                         delayed_rank(b @ q, transported)["rank"])
        np.testing.assert_allclose(f.projector, q @ transported.projector @ q.T,
                                   rtol=0, atol=1e-11)

    def test_circulating_algebra_and_reconstruction(self):
        g = np.array([1., 2., 3., 4.])
        v = np.array([3., -2., 1., 4.])
        parts = split_native_update(v, g)
        self.assertTrue(parts["identified"])
        self.assertAlmostEqual(parts["orthogonality"], 0., places=11)
        np.testing.assert_allclose(parts["descent"] + parts["tangent"], v)
        skew = skew_representer(parts["tangent"], g)
        np.testing.assert_allclose(skew, -skew.T, atol=1e-13)
        np.testing.assert_allclose(skew @ g, parts["tangent"], atol=1e-12)

    def test_gradient_zero_not_mechanistically_identified(self):
        g = np.zeros(4)
        v = np.array([1., -1., 0., 2.])
        parts = split_native_update(v, g)
        self.assertFalse(parts["identified"])
        self.assertIsNone(parts["alpha"])
        np.testing.assert_allclose(parts["tangent"], v)
        with self.assertRaises(ValueError):
            skew_representer(v, g)
        f = energy_fiber(np.eye(4)[:2], g)
        self.assertEqual(f.dimension, 2)

    def test_svd_rank_threshold_exposed(self):
        a = np.diag([1., 1.e-10, 0.])
        basis, singular, rank, threshold = row_nullspace(a)
        self.assertEqual(rank, 1)
        self.assertEqual(basis.shape, (3, 2))
        self.assertEqual(threshold, 1.e-9)
        self.assertAlmostEqual(float(singular[1]), 1.e-10, places=16)

    def test_shams_match_physical_dose_and_both_linear_nulls(self):
        a = np.eye(7)[:3]
        g = np.eye(7)[3]
        f = energy_fiber(a, g)
        shams = tangent_shams(f, radius=0.25, count=8, seed=912)
        self.assertEqual(shams.shape, (8, 7))
        np.testing.assert_allclose(np.linalg.norm(shams, axis=1),
                                   0.25 * np.ones(8), rtol=0, atol=1e-14)
        np.testing.assert_allclose(shams @ a.T, 0., atol=1e-14)
        np.testing.assert_allclose(shams @ g, 0., atol=1e-14)
        np.testing.assert_array_equal(shams, tangent_shams(f, radius=.25,
                                                           count=8, seed=912))

    def test_finite_curvature_can_reverse_first_order_descent(self):
        # E(h)=||h||²/2; at h=(1,0), v=(-.1,5) has g.v<0.
        # Yet full-step E(h+v)>E(h). This is an exact counterexample.
        h = np.array([1., 0.])
        v = np.array([-.1, 5.])
        g = h.copy()
        self.assertLess(float(g @ v), 0.)
        e = lambda s: float(s @ s / 2.)
        self.assertGreater(e(h + v) - e(h), 0.)
        curvature = e(h + v) - e(h) - float(g @ v)
        self.assertAlmostEqual(curvature, float(v @ v) / 2., places=12)

    def test_nullspace_with_no_readout_constraints(self):
        a = np.empty((0, 3))
        f = energy_fiber(a, np.array([0., 0., 1.]))
        self.assertEqual(f.dimension, 2)
        self.assertEqual(f.lower_bound, 2)
        self.assertEqual(delayed_rank(np.eye(3), f)["rank"], 2)

    def test_validate_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            energy_fiber(np.eye(3), np.array([np.nan, 0., 0.]))
        with self.assertRaises(ValueError):
            energy_fiber(np.eye(2), np.zeros(3))
        with self.assertRaises(ValueError):
            row_nullspace(np.array([[np.inf]]))
        with self.assertRaises(ValueError):
            tangent_shams(energy_fiber(np.eye(2), np.zeros(2)), radius=1.,
                          count=8, seed=3)
        with self.assertRaises(ValueError):
            tangent_shams(energy_fiber(np.zeros((0, 3)), np.zeros(3)),
                          radius=0., seed=3)


if __name__ == "__main__":
    unittest.main()
