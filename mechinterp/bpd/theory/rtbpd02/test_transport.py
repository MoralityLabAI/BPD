"""Independent CPU cases for support, coordinate and causal-transport contracts."""
import unittest

import numpy as np

from transport import (
    delayed_dark_rank, direction, disagreement, dominance_constant,
    fixed_future_reader, signed_paths, subspace_angles,
)
from weighted_rank import optimal_sample_rank, optimal_weighted_rank, uncentered_second_moment, weighted_error


class SolverAudit(unittest.TestCase):
    def test_reader_cutoff_is_applied_to_the_target_and_the_lift(self):
        w = np.diag([2., 1e10])
        a = np.diag([1., 1e-9])
        fit = optimal_weighted_rank(w, np.eye(2), 1, a, rtol=1e-6)
        np.testing.assert_allclose(fit.operator, np.diag([2., 0.]))
        self.assertEqual(fit.reader_rank, 1)
        self.assertEqual(fit.error, fit.tail_oracle)
        self.assertEqual(fit.error, 0.)
        self.assertAlmostEqual(fit.full_error, 100.)
        self.assertAlmostEqual(weighted_error(w, fit.operator, np.eye(2), a, full=True), 100.)
        exact = optimal_weighted_rank(w, np.eye(2), 1, a, rtol=1e-12)
        self.assertAlmostEqual(exact.error, 4.)
        self.assertEqual(exact.reader_rank, 2)

    def test_covariance_cutoff_uses_eigenvalues_and_preserves_full_loss(self):
        w = np.diag([2., 1e8])
        c = np.diag([1., 1e-10])
        fit = optimal_weighted_rank(w, c, 1, rtol=1e-8)
        self.assertEqual(fit.input_support_rank, 1)
        self.assertAlmostEqual(fit.error, 0.)
        self.assertAlmostEqual(fit.full_error, 1e6)
        sample = optimal_sample_rank(w, np.sqrt(2) * np.diag([1., 1e-5]), 1, rtol=1e-8)
        np.testing.assert_allclose(sample.operator, fit.operator)
        self.assertAlmostEqual(sample.full_error, fit.full_error)

    def test_thin_sample_matches_covariance_with_contrast_and_weights(self):
        rng = np.random.default_rng(219)
        a = rng.normal(size=(7, 9))
        b = rng.normal(size=a.shape)
        w = rng.normal(size=(5, a.shape[1]))
        reader = rng.normal(size=(2, w.shape[0]))
        weights = np.array([0., 2., 8., 9., 1., 4., 1.])
        rows = a - b
        moment = uncentered_second_moment(rows, weights)
        sample = optimal_sample_rank(w, rows, 1, reader, weights=weights)
        full = optimal_weighted_rank(w, moment, 1, reader)
        np.testing.assert_allclose(sample.operator, full.operator, atol=1e-11)
        self.assertAlmostEqual(sample.error, full.tail_oracle)
        self.assertEqual(sample.input_support_rank, 6)
        huge = uncentered_second_moment(np.eye(3), np.full(3, 1e308))
        np.testing.assert_allclose(huge, np.eye(3) / 3)

    def test_numerical_validation_relative_to_small_scales_and_empty_support(self):
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            optimal_weighted_rank(np.eye(2), np.diag([-1e-30, 2e-30]), 1)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            optimal_weighted_rank(np.eye(2), np.array([[1e-30, 1e-30], [0, 1e-30]]), 1)
        for w, c, a in (
            (np.zeros((3, 0)), np.zeros((0, 0)), None),
            (np.zeros((0, 2)), np.eye(2), None),
            (np.eye(2), np.eye(2), np.zeros((0, 2))),
        ):
            fit = optimal_weighted_rank(w, c, 7, a)
            self.assertEqual(fit.operator.shape, w.shape)
            self.assertEqual((fit.error, fit.full_error, fit.tail_oracle), (0., 0., 0.))

    def test_nonsingular_nonorthogonal_gauges_preserve_the_exact_objective(self):
        w = np.array([[1., -3.], [2., 1.]])
        a = np.array([[1., 0.7], [-0.3, 1.]])
        sigma = np.diag([1., .6])
        g = np.array([[3., 1.], [0., .4]])
        h = np.array([[.8, .1], [.5, 1.3]])
        base = optimal_weighted_rank(w, sigma, 1, a)
        changed = optimal_weighted_rank(
            h @ w @ np.linalg.inv(g), g @ sigma @ g.T, 1, a @ np.linalg.inv(h),
        )
        np.testing.assert_allclose(changed.operator, h @ base.operator @ np.linalg.inv(g), atol=1e-12)
        self.assertAlmostEqual(base.error, changed.error)

    def test_singular_gauge_covariance_applies_to_solution_sets(self):
        w = np.eye(2)
        c = np.diag([1., 0.])
        g = np.array([[1., 1.], [0., 1.]])
        base = optimal_weighted_rank(w, c, 1)
        changed_w = w @ np.linalg.inv(g)
        changed_c = g @ c @ g.T
        changed = optimal_weighted_rank(changed_w, changed_c, 1)
        transported = base.operator @ np.linalg.inv(g)
        self.assertAlmostEqual(weighted_error(changed_w, transported, changed_c), changed.error)
        np.testing.assert_allclose(transported[:, 0], changed.operator[:, 0])
        self.assertGreater(np.linalg.norm(transported - changed.operator), .99)
        self.assertLess(np.linalg.norm(changed.operator), np.linalg.norm(transported))

    def test_tied_singular_optima_and_instability_at_a_cutoff(self):
        c = np.eye(2)
        plus = optimal_weighted_rank(np.diag([1 + 1e-9, 1.]), c, 1).operator
        minus = optimal_weighted_rank(np.diag([1., 1 + 1e-9]), c, 1).operator
        self.assertAlmostEqual(weighted_error(c, np.diag([1., 0.]), c), 1.)
        self.assertAlmostEqual(weighted_error(c, np.diag([0., 1.]), c), 1.)
        self.assertAlmostEqual(subspace_angles(plus, minus)[0], np.pi / 2)


class TransportAudit(unittest.TestCase):
    def test_fixed_and_changed_reader_state_accounting(self):
        a = np.array([[1., 2.], [3., 1.]])
        b = np.array([[3., 1.], [2., 3.]])
        xa = np.array([1., -1.])
        xb = np.array([-2., 3.])
        ca, cb = np.array([[2., 3.]]), np.array([[3., 1.]])
        plain = disagreement(a, b, xa, xb)
        measured = disagreement(a, b, xa, xb, ca, cb)
        np.testing.assert_allclose(plain.total, b @ xb - a @ xa)
        np.testing.assert_allclose(measured.total, cb @ b @ xb - ca @ a @ xa)
        energies = measured.energies()
        self.assertAlmostEqual(energies.pop("total"), sum(energies.values()))
        cancel = disagreement([[1.]], [[2.]], [2.], [1.]).energies()
        self.assertEqual((cancel["operator"], cancel["state"], cancel["total"]), (2.25, 2.25, 0.))
        self.assertEqual(cancel["operator_state_cross"], -4.5)

    def test_delayed_reader_rank_identity_and_no_transport_counterexample(self):
        b = np.eye(2)
        j = np.array([[0., 1.], [0., 0.]])
        now = np.array([[1., 0.]])
        future = fixed_future_reader(b, (j,), (np.zeros((0, 2)), now), (0., 4.))
        np.testing.assert_array_equal(future, [[0., 2.]])
        d = np.array([[1., 0., 1.], [0., 1., 1.]])
        rank_gap = np.linalg.matrix_rank(np.vstack((now, future)) @ d) - np.linalg.matrix_rank(now @ d)
        self.assertEqual(delayed_dark_rank(now, future, d), rank_gap)
        self.assertEqual(delayed_dark_rank(now, np.zeros((1, 2)), d), 0)
        self.assertEqual(delayed_dark_rank(np.zeros((0, 2)), future, d), 1)
        self.assertEqual(delayed_dark_rank(now, future, np.zeros((2, 0))), 0)

    def test_gain_sign_and_equal_dose_witness(self):
        reference = np.array([2., 0.])
        self.assertEqual(direction([1., 0.], reference)["gain"], .5)
        self.assertEqual(direction([-1., 0.], reference)["gain"], -.5)
        self.assertEqual(direction([0., 0.], reference)["status"], "zero_observed")
        self.assertIsNone(direction([1., 0.], [0., 0.])["cosine"])

    def test_transport_bound_and_two_missing_support_witnesses(self):
        source = np.diag([2., 1.])
        target = np.diag([1., 3.])
        ap, aq = np.diag([1., 1.]), np.diag([2., .5])
        m = dominance_constant(source, target)
        c = dominance_constant(ap.T @ ap, aq.T @ aq)
        self.assertAlmostEqual(m, 3.)
        self.assertAlmostEqual(c, 4.)
        w, k = np.eye(2), np.diag([1., 0.])
        self.assertLessEqual(weighted_error(w, k, target, aq), m * c * weighted_error(w, k, source, ap))
        first, second = np.diag([1., 0.]), np.diag([0., 1.])
        self.assertTrue(np.isinf(dominance_constant(first, second)))
        for cov_p, cov_q, reader_p, reader_q in (
            (first, second, None, None),
            (source, source, first, second),
        ):
            self.assertAlmostEqual(weighted_error(w, k, cov_p, reader_p), 0.)
            self.assertAlmostEqual(weighted_error(w, k, cov_q, reader_q), 1.)
        self.assertAlmostEqual(dominance_constant(first, 5 * first), 5.)

    def test_signed_cancellation_and_finite_interaction(self):
        graph = np.zeros((4, 4))
        graph[1, 0] = graph[2, 0] = graph[3, 1] = 1.
        graph[3, 2] = -1.
        self.assertEqual(signed_paths(graph, 0, 3), (0., 2.))
        self.assertEqual(signed_paths(graph, 0, 0), (1., 1.))
        graph[0, 3] = 1.
        with self.assertRaisesRegex(ValueError, "triangular"):
            signed_paths(graph, 0, 3)
        f = lambda u, v: u * v
        self.assertEqual(f(1, 0) - f(0, 0), 0.)
        self.assertEqual(f(0, 1) - f(0, 0), 0.)
        self.assertEqual(f(1, 1) - f(1, 0) - f(0, 1) + f(0, 0), 1.)


if __name__ == "__main__":
    unittest.main()
