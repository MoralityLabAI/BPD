"""Pure CPU independent numerical checks for RT-BPD-02."""
import unittest

import numpy as np

from weighted_rank import (
    optimal_weighted_rank, paired_contrast_moment,
    truncated_svd, uncentered_second_moment, weighted_error,
)


class RTBPD02Tests(unittest.TestCase):
    def setUp(self):
        self.w = np.diag([4.0, 3.0, 2.0])
        self.c = np.diag([0.01, 1.0, 10.0])

    def test_locked_reached_state_fixture(self):
        ordinary = truncated_svd(self.w, 2)
        reached = optimal_weighted_rank(self.w, self.c, 2)
        self.assertAlmostEqual(weighted_error(self.w, ordinary, self.c), 40.0, places=9)
        self.assertAlmostEqual(reached.error, 0.16, places=9)
        self.assertAlmostEqual(reached.error, reached.tail_oracle, places=9)
        self.assertLess(reached.error, weighted_error(self.w, ordinary, self.c))
        self.assertLessEqual(np.linalg.matrix_rank(reached.operator, tol=1e-9), 2)

    def test_locked_future_reader_fixture(self):
        reader = np.array([[0.0, 1.0, 0.0]])
        reader_fit = optimal_weighted_rank(self.w, self.c, 1, reader)
        state_fit = optimal_weighted_rank(self.w, self.c, 1)
        self.assertLess(reader_fit.error, 1e-9)
        self.assertAlmostEqual(weighted_error(self.w, state_fit.operator, self.c, reader), 9.0, places=9)
        self.assertAlmostEqual(reader_fit.error, reader_fit.tail_oracle, places=9)

    def test_singular_psd_and_rank_deficient_reader(self):
        w = np.array([[3., 2., 1.], [1., 4., 7.], [2., 8., 6.], [5., 9., 3.]])
        c = np.diag([4., 1., 0.])
        reader = np.array([[1., 2., 0., 1.], [2., 4., 0., 2.], [0., 0., 0., 0.]])
        for r in (0, 1, 2, 3):
            fit = optimal_weighted_rank(w, c, r, reader)
            self.assertAlmostEqual(fit.error, fit.tail_oracle, places=8)
            self.assertLessEqual(np.linalg.matrix_rank(fit.operator, tol=1e-8), r)
            self.assertEqual(fit.input_support_rank, 2)
            self.assertEqual(fit.reader_rank, 1)

    def test_identity_reduces_to_svd(self):
        rng = np.random.default_rng(20261123)
        w = rng.normal(size=(5, 4))
        for r in range(5):
            fit = optimal_weighted_rank(w, np.eye(4), r)
            np.testing.assert_allclose(fit.operator, truncated_svd(w, r), atol=1e-10)
            self.assertAlmostEqual(fit.error, fit.tail_oracle, places=9)

    def test_orthogonal_input_gauge(self):
        rng = np.random.default_rng(2460)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        w2 = self.w @ q.T
        c2 = q @ self.c @ q.T
        a = np.array([[1., 1., 0.], [0., 0., 1.]])
        for r in (0, 1, 2):
            fit = optimal_weighted_rank(self.w, self.c, r, a)
            rotated = optimal_weighted_rank(w2, c2, r, a)
            np.testing.assert_allclose(rotated.operator, fit.operator @ q.T, atol=1e-9)
            self.assertAlmostEqual(rotated.error, fit.error, places=9)

    def test_zero_covariance_and_zero_reader(self):
        zero_c = optimal_weighted_rank(self.w, np.zeros((3, 3)), 1)
        np.testing.assert_array_equal(zero_c.operator, np.zeros((3, 3)))
        self.assertEqual(zero_c.error, 0.0)
        self.assertEqual(zero_c.input_support_rank, 0)
        zero_a = optimal_weighted_rank(self.w, self.c, 1, np.zeros((2, 3)))
        self.assertEqual(zero_a.error, 0.0)
        self.assertEqual(zero_a.reader_rank, 0)

    def test_paired_contrast_root_weighting(self):
        a = np.array([[1., 3.], [2., 5.]])
        b = np.array([[0., 1.], [2., 2.]])
        expected = np.array([[0.5, 1.0], [1.0, 6.5]])
        np.testing.assert_array_equal(paired_contrast_moment(a, b), expected)
        weighted = paired_contrast_moment(a, b, np.array([1., 3.]))
        np.testing.assert_allclose(weighted, np.array([[0.25, 0.5], [0.5, 7.75]]))
        np.testing.assert_array_equal(uncentered_second_moment(a - b), expected)

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, 'positive semidefinite'):
            optimal_weighted_rank(self.w, np.diag([-1., 1., 2.]), 1)
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            optimal_weighted_rank(self.w, np.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]]), 1)
        with self.assertRaises(ValueError):
            optimal_weighted_rank(self.w, np.eye(3), -1)
        with self.assertRaises(ValueError):
            optimal_weighted_rank(self.w, np.eye(3), True)
        with self.assertRaises(ValueError):
            optimal_weighted_rank(self.w, np.eye(3), 1, np.ones((1, 2)))
        with self.assertRaises(ValueError):
            uncentered_second_moment(np.array([[np.nan]]))
        with self.assertRaises(ValueError):
            paired_contrast_moment(np.ones((2, 2)), np.ones((3, 2)))
        with self.assertRaises(ValueError):
            uncentered_second_moment(np.eye(2), [0., 0.])


if __name__ == '__main__':
    unittest.main()
