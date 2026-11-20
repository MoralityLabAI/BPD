"""Independent finite examples and invalid-input checks for spectral diagnostics."""

import json
import unittest

import numpy as np

from Spectralucative.math import (
    factor_closure_witness,
    matrix_inf_norm,
    ordered_product,
    trace_logdet_diagnostic,
    transport_diagnostic,
    walsh_diagnostic,
    walsh_inverse,
    walsh_transform,
)


class TraceLogTests(unittest.TestCase):
    def test_scalar_geometric_log_and_tail(self):
        result = trace_logdet_diagnostic([[0.75]], z=1.0, terms=10)
        expected = sum(0.75 ** k / k for k in range(1, 11))
        self.assertAlmostEqual(result["trace_log_partial"], expected)
        self.assertLessEqual(result["absolute_log_error"], result["analytic_tail_bound"])
        self.assertTrue(result["floating_point_check_passed"])
        self.assertFalse(result["roundoff_certified"])

    def test_signed_nondiagonal_matrix(self):
        a = np.array([[0.2, 0.3], [-0.4, 0.1]])
        result = trace_logdet_diagnostic(a, z=-0.7, terms=16)
        expected = -np.log(np.linalg.det(np.eye(2) + 0.7 * a))
        self.assertAlmostEqual(result["negative_logdet"], expected)
        self.assertTrue(result["floating_point_check_passed"])

    def test_nonnormal_stable_eigenvalues_do_not_grant_norm_certificate(self):
        a = np.array([[0.8, 10.0], [0.0, 0.8]])
        self.assertLess(max(abs(np.linalg.eigvals(a))), 1)
        result = trace_logdet_diagnostic(a, z=1.0)
        self.assertEqual(result["status"], "outside_norm_domain")
        self.assertIsNone(result["analytic_tail_bound"])
        self.assertIsNone(result["floating_point_check_passed"])
        self.assertTrue(trace_logdet_diagnostic(a, z=0.05)["floating_point_check_passed"])

    def test_strict_boundary_and_zero_cases(self):
        boundary = trace_logdet_diagnostic([[1.0]], z=1.0)
        self.assertEqual(boundary["status"], "outside_norm_domain")
        self.assertIsNone(boundary["negative_logdet"])
        for a, z in ((np.empty((0, 0)), 3.0), (np.eye(2), 0.0)):
            result = trace_logdet_diagnostic(a, z=z, terms=0)
            self.assertEqual(result["trace_log_partial"], 0.0)
            self.assertEqual(result["analytic_tail_bound"], 0.0)
            self.assertTrue(result["floating_point_check_passed"])
            json.dumps(result, allow_nan=False)

    def test_invalid_matrix_and_order(self):
        for a in (np.zeros((2, 3)), [[np.nan]], [[1j]]):
            with self.assertRaises(ValueError):
                trace_logdet_diagnostic(a, z=0.5)
        for terms in (-1, 2.5, True):
            with self.assertRaises(ValueError):
                trace_logdet_diagnostic([[1]], z=0.5, terms=terms)


class TransportTests(unittest.TestCase):
    def test_order_matters_even_with_nilpotent_steps(self):
        a = np.array([[0.0, 2.0], [0.0, 0.0]])
        b = a.T
        np.testing.assert_array_equal(ordered_product([a, b]), b @ a)
        self.assertFalse(np.array_equal(ordered_product([a, b]), ordered_product([b, a])))
        self.assertEqual(max(abs(np.linalg.eigvals(a))), 0)
        self.assertEqual(max(abs(np.linalg.eigvals(b))), 0)
        self.assertEqual(matrix_inf_norm(ordered_product([a, b])), 4.0)

    def test_changing_rectangular_maps(self):
        rng = np.random.default_rng(12019)
        a = [rng.normal(size=(3, 2)), rng.normal(size=(1, 3))]
        b = [rng.normal(size=(2, 4)), rng.normal(size=(3, 2))]
        l = [rng.normal(size=(4, 2)), rng.normal(size=(2, 3)), rng.normal(size=(3, 1))]
        result = transport_diagnostic(a, b, l)
        direct = b[1] @ b[0] @ l[0] - l[2] @ a[1] @ a[0]
        e0 = b[0] @ l[0] - l[1] @ a[0]
        e1 = b[1] @ l[1] - l[2] @ a[1]
        independent_sum = b[1] @ e0 + e1 @ a[0]
        np.testing.assert_allclose(result["final_difference"], direct)
        np.testing.assert_allclose(result["telescoping_reconstruction"], independent_sum)
        self.assertTrue(result["floating_point_check_passed"])
        self.assertLessEqual(result["final_error_norm"], result["final_bound"])

    def test_exact_quotient_and_zero_horizon(self):
        a = [np.diag([0.2, 0.6]), np.diag([0.7, -0.3])]
        l = np.array([[1.0, 0.0]])
        result = transport_diagnostic(a, [np.array([[0.2]]), np.array([[0.7]])], [l, l, l])
        self.assertEqual(result["final_error_norm"], 0)
        self.assertEqual(result["final_bound"], 0)
        zero = transport_diagnostic([], [], [np.empty((0, 2))])
        self.assertEqual(zero["final_error_norm"], 0)
        self.assertTrue(zero["floating_point_check_passed"])
        np.testing.assert_array_equal(ordered_product([], dimension=0), np.empty((0, 0)))

    def test_linear_growth_with_one_step_defect(self):
        result = transport_diagnostic([np.eye(1)] * 4, [np.eye(1)] * 4,
                                      [np.array([[float(t)]]) for t in range(5)])
        self.assertEqual(result["product_error_norms"], [0, 1, 2, 3, 4])
        self.assertEqual(result["norm_envelope"], [0, 1, 2, 3, 4])

    def test_invalid_shapes_and_nonfinite(self):
        with self.assertRaises(ValueError):
            transport_diagnostic([np.eye(2)], [np.eye(2)], [np.eye(2)])
        with self.assertRaises(ValueError):
            transport_diagnostic([np.eye(2)], [np.eye(2)], [np.eye(3), np.eye(3)])
        with self.assertRaises(ValueError):
            ordered_product([np.eye(2), np.ones((1, 3))])
        with self.assertRaises(ValueError):
            transport_diagnostic([], [], [[[np.inf]]])


class WalshTests(unittest.TestCase):
    def test_known_high_order_parity_and_uniform_residual(self):
        ids = np.arange(16)
        values = np.array([3 + 2 * (-1) ** ((int(x) & 0b1011).bit_count()) for x in ids])
        result = walsh_diagnostic(ids, values, max_degree=1)
        self.assertAlmostEqual(result["coefficients"][0], 3)
        self.assertAlmostEqual(result["coefficients"][0b1011], 2)
        self.assertEqual(result["energy_by_degree"], [9, 0, 0, 4, 0])
        self.assertEqual(result["residual_mean_squared_l2_response"], 4)
        self.assertEqual(result["high_degree_fraction_of_variance"], 1)
        self.assertTrue(result["floating_point_check_passed"])

    def test_transform_matches_explicit_character_sum(self):
        rng = np.random.default_rng(5001)
        values = rng.normal(size=(8, 3))
        direct = np.array([sum(values[x] * (-1) ** ((s & x).bit_count())
                               for x in range(8)) / 8 for s in range(8)])
        np.testing.assert_allclose(walsh_transform(values), direct, atol=1e-15)
        np.testing.assert_allclose(walsh_inverse(walsh_transform(values)), values, atol=1e-15)

    def test_shuffled_ids_and_vector_outputs(self):
        ids = [2, 0, 3, 1]
        values = np.array([[float(x), float(x * x)] for x in ids])
        result = walsh_diagnostic(ids, values, max_degree=2)
        self.assertEqual(result["mask_ids"], [0, 1, 2, 3])
        np.testing.assert_allclose(result["truncated_response"], [[0, 0], [1, 1], [2, 4], [3, 9]])
        self.assertEqual(result["discarded_coefficient_energy"], 0)
        self.assertTrue(result["floating_point_check_passed"])

    def test_zero_bit_and_constant_cube(self):
        for ids, values in (([0], [5]), (list(range(8)), np.full(8, 7.0))):
            result = walsh_diagnostic(ids, values)
            self.assertEqual(result["variance"], 0)
            self.assertEqual(result["high_degree_fraction_of_variance"], 0)
            self.assertEqual(result["residual_mean_squared_l2_response"], 0)
            json.dumps(result, allow_nan=False)

    def test_incomplete_duplicate_weighted_and_invalid_values(self):
        cases = [([0, 1, 2], [1, 2, 3]), ([0, 0], [1, 2]), ([0, 2], [1, 2]),
                 ([0.0, 1.0], [1, 2]), ([0, 1], [1, np.nan]), ([], [])]
        for ids, values in cases:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                walsh_diagnostic(ids, values)

    def test_overflow_cannot_be_serialized_as_a_success(self):
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(FloatingPointError):
            walsh_diagnostic([0, 1], [1e200, -1e200])


class FactorClosureTests(unittest.TestCase):
    def test_component_read_closure_with_retained_residual(self):
        rng = np.random.default_rng(901)
        v = rng.normal(size=(4, 2))
        u = rng.normal(size=(2, 4))
        residual = rng.normal(size=(4, 4)) * 0.01
        target = u.T @ v.T + residual
        result = factor_closure_witness(v, u, target, mask=[1, 0.25])
        np.testing.assert_allclose(result["predicted_closure_defect"], -v.T @ residual, atol=1e-14)
        self.assertTrue(result["floating_point_check_passed"])
        self.assertGreater(result["closure_defect_norm"], 0)

    def test_exact_low_rank_recurrence_and_empty_case(self):
        v = np.array([[1.0], [2.0]])
        u = np.array([[0.1, 0.2]])
        result = factor_closure_witness(v, u, u.T @ v.T)
        self.assertAlmostEqual(result["closure_defect_norm"], 0)
        empty = factor_closure_witness(np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)))
        self.assertEqual(empty["identity_residual_norm"], 0)
        self.assertTrue(empty["floating_point_check_passed"])

    def test_reject_nonsquare_recurrence_or_invalid_masks(self):
        with self.assertRaises(ValueError):
            factor_closure_witness(np.ones((3, 2)), np.ones((2, 4)), np.ones((4, 3)))
        with self.assertRaises(ValueError):
            factor_closure_witness(np.ones((3, 2)), np.ones((2, 3)), np.ones((3, 3)), mask=[1, -1])


if __name__ == "__main__":
    unittest.main()
