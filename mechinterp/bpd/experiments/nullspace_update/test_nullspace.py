"""Numerical tests for the protected-update solvers."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import unittest

import numpy as np

from . import nullspace as ns


def keep_matrix(rng, d, k, cond=1.0, rows=None, dtype=np.float64):
    """Rank-k X_keep (rows x d) with singular values spaced geometrically from 1 to 1/cond."""
    rows = rows or k + 4
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    B = Q[:, :k]
    s = np.geomspace(1.0, 1.0 / cond, k) if k > 1 else np.ones(k)
    A, _ = np.linalg.qr(rng.standard_normal((rows, k)))
    return ((A * s) @ B.T).astype(dtype), Q


def problem(rng, d=64, k=32, m=96, d_out=12, cond=1.0, dtype=np.float64):
    Xk, Q = keep_matrix(rng, d, k, cond, dtype=dtype)
    Xn = rng.standard_normal((m, d)).astype(dtype)
    R = (Xn @ rng.standard_normal((d, d_out)) / np.sqrt(d)).astype(dtype)
    return Xk, Xn, R


class StableNullspace(unittest.TestCase):
    def test_exact_float64(self):
        rng = np.random.default_rng(0)
        Xk, Xn, R = problem(rng)
        D = ns.nullspace_update(Xn, Xk, R)
        self.assertLess(np.linalg.norm(Xk @ D) / (np.linalg.norm(Xk) * np.linalg.norm(D)), 1e-13)
        self.assertEqual(D.dtype, np.float64)

    def test_exact_float32(self):
        rng = np.random.default_rng(1)
        Xk, Xn, R = problem(rng, dtype=np.float32)
        D = ns.nullspace_update(Xn, Xk, R)
        self.assertEqual(D.dtype, np.float32)
        self.assertLess(ns.keep_leak(Xk, D), 5e-6)

    def test_rank_truncation_stays_protected(self):
        rng = np.random.default_rng(2)
        Xk, Xn, R = problem(rng)
        D = ns.nullspace_update(Xn, Xk, R, rank=3)
        self.assertEqual(np.linalg.matrix_rank(D, tol=1e-9), 3)
        self.assertLess(ns.keep_leak(Xk, D), 1e-13)

    def test_rank_deficient_and_ill_conditioned(self):
        rng = np.random.default_rng(3)
        for cond in (1e4, 1e8, 1e11):
            Xk, Xn, R = problem(rng, cond=cond)
            D = ns.nullspace_update(Xn, Xk, R)
            self.assertLess(ns.keep_leak(Xk, D), 1e-10, cond)
        Xk, _, _ = problem(rng)
        N, r = ns.null_basis(Xk)
        self.assertEqual(r, 32)          # 36 rows, rank 32
        self.assertEqual(N.shape, (64, 32))

    def test_underdetermined_new_samples(self):
        rng = np.random.default_rng(4)
        Xk, Xn, R = problem(rng, m=10)
        D = ns.nullspace_update(Xn, Xk, R)
        self.assertLess(ns.keep_leak(Xk, D), 1e-13)
        self.assertLess(np.linalg.norm(Xn @ D - R) / np.linalg.norm(R), 1e-9)

    def test_empty_null_space(self):
        rng = np.random.default_rng(5)
        Xk, Xn, R = problem(rng, d=24, k=24, m=40)
        N, r = ns.null_basis(Xk)
        self.assertEqual((r, N.shape[1]), (24, 0))
        D = ns.nullspace_update(Xn, Xk, R)
        self.assertEqual(D.shape, (24, R.shape[1]))
        self.assertEqual(float(np.abs(D).max()), 0.0)


class LiteralFormInstability(unittest.TestCase):
    def test_literal_pinv_leaks_and_blows_up(self):
        rng = np.random.default_rng(6)
        d, k = 192, 96
        Xk = rng.standard_normal((k, d))
        Xn = rng.standard_normal((384, d))
        R = rng.standard_normal((384, 32))
        D_lit, U = ns.literal_projector_update(Xn, Xk, R)
        D_st = ns.nullspace_update(Xn, Xk, R)
        self.assertGreater(np.linalg.norm(U), 1e6 * np.linalg.norm(D_st))
        self.assertGreater(ns.keep_leak(Xk, D_lit), 1e-4)
        self.assertLess(ns.keep_leak(Xk, D_st), 1e-13)


class OtherSolvers(unittest.TestCase):
    def test_soft_protection_monotone_in_lambda(self):
        rng = np.random.default_rng(7)
        Xk, Xn, R = problem(rng)
        leaks = [ns.keep_leak(Xk, ns.soft_protected_ridge(Xn, Xk, R, lam=l)) for l in (0, 1, 100, 1e4)]
        self.assertTrue(all(a > b for a, b in zip(leaks, leaks[1:])), leaks)

    def test_ordinary_ridge_primal_dual_agree(self):
        rng = np.random.default_rng(8)
        X, R = rng.standard_normal((20, 30)), rng.standard_normal((20, 4))
        D = ns.ordinary_ridge(X, R, mu=0.5)
        ref = np.linalg.solve(X.T @ X + 0.5 * np.eye(30), X.T @ R)
        self.assertLess(np.linalg.norm(D - ref), 1e-10)

    def test_tsvd_rank_and_random_match(self):
        rng = np.random.default_rng(9)
        Xk, Xn, R = problem(rng)
        D = ns.truncated_svd_update(Xn, R, 4)
        self.assertEqual(np.linalg.matrix_rank(D, tol=1e-9), 4)
        Z = ns.random_matched(D.shape, 4, np.linalg.norm(D), rng)
        self.assertEqual(np.linalg.matrix_rank(Z, tol=1e-9), 4)
        self.assertAlmostEqual(np.linalg.norm(Z), np.linalg.norm(D), places=10)

    def test_approx_protection_trades_leak_for_fit(self):
        rng = np.random.default_rng(10)
        d = 64
        Xk, Q = keep_matrix(rng, d, 40, cond=1e3)
        Xn = rng.standard_normal((200, d))
        R = Xn @ rng.standard_normal((d, 8)) / 8
        Dfull = ns.nullspace_update(Xn, Xk, R)
        Dapx, info = ns.approx_protected_update(Xn, Xk, R, energy=0.9)
        self.assertLess(info["k"], 40)
        self.assertGreater(info["leak"], ns.keep_leak(Xk, Dfull))
        self.assertLess(np.linalg.norm(Xn @ Dapx - R), np.linalg.norm(Xn @ Dfull - R))

    def test_adaptive_allocation_beats_uniform_and_respects_budget(self):
        spectra = [np.array([10, 9, 8, 1.0]), np.array([1, .5, .1, .1]), np.array([2, 1, .5, .1])]
        a = ns.adaptive_rank_allocation(spectra, 6)
        u = ns.uniform_rank_allocation(spectra, 6)
        self.assertEqual(sum(a), 6)
        cap = lambda al: sum(float((np.sort(s)[::-1][:r] ** 2).sum()) for s, r in zip(spectra, al))
        self.assertGreaterEqual(cap(a), cap(u))
        self.assertEqual(a[0], 4)


if __name__ == "__main__":
    unittest.main()
