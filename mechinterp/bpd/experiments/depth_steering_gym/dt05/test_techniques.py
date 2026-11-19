"""Synthetic contracts; no development roots, checkpoints, or protected datasets."""
from __future__ import annotations

import inspect
import unittest

import torch
from torch import nn

from .techniques import (
    FirstValidStopper,
    HeadNullProjector,
    OnlineTopKLinear,
    principal_angles_deg,
    replace_online_factors,
    sudoku_valid_mask,
    transport_basis_jvp,
)


def solved(size: int) -> torch.Tensor:
    box = 2 if size == 4 else 3
    return torch.tensor([[(row * box + row // box + col) % size + 1
                          for row in range(size) for col in range(size)]])


class SudokuStopTests(unittest.TestCase):
    def test_both_board_sizes_and_public_clues(self):
        for size in (4, 9):
            board = solved(size)
            givens = torch.zeros_like(board)
            givens[:, ::3] = board[:, ::3]
            self.assertEqual(sudoku_valid_mask(board, givens).tolist(), [True])
            relabeled = board.remainder(size) + 1
            self.assertEqual(sudoku_valid_mask(relabeled, givens).tolist(), [False])

    def test_row_column_latin_square_fails_boxes(self):
        latin = torch.tensor([[(row + col) % 4 + 1 for row in range(4) for col in range(4)]])
        self.assertFalse(sudoku_valid_mask(latin, torch.zeros_like(latin)).item())

    def test_invalid_prediction_digit_and_partial_board_fail(self):
        candidates = solved(4).repeat(3, 1)
        candidates[0, 0], candidates[1, 0], candidates[2, 0] = 0, 5, -1
        self.assertEqual(sudoku_valid_mask(candidates, torch.zeros_like(candidates)).tolist(), [False] * 3)

    def test_malformed_and_conflicting_givens_rejected(self):
        for givens in (torch.zeros(1, 25, dtype=torch.long), torch.zeros(1, 16),
                       torch.full((1, 16), 5, dtype=torch.long), torch.ones(1, 16, dtype=torch.long)):
            with self.assertRaises(ValueError):
                FirstValidStopper(givens)

    def test_first_valid_survives_destructive_later_iteration(self):
        board = solved(4).repeat(2, 1)
        invalid = torch.ones_like(board)
        stopper = FirstValidStopper(torch.zeros_like(board))
        first = invalid.clone()
        first[0] = board[0]
        self.assertEqual(stopper.update(first, 2).tolist(), [True, False])
        second = invalid.clone()
        second[1] = board[1]
        stopper.update(second, 4)
        result = stopper.finish(invalid, 8)
        self.assertTrue(torch.equal(result["predictions"], board))
        self.assertEqual(result["selected_depths"].tolist(), [2, 4])
        self.assertEqual(result["valid_found"].tolist(), [True, True])

    def test_unresolved_case_retains_final_prediction(self):
        stopper = FirstValidStopper(torch.zeros((1, 16), dtype=torch.long))
        invalid = torch.ones((1, 16), dtype=torch.long)
        result = stopper.finish(invalid, 6)
        self.assertTrue(torch.equal(result["predictions"], invalid))
        self.assertEqual(result["selected_depths"].tolist(), [6])
        self.assertEqual(result["valid_found"].tolist(), [False])

    def test_depth_order_and_final_consistency_checked(self):
        stopper = FirstValidStopper(torch.zeros((1, 16), dtype=torch.long))
        candidate = solved(4)
        stopper.update(candidate, 3)
        with self.assertRaises(ValueError):
            stopper.update(candidate, 3)
        with self.assertRaises(ValueError):
            stopper.finish(candidate, 2)
        with self.assertRaises(ValueError):
            stopper.finish(torch.ones_like(candidate), 3)
        self.assertEqual(stopper.finish(candidate, 3)["selected_depths"].tolist(), [3])

    def test_stopping_api_has_no_gold_argument(self):
        for call in (FirstValidStopper, FirstValidStopper.update, FirstValidStopper.finish, sudoku_valid_mask):
            self.assertNotIn("gold", inspect.signature(call).parameters)


class FactorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(122)

    def test_full_rank_matches_dense_with_bias(self):
        linear = nn.Linear(7, 5, bias=True, dtype=torch.float64)
        x = torch.randn(2, 3, 7, dtype=torch.float64)
        factor = OnlineTopKLinear.from_linear(linear, 5, 5)
        torch.testing.assert_close(factor(x), linear(x), rtol=1e-10, atol=1e-10)
        self.assertEqual(set(factor.state_dict()), {"V", "U", "bias", "control_indices"})

    def test_online_routing_uses_current_input_each_call(self):
        layer = OnlineTopKLinear.from_factors(torch.eye(2), torch.eye(2), 1)
        torch.testing.assert_close(layer(torch.tensor([[4., 1.]])), torch.tensor([[4., 0.]]))
        torch.testing.assert_close(layer(torch.tensor([[1., 5.]])), torch.tensor([[0., 5.]]))

    def test_online_routing_includes_write_norm(self):
        layer = OnlineTopKLinear.from_factors(torch.eye(2), torch.diag(torch.tensor([1., 10.])), 1)
        torch.testing.assert_close(layer(torch.tensor([[4., 1.]])), torch.tensor([[0., 10.]]))

    def test_fixed_svd_matches_rank_truncated_matrix(self):
        linear = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.diag(torch.tensor([8., 5., 2., 1.])))
        factor = OnlineTopKLinear.from_linear(linear, 4, 2, mode="fixed")
        torch.testing.assert_close(factor(torch.ones(1, 4)), torch.tensor([[8., 5., 0., 0.]]))

    def test_control_modes_preserve_identical_factors(self):
        V, U = torch.randn(4, 4), torch.randn(4, 3)
        modules = [OnlineTopKLinear.from_factors(V, U, 2, mode=mode, seed=37)
                   for mode in ("online", "fixed", "random")]
        for module in modules:
            torch.testing.assert_close(module.V, V)
            torch.testing.assert_close(module.U, U)
        random2 = OnlineTopKLinear.from_factors(V, U, 2, mode="random", seed=37)
        self.assertTrue(torch.equal(modules[-1].control_indices, random2.control_indices))

    def test_factors_receive_gradients_without_dense_teacher(self):
        factor = OnlineTopKLinear.from_factors(torch.eye(3), torch.eye(3), 2)
        factor(torch.tensor([[2., 3., 0.5]])).sum().backward()
        self.assertIsNotNone(factor.V.grad)
        self.assertIsNotNone(factor.U.grad)
        self.assertFalse(hasattr(factor, "W_target"))

    def test_copy_existing_tied_factors_is_independent(self):
        from mechinterp.bpd.components import TiedComponentLinear
        tied = TiedComponentLinear(nn.Linear(4, 3), 3)
        factor = OnlineTopKLinear.from_tied(tied, 3)
        x = torch.randn(2, 4)
        expected = (x @ tied.V.detach()) @ tied.U.detach() + tied.bias
        torch.testing.assert_close(factor(x), expected)
        with torch.no_grad():
            tied.V.add_(20)
        torch.testing.assert_close(factor(x), expected)

    def test_replacements_validate_all_before_mutation(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        original = model[0]
        with self.assertRaises((TypeError, ValueError)):
            replace_online_factors(model, ["0", "1"], 2, 1)
        self.assertIs(model[0], original)
        replacements = replace_online_factors(model, ["0"], 2, 1, mode="random", seed=88)
        self.assertIs(model[0], replacements["0"])

    def test_invalid_factor_contracts_rejected(self):
        for top_k in (0, 4, True):
            with self.assertRaises(ValueError):
                OnlineTopKLinear(torch.eye(3), torch.eye(3), top_k)
        with self.assertRaises(ValueError):
            OnlineTopKLinear(torch.eye(3), torch.eye(2), 1)
        with self.assertRaises(ValueError):
            OnlineTopKLinear(torch.eye(3), torch.eye(3), 1, mode="teacher")


class GeometryTests(unittest.TestCase):
    def test_head_projection_matches_dense_reference_and_is_idempotent(self):
        gen = torch.Generator().manual_seed(709)
        weight = torch.randn(3, 13, generator=gen, dtype=torch.float64)
        vectors = torch.randn(5, 13, generator=gen, dtype=torch.float64)
        projector = HeadNullProjector(weight)
        expected = vectors @ (torch.eye(13, dtype=torch.float64) - torch.linalg.pinv(weight) @ weight)
        actual = projector(vectors)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(projector(actual), actual, atol=1e-10, rtol=1e-10)
        self.assertLess(float((actual @ weight.T).abs().max()), 1e-10)
        self.assertLessEqual(projector.row_basis.numel(), weight.numel())

    def test_dual_projection_locks_both_prefixes_and_preserves_low_board(self):
        weight = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        projector = HeadNullProjector(weight)
        original = torch.arange(30, dtype=torch.float32).reshape(1, -1)
        projected = projector.project_dual(original, 5, 1).reshape(1, 2, 5, 3)
        self.assertTrue(torch.equal(projected[:, :, :1], torch.zeros_like(projected[:, :, :1])))
        torch.testing.assert_close(projected[:, 1, 1:], original.reshape(1, 2, 5, 3)[:, 1, 1:])
        self.assertEqual(float((projected[:, 0] @ weight.T).abs().max()), 0.)

    def test_zero_and_full_rank_heads(self):
        vectors = torch.randn(2, 4)
        zero = HeadNullProjector(torch.zeros(2, 4))
        self.assertEqual(zero.rank, 0)
        torch.testing.assert_close(zero(vectors), vectors)
        full = HeadNullProjector(torch.eye(4))
        torch.testing.assert_close(full(vectors), torch.zeros_like(vectors))

    def test_transport_follows_rotation_without_dense_jacobian(self):
        matrix = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
        state = torch.tensor([2., 3.], dtype=torch.float64)
        basis = torch.tensor([[1.], [0.]], dtype=torch.float64)
        moved, info = transport_basis_jvp(lambda x: matrix @ x, state, basis)
        torch.testing.assert_close(moved @ moved.T, torch.diag(torch.tensor([0., 1.], dtype=torch.float64)))
        self.assertEqual(info["rank_lost"], 0)
        self.assertAlmostEqual(info["one_step_closure_defect"], 1.)
        self.assertFalse(info["dense_state_jacobian_allocated"])

    def test_transport_reports_rank_loss_without_padding(self):
        state = torch.ones(3, dtype=torch.float64)
        matrix = torch.diag(torch.tensor([2., 0., 0.], dtype=torch.float64))
        moved, info = transport_basis_jvp(lambda x: matrix @ x, state, torch.eye(3, dtype=torch.float64))
        self.assertEqual(moved.shape, (3, 1))
        self.assertEqual(info["rank_lost"], 2)

    def test_projection_can_remove_every_transport_direction(self):
        state = torch.ones(3)
        basis = torch.tensor([[1.], [0.], [0.]])
        projector = HeadNullProjector(torch.tensor([[1., 0., 0.]]))
        moved, info = transport_basis_jvp(lambda x: x, state, basis, project=projector)
        self.assertEqual(moved.shape, (3, 0))
        self.assertEqual(info["rank_lost"], 1)
        self.assertAlmostEqual(info["protected_projection_escape"], 1.)

    def test_nonlinear_jvp_matches_finite_difference_span(self):
        state = torch.tensor([0.3, -0.7, 1.2], dtype=torch.float64)
        basis = torch.tensor([[1.], [2.], [-1.]], dtype=torch.float64)
        basis = basis / basis.norm()
        fun = lambda x: torch.sin(x) + 0.2 * x.square()
        moved, _ = transport_basis_jvp(fun, state, basis)
        eps = 1e-5
        diff = (fun(state + eps * basis[:, 0]) - fun(state - eps * basis[:, 0])) / (2 * eps)
        diff = diff / diff.norm()
        torch.testing.assert_close(moved @ moved.T, diff[:, None] @ diff[None], atol=1e-8, rtol=1e-8)

    def test_invalid_tolerance_and_degenerate_input_basis_fail(self):
        with self.assertRaises(ValueError):
            HeadNullProjector(torch.eye(2), rtol=float("nan"))
        with self.assertRaises(ValueError):
            transport_basis_jvp(lambda x: x, torch.ones(2), torch.zeros(2, 1))
        with self.assertRaises(ValueError):
            transport_basis_jvp(lambda x: torch.ones(3), torch.ones(2), torch.eye(2))


class NativeInstrumentTests(unittest.TestCase):
    def test_native_dual_jvp_preserves_shape_and_projected_rank(self):
        from mechinterp.bpd.adapters.trm import build_mini_trm
        model = build_mini_trm(hidden_size=16, num_heads=2, l_layers=1,
                               h_cycles=1, l_cycles=1, puzzle_emb_len=1,
                               seq_len=16, seed=481).eval()
        tokens = torch.ones((1, 16), dtype=torch.long)
        embed = model.encode_inputs(tokens).detach()
        high, low = model.init_states(1)
        state = torch.cat((high.reshape(1, -1), low.reshape(1, -1)), dim=-1)
        projector = HeadNullProjector(model.lm_head.weight)
        project = lambda vectors: projector.project_dual(vectors, model.total_len, model.puzzle_emb_len)
        gen = torch.Generator().manual_seed(614)
        candidate = project(torch.randn(2, state.numel(), generator=gen)).T
        basis = torch.linalg.qr(candidate, mode="reduced").Q

        def step(packed):
            h, l = packed.reshape(1, 2, model.total_len, model.hidden_size).unbind(dim=1)
            h_next, l_next = model.act_step(h, l, embed)
            return torch.cat((h_next.reshape(1, -1), l_next.reshape(1, -1)), dim=-1)

        moved, metrics = transport_basis_jvp(step, state, basis, project=project)
        self.assertEqual(moved.shape, (state.numel(), 2))
        self.assertEqual(metrics["rank_lost"], 0)
        torch.testing.assert_close(project(moved.T), moved.T, atol=2e-6, rtol=2e-6)

    def test_native_factor_replacement_full_width_parity(self):
        from mechinterp.bpd.adapters.trm import build_mini_trm
        model = build_mini_trm(hidden_size=16, num_heads=2, l_layers=1,
                               h_cycles=1, l_cycles=1, puzzle_emb_len=1,
                               seq_len=16, seed=483).eval()
        tokens = torch.ones((2, 16), dtype=torch.long)
        with torch.no_grad():
            embed = model.encode_inputs(tokens)
            high, low = model.init_states(2)
            expected_h, expected_l = model.act_step(high, low, embed)
            replace_online_factors(model, ["layers.0.o_proj", "layers.0.down_proj"], 16, 16)
            actual_h, actual_l = model.act_step(high, low, embed)
        torch.testing.assert_close(actual_h, expected_h, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual_l, expected_l, atol=1e-5, rtol=1e-5)

    def test_vmap_and_loop_transport_match_and_principal_angles_work(self):
        gen = torch.Generator().manual_seed(912)
        a = torch.randn((12, 12), generator=gen, dtype=torch.float64)
        state = torch.randn(12, generator=gen, dtype=torch.float64)
        basis, _ = torch.linalg.qr(torch.randn((12, 4), generator=gen, dtype=torch.float64))
        step = lambda x: torch.tanh(a @ x) + 0.1 * x
        m_vmap, i_vmap = transport_basis_jvp(step, state, basis, chunk_size=2, mode="vmap")
        m_loop, i_loop = transport_basis_jvp(step, state, basis, mode="loop")
        self.assertEqual(i_vmap["jvp_execution_mode"], "vmap")
        self.assertEqual(i_loop["jvp_execution_mode"], "loop")
        torch.testing.assert_close(m_vmap @ m_vmap.T, m_loop @ m_loop.T, atol=1e-10, rtol=1e-10)
        angles = principal_angles_deg(m_vmap, m_loop)
        self.assertAlmostEqual(angles["subspace_overlap"], 1.0, places=9)
        self.assertLess(angles["max_angle_deg"], 1e-4)

    def test_online_topk_eval_cache_and_routing_telemetry(self):
        V = torch.eye(4)
        U = torch.diag(torch.tensor([5., 4., 2., 1.]))
        mod = OnlineTopKLinear.from_factors(V, U, 2, mode="online")
        mod.eval()
        mod.reset_routing_telemetry(True)
        with torch.no_grad():
            y = mod(torch.tensor([[1., 0.1, 2., 0.1], [0.1, 3., 0.1, 4.]]))
        tel = mod.routing_telemetry()
        self.assertTrue(tel["enabled"])
        self.assertAlmostEqual(tel["mean_active_components_per_token"], 2.0)
        self.assertGreaterEqual(tel["unique_active_components"], 2)
        self.assertEqual(set(mod.state_dict()), {"V", "U", "control_indices"})
        self.assertEqual(y.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
