"""Policy-work and artifact regressions using deterministic toy trajectories."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from .evaluation import evaluate_policy, rollout_first_valid, transport_diagnostic
from .resources import BudgetExceeded, MemoryBudget


SOLUTION = torch.tensor([[1, 2, 3, 4, 3, 4, 1, 2, 2, 1, 4, 3, 4, 3, 2, 1]])


class ToyTrajectory(nn.Module):
    """Public clue patterns identify synthetic cases with predetermined rollouts."""
    seq_len = 16

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.batch_counts = []
        plan = torch.ones((3, 4, 16), dtype=torch.long)
        plan[0, 1] = SOLUTION[0]
        plan[1, 3] = SOLUTION[0]
        # Third case oscillates between two invalid boards.
        plan[2, 1] = 2
        plan[2, 3] = 2
        self.register_buffer("plan", plan)

    def encode_inputs(self, tokens):
        ids = torch.where(tokens[:, 0] == 2, 0, torch.where(tokens[:, 1] == 3, 1, 2))
        return ids[:, None, None]

    def init_states(self, batch_size):
        state = torch.zeros((batch_size, 1, 2), device=self.anchor.device)
        return state, state.clone()

    def act_step(self, high, low, embed):
        self.batch_counts.append(len(high))
        result = high.clone()
        result[:, 0, 0] += 1
        result[:, 0, 1] = embed[:, 0, 0]
        return result, low

    def digit_logits(self, high):
        depths = high[:, 0, 0].long() - 1
        ids = high[:, 0, 1].long()
        predictions = self.plan[ids, depths] - 1
        return F.one_hot(predictions, 4).float() * 10


def fixture():
    tokens = torch.ones((3, 16), dtype=torch.long)
    tokens[0, 0] = 2
    tokens[1, 1] = 3
    return tokens, SOLUTION.repeat(3, 1) - 1


def fake_memory(device="cpu"):
    return {
        "device": str(device), "host_rss_bytes": 100_000,
        "host_peak_rss_bytes": 100_000, "host_available_bytes": 8 * 1024**3,
        "cuda_reserved_bytes": None, "cuda_free_bytes": None,
    }


class EvaluationTests(unittest.TestCase):
    def test_compaction_reduces_actual_model_batch_sizes(self):
        model = ToyTrajectory()
        tokens, _ = fixture()
        result = rollout_first_valid(model, tokens, max_depth=4)
        self.assertEqual(model.batch_counts, [3, 3, 2, 2])
        self.assertEqual(result["executed_case_steps"], 10)
        self.assertEqual(result["selected_depths"].tolist(), [2, 4, 4])
        self.assertEqual(result["valid_found"].tolist(), [True, True, False])
        torch.testing.assert_close(result["predictions"][:2], SOLUTION.repeat(2, 1))
        self.assertFalse(result["selection_uses_gold"])

    def test_all_solved_stops_before_budget(self):
        model = ToyTrajectory()
        tokens, _ = fixture()
        result = rollout_first_valid(model, tokens[:1], max_depth=4)
        self.assertEqual(model.batch_counts, [1, 1])
        self.assertEqual(result["executed_outer_steps"], 2)
        self.assertEqual(result["executed_case_steps"], 2)

    @patch("mechinterp.bpd.experiments.depth_steering_gym.dt05.evaluation.memory_snapshot", side_effect=fake_memory)
    def test_same_case_raw_and_policy_outcomes_preserved(self, _memory):
        model = ToyTrajectory()
        tokens, gold = fixture()
        result = evaluate_policy(model, tokens, gold, depths=(1, 4), microbatch=3)
        self.assertEqual(result["observed_depths"], [1, 2, 3, 4])
        self.assertEqual(result["by_depth"]["2"]["per_case_exact"], [True, False, False])
        self.assertEqual(result["by_depth"]["4"]["per_case_exact"], [False, True, False])
        first = result["first_valid"]
        self.assertEqual(first["per_case_exact"], [True, True, False])
        self.assertEqual(first["raw_final_previously_valid_destruction_count"], 1)
        self.assertEqual(first["raw_final_previously_correct_destruction_count"], 1)
        self.assertTrue(first["compact_matches_trajectory_policy"])
        self.assertEqual(first["compact_case_steps"], 10)
        self.assertEqual(first["fixed_case_steps"], 12)
        self.assertEqual(result["tail"]["per_case_period2"], [False, False, True])
        self.assertIn("no speedup claim", result["timing_scope"])

    @patch("mechinterp.bpd.experiments.depth_steering_gym.dt05.evaluation.memory_snapshot", side_effect=fake_memory)
    def test_microbatch_boundaries_preserve_outcomes(self, _memory):
        tokens, gold = fixture()
        results = [evaluate_policy(ToyTrajectory(), tokens, gold, depths=(2, 4), microbatch=n) for n in (1, 2, 3)]
        for result in results[1:]:
            self.assertEqual(result["by_depth"], results[0]["by_depth"])
            for name in ("predictions", "selected_depths", "valid_found", "compact_case_steps"):
                self.assertEqual(result["first_valid"][name], results[0]["first_valid"][name])

    @patch("mechinterp.bpd.experiments.depth_steering_gym.dt05.evaluation.memory_snapshot", side_effect=fake_memory)
    def test_training_mode_restored(self, _memory):
        tokens, gold = fixture()
        model = ToyTrajectory().train()
        rollout_first_valid(model, tokens, max_depth=4)
        self.assertTrue(model.training)
        evaluate_policy(model, tokens, gold, depths=(2, 4))
        self.assertTrue(model.training)

    @patch("mechinterp.bpd.experiments.depth_steering_gym.dt05.evaluation.memory_snapshot", side_effect=fake_memory)
    def test_memory_budget_checked_before_any_rollout(self, _memory):
        tokens, gold = fixture()
        model = ToyTrajectory()
        with self.assertRaises(BudgetExceeded):
            evaluate_policy(model, tokens, gold, depths=(4,), budget=MemoryBudget(max_host_rss_gib=0.00001))
        self.assertEqual(model.batch_counts, [])

    def test_invalid_gold_depth_or_tokens_rejected(self):
        model = ToyTrajectory()
        tokens, gold = fixture()
        with self.assertRaises(ValueError):
            evaluate_policy(model, tokens, torch.zeros_like(gold), depths=(4,))
        with self.assertRaises(ValueError):
            evaluate_policy(model, tokens, gold, depths=(4, 2))
        with self.assertRaises(ValueError):
            rollout_first_valid(model, tokens.float(), max_depth=4)
        with self.assertRaises(ValueError):
            rollout_first_valid(model, tokens, max_depth=0)

    def test_nonfinite_model_output_fails_closed(self):
        model = ToyTrajectory()
        tokens, _ = fixture()
        with patch.object(model, "digit_logits", return_value=torch.full((3, 16, 4), float("nan"))):
            with self.assertRaises(RuntimeError):
                rollout_first_valid(model, tokens, max_depth=4)

    def test_native_transport_diagnostic_is_bounded_and_label_free(self):
        from mechinterp.bpd.adapters.trm import build_mini_trm
        model = build_mini_trm(hidden_size=16, num_heads=2, l_layers=1,
                               h_cycles=1, l_cycles=1, puzzle_emb_len=1,
                               seq_len=16, seed=1843).eval()
        result = transport_diagnostic(model, torch.ones((1, 16), dtype=torch.long), rank=2)
        self.assertEqual(result["full_dual_state_dim"], 544)
        self.assertEqual(result["state_basis_elements"], 544 * 2)
        self.assertLessEqual(result["head_row_basis_elements"], 11 * 16)
        self.assertFalse(result["future_labels_used"])
        self.assertFalse(result["fitted_task_gramian"])
        self.assertFalse(result["measured_causal_benefit"])


if __name__ == "__main__":
    unittest.main()
