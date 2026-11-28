"""Tests of actual training treatment, information parity and update matching."""
import copy
import unittest

import torch
import torch.nn.functional as F

from .data import make_pairs
from .experiment import (Recipe, combine_gradients, full_task_batch, new_model,
                         rotation_for, seed_for, state_hash, training_step)


class TrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        torch.manual_seed(14)
        recipe = Recipe(width=8, components=8, batch_size=4, group_size=2)
        model = new_model(recipe)
        batch = make_pairs(17, 4, root_seed=71, delay=1)
        groups = torch.arange(8).reshape(4, 2)
        return recipe, model, batch, groups, rotation_for(71, 24)

    def test_baseline_matches_plain_task_update_exactly(self):
        recipe, model, batch, groups, rotation = self.fixture()
        plain = copy.deepcopy(model)
        options = dict(lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), **options)
        plain_optimizer = torch.optim.AdamW(plain.parameters(), **options)
        row = training_step(model, optimizer, batch, groups, rotation, recipe, "A_task")
        x, labels = full_task_batch(batch)
        loss = F.cross_entropy(plain(x), labels)
        plain_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plain.parameters(), recipe.gradient_clip)
        plain_optimizer.step()
        self.assertEqual(state_hash(model), state_hash(plain))
        self.assertEqual(row["applied_auxiliary_gradient_l2"], 0)
        self.assertEqual(row["gradient_evaluations"], 3)

    def test_steering_changes_training_and_has_matched_gradient_ratio(self):
        recipe, model, batch, groups, rotation = self.fixture()
        models = {a: copy.deepcopy(model) for a in ("A_task", "B_causal", "C_rotated")}
        rows = {}
        for arm, candidate in models.items():
            opt = torch.optim.AdamW(candidate.parameters(), lr=recipe.learning_rate,
                                    weight_decay=recipe.weight_decay)
            rows[arm] = training_step(candidate, opt, batch, groups, rotation, recipe, arm)
        self.assertEqual(len({state_hash(m) for m in models.values()}), 3)
        for arm in ("B_causal", "C_rotated"):
            self.assertAlmostEqual(rows[arm]["realized_auxiliary_ratio"], .25, places=12)
            self.assertLessEqual(rows[arm]["physical_sham_l2_max_error"], 1e-6)
            self.assertGreater(rows[arm]["parameter_update_l2"], 0)
        self.assertEqual(len({rows[a]["trajectories"] for a in rows}), 1)

    def test_degenerate_auxiliary_stays_zero(self):
        task = (torch.tensor([3., 4.]),)
        aux = (torch.zeros(2),)
        combined, info = combine_gradients(task, aux, .25, True)
        self.assertTrue(torch.equal(combined[0], task[0]))
        self.assertEqual(info["applied_auxiliary_gradient_l2"], 0)
        self.assertTrue(info["auxiliary_degenerate"])

    def test_counterfactual_labels_available_as_ordinary_examples(self):
        _, _, batch, _, _ = self.fixture()
        x, labels = full_task_batch(batch)
        n = len(batch.labels_r)
        self.assertTrue(torch.equal(x[2*n:], batch.hybrid))
        self.assertTrue(torch.equal(labels[2*n:], batch.labels_h))
        domains = ("warmup", "attribution_train", "paired_train",
                   "evaluation_iid", "evaluation_long", "evaluation_causal")
        self.assertEqual(len({seed_for(71, d) for d in domains}), len(domains))


if __name__ == "__main__":
    unittest.main()
