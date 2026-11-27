"""Treatment checks for the actual optimizer update and scored-data barrier."""
from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F

from ..training_steering.data import partition_mask, root_encoding
from ..training_steering.experiment import full_task_batch, state_hash
from .data import binding_batch, make_batch
from .experiment import ARMS, Recipe, new_model, rotation_for, training_step
from .model import projectors


class TrainingContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(774)
        self.recipe = Recipe(width=8, components=8, group_size=2, batch_size=8)
        self.model = new_model(self.recipe)
        self.groups = torch.arange(8).reshape(4, 2)
        self.fixed = projectors(self.model, self.groups)
        self.rotation = rotation_for(99003, self.recipe.width)
        self.batch = make_batch(99003, 'contract_train', 0, 8, 2, 'train', self.recipe)

    def optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.recipe.learning_rate,
                                 weight_decay=self.recipe.weight_decay)

    def step(self, model, arm, optimizer=None):
        return training_step(model, optimizer or self.optimizer(model), self.batch,
                              self.groups, self.fixed, self.rotation, self.recipe, arm)

    def test_task_arm_update_matches_plain_task_training_exactly(self):
        a, plain = copy.deepcopy(self.model), copy.deepcopy(self.model)
        row = self.step(a, 'A_task')
        optimizer = self.optimizer(plain)
        x, labels = full_task_batch(self.batch)
        loss = F.cross_entropy(plain(x), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plain.parameters(), self.recipe.gradient_clip)
        optimizer.step()
        self.assertEqual(state_hash(a), state_hash(plain))
        self.assertEqual(row['applied_auxiliary_gradient_l2'], 0)

    def test_fixed_and_live_updates_match_when_geometry_matches(self):
        fixed, live = copy.deepcopy(self.model), copy.deepcopy(self.model)
        b, d = self.step(fixed, 'B_causal_fixed'), self.step(live, 'D_causal_live')
        self.assertEqual(state_hash(fixed), state_hash(live))
        self.assertEqual(b['fixed_ce'], d['live_ce'])
        self.assertTrue(b['projectors_detached'] and d['projectors_detached'])

    def test_auxiliary_signal_changes_weights_and_direction(self):
        models = {arm: copy.deepcopy(self.model) for arm in ARMS}
        rows = {arm: self.step(model, arm) for arm, model in models.items()}
        self.assertNotEqual(state_hash(models['A_task']), state_hash(models['B_causal_fixed']))
        self.assertNotEqual(state_hash(models['B_causal_fixed']), state_hash(models['C_rotated_fixed']))
        for arm in ARMS:
            row = rows[arm]
            self.assertEqual(row['trajectories'], 48)
            self.assertEqual(row['gradient_evaluations'], 4)
            self.assertLess(row['sham_norm_max_error'], 1e-6)
            self.assertEqual(row['fixed_invalid'], 0)
            self.assertEqual(row['live_invalid'], 0)
            if arm != 'A_task':
                self.assertAlmostEqual(row['realized_auxiliary_ratio'], .25, places=12)
                self.assertGreater(row['applied_auxiliary_gradient_l2'], 0)

    def test_fixed_projector_values_survive_optimizer_updates(self):
        before = self.fixed.clone()
        model = copy.deepcopy(self.model)
        optimizer = self.optimizer(model)
        self.step(model, 'B_causal_fixed', optimizer)
        self.step(model, 'B_causal_fixed', optimizer)
        self.assertTrue(torch.equal(before, self.fixed))
        self.assertFalse(self.fixed.requires_grad)
        self.assertGreater(float((projectors(model, self.groups) - self.fixed).abs().max()), 0)

    def test_all_query_binding_cases_keep_semantic_partition_and_raw_labels(self):
        for split in ('train', 'eval'):
            x, labels, metadata = binding_batch(99003, 'binding_contract_' + split,
                                                32, 3, split, self.recipe)
            memory = torch.tensor(metadata['memories'])
            self.assertTrue(bool(partition_mask(memory, split).all()))
            address, values = root_encoding(99003)
            decoded_address = address.argsort()[x[:, :4, 3:7].argmax(-1)]
            decoded_values = values.argsort()[x[:, :4, 7:].argmax(-1)]
            reconstructed = torch.empty_like(memory).scatter(1, decoded_address, decoded_values)
            self.assertTrue(torch.equal(memory, reconstructed))
            query = address.argsort()[x[:, -1, 3:7].argmax(-1)]
            self.assertTrue(torch.equal(labels, reconstructed.gather(1, query[:, None]).squeeze(1)))
            self.assertEqual(torch.bincount(labels, minlength=4).tolist(), [8, 8, 8, 8])
            self.assertEqual(metadata['modal_baseline_accuracy'], .25)
            self.assertEqual(len(set(labels.tolist())), 4)

    def test_unknown_training_arm_fails(self):
        with self.assertRaises(ValueError):
            self.step(self.model, 'unknown')


if __name__ == '__main__':
    unittest.main()
