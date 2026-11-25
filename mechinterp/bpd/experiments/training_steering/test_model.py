"""Bounded source checks for the RP02 memory and physical intervention contract.

These checks use synthetic engineering examples and a single gradient pass.
They generate no experiment roots or task-learning results.
"""

from __future__ import annotations

import copy
import itertools
import unittest

import torch
import torch.nn.functional as F

from .data import canonical_codes, make_pairs, partition_mask, root_encoding
from .model import ComponentMemory, component_patch


class MemoryDataTests(unittest.TestCase):
    def test_complete_tuple_partition_and_generated_hybrids(self):
        memory = torch.tensor(list(itertools.product(range(4), repeat=4)))
        codes = canonical_codes(memory)
        self.assertEqual(len(torch.unique(codes)), 256)
        training = partition_mask(memory, "train")
        evaluation = partition_mask(memory, "eval")
        self.assertTrue(bool((training ^ evaluation).all()))
        self.assertFalse(bool((training & evaluation).any()))
        self.assertEqual(int(evaluation.sum()), 103)

        for split in ("train", "eval"):
            batch = make_pairs(101, 32, root_seed=41, delay=2, split=split)
            for values in (batch.values_r, batch.values_d, batch.values_h):
                self.assertTrue(bool(partition_mask(values, split).all()))
            row = torch.arange(len(batch.query))
            changed = batch.values_r != batch.values_h
            self.assertTrue(bool((changed.sum(1) == 1).all()))
            self.assertTrue(bool(changed[row, batch.edit_slot].all()))
            self.assertTrue(torch.equal(batch.values_h[row, batch.edit_slot],
                                        batch.values_d[row, batch.edit_slot]))

    def test_counterfactual_labels_and_balanced_queries(self):
        batch = make_pairs(102, 33, root_seed=42, delay=1)
        row = torch.arange(len(batch.query))
        self.assertTrue(torch.equal(batch.labels_r, batch.values_r[row, batch.query]))
        self.assertTrue(torch.equal(batch.labels_d, batch.values_d[row, batch.query]))
        self.assertTrue(torch.equal(batch.labels_h, batch.values_h[row, batch.query]))
        targeted = batch.query == batch.edit_slot
        self.assertEqual(int(targeted.sum()), 17)
        self.assertTrue(bool((batch.labels_h[targeted] != batch.labels_r[targeted]).all()))
        self.assertTrue(torch.equal(batch.labels_h[~targeted], batch.labels_r[~targeted]))
        counts = batch.edit_slot.bincount(minlength=4)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_write_once_encoding_and_late_query(self):
        batch = make_pairs(103, 12, root_seed=43, delay=3, noise=0.0)
        address_permutation, value_permutation = root_encoding(43)
        inverse_address = address_permutation.argsort()
        inverse_value = value_permutation.argsort()
        self.assertEqual(batch.prefix_steps, 4)
        self.assertEqual(tuple(batch.receiver.shape), (12, 8, 11))
        expected_types = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2]).expand(12, -1)
        for features, values in ((batch.receiver, batch.values_r),
                                 (batch.donor, batch.values_d),
                                 (batch.hybrid, batch.values_h)):
            self.assertTrue(torch.equal(features[:, :, :3].argmax(-1), expected_types))
            self.assertTrue(torch.equal(features[:, :, :3].sum(-1), torch.ones(12, 8)))
            addresses = inverse_address[features[:, :4, 3:7].argmax(-1)]
            written_values = inverse_value[features[:, :4, 7:].argmax(-1)]
            self.assertTrue(torch.equal(addresses.sort(1).values,
                                        torch.arange(4).expand(12, -1)))
            decoded = torch.zeros_like(values).scatter(1, addresses, written_values)
            self.assertTrue(torch.equal(decoded, values))
            self.assertTrue(torch.equal(inverse_address[features[:, -1, 3:7].argmax(-1)],
                                        batch.query))
            self.assertEqual(int(torch.count_nonzero(features[:, -1, 7:])), 0)
        self.assertTrue(torch.equal(batch.receiver[:, 4:], batch.donor[:, 4:]))
        self.assertTrue(torch.equal(batch.receiver[:, 4:], batch.hybrid[:, 4:]))

    def test_reproducibility_and_host_label_barrier(self):
        first = make_pairs(104, 8, root_seed=44, delay=2)
        again = make_pairs(104, 8, root_seed=44, delay=2)
        self.assertEqual(first.fingerprint(), again.fingerprint())
        self.assertNotEqual(first.fingerprint(),
                            make_pairs(105, 8, root_seed=44, delay=2).fingerprint())
        torch.manual_seed(44)
        model = ComponentMemory(width=8, components=8)
        original = model(first.receiver).detach()
        first.labels_r[:] = (first.labels_r + 1) % 4
        self.assertTrue(torch.equal(original, model(first.receiver).detach()))
        self.assertNotEqual(first.fingerprint(), again.fingerprint())

    def test_invalid_data_contract(self):
        for changes in ({"n": 0}, {"delay": -1}, {"slots": 1}, {"values": 1},
                        {"noise": float("nan")}, {"split": "confirmation"}):
            kwargs = dict(seed=1, n=4, root_seed=1, delay=1)
            kwargs.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                make_pairs(**kwargs)


class PhysicalComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(701)
        self.model = ComponentMemory(width=8, components=8)
        self.batch = make_pairs(701, 8, root_seed=701, delay=2, noise=0.0)
        self.groups = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])

    def test_target_buffer_is_inactive_and_factors_change_execution(self):
        reference = self.model(self.batch.receiver).detach()
        with torch.no_grad():
            self.model.recurrent.W_target.fill_(12345.0)
        changed_buffer = self.model(self.batch.receiver).detach()
        self.assertTrue(torch.equal(reference, changed_buffer))
        with torch.no_grad():
            self.model.recurrent.U.zero_()
        changed_factors = self.model(self.batch.receiver).detach()
        self.assertGreater(float((reference - changed_factors).abs().max()), 1e-5)
        self.assertEqual(self.model.recurrent.mode, "component")
        self.assertIsNone(self.model.recurrent.schedule)

    def test_task_gradient_reaches_both_shared_factors(self):
        logits = self.model(self.batch.receiver)
        loss = F.cross_entropy(logits, self.batch.labels_r)
        loss.backward()
        for parameter in (self.model.recurrent.V, self.model.recurrent.U,
                          self.model.input.weight, self.model.output.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.norm()), 1e-8)
        self.assertFalse(self.model.recurrent.W_target.requires_grad)

    def test_patch_and_rotated_sham_have_exact_declared_dose(self):
        recipient = self.model.prefix_state(self.batch.receiver)
        donor = self.model.prefix_state(self.batch.donor)
        delta, metadata = component_patch(self.model, recipient, donor, self.groups,
                                          self.batch.edit_slot, dose=.75)
        self.assertTrue(bool(metadata["valid"].all()))
        torch.testing.assert_close(metadata["delivered_norm"], torch.full((8,), .75),
                                   rtol=1e-6, atol=1e-7)
        permutation = torch.arange(24).roll(5)
        signs = torch.where(torch.arange(24) % 2 == 0, 1.0, -1.0)
        sham = delta[:, permutation] * signs
        torch.testing.assert_close(sham.norm(dim=-1), delta.norm(dim=-1),
                                   rtol=1e-6, atol=1e-7)

    def test_zero_direction_is_flagged_and_donor_state_is_detached(self):
        recipient = torch.randn(8, 8, requires_grad=True)
        donor = recipient.detach().clone().requires_grad_(True)
        delta, metadata = component_patch(self.model, recipient, donor, self.groups,
                                          self.batch.edit_slot)
        self.assertFalse(bool(metadata["valid"].any()))
        self.assertEqual(int(torch.count_nonzero(delta)), 0)
        donor = torch.randn(8, 8, requires_grad=True)
        delta, _ = component_patch(self.model, recipient, donor, self.groups,
                                   self.batch.edit_slot)
        objective = (delta * torch.arange(24)).sum()
        grad_recipient, grad_donor = torch.autograd.grad(
            objective, (recipient, donor), allow_unused=True)
        self.assertIsNone(grad_donor)
        self.assertIsNotNone(grad_recipient)
        self.assertGreater(float(grad_recipient.norm()), 1e-7)

    def test_patch_changes_the_named_write_and_preserves_prefix(self):
        ordinary = self.model.rollout(self.batch.receiver)
        recipient = ordinary["states"][:, 3]
        donor = self.model.prefix_state(self.batch.donor)
        delta, _ = component_patch(self.model, recipient, donor, self.groups,
                                   self.batch.edit_slot)
        edited = self.model.rollout(self.batch.receiver, patch=delta, patch_step=4)
        self.assertTrue(torch.equal(ordinary["states"][:, :4], edited["states"][:, :4]))
        self.assertTrue(torch.equal(ordinary["recurrent_writes"][:, :4],
                                    edited["recurrent_writes"][:, :4]))
        torch.testing.assert_close(edited["recurrent_writes"][:, 4]
                                   - ordinary["recurrent_writes"][:, 4], delta,
                                   rtol=1e-5, atol=1e-7)
        self.assertGreater(float((edited["states"][:, 4] - ordinary["states"][:, 4]).norm().detach()),
                           1e-5)
        self.assertGreater(float((edited["logits"] - ordinary["logits"]).norm().detach()), 1e-6)

    def test_component_gauge_permutation_preserves_forward_and_patch(self):
        permuted = copy.deepcopy(self.model)
        permutation = torch.tensor([5, 2, 7, 0, 3, 6, 1, 4])
        with torch.no_grad():
            permuted.recurrent.V.copy_(self.model.recurrent.V[:, permutation])
            permuted.recurrent.U.copy_(self.model.recurrent.U[permutation])
        groups_permuted = permutation.argsort()[self.groups]
        original = self.model.rollout(self.batch.receiver)
        renamed = permuted.rollout(self.batch.receiver)
        torch.testing.assert_close(original["logits"], renamed["logits"], rtol=1e-5, atol=1e-7)
        donor_original = self.model.prefix_state(self.batch.donor)
        donor_renamed = permuted.prefix_state(self.batch.donor)
        delta, _ = component_patch(self.model, original["states"][:, 3], donor_original,
                                   self.groups, self.batch.edit_slot)
        renamed_delta, _ = component_patch(permuted, renamed["states"][:, 3], donor_renamed,
                                           groups_permuted, self.batch.edit_slot)
        torch.testing.assert_close(delta, renamed_delta, rtol=1e-5, atol=2e-7)

    def test_invalid_interventions_fail(self):
        recipient, donor = torch.randn(8, 8), torch.randn(8, 8)
        for dose in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(dose=dose), self.assertRaises(ValueError):
                component_patch(self.model, recipient, donor, self.groups,
                                self.batch.edit_slot, dose=dose)
        bad_groups = [self.groups.float(), self.groups - 1, self.groups + 8,
                      torch.zeros_like(self.groups)]
        for groups in bad_groups:
            with self.subTest(groups=groups.tolist()), self.assertRaises(ValueError):
                component_patch(self.model, recipient, donor, groups, self.batch.edit_slot)
        with self.assertRaises(ValueError):
            self.model.rollout(self.batch.receiver, torch.zeros(8, 24), patch_step=99)


if __name__ == "__main__":
    unittest.main()
