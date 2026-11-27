"""Bounded source checks for RP03 state transport and detached projectors.

These engineering examples perform no optimization and create no study roots.
"""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F

from mechinterp.bpd.experiments.training_steering.data import make_pairs
from mechinterp.bpd.experiments.training_steering.model import ComponentMemory
from .model import ComponentStateMemory, projectors, state_patch


class StateTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7301)
        self.model = ComponentStateMemory(width=8, components=8)
        self.batch = make_pairs(7301, 8, root_seed=7301, delay=3, noise=0.)
        self.groups = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])

    def test_clean_and_zero_patch_match_RP02_bit_for_bit(self):
        original = ComponentMemory(width=8, components=8)
        original.load_state_dict(self.model.state_dict())
        expected = original.rollout(self.batch.receiver)
        clean = self.model.rollout(self.batch.receiver)
        zero = self.model.rollout(self.batch.receiver, patch=torch.zeros(8, 8))
        for key in ("logits", "states", "recurrent_writes"):
            self.assertTrue(torch.equal(clean[key], expected[key]), key)
            self.assertTrue(torch.equal(zero[key], expected[key]), key)
        self.assertTrue(torch.equal(clean["incoming_states"][:, 1:], clean["states"][:, :-1]))
        self.assertEqual(int(torch.count_nonzero(clean["incoming_states"][:, 0])), 0)

    def test_full_hybrid_prefix_state_reproduces_exact_suffix(self):
        hybrid = self.model.rollout(self.batch.hybrid)
        recipient = self.model.rollout(self.batch.receiver)
        replacement = hybrid["states"][:, 3]
        edited = self.model.rollout(self.batch.receiver, state_replacement=replacement)
        self.assertTrue(torch.equal(edited["incoming_states"][:, 4], replacement))
        self.assertTrue(torch.equal(edited["states"][:, :4], recipient["states"][:, :4]))
        for key in ("states", "incoming_states", "recurrent_writes"):
            self.assertTrue(torch.equal(edited[key][:, 4:], hybrid[key][:, 4:]), key)
        self.assertTrue(torch.equal(edited["logits"], hybrid["logits"]))

    def test_patch_enters_both_recurrent_operator_and_carry(self):
        clean = self.model.rollout(self.batch.receiver)
        patch = torch.linspace(-.3, .4, 64).reshape(8, 8)
        edited = self.model.rollout(self.batch.receiver, patch=patch)
        incoming = clean["states"][:, 3] + patch
        torch.testing.assert_close(edited["incoming_states"][:, 4] - clean["incoming_states"][:, 4],
                                   patch, rtol=1e-5, atol=1e-7)
        v, u = self.model.recurrent.V, self.model.recurrent.U
        write = (incoming @ v) @ u
        torch.testing.assert_close(edited["recurrent_writes"][:, 4], write, rtol=1e-5, atol=1e-7)
        ir, iz, inn = self.model.input(self.batch.receiver[:, 4]).chunk(3, dim=-1)
        hr, hz, hn = write.chunk(3, dim=-1)
        reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        expected = (1 - update) * torch.tanh(inn + reset * hn) + update * incoming
        torch.testing.assert_close(edited["states"][:, 4], expected, rtol=1e-5, atol=1e-7)
        stale_carry = (1 - update) * torch.tanh(inn + reset * hn) + update * clean["states"][:, 3]
        self.assertGreater(float((expected - stale_carry).norm().detach()), .01)

    def test_projectors_are_detached_symmetric_idempotent_and_basis_invariant(self):
        p = projectors(self.model, self.groups)
        self.assertFalse(p.requires_grad)
        torch.testing.assert_close(p, p.transpose(1, 2), rtol=1e-5, atol=2e-7)
        torch.testing.assert_close(p @ p, p, rtol=1e-5, atol=2e-7)
        torch.testing.assert_close(p.diagonal(dim1=1, dim2=2).sum(-1), torch.full((4,), 2.),
                                   rtol=1e-5, atol=3e-7)
        changed = copy.deepcopy(self.model)
        basis_change = torch.tensor([[0., -2.], [1.5, .3]])
        with torch.no_grad():
            for group in self.groups:
                changed.recurrent.V[:, group] = changed.recurrent.V[:, group] @ basis_change
        torch.testing.assert_close(projectors(changed, self.groups), p, rtol=2e-5, atol=3e-7)

    def test_rank_deficient_and_invalid_groups_fail_explicitly(self):
        for group in (self.groups.float(), torch.zeros_like(self.groups),
                      self.groups - 1, self.groups + 8):
            with self.subTest(group=group.tolist()), self.assertRaises(ValueError):
                projectors(self.model, group)
        with torch.no_grad():
            self.model.recurrent.V[:, 1].copy_(self.model.recurrent.V[:, 0])
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            projectors(self.model, self.groups)

    def test_physical_dose_zero_direction_and_gradient_barrier(self):
        p = projectors(self.model, self.groups).requires_grad_(True)
        recipient = torch.randn(8, 8, requires_grad=True)
        donor = torch.randn(8, 8, requires_grad=True)
        delta, metadata = state_patch(recipient, donor, p, self.batch.edit_slot)
        self.assertTrue(bool(metadata["valid"].all()))
        torch.testing.assert_close(delta.norm(dim=-1), torch.full((8,), .75), rtol=1e-6, atol=2e-7)
        permutation = torch.tensor([3, 6, 0, 2, 5, 7, 1, 4])
        signs = torch.tensor([1., -1., -1., 1., 1., -1., 1., -1.])
        torch.testing.assert_close((delta[:, permutation] * signs).norm(dim=-1), delta.norm(dim=-1),
                                   rtol=1e-6, atol=2e-7)
        objective = (delta * torch.arange(8)).sum()
        g_recipient, g_donor, g_projector = torch.autograd.grad(objective, (recipient, donor, p),
                                                               allow_unused=True)
        self.assertIsNone(g_donor)
        self.assertIsNone(g_projector)
        self.assertGreater(float(g_recipient.norm()), 1e-7)
        zero, zero_metadata = state_patch(recipient, recipient.detach(), p, self.batch.edit_slot)
        self.assertFalse(bool(zero_metadata["valid"].any()))
        self.assertEqual(int(torch.count_nonzero(zero)), 0)
        zero.sum().backward()
        self.assertTrue(bool(torch.isfinite(recipient.grad).all()))

    def test_equal_frozen_and_current_projectors_give_equal_training_gradients(self):
        frozen = projectors(self.model, self.groups).clone()
        gradients = []
        patches = []
        for use_current in (False, True):
            recipient = self.model.prefix_state(self.batch.receiver)
            donor = self.model.prefix_state(self.batch.donor)
            p = projectors(self.model, self.groups) if use_current else frozen
            delta, _ = state_patch(recipient, donor, p, self.batch.edit_slot)
            loss = F.cross_entropy(self.model.rollout(self.batch.receiver, patch=delta)["logits"],
                                   self.batch.labels_h)
            gradients.append(torch.autograd.grad(loss, tuple(self.model.parameters())))
            patches.append(delta.detach())
        self.assertTrue(torch.equal(patches[0], patches[1]))
        for left, right in zip(*gradients):
            self.assertTrue(torch.equal(left, right))

    def test_native_component_path_has_zero_residual_and_handles_long_sequences(self):
        observations = []
        def observe(module, inputs):
            observations.append((module.mode, bool((module.schedule.delta_mask == 0).all()),
                                 bool((module.schedule.comp_mask == 1).all())))
        handle = self.model.recurrent.register_forward_pre_hook(observe)
        x = torch.randn(2, 68, 11)
        clean = self.model(x).detach()
        handle.remove()
        self.assertEqual(len(observations), 68)
        self.assertTrue(all(value == ("component", True, True) for value in observations))
        with torch.no_grad():
            self.model.recurrent.W_target.fill_(2345.)
        self.assertTrue(torch.equal(clean, self.model(x).detach()))
        self.assertIsNone(self.model.recurrent.schedule)

    def test_invalid_state_patch_and_rollout_shapes_fail(self):
        recipient, donor = torch.randn(8, 8), torch.randn(8, 8)
        p = projectors(self.model, self.groups)
        for dose in (0., -1., float("nan"), float("inf")):
            with self.subTest(dose=dose), self.assertRaises(ValueError):
                state_patch(recipient, donor, p, self.batch.edit_slot, dose=dose)
        for bad_donor, bad_p, bad_slot in (
            (donor[:, :7], p, self.batch.edit_slot),
            (donor, p[:, :, :7], self.batch.edit_slot),
            (donor, p, self.batch.edit_slot.float()),
            (donor, p, self.batch.edit_slot + 4),
        ):
            with self.assertRaises(ValueError):
                state_patch(recipient, bad_donor, bad_p, bad_slot)
        for kwargs in ({"patch": torch.zeros(8, 24)},
                       {"patch": torch.zeros(8, 8), "patch_step": 99},
                       {"patch": torch.zeros(8, 8), "state_replacement": torch.zeros(8, 8)}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                self.model.rollout(self.batch.receiver, **kwargs)


if __name__ == "__main__":
    unittest.main()
