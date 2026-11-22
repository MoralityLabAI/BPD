"""Contract tests for the source-pinned HF Inkling2Loop message_input hook.

All forward calls use a compact *synthetic* torch module implementing the
inspected four-query, two-previous-block equations. No downloaded weights.
"""
from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from pretrained_message_hook import (
    InklingMessageHook, PatchSpec, direction_for_arm, fixed_chord_rotation,
    recompute_message,
)


class FakeUnweightedRMSNorm(nn.Module):
    def forward(self, tensor):
        return tensor.float() * torch.rsqrt(tensor.float().square().mean(-1, keepdim=True) + 1e-6)


class FakeInklingMessage(nn.Module):
    def __init__(self, width=12, n_layers=12):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_layers, 4, width))
        self.norm = FakeUnweightedRMSNorm()

    def forward(self, previous_block_states, previous_hidden_states, loop_layer_idx):
        states = torch.cat(
            [previous_block_states, previous_hidden_states.unsqueeze(-2)], dim=-2
        ).float()
        normalized = self.norm(states)
        query = self.query[loop_layer_idx].float().expand(
            *normalized.shape[:-2], 4, normalized.shape[-1]
        )
        scores = query @ normalized.transpose(-1, -2) / math.sqrt(normalized.shape[-1])
        weights = scores.softmax(-1)
        return weights @ normalized


def synthetic_forward(module, base2, base3, spec=None, *, wrong_sequence=False):
    state = base3.clone()
    if spec is None:
        for _pass in (1, 2):
            previous_block3 = state
            contributions = [
                module(base2, previous_block3, layer)[:, :, 3]
                for layer in range(12)
            ]
            state = state + 0.001 * torch.stack(contributions).sum(0)
        return state, None
    with InklingMessageHook(module, spec) as hook:
        for p in (1, 2):
            # In actual Inkling, all 12 readers of a pass receive the same
            # immutable final block-3 output from the *previous* pass.
            previous_block3 = state
            contributions = []
            for layer in range(12):
                index = layer if not (wrong_sequence and p == 1 and layer == 3) else 4
                message = module(base2, previous_block3, index)
                contributions.append(message[:, :, 3])
            state = state + 0.001 * torch.stack(contributions).sum(0)
    return state, hook.finish()


class PretrainedHookContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(78)
        self.module = FakeInklingMessage()
        self.block2 = torch.randn(1, 3, 1, 12)
        self.block3 = torch.randn(1, 3, 12)

    def spec(self, arm="candidate", passes=(1,)):
        return PatchSpec(
            arm=arm, coordinate=7, passes=passes,
            dose_fraction=0.08, num_loop_layers=12, total_passes=3,
            probe_layer=11, seed=101,
        )

    def test_exact_zero_hook_parity_over_full_toy_continuation(self):
        with torch.no_grad():
            plain, _ = synthetic_forward(self.module, self.block2, self.block3)
            zero, receipt = synthetic_forward(
                self.module, self.block2, self.block3, self.spec("zero")
            )
        self.assertTrue(torch.equal(plain, zero))
        self.assertEqual(receipt["call_count"], 24)
        self.assertEqual(receipt["physical_read_count"], 0)
        self.assertEqual(len(receipt["readers"]), 2)
        self.assertLess(receipt["max_message_parity_abs"], 1e-5)

    def test_all_four_readers_and_delivered_matched_dose(self):
        with torch.no_grad():
            target, t = synthetic_forward(
                self.module, self.block2, self.block3, self.spec("candidate")
            )
            sham, s = synthetic_forward(
                self.module, self.block2, self.block3, self.spec("haar_2")
            )
        self.assertFalse(torch.equal(target, sham))
        self.assertEqual(t["physical_read_count"], 12)
        self.assertEqual(s["physical_read_count"], 12)
        self.assertEqual(len(t["readers"][0]["beta3_change"]), 4)
        self.assertLess(t["max_message_parity_abs"], 1e-5)
        self.assertAlmostEqual(t["delivered"][0]["source_l2"],
                               s["delivered"][0]["source_l2"], places=6)
        self.assertAlmostEqual(t["delivered"][0]["delivered_l2"],
                               s["delivered"][0]["delivered_l2"], places=5)
        for arm in (t, s):
            for patch in arm["delivered"]:
                self.assertAlmostEqual(patch["delivered_l2_fraction"], .08, places=5)
                self.assertAlmostEqual(patch["delivered_norm_ratio"], 1, places=5)
            self.assertTrue(all(
                len(rec["beta3_patched"]) == 4 for rec in arm["readers"]
            ))

    def test_transient_and_sustained_interventions_have_separate_dose_ledgers(self):
        with torch.no_grad():
            _, transient = synthetic_forward(
                self.module, self.block2, self.block3, self.spec(passes=(1,))
            )
            _, sustained = synthetic_forward(
                self.module, self.block2, self.block3, self.spec(passes=(1, 2))
            )
        self.assertEqual(transient["physical_read_count"], 12)
        self.assertEqual(sustained["physical_read_count"], 24)
        self.assertEqual(sorted(set(x["pass"] for x in transient["delivered"])), [1])
        self.assertEqual(sorted(set(x["pass"] for x in sustained["delivered"])), [1, 2])

    def test_pretrained_four_reader_equation_is_independently_recomputed(self):
        with torch.no_grad():
            b2 = self.block2[0, -1, 0]
            b3 = self.block3[0, -1]
            beta, expected = recompute_message(self.module, b2, b3, 7)
            observed = self.module(self.block2, self.block3, 7)[0, -1]
        self.assertEqual(tuple(beta.shape), (4,))
        self.assertEqual(tuple(expected.shape), (4, 12))
        self.assertTrue(torch.allclose(expected, observed, atol=1e-5))
        self.assertTrue(bool(((beta > 0) & (beta < 1)).all()))

    def test_independent_haar_seeds_and_control_coordinate_separation(self):
        x = self.block3[0, -1]
        v1 = direction_for_arm(x, self.spec("haar_0"))
        v2 = direction_for_arm(x, self.spec("haar_0"))
        v3 = direction_for_arm(x, self.spec("haar_3"))
        self.assertTrue(torch.equal(v1, v2))
        self.assertFalse(torch.equal(v1, v3))
        c = direction_for_arm(x, self.spec("coord_0"))
        self.assertEqual(c[7].item(), 0)
        self.assertEqual(int((c != 0).sum()), 1)

    def test_fail_closed_on_source_signature_and_radial_degeneracy(self):
        with self.assertRaisesRegex(RuntimeError, "sequence changed"):
            with InklingMessageHook(self.module, self.spec()) as hook:
                self.module(self.block2, self.block3, 1)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            x = torch.tensor([1., 0., 0., 0.])
            fixed_chord_rotation(x, torch.tensor([1., 0., 0., 0.]), .1)
        with self.assertRaises(ValueError):
            PatchSpec(arm="unregistered", coordinate=7)
        with self.assertRaises(ValueError):
            PatchSpec(arm="candidate", coordinate=7, passes=(0,))

    def test_hooks_detach_after_exception(self):
        with self.assertRaises(RuntimeError):
            with InklingMessageHook(self.module, self.spec()):
                self.module(self.block2, self.block3, 2)
        self.assertEqual(len(self.module._forward_pre_hooks), 0)
        self.assertEqual(len(self.module._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
