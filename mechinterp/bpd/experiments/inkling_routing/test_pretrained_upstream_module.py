"""Optional integration test against the *actual* pinned HF message operator.

Uses a 16D initialized reader only. No pretrained checkpoint shards,
GPU, external hosted inference, or long transformer forward pass.
Run only in the CI job that installs the pinned Transformers source.
"""
from __future__ import annotations

import unittest
import torch

from pretrained_message_hook import InklingMessageHook, PatchSpec
from run_pretrained_dev import verify_transformers_source


class RealHfMessageOperatorTests(unittest.TestCase):
    def test_pinned_hf_source_and_real_module_read_edge(self):
        # Deliberately import here so basic CPU fixture tests need no HF.
        from transformers.models.inkling2_loop.configuration_inkling2_loop import (
            Inkling2LoopTextConfig,
        )
        from transformers.models.inkling2_loop.modeling_inkling2_loop import (
            Inkling2LoopMessageAttentionResidual,
        )
        self.assertEqual(len(verify_transformers_source()), 40)
        torch.set_num_threads(2)
        torch.manual_seed(341)
        config = Inkling2LoopTextConfig(
            hidden_size=16, num_hidden_layers=24,
            loop_layer_index_start=6, loop_layer_index_end=18,
            loop_num_passes=3, n_layers_per_attn_res_block=6,
        )
        module = Inkling2LoopMessageAttentionResidual(config).eval()
        with torch.no_grad():
            module.query.normal_(mean=0, std=.2)
        block2 = torch.randn(1, 4, 1, 16)
        block3 = torch.randn(1, 4, 16)
        spec = PatchSpec(
            arm="candidate", coordinate=7, probe_layer=11,
            num_loop_layers=12, total_passes=3,
        )
        with torch.inference_mode(), InklingMessageHook(module, spec) as hooked:
            for _pass in (1, 2):
                previous_block3 = block3
                contributions = []
                for layer in range(12):
                    msg = module(block2, previous_block3, layer)
                    self.assertEqual(tuple(msg.shape), (1, 4, 4, 16))
                    contributions.append(msg[:, :, 3])
                block3 = block3 + .001 * torch.stack(contributions).sum(0)
        receipt = hooked.finish()
        self.assertEqual(receipt["call_count"], 24)
        self.assertEqual(receipt["physical_read_count"], 12)
        self.assertEqual(len(receipt["readers"]), 2)
        self.assertLess(receipt["max_message_parity_abs"], 1e-4)
        self.assertTrue(all(
            abs(x["delivered_l2_fraction"] - .08) < .001
            for x in receipt["delivered"]
        ))


if __name__ == "__main__":
    unittest.main()
