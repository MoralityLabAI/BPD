"""Native Nanbeige code parity using a tiny randomly initialized checkpoint.

Set BPD_COL3050_SOURCE_DIR to the metadata-only pinned source directory. The
test creates its own small weights; no pretrained checkpoint is downloaded.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streaming


class StreamingNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get("BPD_COL3050_SOURCE_DIR")
        if not source:
            raise unittest.SkipTest("Set BPD_COL3050_SOURCE_DIR for native architecture parity")
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM
            import safetensors
            import psutil
        except ImportError as error:
            raise unittest.SkipTest(f"Native test dependencies unavailable: {error}") from error
        cls.torch = torch
        cls.source = Path(source)
        for name in ("config.json", "configuration_nanbeige.py", "modeling_nanbeige.py"):
            if streaming.sha256_file(cls.source / name) != streaming.SOURCE_SHA256[name]:
                raise AssertionError(f"Unverified official test source: {name}")
        cls.temporary = tempfile.TemporaryDirectory(prefix="bpd-col3050-native-")
        cls.snapshot = Path(cls.temporary.name)
        config = AutoConfig.from_pretrained(str(cls.source), trust_remote_code=True, local_files_only=True)
        for name, value in {
            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
            "vocab_size": 64, "num_loops": 2, "max_position_embeddings": 128,
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
            "rope_theta": 10000.0, "use_cache": False,
        }.items():
            setattr(config, name, value)
        config._attn_implementation = "sdpa"
        torch.manual_seed(813)
        cls.native = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=torch.float32,
            attn_implementation="sdpa").eval()
        cls.native.save_pretrained(cls.snapshot, safe_serialization=True, max_shard_size="8KB")
        for name in ("configuration_nanbeige.py", "modeling_nanbeige.py"):
            shutil.copy2(cls.source / name, cls.snapshot / name)
        cls.ids = torch.tensor([[3, 5, 7, 9, 11]], dtype=torch.long)
        cls.donor_ids = torch.tensor([[4, 6, 8, 10, 11]], dtype=torch.long)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temporary"):
            cls.temporary.cleanup()

    def backend(self):
        return streaming.Backend._from_fixture(copy.deepcopy(self.native), self.snapshot)

    def test_streamed_native_forward_and_complete_head_match(self):
        torch = self.torch
        with torch.inference_mode():
            native_hidden = self.native.model(self.ids, use_cache=False).last_hidden_state[0, -1]
            native_logits = self.native(self.ids, use_cache=False).logits[0, -1]
        with self.backend() as backend:
            result = backend.run(self.ids)
            self.assertTrue(torch.allclose(result.final_hidden_cpu, native_hidden, atol=2e-6, rtol=2e-6))
            logits = backend.score_hidden(result.final_hidden_cpu)[0]
            self.assertTrue(torch.allclose(logits, native_logits, atol=2e-6, rtol=2e-6))
            selected = backend.score_hidden(result.final_hidden_cpu, [3, 11, 2])[0]
            self.assertTrue(torch.allclose(selected, native_logits[[3, 11, 2]], atol=2e-6, rtol=2e-6))
            self.assertEqual(result.receipt["native_layer_visits"], [[0, 0], [0, 1], [1, 0], [1, 1]])
            self.assertEqual(len(result.boundary_states_cpu), 2)
            self.assertTrue(backend.layers_released())
            self.assertTrue(result.receipt["decoder_parameters_released_to_meta"])
            # The forward read complete decoder blocks twice and only requested
            # embedding rows; neither vocabulary-sized table was materialized.
            layer_bytes = sum(p.numel() * 4 for layer in self.native.model.layers for p in layer.parameters())
            expected_bytes = 2 * layer_bytes + self.ids.numel() * self.native.config.hidden_size * 4
            self.assertEqual(result.receipt["streamed_checkpoint_bytes"], expected_bytes)

    def test_protected_prefix_patch_changes_later_native_readout(self):
        torch = self.torch
        with self.backend() as backend:
            recipient = backend.run(self.ids)
            donor = backend.run(self.donor_ids)

            def patch(state, visit):
                if visit:
                    return state
                result = state.clone()
                result[:, :-1, :] = donor.boundary_states_cpu[0][:, :-1, :].to(state)
                return result

            edited = backend.run(self.ids, intervention=patch)
            self.assertTrue(torch.equal(recipient.boundary_last_cpu[0], edited.boundary_last_cpu[0]))
            # Match production's one-vector scoring shape for every readout;
            # GEMM row tiles in a batch can round identical vectors differently.
            current_logits = torch.stack([
                backend.score_hidden(vector)[0]
                for vector in (recipient.boundary_last_cpu[0], edited.boundary_last_cpu[0])
            ])
            self.assertTrue(torch.equal(current_logits[0], current_logits[1]))
            future_logits = torch.stack([
                backend.score_hidden(vector)[0]
                for vector in (recipient.final_hidden_cpu, edited.final_hidden_cpu)
            ])
            self.assertGreater((future_logits[0] - future_logits[1]).abs().max().item(), 1e-7)
            self.assertEqual(edited.receipt["protected_last_token_equal_each_boundary"], [True, True])
            self.assertTrue(backend.layers_released())

    def test_bf16_streaming_matches_native_bf16(self):
        torch = self.torch
        from transformers import AutoModelForCausalLM
        # Native dtype construction keeps explicitly FP32 RoPE frequencies.
        # A blanket model.to(BF16) would incorrectly round those buffers too.
        native_bf16 = AutoModelForCausalLM.from_config(
            copy.deepcopy(self.native.config), trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        native_bf16.load_state_dict(self.native.state_dict())
        with torch.inference_mode():
            expected = native_bf16(self.ids, use_cache=False).logits[0, -1]
            expected_hidden = native_bf16.model(self.ids, use_cache=False).last_hidden_state[0, -1]
        with streaming.Backend._from_fixture(native_bf16, self.snapshot, dtype="bfloat16") as backend:
            result = backend.run(self.ids)
            self.assertTrue(torch.equal(result.final_hidden_cpu, expected_hidden))
            actual = backend.score_hidden(result.final_hidden_cpu)[0]
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(backend.layers_released())

    def test_identity_intervention_and_failure_cleanup(self):
        torch = self.torch
        with self.backend() as backend:
            clean = backend.run(self.ids)
            identity = backend.run(self.ids, intervention=lambda state, visit: state.clone())
            self.assertTrue(torch.equal(clean.final_hidden_cpu, identity.final_hidden_cpu))

            def corrupt_query(state, visit):
                result = state.clone()
                result[:, -1, 0] += 1
                return result

            with self.assertRaisesRegex(ValueError, "protected final-token"):
                backend.run(self.ids, intervention=corrupt_query)
            self.assertTrue(backend.layers_released())
            recovered = backend.run(self.ids)
            self.assertTrue(torch.equal(clean.final_hidden_cpu, recovered.final_hidden_cpu))
            with self.assertRaises(ValueError):
                backend.run(torch.zeros((2, 5), dtype=torch.long))
            with self.assertRaises(ValueError):
                backend.run(torch.zeros((1, 129), dtype=torch.long))
            with self.assertRaisesRegex(ValueError, "must be integers"):
                backend.run([1.5, 3.0])

    def test_failed_layer_load_releases_partial_residency(self):
        with self.backend() as backend:
            original = backend.store.tensor

            def fail_second_parameter(name, **kwargs):
                if name.startswith("model.layers.0.") and "k_proj" in name:
                    raise OSError("fixture read failure")
                return original(name, **kwargs)

            backend.store.tensor = fail_second_parameter
            with self.assertRaisesRegex(OSError, "fixture read failure"):
                backend.run(self.ids)
            self.assertTrue(backend.layers_released())
            backend.store.tensor = original
            self.assertEqual(len(backend.run(self.ids).boundary_states_cpu), 2)

    def test_tensor_store_copies_and_memory_guard_refuses_overrun(self):
        torch = self.torch
        store = streaming.TensorStore(self.snapshot)
        rows = store.rows("model.embed_tokens.weight", [3, 5], device="cpu", dtype=torch.float32)
        expected = rows.clone()
        rows.zero_()
        fresh = store.rows("model.embed_tokens.weight", [3, 5], device="cpu", dtype=torch.float32)
        self.assertTrue(torch.equal(fresh, expected))
        with self.backend() as backend:
            guard = backend.guard
            guard.process = SimpleNamespace(memory_info=lambda: SimpleNamespace(
                rss=guard.baseline_rss + guard.host_limit + 1))
            with self.assertRaises(streaming.MemoryBudgetExceeded):
                guard.check("synthetic admission boundary")

    def test_snapshot_tampering_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text('{"model_type":"changed"}\n')
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                streaming.inspect_snapshot(root)


if __name__ == "__main__":
    unittest.main()
