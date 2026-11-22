"""Source-only tests for the pretrained runner's fail-closed provenance.

Uses fake local files, no HF model load, no model download, no network.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from run_pretrained_dev import (
    ARMS, candidate_token_id, checkpoint_manifest, evaluate_logits,
    load_development_tasks, source_blob_sha, validate_loaded_model,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [2] if text == " A" else [3, 4]


class FakeConfig:
    model_type = "inkling2_loop"
    hidden_size = 2304
    num_hidden_layers = 24
    loop_layer_index_start = 6
    loop_layer_index_end = 18
    loop_num_passes = 3
    n_layers_per_attn_res_block = 6


class FakeQuery:
    shape = (12, 4, 2304)


class FakeMessageInput:
    query = FakeQuery()


class FakeCore:
    config = FakeConfig()
    message_input = FakeMessageInput()


class FakeWrapper:
    config = FakeConfig()

    class model:
        language_model = FakeCore()


class PretrainedRunnerSourceTests(unittest.TestCase):
    def test_fixed_eight_sham_family_and_separate_zero(self):
        self.assertEqual(len(ARMS), 10)
        self.assertEqual(ARMS[:2], ["zero", "candidate"])
        self.assertEqual(len([x for x in ARMS if x.startswith("coord_")]), 4)
        self.assertEqual(len([x for x in ARMS if x.startswith("haar_")]), 4)

    def test_task_file_rejects_confirmation_and_duplicate_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tasks.jsonl"
            item = {
                "phase": "development", "root_id": "dev-01",
                "family_id": "arithmetic", "prompt": "2 + 2 =",
                "correct": " 4", "distractor": " 5",
            }
            p.write_text(json.dumps(item) + "\n")
            rows, sha = load_development_tasks(p)
            self.assertEqual(len(rows), 1)
            self.assertEqual(sha, hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertNotIn("phase", rows[0])
            p.write_text(json.dumps(item | {"phase": "confirmation"}) + "\n")
            with self.assertRaisesRegex(ValueError, "development"):
                load_development_tasks(p)
            p.write_text((json.dumps(item) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_development_tasks(p)

    def test_checkpoint_manifest_includes_local_shards_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"model_type":"inkling2_loop"}')
            (root / "model-00001-of-00001.safetensors").write_bytes(b"synthetic-shard")
            (root / "tokenizer.json").write_text('{"fake":true}')
            m = checkpoint_manifest(root)
            self.assertEqual(m["shard_count"], 1)
            self.assertEqual(m["weight_bytes"], len(b"synthetic-shard"))
            self.assertEqual(len(m["sha256"]), 64)
            self.assertEqual(m["file_count"], 3)
            (root / "model-00001-of-00001.safetensors").write_bytes(b"changed")
            self.assertNotEqual(m["sha256"], checkpoint_manifest(root)["sha256"])
            (root / "model-00001-of-00001.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "no local"):
                checkpoint_manifest(root)

    def test_source_git_blob_lock_follows_sha1_blob_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "modeling.py"
            p.write_bytes(b"hello\n")
            expected = hashlib.sha1(b"blob 6\x00hello\n").hexdigest()
            self.assertEqual(source_blob_sha(p), expected)

    def test_token_and_restricted_outcome_contracts(self):
        t = FakeTokenizer()
        self.assertEqual(candidate_token_id(t, " A"), 2)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            candidate_token_id(t, " B")
        report = evaluate_logits(torch.tensor([[0.1, 0.2, 0.8, 0.7]]), 2, 3)
        self.assertEqual(report["fullhead_correct"], 1)
        self.assertEqual(report["restricted_pair_correct"], 1)
        self.assertAlmostEqual(report["correct_vs_distractor_margin"], 0.1, places=5)
        with self.assertRaises(ValueError):
            evaluate_logits(torch.tensor([[1., 2.], [3., 4.]]), 0, 1)

    def test_loaded_model_geometry_is_explicitly_checked(self):
        # Faux 12 x 4 x 2304 query shape, with a class-named source.
        fake = FakeWrapper()
        with self.assertRaisesRegex(RuntimeError, "not the expected"):
            validate_loaded_model(fake)
        class Inkling2LoopMessageAttentionResidual:
            query = FakeQuery()
        fake.model.language_model.message_input = Inkling2LoopMessageAttentionResidual()
        self.assertIs(validate_loaded_model(fake), fake.model.language_model)
        fake.model.language_model.config.loop_num_passes = 2
        with self.assertRaisesRegex(RuntimeError, "model geometry"):
            validate_loaded_model(fake)
        fake.model.language_model.config.loop_num_passes = 3


if __name__ == "__main__":
    unittest.main()
