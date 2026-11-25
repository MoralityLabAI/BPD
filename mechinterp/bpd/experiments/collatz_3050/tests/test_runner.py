"""Runner contracts, including eight forwards through pinned native code.

The native fixture contains randomly initialized tiny weights. It exercises
execution and saved evidence without measuring pretrained task competence.
Optional tokenizer coverage uses verified local metadata, with no download.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cases
import interventions
import run_gym
import scoring
import streaming
import test_streaming as native_fixture


def frozen_score_contract(manifest):
    return {"schema": run_gym.RUN_SCHEMA,
            "source_sha256": run_gym.source_hashes(),
            "cases_manifest_sha256": manifest["manifest_sha256"]}


class BoundaryTokenizer:
    """Explicit token tables for testing continuation/alignment validation.

    This stand-in supplies small vocabulary IDs to the tiny native model;
    RealPinnedTokenizerTests separately checks the checkpoint's tokenizer.
    """

    def __init__(self, case):
        self.texts = {
            role: self.render(case[role]["prompt"]) + "Answer:"
            for role in cases.ROLES
        }
        self.encodings = {}
        self.template_calls = []
        for role, ids in zip(cases.ROLES, (
            [3, 5, 7, 9, 11], [4, 6, 8, 10, 11], [12, 14, 16, 18, 11],
        )):
            self.set_prompt(role, ids)

    @staticmethod
    def render(prompt):
        return "system:" + run_gym.SYSTEM_PROMPT + "\nuser:" + prompt + "\nassistant:"

    def set_prompt(self, role, ids, answer_ids=None):
        answer_ids = answer_ids or {"A": 19, "B": 20}
        text = self.texts[role]
        self.encodings[text] = list(ids)
        for label, suffix in run_gym.ANSWER_SUFFIXES.items():
            self.encodings[text + suffix] = list(ids) + [answer_ids[label]]

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return self.render(messages[1]["content"])

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("The rendered chat template already supplies its special tokens")
        return list(self.encodings[text])


class RunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = cases.make_case(73, 1, (1, 9, 37), index=0, tier="trace")

    def test_forced_choice_continuations_and_template_options(self):
        tokenizer = BoundaryTokenizer(self.case)
        tokenized = run_gym.tokenize_case(tokenizer, self.case, max_tokens=5)
        self.assertEqual(set(tokenized), set(cases.ROLES))
        self.assertEqual(len(tokenizer.template_calls), 3)
        for role, (messages, options) in zip(cases.ROLES, tokenizer.template_calls):
            self.assertEqual(messages, [
                {"role": "system", "content": run_gym.SYSTEM_PROMPT},
                {"role": "user", "content": self.case[role]["prompt"]},
            ])
            self.assertEqual(options, {
                "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
            })
            self.assertEqual(tokenized[role]["answer_token_ids"], {"A": 19, "B": 20})
            self.assertEqual(tokenized[role]["input_ids"], tokenizer.encodings[tokenizer.texts[role]])
            self.assertEqual(len(tokenized[role]["input_ids"]), 5)

    def test_ambiguous_boundaries_and_unaligned_donors_are_refused(self):
        for fault, expected in (
            ("retokenized_prefix", "single continuation token"),
            ("two_answer_tokens", "single continuation token"),
            ("same_answer_token", "Answer tokens must differ"),
            ("different_answer_ids", "changed across matched prompts"),
            ("unequal_prompt_lengths", "different token counts"),
            ("collapsed_prompt", "identical model prompts"),
            ("too_short", "cap is"),
            ("too_long", "cap is"),
        ):
            with self.subTest(fault=fault):
                tokenizer = BoundaryTokenizer(self.case)
                recipient = tokenizer.texts["recipient"]
                if fault == "retokenized_prefix":
                    tokenizer.encodings[recipient + " A"] = [3, 5, 7, 9, 12, 19]
                elif fault == "two_answer_tokens":
                    tokenizer.encodings[recipient + " A"] += [21]
                elif fault == "same_answer_token":
                    tokenizer.set_prompt("recipient", [3, 5, 7, 9, 11], {"A": 19, "B": 19})
                elif fault == "different_answer_ids":
                    tokenizer.set_prompt("effect_donor", [4, 6, 8, 10, 11], {"A": 21, "B": 20})
                elif fault == "unequal_prompt_lengths":
                    tokenizer.set_prompt("sham_donor", [12, 14, 16, 18, 11, 13])
                elif fault == "collapsed_prompt":
                    tokenizer.set_prompt("sham_donor", [3, 5, 7, 9, 11])
                elif fault == "too_short":
                    tokenizer.set_prompt("recipient", [11])
                cap = 4 if fault == "too_long" else 128
                with self.assertRaisesRegex(ValueError, expected):
                    run_gym.tokenize_case(tokenizer, self.case, max_tokens=cap)

    def test_forward_and_wall_time_limits_precede_model_execution(self):
        for maximum, seconds in ((0, 1), (-1, 1), (True, 1), (1.5, 1),
                                 (1, 0), (1, -1), (1, float("nan")), (1, float("inf"))):
            with self.subTest(maximum=maximum, seconds=seconds):
                with self.assertRaises(ValueError):
                    run_gym.ForwardBudget(maximum, seconds)
        with mock.patch.object(run_gym.time, "monotonic", return_value=100.0) as clock:
            self.assertFalse(run_gym.ForwardBudget(7, 10).admit_case())
            self.assertTrue(run_gym.ForwardBudget(8, 10).admit_case())
            backend = mock.Mock()
            budget = run_gym.ForwardBudget(1, 10)
            budget.run(backend, [3, 5])
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                budget.run(backend, [3, 5])
            self.assertEqual(budget.used, 1)
            self.assertEqual(backend.run.call_count, 1)
            timed = run_gym.ForwardBudget(16, 10)
            clock.return_value = 109.999
            self.assertTrue(timed.admit_case())
            clock.return_value = 110.0
            self.assertFalse(timed.admit_case())
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                timed.run(backend, [3, 5])
            self.assertEqual(timed.used, 0)
            self.assertEqual(backend.run.call_count, 1)

    def test_incomplete_saved_record_is_reported(self):
        with tempfile.TemporaryDirectory(prefix="bpd-col3050-record-") as directory:
            path = Path(directory) / "records.jsonl"
            self.assertEqual(run_gym._load_records(path), [])
            run_gym._append_record(path, {"case_id": self.case["case_id"], "status": "error"})
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"case_id":')
            with self.assertRaisesRegex(ValueError, "Incomplete record at line 2"):
                run_gym._load_records(path)

    def test_changed_scorer_hash_cannot_replace_saved_summary(self):
        manifest = cases.generate_manifest(max_roots=1, tier="trace")
        with tempfile.TemporaryDirectory(prefix="bpd-col3050-frozen-") as directory:
            output = Path(directory)
            run_gym.write_json(output / "cases_manifest.json", manifest)
            original_summary = {"existing_evidence": "preserve until matching source is restored"}
            run_gym.write_json(output / "summary.json", original_summary)
            for changed_source in ("cases.py", "scoring.py"):
                with self.subTest(changed_source=changed_source):
                    contract = frozen_score_contract(manifest)
                    contract["source_sha256"][changed_source] = "0" * 64
                    run_gym.write_json(output / "run_contract.json", contract)
                    with self.assertRaisesRegex(ValueError, "Frozen scorer source changed"):
                        run_gym.command_score(SimpleNamespace(run=output))
                    self.assertEqual(json.loads((output / "summary.json").read_text()), original_summary)


class RecordingBackend:
    """Observe real streamed native calls while retaining their small results."""

    def __init__(self, backend, fail_at=None):
        self.backend, self.fail_at = backend, fail_at
        self.calls, self.results = [], []

    def run(self, ids, **kwargs):
        self.calls.append((list(ids), kwargs))
        if self.fail_at == len(self.calls):
            raise OSError("Injected native-forward interruption")
        result = self.backend.run(ids, **kwargs)
        self.results.append(result)
        if not self.backend.layers_released():
            raise AssertionError("A streamed decoder survived its native forward")
        return result

    def score_hidden(self, *args, **kwargs):
        return self.backend.score_hidden(*args, **kwargs)


class RunnerNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the verified-source fixture setup without inheriting its tests.
        native_fixture.StreamingNativeTests.setUpClass.__func__(cls)
        cls.manifest = cases.generate_manifest(max_roots=2, tier="trace")
        cls.case = cls.manifest["cases"][1]
        cls.tokenized = run_gym.tokenize_case(BoundaryTokenizer(cls.case), cls.case)

    @classmethod
    def tearDownClass(cls):
        native_fixture.StreamingNativeTests.tearDownClass.__func__(cls)

    def backend(self):
        return streaming.Backend._from_fixture(copy.deepcopy(self.native), self.snapshot)

    def test_eight_native_forwards_preserve_readout_and_persist_scorable_evidence(self):
        torch = self.torch
        from safetensors.torch import load_file, save_file

        budget = run_gym.ForwardBudget(8, 600)
        progress = {}
        with self.backend() as backend:
            observed = RecordingBackend(backend)
            self.assertTrue(budget.admit_case())
            record, tensors = run_gym.execute_case(
                observed, self.tokenized, self.case, budget, progress=progress)
            self.assertEqual(budget.used, 8)
            self.assertFalse(budget.admit_case())
            self.assertEqual(len(observed.results), 8)
            self.assertEqual(progress["active_stage"], "completed")
            self.assertEqual(set(record["clean"]), set(cases.ROLES))
            self.assertEqual(tuple(record["arms"]), interventions.ARM_NAMES)
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["input_tokens"], 5)
            self.assertEqual(record["answer_token_ids"], {"A": 19, "B": 20})
            for index, (ids, kwargs) in enumerate(observed.calls):
                expected_role = cases.ROLES[index] if index < 3 else "recipient"
                self.assertEqual(ids, self.tokenized[expected_role]["input_ids"])
                self.assertEqual(kwargs["capture_prefix"], index < 3)
                if index >= 3:
                    self.assertEqual(kwargs["intervention"].name, interventions.ARM_NAMES[index - 3])
            for result in observed.results:
                self.assertEqual(result.receipt["native_layer_visits"],
                                 [[0, 0], [0, 1], [1, 0], [1, 1]])
                self.assertTrue(result.receipt["decoder_parameters_released_to_meta"])
            recipient = observed.results[0]
            protected = torch.stack([recipient.boundary_last_cpu[0]] + [
                result.boundary_last_cpu[0] for result in observed.results[3:]
            ])
            self.assertTrue(torch.equal(protected, protected[0].expand_as(protected)))
            # Keep the one-vector GEMM shape identical to production. Different
            # row tiles in an artificial batch can round equal rows differently.
            full_readout = torch.stack([backend.score_hidden(vector)[0] for vector in protected])
            self.assertEqual(tuple(full_readout.shape), (6, self.native.config.vocab_size))
            self.assertTrue(torch.equal(full_readout, full_readout[0].expand_as(full_readout)))
            self.assertTrue(torch.equal(recipient.final_hidden_cpu, observed.results[3].final_hidden_cpu))
            self.assertEqual(record["clean"]["recipient"]["probabilities"],
                             record["arms"]["identity"]["probabilities"])
            self.assertGreater((recipient.final_hidden_cpu - observed.results[-1].final_hidden_cpu)
                               .abs().max().item(), 1e-7)
            for name, arm in record["arms"].items():
                self.assertEqual(arm["patch_calls"], 1)
                self.assertEqual(arm["protected_readout_max_abs"], 0.0)
                self.assertTrue(arm["admission"], name)
                self.assertAlmostEqual(sum(arm["probabilities"].values()), 1.0, places=14)
            matched_doses = [record["arms"][name]["physical_dose"]
                             for name in scoring.MATCHED_ARMS]
            self.assertGreater(min(matched_doses), 0)
            self.assertLessEqual(max(matched_doses) - min(matched_doses),
                                 1e-6 + .02 * max(matched_doses))

        expected_tensors = {role + suffix for role in cases.ROLES
                            for suffix in (".pass1_boundary", ".final_query")}
        expected_tensors.update("arm." + name + ".final_query" for name in interventions.ARM_NAMES)
        self.assertEqual(set(tensors), expected_tensors)
        with tempfile.TemporaryDirectory(prefix="bpd-col3050-evidence-") as directory:
            output = Path(directory)
            activation = output / "activations" / (self.case["case_id"] + ".safetensors")
            activation.parent.mkdir()
            save_file(tensors, str(activation))
            record["activations"] = {"path": str(activation.relative_to(output)),
                                     "sha256": streaming.sha256_file(activation)}
            run_gym.write_json(output / "cases_manifest.json", self.manifest)
            run_gym.write_json(output / "run_contract.json", frozen_score_contract(self.manifest))
            run_gym._append_record(output / "records.jsonl", record)
            restored_records = run_gym._load_records(output / "records.jsonl")
            self.assertEqual(restored_records, [record])
            restored_tensors = load_file(str(activation))
            self.assertEqual(set(restored_tensors), expected_tensors)
            for name, value in tensors.items():
                self.assertTrue(torch.equal(restored_tensors[name], value), name)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                run_gym.command_score(SimpleNamespace(run=output))
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(json.loads(stdout.getvalue()), summary)
            self.assertEqual(summary, scoring.summarize(restored_records, self.manifest))
            self.assertEqual(summary["schema"], scoring.SCHEMA)
            self.assertEqual(summary["verdict"], "PILOT_ONLY")
            self.assertEqual(summary["n_units"], 2)
            self.assertEqual(summary["n_cases"], 2)
            self.assertEqual(summary["n_attempted_cases"], 1)
            self.assertEqual(summary["actual_completed_comparisons"], 5)
            self.assertEqual(summary["instrument"]["max_protected_readout_abs"], 0.0)
            self.assertEqual(summary["instrument"]["max_identity_probability_error"], 0.0)
            self.assertEqual(summary["case_status_counts"].get("NOT_ATTEMPTED"), 1)
            for name in interventions.ARM_NAMES:
                self.assertEqual(summary["arm_admission"][name], {"admitted": 1, "rejected": 0})

    def test_interrupted_case_retains_partial_scores_without_completed_comparisons(self):
        budget = run_gym.ForwardBudget(8, 600)
        progress = {}
        with self.backend() as backend:
            observed = RecordingBackend(backend, fail_at=5)
            with self.assertRaisesRegex(OSError, "Injected native-forward interruption"):
                run_gym.execute_case(observed, self.tokenized, self.case, budget, progress=progress)
            self.assertEqual(budget.used, 5)
            self.assertEqual(len(observed.results), 4)
            self.assertTrue(backend.layers_released())
        self.assertEqual(progress["active_stage"], "arm.effect")
        self.assertEqual(set(progress["clean"]), set(cases.ROLES))
        self.assertEqual(set(progress["arms"]), {"identity"})
        error_record = {"case_id": self.case["case_id"], "root_id": self.case["root_id"],
                        "status": "error", "reason": "OSError: Injected native-forward interruption",
                        "partial_progress": progress}
        restored = json.loads(json.dumps(error_record, allow_nan=False))
        summary = scoring.summarize([restored], self.manifest)
        self.assertEqual(summary["actual_completed_comparisons"], 0)
        self.assertEqual(summary["n_eligible_cases"], 0)
        self.assertEqual(summary["case_status_counts"], {"ERROR": 1, "NOT_ATTEMPTED": 1})


class RealPinnedTokenizerTests(unittest.TestCase):
    def test_local_pinned_tokenizer_accepts_arithmetic_and_trace_triplets(self):
        source = os.environ.get("BPD_COL3050_SOURCE_DIR")
        if not source:
            self.skipTest("Set BPD_COL3050_SOURCE_DIR for real-tokenizer coverage")
        source = Path(source)
        files = ("config.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model",
                 "special_tokens_map.json", "added_tokens.json")
        if any(not (source / name).is_file() for name in files):
            self.skipTest("Pinned tokenizer metadata is absent; architecture-only source is sufficient for native tests")
        for name in files:
            self.assertEqual(streaming.sha256_file(source / name), streaming.SOURCE_SHA256[name], name)
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            self.skipTest(f"Tokenizer dependencies unavailable: {error}")
        tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True,
                                                  trust_remote_code=False, use_fast=True)
        shared_answers = None
        for tier in cases.TIERS:
            manifest = cases.generate_manifest(max_roots=2, horizons=(1, 2),
                                                cases_per_root=2, tier=tier)
            for case in manifest["cases"]:
                with self.subTest(case=case["case_id"]):
                    variants = run_gym.tokenize_case(tokenizer, case)
                    for variant in variants.values():
                        self.assertLessEqual(len(variant["input_ids"]), 128)
                        if shared_answers is None:
                            shared_answers = variant["answer_token_ids"]
                        self.assertEqual(variant["answer_token_ids"], shared_answers)
                        self.assertEqual(len(set(shared_answers.values())), 2)


if __name__ == "__main__":
    unittest.main()
