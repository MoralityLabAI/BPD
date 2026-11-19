"""Native-mini training regressions using synthetic engineering fixtures only."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import unittest
from unittest import mock

import torch
from torch import nn

from mechinterp.bpd.adapters.trm import build_mini_trm
from ..worlds import sudoku4
from .resources import BoundedTensorCache, MemoryBudget
from . import training as training


class TrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.rng_state = torch.random.get_rng_state()
        torch.manual_seed(81)
        self.budget = MemoryBudget(max_host_rss_gib=12.0, min_host_available_gib=0.1)

    def tearDown(self):
        torch.random.set_rng_state(self.rng_state)

    def model(self):
        return build_mini_trm(**training.MODEL_CONFIGS["mini"], seed=424242).eval()

    def factor_kwargs(self, **overrides):
        return {
            "steps": 2, "components": 8, "top_k": 4,
            "logical_batch": 4, "microbatch": 2,
            "seed": 4242, "budget": self.budget, **overrides,
        }

    def test_uneven_blank_counts_preserve_whole_batch_loss_and_gradients(self):
        generator = torch.Generator().manual_seed(81)
        tokens = torch.randint(2, 6, (4, 16), generator=generator)
        targets = torch.randint(0, 4, (4, 16), generator=generator)
        for row, blanks in enumerate((1, 3, 7, 14)):
            tokens[row, :blanks] = 1
        reference = self.model().train()
        full = copy.deepcopy(reference)
        split = copy.deepcopy(reference)

        # Compute the declared whole-batch objective from the public fit rollout.
        logits, _ = training.fit_rollout(reference, tokens)
        expanded_targets = targets[:, None, :].expand(-1, 3, -1)
        per = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 4), expanded_targets.reshape(-1), reduction="none"
        ).reshape(4, 3, 16)
        blank = tokens.eq(1)[:, None, :]
        expected_loss = (
            (per * blank).sum() / (3 * tokens.eq(1).sum())
            + 0.25 * (per * ~blank).sum() / (3 * tokens.ne(1).sum())
        )
        expected_loss.backward()
        full_loss = training.backward_supervised_batch(full, tokens, targets, microbatch=4)
        split_loss = training.backward_supervised_batch(split, tokens, targets, microbatch=1)
        self.assertAlmostEqual(full_loss, float(expected_loss.detach()), places=6)
        self.assertAlmostEqual(split_loss, full_loss, places=6)
        for candidate in (full, split):
            for (name, expected), (other_name, actual) in zip(
                reference.named_parameters(), candidate.named_parameters()
            ):
                self.assertEqual(name, other_name)
                if expected.grad is None:
                    self.assertIsNone(actual.grad)
                else:
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-5, atol=1e-6)

    def test_optimizer_and_ema_update_once_per_logical_batch(self):
        cases = [sudoku4(860_200 + i, "engineering", blanks=4 + i) for i in range(3)]
        config = training.TrainConfig(steps=2, logical_batch=4, microbatch=1, ema_decay=0.5)
        optimizer_updates = []
        ema_snapshots = []
        initial_shadows = []
        adamw = torch.optim.AdamW
        ema_type = training.ExponentialMovingAverage

        class CountingAdamW(adamw):
            def step(self, closure=None):
                optimizer_updates.append(True)
                return super().step(closure)

        class RecordingEMA(ema_type):
            def __init__(self, model, decay):
                super().__init__(model, decay)
                initial_shadows.append({k: v.clone() for k, v in self.shadow.items()})

            def update(self, model):
                ema_snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})
                super().update(model)

        with mock.patch.object(torch.optim, "AdamW", CountingAdamW), \
                mock.patch.object(training, "ExponentialMovingAverage", RecordingEMA), \
                redirect_stdout(io.StringIO()):
            raw, ema, receipt = training.train_sudoku_model(
                4242, cases, config=config, device="cpu", budget=self.budget, profile="mini"
            )
        self.assertEqual(len(optimizer_updates), 2)
        self.assertEqual(len(ema_snapshots), 2)
        self.assertEqual(receipt["ema_updates"], 2)
        self.assertEqual(receipt["updates"], 2)
        self.assertEqual(receipt["fit_depths"], [2, 4, 6])
        self.assertFalse(raw.training)
        self.assertFalse(ema.training)
        for key, result in ema.state_dict().items():
            if result.is_floating_point():
                expected = (0.25 * initial_shadows[0][key]
                            + 0.25 * ema_snapshots[0][key] + 0.5 * ema_snapshots[1][key])
                torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)

    def test_ema_copies_integer_buffers_and_returns_an_independent_model(self):
        model = nn.Linear(2, 2, bias=False)
        model.register_buffer("counter", torch.tensor(0))
        with torch.no_grad():
            model.weight.fill_(1)
        ema = training.ExponentialMovingAverage(model, 0.5)
        with torch.no_grad():
            model.weight.fill_(3)
            model.counter.fill_(7)
        ema.update(model)
        result = ema.copy_model(model)
        torch.testing.assert_close(result.weight, torch.full((2, 2), 2.0))
        self.assertEqual(int(result.counter), 7)
        with torch.no_grad():
            result.weight.fill_(99)
        torch.testing.assert_close(model.weight, torch.full((2, 2), 3.0))
        torch.testing.assert_close(ema.shadow["weight"], torch.full((2, 2), 2.0))

    def test_cache_parity_frozen_teacher_and_train_only_fit_depths(self):
        teacher = self.model()
        tokens = torch.randint(1, 6, (8, 16), generator=torch.Generator().manual_seed(20))
        allowed_rows = {tuple(row.tolist()) for row in tokens}
        before = training.model_digest(teacher)
        original_rollout = training.fit_rollout
        seen_teacher_rows = []

        def checked_rollout(model, batch):
            if model is teacher:
                for row in batch:
                    digits = tuple(row.tolist())
                    self.assertIn(digits, allowed_rows)
                    seen_teacher_rows.append(digits)
                with mock.patch.object(teacher, "act_step", wraps=teacher.act_step) as steps:
                    output = original_rollout(model, batch)
                self.assertEqual(steps.call_count, 6)
                self.assertEqual(output[0].shape[1], 3)
                return output
            return original_rollout(model, batch)

        cache = BoundedTensorCache(8 * 1024**2)
        with mock.patch.object(training, "fit_rollout", side_effect=checked_rollout):
            cached, cached_receipt = training.fit_online_factors(
                teacher, tokens, cache=cache, **self.factor_kwargs()
            )
            uncached, uncached_receipt = training.fit_online_factors(
                teacher, tokens, cache=BoundedTensorCache(0), **self.factor_kwargs()
            )
        self.assertTrue(seen_teacher_rows)
        self.assertTrue(cached_receipt["target_cache_admitted"])
        self.assertFalse(uncached_receipt["target_cache_admitted"])
        self.assertGreater(cache.hits, 0)
        self.assertGreater(cache.recomputations_avoided, 0)
        torch.testing.assert_close(
            torch.tensor(cached_receipt["losses"], dtype=torch.float64),
            torch.tensor(uncached_receipt["losses"], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertEqual(cached_receipt["train_tokens_sha256"], training.tensor_digest(tokens))
        self.assertEqual(before, training.model_digest(teacher))
        for key, value in cached.state_dict().items():
            if value.is_floating_point():
                torch.testing.assert_close(value, uncached.state_dict()[key], rtol=1e-5, atol=1e-6)
            else:
                self.assertTrue(torch.equal(value, uncached.state_dict()[key]))

    def test_symmetry_pipeline_bitwise_rng_parity_and_checkpoint_gradients(self):
        cases = [sudoku4(860_300 + i, "engineering", blanks=4 + (i % 3)) for i in range(6)]
        givens = torch.tensor([c.givens for c in cases], dtype=torch.long)
        solutions = torch.tensor([c.solution for c in cases], dtype=torch.long)
        gen_ref = torch.Generator(device="cpu").manual_seed(9991)
        gen_opt = torch.Generator(device="cpu").manual_seed(9991)
        cache = BoundedTensorCache(16 * 1024**2)
        pipe = training.SudokuSymmetryPipeline(
            givens, solutions, 8, "cpu", budget=self.budget, cache=cache,
        )
        for step_idx in range(4):
            bx_ref, by_ref = training.sample_augmented_batch(givens, solutions, 8, gen_ref)
            bx_opt, by_opt = pipe.next_batch(step_idx, 4, gen_opt)
            self.assertTrue(torch.equal(bx_ref, bx_opt))
            self.assertTrue(torch.equal(by_ref, by_opt))
            pipe.mark_batch_consumed(step_idx)
        self.assertTrue(pipe.telemetry()["base_admitted_to_cache"])

        # Selective activation checkpointing preserves exact loss and gradients
        m_plain = self.model().train()
        m_ckpt = copy.deepcopy(m_plain).train()
        l_plain = training.backward_supervised_batch(m_plain, bx_ref, by_ref, microbatch=4, checkpoint_act=False)
        l_ckpt = training.backward_supervised_batch(m_ckpt, bx_ref, by_ref, microbatch=4, checkpoint_act=True)
        self.assertAlmostEqual(l_plain, l_ckpt, places=6)
        for (_, p1), (_, p2) in zip(m_plain.named_parameters(), m_ckpt.named_parameters()):
            if p1.grad is not None:
                torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-6, atol=1e-6)

    def test_optimized_training_and_factor_fit_cpu_parity(self):
        cases = [sudoku4(860_350 + i, "engineering", blanks=4 + (i % 3)) for i in range(6)]
        cfg = training.TrainConfig(steps=3, logical_batch=8, microbatch=4, ema_decay=0.9)
        with redirect_stdout(io.StringIO()):
            raw_r, ema_r, rec_r = training.train_sudoku_model(
                4301, cases, config=cfg, device="cpu", budget=self.budget,
                profile="mini", execution_mode="reference",
            )
            raw_o, ema_o, rec_o = training.train_sudoku_model(
                4301, cases, config=cfg, device="cpu", budget=self.budget,
                profile="mini", execution_mode="optimized",
            )
        torch.testing.assert_close(
            torch.tensor([r["loss"] for r in rec_r["trace"]]),
            torch.tensor([r["loss"] for r in rec_o["trace"]]),
            rtol=1e-5, atol=1e-6,
        )
        for p1, p2 in zip(ema_r.parameters(), ema_o.parameters()):
            torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-6)
        tokens = torch.randint(1, 6, (8, 16), generator=torch.Generator().manual_seed(21))
        s_ref, f_ref = training.fit_online_factors(
            ema_r, tokens, cache=BoundedTensorCache(4 * 1024**2),
            execution_mode="reference", **self.factor_kwargs(),
        )
        s_opt, f_opt = training.fit_online_factors(
            ema_r, tokens, cache=BoundedTensorCache(4 * 1024**2),
            execution_mode="optimized", **self.factor_kwargs(),
        )
        torch.testing.assert_close(
            torch.tensor(f_ref["losses"]), torch.tensor(f_opt["losses"]), rtol=1e-5, atol=1e-6,
        )
        for p1, p2 in zip(s_ref.parameters(), s_opt.parameters()):
            torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-6)
        _ = raw_r, raw_o

    def test_factor_training_rejects_invalid_batches_before_allocation(self):
        teacher = self.model()
        tokens = torch.ones((4, 16), dtype=torch.long)
        for logical, micro in ((0, 2), (-4, 2), (4, 0), (4, -2), (3, 2), (True, 1), (4, 1.0)):
            with self.subTest(logical_batch=logical, microbatch=micro), \
                    mock.patch.object(training.copy, "deepcopy", side_effect=AssertionError("allocation before validation")):
                with self.assertRaises(ValueError):
                    training.fit_online_factors(
                        teacher, tokens, cache=BoundedTensorCache(0),
                        **self.factor_kwargs(logical_batch=logical, microbatch=micro),
                    )

    def test_factor_training_rejects_malformed_or_out_of_vocabulary_tokens(self):
        teacher = self.model()
        for tokens in (
            torch.ones((4, 16), dtype=torch.float32), torch.ones((4, 15), dtype=torch.long),
            torch.empty((0, 16), dtype=torch.long), torch.full((4, 16), 11, dtype=torch.long),
            torch.full((4, 16), -1, dtype=torch.long),
        ):
            with self.subTest(shape=tuple(tokens.shape), dtype=tokens.dtype):
                with self.assertRaises(ValueError):
                    training.fit_online_factors(
                        teacher, tokens, cache=BoundedTensorCache(0), **self.factor_kwargs()
                    )

    def test_scalar_tensor_digest_binds_shape_dtype_and_value(self):
        values = (torch.tensor(1.0), torch.tensor(2.0), torch.tensor(1), torch.tensor([1.0]))
        digests = [training.tensor_digest(value) for value in values]
        self.assertEqual(len(set(digests)), len(values))
        for value, digest in zip(values, digests):
            self.assertEqual(digest, training.tensor_digest(value.clone()))
            self.assertEqual(len(digest), 64)

    def test_factor_cache_key_binds_recurrence_recipe(self):
        teacher = self.model()
        changed = copy.deepcopy(teacher)
        changed.l_cycles += 1
        self.assertEqual(training.model_digest(teacher), training.model_digest(changed))
        tokens = torch.ones((2, 16), dtype=torch.long)
        _, original = training.fit_online_factors(
            teacher, tokens, cache=BoundedTensorCache(0), **self.factor_kwargs(steps=0)
        )
        _, altered = training.fit_online_factors(
            changed, tokens, cache=BoundedTensorCache(0), **self.factor_kwargs(steps=0)
        )
        self.assertNotEqual(original["teacher_recipe"], altered["teacher_recipe"])
        self.assertNotEqual(original["target_cache_key"], altered["target_cache_key"])

    def test_bf16_training_is_rejected_on_cpu(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            training.autocast_context(torch.device("cpu"), "bf16")


if __name__ == "__main__":
    unittest.main()
