"""Protocol, leakage, factor-residual and budget tests for COG-04."""
import copy
import math
import unittest

import numpy as np
import torch
from torch import nn

from mechinterp.bpd.components import PassMaskSchedule, TiedComponentLinear
from .experiment import FactorProtocol, Scorer, public_features, rollout
from .transport import HEADER, action_reply, decode, encode, pack_codes, unpack_codes
from .vendor import ledger


class TransportTests(unittest.TestCase):
    def test_every_code_and_padding_roundtrip(self):
        for bits in (2, 4, 8):
            all_codes = np.arange(1 << bits)
            np.testing.assert_equal(unpack_codes(pack_codes(all_codes, bits), len(all_codes), bits), all_codes)
            for n in range(1, 18):
                codes = np.arange(n) % (1 << bits)
                payload = pack_codes(codes, bits)
                self.assertEqual(len(payload), math.ceil(n * bits / 8))
                np.testing.assert_equal(unpack_codes(payload, n, bits), codes)

    def test_quantizer_error_bound(self):
        rng = np.random.default_rng(7)
        scales = np.array([0.2, 1.5, 4], np.float32)
        z = rng.uniform(-1, 1, (13, 3)).astype(np.float32) * scales
        for bits in (2, 4, 8):
            msg = encode(z, scales, bits)
            read = decode(msg.payload, scales, expected_rank=3)
            self.assertEqual(msg.clipped_values, 0)
            self.assertTrue(np.all(np.abs(read - z) <= scales / ((1 << bits) - 1) + 1e-6))
            self.assertEqual(msg.nbits, 8 * (HEADER.size + math.ceil(z.size * bits / 8)))

    def test_float32_roundtrip_and_reply_charge(self):
        z = np.arange(15, dtype=np.float32).reshape(5, 3) / 17
        msg = encode(z, np.ones(3), 32)
        np.testing.assert_equal(decode(msg.payload, np.ones(3), expected_rank=3), z)
        self.assertEqual(msg.nbits, 56 + z.size * 32)
        self.assertEqual(len(action_reply(4, 5)), 2)
        with self.assertRaises(ValueError):
            action_reply(5, 5)

    def test_clipping_and_malformed_messages(self):
        scales = np.array([1, 2], np.float32)
        msg = encode(np.array([[3, -4]], np.float32), scales, 2)
        self.assertEqual(msg.clipped_values, 2)
        np.testing.assert_equal(decode(msg.payload, scales, expected_rank=2), [[1, -2]])
        for payload in (msg.payload[:-1], msg.payload + b"\0", b"x"):
            with self.assertRaises(ValueError):
                decode(payload, scales, expected_rank=2)
        with self.assertRaises(ValueError):
            decode(msg.payload, scales, expected_rank=3)
        with self.assertRaises(ValueError):
            unpack_codes(bytes([1]), 1, 2)
        with self.assertRaises(ValueError):
            encode(np.array([[np.nan]], np.float32), np.ones(1), 8)


class FactorTests(unittest.TestCase):
    def test_full_factor_native_parity_and_low_rank_has_no_residual_bypass(self):
        torch.manual_seed(19)
        base = nn.Linear(5, 7)
        x = torch.randn(1, 11, 5)
        full = TiedComponentLinear(base, 5)
        rank1 = TiedComponentLinear(base, 1)
        for component in (full, rank1):
            component.mode = "component"
            component.schedule = PassMaskSchedule(torch.ones(1, 1, 11, component.C), torch.zeros(1, 1, 11))
            result = component(x)
            manual = component.component_acts(x) @ component.U + component.bias
            torch.testing.assert_close(result, manual, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(full.component_acts(x) @ full.U + full.bias, base(x), atol=2e-6, rtol=2e-6)
        self.assertGreater(float((rank1.component_acts(x) @ rank1.U + rank1.bias - base(x)).norm()), 0.1)

    def test_identity_reference_has_exact_dense_logits(self):
        torch.manual_seed(1701)
        model = Scorer(12)
        x = np.random.default_rng(4).normal(size=(70, 12)).astype(np.float32)
        arm = FactorProtocol(model, np.zeros(12, np.float32), np.ones(12, np.float32), x, 12, 32, identity_factor=True)
        scores, info = arm.scores(x[:13], np.random.default_rng(0))
        direct = np.tanh(x[:13] @ model.input.weight.detach().numpy().T + model.input.bias.detach().numpy()) @ model.head.weight.detach().numpy().ravel() + float(model.head.bias.detach()[0])
        np.testing.assert_equal(scores, direct)
        self.assertEqual(info["logit_mse"], 0)

    def test_argmax_certificate_and_setup_size(self):
        torch.manual_seed(9)
        model = Scorer(12)
        rng = np.random.default_rng(10)
        train = rng.normal(size=(90, 12)).astype(np.float32)
        for rank in (1, 4, 12):
            arm = FactorProtocol(model, np.zeros(12, np.float32), np.ones(12, np.float32), train, rank, 4)
            self.assertGreater(arm.setup_bytes, 0)
            for _ in range(5):
                # The scorer asserts the soundness of every issued certificate.
                scores, info = arm.scores(rng.normal(size=(5, 12)).astype(np.float32), rng)
                self.assertEqual(len(scores), 5)
                self.assertGreaterEqual(info["logit_error_bound_max"], 0)


class LeakageAndBudgetTests(unittest.TestCase):
    def test_candidate_inputs_ignore_metadata_and_outcomes(self):
        task = ledger.generate_tasks(2, 713)[0]
        known = ledger.initial_state(task)
        changed = copy.deepcopy(task)
        changed.update(root_id="secret", split="ood_test", family="secret", labels={"useful": [True]})
        changed["provenance"]["depth"] = -999
        np.testing.assert_equal(public_features(task, known), public_features(changed, known))

    def test_split_roots_and_presentations_are_disjoint(self):
        tasks = ledger.generate_tasks(4, 819)
        self.assertEqual(len({x["root_id"] for x in tasks}), len(tasks))
        self.assertEqual(len({x["task_id"] for x in tasks}), len(tasks))

    def test_budget_and_public_no_repeat_support(self):
        for task in ledger.generate_tasks(2, 112):
            for name in ("uniform_random", "public_structural_heuristic"):
                row, decisions = rollout(task, None, name, budget=3)
                self.assertLessEqual(row["verifier_calls"], 3)
                self.assertEqual(row["verifier_calls"], len(decisions))
                known, rejected = ledger.initial_state(task), set()
                for decision in decisions:
                    action = decision["action"]
                    self.assertNotIn(action, rejected)
                    self.assertNotIn(task["world"]["rules"][action]["conclusion"], known)
                    rec = ledger.check_step(task, known, action)
                    self.assertEqual(rec["accepted"], decision["accepted"])
                    if rec["new_fact"] is not None:
                        rejected.clear()
                    elif not rec["accepted"]:
                        rejected.add(action)
                    known = rec["next_known"]


if __name__ == "__main__":
    unittest.main()
