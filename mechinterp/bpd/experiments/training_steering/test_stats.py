"""Focused checks for independent RP-02 denominator and pairing semantics."""
from __future__ import annotations

import copy
from fractions import Fraction
import json
import unittest

if __package__:
    from .stats import ARMS, exact_sign_flip, summarize
else:
    from stats import ARMS, exact_sign_flip, summarize


def record(bits, *, causal=False, targeted=None):
    bits = list(bits)
    value = {"labels": [0] * len(bits), "predictions": [0 if bit else 1 for bit in bits],
             "correct": bits}
    if causal:
        value["targeted"] = list(targeted) if targeted is not None else [i % 2 == 0 for i in range(len(bits))]
    return value


def root_result(root, a, b, c=None):
    bits = {"A_task": list(a), "B_causal": list(b), "C_rotated": list(a if c is None else c)}
    return {"root": root, "arms": {
        arm: {"eval": {"iid": record([1, 0, 1, 0]), "long": record(bits[arm])},
              "causal": record([1, 0, 1, 1], causal=True)}
        for arm in ARMS
    }}


class StatisticsTests(unittest.TestCase):
    def summarize(self, rows, **kwargs):
        return summarize(rows, bootstrap_draws=256, **kwargs)

    def test_root_means_and_pooled_denominators_are_distinct(self):
        rows = [root_result(3, [0] * 4, [1] * 4),
                root_result(9, [1] * 12, [1] * 6 + [0] * 6)]
        result = self.summarize(rows)
        primary = result["contrasts"][0]
        self.assertEqual(primary["macro_effect"], 0.25)
        self.assertEqual(primary["pooled_effect"], -0.125)
        self.assertEqual(primary["recovery"], 4)
        self.assertEqual(primary["corruption"], 6)
        self.assertEqual(primary["n_units"], 2)
        self.assertEqual(primary["n_cases"], 16)
        self.assertEqual(result["arms"]["B_causal"]["long"]["macro_accuracy"], 0.75)
        self.assertEqual(result["arms"]["B_causal"]["long"]["pooled_accuracy"], 0.625)

    def test_zero_effects_keep_every_root_and_return_p_one(self):
        rows = [root_result(i, [1, 0, 1], [1, 0, 1]) for i in range(8)]
        result = self.summarize(rows)
        for contrast in result["contrasts"]:
            self.assertEqual(contrast["sign_flip"]["p_value"], 1.0)
            self.assertEqual(contrast["sign_flip"]["total_assignments"], 256)
            self.assertEqual(contrast["sign_flip"]["n_zero_effect_units"], 8)
            self.assertEqual(contrast["bootstrap_95"]["lower"], 0.0)
            self.assertEqual(contrast["bootstrap_95"]["upper"], 0.0)
            self.assertFalse(contrast["positive_development_signal"])
        self.assertEqual(result["n_units"], 8)
        self.assertTrue(result["protocol_root_count_complete"])

    def test_positive_significance_keeps_development_verdict(self):
        rows = [root_result(i, [0, 0], [1, 1]) for i in range(8)]
        result = self.summarize(rows)
        for contrast in result["contrasts"]:
            self.assertEqual(contrast["sign_flip"]["p_fraction"], "1/128")
            self.assertEqual(contrast["holm_adjusted_p_fraction"], "1/64")
            self.assertTrue(contrast["positive_development_signal"])
            self.assertFalse(contrast["confirmation_claim"])
            self.assertEqual(contrast["scientific_verdict"], "DEVELOPMENT_ONLY")
        self.assertEqual(result["scientific_verdict"], "DEVELOPMENT_ONLY")
        self.assertEqual(result["planned_comparisons"], 2)

    def test_pairwise_recovery_and_corruption_survive_net_tie(self):
        result = self.summarize([root_result(1, [0, 1, 1, 0], [1, 0, 1, 0])])
        primary = result["contrasts"][0]
        self.assertEqual(primary["recovery"], 1)
        self.assertEqual(primary["corruption"], 1)
        self.assertEqual(primary["net_recovery"], 0)
        self.assertEqual(primary["macro_effect"], 0.0)

    def test_exact_sign_flip_uses_rational_ties(self):
        test = exact_sign_flip([Fraction(1, 3), Fraction(-1, 6), Fraction(-1, 6)])
        self.assertEqual(test["p_value"], 1.0)
        test = exact_sign_flip([Fraction(1, 3), Fraction(1, 6)])
        self.assertEqual(test["p_fraction"], "1/2")

    def test_duplicate_roots_are_rejected(self):
        value = root_result(4, [0, 1], [1, 1])
        with self.assertRaisesRegex(ValueError, "Duplicate root"):
            self.summarize([value, copy.deepcopy(value)])

    def test_label_mismatch_is_rejected_even_with_consistent_bits(self):
        value = root_result(4, [1, 1], [1, 1])
        altered = value["arms"]["B_causal"]["eval"]["long"]
        altered["labels"][0] = 1
        altered["predictions"][0] = 1
        with self.assertRaisesRegex(ValueError, "labels/order/length differ"):
            self.summarize([value])

    def test_false_correctness_bit_is_rejected(self):
        value = root_result(4, [0, 1], [1, 1])
        value["arms"]["B_causal"]["eval"]["long"]["correct"][0] = 0
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.summarize([value])

    def test_lengths_and_nonbinary_bits_are_rejected(self):
        value = root_result(4, [0, 1], [1, 1])
        value["arms"]["B_causal"]["eval"]["iid"]["correct"].append(0)
        with self.assertRaisesRegex(ValueError, "equal-length"):
            self.summarize([value])
        value = root_result(4, [0, 1], [1, 1])
        value["arms"]["C_rotated"]["causal"]["correct"][0] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            self.summarize([value])

    def test_causal_mask_alignment_is_required(self):
        value = root_result(4, [0, 1], [1, 1])
        value["arms"]["C_rotated"]["causal"]["targeted"][0] = False
        with self.assertRaisesRegex(ValueError, "targeted masks differ"):
            self.summarize([value])

    def test_causal_subgroup_denominators_and_absent_subgroup(self):
        value = root_result(4, [0, 1], [1, 1])
        result = self.summarize([value])
        causal = result["arms"]["A_task"]["causal"]
        self.assertEqual(causal["all"]["pooled_accuracy"], 0.75)
        self.assertEqual(causal["targeted"]["n_cases"], 2)
        self.assertEqual(causal["targeted"]["pooled_accuracy"], 1.0)
        self.assertEqual(causal["untargeted"]["pooled_accuracy"], 0.5)
        for arm in ARMS:
            value["arms"][arm]["causal"]["targeted"] = [False] * 4
        result = self.summarize([value])
        causal = result["arms"]["A_task"]["causal"]
        self.assertEqual(causal["targeted"]["n_units"], 1)
        self.assertEqual(causal["targeted"]["n_units_with_cases"], 0)
        self.assertIsNone(causal["targeted"]["macro_accuracy"])
        self.assertIsNone(causal["targeted"]["pooled_accuracy"])
        json.dumps(result, allow_nan=False)

    def test_seeded_bootstrap_and_input_order_are_deterministic(self):
        rows = [root_result(8, [1, 0], [1, 1]), root_result(2, [1, 1], [0, 1])]
        original = copy.deepcopy(rows)
        first = self.summarize(rows, bootstrap_seed=7)
        second = self.summarize(list(reversed(rows)), bootstrap_seed=7)
        self.assertEqual(first, second)
        self.assertEqual(rows, original)
        self.assertEqual(first["roots"], [2, 8])

    def test_empty_roots_and_invalid_bootstrap_budget_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least one root"):
            self.summarize([])
        with self.assertRaisesRegex(ValueError, "must be positive"):
            summarize([root_result(1, [0], [1])], bootstrap_draws=0)
        with self.assertRaisesRegex(ValueError, "at most 18 roots"):
            self.summarize([root_result(i, [0], [1]) for i in range(19)])


if __name__ == "__main__":
    unittest.main()
