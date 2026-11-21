"""Source-only routing decomposition test, not trained-model evidence."""

import math
import unittest

try:
    from mechinterp.bpd.experiments.inkling_reader_v0.reader_probe import (
        coordinate_patch, norm_matched_random_patch, reader_gap, rmsnorm,
    )
except ModuleNotFoundError:
    from reader_probe import coordinate_patch, norm_matched_random_patch, reader_gap, rmsnorm


class ReaderProbeTest(unittest.TestCase):
    def test_rms_and_score_decomposition(self):
        normalized = rmsnorm([2.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(v * v for v in normalized), 4.0, places=5)
        scored = reader_gap([1.0, 0.0], [1.0, 0.0], [0.0, 1.0])
        self.assertAlmostEqual(sum(scored["coordinate_contributions"]), scored["score_gap"])
        self.assertLess(scored["block3_weight"], 0.5)

    def test_coordinate_intervention_recomputes_normalizer(self):
        original = reader_gap([1.0, 1.0], [1.0, 0.0], [0.0, 1.0])
        patch = coordinate_patch([1.0, 1.0], [1.0, 0.0], [0.0, 1.0],
                                 coordinate=0, delta=5.0)
        self.assertEqual(patch["actual_raw_displacement_l2"], 5.0)
        self.assertNotEqual(original["score_gap"], patch["after"]["score_gap"])
        self.assertEqual(original, patch["before"])

    def test_sham_is_dose_matched_and_deterministic(self):
        args = ([1.0, 0.5, 0.2], [0.1, 0.4, 0.0], [0.5, 0.0, 0.1])
        a = norm_matched_random_patch(*args, dose=0.3, seed=71)
        b = norm_matched_random_patch(*args, dose=0.3, seed=71)
        self.assertAlmostEqual(a["actual_raw_displacement_l2"], 0.3)
        self.assertEqual(a["change_in_block3_weight"], b["change_in_block3_weight"])
        self.assertTrue(math.isfinite(a["change_in_block3_weight"]))

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            reader_gap([1.0, 2.0], [1.0], [0.0, 1.0])
        with self.assertRaises(ValueError):
            coordinate_patch([1.0], [1.0], [1.0], coordinate=1, delta=1.0)
        with self.assertRaises(ValueError):
            norm_matched_random_patch([1.0], [1.0], [1.0], dose=-1, seed=2)


if __name__ == "__main__":
    unittest.main()
