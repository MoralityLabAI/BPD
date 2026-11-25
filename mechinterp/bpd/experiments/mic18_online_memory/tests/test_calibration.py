"""Truth-table tests of masking, causal query roles, and exact scope."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import alias_census, make_pair, patch, reachable_states, squared_distance
from register import MemoryProgram


class CalibrationTests(unittest.TestCase):
    def test_matched_singletons_and_cross_query_roles(self):
        program = MemoryProgram.build(3)
        pair = make_pair(program, reachable_states(program), 0, 0, 1)
        self.assertTrue(pair["admitted"])
        target = pair["arms"]["target_patch"]
        nuisance = pair["arms"]["nuisance_patch"]
        self.assertEqual(target["squared_displacement"], nuisance["squared_displacement"])
        self.assertEqual(target["squared_displacement"], 1)
        self.assertEqual(target["mask_cardinality"], nuisance["mask_cardinality"])
        self.assertEqual(target["recipient_correct"], (0, 1, 1))
        self.assertEqual(nuisance["recipient_correct"], (1, 0, 1))
        self.assertEqual(target["donor_effect_recovery"], (1, 0, None))
        self.assertEqual(nuisance["donor_effect_recovery"], (0, 1, None))

    def test_full_donor_and_self_sham_endpoints(self):
        program = MemoryProgram.build(4)
        pair = make_pair(program, reachable_states(program), 5, 0, 3)
        full = pair["arms"]["whole_donor"]
        sham = pair["arms"]["self_sham"]
        self.assertEqual(full["squared_displacement"], 2)
        self.assertEqual(full["donor_agreement"], (1, 1, 1, 1))
        self.assertEqual(sum(full["recipient_correct"]), 2)
        self.assertEqual(sham["squared_displacement"], 0)
        self.assertEqual(sham["recipient_correct"], (1, 1, 1, 1))
        self.assertTrue(all(arm["reachable"] for arm in pair["arms"].values()))

    def test_all_finite_pairs_have_exact_dose_and_reachability(self):
        for width in range(2, 5):
            program = MemoryProgram.build(width)
            states = reachable_states(program)
            for word in range(2**width):
                for target in range(width):
                    for nuisance in range(width):
                        if target != nuisance:
                            pair = make_pair(program, states, word, target, nuisance)
                            self.assertTrue(pair["admitted"])
                            self.assertEqual(sum(x != y for x, y in zip(pair["recipient_state"], pair["donor_state"])), 2)
                            self.assertEqual(pair["phase"], width)

    def test_observer_aliases_and_action_distinctions(self):
        for width in range(2, 7):
            program = MemoryProgram.build(width)
            report = alias_census(program, reachable_states(program))
            n = 2**width
            self.assertEqual(report["current_observer_classes"], 1)
            self.assertEqual(report["future_answer_signatures"], n)
            self.assertEqual(report["equal_observer_pairs_with_future_conflict"], n*(n-1)//2)
            self.assertEqual(report["pair_query_disagreements"], width*2**(2*width-2))

    def test_patch_preserves_unselected_coordinates(self):
        self.assertEqual(patch((1, 0, 1), (0, 1, 0), (1,)), (1, 1, 1))
        self.assertEqual(patch((1, 0, 1), (0, 1, 0), ()), (1, 0, 1))
        self.assertEqual(squared_distance((1, 0, 1), (0, 1, 0)), 3)

    def test_bad_masks_and_sources_fail(self):
        for mask in ((0, 0), (-1,), (3,), (True,)):
            with self.assertRaises(ValueError):
                patch((0, 0, 0), (1, 1, 1), mask)
        with self.assertRaises(ValueError):
            patch((0, 0), (1,), (0,))
        with self.assertRaises(ValueError):
            patch((0, 0), (2, 1), (0,))
        program = MemoryProgram.build(2)
        with self.assertRaises(ValueError):
            make_pair(program, reachable_states(program), 0, 0, 0)

    def test_task_labels_remain_independent_of_a_damaged_readout(self):
        program = MemoryProgram.build(3)
        head = replace(program.head, w_out=((0, 0, 0),))
        damaged = replace(program, head=head)
        pair = make_pair(damaged, reachable_states(damaged), 4, 0, 1)
        self.assertFalse(pair["admitted"])
        self.assertIn("base_task_error", pair["rejection_reasons"])
        self.assertEqual(pair["recipient_targets"], (1, 0, 0))
        self.assertEqual(pair["arms"]["clean"]["recipient_correct"], (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
