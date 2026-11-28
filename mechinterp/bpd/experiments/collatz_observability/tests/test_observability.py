"""Independent finite trajectory checks for BPD-COL-01."""

from itertools import combinations
from math import gcd
from pathlib import Path
import sys
import unittest

# The enclosing BPD __init__ imports Torch. Import this independent stdlib
# package directly so these exact checks also run in Python-only environments.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collatz_observability import audit
from collatz_observability.collatz_core import analyze


def original_odd_step(n, d):
    """Independent original unaccelerated map, counting its even stretch."""
    value = 3 * n + d if n % 4 == 1 else n + d
    halvings = 0
    while value % 2 == 0:
        value //= 2
        halvings += 1
    return value, halvings


def future_word(n, d, observation, horizon):
    word = []
    for _ in range(horizon + 1):
        word.append(observation[n])
        n, _ = original_odd_step(n, d)
    return tuple(word)


class ObservabilityTests(unittest.TestCase):
    def test_all_core_successors_and_quotient_conservation(self):
        states = 0
        for d in range(1, 1026, 4):
            if gcd(d, 6) != 1:
                continue
            core = analyze(d)
            q = audit.observer_views(core)["quotient"]
            refinement = audit.refine_observer(core, q)
            self.assertEqual(len(refinement.partitions), 1)
            for n in core.states:
                successor, halvings = original_odd_step(n, d)
                self.assertEqual(core.successors[n], successor)
                self.assertGreaterEqual(halvings, 2)
                self.assertGreaterEqual(successor, 1)
                self.assertLessEqual(successor, d)
                self.assertEqual(q[n], q[successor])
                states += 1
        self.assertEqual(states, 44204)

    def test_refinement_matches_brute_force_future_words(self):
        for d in (1, 5, 13, 17, 29, 37, 49, 73):
            core = analyze(d)
            for mode, observation in audit.observer_views(core).items():
                with self.subTest(d=d, mode=mode):
                    refinement = audit.refine_observer(core, observation)
                    self.assertLess(len(refinement.partitions), len(core.states) + 1)
                    for horizon, labels in enumerate(refinement.partitions):
                        words = {n: future_word(n, d, observation, horizon)
                                 for n in core.states}
                        for left, right in combinations(core.states, 2):
                            self.assertEqual(labels[left] == labels[right],
                                             words[left] == words[right])
                    # N^2 product states bound any delayed difference between
                    # two deterministic finite trajectories, independently of
                    # the refinement algorithm's fixed-point detection.
                    for left, right in combinations(core.states, 2):
                        if refinement.stable[left] == refinement.stable[right]:
                            self.assertEqual(
                                future_word(left, d, observation, len(core.states) ** 2),
                                future_word(right, d, observation, len(core.states) ** 2))

    def test_stable_quotient_closure_and_target_ceiling(self):
        for d in (13, 29, 73, 217, 313, 577, 1001, 1025):
            core = analyze(d)
            for observation in audit.observer_views(core).values():
                refinement = audit.refine_observer(core, observation)
                for n in core.states:
                    self.assertEqual(refinement.transition[refinement.stable[n]],
                                     refinement.stable[core.successors[n]])
                curve = [audit.ceiling(labels, core.basin)
                         for labels in refinement.partitions]
                self.assertEqual(curve, sorted(curve))
                self.assertLessEqual(curve[-1], len(core.states))

    def test_d13_protected_donor_future_readout_and_rank(self):
        fixture = audit.d13_fixture()
        self.assertEqual(fixture["public_pair"][0], fixture["public_pair"][1])
        self.assertEqual(fixture["basin_pair"], [1, 5])
        self.assertEqual(fixture["positive_donor"]["readout_squared_difference_by_horizon"],
                         [0, 2, 0, 2, 0])
        self.assertEqual(fixture["delayed_donor"]["readout_squared_difference_by_horizon"][0:3],
                         [0, 0, 2])
        self.assertEqual(fixture["quotient_only_null"]["readout_squared_difference_by_horizon"],
                         [0, 0, 0, 0, 0])
        self.assertEqual(fixture["one_hot_stacked_observer_ranks"], [3, 4, 5, 6, 6, 6, 6])
        self.assertEqual([len(cells) for cells in fixture["partition_cells_by_horizon"]],
                         [3, 5, 7])
        self.assertEqual(audit.exact_rank([[1, 2, 3], [2, 4, 6], [0, 0, 1]]), 2)
        # Independent explicit columns for W and O certify the null vector.
        delta = [1, -1, 0, 1, -1, 0, 0]
        states = fixture["states"]
        for target in states:
            self.assertEqual(sum(delta[i] for i, n in enumerate(states)
                                 if original_odd_step(n, 13)[0] == target), 0)
        for branch in (1, 3):
            self.assertEqual(sum(delta[i] for i, n in enumerate(states)
                                 if n != 13 and (3 if n % 4 == 1 else 1) == branch), 0)

    def test_equal_dose_same_basin_effect_and_sham(self):
        fixture = audit.control_fixtures()["d73_matched_effect_and_sham"]
        effect, sham = fixture["effect"], fixture["same_future_sham"]
        self.assertEqual(effect["physical_squared_dose"], 2)
        self.assertEqual(sham["physical_squared_dose"], 2)
        self.assertEqual(set(fixture["basins"].values()), {3})
        self.assertEqual(effect["readout_squared_difference_by_horizon"][0:2], [0, 2])
        self.assertEqual(sham["readout_squared_difference_by_horizon"], [0] * 5)
        self.assertTrue(fixture["same_future_sham_certified_by_stable_partition"])
        # Separate direct trajectory proof: both become/stay c=1 at time 1;
        # their paths enter {3,19,23}, all branch multipliers equal one.
        for start in (1, 37):
            state, _ = original_odd_step(start, 73)
            for _ in range(10):
                self.assertIn(state, (3, 19, 23))
                self.assertEqual(state % 4, 3)
                state, _ = original_odd_step(state, 73)

    def test_same_infinite_branch_trace_can_hide_distinct_basins(self):
        fixture = audit.control_fixtures()["d29_same_future_observation_distinct_basins"]
        self.assertEqual(fixture["basins"], [5, 7])
        self.assertEqual(fixture["cycles"], [[5, 11], [7, 9]])
        self.assertTrue(fixture["same_infinite_observer_trace"])
        self.assertEqual([original_odd_step(n, 29)[0] for n in (5, 11, 9, 7)],
                         [11, 5, 7, 9])
        result = audit.audit_root(29)["observers"]["quotient_branch"]
        self.assertGreater(result["stable_distinct_basin_pairs"], 0)
        self.assertLess(result["stable_basin_correct"], 15)

    def test_public_projection_and_strict_validation(self):
        self.assertEqual(audit.public_observation(13, 1),
                         {"d": 13, "quotient": 1, "branch_multiplier": 3})
        self.assertEqual(audit.public_observation(13, 1), audit.public_observation(13, 5))
        for d, n in ((True, 1), (13, True), (13, 2), (13, 15), (9, 1), (0, 1)):
            with self.subTest(d=d, n=n), self.assertRaises(ValueError):
                audit.public_observation(d, n)
        for prefix in (((True, 2),), ((3, True),), ((2, 2),), ((3, 1),)):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                audit.decode_prefix(13, prefix)
        with self.assertRaisesRegex(ValueError, "scoped"):
            audit.audit_root(1029)

    def test_five_step_extra_channel_decodes_every_core_state(self):
        count = 0
        for d in range(1, 1026, 4):
            if gcd(d, 6) != 1:
                continue
            for n in range(1, d + 1, 2):
                prefix = []
                current = n
                for _ in range(5):
                    next_state, a = original_odd_step(current, d)
                    prefix.append((3 if current % 4 == 1 else 1, a))
                    current = next_state
                residue, modulus = audit.decode_prefix(d, tuple(prefix))
                self.assertEqual(residue, n)
                self.assertGreater(modulus, d - 1)
                count += 1
        self.assertEqual(count, 44204)


if __name__ == "__main__":
    unittest.main()
