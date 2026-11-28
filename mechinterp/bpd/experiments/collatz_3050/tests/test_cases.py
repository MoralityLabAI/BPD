"""Independent transition and prompt-contract checks for public cases."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cases


def independent_trace(d, n, horizon):
    trace = [n]
    for _ in range(horizon):
        n = 3 * n + d if n % 4 == 1 else n + d
        while n % 2 == 0:
            n //= 2
        trace.append(n)
    return trace


class CaseTests(unittest.TestCase):
    def test_deterministic_smoke_and_anchor_semantics(self):
        manifest = cases.generate_manifest()
        self.assertEqual(manifest, cases.generate_manifest())
        self.assertEqual(manifest["root_ids"], [13, 73, 17, 25])
        self.assertEqual(manifest["n_cases"], 4)
        self.assertEqual([c["recipient"]["label"] for c in manifest["cases"]],
                         ["A", "B", "A", "B"])
        d73 = next(c for c in manifest["cases"] if c["d"] == 73)
        self.assertEqual([d73[r]["n"] for r in cases.ROLES], [1, 9, 37])
        self.assertTrue(d73["all_same_basin"])
        self.assertFalse(manifest["cases"][0]["all_same_basin"])
        cases.validate_manifest(manifest)

    def test_full_development_population_with_independent_oracle(self):
        manifest = cases.generate_manifest(32, (1, 2), cases_per_root=2)
        self.assertEqual(manifest["n_roots"], 32)
        self.assertEqual(len({c["case_id"] for c in manifest["cases"]}), manifest["n_cases"])
        for case in manifest["cases"]:
            self.assertEqual(case["root_id"], case["d"])
            for role in cases.ROLES:
                entry = case[role]
                actual = independent_trace(case["d"], entry["n"], case["horizon"])
                self.assertEqual(entry["trace"], actual)
                final_branch = 3 if actual[-1] % 4 == 1 else 1
                self.assertEqual(entry["future_branch"], final_branch)
                self.assertEqual(entry["label"], case["label_map"][str(final_branch)])
                self.assertIn(f"start={entry['n']:06d}", entry["prompt"])
                self.assertNotIn("trace=", entry["prompt"])
            self.assertNotEqual(case["recipient"]["label"], case["effect_donor"]["label"])
            self.assertEqual(case["recipient"]["label"], case["sham_donor"]["label"])
            if case["d"] != 13:
                self.assertTrue(case["all_same_basin"])

    def test_trace_tier_is_explicit_and_oracle_complete(self):
        manifest = cases.generate_manifest(4, (2,), tier="trace")
        self.assertEqual(manifest["specification"]["tier"], "trace")
        for case in manifest["cases"]:
            for role in cases.ROLES:
                entry = case[role]
                self.assertIn("trace=" + " -> ".join(f"{n:06d}" for n in entry["trace"]),
                              entry["prompt"])
                self.assertNotIn("future_branch", entry["prompt"])
                self.assertNotIn("basin", entry["prompt"])

    def test_strict_inputs_and_manifest_tampering(self):
        for kwargs in ({"max_roots": True}, {"max_roots": 0}, {"max_roots": 33},
                       {"horizons": (True,)}, {"horizons": (1, 1)},
                       {"horizons": (3,)}, {"tier": "heldout"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cases.generate_manifest(**kwargs)
        manifest = cases.generate_manifest()
        changed = deepcopy(manifest)
        changed["cases"][0]["recipient"]["label"] = "B"
        with self.assertRaises(ValueError):
            cases.validate_manifest(changed)
        changed = deepcopy(manifest)
        changed["cases"].append(changed["cases"][0])
        with self.assertRaises(ValueError):
            cases.validate_manifest(changed)


if __name__ == "__main__":
    unittest.main()
