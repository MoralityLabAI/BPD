import copy
import unittest
from score import ARMS, score, upper_binomial


def fixture():
    roster, rows = [], []
    for i in range(64):
        for context in ("initial", "challenge"):
            for action in ("deploy", "hold"):
                case = dict(root=str(i), case=context+action,
                            context=context, expected=action)
                roster.append(case)
                for arm in ARMS:
                    rows.append(dict(case, arm=arm, loops=2, site=95,
                                     status="ok", before=-2,
                                     after=2 if arm=="action_sae" else -2,
                                     dose=0 if arm=="same_value" else .05,
                                     radius_error=0, clean_head="clean",
                                     head="clean" if arm=="same_value" else "edited"))
    return roster, rows


class ScoreTests(unittest.TestCase):
    def test_positive(self):
        roster, rows = fixture()
        result = score(roster, rows)
        self.assertTrue(result["fixed_prefix_statistical_gate"])
        self.assertEqual(result["rows"], 1792)

    def test_sham(self):
        self.assertAlmostEqual(upper_binomial(0, 64), 1-.05**(1/64))
        self.assertGreater(upper_binomial(1, 64), .05)

    def test_reject(self):
        roster, rows = fixture()
        for change in ("missing", "duplicate", "dose", "nan", "noop", "failed", "baseline"):
            with self.subTest(change=change):
                bad = copy.deepcopy(rows)
                if change == "missing":
                    bad.pop()
                elif change == "duplicate":
                    bad.append(bad[0])
                elif change == "dose":
                    bad[1]["dose"] = .1
                elif change == "nan":
                    bad[1]["after"] = float("nan")
                elif change == "noop":
                    bad[0]["head"] = "wrong"
                elif change == "failed":
                    bad[1]["status"] = "failed"
                elif change == "baseline":
                    bad[1]["before"] = -3
                with self.assertRaises(ValueError):
                    score(roster, bad)


if __name__ == "__main__":
    unittest.main()
