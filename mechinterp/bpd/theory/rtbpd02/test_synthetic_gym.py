"""Independent root reduction and physical closure for the registered gym."""
import json
from pathlib import Path
from statistics import fmean
import unittest

import numpy as np

from synthetic_gym import build, make_roots, reader_maps, run


class SyntheticGymTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name("EXTENSION_REGISTRATION.json")
        cls.registration = json.loads(path.read_bytes())
        cls.summary, cls.roots, cls.evalmath = run(cls.registration)

    def test_primary_independent_reduction_and_analytic_reader(self):
        g = self.registration["synthetic"]
        gym = build(g)
        roots = make_roots(g, gym, 1, g["development_roots"], g["initial_reach_scale"])
        initial = roots @ gym.basis
        tied_multiplier = np.array(g["background_diagonal"]) + g["recurrent_write_scale"] * np.array(g["spectrum"])
        pass_two = initial * tied_multiplier
        analytic = (pass_two[..., 2:4] * np.array(g["spectrum"][2:4])) @ np.asarray(g["delayed_reader_mix"]).T
        expected = np.mean(np.sum(np.square(analytic), axis=-1), axis=-1)
        rows = [r for r in self.roots if (r["pass"], r["panel"]) == (2, "aligned")]
        self.assertEqual(len(rows), 64)
        self.assertEqual(len({r["root_id"] for r in rows}), 64)
        svd, reader = [[r["methods"][k]["future_mse"] for r in rows] for k in ("svd", "reader")]
        np.testing.assert_allclose(svd, expected, atol=1e-11)
        np.testing.assert_allclose(reader, np.zeros_like(expected), atol=1e-20)
        reduced = fmean(a - b for a, b in zip(svd, reader, strict=True))
        self.assertGreater(reduced, 1.)
        self.assertAlmostEqual(reduced, self.summary["primary_paired_svd_minus_reader"]["mean"])
        self.assertEqual(self.summary["primary_paired_svd_minus_reader"]["positive_roots"], 64)
        self.assertEqual(self.summary["delayed_dark_rank"], 2)

    def test_shift_reverses_write_advantage_and_roots_are_paired(self):
        data = self.summary["panels"]
        aligned, shifted = (data[panel]["2"]["methods"] for panel in ("aligned", "shifted"))
        self.assertLess(aligned["reach"]["write_mse"], aligned["svd"]["write_mse"])
        self.assertGreater(shifted["reach"]["write_mse"], shifted["svd"]["write_mse"])
        groups = {
            (p, b): {r["root_id"] for r in self.roots if (r["pass"], r["panel"]) == (p, b)}
            for p in (1, 2) for b in ("aligned", "shifted")
        }
        self.assertEqual(len(groups[1, "aligned"]), 64)
        self.assertTrue(all(g == groups[1, "aligned"] for g in groups.values()))

    def test_physical_dose_closure_and_evalmath_inputs(self):
        for phase in self.summary["fit_diagnostics"].values():
            self.assertEqual(set(phase["column_rank"].values()), {2})
            self.assertAlmostEqual(phase["reach_fit_loss"], phase["reach_spectral_tail"])
            self.assertAlmostEqual(phase["reader_fit_loss"], phase["reader_spectral_tail"])
        for row in self.roots:
            methods = row["methods"]
            self.assertTrue(all(rec["physical_status"] == "matched" for rec in methods.values()))
            self.assertTrue(all(abs(rec["physical_dose"] - rec["requested_dose"]) < 1e-11 for rec in methods.values()))
            self.assertTrue(all(rec["reader_closure_max_abs"] < 1e-11 for rec in methods.values()))
            self.assertEqual(methods["svd"]["future_direction"]["status"], "zero_observed")
            self.assertAlmostEqual(methods["reader"]["future_direction"]["cosine"], 1.)
        self.assertFalse(self.evalmath["heldout_sealed"])
        self.assertEqual(len(self.evalmath["pilot"]), 64)
        for row in self.evalmath["pilot"]:
            self.assertAlmostEqual(float(row["paired_effect"]), .5)
            self.assertEqual(len(row["sham_effects"]), 1)

    def test_stage_to_memory_is_the_delayed_transport(self):
        spec = self.registration["synthetic"]
        gym = build(spec)
        now, later = reader_maps(gym)
        write = gym.basis[:, 2]
        np.testing.assert_allclose(now @ write, [0., 0.], atol=1e-14)
        before = gym.reaches(make_roots(spec, gym, 1, 3, spec["initial_reach_scale"]), [2])[2][0, 0]
        np.testing.assert_allclose(gym.future_effect(before, write), [1., -.125], atol=1e-14)
        np.testing.assert_allclose(later @ write, [1., -.125], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
