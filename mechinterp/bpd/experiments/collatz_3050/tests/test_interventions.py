from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interventions


class InterventionTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        generator = torch.Generator().manual_seed(83)
        self.source = torch.randn((1, 12, 32), generator=generator)
        self.effect = self.source + 0.3 * torch.randn(self.source.shape, generator=generator)
        self.sham = self.source + 0.4 * torch.randn(self.source.shape, generator=generator)

    def test_delivered_bfloat16_dose_and_protected_query(self):
        torch = self.torch
        source, effect, sham = [v.bfloat16() for v in (self.source, self.effect, self.sham)]
        arms = interventions.prepare_arms(source, effect, sham, case_id="test")
        for name, patch in arms.items():
            result = patch(source.clone(), 0)
            self.assertTrue(torch.equal(source[:, -1], result[:, -1]))
            actual = float(torch.linalg.vector_norm(result[:, :-1].float() - source[:, :-1].float()))
            self.assertAlmostEqual(actual, patch.physical_dose, places=6)
            self.assertTrue(patch.admission)
            self.assertTrue(torch.equal(patch(result, 1), result))
            if name in ("effect", "sham", "random"):
                self.assertLessEqual(abs(actual - patch.requested_dose), 0.01 * patch.requested_dose)

    def test_identity_and_full_prefix_have_known_endpoints(self):
        torch = self.torch
        arms = interventions.prepare_arms(self.source, self.effect, self.sham, case_id="identity")
        self.assertTrue(torch.equal(arms["identity"](self.source, 0), self.source))
        result = arms["full_donor_prefix"](self.source, 0)
        self.assertTrue(torch.equal(result[:, :-1], self.effect[:, :-1]))
        self.assertTrue(torch.equal(result[:, -1], self.source[:, -1]))

    def test_replay_mismatch_and_repeated_patch_are_rejected(self):
        arms = interventions.prepare_arms(self.source, self.effect, self.sham, case_id="mismatch")
        with self.assertRaisesRegex(RuntimeError, "differs"):
            arms["effect"](self.source + 0.01, 0)
        patch = arms["sham"]
        patch(self.source, 0)
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            patch(self.source, 0)

    def test_fixed_random_direction_and_invalid_sources(self):
        torch = self.torch
        a = interventions.prepare_arms(self.source, self.effect, self.sham, case_id="same")
        b = interventions.prepare_arms(self.source, self.effect, self.sham, case_id="same")
        self.assertTrue(torch.equal(a["random"].replacement_prefix, b["random"].replacement_prefix))
        with self.assertRaises(ValueError):
            interventions.prepare_arms(self.source, self.source, self.sham, case_id="zero")
        with self.assertRaises(ValueError):
            interventions.prepare_arms(self.source, self.effect, self.sham, case_id="nan", dose_fraction=float("nan"))


if __name__ == "__main__":
    unittest.main()
