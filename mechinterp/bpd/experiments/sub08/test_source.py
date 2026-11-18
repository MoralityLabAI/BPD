"""Substantive pre-run access, readout and checkpoint checks; no training."""
import tempfile
import unittest
from pathlib import Path

import torch

from .source import TiedGRU, tensor_split, implementation_checks, state_digest, train_trajectory


class SourceTests(unittest.TestCase):
    def test_pilot_cannot_score_heldout_or_dose(self):
        for split in ("heldout", "reserved_dose"):
            with self.assertRaises(ValueError):
                tensor_split(split, "design")
        with self.assertRaises(ValueError):
            tensor_split("reserved_dose", "prospective")

    def test_unregistered_roots_rejected_before_training(self):
        with self.assertRaises(ValueError):
            train_trajectory(1800, 48, 2400, "prospective", None)
        with self.assertRaises(ValueError):
            train_trajectory(1900, 48, 3600, "design", None)

    def test_serialized_model_readout_parity(self):
        # Excluded smoke initialization only; zero optimizer updates.
        torch.manual_seed(1889)
        torch.set_num_threads(1)
        model = TiedGRU(48).eval()
        x, _, _ = tensor_split("validation", "smoke")
        with torch.no_grad():
            before, _ = model.rollout(x)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"state_dict": model.state_dict()}, path)
            reloaded = TiedGRU(48).eval()
            reloaded.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
            with torch.no_grad():
                after, _ = reloaded.rollout(x)
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(state_digest(model), state_digest(reloaded))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 7972)
        checks = implementation_checks(model, "smoke")
        self.assertLessEqual(checks["full_width_parity_max"], 1e-6)
        self.assertLessEqual(checks["prefix2_consistency_max"], 1e-6)
        self.assertLessEqual(checks["prefix4_consistency_max"], 1e-6)


if __name__ == "__main__":
    unittest.main()
