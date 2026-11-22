"""Parity against the actual synthetic recurrent reader, not pretrained Inkling."""
from __future__ import annotations

import unittest

import torch

from native_model import TinyCrossLoopRouter, make_batch
from reader_tangent import (
    physical_patch_audit,
    reader_tangent,
    writer_scan,
)


class NativeReaderTangentTests(unittest.TestCase):
    def test_raw_writer_tangent_matches_full_torch_reader_at_hook(self):
        torch.set_num_threads(2)
        torch.manual_seed(73)
        model = TinyCrossLoopRouter().eval()
        batch = make_batch(7373, 16)
        with torch.no_grad():
            clean = model(batch)
            selected = model(batch, pair=(0, 1), dose=0.02,
                             hook_pass=1, mode="selector_only")
            query = model.reader(batch.query)
        admitted = selected["admitted"]
        self.assertTrue(bool(admitted.any()))
        for i in range(batch.bits.shape[0]):
            u = query[i].tolist()
            v2 = selected["clean2"][i].tolist()
            v3 = selected["clean3"][i].tolist()
            patched3 = selected["patched3"][i].tolist()
            calc = reader_tangent(u, v2, v3)
            self.assertAlmostEqual(calc["beta3"], clean["beta3"][1, i].item(),
                                   places=5)
            audit = physical_patch_audit(u, v2, v3, patched3)
            self.assertAlmostEqual(
                calc["beta3"] + audit["actual_beta3_change"],
                selected["beta3"][1, i].item(), places=5
            )
            if admitted[i]:
                self.assertAlmostEqual(audit["physical_dose_l2"], 0.02,
                                       places=5)
                self.assertAlmostEqual(audit["rms_before"],
                                       audit["rms_after"], places=5)
            cols = {
                "column0": model.writer.weight[:, 0].detach().tolist(),
                "column1": model.writer.weight[:, 1].detach().tolist(),
            }
            scan = writer_scan(u, v2, v3, cols)
            self.assertEqual(set(scan), {"column0", "column1"})
            self.assertTrue(all(
                -1.000001 <= d["raw_reader_cosine"] <= 1.000001
                for d in scan.values()
            ))


if __name__ == "__main__":
    unittest.main()
