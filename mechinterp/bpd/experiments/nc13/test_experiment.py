"""Source-only checks for NC13 lineage, pairing, and endpoint isolation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import experiment as ex


class SourceControls(unittest.TestCase):
    def test_invalid_root_rejected_before_initialization_or_lock(self):
        with patch.object(ex,"DenseLooper") as model:
            for root in (1300,2601316,2601298,True,2601300.0):
                with self.assertRaises(ValueError):
                    ex.train_task(root,"A")
            model.assert_not_called()

    def test_existing_receipt_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/f"root_{ex.SMOKE}_choices.npz"
            path.write_bytes(b"original archive")
            with patch(ex.__package__+".verify_source.verify_source",return_value="test-lock"):
                with patch.object(ex,"train_task") as train:
                    with self.assertRaises(FileExistsError):
                        ex.run(ex.SMOKE,directory)
                    train.assert_not_called()
            self.assertEqual(path.read_bytes(),b"original archive")

    def test_shared_gradient_alignment_recovers_identity(self):
        rng = torch.Generator().manual_seed(913031)
        gradients = torch.randn(8,4,20,generator=rng,dtype=torch.float64)
        gradients = gradients/gradients.norm(dim=-1,keepdim=True)
        identity = torch.eye(20,dtype=torch.float64).expand(4,-1,-1)
        r,diag,spectrum = ex.regularized_alignment(gradients,gradients,identity)
        self.assertTrue(diag["admitted"])
        self.assertEqual(spectrum.shape,(4,20))
        self.assertLess(float((r-identity).abs().max()),1e-12)
        self.assertTrue(all(v>=1 for v in diag["condition_number_by_pass"]))
        with self.assertRaises(ValueError):
            ex.regularized_alignment(gradients[:7],gradients[:7],identity)

    def test_rank_admission_rejects_zero_gradient_rows(self):
        rng = torch.Generator().manual_seed(913032)
        g = torch.randn(40,4,20,generator=rng,dtype=torch.float64)
        g[0,0] = 0
        p,_,diag,_ = ex._rank_four(g)
        self.assertIsNone(p)
        self.assertEqual(diag["zero_gradient_rows"],1)

    def test_failed_sham_preserves_context_endpoint(self):
        names = ex.ARM_NAMES + ("B_clean",)
        scores = {name:{"by_context":{"shared":{"final_nll_nats":.3}}} for name in names}
        scores["mixed"]["by_context"]["shared"]["final_nll_nats"] = .4
        admissions = {name:{"admitted":True,"reasons":[]} for name in names}
        admissions["sham_D_7"] = {"admitted":False,"reasons":["NO_REGISTERED_BRACKET"]}
        out = ex._contrast_values(scores,admissions)
        self.assertTrue(out["context_gain_E"]["admitted"])
        self.assertAlmostEqual(out["context_gain_E"]["value"],.1)
        self.assertFalse(out["median_sham_minus_shared_D"]["admitted"])
        self.assertEqual(out["median_sham_minus_shared_D"]["value"],0)
        del scores["sham_D_7"]
        self.assertIsNone(ex._contrast_values(scores,admissions)["median_sham_minus_shared_D"]["value"])

    def test_correctness_transitions_and_displacement_are_contextual(self):
        x = torch.zeros(4,5)
        x[:,4] = torch.tensor([1.,1.,-1.,-1.])
        labels = torch.tensor([True,False,True,False])
        clean = torch.tensor([[1.,1.,1.,1.],[1.,1.,1.,1.],[-1.,-1.,-1.,-1.],[-1.,-1.,-1.,-1.]])
        changed = torch.tensor([[-1.,-1.,-1.,-1.],[-1.,-1.,-1.,-1.],[1.,1.,1.,1.],[-1.,-1.,-1.,-1.]])
        clean_w = torch.zeros(4,4,20)
        changed_w = torch.ones_like(clean_w)
        result = ex._score(clean,changed,clean_w,changed_w,x,labels)
        self.assertEqual(result["correct_to_wrong"],1)
        self.assertEqual(result["wrong_to_correct"],2)
        self.assertEqual(result["by_context"]["shared"]["correct_to_wrong"],1)
        self.assertEqual(result["by_context"]["specific"]["wrong_to_correct"],1)
        self.assertEqual(result["delivered_write_displacement"]["rms"],1.)
        self.assertEqual(result["delivered_write_displacement"]["rms_by_pass"],[1.]*4)

    def test_failed_target_projector_preserves_context_and_sham_endpoints(self):
        names = ex.ARM_NAMES + ("B_clean",)
        scores = {name:{"by_context":{"shared":{"final_nll_nats":.3}}} for name in names}
        admissions = {name:{"admitted":True,"reasons":[]} for name in names}
        for name in ("target_E","target_D"):
            scores.pop(name)
            admissions[name] = {"admitted":False,"reasons":["target_rank_or_zero_gradient"]}
        effects = ex._contrast_values(scores,admissions)
        for name in ("context_gain_E","context_gain_D","median_sham_minus_shared_D","shared_D_minus_clean"):
            self.assertTrue(effects[name]["admitted"])
        for name in ("shared_gap_reduction_D","target_minus_shared_D"):
            self.assertFalse(effects[name]["admitted"])
            self.assertIsNone(effects[name]["value"])


if __name__ == "__main__":
    unittest.main()
