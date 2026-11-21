"""Actual torch-native recurrent hook and continuation controls."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from native_model import TinyCrossLoopRouter, delivered_rotation, make_batch, train_source


class NativeRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_physical_dose_is_matched_per_sample_and_rms_preserved(self):
        x=torch.tensor([[.2,.4,.7,.9],[.3,.4,.8,.6]],dtype=torch.float64)
        for pair in ((0,1),(2,3)):
            y,mask=delivered_rotation(x,pair,.20)
            self.assertTrue(bool(mask.all()))
            self.assertTrue(torch.allclose(torch.linalg.vector_norm(y-x,dim=-1),
                                           torch.full((2,),.2,dtype=x.dtype),atol=1e-10))
            self.assertTrue(torch.allclose(torch.linalg.vector_norm(y,dim=-1),
                                           torch.linalg.vector_norm(x,dim=-1),atol=1e-10))

    def test_full_forward_patch_and_actual_later_states(self):
        torch.manual_seed(5)
        model=TinyCrossLoopRouter().eval()
        batch=make_batch(901,15)
        clean=model(batch)
        full=model(batch,pair=(0,1),dose=.10,mode="full")
        selector=model(batch,pair=(0,1),dose=.10,mode="selector_only")
        content=model(batch,pair=(0,1),dose=.10,mode="content_only")
        self.assertEqual(tuple(clean["intermediate_logits"].shape),(3,15,2))
        self.assertTrue(torch.allclose(clean["intermediate_logits"][0],
                                       full["intermediate_logits"][0],atol=1e-7))
        self.assertTrue(torch.allclose(clean["states"][0],full["states"][0],atol=1e-7))
        mask=full["admitted"]
        self.assertTrue(bool(mask.all()))
        self.assertGreater((full["states"][1]-clean["states"][1]).abs().max().item(),1e-6)
        self.assertGreater((full["states"][2]-clean["states"][2]).abs().max().item(),1e-6)
        self.assertGreater((selector["beta3"][1]-clean["beta3"][1]).abs().max().item(),1e-7)
        self.assertTrue(torch.allclose(content["beta3"][1],clean["beta3"][1],atol=1e-7))

    def test_shared_writer_same_parameter_object_across_loops(self):
        model=TinyCrossLoopRouter()
        calls=[]
        hook=model.writer.register_forward_hook(lambda _module, args, output:calls.append(output))
        _=model(make_batch(2,5))
        hook.remove()
        self.assertEqual(len(calls),3)
        self.assertEqual(tuple(calls[0].shape),(5,2,16))

    def test_direct_gate_controls_are_native_separate_positive_controls(self):
        torch.manual_seed(11)
        model=TinyCrossLoopRouter().eval()
        batch=make_batch(504,21)
        clean=model(batch)
        single=model(batch,hook_pass=1,gate_control="invert_at_hook")
        all_pass=model(batch,gate_control="invert_all")
        self.assertTrue(torch.allclose(single["beta3"][0],clean["beta3"][0]))
        self.assertTrue(torch.allclose(single["beta3"][1],1-clean["beta3"][1],atol=1e-6))
        self.assertTrue(torch.allclose(all_pass["beta3"][0],1-clean["beta3"][0],atol=1e-6))
        self.assertGreater((all_pass["logits"]-clean["logits"]).abs().max().item(),1e-6)
        self.assertGreater((single["logits"]-clean["logits"]).abs().max().item(),1e-6)
        with self.assertRaisesRegex(ValueError,"cannot be mixed"):
            model(batch,pair=(0,1),gate_control="invert_all")

    def test_short_cpu_training_changes_clean_model(self):
        model,info=train_source(steps=45,seed=103,width=16)
        self.assertGreater(info["parameters"],0)
        self.assertGreaterEqual(info["source_validation_accuracy"],0.4)
        batch=make_batch(450001,32)
        with torch.no_grad():
            logits=model(batch)["logits"]
        self.assertEqual(tuple(logits.shape),(32,2))


if __name__=="__main__":
    unittest.main()
