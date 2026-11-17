"""Standalone engineering tests. Native parity is checked in the hosted repo."""
from __future__ import annotations
import importlib
import math
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pilot import TopOne, TrainedFactors, run_seed, draw, READ, TRANSPORT

class TrainedObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hard_top1_forward_and_gradient(self):
        x=torch.tensor([[-1.,1.],[1.,-1.],[0.,0.]],requires_grad=True)
        mask=TopOne.apply(x)
        self.assertTrue(bool(((mask==0)|(mask==1)).all()))
        self.assertTrue(bool((mask.sum(-1)==1).all()))
        (mask*torch.tensor([[2.,1.],[2.,1.],[2.,1.]])).sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertGreater(float(x.grad.abs().sum()),0.)

    def test_component_forward_and_target_shapes(self):
        model=TrainedFactors(503)
        x=draw(5,7)
        y,m=model(x)
        self.assertEqual(tuple(y.shape),(7,4,4))
        self.assertEqual(tuple(m.shape),(7,4,2))
        self.assertTrue(bool((m.sum(-1)==1).all()))
        self.assertTrue(torch.equal(x[...,2:],torch.zeros_like(x[...,2:])))

    def test_bpd_readwrite_derivative_exact(self):
        model=TrainedFactors(503)
        x=draw(35,5).requires_grad_()
        y,m=model(x)
        analytical=((x @ model.V)*m)@model.U
        torch.testing.assert_close(y,analytical,atol=0,rtol=0)
        self.assertGreater(float(torch.autograd.grad(y.sum(),model.V)[0].abs().sum()),0.)

    def test_rotation_observer_delay(self):
        v=torch.tensor([1.,0.,0.,0.]);z=torch.zeros(4)
        self.assertEqual(float(v@READ),0.)
        self.assertEqual(float(v@TRANSPORT.T@READ),1.)
        self.assertEqual(float(v@torch.eye(4).T@READ),0.)
        self.assertEqual(float(z@TRANSPORT.T@READ),0.)

    def test_seed_cpu_training_and_pair_controls(self):
        row=run_seed(503,steps=200,heldout=128,pairs=64)
        self.assertEqual(row['four_pass_route'],128)
        self.assertEqual(row['component_observable_rank_now'],1)
        self.assertEqual(row['component_observable_rank_after_one'],2)
        self.assertGreater(row['updated_factors']['V'],0.)
        self.assertGreater(row['updated_factors']['U'],0.)
        self.assertGreater(row['mixed_source_mse'],.10)
        self.assertGreater(row['delayed_median'],.9)
        self.assertLess(row['immediate_median'],.10)
        self.assertEqual(row['transfer']['native'],64)
        self.assertEqual(row['transfer']['projected_native'],64)
        self.assertEqual(row['transfer']['sham'],0)
        self.assertEqual(row['transfer']['visible'],0)
        self.assertLess(row['transfer']['random_fiber'],25)
        self.assertLess(row['largest_norm_matching_error'],1e-5)

    def test_seed_replay_bitwise_deterministic(self):
        a=run_seed(521,steps=200,heldout=64,pairs=24)
        b=run_seed(521,steps=200,heldout=64,pairs=24)
        self.assertEqual(a,b)

    def test_untrained_factor_gate_is_not_positive(self):
        model=TrainedFactors(503)
        x=draw(200503,128)
        y,_=model(x)
        self.assertGreater(float((x-y).square().mean().detach()),.05)

    def test_native_wrapper_parity_when_installed(self):
        try:
            importlib.import_module('mechinterp.bpd.components')
        except ImportError:
            self.skipTest('BPD native module absent in standalone container')
        from pilot import NativeTrainedFactors, native_parity
        model=NativeTrainedFactors(503)
        self.assertLess(native_parity(model,503),1e-6)
        x=draw(333,4)
        y,_=model(x)
        self.assertTrue(bool(torch.isfinite(y).all()))

if __name__=='__main__':unittest.main()
