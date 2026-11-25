"""Focused checks for search budgets, data boundaries, and physical execution."""
from dataclasses import replace
import unittest

import torch

from .experiment import (ARMS, Deployment, Features, Recipe, ZERO_CODE, features,
                         fit, proposals, quantize_three_bit, retain, search)
from ..inkling_routing.native_model import make_batch


class SearchTests(unittest.TestCase):
    def test_fixed_budget_and_all_retention_references(self):
        scored = []
        def score(code, candidate_id):
            scored.append((code, candidate_id))
            return sum((c-1)**2 for c in code)
        result = search(score)
        self.assertEqual(len(scored), 496)
        self.assertEqual(result["logical_calls_per_arm"], {a: 172 for a in ARMS})
        known = {row["id"]: row for row in result["candidates"]}
        for row in result["candidates"]:
            if row["parent"] is not None:
                self.assertIn(row["parent"], known)
                allowed = [list(p["code"]) for p in proposals(tuple(known[row["parent"]]["code"]))]
                self.assertIn(row["code"], allowed)
        for retention in result["retentions"]:
            self.assertTrue(set(retention["selected"]).issubset(retention["pool"]))
        self.assertEqual({tuple(r["code"]) for r in result["selected"].values()}, {(1,1,1,1)})

    def test_boundary_slots_stay_charged_and_ties_stay_stable(self):
        rows = proposals((2,-2,2,-2))
        self.assertEqual(len(rows), 9)
        self.assertEqual(sum(r["padded"] for r in rows), 4)
        result = search(lambda code, candidate_id: 0.0)
        for arm in ARMS:
            self.assertEqual(result["selected"][arm]["code"], list(ZERO_CODE))
        pool = [{"code":[0,0,0,0], "score":0.0, "id":"parent"},
                {"code":[1,0,0,0], "score":0.0, "id":"child"}]
        self.assertEqual(retain(pool,1)[0]["id"], "parent")

    def test_quantization_and_label_boundary(self):
        value = torch.tensor([-1., -.1, 0., .4, 1.])
        quant, codes, scale = quantize_three_bit(value)
        self.assertAlmostEqual(scale, 1/3)
        self.assertTrue(torch.equal(quant, codes.float()*scale))
        self.assertTrue(bool((codes.abs() <= 3).all()))
        batch = features(make_batch(17000,8))
        self.assertIsInstance(batch, Features)
        with self.assertRaises(RuntimeError):
            _ = batch.labels


class DeploymentTests(unittest.TestCase):
    def test_actual_core_online_inputs_shared_factors_and_gauge(self):
        # Reduced fixture is test-only, with the allowed old smoke root.
        recipe = replace(Recipe(), source_steps=2, bpd_steps=2)
        model, bpd, trainer, training = fit(17000, recipe)
        self.assertEqual(training["pgd_steps"], 6)
        self.assertEqual(training["pgd_source_shapes"]["mix"], [1,1,1,17])
        deployment = Deployment(model,bpd,trainer,recipe)
        batch = features(make_batch(170000001,8))
        w = bpd.wrappers["mix"]
        v,u = w.V.detach().clone(),w.U.detach().clone()
        with torch.no_grad(), bpd.target_capture():
            clean = model(batch)
        parity,_ = deployment.rollout(batch,all_open=True,delta=1.0)
        self.assertLess(float((parity["intermediate_logits"]-clean["intermediate_logits"]).abs().max()),2e-5)
        base,trace = deployment.rollout(batch,record=True)
        self.assertTrue(torch.equal(trace["pre_input"],trace["ci_input"]))
        self.assertEqual(trace["invocation"][:,0].tolist(),[0,1,2])
        self.assertEqual(len(set(trace["invocation"][:,2].tolist())),1)
        p = torch.arange(15,-1,-1)
        with torch.no_grad():
            w.V.copy_(v[:,p]); w.U.copy_(u[p])
        gauge,_ = deployment.rollout(batch,gate_permutation=p)
        self.assertLess(float((gauge["intermediate_logits"]-base["intermediate_logits"]).abs().max()),2e-5)


if __name__ == "__main__":
    unittest.main()
