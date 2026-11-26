"""Independent replay checks using synthetic models and exposed test fixtures."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ..training_steering.data import make_pairs
from ..training_steering_stability.model import ComponentStateMemory
from .audit import (Checks, AuditFailure, DIRECTIONS, ROUTES, manual_directions,
                    manual_gate_identity, manual_rollout, manual_diagnostic_record,
                    manual_pathway_record, nested_close, semantic_checks,
                    audit_snapshot, check_hash_inventory, algebra_checks)
from .interventions import build_directions, native_rollout, pathway_diagnostics


class IndependentReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4219)
        self.model = ComponentStateMemory(width=8, components=8)
        self.batch = make_pairs(92947, 12, root_seed=617, delay=3, split="eval")
        self.groups = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        matrices = []
        for group in self.groups:
            q = torch.linalg.qr(self.model.recurrent.V[:, group].detach(), mode="reduced").Q
            matrices.append(q @ q.T)
        self.fixed = torch.stack(matrices)
        ordinary = self.model.rollout(torch.cat((self.batch.receiver, self.batch.donor, self.batch.hybrid)))
        n = len(self.batch.query)
        self.states = [ordinary["states"][i*n:(i+1)*n, 3] for i in range(3)]
        self.manual, _, _ = manual_directions(*self.states, self.model.recurrent.V,
            self.groups, self.fixed, self.batch.edit_slot)

    def test_direction_construction_matches_native_from_independent_solve(self):
        expected, _ = build_directions(*self.states, self.model.recurrent.V,
            self.model.recurrent.U, self.groups, self.fixed, self.batch.edit_slot)
        for name in DIRECTIONS:
            torch.testing.assert_close(self.manual[name], expected[name], atol=2e-6, rtol=0)

    def test_all_native_paths_replay_without_intervention_import_in_auditor(self):
        state = self.model.state_dict()
        for name in ("hybrid_live_span", "hybrid_dual"):
            delta = self.manual[name]
            for route in ROUTES:
                expected = native_rollout(self.model, self.batch.receiver, delta,
                    self.groups, self.batch.edit_slot, route)
                actual = manual_rollout(state, self.batch.receiver, delta,
                    self.groups, self.batch.edit_slot, route)
                for key in ("logits", "states", "incoming_states", "recurrent_writes"):
                    torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=0)
                self.assertTrue(torch.equal(actual["logits"].argmax(-1), expected["logits"].argmax(-1)))

    def test_full_hybrid_state_has_native_hybrid_continuation(self):
        actual = manual_rollout(self.model.state_dict(), self.batch.receiver,
                                state_replacement=self.states[2])
        expected = self.model.rollout(self.batch.hybrid)
        torch.testing.assert_close(actual["logits"], expected["logits"], atol=2e-6, rtol=0)

    def test_gate_decomposition_is_exact_to_float32_arithmetic(self):
        record = manual_gate_identity(self.model.state_dict(), self.batch.receiver,
                                      self.manual["hybrid_live_span"])
        for key in ("finite_decomposition_error", "carry_only_identity_error", "interaction_identity_error"):
            self.assertLess(record[key], 1e-6)

    def test_factor_and_pathway_receipts_recompute_independently(self):
        _, expected = build_directions(*self.states, self.model.recurrent.V,
            self.model.recurrent.U, self.groups, self.fixed, self.batch.edit_slot)
        _, actual = manual_diagnostic_record(*self.states, self.model.recurrent.V,
            self.model.recurrent.U, self.groups, self.fixed, self.batch.edit_slot)
        checks = Checks()
        nested_close(actual, expected, checks, "factor receipts")
        a = manual_pathway_record(self.model.state_dict(), self.batch.receiver, self.manual["hybrid_dual"])
        b = pathway_diagnostics(self.model, self.batch.receiver, self.manual["hybrid_dual"])
        nested_close(a, b, checks, "pathway receipts")
        self.assertGreater(checks.count, 500)

    def test_checks_reject_corruption(self):
        with self.assertRaises(AuditFailure):
            Checks().close(.1, .2, "tampered scalar")
        with self.assertRaises(AuditFailure):
            Checks().tensor(torch.tensor([1.]), torch.tensor([0.]), "tampered tensor")

    def test_decoded_labels_reject_tampering(self):
        from .data import make_batch
        base, batch, _ = make_batch(617, "smoke")
        semantic_checks(base, batch, 617, Checks())
        batch.labels_h[0] = (batch.labels_h[0]+1)%4
        with self.assertRaises(AuditFailure):
            semantic_checks(base, batch, 617, Checks())

    def test_hash_inventory_rejects_mutation_and_escape(self):
        from .audit import sha256
        with TemporaryDirectory() as temporary:
            folder=Path(temporary)
            path=folder/"receipt.json"
            path.write_text("{}\n")
            original=sha256(path)
            check_hash_inventory(folder,{path.name:original},Checks(),"receipt")
            path.write_text('{"changed":true}\n')
            with self.assertRaises(AuditFailure):
                check_hash_inventory(folder,{path.name:original},Checks(),"receipt")
            with self.assertRaises(AuditFailure):
                check_hash_inventory(folder,{"../receipt.json":original},Checks(),"receipt")

    def test_gauge_identities_allow_reused_and_partially_overlapping_groups(self):
        for groups in (torch.tensor([[0,1],[2,3],[0,1],[6,7]]),
                       torch.tensor([[0,1],[1,2],[0,1],[6,7]])):
            checks=Checks()
            algebra_checks(self.model.recurrent.V,self.model.recurrent.U,
                           self.manual["hybrid_dual"],groups,self.batch.edit_slot,checks)
            self.assertGreater(checks.count,20)

    def test_gauge_checks_reject_repeated_indices_within_a_group(self):
        groups=self.groups.clone()
        groups[0,1]=groups[0,0]
        with self.assertRaisesRegex(AuditFailure,"within a frozen group"):
            algebra_checks(self.model.recurrent.V,self.model.recurrent.U,
                           self.manual["hybrid_dual"],groups,self.batch.edit_slot,Checks())

    def test_complete_synthetic_snapshot_replays_and_rejects_changed_prediction(self):
        from .common import Recipe, dump
        from .data import make_batch
        from .experiment import snapshot
        recipe=Recipe()
        base,batch,metadata=make_batch(617,"smoke",recipe)
        model=ComponentStateMemory(width=32,components=32)
        groups=torch.arange(16).reshape(4,4)
        groups[2]=groups[0]
        fixed=[]
        for group in groups:
            q=torch.linalg.qr(model.recurrent.V.detach()[:,group],mode="reduced").Q
            fixed.append(q@q.T)
        geometry={"groups":groups,"fixed_projectors":torch.stack(fixed),
                  "permutation":torch.arange(31,-1,-1),"signs":torch.where(torch.arange(32)%2==0,1.,-1.)}
        with TemporaryDirectory() as temporary:
            folder=Path(temporary); checkpoint=folder/"shared_warmup.pt"
            torch.save(model.state_dict(),checkpoint)
            record=snapshot(617,"shared_warmup","smoke",recipe,base,batch,metadata,geometry,checkpoint)
            receipt=folder/"shared_warmup.json"; dump(receipt,record)
            checks=Checks()
            actual=audit_snapshot(receipt,617,"shared_warmup","smoke",recipe,base,batch,metadata,geometry,checkpoint,checks)
            self.assertEqual(set(actual),set(record["conditions"]))
            self.assertGreater(checks.count,5000)
            predictions=record["conditions"]["hybrid_dual__outside_write_only"]["predictions"]
            predictions[0]=(predictions[0]+1)%4
            dump(receipt,record)
            with self.assertRaises(AuditFailure):
                audit_snapshot(receipt,617,"shared_warmup","smoke",recipe,base,batch,metadata,geometry,checkpoint,Checks())


if __name__ == "__main__":
    unittest.main()
