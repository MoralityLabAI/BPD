"""Finite source tests for native BPD component locality and pathway effects."""
from __future__ import annotations

import json
import unittest

import torch

from mechinterp.bpd.experiments.training_steering.data import make_pairs
from mechinterp.bpd.experiments.training_steering_stability.model import ComponentStateMemory
from .interventions import (
    DIRECTIONS, ROUTES, build_directions, direction_diagnostics,
    native_rollout, pathway_diagnostics,
)


def projectors(v, groups):
    result = []
    for group in groups:
        q = torch.linalg.qr(v.double()[:, group], mode="reduced").Q
        result.append(q @ q.T)
    return torch.stack(result).float()


class DirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(40401)
        self.v = torch.eye(8) + .24 * torch.tril(torch.randn(8, 8), diagonal=-1)
        self.u = torch.randn(8, 24) * .3
        self.groups = torch.arange(8).reshape(4, 2)
        self.slots = torch.arange(12).remainder(4)
        self.states = [torch.randn(12, 8) for _ in range(3)]
        self.fixed = projectors(self.v, self.groups)

    def build(self, v=None, u=None, states=None):
        return build_directions(*(states or self.states), self.v if v is None else v,
                                self.u if u is None else u, self.groups, self.fixed, self.slots)

    def test_all_six_directions_deliver_physical_state_dose(self):
        directions, diagnostics = self.build()
        self.assertEqual(tuple(directions), DIRECTIONS)
        for name, delta in directions.items():
            self.assertEqual(delta.dtype, self.states[0].dtype)
            torch.testing.assert_close(delta.norm(dim=-1), torch.full((12,), .75),
                                       atol=1e-7, rtol=2e-7)
            self.assertTrue(all(diagnostics["directions"][name]["valid"]))
            self.assertLess(diagnostics["directions"][name]["dense_write_max_abs_error"], 1e-12)
            self.assertLess(diagnostics["directions"][name]["group_write_sum_max_abs_error"], 1e-12)
        json.dumps(diagnostics, allow_nan=False)

    def test_dual_isolates_reads_and_pays_the_exact_extra_norm(self):
        directions, diagnostics = self.build()
        mask = torch.zeros(12, 8).scatter(1, self.groups[self.slots], 1.)
        for source in ("donor", "hybrid"):
            name = source + "_dual"
            record = diagnostics["directions"][name]
            reads = directions[name] @ self.v
            self.assertLess(float((reads * (1-mask)).abs().max()), 1e-6)
            self.assertLess(max(record["raw_read_residual_relative"]), 1e-12)
            self.assertLess(max(record["normalized_read_residual_relative"]), 1e-6)
            self.assertLess(max(record["pythagorean_relative_residual"]), 1e-12)
            self.assertTrue(all(x >= 1-1e-12 for x in record["dual_to_live_span_norm_ratio"]))
            self.assertTrue(all(0 < x <= 1+1e-12 for x in record["selected_read_amplitude_vs_live_span"]))
        span_reads = directions["hybrid_live_span"] @ self.v
        self.assertGreater(float((span_reads * (1-mask)).norm()), .1)

    def test_rank_deficient_dual_records_infeasible_targets(self):
        v = self.v.clone()
        v[:, -1] = v[:, 0]
        _, diagnostic = self.build(v=v)
        self.assertEqual(diagnostic["factor"]["rank"], 7)
        self.assertIsNone(diagnostic["factor"]["condition_number"])
        self.assertGreater(max(x for x in diagnostic["directions"]["hybrid_dual"]
                               ["raw_read_residual_relative"] if x is not None), .1)
        json.dumps(diagnostic, allow_nan=False)

    def test_zero_displacements_remain_zero_and_undefined_ratios_are_null(self):
        state = self.states[0]
        directions, diagnostic = self.build(states=[state, state, state])
        for name, delta in directions.items():
            self.assertEqual(torch.count_nonzero(delta), 0)
            record = diagnostic["directions"][name]
            self.assertFalse(any(record["valid"]))
            self.assertTrue(all(x is None for x in record["grouped_write_leakage"]))
            self.assertTrue(all(x is None for x in record["grouped_write_cosine"]))
        json.dumps(diagnostic, allow_nan=False)

    def test_inverse_column_row_rescaling_preserves_physical_atom_writes(self):
        scales = torch.tensor([.5, -2., 1.4, -.8, 1.8, -.6, .7, -1.5])
        before, db = self.build()
        after, da = self.build(v=self.v*scales, u=self.u/scales[:, None])
        for name in DIRECTIONS:
            torch.testing.assert_close(before[name], after[name], atol=2e-7, rtol=2e-6)
            for metric in ("grouped_write_leakage", "atom_energy_leakage",
                           "atom_energy_by_component_mean"):
                torch.testing.assert_close(torch.tensor(db["directions"][name][metric]),
                                           torch.tensor(da["directions"][name][metric]),
                                           atol=1e-6, rtol=1e-5)

    def test_block_mixing_preserves_group_writes_while_atoms_can_redistribute(self):
        change = torch.block_diag(*[torch.tensor([[1.4, .7], [-.2, .8]]) for _ in range(4)])
        before, db = self.build()
        after, da = self.build(v=self.v@change, u=torch.linalg.solve(change, self.u))
        for name in DIRECTIONS:
            torch.testing.assert_close(before[name], after[name], atol=3e-7, rtol=3e-6)
            for metric in ("selected_write_norm", "outside_write_norm", "grouped_write_leakage"):
                torch.testing.assert_close(torch.tensor(db["directions"][name][metric]),
                                           torch.tensor(da["directions"][name][metric]),
                                           atol=1e-6, rtol=1e-5)
        atom_before = torch.tensor(db["directions"]["hybrid_full"]["atom_energy_by_component_mean"])
        atom_after = torch.tensor(da["directions"]["hybrid_full"]["atom_energy_by_component_mean"])
        self.assertGreater(float((atom_before-atom_after).norm()), .01)

    def test_U_row_span_measures_physical_write_coverage(self):
        directions, diagnostic = self.build()
        self.assertEqual(diagnostic["factor"]["selected_U_group_ranks"], [2, 2, 2, 2])
        d = (self.states[2]-self.states[0]).double()
        t = (d@self.v.double())@self.u.double()
        expected = []
        for i, slot in enumerate(self.slots):
            selected_u = self.u.double()[self.groups[slot]]
            solution = torch.linalg.lstsq(selected_u.T, t[i, :, None]).solution.squeeze(-1)
            residual = (t[i]-solution@selected_u).norm()/t[i].norm()
            expected.append(float(residual))
        torch.testing.assert_close(torch.tensor(diagnostic["geometry"]["hybrid_selected_U_projection_residual"]),
                                   torch.tensor(expected), atol=2e-7, rtol=1e-6)
        h = self.states[0]+directions["hybrid_dual"]
        _, local = self.build(states=[self.states[0], self.states[1], h])
        self.assertLess(max(local["geometry"]["hybrid_selected_U_projection_residual"]), 1e-6)
        zero_u = torch.zeros_like(self.u)
        _, empty = self.build(u=zero_u)
        self.assertEqual(empty["factor"]["selected_U_group_ranks"], [0, 0, 0, 0])
        self.assertTrue(all(x is None for x in empty["geometry"]["hybrid_selected_U_projection_residual"]))
        json.dumps(empty, allow_nan=False)

    def test_invalid_dose_and_indices_are_rejected(self):
        for dose in (0., -1., float("nan"), float("inf")):
            with self.subTest(dose=dose), self.assertRaises(ValueError):
                build_directions(*self.states, self.v, self.u, self.groups, self.fixed,
                                 self.slots, dose=dose)
        for groups, slots in ((self.groups.float(), self.slots),
                              (self.groups, self.slots+4),
                              (self.groups+8, self.slots),
                              (torch.zeros_like(self.groups), self.slots)):
            with self.assertRaises(ValueError):
                build_directions(*self.states, self.v, self.u, groups, self.fixed, slots)


class NativePathwayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(40402)
        self.model = ComponentStateMemory(width=8, components=8)
        self.batch = make_pairs(40402, 12, root_seed=40402, delay=3, noise=.03)
        self.groups = torch.arange(8).reshape(4, 2)
        self.delta = torch.randn(12, 8)
        self.delta = .75*self.delta/self.delta.norm(dim=-1, keepdim=True)

    def test_full_and_clean_execute_original_native_path_exactly(self):
        for delta in (None, self.delta):
            expected = self.model.rollout(self.batch.receiver, patch=delta)
            actual = native_rollout(self.model, self.batch.receiver, delta=delta)
            for key in expected:
                self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def test_every_route_changes_exactly_its_declared_first_step_paths(self):
        clean = self.model.rollout(self.batch.receiver)
        v, u = self.model.recurrent.V, self.model.recurrent.U
        mask = torch.zeros(12, 8).scatter(1, self.groups[self.batch.edit_slot], 1.)
        reads = self.delta @ v
        group_write = (reads*mask)@u
        outside_write = (reads*(1-mask))@u
        h = clean["incoming_states"][:, 4]
        w = clean["recurrent_writes"][:, 4]
        ir, iz, inn = self.model.input(self.batch.receiver[:, 4]).chunk(3, dim=-1)
        definitions = {
            "full": (True, group_write+outside_write),
            "carry_only": (True, torch.zeros_like(w)),
            "all_write_only": (False, group_write+outside_write),
            "selected_write_only": (False, group_write),
            "outside_write_only": (False, outside_write),
            "carry_selected": (True, group_write),
            "carry_outside": (True, outside_write),
        }
        observed = []
        def inspect(module, inputs):
            observed.append((module.mode, bool((module.schedule.delta_mask==0).all()),
                             bool((module.schedule.comp_mask==1).all())))
        handle = self.model.recurrent.register_forward_pre_hook(inspect)
        try:
            for route, (carry, write_delta) in definitions.items():
                with self.subTest(route=route):
                    output = native_rollout(self.model, self.batch.receiver, self.delta,
                                            self.groups, self.batch.edit_slot, route=route)
                    expected_h = h+self.delta if carry else h
                    wr, wz, wn = (w+write_delta).chunk(3, dim=-1)
                    reset, update = torch.sigmoid(ir+wr), torch.sigmoid(iz+wz)
                    expected_state = (1-update)*torch.tanh(inn+reset*wn)+update*expected_h
                    torch.testing.assert_close(output["incoming_states"][:, 4], expected_h,
                                               atol=2e-7, rtol=1e-6)
                    torch.testing.assert_close(output["recurrent_writes"][:, 4], w+write_delta,
                                               atol=3e-7, rtol=2e-6)
                    torch.testing.assert_close(output["states"][:, 4], expected_state,
                                               atol=3e-7, rtol=2e-6)
                    self.assertTrue(torch.equal(output["states"][:, :4], clean["states"][:, :4]))
                    self.assertGreater(float((output["states"][:, 4]-clean["states"][:, 4]).detach().norm()), 1e-5)
        finally:
            handle.remove()
        self.assertEqual(len(observed), len(ROUTES)*self.batch.receiver.shape[1])
        self.assertTrue(all(x == ("component", True, True) for x in observed))
        self.assertEqual(len(self.model.recurrent._forward_hooks), 0)
        self.assertIsNone(self.model.recurrent.schedule)

    def test_full_hybrid_state_replacement_matches_hybrid_continuation(self):
        hybrid = self.model.rollout(self.batch.hybrid)
        output = native_rollout(self.model, self.batch.receiver,
                                state_replacement=hybrid["states"][:, 3])
        self.assertTrue(torch.equal(output["logits"], hybrid["logits"]))
        self.assertTrue(torch.equal(output["states"][:, 4:], hybrid["states"][:, 4:]))

    def test_native_routes_ignore_unused_target_weight(self):
        before = {route: native_rollout(self.model, self.batch.receiver, self.delta,
                                        self.groups, self.batch.edit_slot, route=route)["logits"]
                  for route in ROUTES}
        with torch.no_grad():
            self.model.recurrent.W_target.fill_(99999.)
        for route in ROUTES:
            after = native_rollout(self.model, self.batch.receiver, self.delta,
                                   self.groups, self.batch.edit_slot, route=route)["logits"]
            self.assertTrue(torch.equal(before[route], after), route)

    def test_forward_hook_cleanup_survives_a_failed_rollout(self):
        with self.assertRaises(ValueError):
            native_rollout(self.model, self.batch.receiver[:, :, :-1], self.delta,
                           self.groups, self.batch.edit_slot, route="carry_selected")
        self.assertEqual(len(self.model.recurrent._forward_hooks), 0)
        self.assertTrue(torch.equal(native_rollout(self.model, self.batch.receiver)["logits"],
                                    self.model(self.batch.receiver)))

    def test_finite_gate_carry_decomposition_matches_native_step(self):
        diagnostic = pathway_diagnostics(self.model, self.batch.receiver, self.delta)
        for key, value in diagnostic.items():
            if key.endswith("max_abs_error"):
                self.assertLess(value, 3e-7, key)
        self.assertGreater(max(diagnostic["interaction_norm"]), 1e-4)
        self.assertGreater(max(diagnostic["carry_term_norm"]), .1)
        json.dumps(diagnostic, allow_nan=False)
        zero = pathway_diagnostics(self.model, self.batch.receiver, torch.zeros_like(self.delta))
        self.assertTrue(all(x is None for x in zero["carry_gate_cosine"]))
        self.assertEqual(max(zero["state_delta_norm"]), 0.)

    def test_invalid_route_and_unmatched_inputs_fail(self):
        for kwargs in ({"route":"unknown"}, {"route":"carry_only"},
                       {"delta":self.delta, "state_replacement":self.delta},
                       {"delta":self.delta, "route":"selected_write_only"},
                       {"delta":self.delta, "patch_step":99}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                native_rollout(self.model, self.batch.receiver, **kwargs)


if __name__ == "__main__":
    unittest.main()
