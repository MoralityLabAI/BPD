"""Deterministic engineering checks with native BPD wrappers and autograd."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import unittest

import torch
from torch import nn

from mechinterp.bpd.components import (
    ComponentIntervention, PassMaskSchedule, TiedComponentLinear,
)
from mechinterp.bpd.experiments.inkling_component_flow.factor_flow import (
    ExecutionSite, ReaderSpec, clean_target_writes, component_direction_patch,
    component_reader_flow, patch_reader_audit, reader_state, to_jsonable,
)


def site(loop: int = 0, *, name: str = "core.down") -> ExecutionSite:
    return ExecutionSite("fixture-root", "fixture-checkpoint", "example-0", 3,
                         loop, 18, name, "depth")


def fixture() -> tuple[TiedComponentLinear, torch.Tensor, torch.Tensor]:
    layer = nn.Linear(3, 4, bias=True)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1., 0., .5], [0., 2., 0.],
                                         [.25, 0., 1.], [1., -.5, .5]]))
        layer.bias.copy_(torch.tensor([.2, -.1, .5, .25]))
    model = TiedComponentLinear(layer, 2, init_svd=False)
    with torch.no_grad():
        model.V.copy_(torch.tensor([[1., .25], [-.5, 1.], [.5, -.5]]))
        model.U.copy_(torch.tensor([[.5, -.25, 1., .25], [-1., .5, .25, .75]]))
    x = torch.tensor([.5, 1., -.25])
    with torch.no_grad():
        observed = model(x)
    return model, x, observed


def readers(width: int = 4, eps: float = .05) -> ReaderSpec:
    queries = torch.arange(1., 1. + 2 * width).reshape(2, width) / 5
    queries[1, 0] *= -1
    return ReaderSpec(replace(site(1, name="message_input"), physical_layer=13),
                      ("Q", "MoE"), queries, eps,
                      torch.linspace(.5, 1.5, width))


class FactorFlowTests(unittest.TestCase):
    def assertClose(self, actual, expected, atol=1e-11, rtol=1e-10):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_native_target_closure_includes_residual_and_bias(self):
        model, x, output = fixture()
        ledger = clean_target_writes(model, x, output, site())
        self.assertGreater(float(ledger.residual_write.norm()), .01)
        self.assertGreater(float(ledger.bias_write.norm()), .01)
        self.assertClose(ledger.component_writes.sum(0) + ledger.residual_write
                         + ledger.bias_write, output.double(), atol=1e-7)
        self.assertEqual(ledger.component_writes.shape, (2, 4))
        self.assertFalse(ledger.component_writes.requires_grad)
        self.assertEqual(ledger.component_writes.dtype, torch.float64)
        self.assertEqual(ledger.component_writes.device.type, "cpu")

    def test_observer_does_not_run_forward_or_mutate_wrapper(self):
        model, x, output = fixture()
        model.record_inputs = True
        model.update_running_means = True
        model.pass_offset = 3
        model.calls = 8
        saved = {key: value.clone() for key, value in model.state_dict().items()}
        clean_target_writes(model, x, output, site())
        self.assertEqual(model.calls, 8)
        self.assertEqual(model.pass_offset, 3)
        self.assertEqual(model.recorded_inputs, [])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(saved[key], value))

    def test_rejects_missing_native_output_closure(self):
        model, x, output = fixture()
        with self.assertRaisesRegex(ValueError, "native output closure"):
            clean_target_writes(model, x, output + .01, site())

    def test_rejects_masked_modes_and_interventions(self):
        model, x, output = fixture()
        model.mode = "component"
        with self.assertRaisesRegex(ValueError, "clean target mode"):
            clean_target_writes(model, x, output, site())
        model.mode = "target"
        model.schedule = PassMaskSchedule(torch.ones(1, 1, 1, 2))
        with self.assertRaisesRegex(ValueError, "clean target mode"):
            clean_target_writes(model, x, output, site())
        model.schedule = None
        model.intervention = ComponentIntervention((0,), None, "ablate")
        with self.assertRaisesRegex(ValueError, "clean target mode"):
            clean_target_writes(model, x, output, site())

    def test_explicit_other_writes_and_score_gap_close(self):
        model, x, output = fixture()
        ledger = clean_target_writes(model, x, output, site())
        remainder = torch.tensor([.5, -.1, .2, .4], dtype=torch.float64)
        b = output.double() + remainder
        result = component_reader_flow(ledger, readers(), torch.ones(4), b, remainder)
        self.assertClose(result["score_gap_terms"].sum(-1), result["reader"]["gap"], atol=1e-7)
        self.assertEqual(result["term_labels"][-3:],
                         ("weight_residual", "bias", "other_block_writes"))
        self.assertAlmostEqual(result["projection_share_sum"], 1., places=7)
        self.assertEqual(result["reader_site"]["physical_layer"], 13)
        self.assertEqual(result["reader_ids"], ("Q", "MoE"))
        with self.assertRaisesRegex(ValueError, "observed block closure"):
            component_reader_flow(ledger, readers(), torch.ones(4), b, remainder + .1)

    def test_jacobian_matches_autograd_with_epsilon_and_learned_scale(self):
        spec = readers(eps=.2)
        a = torch.tensor([.2, -.3, 1., .7], dtype=torch.float64)
        b = torch.tensor([1., .4, -.1, .3], dtype=torch.float64, requires_grad=True)
        state = reader_state(spec, a, b)
        def forward(v):
            norm_b = v * spec.rms_scale / (v.square().mean() + spec.eps).sqrt()
            norm_a = a * spec.rms_scale / (a.square().mean() + spec.eps).sqrt()
            gap = spec.queries @ (norm_b - norm_a) / math.sqrt(4)
            return torch.cat((gap, torch.sigmoid(gap)))
        jac = torch.autograd.functional.jacobian(forward, b)
        self.assertClose(jac[:2], state["gap_gradient_block3"])
        self.assertClose(jac[2:], state["beta3_gradient_block3"])
        self.assertClose(state["radial_gap_derivative"],
                         state["analytic_radial_gap_derivative"])
        self.assertGreater(float(state["radial_gap_derivative"].abs().max()), .01)

    def test_component_derivatives_match_central_finite_differences(self):
        model, x, output = fixture()
        ledger = clean_target_writes(model, x, output, site())
        spec = readers()
        a, b = torch.ones(4), output.double() + .3
        flow = component_reader_flow(ledger, spec, a, b, torch.full((4,), .3, dtype=torch.float64))
        step = 1e-5
        for c in range(2):
            direction = ledger.component_writes[c]
            plus = reader_state(spec, a, b + step * direction)
            minus = reader_state(spec, a, b - step * direction)
            self.assertClose((plus["gap"] - minus["gap"]) / (2 * step),
                             flow["gap_derivative_per_write"][:, c], atol=1e-9)
            self.assertClose((plus["beta3"] - minus["beta3"]) / (2 * step),
                             flow["beta3_derivative_per_write"][:, c], atol=1e-9)

    def test_epsilon_zero_radial_null_and_zero_block_rejection(self):
        spec = readers(eps=0.)
        b = torch.tensor([1., .5, -.7, .3])
        state = reader_state(spec, torch.ones(4), b)
        self.assertClose(state["radial_gap_derivative"], torch.zeros(2, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "undefined normalization"):
            reader_state(spec, torch.ones(4), torch.zeros(4))

    def cancellation_ledger(self, cancel_all=False):
        model = TiedComponentLinear(nn.Linear(1, 2, bias=False), 2, init_svd=False)
        with torch.no_grad():
            model.V.fill_(1.)
            model.U.copy_(torch.tensor([[2., 0.], [-2. if cancel_all else -1., 0.]]))
            model.W_target.copy_(torch.tensor([[0. if cancel_all else 1.], [0.]]))
            output = model(torch.ones(1))
        return clean_target_writes(model, torch.ones(1), output, site())

    def test_signed_projection_shares_preserve_cancellation(self):
        ledger = self.cancellation_ledger()
        flow = component_reader_flow(ledger, readers(2), torch.tensor([0., 1.]),
                                     ledger.observed_output, torch.zeros(2))
        self.assertClose(flow["signed_projection_shares"][:2],
                         torch.tensor([2., -1.], dtype=torch.float64))
        self.assertEqual(flow["projection_share_sum"], 1.)

    def test_zero_state_marks_projection_shares_undefined(self):
        ledger = self.cancellation_ledger(cancel_all=True)
        flow = component_reader_flow(ledger, readers(2), torch.tensor([0., 1.]),
                                     torch.zeros(2), torch.zeros(2))
        self.assertFalse(flow["projection_shares_valid"])
        self.assertIsNone(flow["signed_projection_shares"])
        self.assertIsNone(flow["projection_share_sum"])
        json.dumps(to_jsonable(flow), allow_nan=False)

    def test_zero_edit_is_exact_and_message_decomposition_closes(self):
        spec = readers()
        a, b = torch.tensor([1., .5, .2, -.1]), torch.tensor([.4, -.2, .1, .7])
        zero = patch_reader_audit(spec, a, b, torch.zeros(4), source_site=site(), max_dose=0.)
        for field in ("actual_gap_change", "actual_beta3_change", "message_delta",
                      "route_delta", "content_delta", "interaction_delta"):
            self.assertTrue(torch.equal(zero[field], torch.zeros_like(zero[field])))
        delta = torch.tensor([.3, .1, -.2, .2])
        result = patch_reader_audit(spec, a, b, delta, source_site=site(), max_dose=1.)
        self.assertClose(result["route_delta"] + result["content_delta"]
                         + result["interaction_delta"], result["message_delta"])
        self.assertGreater(float(result["interaction_delta"].abs().max()), 1e-6)
        self.assertAlmostEqual(result["dose_l2"], float(delta.double().norm()))
        self.assertLess(result["decomposition_max_abs_error"], 1e-14)

    def test_direction_patch_delivers_requested_dose_and_enforces_budget(self):
        model, x, output = fixture()
        ledger = clean_target_writes(model, x, output, site())
        delta = component_direction_patch(ledger, 0, .07)
        self.assertAlmostEqual(float(delta.norm()), .07, places=14)
        self.assertLess(float(delta @ ledger.component_writes[0]), 0.)
        result = patch_reader_audit(readers(), torch.ones(4), output, delta, source_site=site(), max_dose=.07)
        self.assertAlmostEqual(result["dose_l2"], .07, places=14)
        with self.assertRaisesRegex(ValueError, "exceeds max_dose"):
            patch_reader_audit(readers(), torch.ones(4), output, delta, source_site=site(), max_dose=.01)
        with self.assertRaisesRegex(ValueError, "invalid component index"):
            component_direction_patch(ledger, -1, .1)

    def test_site_identity_preserves_loop_specific_signed_writes(self):
        model, x, output = fixture()
        first = clean_target_writes(model, x, output, site(0))
        with torch.no_grad():
            later_output = model(-x)
        later = clean_target_writes(model, -x, later_output, site(2))
        self.assertNotEqual(first.site, later.site)
        self.assertClose(first.component_writes, -later.component_writes)
        wrong_context = replace(readers().site, example_id="example-1")
        spec = replace(readers(), site=wrong_context)
        with self.assertRaisesRegex(ValueError, "example_id"):
            component_reader_flow(first, spec, torch.ones(4), output, torch.zeros(4))

    def test_declared_depth_order_rejects_wrong_tokens_and_nonadjacent_loops(self):
        model, x, output = fixture()
        source = site(2)
        ledger = clean_target_writes(model, x, output, source)
        for loop in (0, 1, 2, 4):
            spec = replace(readers(), site=replace(readers().site, loop_index=loop))
            with self.subTest(loop=loop):
                with self.assertRaisesRegex(ValueError, "immediately previous loop"):
                    component_reader_flow(ledger, spec, torch.ones(4), output, torch.zeros(4))
                with self.assertRaisesRegex(ValueError, "immediately previous loop"):
                    patch_reader_audit(spec, torch.ones(4), output, torch.zeros(4),
                                       source_site=source, max_dose=0.)
        spec = replace(readers(), site=replace(readers().site, loop_index=3, token_index=4))
        with self.assertRaisesRegex(ValueError, "preserve token_index"):
            component_reader_flow(ledger, spec, torch.ones(4), output, torch.zeros(4))
        with self.assertRaisesRegex(ValueError, "preserve token_index"):
            patch_reader_audit(spec, torch.ones(4), output, torch.zeros(4),
                               source_site=source, max_dose=0.)

    def test_declared_sequence_time_order_requires_increasing_tokens(self):
        model, x, output = fixture()
        source = replace(site(), step_kind="sequence_time")
        ledger = clean_target_writes(model, x, output, source)
        receiver = replace(readers().site, step_kind="sequence_time", token_index=4)
        spec = replace(readers(), site=receiver)
        flow = component_reader_flow(ledger, spec, torch.ones(4), output, torch.zeros(4))
        self.assertEqual(flow["writer_site"]["token_index"], 3)
        self.assertEqual(flow["reader_site"]["token_index"], 4)
        patch_reader_audit(spec, torch.ones(4), output, torch.zeros(4),
                           source_site=source, max_dose=0.)
        for token in (2, 3):
            bad = replace(spec, site=replace(receiver, token_index=token))
            with self.assertRaisesRegex(ValueError, "strictly increase"):
                patch_reader_audit(bad, torch.ones(4), output, torch.zeros(4),
                                   source_site=source, max_dose=0.)

    def test_patch_receipt_retains_inputs_for_independent_replay(self):
        spec = readers()
        a, b = torch.ones(4), torch.tensor([.3, -.4, .1, .8])
        delta = torch.tensor([.05, .02, -.1, .04])
        audit = patch_reader_audit(spec, a, b, delta, source_site=site(), max_dose=1.)
        saved = json.loads(json.dumps(to_jsonable(audit), allow_nan=False))
        params = saved["reader_spec"]
        queries = torch.tensor(params["queries"], dtype=torch.float64)
        scale = torch.tensor(params["rms_scale"], dtype=torch.float64)
        aa = torch.tensor(saved["block2"], dtype=torch.float64)
        bb = torch.tensor(saved["block3_before"], dtype=torch.float64)
        dd = torch.tensor(saved["applied_displacement"], dtype=torch.float64)
        def independent_probability(v):
            blocks = torch.stack((aa, v))
            normalized = blocks * scale / (blocks.square().mean(-1, keepdim=True) + params["eps"]).sqrt()
            return torch.softmax(queries @ normalized.T / math.sqrt(4), dim=-1)[:, 1]
        expected = independent_probability(bb + dd) - independent_probability(bb)
        self.assertClose(expected, audit["actual_beta3_change"])
        self.assertEqual(saved["source_site"], vars(site()))
        self.assertEqual(saved["reader_site"], params["site"])
        self.assertEqual(saved["intervention_site"], "previous_loop_completed_block3_boundary")

    def test_extreme_finite_inputs_fail_before_nonfinite_results_escape(self):
        with self.assertRaisesRegex(ValueError, "nonfinite derived.*RMS denominator"):
            reader_state(readers(), torch.ones(4), torch.full((4,), 1e200, dtype=torch.float64))
        huge_spec = replace(readers(), queries=torch.full((2, 4), 1e308, dtype=torch.float64),
                            rms_scale=torch.full((4,), 1e308, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "nonfinite derived scaled queries"):
            reader_state(huge_spec, torch.ones(4), torch.ones(4))
        model, x, output = fixture()
        ledger = clean_target_writes(model, x, output, site())
        huge_ledger = replace(ledger, component_writes=torch.full((2, 4), 1e200, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "nonfinite derived component direction norm"):
            component_direction_patch(huge_ledger, 0, .1)
        with self.assertRaisesRegex(ValueError, "nonfinite derived patch dose"):
            patch_reader_audit(readers(), torch.ones(4), output,
                               torch.full((4,), 1e200, dtype=torch.float64),
                               source_site=site(), max_dose=1e250)
        model.double()
        with torch.no_grad():
            model.V.fill_(1e200)
            model.U.fill_(1e200)
        with self.assertRaisesRegex(ValueError, "nonfinite derived component writes"):
            clean_target_writes(model, torch.ones(3), torch.zeros(4), site())
        with self.assertRaisesRegex(ValueError, "nonfinite derived JSON diagnostic"):
            to_jsonable({"bad": torch.tensor(float("inf"))})

    def test_inverse_diagonal_factor_rescaling_preserves_flow(self):
        model, x, output = fixture()
        first = clean_target_writes(model, x, output, site())
        with torch.no_grad():
            model.V.mul_(torch.tensor([8., -.5]))
            model.U.div_(torch.tensor([8., -.5])[:, None])
        scaled = clean_target_writes(model, x, output, site())
        self.assertClose(first.component_writes, scaled.component_writes)
        self.assertClose(first.residual_write, scaled.residual_write)
        kwargs = dict(reader=readers(), block2=torch.ones(4), block3=output,
                      other_block_writes=torch.zeros(4))
        flow1, flow2 = component_reader_flow(first, **kwargs), component_reader_flow(scaled, **kwargs)
        for field in ("score3_terms", "gap_derivative_per_write", "signed_projection_shares"):
            self.assertClose(flow1[field], flow2[field])

    def test_general_component_mixing_can_change_entries_but_preserves_total(self):
        model, x, output = fixture()
        first = clean_target_writes(model, x, output, site())
        transform = torch.tensor([[1., 1.], [0., 1.]])
        with torch.no_grad():
            model.V.copy_(model.V @ transform)
            model.U.copy_(torch.linalg.inv(transform) @ model.U)
        mixed = clean_target_writes(model, x, output, site())
        self.assertFalse(torch.equal(first.component_writes, mixed.component_writes))
        self.assertClose(first.component_writes.sum(0), mixed.component_writes.sum(0))

    def test_invalid_inputs_fail_explicitly(self):
        model, x, output = fixture()
        with self.assertRaisesRegex(ValueError, "rank-1"):
            clean_target_writes(model, x[None], output, site())
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            clean_target_writes(model, x * float("nan"), output, site())
        with self.assertRaisesRegex(ValueError, "widths disagree"):
            reader_state(readers(), torch.ones(3), torch.ones(4))
        with self.assertRaisesRegex(ValueError, "invalid eps"):
            replace(readers(), eps=-.1)
        with self.assertRaisesRegex(ValueError, "reader ID"):
            replace(readers(), reader_ids=("Q", "Q"))
        with self.assertRaisesRegex(ValueError, "physical_layer"):
            replace(site(), physical_layer=0)
        with self.assertRaisesRegex(ValueError, "step_kind"):
            replace(site(), step_kind="unspecified")
        with self.assertRaisesRegex(ValueError, "invalid dose"):
            component_direction_patch(clean_target_writes(model, x, output, site()), 0, -1.)


if __name__ == "__main__":
    unittest.main()
