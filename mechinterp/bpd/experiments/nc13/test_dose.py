"""Algebraic and bounded-search checks; no trained models or development roots."""
import math

import numpy as np
import pytest
import torch

from .dose import (
    ANGLES, calibrate_closed_displacement, dose_metrics, energy_gain, select_shams,
)


def physical_fixture():
    acts = torch.tensor([[[1., 2.]], [[-1., .5]]], dtype=torch.float64)
    u = torch.tensor([[1., 1.], [0., 2.]], dtype=torch.float64)
    p = torch.tensor([[[1., 0.], [0., 0.]]], dtype=torch.float64)
    return acts, u, p


def test_clean_displacement_has_cross_term_for_nonisometric_writes():
    rec = dose_metrics(*physical_fixture(), gain=.7)
    assert rec["quadratic_moments"] == {"a": 1., "b": 1.75, "c": 6.75}
    assert rec["retained_rms"] == pytest.approx(.7)
    assert rec["clean_displacement_rms"] == pytest.approx(math.sqrt(4.79))
    assert rec["clean_displacement_rms_by_pass"] == pytest.approx([math.sqrt(4.79)])
    assert rec["clean_write_rms"] == pytest.approx(math.sqrt(6.75))


@pytest.mark.parametrize("gain", [.5, 1.2, 1.5])
def test_energy_match_at_admitted_gains(gain):
    rec = energy_gain(gain, *physical_fixture())
    assert rec["admitted"] and rec["gain"] == pytest.approx(gain)
    assert rec["matched_retained_rms"] == pytest.approx(gain)


@pytest.mark.parametrize("reference", [.49, 1.51])
def test_energy_failure_has_no_applied_gain(reference):
    rec = energy_gain(reference, *physical_fixture())
    assert not rec["admitted"] and rec["gain"] is None
    assert rec["reason"] == "GAIN_OUTSIDE_REGISTERED_BOUNDS"
    assert rec["matched_retained_rms"] is None


def test_zero_energy_and_invalid_reference_are_rejected():
    a, u, p = physical_fixture()
    assert energy_gain(1., a, u, torch.zeros_like(p))["reason"] == "ZERO_RETAINED_WRITE"
    assert energy_gain(0., a, u, p)["reason"] == "INVALID_REFERENCE"


def test_full_grid_is_recorded_even_when_one_matches_exactly():
    seen = []
    def evaluate(gain):
        seen.append(gain)
        return gain
    rec = calibrate_closed_displacement(evaluate, 1.)
    assert seen == np.linspace(.5, 1.5, 17).tolist()
    assert rec["admitted"] and rec["gain"] == 1.
    assert rec["rollout_calls"] == 17 and rec["bisection_steps"] == 0
    assert len(rec["queries"]) == 17


def test_grid_match_tie_selects_lower_gain():
    rec = calibrate_closed_displacement(
        lambda g: 1. if g in (.9375, 1.0625) else 2., 1.)
    assert rec["gain"] == .9375
    assert rec["rollout_calls"] == 17


def test_constant_matching_function_selects_gain_one():
    rec = calibrate_closed_displacement(lambda _: 1., 1.)
    assert rec["gain"] == 1. and rec["rollout_calls"] == 17


def test_bisection_uses_exact_budget_and_selects_a_queried_match():
    rec = calibrate_closed_displacement(lambda g: .2 + g, 1.207)
    assert rec["admitted"]
    assert rec["selected_bracket"] == [1., 1.0625]
    assert rec["rollout_calls"] == 41 and rec["bisection_steps"] == 24
    assert len([r for r in rec["queries"] if r["stage"] == "bisection"]) == 24
    accepted = [r for r in rec["queries"] if r["relative_error"] <= 1e-4]
    expected = min(accepted, key=lambda r: (abs(r["gain"] - 1), r["gain"]))
    assert rec["gain"] == expected["gain"]
    assert abs(.2 + rec["gain"] - 1.207) / 1.207 <= 1e-4


def test_nonmonotone_search_chooses_nearest_bracket():
    rec = calibrate_closed_displacement(lambda g: 1 + (g - .83) * (g - 1.23), 1.)
    assert rec["admitted"]
    assert rec["selected_bracket"] == [.8125, .875]
    assert abs(rec["gain"] - .83) < .001
    assert rec["rollout_calls"] == 41


def test_equal_distance_brackets_choose_lower_endpoint():
    rec = calibrate_closed_displacement(lambda g: 1 + (g - .81) * (g - 1.19), 1.)
    assert rec["admitted"]
    assert rec["selected_bracket"] == [.75, .8125]


def test_missing_bracket_leaves_global_attainability_open():
    rec = calibrate_closed_displacement(lambda g: 1 + (g - 1.0123) ** 2, 1.)
    assert not rec["admitted"] and rec["gain"] is None
    assert rec["reason"] == "NO_REGISTERED_BRACKET"
    assert rec["rollout_calls"] == 17
    # The actual root exists at 1.0123; the registered grid cannot bracket it.


def test_constant_unmatched_function_has_fixed_failure_budget():
    rec = calibrate_closed_displacement(lambda _: 1.2, 1.)
    assert rec["reason"] == "NO_REGISTERED_BRACKET"
    assert rec["gain"] is None and rec["rollout_calls"] == 17


def test_discontinuous_bracket_can_fail_tolerance_without_fallback():
    rec = calibrate_closed_displacement(lambda g: .8 if g < 1.01 else 1.2, 1.)
    assert not rec["admitted"] and rec["gain"] is None
    assert rec["reason"] == "MATCH_TOLERANCE_FAILED"
    assert rec["rollout_calls"] == 41


def test_invalid_reference_spends_no_rollout_queries():
    for reference in [0., 1e-13, float("inf")]:
        rec = calibrate_closed_displacement(lambda _: pytest.fail("unexpected query"), reference)
        assert rec["reason"] == "INVALID_REFERENCE"
        assert rec["rollout_calls"] == 0 and rec["gain"] is None


def test_nonfinite_grid_observation_is_recorded_and_rejected():
    rec = calibrate_closed_displacement(lambda g: float("nan") if g == .5 else 1., 1.)
    assert rec["reason"] == "INVALID_DISPLACEMENT"
    assert rec["rollout_calls"] == 17 and rec["gain"] is None
    assert rec["queries"][0]["displacement"] is None


def sham_fixture():
    acts = torch.eye(20, dtype=torch.float64)[:, None, :].repeat(1, 4, 1)
    u = torch.eye(20, dtype=torch.float64)
    basis = torch.eye(20, dtype=torch.float64)[:, :4][None].repeat(4, 1, 1)
    return acts, u, basis


def test_isotropic_shams_preserve_rank_energy_and_proposal_order():
    shams, rec = select_shams(*sham_fixture(), .1, root=17)
    assert rec["admitted"] and len(shams) == 8
    assert rec["proposals"] == rec["evaluated_proposals"] == rec["qualified"] == 2048
    assert rec["selected_indices"] == list(range(8))
    assert rec["proposal_indices"] == list(range(2048))
    assert rec["proposal_angles"] == list(ANGLES) * 512
    assert rec["proposal_retained_rms"] == pytest.approx([.1] * 2048)
    assert rec["generator_seed"] == 17 * 1009 + 13013
    for p in shams:
        assert torch.allclose(p, p.transpose(-1, -2), atol=1e-12, rtol=0)
        assert torch.allclose(p @ p, p, atol=1e-12, rtol=0)
        assert torch.diagonal(p, dim1=-2, dim2=-1).sum(-1).tolist() == pytest.approx([4.] * 4)


def test_sham_failure_preserves_all_fixed_proposals():
    shams, rec = select_shams(*sham_fixture(), .5, root=19)
    assert not rec["admitted"] and shams == []
    assert rec["evaluated_proposals"] == 2048
    assert rec["qualified"] == rec["selected"] == 0
    assert rec["rejection_counts"]["outside_energy_tolerance"] == 2048
    assert rec["reason"] == "INSUFFICIENT_ENERGY_MATCHED_SHAMS"
