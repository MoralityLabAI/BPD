"""Source-only tests for SUB-01; no registered development model is trained."""
import pytest
import torch

from mechinterp.bpd.experiments.sub01.experiment import (
    WIDTH, ROOTS, SMOKE, RANKS, Looper, sample_x, external_labels,
    pca_projectors, projected_rollout, gauge_score, fiber_score,
    impulse_directions, sham_null_projector, source_sha, protocol_sha,
    train_source, aggregate,
)


def test_gauge_transports_interventions_and_misalignment_is_visible():
    torch.manual_seed(7)
    model = Looper().eval()
    x = sample_x(SMOKE, 2, 48)
    result = gauge_score(model, x, SMOKE, alpha=0.25)
    assert result["max_clean_logit_error"] < 5e-6
    assert result["max_transported_impulse_logit_error"] < 5e-6
    assert result["max_clean_state_error"] < 5e-6
    assert result["max_intervened_state_error"] < 5e-6
    assert result["untransported_impulse_logit_rms"] > 1e-5


def test_nested_pca_projections_and_identity_rollout():
    torch.manual_seed(8)
    model = Looper().eval()
    x = sample_x(SMOKE, 2, 80)
    projectors, alpha, info = pca_projectors(model, x)
    assert alpha > 0
    assert info["nested_max_abs"] < 1e-5
    for r in RANKS:
        p = projectors[r]
        assert float((p @ p - p).abs().max()) < 1e-5
        assert float((p - p.T).abs().max()) < 1e-5
        assert float(torch.trace(p)) == pytest.approx(r, abs=1e-4)
    direct, _ = model.rollout(x)
    identity, _ = projected_rollout(model, x, torch.eye(WIDTH))
    assert torch.allclose(identity, direct, atol=1e-6, rtol=0)


def block_model(*, cross_coupled: bool):
    model = Looper().eval()
    with torch.no_grad():
        model.inp.weight.zero_()
        model.inp.bias.zero_()
        model.hidden.weight.zero_()
        model.hidden.weight[0, 0] = .7
        if cross_coupled:
            model.hidden.weight[0, 6] = 1.
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.weight[0, 0] = 1.
    return model


def test_null_fiber_exactly_invisible_for_decoupled_dynamics():
    p = torch.diag(torch.tensor([1.] * 6 + [0.] * 6))
    model = block_model(cross_coupled=False)
    result = fiber_score(
        model, torch.zeros(32, 4), p, torch.eye(WIDTH) - p,
        alpha=.4, seed=SMOKE, split=2,
    )
    assert result["terminal_logit_rms"] < 1e-7
    assert result["projected_next_hidden_norm_rms"] < 1e-7
    assert result["max_dose_deviation"] < 1e-6
    assert result["max_null_projection"] < 1e-6


def test_hidden_to_visible_coupling_breaks_quotient():
    p = torch.diag(torch.tensor([1.] * 6 + [0.] * 6))
    model = block_model(cross_coupled=True)
    result = fiber_score(
        model, torch.zeros(64, 4), p, torch.eye(WIDTH) - p,
        alpha=.5, seed=SMOKE, split=2,
    )
    assert result["terminal_logit_rms"] > .001
    assert result["projected_next_hidden_norm_rms"] > .001


def test_fixed_physical_dose_for_pca_and_haar_shams():
    torch.manual_seed(4)
    x = sample_x(SMOKE, 3, 64)
    p = sham_null_projector(6, SMOKE, 0)
    assert float(torch.trace(p)) == pytest.approx(6, abs=1e-4)
    assert float((p @ p - p).abs().max()) < 1e-5
    d = impulse_directions(SMOKE, 3, x, p, .3)
    assert torch.allclose(d.norm(dim=1), torch.full((len(x),), .3), atol=1e-6)
    assert float((d @ (torch.eye(WIDTH) - p)).abs().max()) < 1e-6


def test_roster_and_provenance_guardrails():
    assert list(ROOTS) == list(range(1200, 1208))
    assert SMOKE not in ROOTS
    assert len(protocol_sha()) == len(source_sha()) == 64
    assert torch.equal(sample_x(SMOKE, 3, 8), sample_x(SMOKE, 3, 8))
    assert len(external_labels(sample_x(SMOKE, 1, 8))) == 8
    with pytest.raises(ValueError, match="Unregistered"):
        train_source(1300, steps=1)
    with pytest.raises(ValueError, match="eight"):
        aggregate([])
