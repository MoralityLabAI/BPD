"""Source-only mathematical tests, excluded from SUB-02 development cohort."""
import pytest
import torch

from mechinterp.bpd.experiments.sub01.experiment import (
    WIDTH, Looper, sample_x, pca_projectors,
)
from mechinterp.bpd.experiments.sub02.experiment import (
    SMOKE, DEVELOPMENT_ROOTS, RANKS, future_logit_gradients,
    obs_projectors, score_arm, sham_projector, train_source,
    whole_root_bootstrap, aggregate,
)


def test_observability_projector_minimizes_discovery_linearized_sensitivity():
    torch.manual_seed(4)
    model = Looper().eval()
    x = sample_x(SMOKE, 2, 96)
    gradients, _ = future_logit_gradients(model, x)
    p_obs, info = obs_projectors(model, x)
    p_pca, _, _ = pca_projectors(model, x)
    assert info["finite_difference_max_abs"] < .02
    for rank in RANKS:
        p = p_obs[rank]
        err_obs = (gradients @ (torch.eye(WIDTH)-p)).square().sum()
        err_pca = (gradients @ (torch.eye(WIDTH)-p_pca[rank])).square().sum()
        assert float(err_obs) <= float(err_pca) + 1e-3
        assert info["rank_geometry"][str(rank)]["idempotence_max_abs"] < 1e-5
        assert info["rank_geometry"][str(rank)]["symmetry_max_abs"] < 1e-5
        assert info["rank_geometry"][str(rank)]["trace_error"] < 1e-4


def test_random_shams_rank_dose_and_exact_rollout_compatibility():
    torch.manual_seed(7)
    model = Looper().eval()
    x = sample_x(SMOKE, 4, 32)
    for rank in RANKS:
        p = sham_projector(SMOKE, rank, 3)
        assert float(torch.trace(p)) == pytest.approx(rank,abs=1e-4)
        assert float((p@p-p).abs().max()) < 1e-5
        score = score_arm(model, x, rank, p, .25, SMOKE, 4)
        assert score["fiber"]["max_dose_deviation"] < 5e-6
        assert score["fiber"]["max_null_projection"] < 5e-6


def test_fixed_roster_and_statistical_root_unit():
    assert list(DEVELOPMENT_ROOTS) == list(range(1300,1308))
    assert SMOKE not in DEVELOPMENT_ROOTS
    with pytest.raises(ValueError,match="outside registered roster"):
        train_source(1308,steps=1)
    with pytest.raises(ValueError,match="eight"):
        aggregate([])
    assert len(whole_root_bootstrap([1.,2.,3.,4.])) == 2
