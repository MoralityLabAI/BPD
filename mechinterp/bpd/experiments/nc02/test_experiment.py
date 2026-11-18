"""Deterministic, source-only NC-02 positive-control and audit tests."""
from __future__ import annotations

import math
import torch
import pytest

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02 import experiment as e


def _wrapped(seed: int = 200, steps: int = 12):
    model, data = e.train_model(seed, steps)
    bpd = BPDModel(model, {"mix": e.WIDTH}, max_passes=e.PASSES,
                   ablation=AblationConfig(mode="zero", sphere_rescale=False), init_svd=True)
    return model, bpd, data


def test_training_preserves_physical_path_and_equal_decoy_gains():
    model, result = e.train_model(200, 24)
    assert result["structure"]["passed"]
    assert result["structure"]["max_absolute_offpath_weight"] == 0
    assert result["structure"]["max_within_stage_gain_difference"] == 0
    assert result["training_accuracy"] > 0.95
    assert len(result["structure"]["gains_by_stage_lane"]) == 4


def test_external_labels_ignore_model_and_depend_on_context():
    x = torch.tensor([[2., -4., 8., 16., 1.], [2., -4., 8., 16., -1.]])
    assert e.external_labels(x).tolist() == [True, False]
    assert not torch.equal(e.sample_x(200, 1, 16), e.sample_x(200, 2, 16))
    assert not torch.equal(e.sample_x(200, 2, 16), e.sample_x(200, 3, 16))
    assert not torch.equal(e.sample_x(200, 2, 16), e.sample_x(201, 2, 16))


def test_all_readout_parity_and_exact_factors_on_three_bases():
    model, bpd, _ = _wrapped()
    x = e.sample_x(200, 2, 32)
    for basis in ("native", "bpd_svd", "rotated"):
        e.set_basis(bpd, basis, 200)
        layer = bpd.wrappers["mix"]
        assert torch.allclose((layer.V @ layer.U).T, layer.W_target, atol=1e-5)
        report = e.parity(bpd, model, x)
        assert report["passed"], (basis, report)
        assert report["max_kl_nats"] <= 1e-7


def test_native_attribution_recovers_both_independent_planted_paths():
    model, bpd, _ = _wrapped()
    x = e.sample_x(200, 2, 128)
    e.set_basis(bpd, "native", 200)
    selected = e.selections(bpd, model, x, e.external_labels(x))
    assert selected.shape == (128, e.PASSES * e.WIDTH)
    groups = e.communities(selected)
    oracle = e.best_native_recovery(groups)
    assert oracle["exact_recovered_paths"] == 2, (groups, oracle)
    assert set(groups[0]["nodes"]) in (set(e.true_path(0)), set(e.true_path(1)))
    assert all(len(g["passes"]) == e.PASSES for g in groups[:2])


def test_zero_readout_has_zero_positive_attribution_and_no_community():
    model, bpd, _ = _wrapped()
    e.set_basis(bpd, "native", 200)
    with torch.no_grad():
        model.readout.zero_()
    x = e.sample_x(200, 2, 96)
    hits = e.selections(bpd, model, x, e.external_labels(x))
    assert not hits.any()
    assert e.communities(hits) == []


def test_sparse_temporal_graph_handles_missing_pairwise_edges():
    data = torch.zeros(256, e.PASSES * e.WIDTH)
    # A and C weakly overlap, B active on their union. A-B-C connected.
    data[:48, 0] = 1
    data[48:96, e.WIDTH + 4] = 1
    data[96:112, 0] = 1
    data[96:112, e.WIDTH + 4] = 1
    data[:, 2 * e.WIDTH + 8] = (data[:,0] + data[:,e.WIDTH+4] > 0).float()
    groups = e.communities(data)
    assert len(groups) == 1
    assert groups[0]["nodes"] == [0, e.WIDTH + 4, 2 * e.WIDTH + 8]


def test_exact_gram_rms_matches_direct_physical_write_dose():
    model, bpd, _ = _wrapped()
    e.set_basis(bpd, "native", 200)
    x = e.sample_x(200, 2, 32)
    with bpd.target_capture():
        model(x, capture=True)
        acts = torch.stack(model.acts, dim=1)
    layer = bpd.wrappers["mix"]
    gram = e._dose_gram(acts, layer.V.detach(), layer.U.detach())
    nodes = e.true_path(0)
    m = e.node_mask(nodes)
    direct = float(((((acts @ layer.V) * m) @ layer.U).square().mean()).sqrt().detach())
    analytic = float(e._dose_from_masks(gram, m.numpy()[None])[0])
    assert math.isclose(direct, analytic, abs_tol=1e-6)


def test_oracle_free_shams_match_physical_dose_and_site_profile():
    model, bpd, _ = _wrapped()
    e.set_basis(bpd, "native", 200)
    x = e.sample_x(200, 2, 256)
    with bpd.target_capture():
        model(x, capture=True)
        acts = torch.stack(model.acts, dim=1)
    layer = bpd.wrappers["mix"]
    gram = e._dose_gram(acts, layer.V.detach(), layer.U.detach())
    nodes = e.true_path(0)
    report = e.matched_shams(nodes, gram, "native", 123)
    assert report["eligible"], report
    assert len({tuple(n["nodes"]) for n in report["shams"]}) == 8
    for sham in report["shams"]:
        assert sham["nodes"] != nodes
        assert len(sham["nodes"]) == 4
        assert [sum(n // e.WIDTH == t for n in sham["nodes"]) for t in range(4)] == [1] * 4
        assert sham["relative_dose_error"] <= e.DOSE_TOLERANCE
        assert all(n % e.WIDTH in range(t * e.LANES, (t + 1) * e.LANES)
                   for t, n in enumerate(sham["nodes"]))
    decoy = e.matched_shams(nodes, gram, "native", 123, decoy_only=True)
    assert decoy["eligible"]
    assert all(n % e.LANES in (2, 3) for p in decoy["shams"] for n in p["nodes"])


def test_closed_loop_necessity_and_decoy_negative_intervention():
    model, bpd, _ = _wrapped(200, 32)
    e.set_basis(bpd, "native", 200)
    x = e.sample_x(200, 3, 128)
    y = e.external_labels(x)
    ablate_signal = e.evaluate(bpd, model, x, y, e.true_path(0))
    ablate_decoy = e.evaluate(bpd, model, x, y,
                              [t * e.WIDTH + t * e.LANES + 2 for t in range(e.PASSES)])
    assert ablate_signal["arms"]["ablate"]["nll_delta_nats"] > 0.05
    assert abs(ablate_decoy["arms"]["ablate"]["nll_delta_nats"]) < 1e-6
    assert ablate_decoy["arms"]["ablate"]["final_kl_nats"] < 1e-10
    assert ablate_signal["arms"]["ablate"]["closed_loop_write_change_rms"] > 0


def test_root_bootstrap_treats_models_as_units_and_blocks_confirmation():
    report = e.root_bootstrap([0.1, 0.2, 0.3, 0.4] * 2)
    assert report["n_independent_roots"] == 8
    assert report["scientific_status"] == "ENGINEERING_ONLY"
    lo, hi = report["root_bootstrap_95pct_CI"]
    assert lo > 0 and hi < 0.4
    with pytest.raises(ValueError):
        e.run_pilot(199, steps=1, discovery_size=1, evaluation_size=1)
