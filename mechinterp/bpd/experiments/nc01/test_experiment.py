"""NC-01 audit tests: graph, basis parity, matched shams and split isolation."""
from __future__ import annotations

import torch

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc01.experiment import (
    PASSES, WIDTH, TinyLooper, attribution_selection, choose_matched_shams,
    closure_check, communities_from_selection, direct_write_rms, evaluate_masks, binary_kl,
    labels_from_teacher, make_basis, node_mask, sample_x, set_basis, train_student,
)


def _new_wrapped(seed: int = 19):
    torch.manual_seed(seed)
    model = TinyLooper()
    wrapper = BPDModel(model, {"mix": WIDTH}, max_passes=PASSES,
                       ablation=AblationConfig(mode="zero", sphere_rescale=False))
    return model, wrapper


def test_full_rank_bases_are_exact_and_parity_closes_across_all_passes():
    model, wrapper = _new_wrapped()
    x = sample_x(19, 3, 32)
    w = wrapper.wrappers["mix"].W_target
    for basis in ("native", "bpd_svd", "rotated"):
        v, u = make_basis(w, basis, 19)
        assert torch.allclose((v @ u).T, w, atol=2e-5)
        set_basis(wrapper, basis, 19)
        receipt = closure_check(wrapper, model, x)
        assert receipt["passed"], (basis, receipt)


def test_graph_recovers_two_coactivation_blocks_and_does_not_link_them():
    hits = torch.zeros(256, 14)
    hits[:128, 0:3] = 1
    hits[128:, 7:10] = 1
    groups = communities_from_selection(hits)
    assert [0, 1, 2] in [g["nodes"] for g in groups]
    assert [7, 8, 9] in [g["nodes"] for g in groups]
    assert all(len(g["passes"]) == 1 for g in groups)


def test_null_graph_no_edges_at_impossible_positive_threshold():
    g = torch.Generator().manual_seed(22)
    hits = (torch.rand(256, 16, generator=g) > 0.7).float()
    assert communities_from_selection(hits, min_phi=1.01) == []


def test_masks_reject_duplicate_or_out_of_range_indices():
    for bad in ([0, 0], [-1], [PASSES * WIDTH]):
        try:
            node_mask(bad)
        except ValueError:
            continue
        raise AssertionError("expected failure for invalid nodes")


def test_matching_preserves_pass_profile_and_calibration_rms():
    acts = torch.ones(48, PASSES, WIDTH)
    v = torch.eye(WIDTH)
    u = torch.eye(WIDTH)
    nodes = [0, 3, WIDTH + 1, 2 * WIDTH + 2]
    target = direct_write_rms(acts, v, u, nodes)
    selected = choose_matched_shams(nodes, acts, v, u, 123)
    assert selected["eligible"]
    assert len(selected["shams"]) == 8
    for sham in selected["shams"]:
        assert sham["nodes"] != nodes
        assert sum(i // WIDTH == 0 for i in sham["nodes"]) == 2
        assert sum(i // WIDTH == 1 for i in sham["nodes"]) == 1
        assert sum(i // WIDTH == 2 for i in sham["nodes"]) == 1
        assert abs(sham["discovery_dose_rms"] - target) <= 1e-6


def test_training_and_discovery_datasets_use_separate_streams():
    a = sample_x(100, 2, 16)
    b = sample_x(100, 3, 16)
    c = sample_x(100, 4, 16)
    assert not torch.equal(a, b)
    assert not torch.equal(b, c)


def test_native_attribution_and_closed_loop_intervention_execute():
    teacher, model, training = train_student(100, steps=4)
    wrapper = BPDModel(model, {"mix": WIDTH}, max_passes=PASSES,
                       ablation=AblationConfig(mode="zero", sphere_rescale=False))
    set_basis(wrapper, "native", 100)
    x = sample_x(100, 3, 16)
    y = labels_from_teacher(teacher, x)
    picked = attribution_selection(wrapper, model, x, y, batch_size=8)
    assert picked.shape == (16, PASSES * WIDTH)
    assert torch.isfinite(picked).all()
    assert picked.sum(dim=1).max() <= 8
    effect = evaluate_masks(wrapper, model, x, y, [0, WIDTH])
    assert set(effect) == {"teacher_label_accuracy_clean", "ablate", "retain"}
    assert len(effect["ablate"]["kl_by_pass_nats"]) == PASSES
    assert effect["ablate"]["closed_loop_write_delta_rms"] > 0


def test_sparse_three_node_chain_is_a_valid_connected_community():
    # A and C are independent; B = A OR C, yielding edges A-B and B-C
    # without the A-C edge. Real discovery graphs need not be cliques.
    hits = torch.zeros(256, 3)
    hits[:48, 0] = 1
    hits[48:96, 2] = 1
    hits[96:112, 0] = 1
    hits[96:112, 2] = 1
    hits[:, 1] = (hits[:, 0] + hits[:, 2] > 0).float()
    groups = communities_from_selection(hits)
    assert len(groups) == 1
    assert groups[0]["nodes"] == [0, 1, 2]
    assert groups[0]["internal_phi_mass"] > 0.0


def test_float64_kl_audit_does_not_invent_parity_failure_for_tiny_logits():
    clean = torch.tensor([[0.5, 1.0, -1.0]], dtype=torch.float32)
    near = (clean.double() + 7.45e-8).float()
    assert float(binary_kl(clean, near).abs().max()) < 1e-12
    assert float(binary_kl(clean, clean).abs().max()) < 1e-12
