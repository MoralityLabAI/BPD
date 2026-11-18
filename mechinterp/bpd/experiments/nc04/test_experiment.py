"""NC-04 audit: data blindness, full-model parity, learned dense weights and gauge."""
from __future__ import annotations

import torch
import pytest

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import (
    PASSES, WIDTH, sample_x, external_labels, parity,
)
from mechinterp.bpd.experiments.nc03.experiment import masked_rollout, basis_transport
import mechinterp.bpd.experiments.nc04.experiment as nc04
from mechinterp.bpd.experiments.nc04.experiment import (
    ROOTS, DenseLooper, factor_basis, self_score_discovery,
    self_gradient_projectors, train_model, evaluate_labels, evaluate_root,
)


def setup_model(seed=400, steps=8):
    model, record = train_model(seed, steps=steps)
    bpd = BPDModel(model, {"mix": WIDTH}, max_passes=PASSES,
                   ablation=AblationConfig(mode="zero", sphere_rescale=False))
    return model, bpd, record


def test_trainable_dense_shared_weights_have_no_planted_structural_mask():
    model, _, report = setup_model(400, 8)
    assert not hasattr(model, "structural_mask")
    assert model.mix.weight.shape == (WIDTH, WIDTH)
    assert report["n_trainable_parameters"] > WIDTH * WIDTH
    assert report["mix_weight_density_abs_ge_1e-4"] >= .90


def test_all_pass_native_and_rotated_basis_full_mask_closure():
    model, bpd, _ = setup_model()
    x = sample_x(400,3,48)
    for kind in ("native", "rotated"):
        factor_basis(bpd,kind,400)
        assert parity(bpd,model,x)["passed"], kind


def test_self_scored_discovery_cannot_query_external_labels(monkeypatch):
    model, bpd, _ = setup_model()
    factor_basis(bpd, "native", 400)
    def forbidden(*args, **kwargs):
        raise AssertionError("discovery attempted external_labels")
    monkeypatch.setattr(nc04, "external_labels", forbidden)
    x = sample_x(400,2,64)
    acts, grad, selected, score_info = self_score_discovery(bpd,model,x,32)
    assert acts.shape == grad.shape == (64,PASSES,WIDTH)
    assert selected.shape == (64,PASSES*WIDTH)
    assert selected.sum(dim=-1).max() <= 8
    assert score_info["model_score_nonzero_count"] == 64


def test_uncentered_second_moment_is_invariant_under_any_per_sample_sign():
    g = torch.randn(41,PASSES,WIDTH,dtype=torch.float64)
    signs = torch.where(torch.arange(41)%2==0,1.,-1.).reshape(-1,1,1)
    a = torch.einsum("nti,ntj->tij",g,g)
    b = torch.einsum("nti,ntj->tij",g*signs,g*signs)
    assert torch.allclose(a,b,atol=1e-12)


def test_exact_projector_gauge_transport_reproduces_closed_loop_output():
    model,bpd,_=setup_model()
    x=sample_x(400,3,32)
    q,_=torch.linalg.qr(torch.randn(WIDTH,2,dtype=torch.float64))
    p=(q@q.T).expand(PASSES,-1,-1).contiguous()
    factor_basis(bpd,"native",400)
    baseline,_=masked_rollout(bpd,model,x,p,mode="retain")
    factor_basis(bpd,"rotated",400)
    transported=basis_transport(p,bpd.wrappers["mix"].V)
    rotated,_=masked_rollout(bpd,model,x,transported,mode="retain")
    assert (baseline-rotated).abs().max() < 5e-5


def test_zero_gradients_and_near_degenerate_projectors_are_rejected():
    empty=torch.zeros(16,PASSES,WIDTH,dtype=torch.float64)
    p,diag=self_gradient_projectors(empty)
    assert p is None and diag["status"]=="UNIDENTIFIABLE"
    g=torch.zeros_like(empty)
    g[:,:,3]=1.
    p,diag=self_gradient_projectors(g)
    assert p is None


def test_full_rank_projector_gives_exact_clean_model_on_training_free_weights():
    model,bpd,_=setup_model()
    factor_basis(bpd,"native",400)
    x=sample_x(400,3,32)
    p=torch.eye(WIDTH,dtype=torch.float64).expand(PASSES,-1,-1).contiguous()
    score=evaluate_labels(bpd,model,x,external_labels(x),p)
    assert score["arms"]["retain"]["final_kl_nats"] < 1e-10
    assert len(score["arms"]["ablate"]["kl_by_pass_nats"]) == PASSES


def test_split_isolation():
    a=sample_x(400,1,32)
    b=sample_x(400,2,32)
    c=sample_x(400,3,32)
    assert not torch.equal(a,b) and not torch.equal(b,c)


def test_registered_roots_only_and_no_confirmation_entrypoint():
    assert ROOTS == tuple(range(400,408))
    with pytest.raises(ValueError):
        train_model(300,steps=1)
    with pytest.raises(ValueError):
        evaluate_root(408,steps=1,discovery_size=4,evaluation_size=4)
