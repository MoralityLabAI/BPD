"""BPD-NC-03 engineering assertions for gauge-covariant subspace discovery."""
from __future__ import annotations

import torch
import pytest

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import WIDTH, PASSES, train_model, sample_x, external_labels, parity
from mechinterp.bpd.experiments.nc03.experiment import (
    activation_and_gradients, assess, basis_transport, clean_write_dose,
    configure_basis, covariant_projectors, discovery_moments, evaluate_root,
    masked_rollout, oracle_diagnostics, physical_truth_projectors, sham_projectors,
)


def _wrapped(seed: int = 300, steps: int = 6):
    model, _ = train_model(seed, steps=steps)
    bpd = BPDModel(model, {"mix": WIDTH}, max_passes=PASSES,
                   ablation=AblationConfig(mode="zero", sphere_rescale=False))
    return model, bpd


def test_true_projector_rank_two_every_pass():
    p = physical_truth_projectors()
    assert p.shape == (PASSES, WIDTH, WIDTH)
    assert torch.allclose(p @ p, p)
    assert torch.allclose(p.transpose(-1,-2), p)
    assert torch.allclose(torch.diagonal(p,dim1=-2,dim2=-1).sum(-1),
                          torch.full((PASSES,),2.,dtype=p.dtype))


def test_gradient_pca_recovers_known_physical_path():
    model, bpd = _wrapped()
    x = sample_x(300, 2, 256)
    y = external_labels(x)
    configure_basis(bpd, model, "native", 300)
    acts, grad = activation_and_gradients(bpd, model, x, y)
    p, diag = covariant_projectors(grad)
    assert diag["status"] == "RANK_TWO_IDENTIFIABLE", diag
    assert acts.shape == (256,PASSES,WIDTH)
    assert oracle_diagnostics(p, bpd.wrappers["mix"].V)["min_overlap"] > .999999


def test_exact_orthogonal_transport_and_rank_two_overlap():
    model, bpd = _wrapped()
    x = sample_x(300, 2, 256)
    y = external_labels(x)
    configure_basis(bpd, model, "native", 300)
    _, g0 = activation_and_gradients(bpd, model, x, y)
    p0, _ = covariant_projectors(g0)
    configure_basis(bpd, model, "rotated", 300)
    _, g1 = activation_and_gradients(bpd, model, x, y)
    p1, _ = covariant_projectors(g1)
    v = bpd.wrappers["mix"].V
    assert oracle_diagnostics(p1, v)["min_overlap"] > .999999
    assert (p1 - basis_transport(p0,v)).abs().max() < 1e-5
    assert parity(bpd, model, x)["passed"]


def test_projector_constrained_rollout_is_closed_loop_gauge_invariant():
    model, bpd = _wrapped()
    x = sample_x(300, 3, 64)
    p_true = physical_truth_projectors()
    configure_basis(bpd, model, "native", 300)
    a0, _ = masked_rollout(bpd, model, x, p_true, mode="retain")
    configure_basis(bpd, model, "rotated", 300)
    p1 = basis_transport(p_true, bpd.wrappers["mix"].V)
    a1, _ = masked_rollout(bpd, model, x, p1, mode="retain")
    with bpd.target_capture():
        original, _ = model(x)
    assert (a0-original).abs().max() < 5e-5
    assert (a1-original).abs().max() < 5e-5
    assert (a0-a1).abs().max() < 5e-5


def test_zero_readout_has_no_identifiable_gradient_subspace():
    model, bpd = _wrapped()
    configure_basis(bpd, model, "native", 300)
    with torch.no_grad():
        model.readout.zero_()
    x = sample_x(300, 2, 32)
    _, g = activation_and_gradients(bpd, model, x, external_labels(x))
    p, diag = covariant_projectors(g)
    assert float(g.abs().max()) == 0.0
    assert p is None
    assert diag["status"] == "NO_IDENTIFIABLE_SUBSPACE"


def test_sham_sampler_stays_within_physical_dose_with_oracle_blind_inputs():
    model, bpd = _wrapped()
    configure_basis(bpd, model, "native", 300)
    x = sample_x(300, 2, 128)
    acts, grads = activation_and_gradients(bpd, model, x, external_labels(x))
    p, _ = covariant_projectors(grads)
    layer = bpd.wrappers["mix"]
    masks, info = sham_projectors(acts,layer.U.detach(),p,300,proposals=128,want=4)
    assert info["eligible"], info
    assert len(masks) == 4
    moments,b = discovery_moments(acts,layer.V.detach(),layer.U.detach())
    target = clean_write_dose(moments,b,p)
    for m in masks:
        assert ((m@m)-m).abs().max() < 1e-6
        assert ((m-m.transpose(-1,-2))).abs().max() < 1e-6
        assert abs(clean_write_dose(moments,b,m)/target-1) <= .15+1e-10
    assert len(set(info["selected_candidate_indices"])) == 4


def test_known_causal_projection_removes_nuisance_and_is_necessary():
    model,bpd = _wrapped()
    configure_basis(bpd,model,"native",300)
    x=sample_x(300,3,128)
    y=external_labels(x)
    measure=assess(bpd,model,x,y,physical_truth_projectors())
    assert measure["arms"]["retain"]["final_kl_nats"] <= 1e-7
    assert measure["arms"]["ablate"]["final_kl_nats"] > 0.01
    assert measure["arms"]["ablate"]["nll_delta_nats"] > 0.01


def test_eigenspectrum_rejects_null_and_unidentified_rank():
    zero=torch.zeros(50,PASSES,WIDTH,dtype=torch.float64)
    p,info=covariant_projectors(zero)
    assert p is None
    assert info["status"]=="NO_IDENTIFIABLE_SUBSPACE"
    # A single gradient direction across all samples is rank 1.
    zero[:,:,3]=1.
    p,info=covariant_projectors(zero)
    assert p is None


def test_phase_and_seed_gate_do_not_expose_confirmation_roots():
    with pytest.raises(ValueError):
        evaluate_root(999,steps=1,discovery_size=4,evaluation_size=4)
    with pytest.raises(ValueError):
        evaluate_root(200,steps=1,discovery_size=4,evaluation_size=4)
