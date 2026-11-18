"""NC-05 source-gated tests. No confirmation cohorts or data are used."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import (
    PASSES, WIDTH, sample_x, external_labels, parity,
)
from mechinterp.bpd.experiments.nc04.experiment import (
    factor_basis, self_score_discovery,
)
from mechinterp.bpd.experiments.nc03.experiment import basis_transport
from mechinterp.bpd.experiments.nc05.aggregate import (
    aggregate, family_bootstrap, FAMILY,
)
import mechinterp.bpd.experiments.nc05.experiment as nc05
from mechinterp.bpd.experiments.nc05.experiment import (
    ROOTS, RANKS, GAINS, PROTOCOL_COMMIT, SHAMS,
    train_model, projectors_at_ranks, local_dose_matched_shams,
    projected_rollout, score_keep,
)


def wrapped(steps=4, seed=500):
    model, record=train_model(seed,steps=steps)
    bpd=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                 ablation=AblationConfig(mode="zero",sphere_rescale=False))
    return model,bpd,record


def test_registered_roots_grid_and_family_stay_fixed():
    assert ROOTS == tuple(range(500,508))
    assert RANKS == (2,4,6,8)
    assert GAINS == (0.75,1.0)
    assert FAMILY == 8
    assert PROTOCOL_COMMIT == "f7f1d2346f9d3f871049a3d03416ef7980fe90f8"
    for wrong in (400, 499, 508):
        with pytest.raises(ValueError):
            train_model(wrong,steps=1)


def test_trainable_dense_recurrence_is_not_stage_constrained():
    model,bpd,training=wrapped()
    assert not hasattr(model,"structural_mask")
    assert training["mix_weight_density"] >=0.90
    assert bpd.wrappers["mix"].W_target.shape ==(WIDTH,WIDTH)
    assert training["n_trainable_parameters"] > WIDTH*WIDTH


def test_fixed_rank_projectors_are_symmetric_idempotent_nested():
    torch.manual_seed(503)
    grad=torch.randn(512,PASSES,WIDTH,dtype=torch.float64)
    projectors, details, bases=projectors_at_ranks(grad)
    for rank in RANKS:
        p=projectors[rank]
        assert p is not None, details[rank]
        assert p.shape==(PASSES,WIDTH,WIDTH)
        assert details[rank]["idempotence_max_abs"]<1e-10
        assert torch.allclose(p,p.transpose(-1,-2),atol=1e-10)
        assert torch.allclose(torch.diagonal(p,dim1=-2,dim2=-1).sum(-1),
                              torch.full((PASSES,),float(rank),dtype=torch.float64),
                              atol=1e-8)
        assert bases[rank].shape==(PASSES,WIDTH,rank)
    assert torch.allclose(projectors[4]@projectors[2],projectors[2],atol=1e-8)
    assert torch.allclose(projectors[8]@projectors[6],projectors[6],atol=1e-8)


def test_negative_readout_gradient_blocks_every_rank():
    zero=torch.zeros(16,PASSES,WIDTH,dtype=torch.float64)
    p,diags,_=projectors_at_ranks(zero)
    assert all(p[k] is None for k in RANKS)
    assert all(diags[k]["status"]=="UNIDENTIFIABLE" for k in RANKS)


def test_local_projector_shams_are_rank_preserving_unique_and_dose_matched():
    torch.manual_seed(1001)
    a=torch.randn(1024,PASSES,WIDTH,dtype=torch.float64)
    u=torch.eye(WIDTH,dtype=torch.float64)
    for rank in RANKS:
        q,_=torch.linalg.qr(torch.randn(PASSES,WIDTH,rank,dtype=torch.float64))
        shams, info=local_dose_matched_shams(a,u,q,rank,500,
                                            proposals=128,want=SHAMS)
        assert info["eligible"], (rank,info)
        assert len(shams)==SHAMS
        assert len(set(info["sham_proposal_indices"]))==SHAMS
        for p,rel in zip(shams,info["sham_dose_errors"]):
            assert p.shape==(PASSES,WIDTH,WIDTH)
            assert float((p@p-p).abs().max())<1e-8
            assert abs(float(torch.diagonal(p,dim1=-2,dim2=-1).sum(-1).mean())-rank)<1e-8
            assert rel<=.15


def test_native_vs_rotated_weight_parity_and_gain_projector_causal_output():
    model,bpd,_=wrapped()
    x=sample_x(500,3,48)
    torch.manual_seed(88)
    pbase,_=torch.linalg.qr(torch.randn(PASSES,WIDTH,4,dtype=torch.float64))
    p=pbase@pbase.transpose(-1,-2)
    factor_basis(bpd,"native",500)
    assert parity(bpd,model,x)["passed"]
    for gain in GAINS:
        native,_=projected_rollout(bpd,model,x,p,gain)
        factor_basis(bpd,"rotated",500)
        assert parity(bpd,model,x)["passed"]
        transported=basis_transport(p,bpd.wrappers["mix"].V)
        rot,_=projected_rollout(bpd,model,x,transported,gain)
        assert float((native-rot).abs().max())<5e-5
        factor_basis(bpd,"native",500)


def test_full_rank_identity_gain_one_reconstructs_clean_target():
    model,bpd,_=wrapped()
    x=sample_x(500,3,32)
    y=external_labels(x)
    factor_basis(bpd,"native",500)
    with torch.no_grad():
        with bpd.target_capture():
            clean,_=model(x,capture=True)
            writes=torch.stack(model.writes,dim=1)
    p=torch.eye(WIDTH,dtype=torch.float64).expand(PASSES,-1,-1)
    scored=score_keep(bpd,model,x,y,clean,writes,p,gain=1.0)
    assert scored["final_kl_nats"]<1e-10
    assert abs(scored["nll_delta_from_clean_nats"])<1e-6
    assert scored["correct_count"] <= len(x)


def test_oracle_blind_discovery_uses_only_model_scores(monkeypatch):
    model,bpd,_=wrapped()
    factor_basis(bpd,"native",500)
    import mechinterp.bpd.experiments.nc04.experiment as old
    def forbidden(*args,**kwargs):
        raise AssertionError("external task oracle used during feature discovery")
    monkeypatch.setattr(old,"external_labels",forbidden)
    monkeypatch.setattr(nc05,"external_labels",forbidden)
    x=sample_x(500,2,32)
    _,gradient,_,details=self_score_discovery(bpd,model,x)
    assert details["independent_discovery_examples"]==32
    assert gradient.shape==(32,PASSES,WIDTH)


def test_root_bootstrap_counts_model_roots_and_respects_familywise_correction():
    result=family_bootstrap([1.,2.,3.,4.,5.,6.,7.,8.],17)
    assert result["n_independent_roots"]==8
    assert abs(result["mean_effect_nats"]-4.5)<1e-10
    assert math.isclose(result["familywise_coverage"],.99375,rel_tol=1e-12)
    assert result["familywise_ci"][0]<=4.5<=result["familywise_ci"][1]


def test_failed_scientific_frontier_still_serializes_finite_full_receipt():
    roots=[]
    for seed in ROOTS:
        roots.append({
            "study":"BPD-NC-05","phase":"ENGINEERING","seed":seed,
            "protocol_commit":PROTOCOL_COMMIT,
            "data":{"training":2048,"discovery":1024,"evaluation":1024},
            "training":{"training_steps":384},
            "task_passed":False,
            "negative_controls":{"zero_head":{
                "max_abs_grad":0.,"positive_hits":0,"all_ranks_unidentifiable":True
            }},
            "source_sha256":"test_source","protocol_sha256":"test_protocol",
            "bases":{
                "native":{"closure":{"passed":True},"rank_diagnostics":{}},
                "rotated":{"closure":{"passed":True},"rank_diagnostics":{}},
            },
            "cells":{},
        })
    summary=aggregate(roots)
    assert summary["development_resolution"]=="NOT_RESOLVED_OR_FAILED"
    assert summary["smallest_qualifying_full_gain_rank"] is None
    assert len(summary["cells"]) == 8
    json.dumps(summary,allow_nan=False)
