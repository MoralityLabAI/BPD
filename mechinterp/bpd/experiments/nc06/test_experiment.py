"""NC-06 excluded-root tests: source-only estimator, label separation and physics."""
from __future__ import annotations

import hashlib

import torch

from mechinterp.bpd.experiments.nc06.experiment import (
    DEV_ROOTS, SMOKE, PASSES, WIDTH,
    train_model, norm_weights, shuffled_weights,
    eigen_projectors, source_projectors, evaluate_root,
)


def test_frozen_development_roster():
    assert DEV_ROOTS==tuple(range(600,608))
    assert SMOKE==599 and SMOKE not in DEV_ROOTS
    assert PASSES==4 and WIDTH==20


def test_observed_only_norm_weights_and_exact_histogram():
    gen=torch.Generator().manual_seed(17)
    activ=torch.randn((96,PASSES,WIDTH),generator=gen)
    w,source=norm_weights(activ)
    assert w.shape==(96,4)
    assert source["source_pass_mean_energy"] and bool((w>0).all())
    perm,hashes=shuffled_weights(w,SMOKE)
    assert len(hashes)==4
    assert all(len(h)==64 for h in hashes)
    for k in range(PASSES):
        assert torch.equal(torch.sort(w[:,k]).values,
                           torch.sort(perm[:,k]).values)
        assert torch.allclose(
            w[:,k].sum().square()/w[:,k].square().sum(),
            perm[:,k].sum().square()/perm[:,k].square().sum(),
            atol=1e-11
        )
    again,_=shuffled_weights(w,SMOKE)
    assert torch.equal(perm,again)


def test_covariance_rank_gauge_and_equal_weight_reference():
    g=torch.Generator().manual_seed(41)
    grads=torch.randn((128,PASSES,WIDTH),generator=g)
    ones=torch.ones(128,PASSES,dtype=torch.float64)
    p3,b3,diag=eigen_projectors(grads,ones,3)
    assert b3.shape==(4,20,3)
    assert p3 is not None and p3.shape==(4,20,20)
    assert diag["projector_idempotence_max"]<1e-10
    assert diag["projector_symmetry_max"]<1e-10
    p4,b4,diag4=eigen_projectors(grads,ones,4)
    assert p4 is not None and b4.shape==(4,20,4)
    assert all(v["eigengap"]>=1.001 for v in diag["spectrum"])


def test_only_excluded_smoke_seed_receives_training():
    model,training=train_model(SMOKE,steps=2)
    assert training["seed"]==SMOKE
    assert training["steps"]==2
    assert model.mix.weight.shape==(WIDTH,WIDTH)
    try:
        train_model(508,steps=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Historical NC05 seed permitted")


def test_native_recurrent_smoke_and_separate_heldout_targets():
    record=evaluate_root(SMOKE,steps=8,discovery_size=96,
                         eval_size=96,proposals=64,shams=2)
    assert record["root"]==SMOKE
    assert record["phase"]=="ENGINEERING"
    assert record["discovery_examples"]==96
    assert record["evaluation_examples"]==96
    assert record["clean_parity"]["passed"]
    assert len(record["methods"])==4
    assert record["source_sha256"]
    for name,arm in record["methods"].items():
        assert arm["rank"] in (3,4)
        assert len(arm["diagnostic"]["spectrum"])==4
        assert len(arm["diagnostic"]["weight_effective_sample_fraction_by_pass"])==4
        if arm.get("status")=="ELIGIBLE":
            assert len(arm["sham_nlls"])==2 if arm["rank"]==3 else True
            assert len(arm["scored"]["kl_by_pass_nats"])==4
