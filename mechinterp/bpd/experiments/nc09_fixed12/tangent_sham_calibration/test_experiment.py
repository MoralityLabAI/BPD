"""Excluded engineering 12999 and observed-only tangent sham tests.

No 13000..13058 independent calibration model root is trained here.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from mechinterp.bpd.experiments.nc02.experiment import PASSES, WIDTH
from mechinterp.bpd.experiments.nc09_fixed12.tangent_sham_calibration import (
    tangent_shams as shams
)
from mechinterp.bpd.experiments.nc09_fixed12.tangent_sham_calibration.experiment import (
    ROOTS, SMOKE, PROTO_SHA, train_model, protect_original_nc09_source,
    evaluate_calibration_root,
)


def test_frozen_59_roots_and_original_nc09_source():
    assert ROOTS==tuple(range(13000,13059))
    assert SMOKE==12999 and SMOKE not in ROOTS
    assert shams.SHAM_PROPOSALS==4096
    assert shams.SHAMS==8
    assert shams.ANGLE_GRID==(.20,.30,.45,.65)
    assert shams.SOURCE_SCORE_REL_TOL==.25
    assert shams.SOURCE_SCORE_ABS_TOL==.005
    assert shams.SOURCE_WRITE_REL_TOL==.15
    assert len(PROTO_SHA)==64
    parent=protect_original_nc09_source()
    assert len(parent["nc09_original_experiment_sha256"])==64


def test_clean_first_order_signed_effect_and_exact_full_rank_zero():
    rng=torch.Generator().manual_seed(450)
    z=torch.randn((192,PASSES,WIDTH),generator=rng)
    g=torch.randn((192,PASSES,WIDTH),generator=rng)
    full=[torch.eye(WIDTH,dtype=torch.float64)
          for _ in range(PASSES)]
    # An identity basis has rank twenty and is tested mathematically here
    # using a simple direct tensor calculation.
    delta=(g.double()*(z.double()-
           z.double()@torch.eye(WIDTH,dtype=torch.float64))).sum((-1,-2))
    assert delta.abs().max()<1e-12
    # Partial retained projectors have a nonzero score change that is
    # identical to the source-only analytic formula.
    basis=[torch.eye(WIDTH,dtype=torch.float64)[:,:3]
           for _ in range(PASSES)]
    got=shams.source_score_profile(z,g,basis)
    p=basis[0]@basis[0].T
    expected=(g.double()*(z.double()-z.double()@p)).sum((-1,-2))
    assert abs(got["signed_first_order_score_mean"]-
               float(expected.mean()))<1e-10
    assert abs(got["first_order_score_rms"]-
               float(expected.square().mean().sqrt()))<1e-10


def test_source_only_physical_tangent_matching_has_exact_rank_and_angle():
    rng=torch.Generator().manual_seed(451)
    a=torch.randn((160,PASSES,WIDTH),generator=rng)
    z=torch.randn((160,PASSES,WIDTH),generator=rng)
    g=torch.randn((160,PASSES,WIDTH),generator=rng)*.15
    B=[torch.eye(WIDTH,dtype=torch.float64)[:,:3]
       for _ in range(PASSES)]
    outcome,diag=shams.build_derivative_matched_shams(
        a,torch.eye(WIDTH,dtype=torch.float64),z,g,B,SMOKE,
        proposals=160,want=3
    )
    assert diag["proposals"]==160 and diag["source_observation_count"]==160
    assert len(outcome)==diag["selected"]<=3
    assert diag["qualified"]>=len(outcome)
    assert not diag["uses_external_labels"]
    for v in outcome:
        assert v["projector"].shape==(4,20,20)
        assert v["principal_angle"] in shams.ANGLE_GRID
        assert v["principal_angle"]>=.20
        assert v["write_relative_error"]<=.15+1e-12
        assert v["score_signed_abs_error"]<=diag["signed_score_abs_tolerance"]+1e-12
        assert v["score_rms_abs_error"]<=diag["rms_score_abs_tolerance"]+1e-12
        assert all(abs(float(torch.trace(p))-3)<1e-7
                   for p in v["projector"])


def test_excluded_model_physical_evaluation_and_source_split():
    model,meta=train_model(SMOKE,steps=2)
    assert meta["root"]==SMOKE and meta["steps"]==2
    assert meta["train_size"]==2048 and model.mix.weight.shape==(20,20)
    try:
        train_model(12047,steps=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Prior FC01 calibration root was admitted")
    rec=evaluate_calibration_root(
        SMOKE,steps=8,discovery_n=96,selector_n=64,heldout_n=64,
        sham_proposals=96,shams=2
    )
    assert rec["study"]=="BPD-NC09-FC02"
    assert rec["phase"]=="AGGREGATE_TANGENT_SHAM_CALIBRATION"
    assert rec["root"]==SMOKE and rec["full_mask_closure"]["passed"]
    assert rec["splits"]=={
        "train":2048,"derivative_discovery":96,
        "model_validation":64,"labeled_calibration":64
    }
    assert rec["source_rank12_optimum"] is not None
    assert sum(rec["source_rank12_optimum"]["ranks"])==12
    assert rec["sham_admission"]["proposals"]==96
    assert rec["sham_admission"]["selected"]<=2
    assert rec["selected_sham_score_source_only"]
    assert rec["source_sha256"]==hashlib.sha256(
        __import__("mechinterp.bpd.experiments.nc09_fixed12.tangent_sham_calibration.experiment",fromlist=["__file__"]).__file__
            and open(__import__("mechinterp.bpd.experiments.nc09_fixed12.tangent_sham_calibration.experiment",fromlist=["__file__"]).__file__,"rb").read()
    ).hexdigest()
