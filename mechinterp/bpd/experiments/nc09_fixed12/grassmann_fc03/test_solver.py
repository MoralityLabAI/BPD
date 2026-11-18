"""FC03 mathematical, independent-source matching and excluded-root smoke checks."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from mechinterp.bpd.experiments.nc02.experiment import WIDTH,PASSES
from mechinterp.bpd.experiments.nc09_fixed12.grassmann_fc03 import solver
from mechinterp.bpd.experiments.nc09_fixed12.grassmann_fc03.experiment import (
    SMOKE,ROOTS,PROTO_SHA,train_model,evaluate_calibration_root,
    protect_original_nc09_source,
)


def canonical_bases():
    return [torch.eye(WIDTH,dtype=torch.float64)[:,:r].clone()
            for r in (4,3,3,2)]


def clean_synthetic_source(n=256,seed=101):
    rng=torch.Generator().manual_seed(seed)
    a=torch.randn((n,PASSES,WIDTH),generator=rng)
    z=torch.randn((n,PASSES,WIDTH),generator=rng)
    g=torch.zeros_like(z)
    U=torch.eye(WIDTH)
    return a,U,z,g,canonical_bases()


def test_registered_roster_and_protected_physical_source():
    assert ROOTS==tuple(range(14000,14016))
    assert len(ROOTS)==16 and SMOKE==13999 and SMOKE not in ROOTS
    assert solver.CHART_STARTS==32
    assert solver.MAX_ADAM_STEPS==64
    assert solver.LEARNING_RATE==.08
    assert solver.ANGLE==.25
    assert solver.DP12_MIN_CHORDAL==.20
    assert solver.PAIRWISE_MIN_CHORDAL==.18
    assert solver.CHART_SEED_BASE==14_030_000
    assert len(PROTO_SHA)==64
    protect_original_nc09_source()


def test_exact_rank_constant_angle_and_minimal_normalized_chordal():
    E=canonical_bases()
    frames=solver.complement_frames(E,np.random.default_rng(71))
    params=torch.zeros((PASSES,2),dtype=torch.float64)
    B=solver.chart_bases(E,frames,params)
    P=solver.projectors_from_bases(B)
    target=solver.projectors_from_bases(E)
    assert P.shape==(PASSES,WIDTH,WIDTH)
    assert abs(float(solver.chordal(P,target))-math.sin(.25))<1e-10
    for t,b in enumerate(B):
        rank=E[t].shape[1]
        assert (b.T@b-torch.eye(rank,dtype=torch.float64)).abs().max()<1e-9
        assert abs(float(torch.trace(P[t]))-rank)<1e-9
    assert (P@P-P).abs().max()<1e-9
    assert (P-P.transpose(-1,-2)).abs().max()<1e-9


def test_synthetic_zero_score_source_admits_eight_distinct_nontrivial_shams():
    a,u,z,g,E=clean_synthetic_source(n=384)
    shams,record=solver.fit_constrained_grassmann(
        a,u,z,g,E,seed=13999,starts=32,steps=4,want=8
    )
    assert record["admitted"]
    assert record["selected"]==8 and len(shams)==8
    assert record["charts_configured"]==32
    assert len(record["charts"])==32
    assert not record["uses_external_labels"]
    assert all(x["optimization_step"]<=4 for x in record["accepted"])
    target=solver.projectors_from_bases(E)
    for i,P in enumerate(shams):
        r=record["accepted"][i]
        assert float(solver.chordal(P,target))>=.20
        for j in range(i):
            assert float(solver.chordal(P,shams[j]))>=.18
        assert r["physical_rms_relative_error"]<=.15+1e-12
        assert r["signed_score_abs_error"]<=record["source_signed_score_abs_tolerance"]+1e-12
        assert r["score_rms_abs_error"]<=record["source_score_rms_abs_tolerance"]+1e-12
        assert r["projector_idempotence_max"]<1e-8
        assert r["projector_symmetry_max"]<1e-8


def test_zero_physical_source_is_declared_ineligible():
    a,u,z,g,E=clean_synthetic_source(n=32)
    out,report=solver.fit_constrained_grassmann(
        a*0,u,z,g,E,seed=13999,starts=2,steps=1,want=1
    )
    assert not out and report["source_zero_or_invalid"]


def test_native_four_step_excluded_13999_smoke_without_calibration_roots():
    model,training=train_model(SMOKE,steps=2)
    assert model.mix.weight.shape==(WIDTH,WIDTH)
    assert training["root"]==SMOKE and training["steps"]==2
    try:
        train_model(13000,steps=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Historical FC02 training root was admitted")
    receipt=evaluate_calibration_root(
        SMOKE,steps=8,discovery_n=96,selector_n=64,heldout_n=64,
        sham_proposals=64,shams=2,solver_starts=4,solver_steps=3
    )
    assert receipt["study"]=="BPD-NC09-FC03"
    assert receipt["phase"]=="CONSTRAINED_GRASSMANN_ENGINEERING_DEVELOPMENT"
    assert receipt["root"]==SMOKE and receipt["full_mask_closure"]["passed"]
    assert receipt["splits"]=={
        "train":2048,"derivative_discovery":96,
        "model_validation":64,"labeled_calibration":64
    }
    assert receipt["source_rank12_optimum"] is not None
    assert sum(receipt["source_rank12_optimum"]["ranks"])==12
    assert receipt["sham_admission"]["charts_configured"]==4
    assert receipt["fc02_original_rejection_admission"]["proposals"]==64
    assert receipt["source_dp12_projector_receipt"] is not None
    assert receipt["original_fc02_hold_unchanged"]=="CALIBRATION_INCOMPLETE"
