"""FC04 source-only full tangent Jacobian, independent rank geometry and excluded smoke."""
from __future__ import annotations

import math
import hashlib
from pathlib import Path

import numpy as np
import torch

from mechinterp.bpd.experiments.nc02.experiment import PASSES,WIDTH
from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_fc04 import solver
from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_fc04.experiment import (
    ROOTS,SMOKE,PROTO_SHA,train_model,evaluate_calibration_root,
    protect_original_nc09_source,
)


def native_bases():
    return [torch.eye(WIDTH,dtype=torch.float64)[:,:r].clone()
            for r in (4,3,3,2)]


def mock_data(n=128,seed=614):
    gen=torch.Generator().manual_seed(seed)
    a=torch.randn((n,PASSES,WIDTH),generator=gen)
    z=torch.randn((n,PASSES,WIDTH),generator=gen)
    g=torch.randn((n,PASSES,WIDTH),generator=gen)*.08
    return a,torch.eye(WIDTH,dtype=torch.float64),z,g,native_bases()


def test_fresh_source_roster_and_preregistered_solver():
    assert ROOTS==tuple(range(15000,15016)) and SMOKE==14999
    assert len(PROTO_SHA)==64
    assert solver.CHART_STARTS==32 and solver.MAX_UPDATES==64
    assert solver.CHART_SEED_BASE==15_040_000
    assert solver.LINE_SEARCH==(1.,.5,.25,.125)
    assert solver.STEP_NORM_MAX==.5
    assert solver.GAUSS_NEWTON_DAMPING==.001
    assert solver.ANGLE==.25 and solver.DP12_MIN_CHORDAL==.20
    assert solver.PAIRWISE_MIN_CHORDAL==.18
    protect_original_nc09_source()


def test_full_complement_dimension_fixed_angle_and_rank():
    E=native_bases()
    C=[solver.complementary_basis(e) for e in E]
    assert solver.unconstrained_parameter_count(E)==(
        16*4+17*3+17*3+18*2
    )
    rng=np.random.default_rng(11)
    y=torch.tensor(np.concatenate([
        rng.standard_normal((20-e.shape[1])*e.shape[1])/
        math.sqrt(20-e.shape[1]) for e in E
    ]),dtype=torch.float64)
    p=solver.projectors_from_bases(solver.tangent_chart(E,C,y))
    source=solver.projectors_from_bases(E)
    assert abs(float(solver.chordal(p,source))-math.sin(.25))<1e-10
    for t in range(PASSES):
        rank=E[t].shape[1]
        assert abs(float(torch.trace(p[t]))-rank)<1e-8
    assert float((p@p-p).abs().max())<1e-8


def test_true_full_source_moment_jacobian_correction_direction():
    a,U,z,g,E=mock_data()
    z=z.double();g=g.double();a=a.double()
    C=[solver.complementary_basis(e) for e in E]
    target,tolerance,moments,gram=solver.target_values(z,g,a,U,E)
    gen=np.random.default_rng(15)
    y=torch.tensor(np.concatenate([
        gen.standard_normal((20-e.shape[1])*e.shape[1])/
        math.sqrt(20-e.shape[1]) for e in E
    ]),dtype=torch.float64,requires_grad=True)
    b=solver.tangent_chart(E,C,y)
    p=solver.source_tensor_profile(z,g,b,moments,gram)
    residual=solver.normalized_source_residual(p,target,tolerance)
    delta,diagnostics=solver.minimum_norm_jacobian_step(residual,y)
    assert delta.shape==y.shape and float(delta.norm())<=.5+1e-10
    assert 0<=diagnostics["jacobian_rank"]<=3
    assert len(diagnostics["jacobian_singular_values"])==3
    assert all(np.isfinite(x) for x in delta.detach().numpy())
    # The local Gauss-Newton step reduces squared normalized source residual
    # on an ordinary fully ranked source at a sufficiently small scale.
    current=float(residual.square().sum())
    trial=y.detach()+.125*delta
    b2=solver.tangent_chart(E,C,trial)
    p2=solver.source_tensor_profile(z,g,b2,moments,gram)
    updated=float(solver.normalized_source_residual(p2,target,tolerance).square().sum())
    assert updated<current


def test_eight_nontrivial_shams_from_zero_score_synthetic_source():
    a,U,z,g,E=mock_data(n=256,seed=22)
    g=torch.zeros_like(g)
    selected,receipt=solver.fit_full_tangent_shams(
        a,U,z,g,E,SMOKE,starts=12,steps=5,want=8
    )
    assert receipt["eligible"]
    assert len(selected)==8 and receipt["selected"]==8
    assert len(receipt["charts"])==12
    assert not receipt["uses_external_task_labels"]
    assert receipt["full_tangent_parameter_count"]==202
    p0=solver.projectors_from_bases(E)
    for i,P in enumerate(selected):
        assert float(solver.chordal(P,p0))>=.20
        for j in range(i):
            assert float(solver.chordal(P,selected[j]))>=.18
        entry=receipt["projectors"][i]
        assert entry["physical_rms_relative_error"]<=.15+1e-9
        assert entry["score_rms_abs_error"]<=receipt["source_score_rms_abs_tolerance"]+1e-9
        assert entry["signed_score_abs_error"]<=receipt["source_signed_score_abs_tolerance"]+1e-9
        assert entry["correction_updates"]<=5


def test_new_source_guard_and_native_excluded_14999_model():
    model,stats=train_model(SMOKE,steps=2)
    assert stats["root"]==SMOKE and stats["steps"]==2
    assert model.mix.weight.shape==(20,20)
    try:
        train_model(14001,steps=1)
    except ValueError:
        pass
    else:
        raise AssertionError("FC03 source root retraining was permitted")
    result=evaluate_calibration_root(
        SMOKE,steps=8,discovery_n=96,selector_n=64,heldout_n=64,
        sham_proposals=48,shams=2,solver_starts=4,solver_steps=3
    )
    assert result["study"]=="BPD-NC09-FC04"
    assert result["phase"]=="FULL_TANGENT_JACOBIAN_ENGINEERING_DEVELOPMENT"
    assert result["root"]==SMOKE
    assert result["full_mask_closure"]["passed"]
    assert result["splits"]=={
        "train":2048,"derivative_discovery":96,
        "model_validation":64,"labeled_calibration":64
    }
    assert result["sham_admission"]["charts_configured"]==4
    assert result["fc03_original_eightdim_admission"]["charts_configured"]==4
    assert result["fc02_original_rejection_admission"]["proposals"]==48
    assert result["original_fc01_hold_unchanged"]=="CALIBRATION_REDESIGN_OR_PILOT_ONLY"
    assert result["original_fc02_hold_unchanged"]=="CALIBRATION_INCOMPLETE"
    assert result["original_fc03_hold_unchanged"]=="ENGINEERING_NULL_OR_INELIGIBLE"
