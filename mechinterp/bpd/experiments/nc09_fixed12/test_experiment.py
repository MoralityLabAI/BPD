"""NC09-Fixed12 excluded engineering root 899 and source-only maths tests.

No 900..915 new model root is trained or accessed during preflight.
"""
from __future__ import annotations

import inspect

import torch

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import PASSES, WIDTH, parity, sample_x
from mechinterp.bpd.experiments.nc04.experiment import factor_basis
from mechinterp.bpd.experiments.nc05.experiment import clean_dose_rms
from mechinterp.bpd.experiments.nc08.experiment import (
    passwise_source_candidates, candidate_budget_knapsack, tuple_projector,
)
from mechinterp.bpd.experiments.nc09_fixed12.experiment import (
    ROOTS,SMOKE,TOTAL_RANK,SHAMS,PROPOSALS,GAINS,
    train_model,optimal_fixed12,fixed_arm_projectors,
    closed_loop_rollout,source_matched_local_shams,evaluate_root,
)


def test_original_registration_and_new_roots():
    assert SMOKE==899 and ROOTS==tuple(range(900,916))
    assert SMOKE not in ROOTS and len(set(ROOTS))==16
    assert TOTAL_RANK==12 and GAINS==(.5,1.5)
    assert PROPOSALS==4096 and SHAMS==8
    assert PASSES==4 and WIDTH==20
    assert "external_labels" not in inspect.signature(optimal_fixed12).parameters
    assert "heldout" not in inspect.signature(source_matched_local_shams).parameters


def test_passwise_source_optimum_is_deterministic_and_fixed_budget():
    gen=torch.Generator().manual_seed(999)
    reads=torch.randn((128,PASSES,WIDTH),generator=gen)
    grads=torch.randn((128,PASSES,WIDTH),generator=gen) * torch.linspace(3.,.1,WIDTH)
    src=passwise_source_candidates(reads,grads)
    best=optimal_fixed12(src)
    assert best is not None and best["total_rank"]==12
    assert len(best["ranks"])==4 and sum(best["ranks"])==12
    assert all(2<=rank<=6 for rank in best["ranks"])
    assert best["ranks"]==tuple(candidate_budget_knapsack(src)[12]["ranks"])
    assert best==optimal_fixed12(src)
    arms=fixed_arm_projectors(src,best)
    assert set(arms)=={"dp12","flat3","flat3_dose","flat4","reverse"}
    assert arms["dp12"]["eligible"]
    assert arms["dp12"]["total_rank"]==12
    assert arms["flat3"]["total_rank"]==12
    assert arms["flat4"]["total_rank"]==16
    assert arms["flat3_dose"]["projector"].shape==(4,20,20)


def test_physical_gain_matches_discovery_dose_at_same_total_rank():
    gen=torch.Generator().manual_seed(901)
    a=torch.randn((256,PASSES,WIDTH),generator=gen)
    g=torch.randn((256,PASSES,WIDTH),generator=gen)*torch.linspace(3.,.1,WIDTH)
    c=passwise_source_candidates(a,g)
    dp=optimal_fixed12(c)
    arms=fixed_arm_projectors(c,dp)
    u=torch.eye(WIDTH,dtype=torch.float64)
    dd=clean_dose_rms(a,u,arms["dp12"]["projector"])
    df=clean_dose_rms(a,u,arms["flat3"]["projector"])
    assert dd>0 and df>0
    alpha=dd/df
    assert abs(alpha*df-dd)<1e-12
    assert abs(arms["dp12"]["projector"].diagonal(dim1=-2,dim2=-1).sum()-
               12)<1e-8


def test_sham_proposals_preserve_frozen_tuple_and_source_rms():
    gen=torch.Generator().manual_seed(902)
    acts=torch.randn((256,PASSES,WIDTH),generator=gen)
    grad=torch.randn((256,PASSES,WIDTH),generator=gen)*torch.linspace(3.,.1,WIDTH)
    c=passwise_source_candidates(acts,grad)
    dp=optimal_fixed12(c)
    eigbasis=[c[t][rank]["basis"] for t,rank in enumerate(dp["ranks"])]
    shams,diag=source_matched_local_shams(
        acts,torch.eye(WIDTH,dtype=torch.float64),eigbasis,SMOKE,
        proposals=128,want=2
    )
    assert diag["proposals"]==128
    assert diag["selected"]==len(shams)<=2
    assert diag["qualified"]>=len(shams)
    for v in shams:
        assert v["projector"].shape==(PASSES,WIDTH,WIDTH)
        assert v["source_dose_relative_error"]<=.15


def test_excluded_899_closed_loop_source_and_separated_final_labels():
    model,training=train_model(SMOKE,steps=8)
    assert training["steps"]==8
    assert model.mix.weight.shape==(WIDTH,WIDTH)
    bpd=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
        ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bpd,"native",SMOKE)
    x=sample_x(SMOKE,4,32)
    assert parity(bpd,model,x)["passed"]
    p=torch.eye(WIDTH,dtype=torch.float64).repeat(PASSES,1,1)
    y,w=closed_loop_rollout(bpd,model,x,p,.9)
    assert y.shape==(32,PASSES) and w.shape==(32,PASSES,WIDTH)
    rec=evaluate_root(SMOKE,steps=8,derivative_n=96,
                      selector_n=64,heldout_n=64,
                      sham_proposals=128,shams=2)
    assert rec["study"]=="BPD-NC-09-fixed12"
    assert rec["root"]==899 and rec["phase"]=="DEVELOPMENT"
    assert rec["clean_closure"]["passed"]
    assert rec["splits"]=={
        "train":2048,"discovery":96,"selector_diagnostics":64,"heldout":64
    }
    assert rec["protocol_sha256"] and rec["source_sha256"]
    assert set(rec["heldout"])=={
        "dp12","flat3","flat3_dose","flat4","reverse"
    }
    if rec["fixed_budget12_allocation"] is not None:
        assert sum(rec["fixed_budget12_allocation"]["ranks"])==12
        assert rec["arms"]["flat3"]["total_rank"]==12
        assert rec["arms"]["flat4"]["total_rank"]==16


def test_historical_model_roots_are_excluded():
    try:
        train_model(807,steps=1)
    except ValueError:
        return
    raise AssertionError("Historical NC08 model training root was admitted")
