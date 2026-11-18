"""NC08 source-only tests using excluded engineering seed 799 and fake receipts.

Never train or evaluate development roots 800..807 during preflight.
"""
from __future__ import annotations

import inspect
import math

import torch

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import PASSES,WIDTH,parity,sample_x
from mechinterp.bpd.experiments.nc04.experiment import factor_basis
from mechinterp.bpd.experiments.nc08.experiment import (
    ROOTS,SMOKE,OPTIONS,BUDGETS,SOURCE_THRESHOLDS,
    train_model,passwise_source_candidates,candidate_budget_knapsack,
    tuple_projector,source_selector,local_dose_matched_shams,
    evaluate_root
)


def test_frozen_source_and_budget_roster():
    assert ROOTS==tuple(range(800,808))
    assert SMOKE==799 and SMOKE not in ROOTS
    assert OPTIONS==(2,3,4,5,6) and BUDGETS==(8,10,12,14)
    assert PASSES==4 and WIDTH==20
    assert SOURCE_THRESHOLDS["mean_final_kl_max"]==.027
    assert SOURCE_THRESHOLDS["retained_full_write_fraction_min"]==.35
    assert "true_labels" not in inspect.signature(source_selector).parameters


def test_multiple_choice_knapsack_exact_and_ties():
    mock={t:{r:{"eligible":True,
                 "squared_local_omitted_model_score":(6-r)**2}
             for r in OPTIONS} for t in range(PASSES)}
    got=candidate_budget_knapsack(mock)
    assert tuple(got[8]["ranks"])==(2,2,2,2)
    assert tuple(got[10]["ranks"])==(2,2,3,3)
    assert tuple(got[12]["ranks"])==(3,3,3,3)
    assert tuple(got[14]["ranks"])==(3,3,4,4)
    assert all(sum(got[b]["ranks"])==b for b in BUDGETS)
    mock[0][2]["eligible"]=False
    assert got!=candidate_budget_knapsack(mock)
    no_qual={t:{r:{"eligible":False,
                     "squared_local_omitted_model_score":1}
               for r in OPTIONS} for t in range(PASSES)}
    assert all(v is None for v in candidate_budget_knapsack(no_qual).values())


def test_pass_specific_covariances_and_projectors():
    gen=torch.Generator().manual_seed(8081)
    z=torch.randn((256,PASSES,WIDTH),generator=gen)
    g=torch.randn((256,PASSES,WIDTH),generator=gen)
    g=g*torch.linspace(3.,.1,WIDTH)
    candidates=passwise_source_candidates(z,g)
    assert set(candidates)=={0,1,2,3}
    for t in range(PASSES):
        for r in OPTIONS:
            c=candidates[t][r]
            assert c["rank"]==r
            assert c["projector"].shape==(WIDTH,WIDTH)
            assert c["basis"].shape==(WIDTH,r)
            assert c["idempotence_max"]<1e-9
            assert c["squared_local_omitted_model_score"]>=0
    for b,v in candidate_budget_knapsack(candidates).items():
        if v is None:
            continue
        p=tuple_projector(candidates,v["ranks"])
        assert p.shape==(4,20,20)
        for t in range(PASSES):
            assert abs(float(torch.trace(p[t]))-v["ranks"][t])<1e-8


def test_source_only_selector_native_model_on_excluded_799():
    model,stats=train_model(SMOKE,steps=8)
    assert stats["seed"]==SMOKE and stats["steps"]==8
    xd=sample_x(SMOKE,2,64)
    xs=sample_x(SMOKE,4,64)
    bpd=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                 ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bpd,"native",SMOKE)
    assert parity(bpd,model,xs[:16])["passed"]
    gen=torch.Generator().manual_seed(8082)
    z=torch.randn((64,PASSES,WIDTH),generator=gen)
    g=torch.randn((64,PASSES,WIDTH),generator=gen)*torch.linspace(3.,.1,WIDTH)
    cov=passwise_source_candidates(z,g)
    with bpd.target_capture():
        model(xd,capture=True)
        acts=torch.stack(model.acts,1).detach()
    chosen,cards,budgets=source_selector(bpd,model,xs,z,acts,cov)
    assert set(cards)==set(BUDGETS)
    assert chosen==next((b for b in BUDGETS if cards[b]["eligible"]),None)
    for b,item in cards.items():
        if item["candidate_exists"]:
            assert sum(item["ranks"])==b and "source_proxy_loss" in item
            assert item["mean_final_kl_nats"]>=0


def test_variable_rank_dose_controls_and_original_excluded_smoke():
    gen=torch.Generator().manual_seed(8083)
    acts=torch.randn((128,PASSES,WIDTH),generator=gen)
    q=[]
    for t,r in enumerate((2,4,6,2)):
        raw=torch.randn((WIDTH,r),generator=gen,dtype=torch.float64)
        v,_=torch.linalg.qr(raw,mode="reduced")
        q.append(v)
    selected,diag=local_dose_matched_shams(
        acts,torch.eye(WIDTH,dtype=torch.float64),q,SMOKE,
        proposals=96,want=2
    )
    assert diag["proposals"]==96
    assert diag["selected"]==len(selected)<=2
    assert diag["qualified"]>=len(selected)
    assert all(s["relative_dose_error"]<=.15 for s in selected)
    assert all(s["p"].shape==(4,20,20) for s in selected)
    root=evaluate_root(
        SMOKE,steps=8,derivative_n=64,
        selection_n=64,heldout_n=64,proposals=96,shams=2
    )
    assert root["phase"]=="DEVELOPMENT" and root["root"]==799
    assert root["clean_closure"]["passed"]
    assert root["splits"]=={"train":2048,"discovery":64,"selection":64,"heldout":64}
    assert set(root["candidates_by_budget"])==set(map(str,BUDGETS))
    assert root["selected_total_rank"] is None or root["selected_total_rank"] in BUDGETS
    assert root["protocol_sha256"] and root["source_sha256"]
