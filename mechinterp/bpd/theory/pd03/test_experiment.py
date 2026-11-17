"""BPD-PD-03 engineering audit using excluded seed 1009 only."""
from __future__ import annotations

import math

import torch

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.theory.phase_disjoint_control import draw_batch
from mechinterp.bpd.theory.pd03.experiment import (
    CONFIG, SEEDS, ARMS, gate_mask, masked_forward, loss_parts,
    make_model, run_arm, exact_mcnemar_p, choose_threshold,
    cohort_summary, read_assignment,
)

SMOKE=CONFIG["smoke_seed"]


def test_registration_and_split_seeds():
    assert len(SEEDS)==24
    assert tuple(sorted(set(SEEDS)))==SEEDS
    assert SMOKE not in SEEDS and 71 not in SEEDS and 89 not in SEEDS
    assert len(ARMS)==7
    assert CONFIG["optimization"]["stage1_steps"]==200
    assert CONFIG["success"]["primary_top1_min_successes"]==18
    assert CONFIG["predecessor"]["hard_false_offs"]==498
    for seed in SEEDS:
        offsets = CONFIG["generation"]
        assert len({offsets[p+"_seed_offset"]+seed for p in
                    ("training","calibration","heldout")})==3


def test_straight_through_top1_and_threshold_mask():
    logits=torch.tensor([[[[.2,.4],[-.3,.7]]]],requires_grad=True)
    top1=gate_mask(logits,"top1")
    assert top1.shape==logits.shape
    assert torch.equal(top1.detach(),
                       torch.tensor([[[[0.,1.],[0.,1.]]]]))
    assert bool((top1.sum(dim=-1)==1).all())
    top1[...,0].sum().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum()>0
    logits2=torch.tensor([-.9,0.,.9],requires_grad=True)
    margin=gate_mask(logits2,"margin",threshold=.5)
    assert torch.equal(margin.detach(),torch.tensor([0.,1.,1.]))
    margin.sum().backward()
    assert logits2.grad is not None and logits2.grad.abs().sum()>0


def test_real_component_wrapper_routes_exactly_one_and_receives_grads():
    bpd,gate,_=make_model(SMOKE,frozen=False)
    assert isinstance(bpd,BPDModel)
    x=draw_batch(torch.Generator().manual_seed(101009),8,mixed=False)
    for mode in ("top1","margin","hard"):
        y,mask,probs=masked_forward(bpd,gate,x,mode)
        assert y.shape==x.shape
        assert mask.shape==probs.shape==x.shape
        assert bpd.wrappers["core"].mode=="target"
        assert bpd.wrappers["core"].calls==4
        assert bool(((mask.detach()==0)|(mask.detach()==1)).all())
        if mode=="top1":
            assert bool((mask.detach().sum(dim=-1)==1).all())
    d=loss_parts(bpd,gate,x,"top1")
    d["total"].backward()
    assert all(p.grad is not None for p in bpd.component_parameters())
    assert sum(float(p.grad.square().sum()) for p in bpd.component_parameters())>0
    assert sum(float(p.grad.square().sum()) for p in gate.parameters())>0


def test_same_initialization_and_source_batches():
    initial=[]
    for arm in ARMS:
        if arm.endswith("_frozen"):
            continue
        bpd,gate,h=make_model(SMOKE,frozen=False)
        initial.append(h)
    assert len(set(initial))==1
    a=draw_batch(torch.Generator().manual_seed(101009),16,mixed=False)
    b=draw_batch(torch.Generator().manual_seed(101009),16,mixed=False)
    assert torch.equal(a,b)
    assert bool((a[:,:2,: ,1]==0).all())
    assert bool((a[:,2:,:,0]==0).all())
    mixed=draw_batch(torch.Generator().manual_seed(101009),16,mixed=True)
    assert bool((mixed.abs()>=.75).all())


def test_all_arms_excluded_seed_and_calibration_constraints():
    rows=[
        run_arm(SMOKE,arm,steps=6,batch_size=8,
                calibration_size=24,heldout_size=24)
        for arm in ARMS
    ]
    assert len(rows)==7
    assert all(r["seed"]==SMOKE for r in rows)
    assert len({r["initial_parameter_sha256"] for r in rows if not r["frozen"]})==1
    for r in rows:
        assert r["heldout"]["execution_exact_binary"]
        assert r["heldout"]["required_gate_count"]==96
        assert 0<=r["heldout"]["false_off_count"]<=96
        assert 0<=r["heldout"]["false_on_count"]<=96
        assert r["calibration_threshold"] in CONFIG["optimization"]["calibration_thresholds"] or r["calibration_threshold"]==.5
        if r["mode"]=="top1":
            assert r["heldout"]["gate_fraction"]==.5
            assert r["heldout"]["false_on_count"]==r["heldout"]["false_off_count"]
        if r["mode"]=="margin":
            assert len(r["calibration_candidates"])==6
            scores=[c["calibration_score"] for c in r["calibration_candidates"]]
            assert min(scores)<=max(scores)
            chosen=next(c for c in r["calibration_candidates"] if c["threshold"]==r["calibration_threshold"])
            assert chosen["calibration_score"]<=min(scores)+1e-10
        if r["frozen"]:
            assert r["factor_update_norm"]==0
            assert r["factor_first_gradient_norm"]==0
        else:
            assert r["factor_update_norm"]>0
            assert r["factor_first_gradient_norm"]>0
            assert r["gate_first_gradient_norm"]>0
    result=cohort_summary(rows,registered=False)
    assert not result["primary_passed"]
    assert not result["registered"]
    assert result["seeds"]==1


def test_exact_seed_level_statistics():
    assert exact_mcnemar_p(0,0)==1
    assert exact_mcnemar_p(6,0)==.03125
    assert exact_mcnemar_p(3,7)==.34375
