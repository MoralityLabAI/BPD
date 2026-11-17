"""Excluded-seed engineering tests for BPD-PD-07 (seed 2003 only)."""
from __future__ import annotations

import math
import torch

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.theory.pd06.experiment import sample_batch, rotation_for_seed
from mechinterp.bpd.theory.pd07.experiment import (
    CONFIG, SEEDS, ARMS, FAMILIES, ARM_INFO,
    build, gate_read_coefficients, mask_and_diagnostics,
    masked_forward, objective, run_arm, summarize,
)

SMOKE=CONFIG["runtime"]["smoke_seed"]


def test_locked_registration_and_independent_seeds():
    assert CONFIG["study_id"]=="BPD-PD-07"
    assert len(SEEDS)==24 and len(set(SEEDS))==24
    assert tuple(sorted(SEEDS))==SEEDS
    assert SMOKE not in SEEDS and min(SEEDS)>941
    assert ARMS==(
        "read_coupled","raw_independent","read_detached",
        "raw_count_supervised","read_planted_frozen","oracle_support",
    )
    assert tuple(FAMILIES)==(
        "k1_seen","k2_seen","k2_unseen",
        "k3_seen","k3_unseen","transition_unseen",
    )
    assert CONFIG["train"]["steps"]==240
    assert CONFIG["success"]["primary_min_successes"]==18
    assert CONFIG["success"]["primary_net_seed_wins_min"]==6


def test_exact_read_threshold_usable_on_planted_basis_and_dense_observation():
    bpd,gate,q,_,_=build(SMOKE,"read_planted_frozen")
    assert torch.allclose(q.T@q,torch.eye(4),atol=3e-6)
    assert bpd.wrappers["core"].V.requires_grad is False
    for fam in FAMILIES:
        g=torch.Generator().manual_seed(200000+SMOKE+10000*FAMILIES.index(fam))
        x,support,_=sample_batch(g,24,q,family=fam)
        assert float((x.abs()>1e-6).float().mean()) > .99
        y,mask,diag=masked_forward(bpd,gate,x,support,"read_planted_frozen")
        assert torch.equal(mask.detach(),support)
        assert torch.allclose(y,x,atol=2e-5,rtol=0)
        assert bool(((mask==0)|(mask==1)).all())
        assert abs(float(diag["threshold"].mean())-.5)<1e-7


def test_raw_and_read_gates_are_label_invariant():
    for arm in ("read_coupled","read_detached","raw_independent"):
        bpd,gate,q,_,_=build(SMOKE,arm)
        x,support,_=sample_batch(torch.Generator().manual_seed(102003),12,q,family="train")
        bogus=torch.zeros_like(support)
        bogus[...,0]=1
        y1,mask1,_=masked_forward(bpd,gate,x,support,arm)
        y2,mask2,_=masked_forward(bpd,gate,x,bogus,arm)
        assert torch.equal(mask1.detach(),mask2.detach())
        assert torch.equal(y1.detach(),y2.detach())
        d1=objective(bpd,gate,x,support,arm)
        d2=objective(bpd,gate,x,bogus,arm)
        assert torch.equal(d1.detach(),d2.detach())
    bpd,gate,q,_,_=build(SMOKE,"raw_count_supervised")
    x,support,_=sample_batch(torch.Generator().manual_seed(102003),12,q,family="train")
    bogus=torch.zeros_like(support)
    bogus[...,0]=1
    assert not torch.equal(
        objective(bpd,gate,x,support,"raw_count_supervised").detach(),
        objective(bpd,gate,x,bogus,"raw_count_supervised").detach()
    )


def test_real_bpd_factor_update_and_separate_gate_feature_gradient():
    bpd,gate,q,_,_=build(SMOKE,"read_coupled")
    assert isinstance(bpd,BPDModel)
    x,support,_=sample_batch(torch.Generator().manual_seed(102003),12,q,family="train")
    loss=objective(bpd,gate,x,support,"read_coupled")
    loss.backward()
    params=bpd.component_parameters()
    assert all(p.grad is not None for p in params)
    assert sum(float(p.grad.square().sum()) for p in params)>0
    assert gate.threshold_raw.grad is not None
    assert float(gate.threshold_raw.grad.abs().sum())>0
    assert bpd.wrappers["core"].mode=="target"
    assert bpd.wrappers["core"].calls==4

    bc,gc,qc,_,_=build(SMOKE,"read_coupled")
    bd,gd,qd,_,_=build(SMOKE,"read_detached")
    assert torch.equal(bc.wrappers["core"].V,bd.wrappers["core"].V)
    a=torch.rand(8,4,1,4)
    mc,_=mask_and_diagnostics(bc,gc,a,torch.zeros_like(a),"read_coupled")
    md,_=mask_and_diagnostics(bd,gd,a,torch.zeros_like(a),"read_detached")
    assert torch.equal(mc.detach(),md.detach())
    mc.sum().backward()
    md.sum().backward()
    assert bc.wrappers["core"].V.grad is not None
    assert float(bc.wrappers["core"].V.grad.abs().sum())>0
    assert bd.wrappers["core"].V.grad is None
    # Physical writes still train both factors in detached-feature arm.
    bd.zero_grad(set_to_none=True)
    gd.zero_grad(set_to_none=True)
    x2,support2,_=sample_batch(torch.Generator().manual_seed(102003),8,qd,family="train")
    objective(bd,gd,x2,support2,"read_detached").backward()
    assert bd.wrappers["core"].V.grad is not None
    assert float(bd.wrappers["core"].V.grad.abs().sum())>0


def test_source_support_holdouts_and_unchanged_rotation_rule():
    q=rotation_for_seed(SMOKE)
    _,train_mask,_=sample_batch(torch.Generator().manual_seed(102003),128,q,family="train")
    seen=set(tuple(torch.nonzero(train_mask[b,t,0]).flatten().tolist())
             for b in range(128) for t in range(4))
    unseen={(0,2),(1,3),(0,2,3)}
    assert seen.isdisjoint(unseen)
    for fam in ("k2_unseen","k3_unseen"):
        _,mask,_=sample_batch(torch.Generator().manual_seed(202003),24,q,family=fam)
        actual=set(tuple(torch.nonzero(mask[b,t,0]).flatten().tolist())
                   for b in range(24) for t in range(4))
        assert actual <= unseen


def test_excluded_model_six_arm_smoke_with_same_initialization():
    rows=[]
    for arm in ARMS:
        r=run_arm(SMOKE,arm,steps=5,batch_size=8,heldout_boards=16)
        assert r["seed"]==SMOKE
        assert len(r["heldout"]["per_family"])==6
        assert all(x["mask_binary"] for x in r["heldout"]["per_family"].values())
        assert r["rotation_sha256"]==rows[0]["rotation_sha256"] if rows else True
        if arm=="read_planted_frozen":
            assert r["factor_update_norm"]==0
            assert r["heldout"]["alignment"]["mean_cosine"]>.999999
            assert all(x["exact_fourpass_support"]==1
                       for x in r["heldout"]["per_family"].values())
        else:
            assert r["factor_update_norm"]>0
            assert r["factor_first_gradient_norm"]>0
        if arm in ("read_coupled","read_detached"):
            assert r["threshold_first_gradient_norm"]>0
        rows.append(r)
    assert len({r["initialization_sha256"] for r in rows
                if not r["frozen_planted"]})==1
    report=summarize(rows,registered=False)
    assert report["seed_count"]==1
    assert report["arm_count"]==6
    assert not report["preregistered_primary_passed"]
