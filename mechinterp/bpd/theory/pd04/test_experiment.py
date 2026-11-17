"""PD-04 engineering-only tests on excluded seed 1009.

No registered seed is trained here. Run before EXECUTE_LOCK.json is committed.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mechinterp.bpd.theory.pd04.experiment import (
    ARMS, CONFIG, FAMILIES, SEEDS, AdaptiveTopKSTE,
    build, draw_family, draw_train_batch, gate_mask, masked_forward,
    evaluate_family, read_assignment, objective, run_arm, summarize,
)

SMOKE=CONFIG["execution"]["smoke_seed"]


def test_registration_disjoint_seed_cohort_and_splits():
    assert len(SEEDS)==len(set(SEEDS))==24
    assert tuple(sorted(SEEDS))==SEEDS
    assert SMOKE not in SEEDS
    assert SEEDS[0]>409
    assert len(ARMS)==7
    assert FAMILIES==("disjoint","overlap","nested")
    assert CONFIG["optimization"]["steps"]==240
    assert CONFIG["success"]["primary_adaptive_success_count_min"]==18
    src=CONFIG["source"]
    for seed in SEEDS:
        assert len({src["train_seed_offset"]+seed,
                    src["calibration_seed_offset"]+seed,
                    src["heldout_seed_offset"]+seed})==3
    assert CONFIG["predecessor"]["pd03_primary_p"]==0.0625


def test_stratified_generator_supports_and_balanced_batch():
    src=CONFIG["source"]
    for fam in FAMILIES:
        x=draw_family(torch.Generator().manual_seed(200000+SMOKE),32,fam)
        assert x.shape==(32,4,1,2)
        active=src[f"{fam}_supports"]
        for t in range(4):
            for j in range(2):
                if j in active[t]:
                    assert torch.all((x[:,t,:,j].abs()>=.75)&(x[:,t,:,j].abs()<=1.25))
                else:
                    assert torch.all(x[:,t,:,j]==0)
    a=draw_train_batch(torch.Generator().manual_seed(100000+SMOKE),8)
    b=draw_train_batch(torch.Generator().manual_seed(100000+SMOKE),8)
    assert torch.equal(a,b)
    assert a.shape==(24,4,1,2)
    counts=[int(((a!=0).sum(dim=-1)==2).all(dim=(1,2)).sum()),]
    assert counts[0]==8


def test_adaptive_topk_forward_is_exact_and_backward_reaches_count():
    torch.manual_seed(SMOKE)
    logits=torch.tensor([[[[2.0,-1.0]], [[-1.0,2.0]]]],requires_grad=True)
    count=torch.tensor([[[[-2.0]], [[2.0]]]],requires_grad=True)
    mask=AdaptiveTopKSTE.apply(logits,count)
    assert torch.equal(mask.detach(),torch.tensor([[[[1.,0.]],[[1.,1.]]]]))
    (mask[...,0]*2.+mask[...,1]*3.).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum()>0
    assert count.grad is not None and count.grad.abs().sum()>0
    assert bool(((mask==0)|(mask==1)).all())


def test_actual_shared_component_wrapper_and_count_gradient():
    bpd,gate,_=build(SMOKE,frozen=False)
    x=draw_train_batch(torch.Generator().manual_seed(100000+SMOKE),8)
    y,mask,logits,count=masked_forward(bpd,gate,x,"adaptive")
    assert y.shape==x.shape
    assert mask.shape==logits.shape==x.shape
    assert count.shape==x[...,0:1].shape
    assert bpd.wrappers["core"].mode=="target"
    assert bpd.wrappers["core"].calls==4
    assert bool(((mask==0)|(mask==1)).all())
    value=objective(bpd,gate,x,"adaptive")
    value.backward()
    factors=bpd.component_parameters()
    assert all(p.grad is not None for p in factors)
    assert sum(float(p.grad.square().sum()) for p in factors)>0
    assert sum(float(p.grad.square().sum()) for p in gate.count.parameters())>0
    assert sum(float(p.grad.square().sum()) for p in gate.fc.parameters())>0


def test_cardinality_truth_masks_and_fixed_capacity_controls():
    x=draw_train_batch(torch.Generator().manual_seed(100000+SMOKE),8)
    bpd,gate,_=build(SMOKE,frozen=False)
    a=gate.fc(x.abs()); count=gate.count(x.abs())
    one=gate_mask(a,count,x,"top1")
    two=gate_mask(a,count,x,"all2")
    oracle=gate_mask(a,count,x,"oracle")
    assert bool((one.sum(dim=-1)==1).all())
    assert bool((two.sum(dim=-1)==2).all())
    assert torch.equal(oracle.sum(dim=-1), (x!=0).sum(dim=-1).float())
    assert bool(((oracle==0)|(oracle==1)).all())
    overlaps=((x!=0).sum(dim=-1)==2).sum().item()
    assert overlaps>0
    # Restricting capacity to one always misses an active component on
    # overlapped passes. Always-on capacity creates redundancy on singles.
    assert int(((x!=0).sum(dim=-1)>one.sum(dim=-1)).sum())>0
    assert int(((x!=0).sum(dim=-1)<two.sum(dim=-1)).sum())>0


def test_matched_initialization_and_excluded_full_wrapper_smokes():
    initials=[]
    rows=[]
    for arm in ARMS:
        bpd,gate,h=build(SMOKE,frozen=arm=="adaptive_frozen")
        if arm!="adaptive_frozen":
            initials.append(h)
        row=run_arm(
            SMOKE,arm,steps=4,boards_per_family=12,train_per_family=4
        )
        rows.append(row)
        assert row["heldout"]["axis_match"]>=0
        assert set(row["heldout"]["per_family"])==set(FAMILIES)
        assert all(m["mask_is_binary"] for m in row["heldout"]["per_family"].values())
        if arm=="adaptive_frozen":
            assert row["factor_update_norm"]==0
        else:
            assert row["factor_update_norm"]>0
            assert row["factor_first_gradient_norm"]>0
        if arm=="calibrated_independent":
            assert len(row["calibration_candidates"])==7
            assert row["threshold"] in CONFIG["optimization"]["independent_threshold_candidates"]
            chosen=next(c for c in row["calibration_candidates"] if c["threshold"]==row["threshold"])
            assert chosen["score"]<=min(c["score"] for c in row["calibration_candidates"])+1e-9
        if arm=="adaptive_disjoint_overlap":
            assert row["count_first_gradient_norm"]>0
    assert len(set(initials))==1
    result=summarize(rows,registered=False)
    assert result["seed_count"]==1
    assert result["arm_count"]==7
    assert not result["registered_primary_passed"]
