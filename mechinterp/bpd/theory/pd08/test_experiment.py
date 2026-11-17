"""PD08 source-only excluded-seed 3001 tests, before any cohort training."""
from __future__ import annotations

import torch

from mechinterp.bpd.theory.pd06.experiment import (
    build as build_pd06, sample_batch,
)
from mechinterp.bpd.theory.pd08.experiment import (
    CONFIG, SEEDS, ARMS, FAMILIES, build_and_pretrain,
    init_stage2_gate, factor_snapshot, stage2_mask,
    masked_forward, stage2_objective, run_seed, summary,
)

SMOKE=CONFIG["runtime"]["excluded_smoke_seed"]


def test_preregistered_seeds_are_new_and_sources_separate():
    assert CONFIG["study_id"]=="BPD-PD-08"
    assert len(SEEDS)==len(set(SEEDS))==24
    assert tuple(sorted(SEEDS))==SEEDS
    assert min(SEEDS)>1103 and SMOKE not in SEEDS
    assert ARMS==(
        "read_refit","raw_refit","read_nofit",
        "read_shuffled_refit","stage1_count_topk","oracle_support",
    )
    assert CONFIG["optimization"]["stage1_steps"]==360
    assert CONFIG["optimization"]["stage2_steps"]==180
    assert CONFIG["success"]["primary_min_successes"]==18
    assert FAMILIES==(
        "k1_seen","k2_seen","k2_unseen",
        "k3_seen","k3_unseen","transition_unseen",
    )
    for seed in SEEDS:
        assert len({
            CONFIG["source"]["stage1_generator_offset"]+seed,
            CONFIG["source"]["stage2_generator_offset"]+seed,
            CONFIG["source"]["heldout_generator_offset"]+seed,
        })==3


def test_planted_read_mapping_and_shuffled_read_negative_control():
    bpd,count_gate,q,_,_=build_pd06(SMOKE,frozen_planted=True)
    gate,_=init_stage2_gate(SMOKE)
    gen=torch.Generator().manual_seed(400000+SMOKE)
    x,support,_=sample_batch(gen,24,q,family="train")
    correct=stage2_mask(bpd,gate,x,support,"read_refit")
    shuffled=stage2_mask(bpd,gate,x,support,"read_shuffled_refit")
    assert torch.equal(correct.detach(),support)
    assert not torch.equal(shuffled.detach(),support)
    assert bool(((correct==0)|(correct==1)).all())
    assert bool(((shuffled==0)|(shuffled==1)).all())


def test_label_poison_does_not_affect_gate_only_refit():
    bpd,count_gate,q,_,_=build_pd06(SMOKE,frozen_planted=False)
    gate,_=init_stage2_gate(SMOKE)
    x,support,_=sample_batch(torch.Generator().manual_seed(400000+SMOKE),16,q,family="train")
    poison=torch.zeros_like(support)
    poison[...,0]=1.0
    for arm in ("read_refit","raw_refit","read_nofit","read_shuffled_refit"):
        a=stage2_mask(bpd,gate,x,support,arm)
        b=stage2_mask(bpd,gate,x,poison,arm)
        assert torch.equal(a.detach(),b.detach())
    for arm in ("read_refit","raw_refit","read_shuffled_refit"):
        l1=stage2_objective(bpd,gate,x,support,arm)
        l2=stage2_objective(bpd,gate,x,poison,arm)
        assert torch.equal(l1.detach(),l2.detach())
    assert not torch.equal(
        stage2_mask(bpd,gate,x,support,"oracle_support",mapping=[0,1,2,3]),
        stage2_mask(bpd,gate,x,poison,"oracle_support",mapping=[0,1,2,3]),
    )


def test_stage1_factors_train_and_are_then_bitwise_frozen():
    trained=build_and_pretrain(SMOKE,steps=8,batch_size=8)
    bpd=trained["bpd"]
    assert trained["initial_factor_grad"]>0
    assert trained["factor_update"]>0
    assert all(not p.requires_grad for p in bpd.component_parameters())
    before,sha=factor_snapshot(bpd)
    gate,_=init_stage2_gate(SMOKE)
    rng=torch.Generator().manual_seed(400000+SMOKE)
    x,support,_=sample_batch(rng,8,trained["q"],family="train")
    loss=stage2_objective(bpd,gate,x,support,"read_refit")
    loss.backward()
    assert gate.threshold_raw.grad is not None
    assert float(gate.threshold_raw.grad.abs().sum())>0
    assert all(p.grad is None for p in bpd.component_parameters())
    after,post_sha=factor_snapshot(bpd)
    assert sha==post_sha and all(torch.equal(a,b) for a,b in zip(before,after))


def test_one_stage1_per_seed_six_matched_arm_smoke():
    rows=run_seed(
        SMOKE,stage1_steps=6,stage2_steps=4,
        batch_size=8,heldout_boards=12,
    )
    assert len(rows)==6
    assert [r["arm"] for r in rows]==list(ARMS)
    assert len({r["frozen_factor_sha256"] for r in rows})==1
    assert len({r["stage2_gate_initialization_sha256"] for r in rows})==1
    assert len({r["rotation_sha256"] for r in rows})==1
    assert all(r["factor_bitwise_frozen"] for r in rows)
    assert all(r["stage1_first_factor_gradient"]>0 for r in rows)
    assert all(r["stage1_factor_update"]>0 for r in rows)
    assert all(all(v["mask_binary"] for v in r["heldout"]["per_family"].values())
               for r in rows)
    assert all(r["stage2_steps"]==4 for r in rows
               if r["arm"] in ("read_refit","raw_refit","read_shuffled_refit"))
    assert all(r["stage2_steps"]==0 for r in rows
               if r["arm"] not in ("read_refit","raw_refit","read_shuffled_refit"))
    result=summary(rows,registered=False)
    assert result["seed_count"]==1 and result["arm_count"]==6
    assert not result["preregistered_primary_passed"]
