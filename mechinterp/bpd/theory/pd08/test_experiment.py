"""PD08 engineering tests on excluded seed 2017, before source-locked science."""
from __future__ import annotations

import torch

from mechinterp.bpd.theory.pd06.experiment import sample_batch, rotation_for_seed
from mechinterp.bpd.theory.pd08.experiment import (
    ARMS, CONFIG, SEEDS, FAMILIES, SecondStageGate,
    discover_dictionary, instantiate_stage2, component_mask,
    masked_forward, factors_digest, run_stage2_arm, summarize,
    gate_init_digest, stage2_loss
)
SMOKE = CONFIG["smoke_seed"]


def test_original_source_seeds_and_prospective_gates():
    assert CONFIG["study_id"]=="BPD-PD-08"
    assert len(SEEDS)==24==len(set(SEEDS))
    assert tuple(sorted(SEEDS))==SEEDS
    assert SMOKE not in SEEDS and min(SEEDS)>1103
    assert ARMS==(
      "read_trained","raw_independent","read_fixed_threshold",
      "oracle_support_learned_dictionary","read_planted_frozen",
    )
    assert FAMILIES==(
      "k1_seen","k2_seen","k2_unseen","k3_seen","k3_unseen","transition_unseen"
    )
    assert CONFIG["training"]["stage1_updates"]==360
    assert CONFIG["training"]["stage2_updates"]==240
    assert CONFIG["scoring"]["primary_read_successes_min"]==18


def test_q_rotation_and_forbidden_support_sets_do_not_enter_train():
    q=rotation_for_seed(SMOKE)
    assert torch.allclose(q.T@q,torch.eye(4),atol=2e-6)
    rng=torch.Generator().manual_seed(100000+SMOKE)
    x,support,_=sample_batch(rng,256,q,family="train")
    assert float((x.abs()>1e-6).float().mean())>0.999
    patterns={tuple(torch.nonzero(support[b,t,0]).flatten().tolist())
        for b in range(support.shape[0]) for t in range(4)}
    assert patterns.isdisjoint({(0,2),(1,3),(0,2,3)})
    heldout_rng=torch.Generator().manual_seed(200000+SMOKE+10000*2)
    _,held_support,_=sample_batch(heldout_rng,24,q,family="k2_unseen")
    held_patterns={tuple(torch.nonzero(held_support[b,t,0]).flatten().tolist())
                   for b in range(24) for t in range(4)}
    assert held_patterns <= {(0,2),(1,3)}


def test_source_stage1_updates_real_factors_and_stage2_keeps_exact_snapshot():
    torch.set_num_threads(1)
    learned=discover_dictionary(SMOKE,steps=5,batch_size=8)
    assert learned["factor_update_norm"]>0
    assert learned["first_factor_grad_norm"]>0
    assert len(learned["read_alignment"]["per_component_cosines"])==4
    hashes=[]
    gate_hashes=[]
    for arm in ARMS:
        bpd,gate,q,q_sha=instantiate_stage2(SMOKE,learned,planted=arm=="read_planted_frozen")
        if arm!="read_planted_frozen":
            assert factors_digest(bpd)==learned["factor_sha256"]
            assert torch.equal(bpd.wrappers["core"].V,learned["V"])
            assert torch.equal(bpd.wrappers["core"].U,learned["U"])
            hashes.append(factors_digest(bpd))
        gate_hashes.append(gate_init_digest(gate))
        assert all(not p.requires_grad for p in bpd.component_parameters())
    assert len(set(hashes))==1
    assert len(set(gate_hashes))==1


def test_unsupervised_stage2_mask_and_objective_independent_of_labels():
    torch.set_num_threads(1)
    learned=discover_dictionary(SMOKE,steps=3,batch_size=8)
    for arm in ("read_trained","raw_independent","read_fixed_threshold"):
        bpd,gate,q,_=instantiate_stage2(SMOKE,learned,planted=False)
        x,support,_=sample_batch(torch.Generator().manual_seed(500000+SMOKE),8,q,family="train")
        poisoned=torch.zeros_like(support)
        poisoned[...,0]=1
        a,m1=masked_forward(bpd,gate,x,arm,support=support)
        b,m2=masked_forward(bpd,gate,x,arm,support=poisoned)
        assert torch.equal(a.detach(),b.detach())
        assert torch.equal(m1.detach(),m2.detach())
        assert bool(((m1.detach()==0)|(m1.detach()==1)).all())
        if arm!="read_fixed_threshold":
            loss=stage2_loss(bpd,gate,x,arm)
            loss.backward()
            if arm=="read_trained":
                assert gate.threshold_raw.grad is not None
                assert float(gate.threshold_raw.grad.abs().sum())>0
            else:
                assert gate.raw.assignment.weight.grad is not None
                assert float(gate.raw.assignment.weight.grad.abs().sum())>0
            assert all(p.grad is None for p in bpd.component_parameters())


def test_read_threshold_planted_positive_and_component_gauge():
    torch.set_num_threads(1)
    learned=discover_dictionary(SMOKE,steps=2,batch_size=8)
    bpd,gate,q,_=instantiate_stage2(SMOKE,learned,planted=True)
    assert torch.allclose(bpd.wrappers["core"].V,q,atol=3e-6)
    for family in FAMILIES:
        rng=torch.Generator().manual_seed(200000+SMOKE+10000*FAMILIES.index(family))
        x,support,_=sample_batch(rng,16,q,family=family)
        y,mask=masked_forward(bpd,gate,x,"read_planted_frozen")
        assert torch.equal(mask,support)
        assert torch.allclose(y,x,rtol=0,atol=2e-5)
    assert all(abs(t-.5)<1e-8 for t in gate.thresholds().tolist())


def test_all_five_arms_excluded_seed_and_scientific_sample_guard():
    torch.set_num_threads(1)
    learned=discover_dictionary(SMOKE,steps=4,batch_size=8)
    rows=[]
    for arm in ARMS:
        r=run_stage2_arm(
            SMOKE,arm,learned,updates=5,batch_size=8,
            heldout_boards=16
        )
        rows.append(r)
        assert r["factors_bitwise_frozen"]
        assert set(r["heldout"]["per_family"])==set(FAMILIES)
        assert all(m["exact_binary"] for m in r["heldout"]["per_family"].values())
        if arm in ("read_trained","raw_independent"):
            assert r["stage2_updates"]==5
        else:
            assert r["stage2_updates"]==0
        if arm=="read_planted_frozen":
            assert r["heldout"]["alignment"]["mean_cosine"]>.999999
            assert all(x["support_fraction"]==1
                for x in r["heldout"]["per_family"].values())
        if arm=="oracle_support_learned_dictionary":
            assert all(x["support_fraction"]==1
                for x in r["heldout"]["per_family"].values())
    assert len({r["actual_factor_sha256"] for r in rows
                if r["arm"]!="read_planted_frozen"})==1
    assert len({r["gate_initial_sha256"] for r in rows})==1
    s=summarize(rows,registered=False)
    assert s["seeds"]==1 and s["arms"]==5
    assert not s["primary_passed"]
