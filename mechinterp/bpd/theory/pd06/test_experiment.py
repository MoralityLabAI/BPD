"""Engineering-only PD06 source/mask/rotation tests with excluded seed 1009."""
from __future__ import annotations

import itertools
import torch

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.theory.pd06.experiment import (
    CONFIG, SEEDS, ARMS, FAMILIES,
    rotation_for_seed, build, sample_batch,
    ste_mask, masked_rollout, objective, rank4_alignment,
    score_family, run_arm, cohort_summary, exact_topk_mask,
)

SMOKE=CONFIG["execution"]["smoke_seed"]

def test_source_registration_and_seed_independence():
    assert CONFIG["study_id"]=="BPD-PD-06"
    assert len(SEEDS)==len(set(SEEDS))==32
    assert SEEDS==tuple(sorted(SEEDS))
    assert SMOKE not in SEEDS and SEEDS[0]>719
    assert len(ARMS)==7
    assert set(FAMILIES)=={
        "k1_seen","k2_seen","k2_unseen",
        "k3_seen","k3_unseen","transition_unseen"
    }
    src=CONFIG["source"]
    assert len({src["training_generator_seed_offset"]+s for s in SEEDS})==32
    for seed in SEEDS:
        assert len({src["training_generator_seed_offset"]+seed,
                    src["calibration_generator_seed_offset"]+seed,
                    src["heldout_generator_seed_offset"]+seed})==3
    assert CONFIG["scoring"]["primary_min_successes"]==24

def test_deterministic_rotation_and_dense_observed_coordinates():
    q=rotation_for_seed(SMOKE)
    q2=rotation_for_seed(SMOKE)
    assert torch.equal(q,q2)
    assert torch.allclose(q.T@q,torch.eye(4),atol=2e-6,rtol=0)
    assert not torch.equal(q,torch.eye(4))
    rng=torch.Generator().manual_seed(101009)
    x,support,latent=sample_batch(rng,40,q,family="train")
    assert x.shape==support.shape==latent.shape==(40,4,1,4)
    assert torch.allclose(x.squeeze(2)@q,latent.squeeze(2),atol=2e-6,rtol=0)
    assert bool(((latent==0)==(support==0)).all())
    assert float((x.abs()>1e-6).float().mean()) > .999

def test_train_supports_never_include_reserved_combinations():
    q=rotation_for_seed(SMOKE)
    s=CONFIG["source"]
    train={tuple(o) for k in ("train_support_k1","train_support_k2","train_support_k3")
           for o in s[k]}
    unseen={tuple(o) for k in ("holdout_support_k2","holdout_support_k3")
           for o in s[k]}
    assert train.isdisjoint(unseen)
    _,mask,_=sample_batch(torch.Generator().manual_seed(101009),256,q,family="train")
    patterns={tuple(int(i) for i in torch.nonzero(mask[b,t,0],as_tuple=False).flatten().tolist())
              for b in range(mask.shape[0]) for t in range(4)}
    assert patterns <= train
    assert all(any(len(p)==k for p in patterns) for k in (1,2,3))
    for family in ("k2_unseen","k3_unseen","transition_unseen"):
        _,heldout,_=sample_batch(torch.Generator().manual_seed(201009),24,q,family=family)
        supportsets=[tuple(int(i) for i in torch.nonzero(heldout[b,t,0],as_tuple=False).flatten().tolist())
                     for b in range(heldout.shape[0]) for t in range(4)]
        if family=="k2_unseen" or family=="k3_unseen":
            assert set(supportsets) <= unseen
        else:
            assert all(supportsets[t] in unseen for t in range(len(supportsets))
                       if t%4 in (1,2,3))

def test_exact_binary_topk_and_backprop_through_count_and_assignment():
    assignment=torch.tensor([[[[1.0,.9,.2,-1.0]]]],requires_grad=True)
    count=torch.tensor([[[[-.5,.1,.7]]]],requires_grad=True)
    truth=torch.tensor([[[[1.,1.,1.,0.]]]])
    mask=ste_mask(assignment,count,truth,"unsupervised_topk")
    assert torch.equal(mask.detach(),torch.tensor([[[[1.,1.,1.,0.]]]]))
    assert bool(((mask==0)|(mask==1)).all())
    (mask*torch.tensor([[[[1.,2.,3.,4.]]]])).sum().backward()
    assert count.grad is not None and float(count.grad.abs().sum())>0
    assert assignment.grad is not None and float(assignment.grad.abs().sum())>0
    # Hard cardinality depends on count logits; oracle count uses support size.
    truth1=torch.tensor([[[[1.,0.,0.,0.]]]])
    oracle=ste_mask(assignment.detach(),count.detach(),truth1,"oracle_count")
    assert float(oracle.sum())==1
    assert torch.equal(ste_mask(assignment.detach(),count.detach(),truth1,"oracle_support"),truth1)
    ranks=exact_topk_mask(assignment.detach(),torch.tensor([[[2]]]))
    assert int(ranks.sum())==2

def test_unlabelled_gate_is_invariant_to_corrupted_label_argument():
    bpd,gate,q,_,_=build(SMOKE,frozen_planted=False)
    x,mask,_=sample_batch(torch.Generator().manual_seed(101009),10,q,family="train")
    poisoned=torch.zeros_like(mask)
    # Source labels are passed for optional supervision, but the unsupervised
    # execution path must be fully insensitive to their values.
    y1, m1, _,_=masked_rollout(bpd,gate,x,mask,"unsupervised_topk")
    y2, m2, _,_=masked_rollout(bpd,gate,x,poisoned,"unsupervised_topk")
    assert torch.equal(y1.detach(),y2.detach())
    assert torch.equal(m1.detach(),m2.detach())
    d1,_=objective(bpd,gate,x,mask,"unsupervised_topk")
    d2,_=objective(bpd,gate,x,poisoned,"unsupervised_topk")
    assert torch.equal(d1.detach(),d2.detach())
    sp1,_=objective(bpd,gate,x,mask,"support_supervised")
    sp2,_=objective(bpd,gate,x,poisoned,"support_supervised")
    assert not torch.equal(sp1.detach(),sp2.detach())

def test_matching_gauge_and_real_component_factor_gradients():
    bpd,gate,q,_,_=build(SMOKE,frozen_planted=False)
    assert isinstance(bpd,BPDModel)
    with torch.no_grad():
        bpd.wrappers["core"].V.copy_(q[:,[1,0,3,2]])
    match=rank4_alignment(bpd,q)
    assert match["mean_cosine"] > .999999
    assert match["component_to_latent_permutation"]==[1,0,3,2]
    x,support,_=sample_batch(torch.Generator().manual_seed(101009),10,q,family="train")
    y,gates,_,_=masked_rollout(bpd,gate,x,support,"count_supervised")
    assert y.shape==x.shape
    assert bool(((gates==0)|(gates==1)).all())
    assert bpd.wrappers["core"].mode=="target" and bpd.wrappers["core"].calls==4
    loss,labels=objective(bpd,gate,x,support,"count_supervised")
    assert "count_CE" in labels and "support_BCE" not in labels
    loss.backward()
    assert all(p.grad is not None for p in bpd.component_parameters())
    assert sum(float(p.grad.square().sum()) for p in bpd.component_parameters())>0
    assert sum(float(p.grad.square().sum()) for p in gate.count.parameters())>0
    assert sum(float(p.grad.square().sum()) for p in gate.assignment.parameters())>0

def test_excluded_seed_model_smokes_and_matched_initializations():
    rows=[]
    for name in ARMS:
        row=run_arm(SMOKE,name,steps=3,batch_size=8,heldout_boards=12)
        rows.append(row)
        assert len(row["heldout"]["per_family"])==6
        assert all(v["execution_binary"] for v in row["heldout"]["per_family"].values())
        assert row["rotation_sha256"]==rows[0]["rotation_sha256"]
        if name=="count_supervised_planted_frozen":
            assert row["factor_update_norm"]==0
            assert row["heldout"]["alignment"]["mean_cosine"]>.999999
        else:
            assert row["factor_update_norm"]>0
            assert row["factor_first_gradient_norm"]>0
        if name=="dense_all":
            assert all(abs(v["observed_gate_density"]-1.0)<1e-8
                       for v in row["heldout"]["per_family"].values())
    assert len({r["initial_parameters_sha256"] for r in rows
                if not r["frozen_planted"]})==1
    result=cohort_summary(rows,registered=False)
    assert result["seeds"]==1 and result["arms"]==7
    assert not result["primary_success"]
