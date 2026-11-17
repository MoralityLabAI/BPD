"""PD10 excluded-seed engineering tests, seed 2011 outside registered cohort."""
from __future__ import annotations

import math

import torch

from mechinterp.bpd.theory.pd06.experiment import (
    rotation_for_seed, sample_batch,
)
from mechinterp.bpd.theory.pd09.experiment import qr_orthogonal, smoothed_l0
from mechinterp.bpd.theory.pd10.experiment import (
    CONFIG, SEEDS, ARMS, FAMILIES, sigma_for_step,
    initial_orthogonal_matrix, train_arm, run_arm, summarize,
)
from mechinterp.bpd.theory.pd06.experiment import build as build_pd06

SMOKE=CONFIG["excluded_engineering_seed"]


def test_source_registration_and_disjoint_new_seeds():
    assert CONFIG["study_id"]=="BPD-PD-10"
    assert len(SEEDS)==len(set(SEEDS))==24
    assert tuple(sorted(SEEDS))==SEEDS
    assert SMOKE not in SEEDS and min(SEEDS)>1471
    assert len(ARMS)==6
    assert ARMS==(
        "cold015","cold030","warm030_to015",
        "anneal060_030_015","privileged_pairmix015",
        "count_supervised_positive",
    )
    assert len(FAMILIES)==6
    assert CONFIG["scoring"]["primary_arm"]=="cold015"
    assert CONFIG["scoring"]["primary_control"]=="anneal060_030_015"
    assert CONFIG["training"]["steps"]==360


def test_locked_sigma_schedule_including_pre_crossover_starts():
    expected={
        "cold015":(.15,.15,.15),
        "cold030":(.3,.3,.3),
        "warm030_to015":(.3,.15,.15),
        "anneal060_030_015":(.6,.3,.15),
        "privileged_pairmix015":(.15,.15,.15),
    }
    for arm,values in expected.items():
        assert tuple(sigma_for_step(arm,n) for n in (0,120,240))==values
        assert sigma_for_step(arm,119)==values[0]
        assert sigma_for_step(arm,239)==values[1]
        assert sigma_for_step(arm,359)==values[2]


def test_qr_autograd_and_complete_physical_reconstruction():
    torch.manual_seed(SMOKE)
    A=torch.nn.Parameter(torch.randn(4,4))
    q=qr_orthogonal(A)
    assert torch.allclose(q.T@q,torch.eye(4),atol=3e-6,rtol=0)
    assert torch.allclose(q@q.T,torch.eye(4),atol=3e-6,rtol=0)
    x=torch.randn(32,4,1,4)
    loss=smoothed_l0(x,q,.15)
    loss.backward()
    assert A.grad is not None and float(A.grad.abs().sum())>0


def test_each_unsupervised_objective_only_sees_dense_x():
    q=rotation_for_seed(SMOKE)
    x,support,_=sample_batch(
        torch.Generator().manual_seed(100000+SMOKE),16,q,family="train"
    )
    torch.manual_seed(SMOKE)
    read=qr_orthogonal(torch.randn(4,4))
    fake=torch.ones_like(support)
    assert not torch.equal(fake,support)
    for sigma in (.15,.3,.6):
        # The objective's entire signature is (observed x, V, sigma).
        a=smoothed_l0(x,read,sigma)
        b=smoothed_l0(x.clone(),read,sigma)
        assert torch.equal(a,b)
        assert torch.isfinite(a)
    seen_source=CONFIG["generator"]
    assert seen_source["heldout_generator_offset"]!=seen_source["train_generator_offset"]
    assert float((x.abs()>1e-6).float().mean())>.99


def test_42_degree_pairmix_control_explicitly_uses_q_only_in_stress_arm():
    bpd,gate,q,_,_=build_pd06(SMOKE,frozen_planted=False)
    random=initial_orthogonal_matrix(bpd,q,"cold015")
    baseline=initial_orthogonal_matrix(bpd,q,"anneal060_030_015")
    stress=initial_orthogonal_matrix(bpd,q,"privileged_pairmix015")
    assert torch.equal(random,baseline)
    assert not torch.equal(random,stress)
    assert torch.allclose(stress.T@stress,torch.eye(4),atol=3e-6,rtol=0)
    align=stress.T@q
    c=math.cos(math.radians(42))
    s=math.sin(math.radians(42))
    assert abs(float(align[0,0])-c)<3e-6
    assert abs(float(align[1,0])+s)<3e-6
    assert abs(float(align[0,1])-s)<3e-6
    assert abs(float(align[1,1])-c)<3e-6


def test_matched_train_batches_and_original_holdout_supports():
    q=rotation_for_seed(SMOKE)
    g1=torch.Generator().manual_seed(100000+SMOKE)
    g2=torch.Generator().manual_seed(100000+SMOKE)
    a,sa,_=sample_batch(g1,16,q,family="train")
    b,sb,_=sample_batch(g2,16,q,family="train")
    assert torch.equal(a,b) and torch.equal(sa,sb)
    seen={
        tuple(torch.nonzero(sa[i,t,0]).flatten().tolist())
        for i in range(16) for t in range(4)
    }
    assert seen.isdisjoint({(0,2),(1,3),(0,2,3)})
    _,unseen,_=sample_batch(
        torch.Generator().manual_seed(200000+SMOKE+10000*2),
        16,q,family="k2_unseen",
    )
    assert {
        tuple(torch.nonzero(unseen[i,t,0]).flatten().tolist())
        for i in range(16) for t in range(4)
    } <= {(0,2),(1,3)}


def test_six_arm_excluded_seed_and_real_bpd_mask_execution():
    torch.set_num_threads(1)
    rows=[]
    for arm in ARMS:
        result=run_arm(
            SMOKE,arm,steps=4,batch_size=8,heldout_boards=12,
        )
        rows.append(result)
        assert result["first_trainable_gradient_norm"]>0
        assert result["raw_matrix_parameter_update_norm"]>0
        assert result["optimizer_updates"]==4
        assert len(result["heldout"]["per_family"])==6
        assert all(v["mask_exact_binary"] for v in result["heldout"]["per_family"].values())
        if arm=="count_supervised_positive":
            assert result["used_cardinality_labels"]
        else:
            assert not result["used_cardinality_labels"]
        if arm=="privileged_pairmix015":
            assert result["starting_from_true_Q"]
        else:
            assert not result["starting_from_true_Q"]
        if result["orthogonal_tied_factors"]:
            assert result["heldout"]["physical_weight_delta_mse"]<1e-10
    assert len({r["source_rotation_sha256"] for r in rows})==1
    assert len({r["initial_parameter_sha256"] for r in rows})==1
    assert len({r["starting_raw_basis_sha256"] for r in rows if r["arm"] in
                ("cold015","cold030","warm030_to015","anneal060_030_015")})==1
    result=summarize(rows,registered=False)
    assert result["seed_count"]==1 and result["arm_count"]==6
    assert not result["registered_primary_passed"]
