"""PD09 engineering-only tests on excluded seed 2009, before full science."""
from __future__ import annotations

import torch

from mechinterp.bpd.theory.pd06.experiment import (
    build as build_pd06, rank4_alignment, sample_batch, rotation_for_seed,
)
from mechinterp.bpd.theory.pd09.experiment import (
    CONFIG, SEEDS, ARMS, FAMILIES,
    qr_orthogonal, smoothed_l0, sigma_for_step, source_only_objective,
    set_physical_orthogonal, train_one, run_arm, summarize, evaluate_family,
)

SMOKE=CONFIG["smoke_seed"]


def test_frozen_study_source_and_new_seeds():
    assert CONFIG["study_id"]=="BPD-PD-09"
    assert len(SEEDS)==len(set(SEEDS))==24
    assert SEEDS==tuple(sorted(SEEDS))
    assert SMOKE not in SEEDS and min(SEEDS)>1289
    assert ARMS==("orthogonal_annealed","orthogonal_fixed","untied_sparse",
                  "random_orthogonal","count_supervised_positive")
    assert FAMILIES==("k1_seen","k2_seen","k2_unseen",
                     "k3_seen","k3_unseen","transition_unseen")
    assert CONFIG["optimization"]["steps"]==360
    assert CONFIG["scoring"]["primary_successes_min"]==18


def test_train_and_unseen_supports_are_disjoint():
    q=rotation_for_seed(SMOKE)
    rng=torch.Generator().manual_seed(100000+SMOKE)
    x,support,_=sample_batch(rng,160,q,family="train")
    assert x.shape==support.shape==(160,4,1,4)
    train={tuple(torch.nonzero(support[b,t,0]).flatten().tolist())
           for b in range(160) for t in range(4)}
    assert train.isdisjoint({(0,2),(1,3),(0,2,3)})
    assert torch.allclose(q.T@q,torch.eye(4),rtol=0,atol=3e-6)
    assert float((x.abs()>1e-6).float().mean())>.999


def test_orthogonal_qr_retraction_has_gradients_and_exact_BPD_full_weight():
    torch.set_num_threads(1)
    bpd,_,q,_,_=build_pd06(SMOKE,frozen_planted=False)
    raw=torch.nn.Parameter(bpd.wrappers["core"].V.detach().clone())
    ortho=qr_orthogonal(raw)
    assert torch.allclose(ortho.T@ortho,torch.eye(4),atol=3e-6,rtol=0)
    x,_,_=sample_batch(torch.Generator().manual_seed(102009),16,q,family="train")
    for arm in ("orthogonal_annealed","orthogonal_fixed"):
        raw.grad=None
        source_only_objective(arm,x,raw_rotation=raw,step=0).backward()
        assert raw.grad is not None and float(raw.grad.abs().sum())>0
    set_physical_orthogonal(bpd,ortho.detach())
    assert torch.allclose(
        bpd.wrappers["core"].reconstructed_weight(),torch.eye(4),
        rtol=0,atol=3e-6
    )
    assert torch.allclose(
        bpd.wrappers["core"].V.T@bpd.wrappers["core"].V,
        torch.eye(4),atol=3e-6
    )


def test_annealing_schedule_and_no_label_access_in_unsupervised_losses():
    assert [sigma_for_step("orthogonal_annealed",i) for i in
            (0,119,120,239,240,359)]==[.6,.6,.3,.3,.15,.15]
    assert [sigma_for_step("orthogonal_fixed",i) for i in
            (0,119,120,239,240,359)]==[.6]*6
    bpd,_,q,_,_=build_pd06(SMOKE,frozen_planted=False)
    x,support,_=sample_batch(torch.Generator().manual_seed(102009),
                             8,q,family="train")
    bogus=torch.zeros_like(support)
    assert not torch.equal(support,bogus)
    raw=torch.nn.Parameter(bpd.wrappers["core"].V.detach().clone())
    # Loss API accepts x and factors. Neither labels nor source family exist
    # in these calls; source count is never read in unsupervised arms.
    first=source_only_objective("orthogonal_annealed",x,raw_rotation=raw,step=5)
    again=source_only_objective("orthogonal_annealed",x,raw_rotation=raw,step=5)
    assert torch.equal(first,again)
    untied=source_only_objective("untied_sparse",x,bpd=bpd,step=10)
    untied.backward()
    assert all(p.grad is not None for p in bpd.component_parameters())
    assert sum(float(p.grad.square().sum()) for p in bpd.component_parameters())>0


def test_fixed_planted_read_gate_has_exact_mask_on_all_six_families():
    bpd,gate,q,_,_=build_pd06(SMOKE,frozen_planted=True)
    align=rank4_alignment(bpd,q)
    assert align["mean_cosine"]>.999999
    for family in FAMILIES:
        row=evaluate_family(
            bpd,q,SMOKE,"random_orthogonal",family,
            align["component_to_latent_permutation"],boards=16
        )
        assert row["support_fraction"]==1
        assert row["false_off"]==0 and row["false_on"]==0
        assert row["mse"]<1e-10
        assert row["mask_exact_binary"]


def test_all_source_arms_train_on_excluded_seed_and_keep_same_initial_state():
    torch.set_num_threads(1)
    rows=[
        run_arm(SMOKE,arm,steps=4,batch_size=8,heldout_boards=16)
        for arm in ARMS
    ]
    assert len({r["initial_parameters_sha256"] for r in rows})==1
    assert len({r["rotation_sha256"] for r in rows})==1
    for r in rows:
        assert r["seed"]==SMOKE
        assert len(r["heldout"]["per_family"])==6
        assert all(x["mask_exact_binary"] for x in
                   r["heldout"]["per_family"].values())
        if r["arm"]=="random_orthogonal":
            assert r["training_updates"]==0
            assert r["trainable_update_norm"]==0
        else:
            assert r["trainable_update_norm"]>0
            assert r["first_trainable_gradient_norm"]>0
        if r["raw_matrix_QR_gauge"]:
            assert r["heldout"]["physical_weight_delta_mse"]<1e-8
    report=summarize(rows,registered=False)
    assert report["seed_count"]==1 and report["arm_count"]==5
    assert not report["primary_passed"]
