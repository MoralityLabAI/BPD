"""Excluded-source PD15 engineering checks (seed 4001 only).

Registered scientific sources are never trained, opened or evaluated here.
Frozen scoring is also tested using synthetic booleans with no model outputs.
"""
from __future__ import annotations

import hashlib
import inspect
import math

import torch

from mechinterp.bpd.theory.pd12.experiment import (
    best_pair_angle, initial_random_basis, support_banks,
)
from mechinterp.bpd.theory.pd13.experiment import pool_for_seed as pd13_source
from mechinterp.bpd.theory.pd14.experiment import fit_basis as pd14_fit
from mechinterp.bpd.theory.pd15.experiment import (
    ANGLE_GRID, ARMS, CELL, CONFIG, DIM, PAIRS, SEEDS,
    best_pair_angle_weighted, effective_sample_size, fit_basis,
    independent_objective_diagnostic, run_arm, source_for_seed,
    summarize, weighted_mass, weights_from_observed,
)

SMOKE = CONFIG["engineering_seed"]


def test_frozen_32_independent_roster_source_and_statistical_gates():
    assert CONFIG["study_id"]=="BPD-PD-15"
    assert CELL["margin"]=="strong" and CELL["dimension"]==16
    assert CELL["support_distribution"]=="clustered"
    assert len(SEEDS)==len(set(SEEDS))==32
    assert all(s>2953 and s<4001 for s in SEEDS)
    assert SMOKE not in SEEDS
    assert ARMS==("uniform_wide","weighted_wide","uniform_narrow",
                  "weighted_narrow","planted_Q_oracle")
    assert CONFIG["widths"]=={"wide":.15,"narrow":.05}
    assert CONFIG["weights"]=={"none":0,"norm1":1}
    assert CONFIG["objective"]["sweeps"]==9
    assert CONFIG["objective"]["candidate_angle_scores_per_learned_arm"]==39960
    assert CONFIG["scoring"]["primary_arm"]=="weighted_wide"
    assert CONFIG["scoring"]["primary_control"]=="uniform_wide"
    assert CONFIG["scoring"]["primary_min_successes"]==18
    assert CONFIG["scoring"]["primary_net_paired_wins_min"]==10
    assert CONFIG["scoring"]["objective_consistency_min_dual_signs"]==30
    assert len(PAIRS)==120 and len(ANGLE_GRID)==37


def test_unchanged_pd12_source_parity_with_prior_pd13():
    q,x,sha,support,latent=source_for_seed(SMOKE,boards=96)
    q13,x13,sha13,m13,l13=pd13_source(SMOKE,CELL,boards=96)
    assert torch.equal(q,q13) and torch.equal(x,x13)
    assert torch.equal(support,m13) and torch.equal(latent,l13)
    assert sha==sha13
    assert hashlib.sha256(x.numpy().tobytes()).hexdigest()==sha
    assert x.shape==support.shape==latent.shape==(96,4,1,16)
    assert torch.allclose(x.squeeze(2)@q,latent.squeeze(2),atol=3e-6,rtol=0)
    b=support_banks(16)
    for k in (1,2,3,4):
        assert not set(b["train"][k]).intersection(b["heldout"][k])


def test_sample_weights_use_only_observed_norm_and_are_rotation_invariant():
    q,x,sha,_,_=source_for_seed(SMOKE,boards=128)
    assert set(inspect.signature(fit_basis).parameters)=={
        "observed","initial","sigma","lam","sweeps"
    }
    z=weights_from_observed(x,1.)
    assert z.shape==(128*4,)
    assert bool((z>0).all() and (z<=1).all())
    assert z.max()>z.min()
    assert torch.equal(weights_from_observed(x,0.),torch.ones_like(z))
    v=initial_random_basis(SMOKE,16)
    rotated=(x.reshape(-1,16).double()@v).reshape(128,4,1,16)
    assert torch.allclose(z,weights_from_observed(rotated,1.),
                          rtol=1e-12,atol=1e-12)
    n_eff=effective_sample_size(z)
    assert 0<n_eff<len(z)
    assert abs(effective_sample_size(weights_from_observed(x,0.))-len(z))<1e-8
    assert not any(name in inspect.signature(fit_basis).parameters
                   for name in ("q","latent","support","mask","label","gold"))


def test_weighted_pair_score_reduces_to_original_when_weights_are_one():
    q,x,_,_,_=source_for_seed(SMOKE,boards=32)
    v=initial_random_basis(SMOKE,16)
    u=x.reshape(-1,16).double()@v
    theta_ref,score_ref=best_pair_angle(u[:,0],u[:,1],ANGLE_GRID,.15)
    theta_new,score_new=best_pair_angle_weighted(
        u[:,0],u[:,1],ANGLE_GRID,.15,
        torch.ones(u.shape[0],dtype=torch.float64)
    )
    assert abs(theta_new-theta_ref)<1e-12
    assert abs(score_new-score_ref)<1e-12


def test_givens_budget_determinism_monotone_mass_and_pd14_unweighted_parity():
    q,x,_,_,_=source_for_seed(SMOKE,boards=32)
    initial=initial_random_basis(SMOKE,16)
    for sigma in (.15,.05):
        for lam in (0.,1.):
            read,t=fit_basis(x,initial,sigma,lam,sweeps=2)
            assert t["candidate_angle_scores"]==2*120*37
            assert t["candidate_scalar_projection_evaluations"]==2*120*37*128
            assert t["pair_sweeps"]==2
            assert t["orthogonality_max_error"]<1e-8
            assert torch.allclose(read.T@read,torch.eye(16,dtype=torch.float64),
                                  atol=1e-8,rtol=0)
            assert all(y>=a-1e-9 for a,y in zip(
                t["mass_by_sweep"],t["mass_by_sweep"][1:]))
            replay,tt=fit_basis(x.clone(),initial.clone(),sigma,lam,sweeps=2)
            assert torch.equal(read,replay)
            assert t["final_mass"]==tt["final_mass"]
        # PD15 unweighted branch must be identical to PD14's original
        # single start at a matching two-sweep reduced engineering budget.
        old,_=pd14_fit(x,SMOKE,sigma,"single_long",sweeps_per_restart=1)
        # pd14 one engineering unit is 3 sweeps rather than the 2 above;
        # verify actual same 3-sweep procedure separately.
        fresh,td=fit_basis(x,initial,sigma,0.,sweeps=3)
        assert torch.equal(fresh,old)
        assert td["candidate_angle_scores"]==13320


def test_independent_objective_diagnostic_reverses_on_smoke():
    q,x,sha,_,_=source_for_seed(SMOKE,boards=64)
    d=independent_objective_diagnostic(SMOKE,q)
    assert d["boards"]==1024 and d["dual_sign"]
    assert d["weights"]["none"]["remix_minus_planted"]>0
    assert d["weights"]["norm1"]["remix_minus_planted"]<0
    assert .1<d["weights"]["norm1"]["effective_sample_fraction"]<.8
    assert d["source_sha256"]!=sha
    assert d==independent_objective_diagnostic(SMOKE,q)


def test_excluded_4001_full_physical_masks_and_oracle():
    q,x,sha,_,_=source_for_seed(SMOKE,boards=32)
    for arm in ARMS:
        rec=run_arm(SMOKE,arm,q=q,pool=x,pool_sha=sha,
                    heldout_boards=8,sweeps=1)
        assert rec["training_full_pool_sha256"]==sha
        assert rec["hidden_rotation_sha256"]==hashlib.sha256(q.numpy().tobytes()).hexdigest()
        assert set(rec["heldout"]["per_family"])==set(CONFIG["source"]["heldout_families"])
        assert all(v["binary_mask"] for v in rec["heldout"]["per_family"].values())
        if arm=="planted_Q_oracle":
            assert rec["uses_true_Q_for_optimizer"] and rec["success"]
            assert rec["training"]["candidate_angle_scores"]==0
            assert all(v["support_fraction"]==1.0
                       for v in rec["heldout"]["per_family"].values())
        else:
            assert not rec["uses_true_Q_for_optimizer"]
            assert rec["training"]["candidate_angle_scores"]==120*37
            assert rec["training"]["training_observed_vectors"]==128


def synthetic_receipts(primary:int=18, control:int=0, dual:int=30):
    rows,diagnostics=[],[]
    for i,seed in enumerate(SEEDS):
        diagnostics.append({"seed":seed,"dual_sign":i<dual})
        for arm in ARMS:
            success=(arm=="planted_Q_oracle")
            if arm=="weighted_wide":
                success=i<primary
            if arm=="uniform_wide":
                success=i>=32-control
            rows.append({
                "seed":seed,"arm":arm,"success":success,
                "heldout":{
                    "alignment":{"mean_cosine":1. if success else .5},
                    "per_family":{"k4_unseen":{"support_fraction":1. if success else 0.}}
                },
                "training":{"training_weight_effective_sample_fraction":1.},
            })
    return rows,diagnostics


def test_frozen_mechanism_and_objective_gates_are_conjunctive():
    rows,d=synthetic_receipts()
    s=summarize(rows,d,registered=True)
    assert s["registered"] and s["units"]==32 and s["arms"]==160
    assert s["success_by_arm"]["weighted_wide"]==18
    assert s["primary_only"]==18 and s["control_only"]==0
    assert s["net_primary_wins"]==18
    assert s["McNemar_exact_two_sided_p"]<.05
    assert s["source_objective_dual_sign_count"]==30
    assert s["objective_consistency_gate_passed"]
    assert s["primary_mechanism_gate_passed"] and s["registered_scientific_passed"]

    few,d=synthetic_receipts(primary=17)
    assert not summarize(few,d,registered=True)["registered_scientific_passed"]
    ok,weak_diag=synthetic_receipts(dual=29)
    sub=summarize(ok,weak_diag,registered=True)
    assert sub["primary_mechanism_gate_passed"]
    assert not sub["registered_scientific_passed"]
    discordant,d=synthetic_receipts(primary=18,control=8,dual=32)
    v=summarize(discordant,d,registered=True)
    assert v["net_primary_wins"]==10
    assert v["McNemar_exact_two_sided_p"]>.05
    assert not v["registered_scientific_passed"]
    assert not summarize(rows,d,registered=False)["registered_scientific_passed"]
