"""BPD-PD-16 excluded-seed 4001 engineering tests only.

No registered scientific seed is used for engineering or model optimization.
Synthetic summaries use only predetermined booleans.
"""
from __future__ import annotations

import hashlib
import inspect

import torch

from mechinterp.bpd.theory.pd15.experiment import (
    fit_basis as pd15_fit_basis, weights_from_observed,
    effective_sample_size
)
from mechinterp.bpd.theory.pd16.experiment import (
    ANGLES, ARMS, CELL, CONFIG, FAMILIES, PAIRS, SEEDS,
    correlation, diagnose_weights, fit_shuffled, run_arm,
    shuffled_weights, source_for_seed, summarize,
)
from mechinterp.bpd.theory.pd12.experiment import initial_random_basis

SMOKE = CONFIG["engineering_seed"]


def test_frozen_registration_and_independence():
    assert CONFIG["study_id"] == "BPD-PD-16"
    assert CELL["dimension"] == 16 and CELL["margin"] == "strong"
    assert CELL["support_distribution"] == "clustered"
    assert len(set(SEEDS)) == len(SEEDS) == 32
    assert SMOKE not in SEEDS
    assert min(SEEDS)>3271 and max(SEEDS)<4001
    assert ARMS==("norm_aligned","norm_shuffled","uniform_control","planted_Q_oracle")
    assert CONFIG["objective"]["lambda"] == 1
    assert CONFIG["objective"]["sigma"] == .15
    assert CONFIG["objective"]["sweeps"] == 9
    assert CONFIG["objective"]["candidate_angle_scores"] == 39960
    assert CONFIG["objective"]["candidate_by_vector_scores"] == 327352320
    assert len(ANGLES) == 37 and len(PAIRS) == 120
    sc=CONFIG["scoring"]
    assert sc["primary_arm"] == "norm_aligned"
    assert sc["primary_control"] == "norm_shuffled"
    assert sc["primary_min_successes"] == 18
    assert sc["primary_net_paired_wins_min"] == 10
    assert sc["diagnostic_min_passing_seeds"] == 30


def test_identical_original_pd15_source_and_hash():
    q,x,sha,support,latent=source_for_seed(SMOKE,boards=96)
    assert x.shape==support.shape==latent.shape==(96,4,1,16)
    assert sha==hashlib.sha256(x.numpy().tobytes()).hexdigest()
    assert torch.allclose(
        x.squeeze(2)@q, latent.squeeze(2),atol=3e-6,rtol=0)
    replay=source_for_seed(SMOKE,boards=96)
    assert torch.equal(x,replay[1]) and sha==replay[2]
    assert sum(support.reshape(-1,16).sum(-1)==1)>0


def test_weight_multiset_ESS_and_seeded_permutation_independent_of_labels():
    q,x,_,support,_=source_for_seed(SMOKE,boards=2048)
    aligned=weights_from_observed(x,1.)
    shuffled,sha=shuffled_weights(x,SMOKE)
    assert shuffled.shape==aligned.shape==(8192,)
    assert sha==shuffled_weights(x.clone(),SMOKE)[1]
    assert torch.equal(shuffled,shuffled_weights(x.clone(),SMOKE)[0])
    assert torch.equal(torch.sort(shuffled).values,torch.sort(aligned).values)
    assert (shuffled != aligned).any()
    ess1,ess2=effective_sample_size(aligned),effective_sample_size(shuffled)
    assert abs(ess1-ess2)<1e-10
    assert .3 < ess1/8192 < .7
    d=diagnose_weights(SMOKE,x,support)
    assert d["weight_multiset_exact"]
    assert d["passed"]
    assert d["aligned_weight_k_pearson"] <= -.7
    assert abs(d["shuffled_weight_k_pearson"]) <= .05
    assert d["permutation_sha256"]==sha
    assert set(inspect.signature(fit_shuffled).parameters)=={
        "observed","initial","shuffled","sweeps"
    }
    assert not any(k in inspect.signature(fit_shuffled).parameters
                   for k in ("q","support","labels","latent","heldout"))


def test_weighted_givens_monotonic_budget_and_determinism():
    q,x,_,_,_=source_for_seed(SMOKE,boards=32)
    initial=initial_random_basis(SMOKE,16)
    shuffled,_=shuffled_weights(x,SMOKE)
    read,diag=fit_shuffled(x,initial,shuffled,sweeps=2)
    assert diag["candidate_angle_scores"]==2*120*37
    assert diag["candidate_scalar_projection_evaluations"]==2*120*37*128
    assert diag["training_observed_vectors"]==128
    assert diag["pair_sweeps"]==2
    assert len(diag["mass_by_sweep"])==3
    assert all(b>=a-1e-9 for a,b in zip(
        diag["mass_by_sweep"],diag["mass_by_sweep"][1:]))
    assert diag["orthogonality_max_error"]<1e-8
    again,rest=fit_shuffled(x.clone(),initial.clone(),shuffled.clone(),sweeps=2)
    assert torch.equal(read,again)
    assert diag["final_mass"]==rest["final_mass"]
    aligned,_=pd15_fit_basis(x,initial,.15,1.,sweeps=2)
    assert torch.allclose(aligned.T@aligned,torch.eye(16,dtype=torch.float64),atol=1e-8)


def test_four_physical_arms_and_oracle_on_excluded_seed():
    q,x,sha,support,latent=source_for_seed(SMOKE,boards=32)
    rows=[run_arm(SMOKE,arm,q=q,pool=x,pool_sha=sha,
                  heldout_boards=8,sweeps=1) for arm in ARMS]
    assert len(rows)==4
    for key in ("hidden_rotation_sha256","initial_random_basis_sha256",
                "training_full_pool_sha256"):
        assert len({r[key] for r in rows})==1
    assert rows[-1]["success"]
    for r in rows:
        assert set(r["heldout"]["per_family"])==set(FAMILIES)
        assert all(v["binary_mask"] for v in r["heldout"]["per_family"].values())
        if r["arm"]=="planted_Q_oracle":
            assert r["uses_true_Q_for_optimizer"]
            assert r["training"]["candidate_angle_scores"]==0
            assert all(v["support_fraction"]==1.
                       for v in r["heldout"]["per_family"].values())
        else:
            assert not r["uses_true_Q_for_optimizer"]
            assert r["training"]["candidate_angle_scores"]==120*37
            assert r["training"]["candidate_scalar_projection_evaluations"]==120*37*128
            assert r["training"]["training_board_count"]==32
    aligned=next(r for r in rows if r["arm"]=="norm_aligned")
    shuffled=next(r for r in rows if r["arm"]=="norm_shuffled")
    assert abs(aligned["training"]["training_weight_effective_sample_size"]-
               shuffled["training"]["training_weight_effective_sample_size"])<1e-10


def fake_results(*,aligned:int=20,shuffled:int=0,diagnostics_pass:int=30):
    rows,diagnostics=[],[]
    for i,seed in enumerate(SEEDS):
        diagnostics.append({"seed":seed,"passed":i<diagnostics_pass})
        for arm in ARMS:
            success = (arm=="planted_Q_oracle")
            if arm=="norm_aligned":
                success=i<aligned
            if arm=="norm_shuffled":
                success=i>=32-shuffled
            rows.append({
                "seed":seed,"arm":arm,"success":success,
                "heldout":{"alignment":{"mean_cosine":1. if success else .5},
                    "per_family":{"k4_unseen":{"support_fraction":1. if success else 0.}}}
            })
    return rows,diagnostics


def test_frozen_mcnemar_and_joint_diagnostic_decisions():
    rows,d=fake_results()
    s=summarize(rows,d,registered=True)
    assert s["registered"] and s["units"]==32 and s["arms"]==128
    assert s["success_by_arm"]["norm_aligned"]==20
    assert s["primary_only"]==20 and s["control_only"]==0
    assert s["McNemar_exact_two_sided_p"]<.05
    assert s["norm_cardinality_diagnostic_passes"]==30
    assert s["primary_mechanism_gate_passed"]
    assert s["norm_cardinality_diagnostic_gate_passed"]
    assert s["registered_scientific_passed"]
    rows,d=fake_results(aligned=17)
    assert not summarize(rows,d,registered=True)["registered_scientific_passed"]
    rows,d=fake_results(diagnostics_pass=29)
    assert not summarize(rows,d,registered=True)["registered_scientific_passed"]
    rows,d=fake_results(aligned=20,shuffled=10,diagnostics_pass=32)
    result=summarize(rows,d,registered=True)
    assert result["net_primary_wins"]==10
    assert not result["registered_scientific_passed"]
    assert not summarize(*fake_results(),registered=False)["registered_scientific_passed"]
