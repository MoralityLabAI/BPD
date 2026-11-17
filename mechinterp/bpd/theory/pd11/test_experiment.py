"""Source-only PD11 engineering tests on excluded seed 2029.

The tests may use short sweeps and small source batches; no registered seed
or registered full-budget model is fit before EXECUTE_LOCK.json.
"""
from __future__ import annotations

import math

import torch

from mechinterp.bpd.theory.pd06.experiment import (
    build as build_pd06, sample_batch, rotation_for_seed,
)
from mechinterp.bpd.theory.pd11.experiment import (
    CONFIG, SEEDS, ARMS, FAMILIES, PAIRS,
    angle_grid, best_pair_angle, initialize_basis, gaussian_mass,
    search_basis, source_only_training, run_arm, summarize,
)

SMOKE = CONFIG["smoke_seed"]


def test_exact_registration_and_seed_disjointness():
    assert CONFIG["study_id"] == "BPD-PD-11"
    assert len(SEEDS) == len(set(SEEDS)) == 24
    assert tuple(sorted(SEEDS)) == SEEDS
    assert SMOKE not in SEEDS and min(SEEDS) > 1621
    assert ARMS == (
        "global_givens", "local_givens",
        "global_pairmix_privileged", "local_pairmix_privileged",
        "random_orthogonal", "count_supervised_positive",
    )
    assert len(FAMILIES) == 6
    assert len(PAIRS) == 6
    assert CONFIG["objective"]["sweep_count"] == 6
    assert CONFIG["objective"]["angles_per_pair"] == 73
    assert CONFIG["scoring"]["primary_successes_min"] == 18


def test_full_candidate_grid_and_matched_eval_budget():
    global_grid = angle_grid("global_givens")
    local_grid = angle_grid("local_givens")
    assert len(global_grid) == len(local_grid) == 73
    assert abs(float(global_grid[0]) + math.pi / 2) < 1e-12
    assert abs(float(global_grid[-1]) - math.pi / 2) < 1e-12
    assert abs(float(local_grid[0]) + math.radians(5)) < 1e-12
    assert abs(float(local_grid[-1]) - math.radians(5)) < 1e-12
    assert abs(float(global_grid[36])) < 1e-12
    assert abs(float(local_grid[36])) < 1e-12
    assert CONFIG["objective"]["sweep_count"] * len(PAIRS) * 73 == 2628


def test_source_no_leakage_and_reserved_support_split():
    q = rotation_for_seed(SMOKE)
    rng = torch.Generator().manual_seed(100000 + SMOKE)
    x, source_support, _ = sample_batch(rng, 32, q, family="train")
    assert x.shape == source_support.shape == (32,4,1,4)
    assert float((x.abs() > 1e-6).float().mean()) > .99
    supports = {
        tuple(torch.nonzero(source_support[b,t,0]).flatten().tolist())
        for b in range(32) for t in range(4)
    }
    assert supports.isdisjoint({(0,2),(1,3),(0,2,3)})
    bpd, _, q2, _, _ = build_pd06(SMOKE, frozen_planted=False)
    basis = initialize_basis(bpd, q2, "global_givens")
    # Only observed x, basis and declared optimizer choices reach the search.
    poisoned = torch.ones_like(source_support)
    assert not torch.equal(poisoned, source_support)
    out1, diag1 = search_basis(x, basis, arm="global_givens",
                              sweeps=1, angles_per_pair=9)
    out2, diag2 = search_basis(x.clone(), basis.clone(), arm="global_givens",
                              sweeps=1, angles_per_pair=9)
    assert torch.equal(out1, out2)
    assert diag1 == diag2


def test_pair_search_is_monotone_and_orthogonal():
    q = rotation_for_seed(SMOKE)
    g = torch.Generator().manual_seed(100000 + SMOKE)
    x, _, _ = sample_batch(g, 28, q, family="train")
    bpd, _, true_q, _, _ = build_pd06(SMOKE, frozen_planted=False)
    basis = initialize_basis(bpd, true_q, "global_givens")
    assert torch.allclose(basis.T @ basis, torch.eye(4,dtype=torch.float64),atol=2e-6)
    for mode in ("global_givens","local_givens"):
        result, diag = search_basis(x,basis,arm=mode,sweeps=2,angles_per_pair=9)
        assert diag["number_of_candidate_pair_scores"] == 2*6*9
        assert diag["pair_updates_considered"] == 2*6
        assert diag["angle_candidates_per_pair"] == 9
        assert diag["orthogonality_max_error"] < 2e-6
        assert abs(diag["initial_gaussian_mass"]-gaussian_mass(x.reshape(-1,4).double(),basis,.15))<1e-10
        for a,b in zip(diag["gaussian_mass_by_sweep"],diag["gaussian_mass_by_sweep"][1:]):
            assert b >= a - 1e-10, (mode,a,b)
        assert torch.allclose(result.T@result,torch.eye(4,dtype=torch.float64),atol=2e-6)


def test_privileged_pairmix_is_declared_and_distinct_from_random_initialization():
    bpd, _, q, _, _ = build_pd06(SMOKE, frozen_planted=False)
    normal = initialize_basis(bpd, q, "global_givens")
    local = initialize_basis(bpd, q, "local_givens")
    mix_global = initialize_basis(bpd,q,"global_pairmix_privileged")
    mix_local = initialize_basis(bpd,q,"local_pairmix_privileged")
    assert torch.equal(normal,local)
    assert torch.equal(mix_global,mix_local)
    assert not torch.equal(normal,mix_global)
    assert torch.allclose(mix_global.T@mix_global,torch.eye(4,dtype=torch.float64),atol=2e-6)
    qtx = q.double().T@mix_global
    c,s = math.cos(math.radians(44)),math.sin(math.radians(44))
    assert abs(float(qtx[0,0])-c)<2e-6
    assert abs(float(qtx[0,1])+s)<2e-6
    assert abs(float(qtx[1,0])-s)<2e-6
    assert abs(float(qtx[1,1])-c)<2e-6


def test_all_six_arms_physical_bpd_and_first_run_budgets():
    rows = [
        run_arm(SMOKE,arm,boards=24,sweeps=2,angles_per_pair=9,
                positive_steps=3,heldout_boards=10)
        for arm in ARMS
    ]
    assert len(rows)==6
    assert len({r["source_rotation_sha256"] for r in rows})==1
    assert len({r["initial_parameters_sha256"] for r in rows})==1
    assert len({r["initial_basis_sha256"] for r in rows if r["arm"] in
                ("global_givens","local_givens","random_orthogonal")})==1
    for row in rows:
        assert row["seed"]==SMOKE
        assert len(row["heldout"]["per_family"])==6
        assert all(x["mask_exact_binary"]
                   for x in row["heldout"]["per_family"].values())
        if row["arm"] in ("global_givens","local_givens",
                          "global_pairmix_privileged","local_pairmix_privileged"):
            assert row["number_of_candidate_pair_scores"]==2*6*9
            assert row["orthogonality_max_error"] < 2e-6
            assert 0 <= row["actual_nonzero_pair_updates"] <= 2 * len(PAIRS)
            assert not row["used_cardinality_labels"]
        if row["arm"].endswith("_privileged"):
            assert row["privileged_true_Q_initializer"]
        if row["arm"]=="count_supervised_positive":
            assert row["used_cardinality_labels"]
            assert row["first_factor_gradient_norm"]>0
            assert row["factor_update_norm"]>0
        if row["arm"]=="random_orthogonal":
            assert row["number_of_candidate_pair_scores"]==0
            assert row["actual_nonzero_pair_updates"]==0
    summary=summarize(rows,registered=False)
    assert summary["seed_count"]==1 and summary["arm_count"]==6
    assert not summary["registered_primary_passed"]
