"""Engineering-only PD14 tests on excluded seed 4001 and synthetic summaries.

No registered PD14 seed is optimized, evaluated or queried by this module.
"""
from __future__ import annotations

import hashlib
import inspect
import math

import torch

from mechinterp.bpd.theory.pd13.experiment import pool_for_seed as pd13_source
from mechinterp.bpd.theory.pd14.experiment import (
    ANGLE_GRID, ARMS, CELLS, CONFIG, DIM, PAIRS,
    fit_basis, initial_basis_for_restart, objective_diagnostic,
    run_arm, source_for_seed, summarize,
)

SMOKE = CONFIG["engineering_seed"]


def test_frozen_registration_and_unused_seed_units():
    assert CONFIG["study_id"] == "BPD-PD-14"
    assert len(CELLS) == 2
    assert CELLS[0]["margin"] == "strong" and CELLS[1]["margin"] == "weak"
    assert all(c["dimension"] == 16 and c["support_distribution"] == "clustered" for c in CELLS)
    seeds = [seed for c in CELLS for seed in c["seeds"]]
    assert len(set(seeds)) == len(seeds) == 32
    assert SMOKE not in seeds
    assert min(seeds) > 2633 and max(seeds) < 4001
    assert ARMS == ("wide_single", "narrow_single", "wide_restart",
                    "narrow_restart", "planted_Q_oracle")
    assert CONFIG["widths"]["wide"] == {"strong": .15, "weak": .05}
    assert CONFIG["widths"]["narrow"] == {"strong": .05, "weak": .02}
    assert CONFIG["scoring"]["primary_margin"] == "strong"
    assert CONFIG["scoring"]["primary_arm"] == "narrow_single"
    assert CONFIG["scoring"]["primary_control"] == "wide_single"
    assert CONFIG["scoring"]["primary_min_narrow_strong_successes"] == 8
    assert CONFIG["scoring"]["primary_net_paired_wins_min"] == 6
    assert CONFIG["scoring"]["objective_consistency_min_dual_signs"] == 15
    assert len(PAIRS) == 120 and len(ANGLE_GRID) == 37
    assert CONFIG["optimizer"]["candidates_per_learned_arm"] == 39960


def test_source_matches_original_pd12_and_pd13_without_new_source_logic():
    for cell in CELLS:
        q, x, digest, mask, latent = source_for_seed(SMOKE, cell, boards=96)
        q13, x13, sha13, m13, z13 = pd13_source(SMOKE, cell, boards=96)
        assert torch.equal(q, q13) and torch.equal(x, x13)
        assert torch.equal(mask, m13) and torch.equal(latent, z13)
        assert digest == sha13
        assert x.shape == mask.shape == latent.shape == (96, 4, 1, 16)
        assert hashlib.sha256(x.numpy().tobytes()).hexdigest() == digest
        assert torch.allclose(x.squeeze(2) @ q, latent.squeeze(2), atol=3e-6, rtol=0)


def test_source_only_restart_orthogonality_hashes_and_signature():
    assert set(inspect.signature(fit_basis).parameters) == {
        "observed", "seed", "sigma", "search", "sweeps_per_restart"
    }
    for prohibited in ("Q", "q", "support", "hidden", "latent", "heldout", "labels"):
        assert prohibited not in inspect.signature(fit_basis).parameters
    qs = [initial_basis_for_restart(SMOKE, i) for i in range(3)]
    hs = [hashlib.sha256(q.numpy().tobytes()).hexdigest() for q in qs]
    assert len(set(hs)) == 3
    for r in range(3):
        assert torch.equal(qs[r], initial_basis_for_restart(SMOKE, r))
        assert torch.allclose(qs[r].T @ qs[r], torch.eye(16, dtype=torch.float64),
                              atol=1e-12, rtol=0)
    assert hs[0] == hashlib.sha256(
        __import__("mechinterp.bpd.theory.pd12.experiment", fromlist=["initial_random_basis"])
        .initial_random_basis(SMOKE, 16).numpy().tobytes()
    ).hexdigest()


def test_equal_score_budget_monotone_and_deterministic_winner():
    cell = CELLS[0]
    q, x, _, _, _ = source_for_seed(SMOKE, cell, boards=32)
    for sigma in (.15, .05, .02):
        results = {}
        for method in ("single_long", "three_restarts"):
            read, t = fit_basis(x, SMOKE, sigma, method, sweeps_per_restart=1)
            results[method] = (read, t)
            assert t["candidate_angle_scores"] == 3 * 120 * 37
            assert t["candidate_scalar_projection_evaluations"] == 3 * 120 * 37 * 128
            assert t["candidate_sweeps_total"] == 3
            assert t["training_observed_vectors"] == 128
            assert t["orthogonality_max_error"] < 1e-8
            assert t["initial_basis_sha256_by_restart"][0] == hashlib.sha256(
                initial_basis_for_restart(SMOKE, 0).numpy().tobytes()
            ).hexdigest()
            for path in t["mass_paths_by_restart"]:
                assert all(y >= x - 1e-9 for x, y in zip(path, path[1:]))
            assert max(t["restart_final_masses"]) - t["selected_final_mass"] <= 1e-12
            replay, rt = fit_basis(x.clone(), SMOKE, sigma, method,
                                   sweeps_per_restart=1)
            assert torch.equal(read, replay)
            assert rt["selected_restart_index"] == t["selected_restart_index"]
            assert rt["restart_final_masses"] == t["restart_final_masses"]
        assert len(results["single_long"][1]["restart_final_masses"]) == 1
        assert len(results["three_restarts"][1]["restart_final_masses"]) == 3


def test_source_objective_diagnostic_is_independent_and_dual_sign_on_smoke():
    cell = CELLS[0]
    q, pool, _, _, _ = source_for_seed(SMOKE, cell, boards=128)
    d = objective_diagnostic(SMOKE, cell, q)
    assert d["boards"] == 1024
    assert d["dual_sign_strong"]
    assert d["widths"]["wide"]["remix_minus_planted"] > 0
    assert d["widths"]["narrow"]["remix_minus_planted"] < 0
    assert d["source_sha256"] != hashlib.sha256(pool.numpy().tobytes()).hexdigest()
    assert d == objective_diagnostic(SMOKE, cell, q)
    weak_cell = CELLS[1]
    qw, _, _, _, _ = source_for_seed(SMOKE, weak_cell, boards=16)
    dw = objective_diagnostic(SMOKE, weak_cell, qw, boards=64)
    assert dw["margin"] == "weak" and not dw["dual_sign_strong"]


def test_physical_mask_execution_and_matched_arms_excluded_smoke():
    for cell in CELLS:
        q, pool, sha, _, _ = source_for_seed(SMOKE, cell, boards=32)
        rows = [
            run_arm(SMOKE, cell, arm, q=q, pool=pool, pool_sha=sha,
                    heldout_boards=8, sweeps_per_restart=1)
            for arm in ARMS
        ]
        assert len({r["hidden_rotation_sha256"] for r in rows}) == 1
        assert len({r["initial_random_basis_sha256"] for r in rows}) == 1
        assert len({r["training_full_pool_sha256"] for r in rows}) == 1
        assert rows[-1]["success"]
        assert rows[-1]["heldout"]["alignment"]["mean_cosine"] > .999999
        for row in rows:
            assert set(row["heldout"]["per_family"]) == set(CONFIG["source"]["heldout_families"])
            assert all(v["binary_mask"] for v in row["heldout"]["per_family"].values())
            if row["arm"] == "planted_Q_oracle":
                assert row["uses_true_Q_for_optimizer"]
                assert row["training"]["candidate_angle_scores"] == 0
            else:
                assert not row["uses_true_Q_for_optimizer"]
                assert row["training"]["candidate_angle_scores"] == 13320
                assert row["training"]["candidate_scalar_projection_evaluations"] == 13320*128
                assert row["training"]["training_board_count"] == 32


def synthetic_rows(*, strong_narrow: int, strong_wide: int, dual: int):
    rows, diagnostics = [], []
    for cell in CELLS:
        for ix, seed in enumerate(cell["seeds"]):
            diagnostics.append({
                "seed": seed, "margin": cell["margin"],
                "dual_sign_strong": cell["margin"] == "strong" and ix < dual,
            })
            for arm in ARMS:
                success = arm == "planted_Q_oracle"
                if cell["margin"] == "strong":
                    if arm == "narrow_single":
                        success = ix < strong_narrow
                    elif arm == "wide_single":
                        success = ix >= 16 - strong_wide
                rows.append({
                    "seed": seed, "arm": arm, "success": success,
                    "heldout": {
                        "alignment": {"mean_cosine": 1.0 if success else .5},
                        "per_family": {"k4_unseen": {"support_fraction": 1.0 if success else 0.}},
                    }
                })
    return rows, diagnostics


def test_primary_science_decision_is_frozen_and_conjunctive():
    rows, diagnostics = synthetic_rows(strong_narrow=8, strong_wide=0, dual=15)
    s = summarize(rows, diagnostics, registered=True)
    assert s["registered"]
    assert s["strong_narrow_single_successes"] == 8
    assert s["strong_net_paired_wins"] == 8
    assert s["strong_McNemar_exact_two_sided_p"] <= .05
    assert s["objective_consistency_gate_passed"]
    assert s["primary_mechanism_gate_passed"]
    assert s["registered_scientific_passed"]
    fail_rows, diag = synthetic_rows(strong_narrow=7, strong_wide=0, dual=15)
    assert not summarize(fail_rows, diag, registered=True)["registered_scientific_passed"]
    ok_rows, wrong_diag = synthetic_rows(strong_narrow=8, strong_wide=0, dual=14)
    s2 = summarize(ok_rows, wrong_diag, registered=True)
    assert s2["primary_mechanism_gate_passed"]
    assert not s2["registered_scientific_passed"]
    paired, d = synthetic_rows(strong_narrow=8, strong_wide=2, dual=16)
    s3 = summarize(paired, d, registered=True)
    assert s3["strong_net_paired_wins"] == 6
    assert s3["strong_McNemar_exact_two_sided_p"] > .05
    assert not s3["registered_scientific_passed"]
    assert not summarize(rows, diagnostics, registered=False)["registered_scientific_passed"]
