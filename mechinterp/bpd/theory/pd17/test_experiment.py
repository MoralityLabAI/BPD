"""PD17 engineering tests on excluded seed 4001 and synthetic root summaries.

No registered scientific seed is optimized/evaluated by this test suite.
"""
from __future__ import annotations

import inspect

import torch

from mechinterp.bpd.theory.pd17.source import CONFIG, training_source, FAMILIES
from mechinterp.bpd.theory.pd17.experiment import (
    SEEDS, ARMS, LAWS, SC, evaluate_family_exact,
    paired_root_bootstrap, root_source_diagnostic,
    run_arm, summarize,
)

SMOKE=CONFIG["excluded_engineering_seed"]


def test_original_registration_and_root_independence():
    assert CONFIG["study_id"]=="BPD-PD-17"
    assert len(SEEDS)==len(set(SEEDS))==32
    assert SMOKE not in SEEDS
    assert LAWS==("original","radial_camouflage")
    assert ARMS==("uniform_givens","norm_givens","angular_atoms","planted_Q_oracle")
    assert CONFIG["givens"]["sigma"]==.15
    assert CONFIG["givens"]["sweeps"]==9
    assert CONFIG["givens"]["candidate_angle_scores"]==39960
    assert CONFIG["givens"]["candidate_by_vector_evaluations"]==327352320
    assert len(FAMILIES)==8
    assert SC["original_norm_replication_min_successes"]==24
    assert SC["angular_camouflage_feasibility_min_successes"]==24
    assert SC["source_independence_gate_min_roots"]==30


def test_excluded_original_and_camouflage_source_diagnostic():
    orig=training_source(SMOKE,"original",boards=2048)
    cam=training_source(SMOKE,"radial_camouflage",boards=2048)
    d=root_source_diagnostic(SMOKE,orig,cam)
    assert d["source_independence_gate_passed"]
    assert d["original_corr_K_norm2"]>.7
    assert abs(d["camouflaged_corr_K_norm2"])<.05
    assert .9999<d["camouflage_radial_min"]<1.01
    assert 1.09<d["camouflage_radial_max"]<1.10001
    assert d["paired_q_sha256"]==orig["q_sha256"]
    assert d["paired_support_sha256"]==cam["support_sha256"]


def test_native_BPD_binary_masks_and_oracle_both_sources():
    for law in LAWS:
        pool=training_source(SMOKE,law,boards=32)
        oracle=run_arm(SMOKE,law,"planted_Q_oracle",pool=pool,boards=12)
        assert oracle["success"]
        assert oracle["heldout"]["alignment"]["mean_cosine"]>.999999
        assert oracle["training"]["candidate_angle_scores"]==0
        assert all(row["binary_mask"] and row["support_fraction"]==1.0
                   and row["false_off"]==0 and row["false_on"]==0
                   for row in oracle["heldout"]["per_family"].values())
        for arm in ("uniform_givens","norm_givens"):
            row=run_arm(SMOKE,law,arm,pool=pool,boards=8,sweeps=1)
            assert not row["uses_Q_in_optimizer"]
            assert row["training"]["candidate_angle_scores"]==120*37
            assert row["training"]["candidate_scalar_projection_evaluations"]==120*37*128
            assert row["training"]["training_board_count"]==32
            assert row["training"]["orthogonality_max_error"]<1e-6
            assert all(v["binary_mask"] for v in row["heldout"]["per_family"].values())


def test_native_angular_method_on_excluded_camouflage_source():
    pool=training_source(SMOKE,"radial_camouflage",boards=512)
    row=run_arm(SMOKE,"radial_camouflage","angular_atoms",
                pool=pool,boards=12)
    assert not row["uses_Q_in_optimizer"]
    assert row["training"]["candidate_count"]==1024
    assert row["training"]["nonzero_moves"]==16
    assert not row["training"]["abstained"]
    assert row["success"]
    assert row["heldout"]["alignment"]["mean_cosine"]>.999999
    assert all(v["support_fraction"]==1. for v in row["heldout"]["per_family"].values())


def _synthetic(*,original_norm=32,cam_norm=0,cam_uniform=0,
               cam_angular=32,original_uniform=0,diagnostic_count=32,
               oracle_fails=0):
    # Synthetic root outcomes only; registered scientific seeds are not evaluated.
    rows=[]
    diagnostics=[]
    for i,seed in enumerate(SEEDS):
        diagnostics.append({
            "seed":seed,
            "original_corr_K_norm2":.9,
            "camouflaged_corr_K_norm2":.01,
            "source_independence_gate_passed":i<diagnostic_count,
        })
        for law in LAWS:
            for arm in ARMS:
                ok=arm=="planted_Q_oracle"
                if law=="original":
                    if arm=="norm_givens": ok=i<original_norm
                    if arm=="uniform_givens":ok=i<original_uniform
                else:
                    if arm=="norm_givens":ok=i<cam_norm
                    if arm=="uniform_givens":ok=i<cam_uniform
                    if arm=="angular_atoms":ok=i<cam_angular
                if arm=="planted_Q_oracle" and i<oracle_fails and law=="radial_camouflage":
                    ok=False
                rows.append({"seed":seed,"law":law,"arm":arm,"success":ok})
    return rows,diagnostics


def test_bootstrap_is_root_clustered_and_reproducible():
    vals=[1]*32
    ci=paired_root_bootstrap(vals)
    assert ci["resamples"]==10000
    assert ci["lower95"]==1 and ci["upper95"]==1
    out=paired_root_bootstrap([0]*32)
    assert out["lower95"]==0 and out["upper95"]==0
    assert paired_root_bootstrap(list(range(32)))==paired_root_bootstrap(list(range(32)))


def test_registered_science_joint_gates_and_falsifiers():
    rows,d=_synthetic()
    s=summarize(rows,d,registered=True)
    assert s["registered"] and s["units"]==32 and s["arms"]==256
    assert s["mean_root_interaction"]==1.
    assert s["registered_scientific_passed"]
    assert s["physical_oracles_admitted"]
    assert s["camouflage_observability_gate_passed"]
    assert s["root_bootstrap"]["lower95"]==1.
    fail,d=_synthetic(original_norm=23)
    assert not summarize(fail,d,registered=True)["registered_scientific_passed"]
    fail,d=_synthetic(cam_norm=32)
    assert summarize(fail,d,registered=True)["mean_root_interaction"]==0.
    assert not summarize(fail,d,registered=True)["registered_scientific_passed"]
    fail,d=_synthetic(cam_angular=23)
    assert not summarize(fail,d,registered=True)["registered_scientific_passed"]
    fail,d=_synthetic(diagnostic_count=29)
    assert not summarize(fail,d,registered=True)["registered_scientific_passed"]
    fail,d=_synthetic(oracle_fails=1)
    assert not summarize(fail,d,registered=True)["registered_scientific_passed"]
    assert not summarize(rows,d,registered=False)["registered_scientific_passed"]
