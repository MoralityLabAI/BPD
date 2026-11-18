"""Root-cluster aggregate pseudo-sham null: exact centering and frozen gates.

This suite uses wholly synthetic JSON-like receipts, never trains 13000..13058.
"""
from __future__ import annotations

import math
from statistics import median

import numpy as np
import pytest

from mechinterp.bpd.experiments.nc09_fixed12.tangent_sham_calibration.aggregate import (
    ROOTS,aggregate,aggregate_root_mean_null,root_mean_bootstrap,
    approximate_power,
)
from mechinterp.bpd.experiments.nc09_fixed12.tangent_sham_calibration.experiment import PROTO_SHA


def fake_receipts(*,sham_width=.004,advantage=.053,block_seed=None):
    docs=[]
    for i,root in enumerate(ROOTS):
        center=.16+.001*(i%5-2)
        shams=[center+sham_width*(j-3.5)/3.5 for j in range(8)]
        sham_med=median(shams)
        dp=float(np.mean(shams)-advantage-.001*(i%4-1.5))
        null=[float((sum(shams)-v)/7-v) for v in shams]
        j0=int(np.random.default_rng(18_180_000+root).integers(0,8))
        effects={
            "A_flat3_minus_dp12":.033+.001*(i%5-2),
            "D_dose_flat3_minus_dp12":.032+.001*(i%5-2),
            "S_mean_sham_minus_dp12":float(np.mean(shams)-dp),
            "S_median_sham_minus_dp12":float(sham_med-dp),
            "C_dp12_minus_clean":.014+.001*(i%5-2),
            "F_dp12_minus_flat4":.006+.001*(i%5-2),
            "null_sham_signed_effects":[float(v-sham_med) for v in shams],
            "T_null_worst_abs":float(max(abs(v-sham_med) for v in shams)),
            "Z_pseudo_null_by_sham":null,
            "pseudo_selected_sham_index":j0,
            "Z_preselected_pseudo_sham":null[j0],
        }
        docs.append({
            "study":"BPD-NC09-FC02",
            "phase":"AGGREGATE_TANGENT_SHAM_CALIBRATION",
            "root":root,
            "protocol_sha256":PROTO_SHA,
            "source_sha256":"independently frozen synthetic test",
            "eligible":root!=block_seed,
            "source_rank12_optimum":{"ranks":[4,3,3,2],"total_rank":12},
            "sham_nlls":shams,
            "sham_admission":{
                "selected":8 if root!=block_seed else 7,
                "qualified":34,"proposals":4096,
                "uses_external_labels":False,
                "target_signed_score_mean":.03,
                "target_first_order_score_rms":.1,
                "signed_score_abs_tolerance":.0075,
                "rms_score_abs_tolerance":.025,
                "selected_dose_relative_errors":[.01]*8,
                "selected_score_mean_abs_errors":[.001]*8,
                "selected_score_rms_abs_errors":[.005]*8,
            },
            "task_nll_by_arm":{"dp12":{"nll_nats":dp}},
            "effects":effects,
        })
    return docs


def test_exact_centering_and_mean_bootstrap_determinism():
    roots=fake_receipts()
    z=np.asarray([r["effects"]["Z_pseudo_null_by_sham"] for r in roots])
    assert z.shape==(59,8)
    assert np.max(np.abs(z.mean(1)))<1e-12
    s=aggregate_root_mean_null(z)
    assert s["outer_resamples"]==256
    assert s["inner_simulations_per_outer"]==4096
    assert s["future_model_roots_per_simulated_mean"]==32
    assert 0<=s["upper95_across_outer_95pct_estimates"]<.01
    assert len(s["outer_95pct_tail_estimates"])==256
    assert not s["mathematical_exact_coverage_claim"]


def test_frozen_positive_engineering_calibration_on_synthetic_root_sources():
    result=aggregate(fake_receipts())
    assert result["independent_model_roots"]==59
    assert result["eligible_roots"]==59
    assert result["registered_development_verdict"]=="AGGREGATE_TANGENT_NULL_DEVELOPMENT_POSITIVE"
    assert all(result["gates"].values())
    assert result["approximate_power"]["engineering_power_gate_passed"]
    assert result["root_mean_bootstrap"]["S_mean_sham_minus_dp12"]["mean"]>.05
    assert result["rootwise_sham_max_95_95_diagnostic"]<.01
    assert result["scientific_confirmation_roots_evaluated"]==0
    assert not result["calibration_scientific_confirmation_ready"]


def test_negative_incomplete_and_source_drift_fail_closed():
    s=aggregate(fake_receipts(block_seed=13027))
    assert s["registered_development_verdict"]=="CALIBRATION_INCOMPLETE"
    assert s["eligible_roots"]==58
    assert s["aggregate_32_root_pseudo_null"] is None
    assert s["root_mean_bootstrap"] is None
    assert not s["calibration_scientific_confirmation_ready"]
    wrong=fake_receipts()
    wrong[0]["protocol_sha256"]="not-original-source"
    with pytest.raises(AssertionError):
        aggregate(wrong)
    duplicate=fake_receipts()
    duplicate[1]["root"]=duplicate[0]["root"]
    with pytest.raises(ValueError):
        aggregate(duplicate)
    invalid=fake_receipts()
    invalid[10]["effects"]["Z_pseudo_null_by_sham"][0]+=.1
    with pytest.raises(AssertionError):
        aggregate(invalid)


def test_frozen_normal_planning_floor_and_minimum_exact_sign_roots():
    p=approximate_power(.01,.03,.02)
    assert p["normal_approx_80pct_power_roots_required"] is None
    assert not p["engineering_power_gate_passed"]
    assert p["exact_minimum_all_positive_root_sign_units"]>=6
    rich=approximate_power(.10,.01,.01)
    assert rich["engineering_power_gate_passed"]
