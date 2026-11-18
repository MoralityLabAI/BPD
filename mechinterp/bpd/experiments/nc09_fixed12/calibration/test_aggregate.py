"""FC01 synthetic original-sham calibration and exact EvalMath output tests.

These tests never fit development/calibration source models.
"""
from __future__ import annotations

from copy import deepcopy
from statistics import median

import pytest

from mechinterp.bpd.experiments.nc09_fixed12.calibration.aggregate import (
    aggregate,make_evalmath_pilot,ROOTS,CALIBRATION_CONFIDENCE
)
from mechinterp.bpd.experiments.nc09_fixed12.calibration.experiment import (
    PROTO_SHA,PARENT_SOURCE_SHA
)


def synthetic(*,effect=.032,null_magnitude=.004,fail_root=None):
    docs=[]
    for i,root in enumerate(ROOTS):
        shams=[.24+null_magnitude*(j-3.5)/3.5 for j in range(8)]
        m=median(shams)
        null=[float(v-m) for v in shams]
        docs.append({
            "study":"BPD-NC09-FC01","phase":"INDEPENDENT_SHAM_CALIBRATION",
            "root":root,"protocol_sha256":PROTO_SHA,
            "parent_integrity":{
                "nc09_original_experiment_sha256":PARENT_SOURCE_SHA
            },
            "source_sha256":"dummy-branch-scientific-source",
            "eligible":root!=fail_root,
            "source_rank12_optimum":{"ranks":[4,3,3,2],"total_rank":12},
            "physical_source_gain":1.02,
            "sham_admission":{"selected":8,"qualified":16,"proposals":4096},
            "sham_nlls":shams,
            "effects":{
                "A_flat3_minus_dp12":effect+(i%3-1)*.001,
                "D_dose_flat3_minus_dp12":effect+.003+(i%3-1)*.001,
                "S_median_sham_minus_dp12":.051+(i%3-1)*.001,
                "C_dp12_minus_clean":.011+(i%3-1)*.001,
                "F_dp12_minus_flat4":.004+(i%3-1)*.001,
                "null_sham_signed_effects":null,
                "T_null_worst_abs":max(abs(v) for v in null),
            }
        })
    return docs


def test_59_independent_sham_worlds_admit_evalmath_after_exact_null():
    s,power_spec,power_result=aggregate(synthetic())
    assert s["root_count"]==59 and s["eligible_root_count"]==59
    assert s["full_59_root_admission"] and s["order_statistic_rank"]==59
    assert s["95_95_sham_floor"]>0
    assert s["original_95_95_root_tolerance_coverage"]>=.95
    assert s["calibration_verdict"]=="CALIBRATION_READY_FOR_SEPARATE_CONFIRMATION_PROTOCOL"
    assert s["evalmath_decision"]=="READY_FOR_CONFIRMATION"
    assert power_result["order_statistic_rank"]==59
    assert len(power_spec["pilot"])==59
    assert power_spec["confirm_units_planned"]==32
    assert power_spec["planned_comparisons"]==3
    assert power_spec["heldout_sealed"]
    assert s["independent_confirmation_roots_evaluated"]==0


def test_sham_variation_above_effect_floor_blocks_confirmation():
    s,p,e=aggregate(synthetic(null_magnitude=.11))
    assert s["full_59_root_admission"]
    assert s["95_95_sham_floor"]>.027
    assert s["calibration_verdict"]=="CALIBRATION_REDESIGN_OR_PILOT_ONLY"
    assert "pilot_effect_not_above_mechanistic_variation_floor" in e["reasons"]


def test_missing_model_keeps_tolerance_uncalibrated():
    s,p,e=aggregate(synthetic(fail_root=12017))
    assert s["eligible_root_count"]==58
    assert s["calibration_verdict"]=="CALIBRATION_INCOMPLETE"
    assert s["95_95_sham_floor"] is None
    assert s["evalmath_decision"] is None and p is None and e is None
    assert s["independent_confirmation_roots_evaluated"]==0


def test_unique_roots_required_without_replenishment():
    a=synthetic()
    with pytest.raises(ValueError):
        aggregate(a[:-1])
    b=synthetic()
    b[0]["root"]=b[1]["root"]
    with pytest.raises(ValueError):
        aggregate(b)
    c=synthetic()
    c[0]["parent_integrity"]["nc09_original_experiment_sha256"]="wrong"
    with pytest.raises(AssertionError):
        aggregate(c)
