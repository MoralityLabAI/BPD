"""Synthetic 16-root FC04 decision and independent archived physical geometry tests."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

import pytest
import torch
import numpy as np

from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_fc04.solver import (
    fit_full_tangent_shams,projectors_from_bases
)
from mechinterp.bpd.experiments.nc09_fixed12.grassmann_fc03.solver import projector_digest
from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_fc04.aggregate import (
    aggregate,independently_audit_full_tangent_projectors,root_bootstrap
)
from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_fc04.experiment import (
    ROOTS,PROTO_SHA,PARENT_SOURCE_SHA
)


@lru_cache(maxsize=1)
def source_fixture():
    rng=torch.Generator().manual_seed(15151)
    a=torch.randn((256,4,20),generator=rng)
    z=torch.randn((256,4,20),generator=rng)
    g=torch.zeros_like(z)
    e=[torch.eye(20,dtype=torch.float64)[:,:rank]
       for rank in (4,3,3,2)]
    shams,receipt=fit_full_tangent_shams(
        a,torch.eye(20,dtype=torch.float64),z,g,e,14999,
        starts=16,steps=4,want=8
    )
    assert len(shams)==8
    result=deepcopy(receipt)
    result["charts_configured"]=32
    result["max_corrections_per_chart"]=64
    result["charts"].extend({
        "chart":i,"status":"skipped_after_full","updates":0,
        "source_metric_evaluations":0
    } for i in range(len(result["charts"]),32))
    return result,projector_digest(projectors_from_bases(e))


def roots(*,baseline_fc03=10,blocked=None):
    source,original=source_fixture()
    out=[]
    for i,seed in enumerate(ROOTS):
        adm=deepcopy(source)
        sc=[.17]*8
        if i==blocked:
            adm["eligible"]=False
            adm["selected"]=7
            adm["projectors"]=adm["projectors"][:7]
            sc=sc[:7]
        baseline_full=i<baseline_fc03
        old_fc03={
            "charts_configured":32,"max_adam_steps_per_chart":64,
            "selected":8 if baseline_full else 5,
        }
        old_fc02={
            "proposals":4096,"selected":8 if i<8 else 4,
        }
        out.append({
            "study":"BPD-NC09-FC04",
            "phase":"FULL_TANGENT_JACOBIAN_ENGINEERING_DEVELOPMENT",
            "protocol_sha256":PROTO_SHA,"root":seed,
            "parent_integrity":{"nc09_original_experiment_sha256":PARENT_SOURCE_SHA},
            "source_sha256":"synthetic-first-root-testing",
            "eligible":i!=blocked,
            "source_gain_eligible":True,
            "full_mask_closure":{"passed":True},
            "source_rank12_optimum":{"ranks":[4,3,3,2],"total_rank":12},
            "source_dp12_projector_receipt":deepcopy(original),
            "sham_admission":adm,
            "sham_nlls":sc,
            "fc03_original_eightdim_admission":old_fc03,
            "fc03_original_eightdim_sham_nlls":[.17]*old_fc03["selected"],
            "fc02_original_rejection_admission":old_fc02,
            "fc02_original_rejection_sham_nlls":[.17]*old_fc02["selected"],
            "task_nll_by_arm":{
                "dp12":{"nll_nats":.11},
                "flat3":{"nll_nats":.14},
                "flat3_dose":{"nll_nats":.14},
                "flat4":{"nll_nats":.105},
            },
            "native_rotated_gauge":{
                k:{"physical_map_relative_error":0.,
                   "max_complete_rollout_logit_error":0.}
                for k in ("dp12","flat3","flat3_dose","flat4")
            },
        })
    return out


def test_frozen_positive_full_tangent_source_admission_and_baselines():
    result=aggregate(roots(baseline_fc03=10))
    assert result["first_source_locked_development_verdict"]=="FULL_TANGENT_ENGINEERING_POSITIVE"
    assert result["full_tangent_admitted_roots"]==16
    assert result["FC03_old_eightdim_solver_admitted_roots"]==10
    assert result["FC02_original_random_sampler_admitted_roots"]==8
    assert all(result["gates"].values())
    assert result["solver_sham_effect_descriptive_bootstrap"]["mean"]==pytest.approx(.06)
    assert result["protected_32_root_confirmation_evaluated"]==0
    assert result["protected_59_root_recalibration_evaluated"]==0


def test_original_registered_failures_and_intention_to_test():
    neutral=aggregate(roots(baseline_fc03=15))
    assert not neutral["gates"]["two_more_full_model_root_sham_admissions_than_FC03"]
    assert neutral["first_source_locked_development_verdict"]=="ENGINEERING_NULL_OR_INELIGIBLE"
    fail=aggregate(roots(baseline_fc03=10,blocked=0))
    assert fail["full_tangent_admitted_roots"]==15
    assert not fail["gates"]["sixteen_independent_full_source_and_eight_tangent_shams"]
    assert fail["solver_sham_effect_descriptive_bootstrap"] is None


def test_independent_projector_SHA_geometry_and_source_constraints():
    original=roots()[0]
    verified=independently_audit_full_tangent_projectors(original)
    assert verified["complete"] and verified["matched"]==8
    assert min(x["dp12_distance"] for x in verified["geometry"])>=.20
    bad=roots()[0]
    bad["sham_admission"]["projectors"][0]["projector_sha256"]="0"*64
    with pytest.raises(AssertionError):
        independently_audit_full_tangent_projectors(bad)


def test_exact_root_roster_and_full_root_bootstrap():
    with pytest.raises(ValueError):
        aggregate(roots()[:-1])
    duplicate=roots()
    duplicate[0]["root"]=duplicate[1]["root"]
    with pytest.raises(ValueError):
        aggregate(duplicate)
    b=root_bootstrap([.06]*16)
    assert b==root_bootstrap([.06]*16)
    assert b["seed"]==15041504 and b["draws"]==4096
