"""Pure synthetic FC03 16-root engineering verdict, no development roots sampled."""
from __future__ import annotations

from copy import deepcopy

import pytest

from mechinterp.bpd.experiments.nc09_fixed12.grassmann_fc03.aggregate import (
    aggregate,independent_source_audit,root_bootstrap,ROOTS
)
from mechinterp.bpd.experiments.nc09_fixed12.grassmann_fc03.experiment import (
    PROTO_SHA,PARENT_SOURCE_SHA
)
from mechinterp.bpd.experiments.nc09_fixed12.grassmann_fc03.solver import (
    fit_constrained_grassmann,projectors_from_bases,projector_digest
)
from mechinterp.bpd.experiments.nc09_fixed12.grassmann_fc03.test_solver import (
    clean_synthetic_source
)


def manufactured_roots(*,baseline_pass=0,blocked=None):
    a,u,z,g,E=clean_synthetic_source(n=192)
    vectors,adm=fit_constrained_grassmann(
        a,u,z,g,E,seed=13999,starts=32,steps=4,want=8
    )
    assert len(vectors)==8
    full=projector_digest(projectors_from_bases(E))
    frozen=deepcopy(adm)
    frozen["max_adam_steps_per_chart"]=64
    data=[]
    for i,seed in enumerate(ROOTS):
        baseline=bool(i<baseline_pass)
        blocked_here=(i==blocked)
        record={
            "study":"BPD-NC09-FC03",
            "phase":"CONSTRAINED_GRASSMANN_ENGINEERING_DEVELOPMENT",
            "root":seed,"protocol_sha256":PROTO_SHA,
            "parent_integrity":{"nc09_original_experiment_sha256":PARENT_SOURCE_SHA},
            "eligible":not blocked_here,
            "full_mask_closure":{"passed":True},
            "source_gain_eligible":True,
            "source_rank12_optimum":{"ranks":[4,3,3,2],"total_rank":12},
            "source_dp12_projector_receipt":deepcopy(full),
            "task_nll_by_arm":{
                "dp12":{"nll_nats":.11},
                "flat3":{"nll_nats":.14},
                "flat3_dose":{"nll_nats":.14},
                "flat4":{"nll_nats":.10},
            },
            "sham_admission":deepcopy(frozen),
            "sham_nlls":[.16]*8,
            "fc02_original_rejection_admission":{
                "proposals":4096,
                "qualified":20 if baseline else 5,
                "selected":8 if baseline else 5,
            },
            "fc02_original_rejection_sham_nlls":[.16]*(8 if baseline else 5),
        }
        if blocked_here:
            record["eligible"]=False
            record["sham_admission"]["admitted"]=False
            record["sham_admission"]["selected"]=7
            record["sham_admission"]["accepted"]=record["sham_admission"]["accepted"][:7]
            record["sham_nlls"]=record["sham_nlls"][:7]
        data.append(record)
    return data


def test_source_exact_geometry_audit_and_sixteen_root_positive_gate():
    docs=manufactured_roots(baseline_pass=9)
    rec=independent_source_audit(docs[0])
    assert rec["source_eligible"]
    assert len(rec["geometry"])==8
    assert rec["fc02_full_original_admission"]
    out=aggregate(docs)
    assert out["source_roots"]==16
    assert out["solver_fully_admitted_roots"]==16
    assert out["FC02_frozen_sampler_fully_admitted_roots"]==9
    assert out["first_original_development_verdict"]=="CONSTRAINED_GRASSMANN_ENGINEERING_POSITIVE"
    assert all(out["gates"].values())
    assert out["solver_sham_mean_effect_descriptive_bootstrap"]["mean"]>.04
    assert out["scientific_confirmation_roots_evaluated"]==0
    assert out["new_independent_59_root_recalibration_evaluated"]==0


def test_original_rejection_baseline_and_invalid_root_fail_closed():
    baseline=aggregate(manufactured_roots(baseline_pass=15))
    assert not baseline["gates"]["two_more_root_admissions_than_original_fc02_random"]
    assert baseline["first_original_development_verdict"]=="ENGINEERING_NULL_OR_INELIGIBLE"
    ineligible=aggregate(manufactured_roots(baseline_pass=9,blocked=3))
    assert ineligible["solver_fully_admitted_roots"]==15
    assert not ineligible["gates"]["sixteen_full_native_models_and_eight_constrained_shams"]
    assert ineligible["solver_sham_mean_effect_descriptive_bootstrap"] is None


def test_projector_hash_protection_and_unique_roots():
    bad=manufactured_roots(baseline_pass=8)
    bad[0]["sham_admission"]["accepted"][0]["projector_sha256"]="0"*64
    with pytest.raises(AssertionError):
        aggregate(bad)
    docs=manufactured_roots()
    docs[0]["root"]=docs[1]["root"]
    with pytest.raises(ValueError):
        aggregate(docs)
    with pytest.raises(ValueError):
        aggregate(manufactured_roots()[:-1])


def test_deterministic_whole_root_bootstrap():
    x=root_bootstrap([.05]*16)
    assert x==root_bootstrap([.05]*16)
    assert x["draws"]==4096 and x["seed"]==14031403
    assert abs(x["lower95"]-.05)<1e-12
