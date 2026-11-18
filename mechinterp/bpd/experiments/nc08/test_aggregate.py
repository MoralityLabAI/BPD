"""Synthetic NC08 engineering gate tests; no development root data are opened."""
from __future__ import annotations

import pytest

from mechinterp.bpd.experiments.nc08.aggregate import aggregate,root_bootstrap
from mechinterp.bpd.experiments.nc08.experiment import ROOTS, PROTO_HASH, BUDGETS


def fake_roots(*, budgets=None,keep_cost=.02,sham_adv=.04,
               extra_cost_flat4=.006,blocked_root=None):
    budgets=budgets or [10,10,12,12,12,12,14,14]
    assert len(budgets)==8
    rows=[]
    for i,seed in enumerate(ROOTS):
        b=budgets[i]
        tup={
            8:(2,2,2,2),10:(2,2,3,3),12:(3,3,3,3),14:(3,3,4,4)
        }[b]
        c={str(n):{"eligible":n==b,"budget":n} for n in BUDGETS}
        r={
            "study":"BPD-NC-08","phase":"DEVELOPMENT",
            "root":seed,"protocol_sha256":PROTO_HASH,
            "eligible":i!=blocked_root,
            "selected_total_rank":b,"selected_rank_tuple":tup,
            "candidates_by_budget":c,
            "clean_final_nll_nats":.10,
            "sham_admission":{"selected":8 if i!=blocked_root else 7},
            "median_sham_minus_selected_nll_nats":sham_adv,
            "heldout":{
                "selected":{"nll_nats":.10+keep_cost},
                "flat_rank3":{"nll_nats":.10+keep_cost+.01},
                "flat_rank4":{"nll_nats":.10+keep_cost-extra_cost_flat4},
                "dp_budget12":{"nll_nats":.10+keep_cost+.001},
            }
        }
        rows.append(r)
    return rows


def test_source_registered_positive_rank_budget():
    got=aggregate(fake_roots())
    assert got["original_development_verdict"]=="ENGINEERING_POSITIVE"
    assert got["independent_model_roots"]==8
    assert got["admitted_roots"]==8
    assert got["mean_selected_budget"]==12
    assert got["selected_budget_le12_roots"]==6
    assert got["bootstrap"]["keep_cost"]["upper95"]<=.05
    assert got["bootstrap"]["sham_margin"]["lower95"]>0
    assert all(got["gates"].values())
    assert not got["scientific_confirmation_performed"]
    assert got["independent_sham_calibration_roots_used"]==0
    assert got["independent_confirmation_roots_used"]==0


def test_frozen_negative_rank_fidelity_and_source_controls():
    assert not aggregate(fake_roots(budgets=[14]*8))["gates"]["strict_total_rank_reduction"]
    assert not aggregate(fake_roots(keep_cost=.07))["gates"]["absolute_external_task_fidelity"]
    assert not aggregate(fake_roots(sham_adv=0.))["gates"]["source_matched_sham_specificity"]
    assert not aggregate(fake_roots(extra_cost_flat4=.04))["gates"]["bounded_extra_task_cost_vs_flat_rank4"]
    blocked=aggregate(fake_roots(blocked_root=2))
    assert blocked["original_development_verdict"]=="ENGINEERING_NULL_OR_INELIGIBLE"
    assert blocked["bootstrap"] is None and blocked["mean_selected_budget"] is None
    assert not any(blocked["gates"].values())


def test_whole_root_bootstrap_determinism():
    v=[(.02,.04,.006)]*8
    got=root_bootstrap(v)
    assert got==root_bootstrap(v)
    assert got["repetitions"]==2000
    assert abs(got["keep_cost"]["upper95"]-.02)<1e-12


def test_require_unique_frozen_roots_and_smallest_budget():
    with pytest.raises(ValueError):
        aggregate(fake_roots()[:-1])
    bad=fake_roots()
    bad[3]["root"]=800
    with pytest.raises(ValueError):
        aggregate(bad)
    changed=fake_roots()
    changed[0]["selected_total_rank"]=12
    with pytest.raises(AssertionError):
        aggregate(changed)
