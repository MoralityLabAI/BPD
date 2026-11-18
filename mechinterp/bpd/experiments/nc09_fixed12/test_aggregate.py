"""NC09 fixed twelve-slot development decision tests with synthetic root records.

The 16 registered model roots are IDs only; no model is trained in this test.
"""
from __future__ import annotations

import copy

import pytest

from mechinterp.bpd.experiments.nc08.experiment import candidate_budget_knapsack
from mechinterp.bpd.experiments.nc09_fixed12.experiment import PROTO_SHA,ROOTS,OPTIONS
from mechinterp.bpd.experiments.nc09_fixed12.aggregate import aggregate,rooted_bootstrap


def source_candidates():
    weights=(10.,6.,4.,1.)
    return {
        str(t):{
            str(rank):{
                "eligible":True,
                "squared_local_omitted_model_score":
                     weights[t]*(6-rank)**2
            } for rank in OPTIONS
        } for t in range(4)
    }


def receipts(*,flat3_margin=.015,dose_margin=.012,
             cost=.02,reference_cost=.005,sham_margin=.05,
             blocked_root=None):
    c=source_candidates()
    by={int(t):{int(r):rec for r,rec in ranks.items()}
        for t,ranks in c.items()}
    best=candidate_budget_knapsack(by)[12]
    assert best is not None
    result=[]
    for i,root in enumerate(ROOTS):
        budget=list(best["ranks"])
        dp=.1+cost
        methods={
            "dp12":{"total_rank":12,"source_dose_rms":1.0},
            "flat3":{"total_rank":12,"source_dose_rms":1.0},
            "flat3_dose":{"total_rank":12,"source_dose_rms":1.0},
            "flat4":{"total_rank":16,"source_dose_rms":1.0},
            "reverse":{"total_rank":12,"source_dose_rms":1.0},
        }
        rows={
            "study":"BPD-NC-09-fixed12","phase":"DEVELOPMENT","root":root,
            "protocol_sha256":PROTO_SHA,
            "eligible":i!=blocked_root,
            "passwise_source_candidates":copy.deepcopy(c),
            "fixed_budget12_allocation":{
                "ranks":budget,"total_rank":12,
                "source_proxy_loss":best["proxy_loss"],
            },
            "arms":methods,
            "dp12_source_write_rms":1.0,
            "flat3_source_write_rms":1.0,
            "dose_matched_flat3_gain":1.0,
            "dose_matched_flat3_gain_eligible":True,
            "sham_admission":{
                "qualified":32,"selected":8 if i!=blocked_root else 7,
                "proposals":4096
            },
            "gauge":{
                a:{"physical_map_relative_error":0.,
                   "max_complete_rollout_logit_error":0.}
                for a in ("dp12","flat3","flat3_dose","flat4")
            },
            "clean_closure":{"passed":True},
            "clean_final_nll_nats":.1,
            "heldout":{
                "dp12":{"nll_nats":dp},
                "flat3":{"nll_nats":dp+flat3_margin},
                "flat3_dose":{"nll_nats":dp+dose_margin},
                "flat4":{"nll_nats":dp-reference_cost},
                "reverse":{"nll_nats":dp+.009},
            },
            "median_sham_minus_dp12_nll":sham_margin
        }
        result.append(rows)
    return result


def test_source_registered_positive_pairwise_development():
    s=aggregate(receipts())
    assert s["registered_verdict"]=="DEVELOPMENT_POSITIVE"
    assert s["admitted_roots"]==16 and s["independent_model_roots"]==16
    assert s["model_budget"]==12 and s["physical_reference_budget"]==16
    assert s["budget_saving_percent"]==25.
    assert s["nonflat_allocations"]==16
    assert all(s["gates"].values())
    assert s["bootstrap"]["draws"]==4096
    assert s["bootstrap"]["metrics"]["doseflat3_minus_dp12"]["lower95"]>0
    assert not s["confirmed_scientific_result"]
    assert s["calibration_roots_evaluated"]==0
    assert s["confirmation_roots_evaluated"]==0


def test_frozen_distinct_rejection_modes():
    assert not aggregate(receipts(flat3_margin=.001))[
        "gates"]["fixed12_beats_equal12_flat3"]
    assert not aggregate(receipts(dose_margin=.001))[
        "gates"]["fixed12_beats_dose_matched_flat3"]
    assert not aggregate(receipts(cost=.08))[
        "gates"]["absolute_task_fidelity"]
    assert not aggregate(receipts(reference_cost=.03))[
        "gates"]["near_rank4_task_fidelity"]
    assert not aggregate(receipts(sham_margin=0.))[
        "gates"]["source_dose_matched_sham_specificity"]
    blocked=aggregate(receipts(blocked_root=7))
    assert blocked["registered_verdict"]=="DEVELOPMENT_NULL_OR_INELIGIBLE"
    assert blocked["bootstrap"] is None
    assert not any(blocked["gates"].values())


def test_source_dp_source_only_and_unique_root_admission():
    wrong=receipts()
    wrong[0]["fixed_budget12_allocation"]["ranks"]=[3,3,3,3]
    with pytest.raises(AssertionError):
        aggregate(wrong)
    with pytest.raises(ValueError):
        aggregate(receipts()[:-1])
    dup=receipts()
    dup[0]["root"]=dup[1]["root"]
    with pytest.raises(ValueError):
        aggregate(dup)


def test_independent_root_bootstrap():
    a=[(.015,.012,.02,.005,.05)]*16
    x=rooted_bootstrap(a)
    assert x==rooted_bootstrap(a)
    assert x["draws"]==4096 and x["seed"]==909009
    assert abs(x["metrics"]["flat3_minus_dp12"]["mean"]-.015)<1e-12
