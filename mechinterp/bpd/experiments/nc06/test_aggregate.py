"""Synthetic root-level NC06 development gate tests, no registered roots evaluated."""
from __future__ import annotations

from mechinterp.bpd.experiments.nc06.aggregate import aggregate
from mechinterp.bpd.experiments.nc06.experiment import DEV_ROOTS, PROTOCOL_HASH


def receipts(*, cost: float=.02, uniform_gap:float=.02, shuffled_gap:float=.02,
             sham_gap:float=.04, blocked:int=0):
    out=[]
    for j,root in enumerate(DEV_ROOTS):
        base=.1
        def arm(nll, rank=3):
            return {"rank":rank,"status":"ELIGIBLE","scored":{"nll_nats":nll},
                    "median_sham_minus_kept_nll":sham_gap}
        methods={
            "norm_r3":arm(base+cost),
            "uniform_r3":arm(base+cost+uniform_gap),
            "shuffle_r3":arm(base+cost+shuffled_gap),
            "uniform_r4_reference":arm(base+.015,rank=4),
        }
        if j<blocked:methods["norm_r3"]["status"]="BLOCKED_SHAMS"
        out.append({
            "study":"BPD-NC-06","phase":"ENGINEERING",
            "protocol_sha256":PROTOCOL_HASH,
            "root":root,"eligible":not j<blocked,
            "clean_nll":base,"methods":methods
        })
    return out


def test_frozen_development_positive_synthetic_roots():
    r=aggregate(receipts())
    assert r["model_roots"]==8
    assert r["source_frozen"]
    assert r["frozen_development_positive"]
    assert r["registered_verdict"]=="ENGINEERING_POSITIVE"
    assert r["statistics"]["norm_keep_cost"]["upper95"]<=.05


def test_frozen_development_negative_and_missing_shams():
    assert not aggregate(receipts(cost=.1))["frozen_development_positive"]
    assert not aggregate(receipts(uniform_gap=0.))["frozen_development_positive"]
    assert not aggregate(receipts(shuffled_gap=0.))["frozen_development_positive"]
    assert not aggregate(receipts(sham_gap=0.))["frozen_development_positive"]
    blocked=aggregate(receipts(blocked=1))
    assert not blocked["frozen_development_positive"]
    assert blocked["registered_verdict"]=="ENGINEERING_NULL_OR_INELIGIBLE"
