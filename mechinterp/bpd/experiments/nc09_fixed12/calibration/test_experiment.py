"""FC01 excluded-root and source arithmetic tests, no calibration root training."""
from __future__ import annotations

import hashlib
import math

import torch

from mechinterp.bpd.experiments.nc09_fixed12.calibration import evalmath_preflight
from mechinterp.bpd.experiments.nc09_fixed12.calibration.experiment import (
    ROOTS,SMOKE,PARENT_SOURCE_SHA,PROTO_SHA,
    protect_original_nc09_source,train_model,evaluate_calibration_root
)
from mechinterp.bpd.experiments.nc09_fixed12.calibration.aggregate import (
    CALIBRATION_CONFIDENCE,EVALMATH_BLOB_SHA,git_blob_sha,verify_pinned_evalmath
)
from mechinterp.bpd.experiments.nc09_fixed12.calibration.experiment import ROOT


def test_frozen_roster_and_parent_original_source_sha():
    assert len(ROOTS)==59 and ROOTS[0]==12000 and ROOTS[-1]==12058
    assert SMOKE==11999 and SMOKE not in ROOTS
    assert min(ROOTS)>1007
    out=protect_original_nc09_source()
    assert out["nc09_original_experiment_sha256"]==PARENT_SOURCE_SHA
    assert len(PROTO_SHA)==64
    assert 1-.95**59>=.95


def test_exact_evalmath_import_and_original_upstream_blob():
    assert verify_pinned_evalmath()==EVALMATH_BLOB_SHA
    script=(ROOT/"evalmath_preflight.py").read_bytes()
    assert git_blob_sha(script)==EVALMATH_BLOB_SHA
    x=[.01]*59
    from fractions import Fraction
    assert evalmath_preflight.upper_quantile(
        [Fraction(1,100)]*59
    )==(Fraction(1,100),59)
    assert abs(CALIBRATION_CONFIDENCE-(1-.95**59))<1e-15


def test_root_guard_and_original_train_contract_excluded_11999():
    model,meta=train_model(SMOKE,steps=2)
    assert meta["root"]==SMOKE and meta["steps"]==2
    assert meta["train_size"]==2048 and model.mix.weight.shape==(20,20)
    try:
        train_model(915,steps=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Historical NC09 model root reused")


def test_smoke_first_real_physical_recurrence_on_excluded_11999():
    doc=evaluate_calibration_root(
        SMOKE,steps=8,discovery_n=96,selector_n=64,heldout_n=64,
        sham_proposals=128,shams=2
    )
    assert doc["study"]=="BPD-NC09-FC01"
    assert doc["phase"]=="INDEPENDENT_SHAM_CALIBRATION"
    assert doc["root"]==SMOKE
    assert doc["full_mask_closure"]["passed"]
    assert doc["splits"]=={
        "train":2048,"derivative_discovery":96,"model_validation":64,
        "labeled_calibration":64
    }
    assert doc["parent_integrity"]["nc09_original_experiment_sha256"]==PARENT_SOURCE_SHA
    assert doc["training"]["steps"]==8
    assert doc["source_rank12_optimum"] is not None
    assert sum(doc["source_rank12_optimum"]["ranks"])==12
    assert set(doc["task_nll_by_arm"])=={
        "dp12","flat3","flat3_dose","flat4","reverse"
    }
    assert doc["sham_admission"]["selected"]<=2
    assert doc["sham_admission"]["proposals"]==128
    assert doc["source_sha256"]
