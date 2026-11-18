"""SUB-06 source-only checks; excluded source root, no development data."""
import pytest
import torch

from mechinterp.bpd.experiments.sub01.experiment import (
    WIDTH, Looper, sample_x,
)
from mechinterp.bpd.experiments.sub04.experiment import source_joint_projector
from mechinterp.bpd.experiments.sub05.experiment import (
    source_bundle, bundle_loss,
)
from mechinterp.bpd.experiments.sub06.experiment import (
    SMOKE, ROOTS, RANK, CLOCK_MAPS, STEP_MAPS, FIT_STEPS,
    source_arrays, schedule_arrays, source_fit_family, causal_response,
    stationary_bundle, fit_source_schedule, train_model,
    preflight_spec, aggregate, bootstrap,
)


def test_seven_schedules_preserve_source_multisets_and_are_fixed():
    torch.manual_seed(44)
    m=Looper().eval()
    x=sample_x(SMOKE,2,32)
    d,c,_=source_arrays(m,x)
    assert d.shape==(len(x),3,WIDTH,WIDTH)
    assert c.shape==(4,WIDTH,WIDTH)
    assert len(CLOCK_MAPS)==len(STEP_MAPS)==7
    assert (0,2,1,3) in CLOCK_MAPS
    for arm in range(7):
        ds,cs,tau,pi=schedule_arrays(d,c,arm)
        assert sorted(tau)==[0,1,2]
        assert sorted(pi)==[0,1,2,3]
        for t in range(3):
            assert torch.equal(ds[:,t],d[:,tau[t]])
        for t in range(4):
            assert torch.equal(cs[t],c[pi[t]])
        # Identical sets of source Jacobians and output covariances.
        assert torch.allclose(ds.square().sum(),d.square().sum(),rtol=1e-12,atol=1e-12)
        assert torch.allclose(cs.square().sum(),c.square().sum(),rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError,match="Unregistered"):
        schedule_arrays(d,c,7)


def test_frozen_aligned_source_fit_matches_sub05_exactly():
    torch.manual_seed(102)
    m=Looper().eval()
    x=sample_x(SMOKE,2,32)
    static,basis,_,_=source_joint_projector(m,x)
    p_original,q_original,c_original,diag=source_bundle(m,x,basis)
    all_p,all_q,all_diag,c,info=source_fit_family(m,x,basis)
    assert len(all_p)==len(all_q)==len(all_diag)==8
    assert torch.allclose(all_p["aligned"],p_original,atol=1e-7,rtol=0)
    assert torch.allclose(all_q["aligned"],q_original,atol=1e-7,rtol=0)
    assert torch.allclose(c,c_original,atol=1e-12,rtol=0)
    assert all_diag["aligned"]["optimizer_steps"]==FIT_STEPS
    assert all_diag["aligned"]["final_objective"]==pytest.approx(
        diag["final_objective"],abs=1e-7)
    for arm in range(7):
        row=all_diag[f"wrong_{arm}"]
        assert row["optimizer_steps"]==120
        assert len(row["source_loss_trace"])==120
        assert row["rank_trace_max_error"]<1e-5
        assert row["symmetry_max"]<1e-5
        assert row["idempotence_max"]<1e-5
        assert torch.isfinite(all_p[f"wrong_{arm}"]).all()
    assert max(float((all_p["aligned"]-all_p[f"wrong_{i}"]).norm())
               for i in range(7))>.001
    assert info["source_discovery_size"]==len(x)


def test_same_sham_family_score_in_true_clock_and_dose():
    torch.manual_seed(7)
    m=Looper().eval()
    x=sample_x(SMOKE,4,44)
    p=torch.diag(torch.tensor([1.]*RANK+[0.]*(WIDTH-RANK)))
    bundle=stationary_bundle(p)
    z=causal_response(m,x,bundle,.35,SMOKE,4)
    assert len(z["per_pass_next_state_rms"])==3
    assert z["max_physical_dose_error"]<5e-6
    assert z["max_null_impulse_projection"]<5e-6
    assert z["physical_alpha"]==pytest.approx(.35)


def test_exact_clocked_linear_cocycle_is_synthetic():
    w=torch.eye(WIDTH,dtype=torch.float64).roll(4,dims=0)
    initial=torch.eye(WIDTH,dtype=torch.float64)[:,:8]
    ps=[]
    for t in range(4):
        e=torch.linalg.matrix_power(w,t)@initial
        ps.append(e@e.T)
    assert max(float((ps[t+1]@w@(torch.eye(WIDTH,dtype=torch.float64)-ps[t]))
                     .norm()) for t in range(3)) <1e-10


def test_sealed_roster_and_root_only_preflight():
    assert ROOTS==tuple(range(1700,1708))
    assert SMOKE not in ROOTS
    with pytest.raises(ValueError,match="outside frozen"):
        train_model(1900,steps=1)
    with pytest.raises(ValueError,match="8 registered"):
        aggregate([])
    fake={
        "root":1700,
        "primary_strongest_null_minus_aligned":.015,
        "scores":{
            "stationary":{"pooled_next_state_rms":.1},
            **{f"wrong_{i}":{"pooled_next_state_rms":.08} for i in range(7)},
        },
    }
    spec=preflight_spec([fake])
    assert spec["confirm_units_planned"]==64
    assert spec["scientific_min_effect"]=="0.009900990099009901"
    assert len(spec["pilot"])==1
    assert len(spec["pilot"][0]["sham_effects"])==7
    assert spec["pilot"][0]["unit_id"]=="root-1700"
    assert len(bootstrap([.01]*8))==2
