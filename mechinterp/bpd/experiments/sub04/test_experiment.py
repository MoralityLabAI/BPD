"""SUB-04 structural source tests, entirely outside the eight-root cohort."""
import math

import pytest
import torch

from mechinterp.bpd.experiments.sub01.experiment import WIDTH, Looper, sample_x
from mechinterp.bpd.experiments.sub04.experiment import (
    DEV_ROOTS, SMOKE, RANK, OPT_STEPS,
    source_joint_projector, joint_rotated_shams, physical_delta,
    score_transition, transported_gauge_passes, linearized_source,
    linearized_loss, preflight_spec, aggregate, train_source,
)


def test_projected_step_commutator_zero_when_fiber_is_invariant():
    m=Looper().eval()
    with torch.no_grad():
        m.hidden.weight.zero_()
        m.inp.weight.zero_()
        m.inp.bias.zero_()
        m.head.weight.zero_()
        m.head.bias.zero_()
        m.head.weight[0,0]=1.
        m.hidden.weight[0,0]=.8
    p=torch.diag(torch.tensor([1.]*RANK+[0.]*(WIDTH-RANK)))
    x=torch.zeros(48,4)
    result=score_transition(m,x,p,.30,SMOKE,4)
    assert result["pooled_projected_next_state_euclidean_rms"]<1e-8
    assert result["max_physical_dose_error"]<1e-6
    assert result["max_null_projection_norm"]<1e-6
    with torch.no_grad():
        m.hidden.weight[0,8]=1.0
    changed=score_transition(m,x,p,.30,SMOKE,4)
    assert changed["pooled_projected_next_state_euclidean_rms"]>.001


def test_joint_source_fit_and_exact_rank_nonlinear_geometry():
    torch.manual_seed(22)
    m=Looper().eval()
    x=sample_x(SMOKE,2,48)
    p,basis,c_out,info=source_joint_projector(m,x)
    assert len(info["source_loss_trace"])==OPT_STEPS
    assert info["projector_idempotence_max"]<1e-6
    assert info["projector_symmetry_max"]<1e-6
    assert info["projector_trace_error"]<1e-5
    assert math.isfinite(info["final_objective"])
    assert info["initial_objective"]==pytest.approx(1.5,abs=1e-4)
    assert info["final_objective"] < info["initial_objective"]
    assert torch.allclose(basis.T@basis,
                          torch.eye(RANK,dtype=torch.float64),atol=1e-8)
    assert torch.allclose(p.double(),basis@basis.T,atol=1e-6)
    assert torch.isfinite(linearized_source(m,x)).all()


def test_joint_local_and_global_shams_share_rank_and_artificial_spectrum():
    torch.manual_seed(21)
    m=Looper().eval()
    p,basis,c_out,_=source_joint_projector(m,sample_x(SMOKE,2,48))
    shams=joint_rotated_shams(SMOKE,p,basis,c_out)
    assert len(shams)==8
    assert [z["theta"] for z in shams[:4]]==[.20,.40,.60,.80]
    assert all(z["theta"] is None for z in shams[4:])
    for s in shams:
        g=s["geometry"]
        assert g["orthogonality_max"]<1e-9
        assert g["projector_idempotence_max"]<1e-9
        assert g["projector_trace_error"]<1e-6
        assert g["artificial_spectrum_difference_max"]<1e-9
        assert float(torch.trace(s["projector"]))==pytest.approx(
            RANK,abs=1e-4)


def test_fixed_dose_same_raw_directions_and_gauge_at_all_passes():
    torch.manual_seed(23)
    m=Looper().eval()
    x=sample_x(SMOKE,4,40)
    p=torch.diag(torch.tensor([1.]*RANK+[0.]*(WIDTH-RANK)))
    d=physical_delta(SMOKE,4,0,len(x),p,.35)
    assert torch.allclose(d.norm(dim=-1),torch.full((len(x),),.35),
                          atol=1e-6)
    assert float((d@p).norm(dim=-1).max())<1e-6
    assert torch.equal(d,physical_delta(SMOKE,4,0,len(x),p,.35))
    assert not torch.equal(d,physical_delta(SMOKE,4,1,len(x),p,.35))
    gauge=transported_gauge_passes(m,x,p,.35,SMOKE)
    assert gauge["clean_logit_max_error"]<5e-6
    assert gauge["intervention_logit_max_error"]<5e-6
    assert gauge["clean_state_max_error"]<5e-6
    assert gauge["intervention_state_max_error"]<5e-6
    assert max(gauge["wrong_untransported_logit_rms_by_pass"])>1e-5


def test_roster_preflight_and_no_confirmation_function():
    assert DEV_ROOTS==tuple(range(1500,1508))
    assert SMOKE not in DEV_ROOTS
    with pytest.raises(ValueError,match="outside"):
        train_source(1550,steps=1)
    with pytest.raises(ValueError,match="eight"):
        aggregate([])
    r={"root":1500,"primary_obs_minus_joint_state_rms":.03,
       "methods":{"obs":{"transition":{
           "pooled_projected_next_state_euclidean_rms":.12}}},
       "shams":[{"transition":{
           "pooled_projected_next_state_euclidean_rms":.10}}]*8}
    spec=preflight_spec([r])
    assert spec["pilot"][0]["unit_id"]=="root-1500"
    assert len(spec["pilot"][0]["sham_effects"])==8
    assert spec["confirm_units_planned"]==64
    assert spec["scientific_min_effect"]=="0.0196078431372549"
