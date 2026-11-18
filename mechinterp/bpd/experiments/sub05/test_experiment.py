"""SUB-05: excluded-source structural tests for pass-specific quotient bundles."""
import pytest
import torch

from mechinterp.bpd.experiments.sub01.experiment import WIDTH, Looper, sample_x
from mechinterp.bpd.experiments.sub04.experiment import (
    source_joint_projector, physical_delta,
)
from mechinterp.bpd.experiments.sub05.experiment import (
    ROOTS, SMOKE, RANK, OPT_STEPS, TIME_PERMUTATIONS, ANGLES,
    pass_output_covariances, bundle_loss, source_bundle,
    stationary_bundle, permuted_bundle, bundle_shams,
    causal_response, clocked_student, gauge_checks, preflight_spec,
    aggregate, train_source,
)


def test_exact_time_indexed_linear_cocycle_with_bad_time_permutation():
    """A synthetic linear 4+4+4 cyclic permutation has exact moving fibers."""
    eye=torch.eye(WIDTH,dtype=torch.float64)
    w=eye.roll(4,dims=0)
    basis=torch.cat([eye[:,:8]],dim=1)
    ps=[]
    for t in range(4):
        q=torch.linalg.matrix_power(w,t)@basis
        ps.append(q@q.T)
    for t in range(3):
        bridge=ps[t+1]@w@(eye-ps[t])
        assert float(bridge.norm())<1e-12
    stationary=ps[0]@w@(eye-ps[0])
    assert float(stationary.norm())>1.
    bad=TIME_PERMUTATIONS[3]
    reordered=[ps[t] for t in bad]
    assert max(float((reordered[t+1]@w@(eye-reordered[t])).norm())
               for t in range(3))>1.


def test_bundle_source_qr_objective_and_all_pass_gradients():
    torch.manual_seed(123)
    model=Looper().eval()
    x=sample_x(SMOKE,2,48)
    static,basis,_,s=source_joint_projector(model,x)
    p,q,cov,info=source_bundle(model,x,basis)
    assert len(info["trace"])==OPT_STEPS
    assert info["final_objective"]<info["initial_objective"]
    assert info["projector_idempotence_max"]<1e-5
    assert info["projector_symmetry_max"]<1e-5
    assert info["projector_trace_max_error"]<1e-4
    assert all(v>0 for v in info["grad_norm_per_pass"])
    assert torch.allclose(q.transpose(-1,-2)@q,
       torch.eye(RANK,dtype=torch.float64),atol=1e-8)
    jac,gn=pass_output_covariances(model,x)
    assert jac.shape==(4,12,12)
    assert len(gn)==4


def test_temporal_sham_roster_angle_controls_and_dose():
    torch.manual_seed(42)
    model=Looper().eval()
    x=sample_x(SMOKE,2,48)
    p,basis,_,_=source_joint_projector(model,x)
    bundle,q,cov,_=source_bundle(model,x,basis)
    shams=bundle_shams(SMOKE,q,cov)
    assert len(TIME_PERMUTATIONS)==7
    assert len(set(TIME_PERMUTATIONS))==7
    assert len(shams)==8
    assert [s["theta"] for s in shams[:4]]==list(ANGLES)
    for a in shams:
        for key in ("orthogonality_max","rank_trace_max_error",
                    "idempotence_max","symmetry_max"):
            assert a["geometry"][key]<1e-5
    for permutation in TIME_PERMUTATIONS:
        shuffled=permuted_bundle(bundle,permutation)
        for t in range(4):
            assert torch.equal(shuffled[t],bundle[permutation[t]])
    with pytest.raises(ValueError,match="Unregistered"):
        permuted_bundle(bundle,(0,1,2,3))
    d=physical_delta(SMOKE,4,1,32,bundle[1],.3)
    assert float((d@bundle[1]).norm(dim=-1).max())<1e-5
    assert torch.allclose(d.norm(dim=-1),torch.full((32,),.3),atol=1e-6)


def test_exact_full_rank_parity_and_all_pass_gauge():
    torch.manual_seed(66)
    model=Looper().eval()
    x=sample_x(SMOKE,4,48)
    full=stationary_bundle(torch.eye(WIDTH))
    parity=clocked_student(model,x,full)
    assert parity["mean_clean_bernoulli_kl"]<1e-8
    p=stationary_bundle(
        torch.diag(torch.tensor([1.]*RANK+[0.]*(WIDTH-RANK))))
    gauge=gauge_checks(model,x,p,.35,SMOKE)
    assert gauge["clean_logit_error"]<5e-6
    assert gauge["transported_next_state_error"]<5e-6
    assert gauge["transported_logit_error"]<5e-6
    assert max(gauge["wrong_untransported_logit_rms_by_pass"])>1e-5
    r=causal_response(model,x,p,.35,SMOKE,4)
    assert len(r["per_pass_next_state_rms"])==3
    assert r["max_physical_dose_error"]<5e-6


def test_roster_independent_unit_and_sealed_future():
    assert ROOTS==tuple(range(1600,1608))
    assert SMOKE not in ROOTS
    with pytest.raises(ValueError,match="outside frozen"):
        train_source(1700,steps=1)
    with pytest.raises(ValueError,match="eight"):
        aggregate([])
    fake={"root":1600,"primary_strongest_null_minus_aligned":.023,
          "scores":{
            "stationary":{"pooled_next_state_rms":.3},
            **{"time_perm_"+str(i):{"pooled_next_state_rms":.28}
               for i in range(7)}
          }}
    spec=preflight_spec([fake])
    assert spec["confirm_units_planned"]==64
    assert spec["pilot"][0]["unit_id"]=="root-1600"
    assert len(spec["pilot"][0]["sham_effects"])==7
    assert spec["scientific_min_effect"]=="0.014778325123152709"
