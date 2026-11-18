"""NC-11 frozen source tests for independent model causal-Jacobian alignment."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import PASSES, WIDTH, sample_x, parity
from mechinterp.bpd.experiments.nc04.experiment import factor_basis
from mechinterp.bpd.experiments.nc03.experiment import basis_transport
from mechinterp.bpd.experiments.nc11.aggregate import aggregate, root_bootstrap, PRIMARY
import mechinterp.bpd.experiments.nc11.experiment as nc


def wrapped(steps=5,root=nc.SMOKE,task="B"):
    model,_=nc.train_pair_task(root,task,steps=steps)
    bpd=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                 ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bpd,"native",root if task=="A" else root+100000)
    return model,bpd


def test_new_registration_and_disjoint_root_population():
    assert nc.DEVELOPMENT == tuple(range(1100,1108))
    assert nc.SMOKE == 1099
    assert nc.PROTOCOL_COMMIT == "60b65642c03cb3064912aa3929582479aa21dd20"
    assert nc.TARGET_CALIBRATION == 128
    assert len(PRIMARY) == 4
    for old in (899,900,1000,1108):
        with pytest.raises(ValueError):
            nc.train_pair_task(old,"B",steps=1)
        with pytest.raises(ValueError):
            nc.run(old,steps=1,discovery_size=4,calibration_pool_size=4,
                   dose_size=4,evaluation_size=4,proposals=4)


def test_task_relation_and_training_stream_separation():
    x=torch.tensor([
        [2.,-3.,-4.,0.,1.],
        [-2.,3.,4.,0.,1.],
        [2.,-3.,4.,0.,-1.],
        [2.,3.,-4.,0.,-1.],
    ])
    assert nc.task_labels(x,"A").tolist() == [True,False,False,True]
    assert nc.task_labels(x,"B").tolist() == [True,False,True,False]
    a,ta=nc.train_pair_task(nc.SMOKE,"A",steps=4)
    b,tb=nc.train_pair_task(nc.SMOKE,"B",steps=4)
    assert ta["model_seed"] != tb["model_seed"]
    assert ta["train_split"]==1 and tb["train_split"]==11
    assert not torch.equal(a.mix.weight,b.mix.weight)
    assert not torch.equal(sample_x(nc.SMOKE,1,10),sample_x(nc.SMOKE,11,10))


def test_native_and_rotated_fullmask_parity():
    model,bpd=wrapped(4)
    x=sample_x(nc.SMOKE,6,32)
    assert parity(bpd,model,x)["passed"]
    factor_basis(bpd,"rotated",nc.SMOKE+100000)
    assert parity(bpd,model,x)["passed"]


def test_raw_jacobians_are_label_blind_and_have_expected_shape(monkeypatch):
    model,bpd=wrapped(4)
    def forbidden(*args,**kwargs):
        raise AssertionError("external labels consulted during model-only calibration")
    monkeypatch.setattr(nc,"task_labels",forbidden)
    a,g=nc.read_and_score_jacobian(bpd,model,sample_x(nc.SMOKE,4,64),batch_size=16)
    assert a.shape == g.shape == (64,PASSES,WIDTH)
    assert torch.isfinite(a).all() and torch.isfinite(g).all()
    assert float(g.abs().max())>0
    u,stats=nc.unit_jacobians(g)
    assert stats["zero_row_count"]==0
    assert torch.allclose(u.norm(dim=-1),torch.ones(64,PASSES,dtype=torch.float64),
                          atol=1e-6)


def test_zero_jacobian_rows_are_reported():
    g=torch.zeros(10,PASSES,WIDTH,dtype=torch.float64)
    normalized,diag=nc.unit_jacobians(g)
    assert diag["zero_row_count"]==40
    assert float(normalized.abs().max())==0


def test_procrustes_recovers_synthetic_causal_response_rotation():
    generator=torch.Generator().manual_seed(55)
    a=torch.randn(256,PASSES,WIDTH,generator=generator,dtype=torch.float64)
    q=[]
    for _ in range(PASSES):
        u,r=torch.linalg.qr(torch.randn(WIDTH,WIDTH,generator=generator,dtype=torch.float64))
        q.append(u*torch.where(torch.diag(r)>=0,1.,-1.).unsqueeze(0))
    q=torch.stack(q)
    b=torch.einsum("nti,tij->ntj",a,q)
    fitted,diag=nc.procrustes(a,b)
    assert max(diag["relative_alignment_residual_by_pass"])<1e-10
    assert (fitted-q).abs().max()<1e-10
    assert diag["max_orthogonality_error"]<1e-10


def test_transport_preserves_projector_rank_and_physical_orientation():
    generator=torch.Generator().manual_seed(88)
    a,_=torch.linalg.qr(torch.randn(PASSES,WIDTH,4,generator=generator,dtype=torch.float64))
    q,_=torch.linalg.qr(torch.randn(PASSES,WIDTH,WIDTH,generator=generator,dtype=torch.float64))
    p,e=nc.transfer_map(a,q)
    expected=q.transpose(-1,-2)@(a@a.transpose(-1,-2))@q
    assert (p-expected).abs().max()<1e-10
    assert (p@p-p).abs().max()<1e-10
    assert all(abs(v-4)<1e-10 for v in nc._projection_diagnostics(p)["rank_trace_by_pass"])


def test_retained_full_rank_recovers_target_when_gain_one():
    model,bpd=wrapped(4)
    x=sample_x(nc.SMOKE,6,24)
    p=torch.eye(WIDTH,dtype=torch.float64).expand(PASSES,-1,-1)
    with torch.no_grad():
        with bpd.target_capture():
            target,_=model(x)
        kept,_=nc.projected_with_gain(bpd,model,x,p,1.)
    assert (target-kept).abs().max()<1e-5


def test_exact_within_target_factor_gauge_transport():
    model,bpd=wrapped(4)
    x=sample_x(nc.SMOKE,6,32)
    rng=torch.Generator().manual_seed(10)
    e,_=torch.linalg.qr(torch.randn(PASSES,WIDTH,4,dtype=torch.float64,generator=rng))
    p=e@e.transpose(-1,-2)
    native,_=nc.projected_with_gain(bpd,model,x,p,1.)
    factor_basis(bpd,"rotated",nc.SMOKE+100000)
    rotated_p=basis_transport(p,bpd.wrappers["mix"].V.detach())
    changed,_=nc.projected_with_gain(bpd,model,x,rotated_p,1.)
    assert (native-changed).abs().max()<5e-5


def test_gain_matching_uses_discovery_only_bounds():
    rng=torch.Generator().manual_seed(123)
    acts=torch.randn(32,PASSES,WIDTH,generator=rng,dtype=torch.float64)
    u=torch.eye(WIDTH)
    p=torch.eye(WIDTH,dtype=torch.float64).expand(PASSES,-1,-1)
    from mechinterp.bpd.experiments.nc05.experiment import clean_dose_rms
    d=clean_dose_rms(acts,u,p)
    eligible=nc.fixed_gain(d,acts,u,p)
    assert eligible["eligible"]
    assert abs(eligible["applied_gain"]-1.)<1e-8
    blocked=nc.fixed_gain(5*d,acts,u,p)
    assert not blocked["eligible"]
    assert blocked["applied_gain"]==1


def test_local_rank4_shams_on_dose_only_source():
    from mechinterp.bpd.experiments.nc06.experiment import local_matched_shams
    rng=torch.Generator().manual_seed(1234)
    act=torch.randn(128,PASSES,WIDTH,generator=rng,dtype=torch.float64)
    q,_=torch.linalg.qr(torch.randn(PASSES,WIDTH,4,generator=rng,dtype=torch.float64))
    shams,report=local_matched_shams(act,torch.eye(WIDTH),q,
                                     seed=nc.SMOKE,arm_index=11,proposals=128,want=8)
    assert report["admitted"],report
    assert len(shams)==8
    for proj in shams:
        assert proj.shape==(PASSES,WIDTH,WIDTH)
        assert (proj@proj-proj).abs().max()<1e-8


def test_root_level_familywise_bootstrap_and_bad_roster():
    v=root_bootstrap([.01,.02,.03,.04,.05,.06,.07,.08],seed=91)
    assert v["n_independent_root_pairs"]==8
    assert math.isclose(v["family_coverage"],.9875,rel_tol=1e-10)
    assert v["familywise_9875pct_ci"][0]<=v["mean_nats"]<=v["familywise_9875pct_ci"][1]
    with pytest.raises(ValueError):
        aggregate([{"root":1100}]*8)


def test_excluded_smoke_completes_with_failure_receipt():
    out=nc.run(nc.SMOKE,steps=4,discovery_size=128,calibration_pool_size=512,
               dose_size=128,evaluation_size=128,proposals=64)
    assert out["phase"]=="ENGINEERING"
    assert out["root"]==nc.SMOKE
    assert out["protocol_commit"]==nc.PROTOCOL_COMMIT
    assert out["budgets"]["target_gradient_calibration"]==128
    assert isinstance(out["excluded_reasons"],list)
    assert out["source_hash_sha256"] and out["protocol_sha256"]
    json.dumps(out,allow_nan=False)
