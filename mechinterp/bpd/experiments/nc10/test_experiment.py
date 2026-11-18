"""NC-10 tests: cross-model alignment, oracle separation and native BPD parity."""
from __future__ import annotations

import json

import pytest
import torch

import mechinterp.bpd.experiments.nc10.experiment as nc
from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import PASSES, WIDTH, sample_x, parity
from mechinterp.bpd.experiments.nc04.experiment import factor_basis
from mechinterp.bpd.experiments.nc05.experiment import projectors_at_ranks, projected_rollout
from mechinterp.bpd.experiments.nc10.aggregate import aggregate, bootstrap, COMPARISON_NAMES


def test_two_task_labels_share_only_positive_context_branch():
    x = torch.tensor([
        [2.,-3.,-4.,0.,+1.],
        [-2.,+3.,+4.,0.,+1.],
        [2.,-3.,+4.,0.,-1.],
        [2.,+3.,-4.,0.,-1.],
    ])
    a=nc.task_labels(x,"A")
    b=nc.task_labels(x,"B")
    assert a.tolist()==[True,False,False,True]
    assert b.tolist()==[True,False,True,False]


def test_procrustes_exact_orthogonal_alignment_and_projector_transport():
    g=torch.Generator().manual_seed(90200)
    a=torch.randn(256,PASSES,WIDTH,generator=g,dtype=torch.float64)
    true=[]
    for t in range(PASSES):
        q,r=torch.linalg.qr(torch.randn(WIDTH,WIDTH,generator=g,dtype=torch.float64))
        true.append(q * torch.where(torch.diag(r)>=0,1.,-1.).unsqueeze(0))
    q=torch.stack(true)
    b=torch.einsum("nti,tij->ntj",a,q)
    recovered,diag=nc.procrustes(a,b)
    assert diag["max_orthogonality_error"]<1e-10
    assert max(diag["relative_alignment_residual_by_pass"])<1e-10
    assert (recovered-q).abs().max()<1e-9
    basis,_=torch.linalg.qr(torch.randn(PASSES,WIDTH,nc.RANK,generator=g,dtype=torch.float64))
    p,e=nc.transport(basis,recovered)
    expected=q.transpose(-1,-2)@(basis@basis.transpose(-1,-2))@q
    assert torch.allclose(p,expected,atol=1e-10)
    assert torch.allclose(p@p,p,atol=1e-10)
    assert torch.allclose(torch.diagonal(p,dim1=-2,dim2=-1).sum(-1),
                          torch.full((PASSES,),float(nc.RANK),dtype=torch.float64),atol=1e-10)


def test_a_b_training_seed_and_examples_separated():
    model_a,a=nc.train_task(nc.SMOKE,"A",steps=4)
    model_b,b=nc.train_task(nc.SMOKE,"B",steps=4)
    assert a["model_seed"]!=b["model_seed"]
    assert a["training_split"]==1 and b["training_split"]==11
    assert not torch.equal(sample_x(nc.SMOKE,1,16),sample_x(nc.SMOKE,11,16))
    assert not torch.equal(model_a.mix.weight,model_b.mix.weight)
    assert a["training_steps"]==b["training_steps"]==4


def test_independently_trained_models_keep_native_fullmask_parity():
    for task in ("A","B"):
        model,_=nc.train_task(nc.SMOKE,task,steps=5)
        bp=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                    ablation=AblationConfig(mode="zero",sphere_rescale=False))
        factor_basis(bp,"native",nc.SMOKE)
        assert parity(bp,model,sample_x(nc.SMOKE,3,24))["passed"]


def test_discovery_and_positive_alignment_never_call_labels(monkeypatch):
    model,_=nc.train_task(nc.SMOKE,"A",steps=3)
    bp=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bp,"native",nc.SMOKE)
    def forbidden(*args,**kwargs):
        raise AssertionError("task label leaked to discovery")
    monkeypatch.setattr(nc,"task_labels",forbidden)
    p,b,diagnostics,acts=nc.discover(
        bp,model,sample_x(nc.SMOKE,2,64),nc.SMOKE,"A"
    )
    assert acts.shape==(64,PASSES,WIDTH)
    assert diagnostics["source_info"]["independent_discovery_examples"]==64
    source=nc.prewrite_states(bp,model,sample_x(nc.SMOKE,4,64))
    r,info=nc.procrustes(source,source.clone())
    assert info["max_orthogonality_error"]<1e-9
    assert max(info["relative_alignment_residual_by_pass"])<1e-9


def test_native_bpd_intervention_is_gauge_equivariant_with_exact_transport():
    model,_=nc.train_task(nc.SMOKE,"B",steps=4)
    bp=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                ablation=AblationConfig(mode="zero",sphere_rescale=False))
    x=sample_x(nc.SMOKE,3,32)
    p=torch.eye(WIDTH,dtype=torch.float64)[:,:nc.RANK]
    p=p@p.T
    p=p.expand(PASSES,-1,-1)
    factor_basis(bp,"native",nc.SMOKE+100000)
    native,_=projected_rollout(bp,model,x,p,1.0)
    factor_basis(bp,"rotated",nc.SMOKE+100000)
    transported=nc.basis_transport(p,bp.wrappers["mix"].V)
    rotated,_=projected_rollout(bp,model,x,transported,1.0)
    assert (native-rotated).abs().max()<5e-5
    assert parity(bp,model,x)["passed"]


def test_score_is_full_closed_loop_and_context_separated():
    model,_=nc.train_task(nc.SMOKE,"A",steps=4)
    bp=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bp,"native",nc.SMOKE)
    x=sample_x(nc.SMOKE,3,64)
    y=nc.task_labels(x,"A")
    clean,w,report=nc.clean_reference(bp,model,x,y)
    p=torch.eye(WIDTH,dtype=torch.float64).expand(PASSES,-1,-1)
    scored=nc.score_projector(bp,model,x,y,clean,w,p)
    assert scored["kl_by_pass_nats"][-1]<1e-10
    assert abs(scored["nll_change_from_clean_nats"])<1e-6
    assert scored["by_context"]["shared"]["n"]+scored["by_context"]["specific"]["n"]==len(x)


def test_root_eligibility_and_confirmation_seal():
    with pytest.raises(ValueError):
        nc.train_task(1100,"A",steps=1)
    with pytest.raises(ValueError):
        nc.run(1100,steps=1,discovery_size=4,alignment_size=4,evaluation_size=4)


def test_familywise_bootstrap_and_missing_roots():
    x=bootstrap([.01,.02,.03,.04,.01,.02,.03,.04],seed=44)
    assert x["n_independent_pairs"]==8
    assert abs(x["familywise_coverage"]-.9875)<1e-12
    assert x["familywise_ci"][0]<=x["mean_nats"]<=x["familywise_ci"][1]
    assert len(COMPARISON_NAMES)==4
    with pytest.raises(ValueError):
        aggregate([{"root":1000}]*8)


def test_smoke_is_usable_even_when_source_alignment_count_is_below_full_gate():
    receipt=nc.run(nc.SMOKE,steps=4,discovery_size=64,alignment_size=64,
                   evaluation_size=64,proposals=32)
    assert receipt["root"]==nc.SMOKE
    assert receipt["phase"]=="ENGINEERING"
    assert receipt["alignment"].get("positive_context_count",0)<300
    assert not receipt["eligible"]
    assert "insufficient_positive_context_alignment" in receipt["eligibility_reasons"]
    json.dumps(receipt,allow_nan=False)
