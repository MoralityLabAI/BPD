"""NC-12 source audit: exact low-budget count, ridge geometry and failure gates."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

import mechinterp.bpd.experiments.nc12.experiment as nc
from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import PASSES,WIDTH,sample_x,parity
from mechinterp.bpd.experiments.nc10.experiment import procrustes
from mechinterp.bpd.experiments.nc11.experiment import read_and_score_jacobian,score_projector,clean_reference
from mechinterp.bpd.experiments.nc04.experiment import factor_basis
from mechinterp.bpd.experiments.nc12.aggregate import aggregate,paired_root_interval,MULTIPLICITY


def _wrapped(root=nc.SMOKE,task="B",steps=3):
    model,_=nc.train_task(root,task,steps)
    bpd=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                 ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bpd,"native",root if task=="A" else root+100_000)
    return model,bpd


def test_prospective_roots_query_grid_and_provenance():
    assert nc.ROOTS==tuple(range(1200,1208))
    assert nc.SMOKE==1199
    assert nc.BUDGETS==(4,8,16,32)
    assert nc.PROTOCOL_COMMIT=="6a4b5ebc16fae730f17b0e5dac5564eaf419c071"
    assert MULTIPLICITY==16
    assert nc.RIDGE==.05
    for old in (1099,1100,1198,1208):
        with pytest.raises(ValueError):
            nc.train_task(old,"A",steps=1)
        with pytest.raises(ValueError):
            nc.run(old,steps=1,discovery_size=4,dose_size=4,evaluation_size=4,proposals=4)


def test_training_separate_seeds_and_input_streams():
    a,ba=nc.train_task(nc.SMOKE,"A",steps=4)
    b,bb=nc.train_task(nc.SMOKE,"B",steps=4)
    assert ba["model_seed"] != bb["model_seed"]
    assert ba["training_split"]==1 and bb["training_split"]==11
    assert not torch.equal(a.mix.weight,b.mix.weight)
    assert not torch.equal(sample_x(nc.SMOKE,1,16),sample_x(nc.SMOKE,11,16))
    assert ba["steps"]==bb["steps"]==4


def test_registration_has_nested_4_8_16_32_gradient_prefixes():
    p,x,diag=nc.split_calibration(sample_x(nc.SMOKE,4,512))
    assert p.shape==(128,5)
    assert x.shape==(32,5)
    assert (p[:,4]>0).all()
    assert torch.equal(p[:32],x)
    assert diag["nested_gradient_union"]==32
    assert diag["positive_context_count"]>=128
    assert diag["nested_budgets"]==[4,8,16,32]


def test_incomplete_calibration_pool_is_rejected_before_scoring():
    with pytest.raises(ValueError):
        nc.split_calibration(torch.randn(128,5))


def test_regularized_jacobian_procrustes_identifies_known_rotation():
    g=torch.Generator().manual_seed(121212)
    x=torch.randn(128,PASSES,WIDTH,generator=g,dtype=torch.float64)
    x=x/x.norm(dim=-1,keepdim=True)
    qs=[]
    for _ in range(PASSES):
        q,r=torch.linalg.qr(torch.randn(WIDTH,WIDTH,generator=g,dtype=torch.float64))
        qs.append(q*torch.where(torch.diag(r)>=0,1.,-1.).unsqueeze(0))
    truth=torch.stack(qs)
    y=torch.einsum("nti,tij->ntj",x,truth)
    out,diag=nc.regularized_jacobian_procrustes(x,y,truth)
    assert diag["eligible"]
    assert diag["fixed_ridge"]==.05
    assert max(diag["normalized_gradient_fit_residual_by_pass"])<1e-9
    assert (out-truth).abs().max()<1e-9
    assert diag["max_orthogonality_abs"]<1e-10


def test_q4_prior_resolves_underidentified_20d_jacobian_chart():
    g=torch.Generator().manual_seed(8763)
    source=torch.randn(4,PASSES,WIDTH,generator=g,dtype=torch.float64)
    target=torch.randn(4,PASSES,WIDTH,generator=g,dtype=torch.float64)
    source=source/source.norm(dim=-1,keepdim=True)
    target=target/target.norm(dim=-1,keepdim=True)
    prior=torch.eye(WIDTH,dtype=torch.float64).expand(PASSES,-1,-1)
    p,diag=nc.regularized_jacobian_procrustes(source,target,prior)
    assert p.shape==(PASSES,WIDTH,WIDTH)
    assert diag["max_orthogonality_abs"]<1e-10
    assert all(v>0 for v in diag["minimum_regularized_singular_by_pass"])


def test_circular_permutation_changes_q4_case_correspondence():
    q=torch.arange(4*PASSES*WIDTH,dtype=torch.float64).reshape(4,PASSES,WIDTH)
    assert not torch.equal(q,q.roll(1,0))
    assert torch.equal(q.sort(dim=0).values,q.roll(1,0).sort(dim=0).values)


def test_native_and_rotated_fullmask_parity():
    m,b=_wrapped()
    x=sample_x(nc.SMOKE,6,32)
    assert parity(b,m,x)["passed"]
    factor_basis(b,"rotated",nc.SMOKE+100000)
    assert parity(b,m,x)["passed"]


def test_oracle_blind_jacobian_calibration_and_source_projection(monkeypatch):
    m,b=_wrapped()
    def forbidden(*args,**kwargs):
        raise AssertionError("external task label in source-only gradient calibration")
    monkeypatch.setattr(nc,"task_labels",forbidden)
    x=sample_x(nc.SMOKE,4,32)
    a,g=read_and_score_jacobian(b,m,x)
    assert a.shape==g.shape==(32,PASSES,WIDTH)
    assert float(g.abs().max())>0
    p,d,_=nc.projectors_at_ranks(g)
    assert p[4] is None or p[4].shape==(PASSES,WIDTH,WIDTH)
    assert "spectrum" in d[4]


def test_gain_alignment_uses_only_clean_dose_data():
    g=torch.Generator().manual_seed(511)
    acts=torch.randn(48,PASSES,WIDTH,generator=g,dtype=torch.float64)
    p=torch.eye(WIDTH,dtype=torch.float64).expand(PASSES,-1,-1)
    from mechinterp.bpd.experiments.nc11.experiment import fixed_gain
    d=nc.clean_dose_rms(acts,torch.eye(WIDTH),p)
    match=fixed_gain(d,acts,torch.eye(WIDTH),p)
    assert match["eligible"] and abs(match["applied_gain"]-1)<1e-10
    blocked=fixed_gain(4*d,acts,torch.eye(WIDTH),p)
    assert not blocked["eligible"]


def test_exact_full_read_projector_matches_clean_model_on_all_passes():
    m,b=_wrapped()
    x=sample_x(nc.SMOKE,3,32)
    from mechinterp.bpd.experiments.nc10.experiment import task_labels
    y=task_labels(x,"B")
    c,w,_=clean_reference(b,m,x,y)
    ident=torch.eye(WIDTH,dtype=torch.float64).expand(PASSES,-1,-1)
    p=score_projector(b,m,x,y,c,w,ident,1.)
    assert max(abs(v) for v in p["kl_by_pass_nats"])<1e-9
    assert abs(p["nll_change_from_clean_nats"])<1e-6


def test_local_sham_doses_are_source_only_and_rank_four():
    from mechinterp.bpd.experiments.nc06.experiment import local_matched_shams
    rng=torch.Generator().manual_seed(231)
    a=torch.randn(128,PASSES,WIDTH,generator=rng,dtype=torch.float64)
    e,_=torch.linalg.qr(torch.randn(PASSES,WIDTH,4,generator=rng,dtype=torch.float64))
    selected,receipt=local_matched_shams(a,torch.eye(WIDTH),e,seed=nc.SMOKE,
                                         arm_index=8,proposals=256,want=8)
    assert receipt["admitted"],receipt
    assert len(selected)==8
    assert all(float((p@p-p).abs().max())<1e-8 for p in selected)


def test_zero_output_head_has_zero_task_score_gradient():
    m,b=_wrapped()
    with torch.no_grad():
        m.head.weight.zero_()
        m.head.bias.zero_()
    _,g=read_and_score_jacobian(b,m,sample_x(nc.SMOKE,6,16))
    assert float(g.abs().max())==0.


def test_familywise_root_bootstrap_uses_pairs_and_all_sixteen_comparisons():
    result=paired_root_interval([.01,.02,.03,.04,.05,.06,.07,.08],seed=1312)
    assert result["n_independent_model_pairs"]==8
    assert math.isclose(result["confidence"],.996875,abs_tol=1e-12)
    assert result["familywise_ci"][0]<=result["mean_nats"]<=result["familywise_ci"][1]


def test_blocked_roots_remain_a_complete_scientific_negative():
    rows=[]
    for root in nc.ROOTS:
        rows.append({
            "study":"BPD-NC-12","phase":"ENGINEERING","root":root,
            "protocol_commit":nc.PROTOCOL_COMMIT,
            "budgets":{"steps":384,"train_A":2048,"train_B":2048,
                "discovery":1024,"calibration_pool":512,
                "activation_calibration":128,"target_gradient_union":32,
                "dose":1024,"evaluation":1024,"sham_proposals":2048},
            "source_sha256":"test-source","protocol_sha256":"test-protocol",
            "excluded_global":["blocked_fixture"],
            "q_arms":{str(q):{"budget":q,"excluded_reasons":["blocked_fixture"],
                            "technically_admitted":False} for q in nc.BUDGETS},
            "heldout":{},
        })
    r=aggregate(rows)
    assert r["original_development_verdict"]=="ENGINEERING_NULL_OR_INELIGIBLE"
    assert r["lowest_qualifying_low_budget"] is None
    assert len(r["arms"])==4
    assert all(len(v["excluded_roots"])==8 for v in r["arms"].values())
    json.dumps(r,allow_nan=False)


def test_reduced_engineering_smoke_serializes_without_claiming_eight_roots():
    r=nc.run(nc.SMOKE,steps=4,discovery_size=64,
             dose_size=128,evaluation_size=128,proposals=128)
    assert r["study"]=="BPD-NC-12" and r["phase"]=="ENGINEERING"
    assert r["root"]==nc.SMOKE and len(r["q_arms"])==4
    assert r["calibration"]["nested_gradient_union"]==32
    assert r["protocol_commit"]==nc.PROTOCOL_COMMIT
    assert isinstance(r["excluded_global"],list)
    assert r["source_sha256"] and r["protocol_sha256"]
    json.dumps(r,allow_nan=False)
