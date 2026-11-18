"""NC-09: causal mechanics, oracle-blindness and sealed-root audit tests."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import mechinterp.bpd.experiments.nc09.core as core
from mechinterp.bpd.experiments.nc09.core import (
    ROOTS, EXCLUDED_ENGINEERING_ROOT, PROTOCOL_COMMIT, PASSES, WIDTH, RANK,
    SHAMS, sample_base, mark_task, task_labels, context_masks,
    train_model, wrap_bpd, set_basis, full_mask_parity, discover,
    projector_from_gradients, physical_write_rms, source_doses,
    local_matched_shams, run_projector, transport_to_rotated,
)
from mechinterp.bpd.experiments.nc09.aggregate import aggregate,interval_root
from mechinterp.bpd.experiments.nc09.preflight_manifest import build_inputs
from mechinterp.bpd.experiments.nc09.experiment import assess_root, evaluate


def tiny_model(seed=899,steps=4):
    model,metrics=train_model(seed,steps=steps)
    bpd=wrap_bpd(model)
    set_basis(bpd,seed,"native")
    return model,bpd,metrics


def test_registered_roots_and_disjoint_sample_streams():
    assert ROOTS==tuple(range(900,908))
    assert EXCLUDED_ENGINEERING_ROOT==899
    assert PROTOCOL_COMMIT=="22e00fc22cdd5e3403d936294232bcfc051aa2e2"
    with pytest.raises(ValueError):
        train_model(898,steps=1)
    with pytest.raises(ValueError):
        assess_root(908,steps=1,discovery_size=4,evaluation_size=4)
    assert not torch.equal(sample_base(899,1,32),sample_base(899,2,32))
    assert not torch.equal(sample_base(899,2,32),sample_base(899,3,32))


def test_paired_task_labels_share_positive_branch_and_differ_on_private():
    base=sample_base(899,2,1024)
    a,b=task_labels(base,"A"),task_labels(base,"B")
    masks=context_masks(base)
    assert torch.equal(a[masks["shared"]],b[masks["shared"]])
    assert float((a[masks["private"]]!=b[masks["private"]]).float().mean())>.35
    xa,xb=mark_task(base,"A"),mark_task(base,"B")
    assert torch.equal(xa[:,:5],xb[:,:5])
    assert (xa[:,5]==-1).all() and (xb[:,5]==1).all()
    with pytest.raises(ValueError):
        mark_task(base,"unknown")


def test_training_has_unconstrained_dense_physical_weight_and_both_task_labels():
    model,bpd,report=tiny_model()
    assert report["n_trainable_parameters"]>WIDTH*WIDTH
    assert report["mix_weight_density"]>=.90
    assert not hasattr(model,"structural_mask")
    assert bpd.wrappers["mix"].W_target.shape==(WIDTH,WIDTH)
    assert report["task_marked_training_cases"]==4096


def test_all_pass_parity_of_native_and_rotated_factor_charts():
    model,bpd,_=tiny_model()
    x=mark_task(sample_base(899,3,64),"B")
    for basis in ("native","rotated"):
        set_basis(bpd,899,basis)
        check=full_mask_parity(bpd,model,x)
        assert check["passed"],(basis,check)


def test_discovery_cannot_use_external_task_labels(monkeypatch):
    model,bpd,_=tiny_model()
    def forbidden(*_a,**_kw):
        raise AssertionError("external task labels requested during discovery")
    monkeypatch.setattr(core,"task_labels",forbidden)
    x=mark_task(sample_base(899,2,48),"A")
    z,g=discover(bpd,model,x,batch_size=16)
    assert z.shape==g.shape==(48,PASSES,WIDTH)
    assert torch.isfinite(z).all()
    assert torch.isfinite(g).all()


def test_sign_flip_cancels_in_uncentered_gradient_covariance():
    rng=torch.Generator().manual_seed(120)
    grad=torch.randn(50,PASSES,WIDTH,generator=rng,dtype=torch.float64)
    sign=torch.where(torch.arange(len(grad))%2==0,1.,-1.)[:,None,None]
    a=torch.einsum("nti,ntj->tij",grad,grad)
    b=torch.einsum("nti,ntj->tij",grad*sign,grad*sign)
    assert torch.allclose(a,b,atol=1e-12)


def test_top_rank_projector_is_idempotent_and_identifies_an_eigenspace():
    torch.manual_seed(19)
    g=torch.randn(512,PASSES,WIDTH,dtype=torch.float64)
    p,e,details=projector_from_gradients(g)
    assert details["status"]=="IDENTIFIABLE",details
    assert p is not None
    assert e.shape==(PASSES,WIDTH,RANK)
    assert float((p@p-p).abs().max())<1e-9
    assert float((p-p.transpose(1,2)).abs().max())<1e-9
    assert torch.allclose(torch.diagonal(p,dim1=-2,dim2=-1).sum(-1),
                          torch.full((PASSES,),4.,dtype=torch.float64),atol=1e-7)


def test_zero_head_gradient_cannot_produce_projector():
    model,bpd,_=tiny_model()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    z,g=discover(bpd,model,mark_task(sample_base(899,2,24),"A"))
    p,_,d=projector_from_gradients(g)
    assert p is None and d["status"]=="UNIDENTIFIABLE"
    assert float(g.abs().max())==0.


def test_shared_source_equals_target_has_gain_one_and_zero_dose_mismatch():
    torch.manual_seed(23)
    acts=torch.randn(48,PASSES,WIDTH,dtype=torch.float64)
    input_base=sample_base(899,2,48)
    q,_=torch.linalg.qr(torch.randn(PASSES,WIDTH,RANK,dtype=torch.float64))
    p=q@q.transpose(1,2)
    d=source_doses(acts,input_base,torch.eye(WIDTH),p,p)
    assert abs(d["transfer_gain"]-1)<1e-12
    assert d["gain_admissible"]
    assert d["branch_dose_admissible"]
    assert all(x<1e-12 for x in d["gain_adjusted_relative_mismatch"].values())


def test_local_shams_preserve_rank_and_contextwise_discovery_dose():
    rng=torch.Generator().manual_seed(32)
    acts=torch.randn(256,PASSES,WIDTH,generator=rng,dtype=torch.float64)
    e,_=torch.linalg.qr(torch.randn(PASSES,WIDTH,RANK,generator=rng,dtype=torch.float64))
    controls,receipt=local_matched_shams(
        acts,sample_base(899,2,256),torch.eye(WIDTH),e,899,n_proposals=64)
    assert receipt["eligible"],receipt
    assert len(controls)==SHAMS
    assert len({row["proposal_index"] for row in receipt["selected"]})==SHAMS
    for p in controls:
        assert float((p@p-p).abs().max())<1e-9
    for c in receipt["selected"]:
        assert c["relative_errors"]["all"]<=.15
        assert c["relative_errors"]["shared"]<=.20
        assert c["relative_errors"]["private"]<=.20


def test_full_identity_projector_reproduces_original_native_model():
    model,bpd,_=tiny_model()
    x=mark_task(sample_base(899,3,32),"B")
    p=torch.eye(WIDTH,dtype=torch.float64).expand(PASSES,-1,-1)
    with torch.no_grad():
        with bpd.target_capture():
            clean,_=model(x)
        projected,_=run_projector(bpd,model,x,p,gain=1.)
    assert float((clean-projected).abs().max())<5e-5


def test_orthogonal_projector_transport_reproduces_full_closed_loop():
    model,bpd,_=tiny_model()
    x=mark_task(sample_base(899,3,32),"B")
    q,_=torch.linalg.qr(torch.randn(PASSES,WIDTH,RANK,dtype=torch.float64))
    p=q@q.transpose(1,2)
    native,_=run_projector(bpd,model,x,p,gain=.95)
    set_basis(bpd,899,"rotated")
    pr=transport_to_rotated(p,bpd.wrappers["mix"].V.detach())
    rot,_=run_projector(bpd,model,x,pr,gain=.95)
    assert float((native-rot).abs().max())<5e-5


def test_three_comparison_bootstrap_uses_only_trained_model_roots():
    result=interval_root([1/10,2/10,3/10,4/10,5/10,6/10,7/10,8/10],919)
    assert result["n_independent_roots"]==8
    assert abs(result["mean"]-.45)<1e-12
    assert abs(result["coverage"]-(1-.05/3))<1e-12
    assert result["familywise_interval"][0]<=.45<=result["familywise_interval"][1]


def test_blocked_scientific_result_and_preflight_do_not_unseal_anything():
    receipts=[]
    for seed in ROOTS:
        receipts.append({
            "study":"BPD-NC-09","phase":"ENGINEERING","seed":seed,
            "protocol_commit":PROTOCOL_COMMIT,
            "source_sha256":"frozen_source","core_sha256":"frozen_core",
            "protocol_sha256":"frozen_protocol",
            "data":{"train_base":2048,"train_task_marked":4096,
                    "discovery_base":1024,"evaluation_base":1024},
            "training":{"steps":640},
            "eligible":False,"status":"ENGINEERING_BLOCKED",
            "admission":{"task_model":False},
            "heldout":{
                "clean_A":{"by_context":{
                    "shared":{"accuracy":.5},"private":{"accuracy":.5}}},
                "clean_B":{"by_context":{
                    "shared":{"accuracy":.5},"private":{"accuracy":.5}}},
            },
            "dose_matching":{"gain_admissible":False},
            "sham_matching":{"eligible":False},
            "discovery":{},"primary":{},
        })
    a=aggregate(receipts)
    assert a["development_resolution"]=="ENGINEERING_NULL_OR_INELIGIBLE"
    assert len(a["excluded_seeds"])==8
    p=build_inputs(receipts)
    assert p["status"]=="HOLD_INSUFFICIENT_ELIGIBLE_DEVELOPMENT_ROOTS"
    assert p["confirmation"]=="SEALED"
    json.dumps(a,allow_nan=False)
    json.dumps(p,allow_nan=False)
