"""SUB-07 source-only tests: exact oracle, noncommuting depth and driven GRU math."""
import math

import pytest
import torch
from torch.nn import functional as F

from mechinterp.bpd.experiments.sub07.oracle import (
    TOKENS, SEQ_LEN, INITIAL_BITS, oracle, transition, universe,
    partition, split_indices, exact_task_certificate,
)
from mechinterp.bpd.experiments.sub07.source import (
    WIDTH, RANK, PASSES, ROOTS, SMOKE, CLOCK_MAPS, STEP_MAPS,
    DISCOVERY_JACOBIAN_ROWS, TiedGRU, tensor_split, discovered_source,
    source_fit_family, stationary_source, bundle_loss, schedule_arrays,
    train_source,
)
from mechinterp.bpd.experiments.sub07.experiment import (
    physical_delta, fixed_stationary, source_causal_response,
    clocked_student, gauge_transport, preflight_spec, aggregate, bootstrap,
)


def test_exact_oracle_enumerated_information_theory_and_order_witness():
    cert=exact_task_certificate()
    assert cert["positives"]==2291
    assert cert["oracle_sequences"]==4096
    assert cert["prefix2"]["bayes_correct"]==2399
    assert cert["prefix4"]["bayes_correct"]==2735
    assert cert["last_token"]["bayes_correct"]==2291
    assert cert["token_counts"]["bayes_correct"]==2335
    assert cert["label_entropy_bits"]>.98
    assert cert["prefix2"]["conditional_entropy_bits"]>.96
    assert cert["prefix4"]["conditional_entropy_bits"]>.87
    assert oracle((0,0,1,1,3,2))!=oracle((0,0,1,1,2,3))
    for t in range(4):
        s=INITIAL_BITS
        for _ in range(4):
            s=transition(s,t)
        # Bijectivity on all 4 states is also independently tested below.
    with pytest.raises(ValueError):
        oracle((0,1))
    for token in range(4):
        images={transition((a,b),token) for a in (0,1) for b in (0,1)}
        assert len(images)==4


def test_split_disjoint_complete_deterministic_and_onehot_tensors():
    first=split_indices()
    second=split_indices()
    for name in ("train","discovery","dose","heldout"):
        assert first[name].tolist()==second[name].tolist()
    ids=[int(z) for x in first.values() for z in x]
    assert len(ids)==4096 and len(set(ids))==4096
    x,y,keys=tensor_split("heldout")
    assert x.shape==(1024,6,4) and y.shape==(1024,)
    assert all(v in (0.,1.) for v in set(y.tolist()))
    assert torch.equal(x.sum(dim=-1),torch.ones(1024,6))
    from_oracle=[oracle(tuple(int(t) for t in row.argmax(-1)))
                 for row in x[:32]]
    assert from_oracle==[int(v) for v in y[:32]]


def test_cyclic_orthogonal_fiber_cocycle_requires_correct_pass_index():
    eye=torch.eye(WIDTH,dtype=torch.float64)
    w=eye.roll(8,dims=0)
    p0=torch.diag(torch.tensor([1.]*RANK+[0.]*(WIDTH-RANK),
                               dtype=torch.float64))
    ps=[]
    for t in range(PASSES):
        wt=torch.linalg.matrix_power(w,t)
        ps.append(wt@p0@wt.T)
    assert max(float((ps[t+1]@w@(eye-ps[t])).norm())
               for t in range(PASSES-1))<1e-10
    assert float((p0@w@(eye-p0)).norm())>1
    bad=[ps[t] for t in CLOCK_MAPS[5]]
    assert max(float((bad[t+1]@w@(eye-bad[t])).norm())
               for t in range(PASSES-1))>1


def test_gru_jacobians_nontrivial_numerical_parity_and_source_shapes():
    torch.set_num_threads(1)
    torch.manual_seed(99)
    m=TiedGRU().eval()
    x,_,_=tensor_split("discovery")
    x=x[:DISCOVERY_JACOBIAN_ROWS]
    d,c,gradnorm=discovered_source(m,x)
    assert d.shape==(DISCOVERY_JACOBIAN_ROWS,5,WIDTH,WIDTH)
    assert c.shape==(6,WIDTH,WIDTH)
    assert torch.isfinite(d).all() and torch.isfinite(c).all()
    assert min(gradnorm)>0
    assert torch.linalg.eigvalsh(c).min()>=-1e-10
    _,states=m.rollout(x)
    h=states[1][:1].detach()
    step=x[:1,2]
    eps=.001
    unit=torch.zeros_like(h)
    unit[:,0]=eps
    with torch.no_grad():
        numerical=((m.step(h+unit,step)-m.step(h-unit,step))/(2*eps))
    expected=d[0,1,:,0]
    assert float((numerical[0].double()-expected).abs().max())<.002


def test_frozen_source_schedules_and_matched_budget_geometry():
    torch.set_num_threads(1)
    torch.manual_seed(44)
    m=TiedGRU().eval()
    x,_,_=tensor_split("discovery")
    x=x[:DISCOVERY_JACOBIAN_ROWS]
    d,c,norms=discovered_source(m,x)
    assert len(CLOCK_MAPS)==len(STEP_MAPS)==7
    for i in range(7):
        ds,cs,tau,pi=schedule_arrays(d,c,i)
        assert sorted(tau)==list(range(5))
        assert sorted(pi)==list(range(6))
        assert all(torch.equal(ds[:,j],d[:,tau[j]]) for j in range(5))
        assert all(torch.equal(cs[j],c[pi[j]]) for j in range(6))
        assert torch.allclose(ds.square().sum(),d.square().sum(),
                              atol=1e-10,rtol=1e-10)
    p,q,static_info=stationary_source(d,c)
    assert static_info["optimizer_steps"]==80
    ps,bases,diag,common=source_fit_family(d,c,q)
    assert len(ps)==len(bases)==len(diag)==8
    for name in diag:
        assert len(diag[name]["source_loss_trace"])==120
        assert diag[name]["optimizer_steps"]==120
        assert diag[name]["rank_trace_max_error"]<1e-4
        assert diag[name]["idempotence_max"]<1e-5
        assert diag[name]["symmetry_max"]<1e-5
        assert ps[name].shape==(PASSES,WIDTH,WIDTH)
    assert common["a0"]>0 and common["b0"]>0
    assert diag["aligned"]["source_output_schedule"]==list(range(6))
    assert diag["aligned"]["source_step_schedule"]==list(range(5))


def test_excluded_smoke_parity_dose_task_and_gauge():
    torch.set_num_threads(1)
    m,_=train_source(SMOKE,steps=16)
    x,y,_=tensor_split("heldout")
    x,y=x[:40],y[:40]
    p=torch.diag(torch.tensor([1.]*RANK+[0.]*(WIDTH-RANK)))
    bundle=fixed_stationary(p)
    delta=physical_delta(SMOKE,4,0,len(x),p,.3)
    assert torch.allclose(delta.norm(dim=-1),
                          torch.full((len(x),),.3),atol=1e-6)
    assert float((delta@p).norm(dim=-1).max())<1e-5
    scored=source_causal_response(m,x,y,bundle,.3,SMOKE,4)
    assert len(scored["per_pass_projected_next_state_rms"])==5
    assert scored["max_physical_dose_error"]<5e-6
    full=clocked_student(m,x,y,fixed_stationary(torch.eye(WIDTH)))
    assert full["full_head_max_abs_error"]<1e-6
    gauge=gauge_transport(m,x,bundle,.3,SMOKE)
    assert gauge["clean_logit_error"]<5e-6
    assert gauge["next_state_transported_max_error"]<5e-6
    assert gauge["terminal_logit_transported_max_error"]<5e-6
    assert max(gauge["wrong_untransported_logit_rms"])>1e-5


def test_root_roster_and_preflight_clustered_units():
    assert ROOTS==tuple(range(1800,1808)) and SMOKE not in ROOTS
    with pytest.raises(ValueError,match="outside frozen"):
        train_source(1900,steps=1)
    with pytest.raises(ValueError,match="eight registered"):
        aggregate([])
    fake={
        "root":1800,"primary_strongest_minus_aligned":.02,
        "scores":{
            "stationary":{"pooled_next_state_rms":.3},
            **{f"wrong_{i}":{"pooled_next_state_rms":.23} for i in range(7)},
        },
    }
    pre=preflight_spec([fake])
    assert pre["confirm_units_planned"]==64
    assert len(pre["pilot"])==1
    assert len(pre["pilot"][0]["sham_effects"])==7
    assert pre["scientific_min_effect"]=="0.009900990099009901"
    assert len(bootstrap([.01]*8))==2
