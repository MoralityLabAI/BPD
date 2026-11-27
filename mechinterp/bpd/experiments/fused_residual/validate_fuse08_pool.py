"""FUSE-08: native step-wide pool stress and lifecycle gate."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from mechinterp.bpd.components import TiedComponentLinear as Base, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig
from .step_cache import StepResidualPool, planned_sites

ATOL, RTOL = 5e-5, 5e-4


def make(width=16, rank=4):
    return Base(nn.Linear(width,width), rank, max_passes=3,
                ablation=AblationConfig(mode="zero",sphere_rescale=False),
                init_svd=False)


def stress():
    cases=checks=exact=failed_cases=failures=nonfinite=0
    max_abs=0.; rows=[]; examples=[]; storage_checks=cross_hits=0
    for scale,target in itertools.product((1.,10.,100.,10000.),("ordinary","exact","near")):
        row=dict(scale=scale,target=target,cases=0,checks=0,exact=0,failed_cases=0)
        for seed in range(4000001,4000009):
            torch.manual_seed(seed)
            w=Base(nn.Linear(64,64),32,max_passes=3,
                   ablation=AblationConfig(mode="pass_mean",sphere_rescale=True),
                   init_svd=False)
            with torch.no_grad():
                w.V.mul_(scale);w.U.mul_(scale)
                if target!="ordinary":
                    w.W_target.copy_(w.reconstructed_weight()+
                        (torch.randn_like(w.W_target)*1e-4 if target=="near" else 0.))
                x=torch.randn(2,3,1,8,64);cot=torch.randn_like(x)
                m=torch.randn(1,3,8,32);ds=torch.randn(2,1,3,8)
                for i in range(3):w.observe_clean_pass(x[0,i],i)
            def run(fast):
                q=copy.deepcopy(w);q.mode="component"
                xx=x.clone().requires_grad_();mm=m.clone().requires_grad_()
                dd=ds.clone().requires_grad_()
                pool=StepResidualPool([q],max_bytes=q.W_target.numel()*4) if fast else None
                out=[];losses=[];stats=[];ptrs=[]
                try:
                    for j in range(2):
                        q.reset_pass_state()
                        q.schedule=PassMaskSchedule((mm+.25*j).sigmoid(),dd[j].sigmoid())
                        with (pool.scope() if fast else nullcontext(None)) as st:
                            y=torch.stack([q(xx[j,i]) for i in range(3)])
                        assert q.calls==3 and "weight_delta" not in q.__dict__
                        out.append(y);losses.append((y*cot[j]).sum());stats.append(st)
                        if fast:ptrs.append(pool.entries[id(q)][1].untyped_storage().data_ptr())
                    loss=losses[0]+losses[1]+q.weight_delta().square().mean()*.01
                    grads=torch.autograd.grad(loss,(xx,q.V,q.U,mm,dd))
                    if fast:
                        assert ptrs[0]==ptrs[1]
                        assert pool.sealed and not pool.poisoned
                        assert pool.peak_bytes==q.W_target.numel()*4
                        assert (stats[0].misses,stats[0].hits,stats[0].cross_scope_hits)==(1,2,0)
                        assert (stats[1].misses,stats[1].hits,stats[1].cross_scope_hits)==(0,3,3)
                        assert all(s.storage_checks==3 and s.backward_nodes==3 for s in stats)
                    return [v.detach() for v in (*out,*grads)],stats
                finally:
                    if pool is not None:
                        pool.close()
                        assert not pool.entries and pool.resident_bytes==0 and pool.closed
            a,_=run(False);b,stats=run(True);bad=[]
            storage_checks+=sum(st.storage_checks for st in stats)
            cross_hits+=sum(st.cross_scope_hits for st in stats)
            for name,u,v in zip(("out0","out1","x","V","U","m","d"),a,b):
                checks+=1;row["checks"]+=1
                eq=torch.equal(u,v);exact+=int(eq);row["exact"]+=int(eq)
                finite=bool(torch.isfinite(u).all() and torch.isfinite(v).all())
                nonfinite+=int(not finite)
                if finite:max_abs=max(max_abs,float((u-v).abs().max()))
                if not finite or not torch.allclose(u,v,atol=ATOL,rtol=RTOL):
                    failures+=1;bad.append(name)
            cases+=1;row["cases"]+=1
            if bad:
                failed_cases+=1;row["failed_cases"]+=1
                if len(examples)<12:examples.append(dict(seed=seed,scale=scale,target=target,failed=bad))
        rows.append(row)
    return dict(cases=cases,checks=checks,exact=exact,failures=failures,
                failed_cases=failed_cases,nonfinite=nonfinite,max_abs=max_abs,
                storage_checks=storage_checks,cross_scope_hits=cross_hits,
                examples=examples,rows=rows)


def lifecycle():
    torch.manual_seed(4000901)
    checks={}
    def check(k,v):
        checks[k]=bool(v)
        assert v,k
    def raises(fn):
        try:fn()
        except (RuntimeError,ValueError):return True
        return False
    def enter(pool):
        with pool.scope():pass

    w=make();n=w.W_target.numel()*4
    p=StepResidualPool([w],max_bytes=n)
    with p.scope() as a:
        v0=w.weight_delta();v1=w.weight_delta()
        check("independent_nodes",v0.grad_fn is not v1.grad_fn)
        check("same_scope_storage",v0.untyped_storage().data_ptr()==v1.untyped_storage().data_ptr())
        check("nested_rejected",raises(lambda:enter(p)))
    with p.scope() as b:
        v2=w.weight_delta()
        check("cross_scope_storage",v2.untyped_storage().data_ptr()==v0.untyped_storage().data_ptr())
        check("cross_scope_stats",b.misses==0 and b.cross_scope_hits==1)
        check("distinct_adjoints_across_scopes",v0.grad_fn is not v2.grad_fn)
    check("pool_cap",p.peak_bytes==n and p.resident_bytes==n)
    (v0+v1+v2).square().sum().backward()
    check("adjoints_executed",a.backward_nodes==2 and b.backward_nodes==1)
    check("sealed_after_backward",p.sealed and raises(lambda:enter(p)))
    p.close()
    check("closed_no_reentry",p.closed and p.resident_bytes==0 and raises(lambda:enter(p)))
    check("method_restored","weight_delta" not in w.__dict__ and "_residual_cache_active" not in w.__dict__)

    w=make();p=StepResidualPool([w],max_bytes=n)
    with p.scope():w.weight_delta()
    with torch.no_grad():w.V.add_(.01)
    check("mutation_between_scopes",raises(lambda:enter(p)) and p.poisoned)
    p.close()

    w=make();p=StepResidualPool([w],max_bytes=n)
    with p.scope():w.weight_delta()
    w.U=nn.Parameter(w.U.detach().clone())
    check("identity_replacement",raises(lambda:enter(p)) and p.poisoned)
    p.close()

    w=make();p=StepResidualPool([w],max_bytes=n)
    with p.scope():w.weight_delta()
    with torch.no_grad():p.entries[id(w)][1].add_(1)
    check("cached_storage_mutation",raises(lambda:enter(p)) and p.poisoned)
    p.close()

    w=make();p=StepResidualPool([w],max_bytes=n)
    def early():
        with p.scope():w.weight_delta().square().sum().backward()
    check("early_backward",raises(early) and p.poisoned and p.sealed)
    check("early_backward_restored","weight_delta" not in w.__dict__)
    p.close()

    w=make();p=StepResidualPool([w],max_bytes=n)
    try:
        with p.scope():
            w.weight_delta()
            raise RuntimeError("intentional")
    except RuntimeError:pass
    check("exception_restored",p.poisoned and "weight_delta" not in w.__dict__)
    p.close()

    w=make();p=StepResidualPool([w],max_bytes=n)
    with p.scope() as st:
        with torch.no_grad():v=w.weight_delta()
        check("no_grad_bypass",not v.requires_grad and st.bypasses==1 and st.misses==0)
        with torch.autocast("cpu",dtype=torch.bfloat16):w.weight_delta()
        check("autocast_bypass",st.bypasses==2 and st.misses==0)
        z=w.weight_delta()
        check("grad_cache_after_bypass",z.requires_grad and st.misses==1)
    z.sum().backward();p.close()

    w=make();p=StepResidualPool([w],max_bytes=0)
    with p.scope() as st:
        v=w.weight_delta();v1=w.weight_delta()
        check("zero_cap",st.cap_rejections==2 and st.peak_bytes==0 and torch.equal(v,v1))
    (v+v1).sum().backward();p.close()

    w=make();q=make();p=StepResidualPool([w,q],max_bytes=n)
    with p.scope() as st:
        a=w.weight_delta();b=q.weight_delta()
        check("aggregate_binding_cap",st.misses==1 and st.cap_rejections==1 and p.peak_bytes==n)
    (a+b).sum().backward();p.close()

    w=make();q=make()
    with torch.no_grad():
        chosen,plan=planned_sites([w,q],6,2*n)
    check("unsupported_planner",not chosen and plan.value==0)
    check("frontier_limit",raises(lambda:planned_sites([w,q],6,2*n,max_frontier=1)))
    chosen,plan=planned_sites([w,q],1,2*n)
    check("zero_benefit",not chosen and plan.value==0)

    w=make();opt=torch.optim.AdamW(w.parameters(),lr=.01)
    p=StepResidualPool([w],max_bytes=n)
    with p.scope():old=w.weight_delta()
    old.square().sum().backward();p.close();opt.step();opt.zero_grad(set_to_none=True)
    q=StepResidualPool([w],max_bytes=n)
    with q.scope() as st:
        new=w.weight_delta()
        check("fresh_after_optimizer",st.misses==1 and not torch.equal(old,new))
    new.square().sum().backward();q.close()
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    life=lifecycle();result=stress()
    out=dict(torch=torch.__version__,threads=torch.get_num_threads(),lifecycle=life,stress=result)
    Path(a.output).write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(lifecycle=life,stress={k:v for k,v in result.items() if k!="rows"}),indent=2))
    assert not result["failures"] and result["exact"]==result["checks"]


if __name__=="__main__":main()
