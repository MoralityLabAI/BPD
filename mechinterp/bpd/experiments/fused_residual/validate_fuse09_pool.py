"""FUSE09: source-only PGD cache stress; separate training adjoints."""
from __future__ import annotations
import argparse, copy, itertools, json
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from mechinterp.bpd.components import TiedComponentLinear as Base, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig
from .step_cache import StepResidualPool

ATOL, RTOL = 5e-5, 5e-4


def stress():
    checks=exact=cases=failures=failed_cases=nonfinite=0
    storage_checks=mask_calls=adjoints=cross_hits=0
    maximum=0.; examples=[]; rows=[]
    for scale,target in itertools.product((1.,10.,100.,10000.),("ordinary","exact","near")):
        row=dict(scale=scale,target=target,cases=0,checks=0,exact=0,failed_cases=0)
        for seed in range(4009001,4009009):
            torch.manual_seed(seed)
            w=Base(nn.Linear(64,64),32,max_passes=3,
                   ablation=AblationConfig(mode="pass_mean",sphere_rescale=True),init_svd=False)
            with torch.no_grad():
                w.V.mul_(scale);w.U.mul_(scale)
                if target!="ordinary":
                    w.W_target.copy_(w.reconstructed_weight()+
                       (torch.randn_like(w.W_target)*1e-4 if target=="near" else 0.))
                x=torch.randn(3,3,1,8,64);cot=torch.randn_like(x)
                m=torch.randn(3,1,3,8,32);d=torch.randn(3,1,3,8)
                for i in range(3):w.observe_clean_pass(x[0,i],i)
            def run(fast):
                q=copy.deepcopy(w);q.mode="component"
                pool=StepResidualPool([q],max_bytes=q.W_target.numel()*4) if fast else None
                vals=[];stats=[];ptrs=[]
                try:
                    for j in range(2):
                        # The inner graph is consumed only for source derivatives.
                        mm=m[j].clone().requires_grad_();dd=d[j].clone().requires_grad_()
                        q.reset_pass_state();q.schedule=PassMaskSchedule(mm.sigmoid(),dd.sigmoid())
                        with (pool.scope(mask_only=True) if fast else nullcontext(None)) as st:
                            y=torch.stack([q(x[j,i]) for i in range(3)])
                        assert q.calls==3 and "weight_delta" not in q.__dict__
                        loss=(y*cot[j]).sum()
                        gm,gd=torch.autograd.grad(loss,(mm,dd))
                        vals.extend([y.detach(),gm.detach(),gd.detach()])
                        del y,loss,gm,gd,mm,dd
                        stats.append(st)
                        if fast:
                            ptrs.append(pool.entries[id(q)][1].untyped_storage().data_ptr())
                            assert st.mask_only_calls==3 and st.independent_calls==0 and st.backward_nodes==0
                            assert not pool.sealed and not pool.poisoned
                    # A new, independent training graph, not either inner loss.
                    xx=x[2].clone().requires_grad_();mm=m[2].clone().requires_grad_()
                    dd=d[2].clone().requires_grad_();q.reset_pass_state()
                    q.schedule=PassMaskSchedule(mm.sigmoid(),dd.sigmoid())
                    with (pool.scope() if fast else nullcontext(None)) as st:
                        y=torch.stack([q(xx[i]) for i in range(3)])
                    loss=(y*cot[2]).sum()+q.weight_delta().square().mean()*.01
                    gs=torch.autograd.grad(loss,(xx,q.V,q.U,mm,dd))
                    vals.extend([y.detach(),*[g.detach() for g in gs]])
                    stats.append(st)
                    if fast:
                        ptrs.append(pool.entries[id(q)][1].untyped_storage().data_ptr())
                        assert len(set(ptrs))==1
                        assert st.mask_only_calls==0 and st.independent_calls==3 and st.backward_nodes==3
                        assert pool.sealed and not pool.poisoned
                        assert pool.peak_bytes==q.W_target.numel()*4
                        assert [(s.misses,s.hits,s.cross_scope_hits) for s in stats]==[(1,2,0),(0,3,3),(0,3,3)]
                    return vals,stats
                finally:
                    if pool is not None:
                        pool.close()
                        assert pool.closed and pool.resident_bytes==0 and not pool.entries
            a,_=run(False);b,stats=run(True);bad=[]
            storage_checks+=sum(s.storage_checks for s in stats)
            mask_calls+=sum(s.mask_only_calls for s in stats)
            adjoints+=sum(s.backward_nodes for s in stats)
            cross_hits+=sum(s.cross_scope_hits for s in stats)
            names=("inner0_output","inner0_m","inner0_d","inner1_output","inner1_m","inner1_d",
                   "joint_output","x","V","U","m","d")
            for name,u,v in zip(names,a,b):
                checks+=1;row["checks"]+=1
                eq=torch.equal(u,v);exact+=int(eq);row["exact"]+=int(eq)
                finite=bool(torch.isfinite(u).all() and torch.isfinite(v).all());nonfinite+=int(not finite)
                if finite:maximum=max(maximum,float((u-v).abs().max()))
                if not finite or not torch.allclose(u,v,atol=ATOL,rtol=RTOL):
                    failures+=1;bad.append(name)
            cases+=1;row["cases"]+=1
            if bad:
                failed_cases+=1;row["failed_cases"]+=1
                if len(examples)<12:examples.append(dict(seed=seed,scale=scale,target=target,failed=bad))
        rows.append(row)
    return dict(cases=cases,checks=checks,exact=exact,failures=failures,failed_cases=failed_cases,
                nonfinite=nonfinite,max_abs=maximum,storage_checks=storage_checks,
                mask_only_calls=mask_calls,backward_nodes=adjoints,cross_scope_hits=cross_hits,
                examples=examples,rows=rows)


def lifecycle():
    checks={}
    def check(k,v):
        checks[k]=bool(v);assert v,k
    def raises(fn):
        try:fn()
        except (RuntimeError,ValueError):return True
        return False
    def enter(p):
        with p.scope(mask_only=True):pass
    def make():
        return Base(nn.Linear(16,16),4,max_passes=3,
                    ablation=AblationConfig(mode="zero",sphere_rescale=False),init_svd=False)
    w=make();n=w.W_target.numel()*4;p=StepResidualPool([w],max_bytes=n)
    with p.scope(mask_only=True) as st:
        a=w.weight_delta();b=w.weight_delta()
        check("detached_value",not a.requires_grad and a.grad_fn is None and torch.equal(a,b))
        check("mask_scope_counts",st.mask_only_calls==2 and st.independent_calls==0)
        check("nested_mask_scope_rejected",raises(lambda:enter(p)))
        check("close_during_scope_rejected",raises(p.close))
        # The derivative with respect to an independent activation remains.
        x=torch.randn(2,16,requires_grad=True)
        gx=torch.autograd.grad((x@a.T).sum(),x)[0]
        check("activation_derivative",torch.equal(gx,a.sum(0).expand_as(x)))
    check("source_gradient_does_not_seal",not p.sealed and not p.poisoned)
    with p.scope() as tr:
        c=w.weight_delta()
        check("joint_has_independent_adjoint",c.requires_grad and c.grad_fn is not None)
        check("same_storage_after_mask",c.untyped_storage().data_ptr()==a.untyped_storage().data_ptr())
    c.square().sum().backward()
    check("joint_backward_seals",p.sealed and tr.backward_nodes==1 and raises(lambda:enter(p)))
    p.close();check("closed_restored",p.closed and p.resident_bytes==0 and "weight_delta" not in w.__dict__)
    w=make();p=StepResidualPool([w],max_bytes=0)
    with p.scope(mask_only=True) as st:
        a=w.weight_delta()
        check("zero_cap_reference",st.cap_rejections==1 and st.mask_only_calls==0 and a.requires_grad)
    p.close()
    w=make();p=StepResidualPool([w],max_bytes=n)
    with p.scope(mask_only=True) as st:
        with torch.no_grad():w.weight_delta()
        with torch.autocast("cpu",dtype=torch.bfloat16):w.weight_delta()
        check("unsupported_bypass",st.bypasses==2 and st.misses==0)
        w.weight_delta()
    with torch.no_grad():w.V.add_(.01)
    check("mutation_poison",raises(lambda:enter(p)) and p.poisoned);p.close()
    w=make();p=StepResidualPool([w],max_bytes=n)
    try:
        with p.scope(mask_only=True):
            w.weight_delta();raise ValueError("intentional")
    except ValueError:pass
    check("exception_cleanup",p.poisoned and "weight_delta" not in w.__dict__)
    p.close()
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);a=parser.parse_args()
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest");torch.manual_seed(4009901)
    life=lifecycle();s=stress()
    Path(a.output).write_text(json.dumps(dict(torch=torch.__version__,lifecycle=life,stress=s),indent=2,allow_nan=False)+"\n")
    print("FUSE09_POOL "+json.dumps(dict(lifecycle=life,stress={k:v for k,v in s.items() if k!="rows"}),allow_nan=False))
    assert not s["failures"] and s["exact"]==s["checks"]


if __name__=="__main__":main()
