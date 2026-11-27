"""FUSE-11: retained PGD source query followed by exact joint backward."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from mechinterp.bpd.components import TiedComponentLinear, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig, NullCalibrationConfig
from .admission import Site, select
from .step_cache import StepResidualPool
from .validate_fuse09_native import build


def compare(x, y, counts, path):
    if isinstance(x, torch.Tensor):
        assert isinstance(y, torch.Tensor) and x.dtype == y.dtype, path
        assert torch.equal(x, y), path
        if x.is_floating_point() or x.is_complex():
            assert torch.isfinite(x).all(), path
        counts["tensors"] += 1
    elif dataclasses.is_dataclass(x):
        compare(dataclasses.asdict(x), dataclasses.asdict(y), counts, path)
    elif isinstance(x, dict):
        assert x.keys() == y.keys(), path
        for k in x:
            compare(x[k], y[k], counts, f"{path}/{k}")
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y), path
        for i, (u, v) in enumerate(zip(x, y)):
            compare(u, v, counts, f"{path}/{i}")
    else:
        assert x == y, path
        if isinstance(x, float):
            assert math.isfinite(x), path
        counts["scalars"] += 1


def stress():
    counts = {"tensors":0, "scalars":0}
    cases = nodes = hits = 0
    for seed, scale, target in itertools.product(
            range(4011001,4011009), (1.,10.,100.,10000.), ("ordinary","exact","near")):
        torch.manual_seed(seed)
        w = TiedComponentLinear(torch.nn.Linear(64,64),32,max_passes=3,
                ablation=AblationConfig(mode="pass_mean",sphere_rescale=True),
                init_svd=False)
        with torch.no_grad():
            w.V.mul_(scale)
            w.U.mul_(scale)
            if target != "ordinary":
                w.W_target.copy_(w.reconstructed_weight() +
                    (torch.randn_like(w.W_target)*1e-4 if target=="near" else 0.))
            x = torch.randn(3,3,1,8,64)
            m = torch.randn(3,1,3,8,32)
            d = torch.randn(3,1,3,8)
            cot = torch.randn_like(x)
            for j in range(3):
                w.observe_clean_pass(x[0,j],j)

        def run(fast):
            q = copy.deepcopy(w)
            q.mode = "component"
            p = StepResidualPool([q],max_bytes=q.W_target.numel()*4) if fast else None
            stats = []
            vals = []
            losses = []
            leaves = []
            try:
                for j in range(3):
                    xx = x[j].clone().requires_grad_()
                    mm = m[j].clone().requires_grad_()
                    dd = d[j].clone().requires_grad_()
                    q.reset_pass_state()
                    q.schedule = PassMaskSchedule(mm.sigmoid(),dd.sigmoid())
                    with (p.scope(mask_only=(j==1)) if fast else nullcontext(None)) as st:
                        yy = torch.stack([q(xx[i]) for i in range(3)])
                    loss = (yy*cot[j]).sum()
                    vals.append(yy.detach())
                    stats.append(st)
                    if j == 1:
                        g = torch.autograd.grad(loss,(mm,dd))
                        vals.extend(t.detach() for t in g)
                    else:
                        losses.append(loss)
                        leaves.extend((xx,mm,dd))
                        if j == 2:
                            if fast:
                                p.seal_forward()
                                assert stats[0].backward_nodes == stats[2].backward_nodes == 0
                            g = torch.autograd.grad(loss,(mm,dd),retain_graph=True)
                            vals.extend(t.detach() for t in g)
                            if fast:
                                assert stats[0].backward_nodes == stats[2].backward_nodes == 0
                                try:
                                    with p.scope(): pass
                                except RuntimeError:
                                    pass
                                else:
                                    raise AssertionError("scope after seal")
                loss = losses[0] + .5*losses[1] + .01*q.weight_delta().square().mean()
                loss.backward()
                vals.extend(t.grad.detach().clone() for t in (q.V,q.U,*leaves))
                if fast:
                    assert [s.backward_nodes for s in stats] == [3,0,3]
                    assert [s.hits for s in stats] == [2,3,3]
                    assert p.sealed and not p.poisoned
                    assert p.peak_bytes == q.W_target.numel()*4
                return vals,stats
            finally:
                if p is not None:
                    p.close()
                    assert p.closed and p.resident_bytes == 0
                    assert "weight_delta" not in q.__dict__
        a,_ = run(False)
        b,ss = run(True)
        compare(a,b,counts,f"stress/{seed}/{scale}/{target}")
        cases += 1
        nodes += sum(s.backward_nodes for s in ss)
        hits += sum(s.hits for s in ss)
    return dict(cases=cases,exact_checks=counts,joint_factor_nodes=nodes,hits=hits)


def campaign(width, passes, regime, cap_mode, inner, stoch, adv,
             inner_cache, cal_cache, root):
    seed = 4111001 + root + width*100 + passes*10000
    a, aa = build(width,regime,seed,True,"pgd_step",passes,
                  "full" if cap_mode=="zero" else cap_mode,2)
    b, bb = build(width,regime,seed,True,"pgd_step",passes,
                  "full" if cap_mode=="zero" else cap_mode,2)
    assert not a.cache_pgd_final_residuals and not b.cache_pgd_final_residuals
    b.cache_pgd_final_residuals = True
    for tr in (a,b):
        tr.cache_pgd_mask_residuals = inner_cache
        tr.cache_calibration_residuals = cal_cache
        tr.pgd.cfg = dataclasses.replace(tr.pgd.cfg,inner_steps=inner)
        tr.cfg = dataclasses.replace(tr.cfg,adversary=tr.pgd.cfg,
                    coeff_stoch=.5 if stoch else 0., coeff_adv=.5 if adv else 0.,
                    null_calibration=NullCalibrationConfig(use_tolerance_weights=cal_cache,
                        n_rotations_train=1,calibrate_every=1))
        if cap_mode=="zero":
            tr.residual_cache_max_bytes=0
    counts = {"tensors":0,"scalars":0}
    ascent = [[],[]]
    raw = [{},{}]
    for j,tr in enumerate((a,b)):
        original = tr.pgd.ascent_step
        def tap(grads, *, lr, j=j, original=original):
            ascent[j].append({k:v.detach().clone() for k,v in grads.items()})
            return original(grads,lr=lr)
        tr.pgd.ascent_step=tap
        for tag, module in (("bpd",tr.bpd_model),("ci",tr.ci_net)):
            for name,p in module.named_parameters():
                if p.requires_grad:
                    def capture(param,j=j,key=tag+"/"+name):
                        raw[j][key]=param.grad.detach().clone()
                    p.register_post_accumulate_grad_hook(capture)

    def eq(x,y,path):
        compare(x,y,counts,path)

    def state():
        for key in ("bpd_model","ci_net","optimizer"):
            eq(getattr(a,key).state_dict(),getattr(b,key).state_dict(),key)
        for key in ("bpd_model","ci_net"):
            eq({k:p.grad for k,p in getattr(a,key).named_parameters()},
               {k:p.grad for k,p in getattr(b,key).named_parameters()},key+"/grad")
        for key in ("sources","exp_avg","exp_avg_sq","step_count"):
            eq(getattr(a.pgd,key),getattr(b.pgd,key),"pgd/"+key)
        eq(a.pass_tolerances,b.pass_tolerances,"tolerances")
        eq(a.rng.get_state(),b.rng.get_state(),"rng")

    state()
    eq(a.run_faithfulness_warmup(1),b.run_faithfulness_warmup(1),"warmup")
    state()
    gen=torch.Generator().manual_seed(seed+100)
    rows=[]
    for step in range(2):
        inp=torch.randint(1,11,(2,16),generator=gen)
        xa=aa.make_batch_with_clean_cache(inp,passes)
        xb=bb.make_batch_with_clean_cache(inp,passes)
        eq(xa.clean_states,xb.clean_states,"clean_states")
        rng=torch.get_rng_state()
        metrics,secs,after=[None]*2,[None]*2,[None]*2
        for j in ((0,1) if step==0 else (1,0)):
            ascent[j].clear()
            raw[j].clear()
            torch.set_rng_state(rng)
            start=time.perf_counter()
            metrics[j]=(a,b)[j].train_step((xa,xb)[j],step)
            secs[j]=time.perf_counter()-start
            after[j]=torch.get_rng_state()
        eq(*metrics,"metrics")
        eq(*after,"ambient_rng")
        eq(*ascent,"actual_pgd_source_gradients")
        eq(*raw,"preclip_joint_gradients")
        assert raw[0]
        assert len(ascent[0])==(inner+1 if adv and regime=="closed_loop" else 0)
        state()
        assert not a.residual_final_pgd_stats and a.residual_final_pgd_query is None
        final=adv and regime=="closed_loop" and cap_mode!="zero"
        assert len(b.residual_final_pgd_stats)==int(final)
        n=passes if regime=="closed_loop" else passes//2
        planned=n*(int(stoch)+int(adv)+
                    (inner if adv and inner_cache and regime=="closed_loop" else 0))
        if final:
            s=b.residual_final_pgd_stats[0]
            q=b.residual_final_pgd_query
            assert q["factor_nodes_before"]==q["factor_nodes_after"]==0
            assert q["forward_sealed"]
            ws=[b.bpd_model.wrappers[k] for k in b.bpd_model.module_order]
            plan=select([Site.residual(w.d_in,w.d_out,w.C,planned) for w in ws],
                        b.residual_cache_max_bytes,"exact")
            k=len(plan.indices)
            assert b.residual_cache_plans[0]["calls"]==planned
            assert tuple(b.residual_cache_plans[0]["indices"])==plan.indices
            assert s.calls==s.independent_calls==k*n
            assert s.mask_only_calls==0 and s.bypasses==0
            assert 0<s.backward_nodes<=k*n
            assert s.misses==(0 if stoch or (inner_cache and inner) else k)
            assert s.hits==k*n-s.misses
            assert s.cross_scope_hits==(k*n if stoch or (inner_cache and inner) else 0)
            assert q["resident_bytes"]==plan.size<=b.residual_cache_max_bytes
            receipt=b.residual_step_receipt
            assert receipt["sealed"] and receipt["closed"] and not receipt["poisoned"]
            assert receipt["resident_bytes"]==0 and receipt["peak_bytes"]==plan.size
        else:
            assert b.residual_final_pgd_query is None
        assert b._step_residual_pool is None and not b._residual_step_active
        assert all("weight_delta" not in w.__dict__ for w in b.bpd_model.wrappers.values())
        with torch.no_grad():
            ea,ca,ta=a.evaluate(xa)
            eb,cb,tb=b.evaluate(xb)
            eq(ea,eb,"evaluation")
            eq(ca,cb,"eval_ci")
            eq(ta.readouts,tb.readouts,"readouts")
            eq(ta.boundary_states,tb.boundary_states,"boundaries")
        rows.append(dict(step=step,seconds=secs,planned_calls=planned,
            final=[dataclasses.asdict(s) for s in b.residual_final_pgd_stats],
            source_query=b.residual_final_pgd_query,pool=b.residual_step_receipt,
            reference_reconstructions=4*n*(int(stoch)+int(adv)+
                (inner if adv and regime=="closed_loop" else 0))-
                sum(s.hits for s in a.residual_cache_stats+a.residual_pgd_cache_stats),
            cached_reconstructions=4*n*(int(stoch)+int(adv)+
                (inner if adv and regime=="closed_loop" else 0))-
                sum(s.hits for s in b.residual_cache_stats+b.residual_pgd_cache_stats+
                    b.residual_final_pgd_stats)))
    return dict(width=width,passes=passes,regime=regime,cap=cap_mode,inner=inner,
                stoch=stoch,adv=adv,inner_cache=inner_cache,calibration_cache=cal_cache,
                seed=seed,exact_checks=counts,steps=rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--passes",type=int,default=6)
    p.add_argument("--wide",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    # regime, cap, K, stochastic, adversary, inner reuse, clean calibration reuse
    configs=[
        ("closed_loop","binding",2,True,True,True,False),
        ("closed_loop","full",2,True,True,True,True),
        ("closed_loop","full",0,True,True,True,False),
        ("closed_loop","full",0,False,True,False,False),
        ("closed_loop","binding",2,False,True,True,False),
        ("closed_loop","binding",2,True,True,False,False),
        ("closed_loop","zero",2,True,True,True,False),
        ("closed_loop","binding",2,True,False,True,False),
        ("clean_window","binding",2,True,True,True,True),
    ]
    if args.wide:
        configs=configs[:2]
    stress_result=None if args.wide else stress()
    rows=[campaign(args.width,args.passes,*c,i) for i,c in enumerate(configs)]
    for row in rows:
        print("FUSE11_ROW",json.dumps(row,allow_nan=False),flush=True)
    result=dict(torch=str(torch.__version__),stress=stress_result,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("FUSE11_DONE",json.dumps(dict(campaigns=len(rows),paired_steps=2*len(rows),
          native_exact=sum(r["exact_checks"]["tensors"] for r in rows),stress=stress_result)))


if __name__=="__main__":
    main()
