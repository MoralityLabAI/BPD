"""FUSE-09: matched actual trainer with mask-only PGD and wider capacities."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import resource
import statistics
import time
from pathlib import Path

import torch

from mechinterp.bpd.adapters.trm import TRMBPDAdapter, build_mini_trm
from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import (
    AblationConfig, AdversaryConfig, BPDConfig, CIConfig, NullCalibrationConfig,
)
from mechinterp.bpd.losses import deterministic_ci_schedules
from mechinterp.bpd.trainer import BPDTrainer
from .admission import Site, select
from .validate_fuse07 import oracle

ATOL, RTOL = 5e-5, 5e-4


def build(width, regime, seed, fast, policy, n_passes, cap_mode, n_steps):
    torch.manual_seed(seed)
    model = build_mini_trm(hidden_size=width, num_heads=4, l_layers=2,
                           h_cycles=1, l_cycles=2, puzzle_emb_len=4,
                           seq_len=16, seed=seed)
    paths = [f"layers.{i}.{name}" for i in range(2) for name in ("o_proj", "down_proj")]
    sites = dict(zip(paths, (width//32, width//16, 3*width//32, width//8)))
    cfg = BPDConfig(
        c_per_module=sites, n_passes=n_passes, n_steps=n_steps, batch_size=2, seq_len=20,
        seed=seed, init_svd=False, faithfulness_warmup_steps=1,
        coeff_faith=1e5, coeff_stoch=.5, coeff_adv=.5,
        readout_policy="readout_or_boundary", rollout_mode=regime, window_steps=n_passes//2,
        ci=CIConfig(variant="state_only", d_model=32, n_blocks=1,
                    n_heads=4, mlp_hidden=64, max_passes=n_passes, max_seq_len=20),
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        adversary=AdversaryConfig(inner_steps=2),
        null_calibration=NullCalibrationConfig(use_tolerance_weights=False),
    )
    bpd = BPDModel(model, sites, max_passes=n_passes, ablation=cfg.ablation, init_svd=False)
    adapter = TRMBPDAdapter(model)
    cap = sum(w.W_target.numel()*4 for name,w in bpd.wrappers.items()
              if cap_mode=="full" or name.startswith("layers.0."))
    trainer = BPDTrainer(
        bpd, cfg, adapter.rollout, clean_window_state_fn=adapter.sample_clean_window_state,
        visible_dim=adapter.visible_dim, prompt_dim=adapter.prompt_dim,
        device="cpu", cache_single_backward_residuals=fast,
        residual_cache_max_bytes=cap,
        residual_cache_policy="step_exact" if policy=="pgd_step" else policy,
        cache_pgd_mask_residuals=bool(fast and policy=="pgd_step"),
    )
    return trainer, adapter


def campaign(width, regime, policy, n_passes, cap_mode, root, stage):
    n_steps = 4 if stage=="small" else 2
    seed = (4001001+width if stage=="small" else
            4101001+width+n_passes*10000+(100000 if regime=="clean_window" else 0)
            +(1000000 if cap_mode=="full" else 0)+root)
    a, aa = build(width, regime, seed, False, policy, n_passes, cap_mode, n_steps)
    b, bb = build(width, regime, seed, True, policy, n_passes, cap_mode, n_steps)
    cap = b.residual_cache_max_bytes
    names = b.bpd_model.module_order
    ws = [b.bpd_model.wrappers[n] for n in names]
    passes = n_passes if regime=="closed_loop" else n_passes//2
    branches = 1 if regime=="closed_loop" else 2
    inner = 2 if regime=="closed_loop" else 0
    step_pool = policy in ("step_exact","pgd_step")
    mask_inner = inner if policy=="pgd_step" else 0
    calls = passes*(branches+mask_inner) if step_pool else passes
    sites = [Site.residual(w.d_in,w.d_out,w.C,calls) for w in ws]
    opt = oracle(sites,cap)
    execution_order = [names.index(f"layers.{i}.{n}") for i in range(2) for n in ("o_proj","down_proj")]
    ordered = [sites[i] for i in execution_order]
    ff = select(ordered,cap,"first")
    chosen = tuple(sorted(execution_order[k] for k in ff.indices)) if policy=="first_fit" else opt[2]
    selected_names = [names[k] for k in chosen]
    selected_size = sum(sites[k].size for k in chosen)
    assert selected_size == cap
    assert (0<len(chosen)<len(names)) if cap_mode=="binding" else len(chosen)==len(names)
    assert opt[0]>=ff.value
    if cap_mode=="binding":assert opt[0]>ff.value
    # Audit exactly what native PGD ascent consumes, including the final
    # source-only retained-graph query. Copies are detached and outside timing
    # interpretation (same instrumentation on both arms).
    traces=[[],[]]
    def tap(trainer, dest):
        original=trainer.pgd.ascent_step
        def wrapped(grads, *, lr):
            dest.append({k:v.detach().clone() for k,v in grads.items()})
            return original(grads,lr=lr)
        trainer.pgd.ascent_step=wrapped
    tap(a,traces[0]);tap(b,traces[1])
    checks = {"tensors": 0, "exact": 0, "scalars": 0, "failures": 0,
              "greedy": 0, "greedy_failures": 0}
    maxima = {}
    failures = []

    def compare(u, v, path):
        if isinstance(u, torch.Tensor):
            assert isinstance(v, torch.Tensor) and u.shape == v.shape, path
            checks["tensors"] += 1
            checks["exact"] += int(torch.equal(u, v))
            if u.is_floating_point() or u.is_complex():
                finite = bool(torch.isfinite(u).all() and torch.isfinite(v).all())
                equal = finite and bool(torch.allclose(u, v, atol=ATOL, rtol=RTOL))
                if finite and u.numel():
                    group = path.split("/")[0]
                    maxima[group] = max(maxima.get(group, 0.), float((u-v).abs().max()))
            else:
                equal = bool(torch.equal(u, v))
        elif dataclasses.is_dataclass(u):
            compare(dataclasses.asdict(u), dataclasses.asdict(v), path)
            return
        elif isinstance(u, dict):
            assert u.keys() == v.keys(), path
            for k in u:
                compare(u[k], v[k], f"{path}/{k}")
            return
        elif isinstance(u, (tuple, list)):
            assert len(u) == len(v), path
            for i, (x, y) in enumerate(zip(u, v)):
                compare(x, y, f"{path}/{i}")
            return
        elif isinstance(u, float):
            checks["scalars"] += 1
            equal = math.isfinite(u) and math.isfinite(v) and abs(u-v) <= ATOL+RTOL*abs(v)
        else:
            checks["scalars"] += 1
            equal = u == v
        if not equal:
            checks["failures"] += 1
            if len(failures) < 32:
                failures.append(path)

    def state():
        compare(a.bpd_model.state_dict(), b.bpd_model.state_dict(), "bpd_state")
        compare(a.ci_net.state_dict(), b.ci_net.state_dict(), "ci_state")
        compare(a.optimizer.state_dict(), b.optimizer.state_dict(), "optimizer")
        for tag, x, y in (("component_grad", a.bpd_model, b.bpd_model),
                          ("ci_grad", a.ci_net, b.ci_net)):
            compare({k:p.grad for k,p in x.named_parameters()},
                    {k:p.grad for k,p in y.named_parameters()}, tag)
        for name in ("sources", "exp_avg", "exp_avg_sq", "step_count"):
            compare(getattr(a.pgd,name), getattr(b.pgd,name), "pgd/"+name)
        compare(a.rng.get_state(), b.rng.get_state(), "trainer_rng")

    state()
    warm = (a.run_faithfulness_warmup(1), b.run_faithfulness_warmup(1))
    compare(*warm, "warmup")
    state()
    steps = []
    gen = torch.Generator().manual_seed(seed+100)
    for step in range(n_steps):
        inp = torch.randint(1, 11, (2, 16), generator=gen)
        a.bpd_model.clear_schedules()
        b.bpd_model.clear_schedules()
        xa = aa.make_batch_with_clean_cache(inp, n_passes)
        xb = bb.make_batch_with_clean_cache(inp, n_passes)
        compare(xa.clean_states, xb.clean_states, "clean_cache")
        # Keep the ambient torch RNG identical in addition to each trainer generator.
        rng = torch.get_rng_state()
        for t in traces:t.clear()
        metrics = [None,None]; elapsed = [None,None]; after_rng = [None,None]
        for j in ((0,1) if step%2==0 else (1,0)):
            torch.set_rng_state(rng)
            t = time.perf_counter()
            metrics[j] = (a,b)[j].train_step((xa,xb)[j],step)
            elapsed[j] = time.perf_counter()-t
            after_rng[j] = torch.get_rng_state()
        compare(after_rng[0],after_rng[1],"ambient_rng")
        ma,mb = metrics; elapsed_a,elapsed_b = elapsed
        compare(ma, mb, "metrics")
        assert len(traces[0])==len(traces[1])==(inner+1 if regime=="closed_loop" else 0)
        compare(traces[0],traces[1],"pgd_ascent_gradients")
        assert ma.mode == regime and mb.mode == regime
        assert not a.residual_cache_stats and not a.residual_pgd_cache_stats
        assert len(b.residual_cache_stats)==branches
        assert len(b.residual_pgd_cache_stats)==mask_inner
        expected_plans=0 if policy=="first_fit" else (1 if step_pool else branches)
        assert len(b.residual_cache_plans)==expected_plans
        for p in b.residual_cache_plans:
            assert p["status"]=="selected" and p["calls"]==calls
            assert (p["value"],p["size"],tuple(p["indices"]))==opt
            assert p["names"]==selected_names
        stats=[dataclasses.asdict(st) for st in b.residual_cache_stats]
        pgd_stats=[dataclasses.asdict(st) for st in b.residual_pgd_cache_stats]
        n=len(chosen)
        for i,st in enumerate(stats+pgd_stats):
            is_pgd=i>=branches
            misses=n if not step_pool or i==0 else 0
            hit=n*passes-misses
            bypass=(4-n)*passes if policy=="first_fit" else 0
            assert (st["calls"],st["misses"],st["hits"],st["reconstructions"])==(
                n*passes+bypass,misses,hit,misses+bypass),st
            assert st["resident_bytes"]==0 and not st["consumed"]
            assert st["bypasses"]==st["cap_rejections"]==bypass
            assert st["peak_bytes"]==selected_size,st
            if is_pgd:assert st["backward_nodes"]==0
            else:assert 0<st["backward_nodes"]<=n*passes,st
            if step_pool:
                assert st["storage_checks"]==n*passes
                assert st["cross_scope_hits"]==(0 if i==0 else n*passes)
                assert st["pool_bytes_after_scope"]==selected_size
                assert st["slots"]==stats[0]["slots"]
                assert st["mask_only_calls"]==(n*passes if is_pgd else 0)
                assert st["independent_calls"]==(0 if is_pgd else n*passes)
        assert b._step_residual_pool is None and not b._residual_step_active
        if step_pool:
            assert b.residual_step_receipt=={
                "peak_bytes":selected_size,"resident_bytes":0,"scopes":branches+mask_inner,
                "sealed":True,"poisoned":False,"closed":True,
            },b.residual_step_receipt
        else:assert b.residual_step_receipt is None
        assert not a.residual_cache_plans and a.residual_step_receipt is None
        assert all("weight_delta" not in w.__dict__
                   for w in b.bpd_model.wrappers.values())
        state()
        with torch.no_grad():
            ea, cia, ca = a.evaluate(xa)
            eb, cib, cb = b.evaluate(xb)
            compare(ea, eb, "eval_metrics")
            compare(cia, cib, "eval_ci")
            compare(ca.readouts, cb.readouts, "clean_readout")
            sa = deterministic_ci_schedules(cia.lower_leaky, delta_value=0.)
            sb = deterministic_ci_schedules(cib.lower_leaky, delta_value=0.)
            pa = a.run_masked_rollout(xa, sa, n_passes=n_passes)
            pb = b.run_masked_rollout(xb, sb, n_passes=n_passes)
            compare(pa.readouts, pb.readouts, "masked_readout")
            compare(pa.boundary_states, pb.boundary_states, "masked_boundary")
            for k in pa.readouts:
                checks["greedy"] += 1
                checks["greedy_failures"] += int(not torch.equal(
                    pa.readouts[k].argmax(-1), pb.readouts[k].argmax(-1)))
        steps.append(dict(step=step, reference=dataclasses.asdict(ma),
                          cached=dataclasses.asdict(mb), cache_stats=stats, pgd_cache_stats=pgd_stats,
                          plans=b.residual_cache_plans, step_pool=b.residual_step_receipt,
                          reference_seconds=elapsed_a, cached_seconds=elapsed_b,
                          pgd_ascent_calls=len(traces[0]),
                          reconstruction_count=4*passes*(branches+inner)-sum(st["hits"] for st in stats+pgd_stats)))
    return dict(width=width, regime=regime, n_passes=n_passes, cap_mode=cap_mode, root=root,
                policy=policy, seed=seed, checks=checks, maxima=maxima, failures=failures,
                steps=steps, cap_bytes=cap, planned_calls=calls, names=names, selected_names=selected_names,
                optimal_value=opt[0],first_fit_value=ff.value, selected_bytes=selected_size,
                first_fit_bytes=ff.size, process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def summary(r):
    return dict(**{k:r[k] for k in ("width","regime","n_passes","cap_mode","root","policy","checks","maxima","failures","selected_names","cap_bytes","optimal_value","first_fit_value","selected_bytes","process_peak_rss_kib")},
                reconstruction_count=[x["reconstruction_count"] for x in r["steps"]],
                cross_scope_hits=[sum(st.get("cross_scope_hits",0) for st in x["cache_stats"]+x["pgd_cache_stats"]) for x in r["steps"]],
                mask_only_calls=[sum(st["mask_only_calls"] for st in x["pgd_cache_stats"]) for x in r["steps"]],
                pgd_ascent_calls=[x["pgd_ascent_calls"] for x in r["steps"]],
                reference_ms=1000*statistics.median(x["reference_seconds"] for x in r["steps"]),
                cached_ms=1000*statistics.median(x["cached_seconds"] for x in r["steps"]),
                raw_ms=[[1000*x["reference_seconds"],1000*x["cached_seconds"]] for x in r["steps"]])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--stage", choices=("small","wide"), required=True)
    p.add_argument("--width", type=int, choices=(512,1024))
    p.add_argument("--passes", type=int, choices=(6,12))
    p.add_argument("--regime", choices=("closed_loop","clean_window"))
    p.add_argument("--cap", choices=("binding","full"))
    args=p.parse_args()
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    if args.stage=="small":
        settings=[(w,"closed_loop","pgd_step",6,"binding",0) for w in (64,128,256)]
    else:
        assert args.width and args.passes and args.regime and args.cap
        policies=["first_fit","exact","step_exact"]+(["pgd_step"] if args.regime=="closed_loop" else [])
        settings=[(args.width,args.regime,pol,args.passes,args.cap,root)
                  for root in range(2) for pol in policies]
    rows=[]
    for setting in settings:
        r=campaign(*setting,args.stage);rows.append(r)
        print("FUSE09_ROW "+json.dumps(summary(r),allow_nan=False),flush=True)
        Path(args.output).write_text(json.dumps(dict(torch=torch.__version__,threads=torch.get_num_threads(),stage=args.stage,rows=rows),indent=2,allow_nan=False)+"\n")
    assert not any(r["checks"]["failures"] or r["checks"]["greedy_failures"] for r in rows)
    print("FUSE09_DONE "+json.dumps({"campaigns":len(rows),"paired_steps":sum(len(r["steps"]) for r in rows)}),flush=True)


if __name__=="__main__":main()
