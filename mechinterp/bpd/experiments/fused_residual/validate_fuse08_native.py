"""FUSE-08: actual native trainer with binding caps and step-wide primal pool."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
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


def build(width, regime, seed, fast, policy):
    torch.manual_seed(seed)
    model = build_mini_trm(hidden_size=width, num_heads=4, l_layers=2,
                           h_cycles=1, l_cycles=2, puzzle_emb_len=4,
                           seq_len=16, seed=seed)
    paths = [f"layers.{i}.{name}" for i in range(2) for name in ("o_proj", "down_proj")]
    sites = dict(zip(paths, (width//32, width//16, 3*width//32, width//8)))
    cfg = BPDConfig(
        c_per_module=sites, n_passes=6, n_steps=4, batch_size=2, seq_len=20,
        seed=seed, init_svd=False, faithfulness_warmup_steps=1,
        coeff_faith=1e5, coeff_stoch=.5, coeff_adv=.5,
        readout_policy="readout_or_boundary", rollout_mode=regime, window_steps=3,
        ci=CIConfig(variant="state_only", d_model=32, n_blocks=1,
                    n_heads=4, mlp_hidden=64, max_passes=6, max_seq_len=20),
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        adversary=AdversaryConfig(inner_steps=2),
        null_calibration=NullCalibrationConfig(use_tolerance_weights=False),
    )
    bpd = BPDModel(model, sites, max_passes=6, ablation=cfg.ablation, init_svd=False)
    adapter = TRMBPDAdapter(model)
    trainer = BPDTrainer(
        bpd, cfg, adapter.rollout, clean_window_state_fn=adapter.sample_clean_window_state,
        visible_dim=adapter.visible_dim, prompt_dim=adapter.prompt_dim,
        device="cpu", cache_single_backward_residuals=fast,
        residual_cache_max_bytes=16 * width**2,
        residual_cache_policy=policy,
    )
    return trainer, adapter


def campaign(width, regime, policy):
    seed = 4001001 + width + (10000 if regime == "clean_window" else 0)
    a, aa = build(width, regime, seed, False, policy)
    b, bb = build(width, regime, seed, True, policy)
    cap = b.residual_cache_max_bytes
    names = b.bpd_model.module_order
    ws = [b.bpd_model.wrappers[n] for n in names]
    passes = 6 if regime == "closed_loop" else 3
    branches = 1 if regime == "closed_loop" else 2
    calls = passes*branches if policy == "step_exact" else passes
    sites = [Site.residual(w.d_in,w.d_out,w.C,calls) for w in ws]
    opt = oracle(sites,cap)
    selected_names = [names[i] for i in opt[2]]
    assert 0 < len(selected_names) < len(names) and opt[1] == cap
    execution_order = [names.index(f"layers.{i}.{n}") for i in range(2) for n in ("o_proj","down_proj")]
    ordered = [sites[i] for i in execution_order]
    ff = select(ordered,cap,"first")
    assert opt[0] > ff.value
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
    for step in range(4):
        inp = torch.randint(1, 11, (2, 16), generator=gen)
        a.bpd_model.clear_schedules()
        b.bpd_model.clear_schedules()
        xa = aa.make_batch_with_clean_cache(inp, 6)
        xb = bb.make_batch_with_clean_cache(inp, 6)
        compare(xa.clean_states, xb.clean_states, "clean_cache")
        # Keep the ambient torch RNG identical in addition to each trainer generator.
        rng = torch.get_rng_state()
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
        assert ma.mode == regime and mb.mode == regime
        assert not a.residual_cache_stats
        assert len(b.residual_cache_stats) == branches
        assert len(b.residual_cache_plans) == (1 if policy == "step_exact" else branches)
        for p in b.residual_cache_plans:
            assert p["status"] == "selected" and p["calls"] == calls
            assert (p["value"],p["size"],tuple(p["indices"])) == opt
            assert p["names"] == selected_names
        stats = [dataclasses.asdict(s) for s in b.residual_cache_stats]
        n = len(selected_names)
        for i, st in enumerate(stats):
            misses = n if policy == "exact" or i == 0 else 0
            assert (st["calls"],st["misses"],st["hits"],st["reconstructions"]) == (
                n*passes,misses,n*passes-misses,misses), st
            assert st["resident_bytes"] == 0 and not st["consumed"]
            assert st["bypasses"] == st["cap_rejections"] == 0
            assert 0 < st["backward_nodes"] <= st["calls"], st
            assert st["peak_bytes"] == opt[1], st
            if policy == "step_exact":
                assert st["storage_checks"] == st["calls"]
                assert st["cross_scope_hits"] == (0 if i==0 else st["calls"])
                assert st["pool_bytes_after_scope"] == opt[1]
                assert st["slots"] == stats[0]["slots"]
        assert b._step_residual_pool is None and not b._residual_step_active
        if policy == "step_exact":
            assert b.residual_step_receipt == {
                "peak_bytes":opt[1],"resident_bytes":0,"scopes":branches,
                "sealed":True,"poisoned":False,"closed":True,
            }, b.residual_step_receipt
        else:
            assert b.residual_step_receipt is None
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
            pa = a.run_masked_rollout(xa, sa, n_passes=6)
            pb = b.run_masked_rollout(xb, sb, n_passes=6)
            compare(pa.readouts, pb.readouts, "masked_readout")
            compare(pa.boundary_states, pb.boundary_states, "masked_boundary")
            for k in pa.readouts:
                checks["greedy"] += 1
                checks["greedy_failures"] += int(not torch.equal(
                    pa.readouts[k].argmax(-1), pb.readouts[k].argmax(-1)))
        steps.append(dict(step=step, reference=dataclasses.asdict(ma),
                          cached=dataclasses.asdict(mb), cache_stats=stats,
                          plans=b.residual_cache_plans, step_pool=b.residual_step_receipt,
                          reference_seconds=elapsed_a, cached_seconds=elapsed_b,
                          reconstruction_count=4*passes*branches-sum(st["hits"] for st in stats)))
    return dict(width=width, regime=regime, policy=policy, seed=seed, checks=checks,
                maxima=maxima, failures=failures, steps=steps, cap_bytes=cap,
                planned_calls=calls, names=names, selected_names=selected_names,
                optimal_value=opt[0],first_fit_value=ff.value,
                selected_bytes=opt[1],first_fit_bytes=ff.size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    rows = [campaign(width, regime, policy) for width in (64,128,256)
            for regime in ("closed_loop","clean_window")
            for policy in ("exact","step_exact")]
    result = dict(torch=torch.__version__, threads=torch.get_num_threads(), rows=rows)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    import statistics
    print(json.dumps([dict(width=r["width"], regime=r["regime"], policy=r["policy"], checks=r["checks"],
                           maxima=r["maxima"], failures=r["failures"], selected_names=r["selected_names"],
                           cap_bytes=r["cap_bytes"], optimal_value=r["optimal_value"], first_fit_value=r["first_fit_value"],
                           reconstruction_count=[x["reconstruction_count"] for x in r["steps"]],
                           cross_scope_hits=[sum(st.get("cross_scope_hits",0) for st in x["cache_stats"]) for x in r["steps"]],
                           reference_ms=1000*statistics.median(x["reference_seconds"] for x in r["steps"]),
                           cached_ms=1000*statistics.median(x["cached_seconds"] for x in r["steps"])) for r in rows], indent=2))
    assert not any(r["checks"]["failures"] or r["checks"]["greedy_failures"] for r in rows)


if __name__ == "__main__":
    main()
