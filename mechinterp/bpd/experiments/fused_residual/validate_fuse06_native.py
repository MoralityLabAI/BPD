"""FUSE-06: full-package native BPDTrainer parity gate (CPU, opt-in)."""
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

ATOL, RTOL = 5e-5, 5e-4


def build(width, regime, seed, fast):
    torch.manual_seed(seed)
    model = build_mini_trm(hidden_size=width, num_heads=4, l_layers=2,
                           h_cycles=1, l_cycles=2, puzzle_emb_len=4,
                           seq_len=16, seed=seed)
    sites = {f"layers.{i}.{name}": width // 4 for i in range(2)
             for name in ("o_proj", "down_proj")}
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
        residual_cache_max_bytes=16 * 1024**2,
    )
    return trainer, adapter


def campaign(width, regime):
    seed = 3803001 + width + (10000 if regime == "clean_window" else 0)
    a, aa = build(width, regime, seed, False)
    b, bb = build(width, regime, seed, True)
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
        t = time.perf_counter()
        ma = a.train_step(xa, step)
        elapsed_a = time.perf_counter()-t
        torch.set_rng_state(rng)
        t = time.perf_counter()
        mb = b.train_step(xb, step)
        elapsed_b = time.perf_counter()-t
        compare(ma, mb, "metrics")
        assert ma.mode == regime and mb.mode == regime
        assert not a.residual_cache_stats
        expected_scopes = 1 if regime == "closed_loop" else 2
        expected_passes = 6 if regime == "closed_loop" else 3
        assert len(b.residual_cache_stats) == expected_scopes
        stats = [dataclasses.asdict(s) for s in b.residual_cache_stats]
        for s in stats:
            assert (s["calls"], s["misses"], s["hits"]) == (
                4*expected_passes, 4, 4*(expected_passes-1)), s
            assert s["resident_bytes"] == 0 and not s["consumed"]
            assert s["bypasses"] == s["cap_rejections"] == 0
            # Autograd may legitimately prune an unused node; verify nonzero work.
            assert 0 < s["backward_nodes"] <= s["calls"], s
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
                          reference_seconds=elapsed_a, cached_seconds=elapsed_b))
    return dict(width=width, regime=regime, seed=seed, checks=checks,
                maxima=maxima, failures=failures, steps=steps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    rows = [campaign(width, regime) for width in (64,128)
            for regime in ("closed_loop","clean_window")]
    result = dict(torch=torch.__version__, threads=torch.get_num_threads(), rows=rows)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps([dict(width=r["width"], regime=r["regime"], checks=r["checks"],
                           maxima=r["maxima"], failures=r["failures"]) for r in rows], indent=2))
    assert not any(r["checks"]["failures"] or r["checks"]["greedy_failures"] for r in rows)


if __name__ == "__main__":
    main()
