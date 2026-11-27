"""FUSE-10 native CPU gate: exact calibration and training state parity."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch

from mechinterp.bpd.components import TiedComponentLinear
from mechinterp.bpd.config import NullCalibrationConfig
from .calibration_cache import calibration_residuals
from .validate_fuse09_native import build


def lifecycle():
    torch.manual_seed(4010000)
    w = TiedComponentLinear(torch.nn.Linear(8, 12), 3, init_svd=False)
    w.update_running_means = True
    named = [("site", w)]
    cap = w.W_target.numel()*4
    with torch.no_grad():
        native = w.weight_delta()
        with calibration_residuals(named, calls=3, max_bytes=cap) as st:
            assert torch.equal(w.weight_delta(), native)
            assert torch.equal(w.weight_delta(), native)
            assert st["misses"] == st["hits"] == 1
        assert st["closed"] and st["resident_bytes"] == 0
        assert "weight_delta" not in w.__dict__
        with calibration_residuals(named, calls=1, max_bytes=cap) as st:
            assert not st["selected"]
        with calibration_residuals(named, calls=4, max_bytes=0) as st:
            assert not st["selected"]
        try:
            with calibration_residuals(named, calls=3, max_bytes=cap) as st:
                w.weight_delta()
                w.V.add_(1)
                w.weight_delta()
        except RuntimeError as exc:
            assert "mutated" in str(exc)
        else:
            raise AssertionError("mutation escaped")
        assert st["closed"] and "weight_delta" not in w.__dict__
        try:
            with calibration_residuals(named, calls=3, max_bytes=cap):
                w.weight_delta()
                raise ValueError("injected")
        except ValueError as exc:
            assert str(exc) == "injected"
        assert "weight_delta" not in w.__dict__
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with calibration_residuals(named, calls=3, max_bytes=cap) as st:
                assert not st["selected"]
    with calibration_residuals(named, calls=3, max_bytes=cap) as st:
        assert not st["selected"]
        assert w.weight_delta().requires_grad
    with torch.inference_mode():
        with calibration_residuals(named, calls=3, max_bytes=cap) as st:
            assert not st["selected"]
    return {"lifecycle": "pass"}


def campaign(width, passes, regime, cap_mode, combined, root):
    seed = 4010001 + root
    a, aa = build(width, regime, seed, combined, "pgd_step", passes, cap_mode, 2)
    b, bb = build(width, regime, seed, combined, "pgd_step", passes, cap_mode, 2)
    assert not a.cache_calibration_residuals and not b.cache_calibration_residuals
    b.cache_calibration_residuals = True
    for tr in (a, b):
        tr.cfg = dataclasses.replace(
            tr.cfg, null_calibration=NullCalibrationConfig(
                use_tolerance_weights=True, n_rotations_train=1, calibrate_every=1))
    counts = {"tensors": 0, "scalars": 0}

    def eq(x, y, path):
        if isinstance(x, torch.Tensor):
            assert isinstance(y, torch.Tensor) and x.dtype == y.dtype, path
            assert torch.equal(x, y), path
            if x.is_floating_point() or x.is_complex():
                assert torch.isfinite(x).all(), path
            counts["tensors"] += 1
        elif dataclasses.is_dataclass(x):
            eq(dataclasses.asdict(x), dataclasses.asdict(y), path)
        elif isinstance(x, dict):
            assert x.keys() == y.keys(), path
            for k in x:
                eq(x[k], y[k], f"{path}/{k}")
        elif isinstance(x, (tuple, list)):
            assert len(x) == len(y), path
            for i, (u, v) in enumerate(zip(x, y)):
                eq(u, v, f"{path}/{i}")
        else:
            assert x == y, path
            if isinstance(x, float):
                assert math.isfinite(x), path
            counts["scalars"] += 1

    def state():
        for key in ("bpd_model", "ci_net", "optimizer"):
            eq(getattr(a, key).state_dict(), getattr(b, key).state_dict(), key)
        for key in ("bpd_model", "ci_net"):
            eq({k:p.grad for k,p in getattr(a,key).named_parameters()},
               {k:p.grad for k,p in getattr(b,key).named_parameters()}, key+"/grads")
        for key in ("sources", "exp_avg", "exp_avg_sq", "step_count"):
            eq(getattr(a.pgd, key), getattr(b.pgd, key), "pgd/"+key)
        eq(a.pass_tolerances, b.pass_tolerances, "tolerances")
        eq(a.rng.get_state(), b.rng.get_state(), "rng")

    state()
    eq(a.run_faithfulness_warmup(1), b.run_faithfulness_warmup(1), "warmup")
    state()
    steps = []
    gen = torch.Generator().manual_seed(seed+100)
    for step in range(2):
        inp = torch.randint(1, 11, (2,16), generator=gen)
        xa = aa.make_batch_with_clean_cache(inp, passes)
        xb = bb.make_batch_with_clean_cache(inp, passes)
        eq(xa.clean_states, xb.clean_states, "clean_states")
        rng = torch.get_rng_state()
        metrics, elapsed, end_rng = [None]*2, [None]*2, [None]*2
        for j in ((0,1) if step == 0 else (1,0)):
            torch.set_rng_state(rng)
            start = time.perf_counter()
            metrics[j] = (a,b)[j].train_step((xa,xb)[j], step)
            elapsed[j] = time.perf_counter()-start
            end_rng[j] = torch.get_rng_state()
        eq(*metrics, "metrics")
        eq(*end_rng, "ambient_rng")
        state()
        assert not a.residual_calibration_stats
        # Full tolerance clean capture + ordinary training clean capture.
        receipts = b.residual_calibration_stats
        assert len(receipts) == 2, receipts
        expected = [passes, passes if regime == "closed_loop" else passes//2]
        for st, n in zip(receipts, expected):
            k = len(st["selected"])
            assert st["planned_calls"] == n
            assert k == 4 if cap_mode == "full" else 0 < k < 4
            assert st["calls"] == k*n and st["misses"] == k
            assert st["hits"] == k*(n-1) and st["bypasses"] == 0
            assert st["reconstructions"] == k
            assert st["peak_bytes"] == st["planned_bytes"] <= b.residual_cache_max_bytes
            assert st["resident_bytes"] == 0 and st["closed"]
        assert b._step_residual_pool is None
        assert all("weight_delta" not in w.__dict__ for w in b.bpd_model.wrappers.values())
        with torch.no_grad():
            ev_a, ci_a, tr_a = a.evaluate(xa)
            ev_b, ci_b, tr_b = b.evaluate(xb)
            eq(ev_a, ev_b, "eval")
            eq(ci_a, ci_b, "eval_ci")
            eq(tr_a.readouts, tr_b.readouts, "readouts")
            eq(tr_a.boundary_states, tr_b.boundary_states, "boundaries")
        steps.append({"step":step, "seconds":elapsed, "calibration":receipts})
    return dict(width=width, passes=passes, regime=regime, cap=cap_mode,
                combined=combined, seed=seed, exact_checks=counts, steps=steps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--passes", type=int, default=6)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    rows = [campaign(args.width, args.passes, regime, cap, combined, i)
            for i, (regime, cap, combined) in enumerate([
                ("closed_loop", "binding", False),
                ("closed_loop", "full", True),
                ("clean_window", "binding", True),
                ("clean_window", "full", False),
            ])]
    receipt = {"torch":str(torch.__version__), "lifecycle":lifecycle(), "rows":rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")
    print("FUSE10_DONE", json.dumps({"campaigns":len(rows), "paired_steps":2*len(rows),
                                    "exact_checks":sum(r["exact_checks"]["tensors"] for r in rows)}))


if __name__ == "__main__":
    main()
