"""FUSE-13 preregistered diagnostic gym. No production code or defaults changed.

CPU-only receipt audit by default; explicit --probe for experimental first-order
autocast residual candidates. GPU probe must be queued under VRAMorama.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "f0d042b25306e4efcbb1d35b266d2ab85fb7bb49"
ATOL, RTOL = 5e-5, 5e-4  # frozen FUSE-12 tolerances
ROOTS = (4013001, 4013002, 4013003, 4013004)
FRONTIER = ((2048, 48), (4096, 24), (4096, 48), (6144, 24), (8192, 12))
MODES = ("fp32", "tf32", "bf16", "fp16")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(directory=HERE):
    paths = [directory / "receipt_fuse12_cuda_validate.json",
             directory / "fuse12_cuda_receipts.jsonl"]
    val = json.loads(paths[0].read_text(encoding="utf-8"))
    rows = read_jsonl(paths[1])
    ids = [r["cell_id"] for r in rows]
    counts = Counter((r["precision_mode"], r["status"]) for r in rows)
    expected = {("fp32", "ok"): 38, ("tf32", "ok"): 4,
                ("bf16", "ok"): 2, ("fp16", "ok"): 2,
                ("bf16", "native_autocast_error"): 2,
                ("fp16", "native_autocast_error"): 2}
    checks = {"50_unique_cells": len(ids) == len(set(ids)) == 50,
              "status_matrix": dict(counts) == expected,
              "fp32_gate": val.get("fp32_gate_passed") is True}
    stress = {}
    for mode, row in {"fp32": val["cuda_fp32_stress"], **val["fuse12b_stress"]}.items():
        stress[mode] = {k: row[k] for k in ("cases", "tensors", "bitwise_exact",
                        "within_tol", "nonfinite_parity_tensors", "failures")}
        if mode in ("fp32", "tf32"):
            checks[mode + "_stress"] = (row["bitwise_exact"] == row["tensors"] == 1536
                                        and row["failures"] == row["nonfinite_parity_tensors"] == 0)
    mixed = {}
    for row in rows:
        mode = row["precision_mode"]
        if mode not in ("bf16", "fp16") or row.get("status") != "ok" or row["width"] != 512:
            continue
        per_arm = {name: {k: arm["parity"][k] for k in ("tensors", "bitwise_exact", "failures")}
                   for name, arm in row["arms"].items() if arm.get("parity")}
        mixed[mode] = {"per_arm": per_arm,
                       "all_candidate_failures": sum(p["failures"] for p in per_arm.values())}
    checks["mixed_arm_counts"] = (mixed.get("bf16", {}).get("all_candidate_failures") == 154
                                   and mixed.get("fp16", {}).get("all_candidate_failures") == 126
                                   and mixed["bf16"]["per_arm"]["G"]["failures"] == 52
                                   and mixed["fp16"]["per_arm"]["G"]["failures"] == 42)
    frontier = [{"cell_id": r["cell_id"],
                 "parity_arms": [k for k, v in r.get("arms", {}).items() if v.get("parity")]}
                for r in rows if r["width"] == 4096]
    return {"base": BASE, "checks": checks, "pass": all(checks.values()),
            "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "status_counts": {"/".join(k): v for k, v in sorted(counts.items())},
            "stress": stress, "mixed_trainer": mixed, "frontier": frontier,
            "interpretation": "nonfinite_parity_tensors counts matching nonfinite cases, not finite passes; "
                              "FUSE-12 executive arm-G failures differ in denominator from all-arm totals"}


def matrix():
    # Frozen priorities; each downstream stage needs a separate receipt.
    return {"base": BASE, "roots": ROOTS, "atol": ATOL, "rtol": RTOL,
            "stages": {
                "A": {"task": "native EMA semantics", "widths": [128, 512, 1024],
                      "calibration": ["off", "1", "50"], "modes": MODES},
                "B_C": {"task": "cast storage and independent adjoints",
                        "micro_widths": [64, 128, 256], "passes": [1, 6, 12, 24],
                        "scales": [1, 10, 100, 10000],
                        "targets": ["ordinary", "exact", "near"],
                        "arms": ["native", "cached_per_use_cast", "cached_shared_factor_cast"]},
                "D": {"task": "frontier and checkpoint transfer",
                      "synthetic_frontier": FRONTIER, "checkpoint_required": True},
                "E": {"task": "admission", "baselines": ["exact_saved_mac", "density", "first_fit",
                                                       "log_ridge", "physical_plus_learned"],
                      "holdouts": ["width", "depth", "device", "checkpoint"]}},
            "hardware": {"lease_mib": 12288, "headroom_mib": 2048, "min_power_w": 60},
            "defaults": "off", "gpu_execution": "explicit only"}


def probe(device_name, width, passes, seed, mode, scale=1.0, target="ordinary"):
    import torch
    import torch.nn.functional as F

    device = torch.device(device_name)
    dtype = {"fp32": torch.float32, "tf32": torch.float32,
             "bf16": torch.bfloat16, "fp16": torch.float16}[mode]
    if mode == "tf32" and device.type != "cuda":
        return {"status": "unsupported", "reason": "TF32 requires CUDA"}
    if mode == "fp16" and device.type == "cpu":
        return {"status": "unsupported", "reason": "CPU FP16 outside this gym"}
    rank = max(4, width // 8)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    def rand(*shape):
        return torch.randn(shape, generator=gen).to(device)
    # Fixed inputs shared across all arms, no stochastic forward differences.
    base_v = rand(width, rank) * (scale / width**0.5)
    base_u = rand(rank, width) / rank**0.5
    xs = [rand(2, width) for _ in range(passes)]
    cotangents = [rand(2, width) for _ in range(passes)]
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if mode == "tf32" else "highest")
    torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
    torch.backends.cudnn.allow_tf32 = mode == "tf32"

    def ac():
        return torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32)

    class SharedCastResidual(torch.autograd.Function):
        # Takes the *actual compute-dtype* factors. Their casts remain in the
        # autograd graph, so this experiment can vary cast-node sharing.
        @staticmethod
        def forward(ctx, v, u, payload):
            ctx.save_for_backward(v, u)
            return payload

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, g):
            v, u = ctx.saved_tensors
            # The native FP32 residual subtract and cast is an explicit
            # precision boundary; sign/transpose occurs before low-dtype mm.
            with torch.autocast(device.type, enabled=False):
                dz = (-g.float().T).to(v.dtype)
                return dz @ u.T, v.T @ dz, None

    def compare(a, b):
        shape_dtype = a.shape == b.shape and a.dtype == b.dtype
        if not shape_dtype:
            return {"bitwise": False, "finite": False, "within_tol": False}
        a, b = a.detach().contiguous(), b.detach().contiguous()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        # True byte equality, including signed zeros and NaN payloads.
        bitwise = bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
        close = finite and bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL))
        return {"bitwise": bitwise, "finite": finite, "within_tol": close,
                "max_abs": float((a.float()-b.float()).abs().max()) if finite else None,
                "nan_masks_equal": bool(torch.equal(torch.isnan(a), torch.isnan(b))),
                "posinf_masks_equal": bool(torch.equal(torch.isposinf(a), torch.isposinf(b))),
                "neginf_masks_equal": bool(torch.equal(torch.isneginf(a), torch.isneginf(b)))}

    try:
        with torch.no_grad(), ac():
            product = base_v @ base_u
        if target == "ordinary":
            w = rand(width, width)
        elif target == "exact":
            w = product.T.float().clone()
        elif target == "near":
            w = product.T.float() + 1e-5 * rand(width, width)
        else:
            raise ValueError(target)

        def arm(name):
            v = base_v.detach().clone().requires_grad_(True)
            u = base_u.detach().clone().requires_grad_(True)
            inputs = [x.detach().clone().requires_grad_(True) for x in xs]
            saved_square = {}
            calls = 0
            def pack(t):
                nonlocal calls
                if t.ndim == 2 and tuple(t.shape) == (width, width):
                    calls += 1
                    storage = t.untyped_storage()
                    saved_square[(str(t.device), storage.data_ptr())] = storage.nbytes()
                return t
            def unpack(t):
                return t

            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                outputs = []
                with ac():
                    if name != "native":
                        with torch.no_grad():
                            payload = (w.float() - (v @ u).T).to(dtype)
                        if name == "cached_shared_factor_cast":
                            vc, uc = v.to(dtype), u.to(dtype)
                    for x in inputs:
                        if name == "native":
                            delta = w.float() - (v @ u).T
                            y = F.linear(x, delta)
                        else:
                            if name == "cached_per_use_cast":
                                vc, uc = v.to(dtype), u.to(dtype)
                            delta = SharedCastResidual.apply(vc, uc, payload)
                            # Explicit low-dtype input and residual; no implicit copy.
                            with torch.autocast(device.type, enabled=False):
                                y = F.linear(x.to(dtype), delta)
                        outputs.append(y)
                    loss = sum((y.float() * g).sum() for y, g in zip(outputs, cotangents))
                loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tensors = {**{"y"+str(i): y.detach().cpu() for i, y in enumerate(outputs)},
                       **{"xgrad"+str(i): x.grad.detach().cpu() for i, x in enumerate(inputs)},
                       "g_V": v.grad.detach().cpu(), "g_U": u.grad.detach().cpu()}
            return tensors, {"saved_square_calls": calls, "unique_square_storages": len(saved_square),
                             "unique_square_bytes": sum(saved_square.values()),
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                             if device.type == "cuda" else None}

        ref, ref_stats = arm("native")
        candidates = {}
        for name in ("cached_per_use_cast", "cached_shared_factor_cast"):
            cand, stats = arm(name)
            metrics = {k: compare(ref[k], cand[k]) for k in ref}
            candidates[name] = {"storage": stats, "comparisons": metrics,
                                "finite_bitwise": all(m["finite"] and m["bitwise"] for m in metrics.values()),
                                "within_tol": all(m["within_tol"] for m in metrics.values())}
        return {"status": "diagnostic_only", "device": device_name, "mode": mode, "seed": seed,
                "width": width, "passes": passes, "scale": scale, "target": target,
                "native_storage": ref_stats, "candidates": candidates}
    finally:
        torch.set_float32_matmul_precision(old_prec)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--allow-cuda", action="store_true")
    p.add_argument("--width", type=int, choices=(64,128,256), default=64)
    p.add_argument("--passes", type=int, choices=(1,6,12,24), default=6)
    p.add_argument("--seed", type=int, choices=ROOTS, default=ROOTS[0])
    p.add_argument("--mode", choices=MODES, default="bf16")
    p.add_argument("--scale", type=float, choices=(1,10,100,10000), default=1)
    p.add_argument("--target", choices=("ordinary","exact","near"), default="ordinary")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.plan and args.probe:
        p.error("choose plan or probe")
    if args.probe and args.device == "cuda":
        if not args.allow_cuda:
            p.error("CUDA needs --allow-cuda inside an active VRAMorama lease")
        from .validate_fuse12_cuda import check_gpu_safety
        snap = check_gpu_safety(min_watts=60, min_free_mib=2048)
        if snap.get("error") or snap.get("memory_free_mib") is None or snap.get("enforced_power_limit_w") is None:
            raise RuntimeError("CUDA telemetry unavailable: fail closed")
    data = (matrix() if args.plan else
            probe(args.device, args.width, args.passes, args.seed, args.mode, args.scale, args.target)
            if args.probe else audit())
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(args.output)
    print(payload, end="")
    return 0 if data.get("pass", True) else 1


if __name__ == "__main__":
    sys.exit(main())
