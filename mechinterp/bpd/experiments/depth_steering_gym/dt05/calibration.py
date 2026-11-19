"""Native TRM microbatch measurements using random engineering inputs.

Each call owns a temporary random model, with CPU/CUDA RNG state restored when it
returns. ``run_calibration`` times isolated forward/backward microbatches without
optimizer updates; ``run_full_step_benchmark`` separately measures complete
optimizer updates including CPU data preparation, host-to-device transfer,
gradient clipping, AdamW step, LR schedule, and EMA shadow updates.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import statistics
import time
from typing import Iterable

import torch

from mechinterp.bpd.adapters.trm import build_mini_trm
from . import FIT_DEPTHS
from .resources import (
    GIB,
    BudgetExceeded,
    MemoryBudget,
    ReusableTransferBuffer,
    _is_oom,
    _required_number,
    _reset_cuda_peak,
    _synchronize,
    assert_budget,
    calibrate_microbatch,
    memory_snapshot,
)
from .training import (
    ExponentialMovingAverage,
    GraphedSupervisedExecutor,
    MODEL_CONFIGS,
    _make_adamw,
    autocast_context,
    backward_supervised_batch,
    model_digest,
    sample_augmented_batch,
)

ENGINEERING_SEED = 2605991


def _validate_device(device: str, precision: str) -> torch.device:
    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda"):
        raise ValueError("calibration supports CPU and CUDA")
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        idx = torch.cuda.current_device() if dev.index is None else dev.index
        if idx < 0 or idx >= torch.cuda.device_count():
            raise ValueError("requested CUDA device does not exist")
        dev = torch.device("cuda", idx)
        with torch.cuda.device(dev):
            if precision == "bf16" and not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 requires support on the selected CUDA device")
    elif precision == "bf16":
        raise ValueError("BF16 calibration requires CUDA")
    return dev


def _engineering_cpu_prep(
    pool_givens: torch.Tensor,
    pool_solutions: torch.Tensor,
    logical_batch: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare one logical batch of augmented Sudoku-shaped engineering tokens on CPU."""
    size = int(pool_givens.shape[1])
    if size == 4:
        return sample_augmented_batch(pool_givens, pool_solutions, logical_batch, gen)
    # 9x9 engineering augmentation: random pool gather + transpose + digit relabeling
    idx = torch.randint(len(pool_givens), (logical_batch,), generator=gen)
    g = pool_givens[idx]
    s = pool_solutions[idx]
    tr = torch.randint(2, (logical_batch, 1, 1), generator=gen).bool()
    g = torch.where(tr, g.transpose(1, 2), g)
    s = torch.where(tr, s.transpose(1, 2), s)
    perm_digits = torch.argsort(torch.rand((logical_batch, size), generator=gen), dim=-1) + 1
    lut = torch.cat((torch.zeros((logical_batch, 1), dtype=torch.long), perm_digits), dim=-1)
    bx = lut.gather(1, g.reshape(logical_batch, size * size)) + 1
    by = lut.gather(1, s.reshape(logical_batch, size * size)) - 1
    return bx, by


def run_calibration(
    profile: str,
    stage: str,
    logical_batch: int,
    candidates: Iterable[int],
    device: str,
    budget: MemoryBudget,
    precision: str = "fp32",
    warmups: int = 1,
    repeats: int = 3,
    *,
    checkpoint_act: bool = False,
    cuda_graph: bool = False,
) -> dict:
    """Measure isolated native inference or training microbatches without learning.

    Training uses the same fit-depth blank/clue objective as ``training.py``.
    Its gradients remain alive for the post-work memory sample. Training speed
    excludes the optimizer step, EMA update, data preparation and transfer.
    The fixed logical batch determines accumulation; this function measures one
    microbatch at a time, so a full optimizer-step duration is not measured.

    Inference measures eight native ACT steps. Both precision modes keep FP32
    master weights; ``bf16`` applies CUDA autocast to the forward calculations.
    Host limits remain sampled guards. A sudden temporary host allocation can
    exceed them between samples, while CUDA allocator peaks are recorded.
    """
    if profile not in MODEL_CONFIGS:
        raise ValueError(f"unknown model profile: {profile}")
    if stage not in ("inference", "training"):
        raise ValueError("stage must be inference or training")
    if type(logical_batch) is not int or logical_batch < 1:
        raise ValueError("logical_batch must be a positive integer")
    batches = list(candidates)
    if not batches or any(type(b) is not int or b < 1 or logical_batch % b for b in batches):
        raise ValueError("candidate microbatches must divide the logical batch")
    if type(warmups) is not int or warmups < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats positive")
    dev = _validate_device(device, precision)
    if cuda_graph and dev.type != "cuda":
        raise ValueError("cuda_graph calibration requires a CUDA device")
    if cuda_graph and checkpoint_act:
        raise ValueError("cuda_graph and checkpoint_act cannot be combined in one trial")
    device_name = str(dev)
    initial_memory = memory_snapshot(device_name)
    assert_budget(initial_memory, budget)
    cuda_devices = [dev.index] if dev.type == "cuda" else []
    original_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    original_tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode(False), torch.enable_grad(), torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(ENGINEERING_SEED)
            model = build_mini_trm(**MODEL_CONFIGS[profile], seed=ENGINEERING_SEED).float().to(dev)
            model.train(stage == "training")
            frozen_digest = model_digest(model)
            parameters = list(model.parameters())
            parameter_count = sum(p.numel() for p in parameters)
            parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
            generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
            cells = int(model.seq_len)
            digits = int(cells ** 0.5)
            tokens = torch.randint(1, digits + 2, (logical_batch, cells), generator=generator).to(dev)
            targets = torch.randint(0, digits, (logical_batch, cells), generator=generator).to(dev)
            tokens[:, ::2] = 1
            tokens[:, 1] = 2
            optimizer_moments = []
            ema_shadow = []
            if stage == "training":
                optimizer_moments = [torch.zeros_like(p) for p in parameters for _ in range(2)]
                ema_shadow = [value.detach().clone() for value in model.state_dict().values()]
            optimizer_moment_bytes = sum(t.numel() * t.element_size() for t in optimizer_moments)
            ema_shadow_bytes = sum(t.numel() * t.element_size() for t in ema_shadow)
            resident_memory = memory_snapshot(device_name)
            assert_budget(resident_memory, budget)

            graphed_cache: dict[int, object] = {}

            def run_batch(microbatch: int) -> None:
                if stage == "training":
                    if cuda_graph:
                        exec_obj = graphed_cache.get(microbatch)
                        if exec_obj is None:
                            # Release prior microbatch graph buffers before capturing next candidate
                            graphed_cache.clear()
                            exec_obj = GraphedSupervisedExecutor(model, microbatch, precision=precision)
                            graphed_cache[microbatch] = exec_obj
                        exec_obj.run(tokens[:microbatch], targets[:microbatch])  # type: ignore[attr-defined]
                    else:
                        model.zero_grad(set_to_none=True)
                        backward_supervised_batch(
                            model, tokens[:microbatch], targets[:microbatch],
                            microbatch=microbatch, precision=precision, checkpoint_act=checkpoint_act,
                        )
                else:
                    if cuda_graph:
                        entry = graphed_cache.get(microbatch)
                        if entry is None:
                            graphed_cache.clear()
                            static_x = tokens[:microbatch].clone()
                            w_stream = torch.cuda.Stream(device=dev)
                            w_stream.wait_stream(torch.cuda.current_stream(dev))
                            with torch.cuda.stream(w_stream), torch.no_grad(), autocast_context(dev, precision):
                                for _ in range(3):
                                    emb_w = model.encode_inputs(static_x)
                                    zh_w, zl_w = model.init_states(microbatch)
                                    for _ in range(8):
                                        zh_w, zl_w = model.act_step(zh_w, zl_w, emb_w)
                                    _ = model.digit_logits(zh_w)
                            torch.cuda.current_stream(dev).wait_stream(w_stream)
                            g_inf = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g_inf), torch.no_grad(), autocast_context(dev, precision):
                                emb_g = model.encode_inputs(static_x)
                                zh_g, zl_g = model.init_states(microbatch)
                                for _ in range(8):
                                    zh_g, zl_g = model.act_step(zh_g, zl_g, emb_g)
                                out_g = model.digit_logits(zh_g)
                            entry = (static_x, g_inf, out_g)
                            graphed_cache[microbatch] = entry
                        sx, g_inf, _ = entry  # type: ignore[misc]
                        sx.copy_(tokens[:microbatch])
                        g_inf.replay()
                    else:
                        with torch.inference_mode(), autocast_context(dev, precision):
                            emb = model.encode_inputs(tokens[:microbatch])
                            zh, zl = model.init_states(microbatch)
                            for _ in range(8):
                                zh, zl = model.act_step(zh, zl, emb)
                            model.digit_logits(zh)

            try:
                calibration = calibrate_microbatch(
                    batches, logical_batch, run_batch,
                    device=device_name, budget=budget, warmups=warmups, repeats=repeats,
                )
                after_digest = model_digest(model)
                if after_digest != frozen_digest:
                    raise AssertionError("calibration changed model weights")
                final_memory = memory_snapshot(device_name)
            finally:
                graphed_cache.clear()
                model.zero_grad(set_to_none=True)
            result = {
                "schema": "BPD-DT05/native-resource-calibration/v1",
                "status": "CALIBRATED" if calibration["selected_microbatch"] is not None else "NO_ADMISSIBLE_MICROBATCH",
                "purpose": "engineering throughput and sampled memory calibration",
                "evidence_class": "engineering measurement on random inputs",
                "profile": profile, "model_config": MODEL_CONFIGS[profile].copy(),
                "stage": stage, "device": device_name, "precision": precision,
                "activation_checkpointing": checkpoint_act,
                "cuda_graph": cuda_graph,
                "master_weight_dtype": "float32", "autocast_enabled": precision == "bf16",
                "parameter_count": parameter_count, "parameter_bytes": parameter_bytes,
                "logical_batch": logical_batch,
                "training_fit_depths": list(FIT_DEPTHS) if stage == "training" else None,
                "inference_act_steps": 8 if stage == "inference" else None,
                "block_executions_per_act": len(model.layers) * model.h_cycles * (model.l_cycles + 1),
                "optimizer_updates": 0,
                "persistent_adam_moment_bytes": optimizer_moment_bytes,
                "persistent_ema_shadow_bytes": ema_shadow_bytes,
                "gradient_storage": "retained through post-trial sample" if stage == "training" else "none",
                "timing_scope": "microbatch forward/backward; no optimizer/EMA update or input transfer" if stage == "training" else "microbatch encode + eight ACT steps + output head; no input transfer",
                "input_kind": "random engineering tokens and labels; no puzzle or evaluation cohort",
                "engineering_seed": ENGINEERING_SEED,
                "source_weights_sha256": frozen_digest,
                "final_weights_sha256": after_digest,
                "parameters_unchanged": True,
                "rng_restored": True,
                "tf32_matmul": False, "tf32_cudnn": False,
                "budget": asdict(budget),
                "memory_initial": initial_memory, "memory_resident_buffers": resident_memory,
                "memory_final_before_cleanup": final_memory,
                "calibration": calibration,
                "selected_microbatch": calibration["selected_microbatch"],
                "accumulation_steps": calibration["accumulation_steps"],
                "memory_limit_scope": "sampled process and device admission; not an OS memory cap",
                "scientific_cohorts_accessed": False,
            }
            del optimizer_moments, ema_shadow, model, parameters, tokens, targets
        return result
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32_matmul
        torch.backends.cudnn.allow_tf32 = original_tf32_cudnn


def run_full_step_benchmark(
    profile: str,
    logical_batch: int,
    candidates: Iterable[int],
    device: str,
    budget: MemoryBudget,
    precision: str = "fp32",
    warmups: int = 1,
    repeats: int = 3,
    *,
    execution_mode: str = "optimized",
    checkpoint_act: bool = False,
    cuda_graph: bool | None = None,
) -> dict:
    """Measure complete end-to-end optimizer updates on random engineering inputs.

    Unlike ``run_calibration``, each trial executes a full logical-batch update:
    CPU data preparation, host-to-device transfer, ``logical_batch // microbatch``
    accumulated forward/backward microbatches, gradient norm clipping, AdamW
    step, LR scheduler step, and EMA shadow update. Stage sub-timings and full
    overlapped step throughput are reported separately.
    """
    if profile not in MODEL_CONFIGS:
        raise ValueError(f"unknown model profile: {profile}")
    if execution_mode not in ("reference", "optimized"):
        raise ValueError("execution_mode must be reference or optimized")
    if type(logical_batch) is not int or logical_batch < 1:
        raise ValueError("logical_batch must be a positive integer")
    batches = sorted(set(candidates))
    if not batches or any(type(b) is not int or b < 1 or logical_batch % b for b in batches):
        raise ValueError("candidate microbatches must divide the logical batch")
    if type(warmups) is not int or warmups < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats positive")

    dev = _validate_device(device, precision)
    use_opt = execution_mode == "optimized"
    use_graph = bool(
        (dev.type == "cuda" and not checkpoint_act)
        if cuda_graph is None and use_opt
        else (cuda_graph and dev.type == "cuda" and not checkpoint_act)
    )
    if cuda_graph and dev.type != "cuda":
        raise ValueError("cuda_graph requires CUDA")
    if cuda_graph and checkpoint_act:
        raise ValueError("cuda_graph and checkpoint_act cannot be combined")

    device_name = str(dev)
    initial_memory = memory_snapshot(device_name)
    assert_budget(initial_memory, budget)
    cuda_devices = [dev.index] if dev.type == "cuda" else []
    original_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    original_tf32_cudnn = torch.backends.cudnn.allow_tf32

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode(False), torch.enable_grad(), torch.random.fork_rng(devices=cuda_devices):
            cells = int(MODEL_CONFIGS[profile]["seq_len"])
            size = int(math.isqrt(cells))
            gen_pool = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 3)
            pool_solutions = torch.randint(1, size + 1, (256, size, size), generator=gen_pool)
            pool_givens = pool_solutions.clone()
            pool_givens.reshape(256, -1)[:, ::2] = 0

            trials: list[dict] = []
            previous_ok: dict | None = None
            stop_reason: str | None = None
            parameter_count = 0
            parameter_bytes = 0
            initial_digest = ""

            for mb in batches:
                row: dict = {
                    "microbatch": mb,
                    "accumulation_steps": logical_batch // mb,
                    "execution_mode": execution_mode,
                    "cuda_graph": use_graph,
                    "activation_checkpointing": checkpoint_act,
                }
                if stop_reason is not None:
                    trials.append({**row, "status": "SKIPPED_AFTER_LIMIT", "reason": stop_reason})
                    continue
                model = None
                opt = None
                ema = None
                graphed_exec = None
                try:
                    before = memory_snapshot(device_name)
                    assert_budget(before, budget)
                    row["before"] = before
                    if previous_ok is not None and dev.type == "cuda":
                        growth = mb / previous_ok["microbatch"] - 1.0
                        gpu_inc = max(0, previous_ok["peak_cuda_reserved_bytes"] - previous_ok["before"]["cuda_reserved_bytes"])
                        extra = gpu_inc * growth * 1.2
                        if before["cuda_reserved_bytes"] + extra > budget.max_cuda_reserved_gib * GIB:
                            raise BudgetExceeded("predicted CUDA growth exceeds ceiling")
                        if before["cuda_free_bytes"] - extra < budget.min_cuda_free_gib * GIB:
                            raise BudgetExceeded("predicted CUDA growth would consume the reserve")

                    torch.manual_seed(ENGINEERING_SEED)
                    model = build_mini_trm(**MODEL_CONFIGS[profile], seed=ENGINEERING_SEED).float().to(dev).train()
                    if not initial_digest:
                        initial_digest = model_digest(model)
                        params = list(model.parameters())
                        parameter_count = sum(p.numel() for p in params)
                        parameter_bytes = sum(p.numel() * p.element_size() for p in params)
                    opt = _make_adamw(model.parameters(), lr=0.002, weight_decay=1e-4, dev=dev, allow_fused=use_opt)
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(10, warmups + repeats))
                    ema = ExponentialMovingAverage(model, 0.995)
                    gen_step = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 17)

                    bx_stage = ReusableTransferBuffer((logical_batch, cells), torch.long, dev, pin_memory=use_opt) if use_opt else None
                    by_stage = ReusableTransferBuffer((logical_batch, cells), torch.long, dev, pin_memory=use_opt) if use_opt else None
                    copy_stream = torch.cuda.Stream(device=dev) if (use_opt and dev.type == "cuda") else None
                    if use_graph:
                        graphed_exec = GraphedSupervisedExecutor(model, mb, precision=precision)

                    _reset_cuda_peak(device_name)
                    observations = [before]
                    t_prep, t_h2d, t_fwbw, t_opt, t_ema, t_full = [], [], [], [], [], []

                    for it in range(warmups + repeats):
                        _synchronize(device_name)
                        s0 = time.perf_counter()
                        bx_cpu, by_cpu = _engineering_cpu_prep(pool_givens, pool_solutions, logical_batch, gen_step)
                        s1 = time.perf_counter()
                        if bx_stage is not None and by_stage is not None:
                            bx_dev = bx_stage.stage_and_send(bx_cpu, stream=copy_stream)
                            by_dev = by_stage.stage_and_send(by_cpu, stream=copy_stream)
                            if copy_stream is not None:
                                torch.cuda.current_stream(dev).wait_stream(copy_stream)
                        else:
                            bx_dev = bx_cpu.to(dev)
                            by_dev = by_cpu.to(dev)
                        _synchronize(device_name)
                        s2 = time.perf_counter()

                        if graphed_exec is not None:
                            _ = graphed_exec.run(bx_dev, by_dev)
                        else:
                            opt.zero_grad(set_to_none=True)
                            _ = backward_supervised_batch(
                                model, bx_dev, by_dev, microbatch=mb,
                                precision=precision, checkpoint_act=checkpoint_act,
                            )
                        _synchronize(device_name)
                        s3 = time.perf_counter()

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                        opt.step()
                        sched.step()
                        _synchronize(device_name)
                        s4 = time.perf_counter()

                        ema.update(model)
                        _synchronize(device_name)
                        s5 = time.perf_counter()

                        after = memory_snapshot(device_name)
                        observations.append(after)
                        assert_budget(after, budget)
                        if dev.type == "cuda" and _required_number(after, "cuda_peak_reserved_bytes") > budget.max_cuda_reserved_gib * GIB:
                            raise BudgetExceeded("trial peak CUDA reservation exceeds ceiling")
                        if it >= warmups:
                            t_prep.append(s1 - s0)
                            t_h2d.append(s2 - s1)
                            t_fwbw.append(s3 - s2)
                            t_opt.append(s4 - s3)
                            t_ema.append(s5 - s4)
                            t_full.append(s5 - s0)

                    med_full = statistics.median(t_full)
                    row.update({
                        "status": "OK",
                        "full_step_seconds": t_full,
                        "full_step_median_seconds": med_full,
                        "examples_per_second": logical_batch / med_full,
                        "optimizer_updates_per_second": 1.0 / med_full,
                        "stage_median_seconds": {
                            "data_preparation": statistics.median(t_prep),
                            "h2d_transfer": statistics.median(t_h2d),
                            "forward_backward_accumulated": statistics.median(t_fwbw),
                            "grad_clip_and_optimizer_step": statistics.median(t_opt),
                            "ema_update": statistics.median(t_ema),
                        },
                        "ema_updates_completed": ema.updates,
                        "post_trial_weights_sha256": model_digest(model),
                        "peak_sampled_host_rss_bytes": max(s["host_rss_bytes"] for s in observations),
                        "peak_cuda_reserved_bytes": (
                            max((s["cuda_peak_reserved_bytes"] or 0) for s in observations[1:])
                            if dev.type == "cuda" else None
                        ),
                        "peak_cuda_allocated_bytes": (
                            max((s["cuda_peak_allocated_bytes"] or 0) for s in observations[1:])
                            if dev.type == "cuda" else None
                        ),
                        "after": observations[-1],
                    })
                    previous_ok = row
                except BudgetExceeded as exc:
                    row.update({"status": "BUDGET_REJECTED", "reason": str(exc)})
                    stop_reason = str(exc)
                except BaseException as exc:
                    if not _is_oom(exc):
                        raise
                    row.update({"status": "OOM", "reason": f"{type(exc).__name__}: {exc}"})
                    stop_reason = "smaller microbatch exhausted memory"
                finally:
                    del graphed_exec, ema, opt, model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                trials.append(row)

            ok_rows = [r for r in trials if r["status"] == "OK"]
            best = max(ok_rows, key=lambda r: (r["examples_per_second"], -r["microbatch"])) if ok_rows else None
            return {
                "schema": "BPD-DT05/full-step-benchmark/v1",
                "status": "BENCHMARKED" if best is not None else "NO_ADMISSIBLE_MICROBATCH",
                "purpose": "end-to-end optimizer step, data preparation, H2D transfer, and EMA benchmark",
                "evidence_class": "engineering measurement on random inputs; separate from microbatch forward/backward calibrator",
                "profile": profile,
                "model_config": MODEL_CONFIGS[profile].copy(),
                "device": device_name,
                "precision": precision,
                "execution_mode": execution_mode,
                "activation_checkpointing": checkpoint_act,
                "cuda_graph": use_graph,
                "parameter_count": parameter_count,
                "parameter_bytes": parameter_bytes,
                "logical_batch": logical_batch,
                "warmups": warmups,
                "repeats": repeats,
                "training_fit_depths": list(FIT_DEPTHS),
                "timing_scope": "complete logical-batch optimizer step: CPU prep + H2D transfer + accumulated fw/bw + grad clip + AdamW + scheduler + EMA",
                "engineering_seed": ENGINEERING_SEED,
                "initial_weights_sha256": initial_digest,
                "rng_restored": True,
                "tf32_matmul": False,
                "tf32_cudnn": False,
                "budget": asdict(budget),
                "memory_initial": initial_memory,
                "trials": trials,
                "selected_microbatch": best["microbatch"] if best else None,
                "accumulation_steps": best["accumulation_steps"] if best else None,
                "best_examples_per_second": best["examples_per_second"] if best else None,
                "best_optimizer_updates_per_second": best["optimizer_updates_per_second"] if best else None,
                "scientific_cohorts_accessed": False,
            }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32_matmul
        torch.backends.cudnn.allow_tf32 = original_tf32_cudnn
