"""Native TRM microbatch measurements using random engineering inputs.

Each call owns a temporary random model, with CPU/CUDA RNG state restored when it
returns. Training trials include persistent Adam moment and EMA storage but time
forward/backward only. No optimizer update, checkpoint selection, model-root
training, puzzle generation, or evaluation-cohort access occurs here.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import torch

from mechinterp.bpd.adapters.trm import build_mini_trm
from . import FIT_DEPTHS
from .resources import MemoryBudget, assert_budget, calibrate_microbatch, memory_snapshot
from .training import MODEL_CONFIGS, autocast_context, backward_supervised_batch, model_digest

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
    device_name = str(dev)
    initial_memory = memory_snapshot(device_name)
    assert_budget(initial_memory, budget)
    cuda_devices = [dev.index] if dev.type == "cuda" else []
    original_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    original_tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Disable a caller's inference context during construction so training
        # tensors and model parameters always support ordinary autograd.
        with torch.inference_mode(False), torch.enable_grad(), torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(ENGINEERING_SEED)
            model = build_mini_trm(**MODEL_CONFIGS[profile], seed=ENGINEERING_SEED).float().to(dev)
            model.train(stage == "training")
            frozen_digest = model_digest(model)
            parameters = list(model.parameters())
            parameter_count = sum(p.numel() for p in parameters)
            parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
            # Use a local CPU generator, then transfer once outside timing.
            generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 1)
            cells = int(model.seq_len)
            digits = int(cells ** 0.5)
            tokens = torch.randint(1, digits + 2, (logical_batch, cells), generator=generator).to(dev)
            targets = torch.randint(0, digits, (logical_batch, cells), generator=generator).to(dev)
            # A repeatable mixture ensures both blank and clue objective terms.
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

            def run_batch(microbatch: int) -> None:
                if stage == "training":
                    model.zero_grad(set_to_none=True)
                    backward_supervised_batch(
                        model, tokens[:microbatch], targets[:microbatch],
                        microbatch=microbatch, precision=precision,
                    )
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
                model.zero_grad(set_to_none=True)
            result = {
                "schema": "BPD-DT05/native-resource-calibration/v1",
                "status": "CALIBRATED" if calibration["selected_microbatch"] is not None else "NO_ADMISSIBLE_MICROBATCH",
                "purpose": "engineering throughput and sampled memory calibration",
                "evidence_class": "engineering measurement on random inputs",
                "profile": profile, "model_config": MODEL_CONFIGS[profile].copy(),
                "stage": stage, "device": device_name, "precision": precision,
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
            # Buffers intentionally stayed referenced throughout every measurement.
            del optimizer_moments, ema_shadow, model, parameters, tokens, targets
        return result
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32_matmul
        torch.backends.cudnn.allow_tf32 = original_tf32_cudnn
