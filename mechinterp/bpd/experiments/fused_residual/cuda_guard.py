"""FUSE-12: CUDA device, single-stream ownership, and precision-mode guards.

Default behavior (`allow_cuda=False`, `precision_mode="fp32"`) preserves the
exact FUSE-08..11 sequential CPU FP32 contract and keeps production defaults
untouched. CUDA admission requires explicit experimental opt-in and enforces
single sequential CUDA stream ownership plus runtime precision-state binding.
"""
from __future__ import annotations

import torch

PRECISION_MODES = ("fp32", "tf32", "bf16", "fp16")
STORAGE_MODES = ("fp32", "compute_dtype")


def validate_cuda_options(
    allow_cuda: bool,
    precision_mode: str,
    storage_mode: str = "fp32",
) -> None:
    if type(allow_cuda) is not bool:
        raise TypeError("allow_cuda must be bool")
    if precision_mode not in PRECISION_MODES:
        raise ValueError(f"unknown residual cache precision_mode: {precision_mode!r}")
    if storage_mode not in STORAGE_MODES:
        raise ValueError(f"unknown residual cache storage_mode: {storage_mode!r}")
    if precision_mode != "fp32" and not allow_cuda:
        raise ValueError("non-fp32 residual cache precision modes require allow_cuda=True")
    if storage_mode != "fp32" and not allow_cuda:
        raise ValueError("non-fp32 residual cache storage_mode requires allow_cuda=True")


def target_storage_dtype(precision_mode: str, storage_mode: str) -> torch.dtype:
    if storage_mode == "compute_dtype":
        if precision_mode == "bf16":
            return torch.bfloat16
        if precision_mode == "fp16":
            return torch.float16
    return torch.float32


def runtime_precision_state(device_type: str) -> tuple[object, ...]:
    """Hashable snapshot of ambient precision/autocast/TF32 state for cache identity."""
    fp32_prec = torch.get_float32_matmul_precision()
    ac_cache = bool(torch.is_autocast_cache_enabled())
    cpu_ac = bool(torch.is_autocast_enabled("cpu"))
    cpu_ac_dt = str(torch.get_autocast_dtype("cpu")) if cpu_ac else None
    if device_type == "cuda":
        tf32_mm = bool(torch.backends.cuda.matmul.allow_tf32)
        cuda_ac = bool(torch.is_autocast_enabled("cuda"))
        cuda_ac_dt = str(torch.get_autocast_dtype("cuda")) if cuda_ac else None
        return ("cuda", tf32_mm, fp32_prec, cuda_ac, cuda_ac_dt, ac_cache, cpu_ac, cpu_ac_dt)
    return (device_type, False, fp32_prec, False, None, ac_cache, cpu_ac, cpu_ac_dt)


def current_stream_token(device: torch.device) -> tuple[str, int, int]:
    """Identify the active sequential execution stream for ownership checks."""
    if device.type == "cuda":
        idx = int(device.index if device.index is not None else torch.cuda.current_device())
        stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)
        return ("cuda", idx, stream_ptr)
    return (device.type, -1, 0)


def is_stream_capturing(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    return bool(torch.cuda.is_current_stream_capturing())


def device_and_precision_supported(
    ts: tuple[torch.Tensor, ...],
    *,
    allow_cuda: bool = False,
    precision_mode: str = "fp32",
) -> bool:
    """Return True iff tensors and ambient runtime state match the contract."""
    if not ts:
        return False
    dev = ts[0].device
    if not all(
        t.device == dev and t.ndim == 2 and t.is_contiguous() and not t.is_inference()
        for t in ts
    ):
        return False
    if dev.type == "cpu":
        return (
            precision_mode == "fp32"
            and not torch.is_autocast_enabled("cpu")
            and torch.get_float32_matmul_precision() == "highest"
            and all(t.dtype == torch.float32 for t in ts)
        )
    if dev.type == "cuda":
        if not allow_cuda:
            return False
        if is_stream_capturing(dev) or torch.is_autocast_enabled("cpu"):
            return False
        ac = bool(torch.is_autocast_enabled("cuda"))
        ac_dt = torch.get_autocast_dtype("cuda") if ac else None
        tf32_mm = bool(torch.backends.cuda.matmul.allow_tf32)
        if precision_mode == "fp32":
            return (
                not ac
                and not tf32_mm
                and torch.get_float32_matmul_precision() == "highest"
                and all(t.dtype == torch.float32 for t in ts)
            )
        if precision_mode == "tf32":
            return (
                not ac
                and tf32_mm
                and all(t.dtype == torch.float32 for t in ts)
            )
        if precision_mode == "bf16":
            return (
                ac
                and ac_dt == torch.bfloat16
                and all(t.dtype in (torch.float32, torch.bfloat16) for t in ts)
            )
        if precision_mode == "fp16":
            return (
                ac
                and ac_dt == torch.float16
                and all(t.dtype in (torch.float32, torch.float16) for t in ts)
            )
        return False
    return False
