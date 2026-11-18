"""Host DRAM (RSS) and CUDA VRAM telemetry + hard safety guard for BPD-DT04.

Designed for unattended overnight execution on 16 GB RTX 5080 + ~22 GB free host RAM:
enforces per-process CUDA memory cap, records Win32/POSIX RSS and peak reserved VRAM,
and fails closed before host paging if RSS approaches the configured ceiling.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import gc
import os
import sys
import torch


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def host_memory_bytes() -> dict[str, int]:
    """Return current and peak process resident set size (RSS) in bytes."""
    if sys.platform == "win32":
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if ok:
            return {
                "rss_bytes": int(counters.WorkingSetSize),
                "peak_rss_bytes": int(counters.PeakWorkingSetSize),
                "commit_bytes": int(counters.PagefileUsage),
            }
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mult = 1024 if sys.platform != "darwin" else 1
        return {
            "rss_bytes": int(usage * mult),
            "peak_rss_bytes": int(usage * mult),
            "commit_bytes": int(usage * mult),
        }
    except Exception:
        return {"rss_bytes": 0, "peak_rss_bytes": 0, "commit_bytes": 0}


def configure_device_budget(
    device: torch.device | str,
    *,
    cuda_fraction: float = 0.75,
    cpu_threads: int = 4,
) -> dict:
    """Configure deterministic TF32-off arithmetic and cap CUDA memory usage."""
    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(device)
    info = {
        "device": str(dev),
        "cpu_threads": cpu_threads,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if dev.type == "cuda" and torch.cuda.is_available():
        idx = dev.index or 0
        torch.cuda.set_per_process_memory_fraction(cuda_fraction, idx)
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "gpu_name": props.name,
            "gpu_total_bytes": int(props.total_memory),
            "cuda_fraction_cap": float(cuda_fraction),
            "cuda_cap_bytes": int(props.total_memory * cuda_fraction),
        })
    return info


def memory_snapshot(device: torch.device | str = "cpu") -> dict[str, float | int]:
    """Capture current/peak host RSS and CUDA allocated/reserved MiB."""
    host = host_memory_bytes()
    snap: dict[str, float | int] = {
        "host_rss_mib": round(host["rss_bytes"] / (1024.0 ** 2), 2),
        "host_peak_rss_mib": round(host["peak_rss_bytes"] / (1024.0 ** 2), 2),
        "host_commit_mib": round(host["commit_bytes"] / (1024.0 ** 2), 2),
    }
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        idx = dev.index or 0
        snap.update({
            "cuda_allocated_mib": round(torch.cuda.memory_allocated(idx) / (1024.0 ** 2), 2),
            "cuda_reserved_mib": round(torch.cuda.memory_reserved(idx) / (1024.0 ** 2), 2),
            "cuda_max_allocated_mib": round(torch.cuda.max_memory_allocated(idx) / (1024.0 ** 2), 2),
            "cuda_max_reserved_mib": round(torch.cuda.max_memory_reserved(idx) / (1024.0 ** 2), 2),
        })
    return snap


def enforce_memory_ceiling(
    device: torch.device | str = "cpu",
    *,
    max_host_rss_gib: float = 12.0,
    max_cuda_reserved_gib: float = 12.5,
) -> dict[str, float | int]:
    """Run GC/CUDA cache cleanup and raise RuntimeError if host RSS or VRAM exceeds ceiling."""
    gc.collect()
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    snap = memory_snapshot(dev)
    if float(snap["host_rss_mib"]) > max_host_rss_gib * 1024.0:
        raise RuntimeError(
            f"Host RSS {snap['host_rss_mib']} MiB exceeded safety ceiling {max_host_rss_gib} GiB"
        )
    if "cuda_reserved_mib" in snap and float(snap["cuda_reserved_mib"]) > max_cuda_reserved_gib * 1024.0:
        raise RuntimeError(
            f"CUDA reserved {snap['cuda_reserved_mib']} MiB exceeded safety ceiling {max_cuda_reserved_gib} GiB"
        )
    return snap
