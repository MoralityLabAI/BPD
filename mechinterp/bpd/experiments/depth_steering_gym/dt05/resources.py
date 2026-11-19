"""Bounded DT05 engineering resources and measured microbatch selection.

Snapshots distinguish current RSS from the process lifetime peak. Missing telemetry
is represented by ``None`` and an error, never by a successful zero-byte reading.
The tuner measures a caller-owned work unit; it does not train, update weights,
change precision, read evaluation labels, or launch concurrent model roots.
Budgets are checked at sampling boundaries; they are not OS memory limits and
cannot guarantee that a temporary allocation stays below the process ceiling.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import ctypes
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Callable, Iterable

GIB = 1024 ** 3


@dataclass(frozen=True)
class MemoryBudget:
    max_host_rss_gib: float = 12.0
    min_host_available_gib: float = 4.0
    max_cuda_reserved_gib: float = 12.5
    min_cuda_free_gib: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_host_rss_gib <= 0 or self.max_cuda_reserved_gib <= 0:
            raise ValueError("memory ceilings must be positive")


class BudgetExceeded(RuntimeError):
    """The resource envelope cannot be verified or has been exceeded."""


def _proc_fields(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) >= 2 and parts[1].isdigit():
            result[parts[0]] = int(parts[1]) * (1024 if len(parts) > 2 and parts[2] == "kB" else 1)
    return result


def _linux_cgroup_available() -> int | None:
    """Honor an effective cgroup v2 RAM limit when visible in the current mount."""
    try:
        roots = [Path("/sys/fs/cgroup")]
        for line in Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines():
            if line.startswith("0::"):
                relative = line[3:].lstrip("/")
                if relative and ".." not in Path(relative).parts:
                    roots.insert(0, Path("/sys/fs/cgroup") / relative)
        available: list[int] = []
        for root in roots:
            try:
                raw_limit = (root / "memory.max").read_text(encoding="ascii").strip()
                if raw_limit != "max":
                    limit = int(raw_limit)
                    current = int((root / "memory.current").read_text(encoding="ascii").strip())
                    available.append(max(0, limit - current))
            except (OSError, ValueError):
                continue
        return min(available) if available else None
    except (OSError, ValueError):
        return None


def _windows_memory() -> dict[str, int]:
    """Use explicit Win32 pointer-width signatures, including the process handle."""
    from ctypes import wintypes

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in (
                "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCountersEx), wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatusEx)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return {
        "host_rss_bytes": int(counters.WorkingSetSize),
        "host_peak_rss_bytes": int(counters.PeakWorkingSetSize),
        "host_commit_bytes": int(counters.PrivateUsage),
        "host_available_bytes": int(status.ullAvailPhys),
        "host_total_bytes": int(status.ullTotalPhys),
    }


def memory_snapshot(device: str = "cpu") -> dict:
    """Return sampled host/device memory in bytes, with explicit availability errors.

    CUDA ``free`` covers the device globally; allocated/reserved cover this process's
    PyTorch allocator. RSS excludes descendants. Peak RSS is process lifetime.
    No collection or CUDA cache flush takes place before a measurement.
    """
    dev = str(device)
    if dev != "cpu" and not (dev == "cuda" or (dev.startswith("cuda:") and dev[5:].isdigit())):
        raise ValueError("device must be cpu or cuda[:index]")
    snap = {
        "device": dev,
        "host_rss_bytes": None, "host_peak_rss_bytes": None,
        "host_commit_bytes": None, "host_available_bytes": None, "host_total_bytes": None,
        "cuda_allocated_bytes": None, "cuda_reserved_bytes": None,
        "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None,
        "cuda_free_bytes": None, "cuda_total_bytes": None,
        "host_source": None, "errors": [],
    }
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        system = psutil.virtual_memory()
        snap.update({
            "host_rss_bytes": int(mem.rss),
            "host_peak_rss_bytes": int(mem.peak_wset) if hasattr(mem, "peak_wset") else None,
            "host_commit_bytes": int(mem.private) if hasattr(mem, "private") else None,
            "host_available_bytes": int(system.available),
            "host_total_bytes": int(system.total), "host_source": "psutil",
        })
    except Exception as exc:
        snap["errors"].append(f"psutil: {type(exc).__name__}: {exc}")
        try:
            if sys.platform == "win32":
                snap.update(_windows_memory())
                snap["host_source"] = "win32"
            elif sys.platform.startswith("linux"):
                process = _proc_fields(Path("/proc/self/status"))
                system = _proc_fields(Path("/proc/meminfo"))
                snap.update({
                    "host_rss_bytes": process.get("VmRSS"),
                    "host_peak_rss_bytes": process.get("VmHWM"),
                    "host_available_bytes": system.get("MemAvailable"),
                    "host_total_bytes": system.get("MemTotal"), "host_source": "procfs",
                })
            else:
                snap["errors"].append("current RSS and system availability require psutil on this platform")
        except (OSError, ValueError) as fallback_exc:
            snap["errors"].append(f"host fallback: {type(fallback_exc).__name__}: {fallback_exc}")
    if snap["host_peak_rss_bytes"] is None and sys.platform != "win32":
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            snap["host_peak_rss_bytes"] = int(peak * (1 if sys.platform == "darwin" else 1024))
        except (ImportError, OSError):
            pass
    if snap["host_rss_bytes"] is not None and snap["host_peak_rss_bytes"] is not None:
        # Kernel interfaces sample at slightly different instants; an observed
        # current value is also a lower bound on the lifetime peak.
        snap["host_peak_rss_bytes"] = max(snap["host_peak_rss_bytes"], snap["host_rss_bytes"])
    if sys.platform.startswith("linux"):
        cgroup_available = _linux_cgroup_available()
        snap["cgroup_available_bytes"] = cgroup_available
        if cgroup_available is not None and snap["host_available_bytes"] is not None:
            snap["host_available_bytes"] = min(snap["host_available_bytes"], cgroup_available)
    if dev.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            free, total = torch.cuda.mem_get_info(dev)
            snap.update({
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(dev)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(dev)),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
                "cuda_free_bytes": int(free), "cuda_total_bytes": int(total),
            })
        except (ImportError, RuntimeError, OSError) as exc:
            snap["errors"].append(f"CUDA: {type(exc).__name__}: {exc}")
    return snap


def _required_number(snapshot: dict, key: str) -> float:
    value = snapshot.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise BudgetExceeded(f"Required memory telemetry unavailable: {key}")
    return float(value)


def assert_budget(snapshot: dict, budget: MemoryBudget) -> None:
    """Fail closed on missing required telemetry or a current memory limit breach."""
    rss = _required_number(snapshot, "host_rss_bytes")
    available = _required_number(snapshot, "host_available_bytes")
    if rss > budget.max_host_rss_gib * GIB:
        raise BudgetExceeded("current process RSS exceeds host ceiling")
    if available < budget.min_host_available_gib * GIB:
        raise BudgetExceeded("available host memory is below reserve")
    if str(snapshot.get("device", "cpu")).startswith("cuda"):
        reserved = _required_number(snapshot, "cuda_reserved_bytes")
        free = _required_number(snapshot, "cuda_free_bytes")
        if reserved > budget.max_cuda_reserved_gib * GIB:
            raise BudgetExceeded("CUDA allocator reservation exceeds ceiling")
        if free < budget.min_cuda_free_gib * GIB:
            raise BudgetExceeded("CUDA device free memory is below reserve")


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def _reset_cuda_peak(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.reset_peak_memory_stats(device)


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    try:
        import torch
        return isinstance(exc, torch.OutOfMemoryError)
    except ImportError:
        return False


def calibrate_microbatch(
    candidates: Iterable[int],
    logical_batch: int,
    run_batch: Callable[[int], object],
    *,
    device: str = "cpu",
    budget: MemoryBudget = MemoryBudget(),
    warmups: int = 1,
    repeats: int = 3,
    snapshot_fn: Callable[[str], dict] = memory_snapshot,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Measure divisor microbatches and select the highest median example throughput.

    ``run_batch(b)`` performs one representative microbatch, releasing temporary
    outputs before returning. A training callback may do forward/backward but must
    clear its gradients and must never step the optimizer. Calibrate training and
    inference separately. The caller is responsible for restoring RNG/state.
    ``logical_batch`` stays fixed; accumulation is ``logical_batch / selected``.
    Every attempt is retained. A failed fit stops all larger candidates.
    """
    if isinstance(logical_batch, bool) or not isinstance(logical_batch, int) or logical_batch < 1:
        raise ValueError("logical_batch must be a positive integer")
    raw = list(candidates)
    if not raw or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 or logical_batch % x for x in raw):
        raise ValueError("each microbatch must be a positive divisor of logical_batch")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("warmups must be a nonnegative integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    rows: list[dict] = []
    previous: dict | None = None
    stop_reason: str | None = None
    for batch in sorted(set(raw)):
        row = {"microbatch": batch, "accumulation_steps": logical_batch // batch}
        if stop_reason:
            rows.append({**row, "status": "SKIPPED_AFTER_LIMIT", "reason": stop_reason})
            continue
        try:
            before = snapshot_fn(device)
            assert_budget(before, budget)
            row["before"] = before
            if previous is not None:
                growth = batch / previous["microbatch"] - 1
                # Reserve extra space before growing from the last observed increment.
                host_increment = max(0, previous["peak_sampled_host_rss_bytes"] - previous["before"]["host_rss_bytes"])
                if before["host_available_bytes"] - host_increment * growth * 1.2 < budget.min_host_available_gib * GIB:
                    raise BudgetExceeded("predicted host growth would consume the reserve")
                if str(device).startswith("cuda"):
                    gpu_increment = max(0, previous["peak_cuda_reserved_bytes"] - previous["before"]["cuda_reserved_bytes"])
                    extra = gpu_increment * growth * 1.2
                    if before["cuda_reserved_bytes"] + extra > budget.max_cuda_reserved_gib * GIB:
                        raise BudgetExceeded("predicted CUDA growth exceeds ceiling")
                    if before["cuda_free_bytes"] - extra < budget.min_cuda_free_gib * GIB:
                        raise BudgetExceeded("predicted CUDA growth would consume the reserve")
            _reset_cuda_peak(device)
            observations = [before]
            timings = []
            for iteration in range(warmups + repeats):
                _synchronize(device)
                started = clock()
                run_batch(batch)
                _synchronize(device)
                elapsed = clock() - started
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("measurement clock must advance by a finite positive duration")
                after = snapshot_fn(device)
                observations.append(after)
                row["last_snapshot"] = after
                assert_budget(after, budget)
                if str(device).startswith("cuda") and _required_number(after, "cuda_peak_reserved_bytes") > budget.max_cuda_reserved_gib * GIB:
                    raise BudgetExceeded("trial peak CUDA reservation exceeds ceiling")
                if iteration >= warmups:
                    timings.append(elapsed)
            median = statistics.median(timings)
            row.update({
                "status": "OK", "seconds": timings, "median_seconds": median,
                "examples_per_second": batch / median,
                "peak_sampled_host_rss_bytes": max(s["host_rss_bytes"] for s in observations),
                "peak_cuda_reserved_bytes": max((s["cuda_peak_reserved_bytes"] or 0) for s in observations[1:]) if str(device).startswith("cuda") else None,
                "after": observations[-1],
            })
            previous = row
        except BudgetExceeded as exc:
            row.update({"status": "BUDGET_REJECTED", "reason": str(exc)})
            stop_reason = str(exc)
        except BaseException as exc:
            if not _is_oom(exc):
                raise
            row.update({"status": "OOM", "reason": f"{type(exc).__name__}: {exc}"})
            stop_reason = "smaller microbatch exhausted memory"
        rows.append(row)
    successful = [r for r in rows if r["status"] == "OK"]
    best = max(successful, key=lambda r: (r["examples_per_second"], -r["microbatch"])) if successful else None
    return {
        "schema": "BPD-DT05/microbatch-calibration/v1", "device": str(device),
        "logical_batch": logical_batch, "warmups": warmups, "repeats": repeats,
        "budget": asdict(budget), "trials": rows,
        "selected_microbatch": best["microbatch"] if best else None,
        "accumulation_steps": best["accumulation_steps"] if best else None,
        "selection": "maximum median examples/second, smaller microbatch breaks ties",
        "host_peak_scope": "sampled current RSS; process lifetime peak is separate telemetry",
    }


class BoundedTensorCache:
    """An LRU cache owning immutable CPU tensor copies under an explicit byte cap.

    Cache keys should include source/checkpoint, split and transform hashes. Copies
    returned by ``get`` belong to the caller; editing one cannot change a future hit.
    Only owned tensor storage is charged. Caller copies and Python metadata remain
    outside this cap and are covered by the process memory budget.
    """

    def __init__(self, max_bytes: int, max_entries: int = 128):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._items: OrderedDict[str, tuple[object, int]] = OrderedDict()

    def put(self, key: str, value: object) -> bool:
        import torch
        if not isinstance(key, str) or not key or len(key) > 512:
            raise ValueError("nonempty bounded string key required")
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.device.type != "cpu":
            raise ValueError("cache accepts dense CPU tensors only")
        size = int(value.numel() * value.element_size())
        if size == 0 or size > self.max_bytes:
            return False
        while self._items and (
            self.bytes_used - (self._items[key][1] if key in self._items else 0) + size > self.max_bytes
            or len(self._items) - int(key in self._items) >= self.max_entries
        ):
            old_key = next(iter(self._items))
            if old_key == key:
                old_key = next(k for k in self._items if k != key)
            _, removed_size = self._items.pop(old_key)
            self.bytes_used -= removed_size
            self.evictions += 1
        # Clone before replacing the same key, so an allocation failure retains it.
        frozen = value.detach().clone(memory_format=torch.contiguous_format)
        if key in self._items:
            self.bytes_used -= self._items.pop(key)[1]
        self._items[key] = (frozen, size)
        self.bytes_used += size
        return True

    def get(self, key: str):
        if key not in self._items:
            self.misses += 1
            return None
        tensor, _ = self._items[key]
        result = tensor.clone()
        self._items.move_to_end(key)
        self.hits += 1
        return result

    def get_rows(self, key: str, indices):
        """Copy selected rows with CPU int64 indices, without cloning the full entry."""
        import torch
        if not isinstance(indices, torch.Tensor) or indices.device.type != "cpu" or indices.dtype != torch.int64 or indices.ndim != 1:
            raise ValueError("row indices must be a one-dimensional CPU int64 tensor")
        if key not in self._items:
            self.misses += 1
            return None
        tensor, _ = self._items[key]
        if tensor.ndim == 0:
            raise ValueError("row selection requires a tensor with at least one axis")
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= tensor.shape[0]):
            raise ValueError("cache row index out of range")
        result = tensor.index_select(0, indices)
        self._items.move_to_end(key)
        self.hits += 1
        return result

    def clear(self) -> None:
        self._items.clear()
        self.bytes_used = 0

    def stats(self) -> dict[str, int]:
        return {"max_bytes": self.max_bytes, "max_entries": self.max_entries, "bytes_used": self.bytes_used,
                "entries": len(self._items), "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions}
