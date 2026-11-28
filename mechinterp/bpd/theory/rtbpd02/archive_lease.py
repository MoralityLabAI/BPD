"""Bounded local CPU lease for the optional archive replay."""
from contextlib import contextmanager
import ctypes as ct
import os
from pathlib import Path
import sys
import threading
import time


class MemoryStatus(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("length", "load")] + [
        (name, ct.c_uint64) for name in (
            "total_phys", "available_phys", "total_page", "available_page",
            "total_virtual", "available_virtual", "available_extended_virtual",
        )
    ]


class ProcessMemory(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("length", "page_faults")] + [
        (name, ct.c_size_t) for name in (
            "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
            "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile", "private",
        )
    ]


def memory():
    if os.name == "nt":
        kernel = ct.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ct.c_void_p
        kernel.GlobalMemoryStatusEx.argtypes = (ct.POINTER(MemoryStatus),)
        kernel.K32GetProcessMemoryInfo.argtypes = (ct.c_void_p, ct.POINTER(ProcessMemory), ct.c_uint32)
        system, process = MemoryStatus(), ProcessMemory()
        system.length, process.length = ct.sizeof(system), ct.sizeof(process)
        if not kernel.GlobalMemoryStatusEx(ct.byref(system)):
            raise ct.WinError(ct.get_last_error())
        if not kernel.K32GetProcessMemoryInfo(kernel.GetCurrentProcess(), ct.byref(process), process.length):
            raise ct.WinError(ct.get_last_error())
        return int(system.available_phys), int(process.private)
    if sys.platform == "linux":
        def kib(path, names):
            rows = (line.split() for line in Path(path).read_text().splitlines())
            return 1024 * sum(int(fields[1]) for fields in rows if fields and fields[0] in names)
        return (
            kib("/proc/meminfo", {"MemAvailable:"}),
            kib("/proc/self/smaps_rollup", {"Private_Clean:", "Private_Dirty:", "Private_Hugetlb:"}),
        )
    raise RuntimeError("archive resource admission requires Windows or Linux process memory counters")


def lower_priority():
    if os.name == "nt":
        kernel = ct.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ct.c_void_p
        kernel.SetPriorityClass.argtypes = (ct.c_void_p, ct.c_uint32)
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise ct.WinError(ct.get_last_error())
    else:
        os.nice(5)


@contextmanager
def lease(spec):
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    available, _ = memory()
    if available < spec["minimum_start_available_bytes"]:
        raise RuntimeError(f"archive start requires 5 GiB available, observed {available / 2**30:.2f} GiB")
    lower_priority()
    started = time.monotonic()
    stop = threading.Event()
    samples = []

    def check():
        available, private = memory()
        elapsed = time.monotonic() - started
        samples.append((available, private, elapsed))
        if (available < spec["minimum_reserve_bytes"]
                or private > spec["max_private_bytes"]
                or elapsed > spec["max_seconds"]):
            raise RuntimeError("archive lease limit exceeded")

    def monitor():
        while not stop.wait(2.0):
            try:
                check()
            except Exception as error:
                print(error, file=sys.stderr, flush=True)
                os._exit(3)  # Bound even an in-flight BLAS or archive read.
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        check()
        yield check, samples
        check()
    finally:
        stop.set()
        watcher.join()
