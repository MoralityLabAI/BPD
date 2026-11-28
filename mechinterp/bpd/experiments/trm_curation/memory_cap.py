"""Fail-closed process-tree 6 GiB memory enforcement (Linux cgroup v2 & Windows Job Object).

On Linux: verifies active cgroup v2 ``memory.max == 6442450944`` and swap disabled
(``memory.swap.max == 0``), charging process-tree stats from cgroup files.
On Windows: binds the current process (and all spawned descendants) into a
Win32 Job Object with ``JOB_OBJECT_LIMIT_JOB_MEMORY`` set to ``max_bytes``
(6 GiB commit ceiling across the entire process tree) plus ``psutil`` RSS and
host available-RAM monitoring. Explicitly labels platform memory semantics and
fails closed if OS-level enforcement cannot be established.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import sys
import time

import psutil

DEFAULT_CAP_BYTES = 6 * 1024 * 1024 * 1024  # 6,442,450,944 bytes (6 GiB)


@dataclass
class MemoryAccounting:
    platform: str
    enforcement_mechanism: str
    semantics: str
    cap_bytes: int
    enforced: bool
    peak_job_commit_bytes: int | None
    peak_tree_rss_bytes: int
    current_tree_rss_bytes: int
    host_available_bytes_min: int
    wall_seconds: float
    process_cpu_seconds: float


class ProcessTreeMemoryCap:
    """Hard OS-level process-tree memory cap with fail-closed semantics."""

    def __init__(self, cap_bytes: int = DEFAULT_CAP_BYTES, *,
                 allow_nested_windows_job: bool = True) -> None:
        if type(cap_bytes) is not int or cap_bytes <= 0:
            raise ValueError('cap_bytes must be a positive int')
        self.cap_bytes = cap_bytes
        self.allow_nested_windows_job = allow_nested_windows_job
        self._job_handle = None
        self._enforced = False
        self._mechanism = 'none'
        self._semantics = 'none'
        self._start_wall = 0.0
        self._start_cpu = 0.0
        self._peak_tree_rss = 0
        self._min_host_avail = int(psutil.virtual_memory().available)
        self._cgroup_dir: Path | None = None

    def _enter_windows(self) -> None:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('ReadOperationCount', ctypes.c_uint64),
                ('WriteOperationCount', ctypes.c_uint64),
                ('OtherOperationCount', ctypes.c_uint64),
                ('ReadTransferCount', ctypes.c_uint64),
                ('WriteTransferCount', ctypes.c_uint64),
                ('OtherTransferCount', ctypes.c_uint64),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', ctypes.c_int64),
                ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t),
            ]

        self._ext_info_cls = JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD
        ]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD)
        ]
        k32.QueryInformationJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.GetCurrentProcess.argtypes = []
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        self._k32 = k32

        job = k32.CreateJobObjectW(None, None)
        if not job:
            err = ctypes.get_last_error()
            raise RuntimeError(f'Fail-closed: CreateJobObjectW failed (err={err})')

        JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        info.JobMemoryLimit = self.cap_bytes

        ok = k32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info)
        )
        if not ok:
            err = ctypes.get_last_error()
            k32.CloseHandle(job)
            raise RuntimeError(f'Fail-closed: SetInformationJobObject failed (err={err})')

        # Verify queried limit equals requested cap_bytes
        verify = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        ret_len = wintypes.DWORD(0)
        ok_q = k32.QueryInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(verify), ctypes.sizeof(verify), ctypes.byref(ret_len)
        )
        if not ok_q or int(verify.JobMemoryLimit) != self.cap_bytes:
            k32.CloseHandle(job)
            raise RuntimeError('Fail-closed: JobObject limit verification mismatch')

        proc = k32.GetCurrentProcess()
        ok_assign = k32.AssignProcessToJobObject(job, proc)
        if not ok_assign:
            err = ctypes.get_last_error()
            k32.CloseHandle(job)
            raise RuntimeError(
                f'Fail-closed: AssignProcessToJobObject failed (err={err}); '
                'hard memory cap unavailable'
            )

        self._job_handle = job
        self._enforced = True
        self._mechanism = 'windows_job_object'
        self._semantics = (
            'Win32 Job Object hard commit-memory cap (JOB_OBJECT_LIMIT_JOB_MEMORY) '
            'across current process and all descendants, paired with psutil tree RSS '
            'and host available-RAM monitoring'
        )

    def _enter_linux(self) -> None:
        cg_proc = Path('/proc/self/cgroup')
        if not cg_proc.is_file():
            raise RuntimeError('Fail-closed: /proc/self/cgroup unavailable')
        rel = None
        for line in cg_proc.read_text(encoding='utf-8').splitlines():
            if line.startswith('0::'):
                rel = line[3:].strip()
                break
        if not rel:
            raise RuntimeError('Fail-closed: cgroup v2 hierarchy not active')
        cg_dir = Path('/sys/fs/cgroup') / rel.lstrip('/')
        mem_max = (cg_dir / 'memory.max').read_text(encoding='utf-8').strip()
        swap_max = (cg_dir / 'memory.swap.max').read_text(encoding='utf-8').strip()
        if mem_max == 'max' or int(mem_max) > self.cap_bytes:
            raise RuntimeError(
                f'Fail-closed: cgroup v2 memory.max={mem_max} exceeds cap {self.cap_bytes}'
            )
        if swap_max != '0':
            raise RuntimeError(
                f'Fail-closed: cgroup v2 memory.swap.max={swap_max} (swap must be disabled=0)'
            )
        self._cgroup_dir = cg_dir
        self._enforced = True
        self._mechanism = 'linux_cgroup_v2'
        self._semantics = (
            f'Linux cgroup v2 memory.max={mem_max}, memory.swap.max=0 hard resident+cache cap'
        )

    def sample(self) -> tuple[int, int]:
        """Sample current process-tree RSS and host available bytes."""
        root = psutil.Process(os.getpid())
        rss = root.memory_info().rss
        for child in root.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        avail = int(psutil.virtual_memory().available)
        if rss > self._peak_tree_rss:
            self._peak_tree_rss = rss
        if avail < self._min_host_avail:
            self._min_host_avail = avail
        return rss, avail

    def query_peak_job_bytes(self) -> int | None:
        if self._mechanism == 'windows_job_object' and self._job_handle:
            import ctypes
            from ctypes import wintypes
            verify = self._ext_info_cls()
            ret_len = wintypes.DWORD(0)
            ok = self._k32.QueryInformationJobObject(
                self._job_handle, 9, ctypes.byref(verify),
                ctypes.sizeof(verify), ctypes.byref(ret_len)
            )
            if ok:
                return int(verify.PeakJobMemoryUsed)
        if self._mechanism == 'linux_cgroup_v2' and self._cgroup_dir:
            peak_file = self._cgroup_dir / 'memory.peak'
            if peak_file.is_file():
                return int(peak_file.read_text(encoding='utf-8').strip())
        return None

    def accounting(self) -> MemoryAccounting:
        rss, _ = self.sample()
        return MemoryAccounting(
            platform=f'{platform.system()} {platform.release()}',
            enforcement_mechanism=self._mechanism,
            semantics=self._semantics,
            cap_bytes=self.cap_bytes,
            enforced=self._enforced,
            peak_job_commit_bytes=self.query_peak_job_bytes(),
            peak_tree_rss_bytes=self._peak_tree_rss,
            current_tree_rss_bytes=rss,
            host_available_bytes_min=self._min_host_avail,
            wall_seconds=time.perf_counter() - self._start_wall,
            process_cpu_seconds=time.process_time() - self._start_cpu,
        )

    def __enter__(self) -> 'ProcessTreeMemoryCap':
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        if sys.platform == 'win32':
            self._enter_windows()
        elif sys.platform.startswith('linux'):
            self._enter_linux()
        else:
            raise RuntimeError(f'Fail-closed: unsupported platform {sys.platform} for hard cap')
        self.sample()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.sample()
        # Do NOT close the Job Object handle with KILL_ON_JOB_CLOSE while we are
        # still inside the current process, or Windows would terminate us!
        # Keep handle alive until process exit (OS cleans up automatically),
        # or only release when exiting a child launcher.
        return None


def run_capped_subprocess(cmd: list[str], *, cap_bytes: int = DEFAULT_CAP_BYTES,
                          cwd: str | Path | None = None,
                          timeout: float | None = None) -> dict:
    """Launch a child process under an isolated 6 GiB Win32 Job Object cap."""
    import subprocess
    if sys.platform != 'win32':
        raise RuntimeError('run_capped_subprocess currently implements Win32 Job Object isolation')
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(f, ctypes.c_uint64) for f in (
            'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
            'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount'
        )]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('PerProcessUserTimeLimit', ctypes.c_int64),
            ('PerJobUserTimeLimit', ctypes.c_int64),
            ('LimitFlags', wintypes.DWORD),
            ('MinimumWorkingSetSize', ctypes.c_size_t),
            ('MaximumWorkingSetSize', ctypes.c_size_t),
            ('ActiveProcessLimit', wintypes.DWORD),
            ('Affinity', ctypes.c_size_t),
            ('PriorityClass', wintypes.DWORD),
            ('SchedulingClass', wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ('IoInfo', IO_COUNTERS),
            ('ProcessMemoryLimit', ctypes.c_size_t),
            ('JobMemoryLimit', ctypes.c_size_t),
            ('PeakProcessMemoryUsed', ctypes.c_size_t),
            ('PeakJobMemoryUsed', ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD
    ]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD)
    ]
    k32.QueryInformationJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL

    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise RuntimeError('CreateJobObjectW failed')
    try:
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x00000200 | 0x00002000
        info.JobMemoryLimit = cap_bytes
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise RuntimeError('SetInformationJobObject failed')
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if not k32.AssignProcessToJobObject(job, int(proc._handle)):
            proc.kill()
            raise RuntimeError('AssignProcessToJobObject failed on child')
        out, err = proc.communicate(timeout=timeout)
        elapsed = time.perf_counter() - t0
        q = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        ret_len = wintypes.DWORD(0)
        k32.QueryInformationJobObject(job, 9, ctypes.byref(q), ctypes.sizeof(q), ctypes.byref(ret_len))
        return {
            'returncode': proc.returncode,
            'stdout': out,
            'stderr': err,
            'wall_seconds': elapsed,
            'peak_job_commit_bytes': int(q.PeakJobMemoryUsed),
            'cap_bytes': cap_bytes,
            'mechanism': 'windows_job_object_subprocess',
        }
    finally:
        k32.CloseHandle(job)
