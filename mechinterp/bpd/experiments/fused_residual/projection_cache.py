"""FUSE-14: Bounded, step-local frozen projection BF16 cast compactification.

For frozen, bias-free, contiguous FP32 ``nn.Linear`` projections (e.g., ``qkv_proj``,
``gate_up_proj``) executed under eager ``torch.autocast(..., dtype=torch.bfloat16)``,
PyTorch allocates a fresh BF16 weight copy saved for backward on every recurrent visit.
This pool caches one detached BF16 weight cast per physical projection per training step
and serves it via ``F.linear(x, cast)`` so input adjoints execute individually with
identical kernels, input casts, and layouts.

Strict fail-closed contract:
- Opt-in only; production defaults remain untouched.
- BF16 eager autocast only (FP16 remains HOLD; FP32/TF32 stay native).
- Rejects trainable, biased, subclassed, parametrized, non-contiguous, inference,
  or prepatched ``nn.Linear`` modules.
- Rejects weight replacement, versioned mutation, stream migration, or nested scopes.
- Strictly step-local: closed before ``optimizer.step()``; no cross-step reuse.
- Trainable projection cast sharing is explicitly excluded.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize as _parametrize

from .cuda_guard import current_stream_token, is_stream_capturing

TARGET_MODES = ("block_projs", "qkv_only", "gate_up_only", "all")
_MISSING = object()


def _matches_target(name: str, targets: str | tuple[str, ...]) -> bool:
    if isinstance(targets, str):
        if targets == "block_projs":
            return name.endswith(".qkv_proj") or name.endswith(".gate_up_proj") or name in ("qkv_proj", "gate_up_proj")
        if targets == "qkv_only":
            return name.endswith(".qkv_proj") or name == "qkv_proj"
        if targets == "gate_up_only":
            return name.endswith(".gate_up_proj") or name == "gate_up_proj"
        if targets == "all":
            return True
        raise ValueError(f"unknown projection_cache_targets: {targets!r}")
    return any(name == t or name.endswith("." + t) for t in targets)


def is_eligible_frozen_linear(m: nn.Module) -> bool:
    """Static structural check for dense, bias-free, frozen FP32 exact nn.Linear."""
    if type(m) is not nn.Linear:
        return False
    if m.bias is not None:
        return False
    if _parametrize.is_parametrized(m):
        return False
    w = m.weight
    if (
        not isinstance(w, torch.Tensor)
        or w.ndim != 2
        or w.requires_grad
        or w.dtype != torch.float32
        or not w.is_contiguous()
        or w.is_inference()
    ):
        return False
    return True


def eligible_frozen_projections(
    root: nn.Module,
    *,
    targets: str | tuple[str, ...] = "block_projs",
) -> list[tuple[str, nn.Linear]]:
    """Discover eligible frozen bias-free FP32 ``nn.Linear`` modules in deterministic order."""
    if isinstance(targets, str) and targets not in TARGET_MODES:
        raise ValueError(f"unknown projection_cache_targets: {targets!r}")
    found: list[tuple[str, nn.Linear]] = []
    seen: set[int] = set()
    for name, mod in root.named_modules():
        if id(mod) in seen:
            continue
        if not _matches_target(name, targets):
            continue
        if is_eligible_frozen_linear(mod):
            seen.add(id(mod))
            found.append((name, mod))
    return found


@dataclass
class ProjectionScopeStats:
    calls: int = 0
    hits: int = 0
    misses: int = 0
    fallbacks: int = 0
    cap_rejections: int = 0
    resident_bytes: int = 0
    peak_bytes: int = 0
    stream_checks: int = 0
    per_site_hits: dict[str, int] = field(default_factory=dict)
    per_site_misses: dict[str, int] = field(default_factory=dict)
    per_site_fallbacks: dict[str, int] = field(default_factory=dict)


class FrozenProjectionCastPool:
    """Step-local pool of detached BF16 casts for frozen FP32 linear projections."""

    def __init__(
        self,
        named_modules: Iterable[tuple[str, nn.Linear]],
        *,
        max_bytes: int | None = None,
        allow_cuda: bool = False,
        allow_recompute: bool = False,
    ) -> None:
        if type(allow_cuda) is not bool:
            raise TypeError("allow_cuda must be bool")
        if type(allow_recompute) is not bool:
            raise TypeError("allow_recompute must be bool")
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ValueError("projection_cache_max_bytes must be None or a nonnegative int")
        items = list(named_modules)
        seen_ids: set[int] = set()
        for name, m in items:
            if id(m) in seen_ids:
                raise ValueError(f"duplicate projection module: {name}")
            if type(m) is not nn.Linear:
                raise ValueError(f"expected exact nn.Linear for {name}, got {type(m).__name__}")
            if m.bias is not None:
                raise ValueError(f"biased projection rejected: {name}")
            if _parametrize.is_parametrized(m):
                raise ValueError(f"parametrized projection rejected: {name}")
            if getattr(m, "_projection_cache_active", False):
                raise RuntimeError(f"nested projection cast scope: {name}")
            if "forward" in m.__dict__:
                raise ValueError(f"prepatched projection rejected: {name}")
            seen_ids.add(id(m))

        self.named = tuple(items)
        self.max_bytes = max_bytes
        self.allow_cuda = allow_cuda
        self.allow_recompute = allow_recompute
        self.owner_stream: tuple[str, int, int] | None = None
        self.entries: dict[int, tuple[torch.Tensor, int, tuple, torch.Tensor]] = {}
        self.resident_bytes = 0
        self.peak_bytes = 0
        self.scope_count = 0
        self.stream_checks = 0
        self.total_hits = 0
        self.total_misses = 0
        self.total_fallbacks = 0
        self.total_cap_rejections = 0
        self.per_site_hits: dict[str, int] = {n: 0 for n, _ in self.named}
        self.per_site_misses: dict[str, int] = {n: 0 for n, _ in self.named}
        self.per_site_fallbacks: dict[str, int] = {n: 0 for n, _ in self.named}
        self.active = False
        self.closed = False
        self.poisoned = False

    def _check_stream(self, device: torch.device) -> tuple[str, int, int]:
        tok = current_stream_token(device)
        if device.type == "cuda":
            if is_stream_capturing(device):
                self.poisoned = True
                raise RuntimeError("CUDA graph capture unsupported in frozen projection scope")
            if self.owner_stream is None:
                self.owner_stream = tok
            elif tok != self.owner_stream:
                self.poisoned = True
                raise RuntimeError("CUDA stream ownership violation in frozen projection scope")
            self.stream_checks += 1
        return tok

    def _verify_entry(self, name: str, m: nn.Linear, entry) -> None:
        held_w, version, key, cast = entry
        w = m.weight
        if (
            w is not held_w
            or w._version != version
            or w.requires_grad
            or w.is_inference()
            or m.bias is not None
            or _parametrize.is_parametrized(m)
            or cast._version != 0
        ):
            self.poisoned = True
            raise RuntimeError(f"frozen projection changed during scope: {name}")
        if w.device.type == "cuda":
            self._check_stream(w.device)

    def verify_all(self) -> None:
        for name, m in self.named:
            entry = self.entries.get(id(m))
            if entry is not None:
                self._verify_entry(name, m, entry)

    def close(self) -> None:
        if self.active:
            raise RuntimeError("cannot close active frozen projection pool")
        try:
            if not self.poisoned and not self.closed:
                self.verify_all()
        finally:
            self.closed = True
            self.entries.clear()
            self.resident_bytes = 0

    def receipt(self) -> dict[str, object]:
        return {
            "sites": [n for n, _ in self.named],
            "n_sites": len(self.named),
            "allow_cuda": self.allow_cuda,
            "allow_recompute": self.allow_recompute,
            "max_bytes": self.max_bytes,
            "peak_bytes": self.peak_bytes,
            "resident_bytes": self.resident_bytes,
            "scopes": self.scope_count,
            "hits": self.total_hits,
            "misses": self.total_misses,
            "fallbacks": self.total_fallbacks,
            "cap_rejections": self.total_cap_rejections,
            "stream_checks": self.stream_checks,
            "owner_stream": list(self.owner_stream) if self.owner_stream is not None else None,
            "per_site_hits": dict(self.per_site_hits),
            "per_site_misses": dict(self.per_site_misses),
            "per_site_fallbacks": dict(self.per_site_fallbacks),
            "poisoned": self.poisoned,
            "closed": self.closed,
        }

    @contextmanager
    def scope(self):
        """Patch eligible frozen ``nn.Linear`` forwards for this bounded execution window."""
        if self.closed or self.active or self.poisoned:
            raise RuntimeError("unavailable frozen projection cast pool")
        if any(getattr(m, "_projection_cache_active", False) for _, m in self.named):
            raise RuntimeError("nested frozen projection cast scope")
        self.verify_all()
        self.scope_count += 1
        stats = ProjectionScopeStats(
            peak_bytes=self.resident_bytes,
            per_site_hits={n: 0 for n, _ in self.named},
            per_site_misses={n: 0 for n, _ in self.named},
            per_site_fallbacks={n: 0 for n, _ in self.named},
        )
        originals = []
        self.active = True

        def build(name: str, mod: nn.Linear, original):
            def forward(self_m, x: torch.Tensor):
                stats.calls += 1
                if self.closed or self.poisoned or not self.active:
                    raise RuntimeError("unavailable frozen projection cast pool")
                w = self_m.weight
                old = self.entries.get(id(self_m))
                if old is not None:
                    self._verify_entry(name, self_m, old)

                def fallback():
                    stats.fallbacks += 1
                    stats.per_site_fallbacks[name] += 1
                    self.total_fallbacks += 1
                    self.per_site_fallbacks[name] += 1
                    return original(x)

                dev = w.device.type
                if dev == "cuda" and not self.allow_cuda:
                    return fallback()
                if dev not in ("cpu", "cuda"):
                    return fallback()
                if not torch.is_autocast_enabled(dev):
                    return fallback()
                ac_dtype = torch.get_autocast_dtype(dev)
                # Strict FUSE-14 gate: BF16 only (FP16 remains HOLD; FP32/TF32 incumbent).
                if ac_dtype != torch.bfloat16:
                    return fallback()
                if (
                    type(self_m) is not nn.Linear
                    or self_m.bias is not None
                    or _parametrize.is_parametrized(self_m)
                    or w.requires_grad
                    or w.dtype != torch.float32
                    or not w.is_contiguous()
                    or w.is_inference()
                    or x.device != w.device
                    or x.dtype not in (torch.float32, ac_dtype)
                    or not torch.is_grad_enabled()
                    or torch.is_inference_mode_enabled()
                    or torch.jit.is_tracing()
                ):
                    return fallback()

                stream_tok = self._check_stream(w.device)
                stats.stream_checks = self.stream_checks
                key = (w.device, ac_dtype, stream_tok, tuple(w.shape), tuple(w.stride()), w.data_ptr())
                if old is not None:
                    if old[2] != key:
                        self.poisoned = True
                        raise RuntimeError(f"projection cast scope key changed: {name}")
                    cast = old[3]
                    stats.hits += 1
                    stats.per_site_hits[name] += 1
                    self.total_hits += 1
                    self.per_site_hits[name] += 1
                else:
                    nbytes = w.numel() * 2  # bfloat16 = 2 bytes/elem
                    if self.max_bytes is not None and self.resident_bytes + nbytes > self.max_bytes:
                        stats.cap_rejections += 1
                        self.total_cap_rejections += 1
                        return fallback()
                    with torch.no_grad():
                        cast = w.detach().to(ac_dtype)
                    if (
                        cast.dtype != ac_dtype
                        or cast.device != w.device
                        or cast.shape != w.shape
                        or not cast.is_contiguous()
                        or cast.untyped_storage().nbytes() != nbytes
                    ):
                        self.poisoned = True
                        raise RuntimeError(f"unexpected frozen projection cast storage: {name}")
                    if any(
                        cast.untyped_storage().data_ptr() == e[3].untyped_storage().data_ptr()
                        for e in self.entries.values()
                    ):
                        self.poisoned = True
                        raise RuntimeError(f"unexpected cross-site projection cast alias: {name}")
                    self.entries[id(self_m)] = (w, w._version, key, cast)
                    self.resident_bytes += nbytes
                    self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
                    stats.misses += 1
                    stats.per_site_misses[name] += 1
                    self.total_misses += 1
                    self.per_site_misses[name] += 1

                stats.resident_bytes = self.resident_bytes
                stats.peak_bytes = max(stats.peak_bytes, self.resident_bytes)
                return F.linear(x, cast)

            return MethodType(forward, mod)

        try:
            for name, m in self.named:
                if "forward" in m.__dict__:
                    raise RuntimeError(f"prepatched projection encountered: {name}")
                originals.append(
                    (
                        m,
                        m.__dict__.get("forward", _MISSING),
                        m.__dict__.get("_projection_cache_active", _MISSING),
                    )
                )
                m._projection_cache_active = True
                m.forward = build(name, m, m.forward)
            yield stats
            self.verify_all()
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self.active = False
            for m, orig_fwd, orig_flag in reversed(originals):
                if orig_fwd is _MISSING:
                    if "forward" in m.__dict__:
                        del m.forward
                else:
                    m.forward = orig_fwd
                if orig_flag is _MISSING:
                    if "_projection_cache_active" in m.__dict__:
                        del m._projection_cache_active
                else:
                    m._projection_cache_active = orig_flag
