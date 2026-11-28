"""FUSE-05: experimental cached primal, independent first-order adjoints."""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import MethodType
import torch
from torch.autograd.function import once_differentiable

from .cuda_guard import (
    current_stream_token,
    device_and_precision_supported,
    runtime_precision_state,
    target_storage_dtype,
    validate_cuda_options,
)


@dataclass
class Stats:
    calls: int = 0
    hits: int = 0
    misses: int = 0
    bypasses: int = 0
    reconstructions: int = 0
    backward_nodes: int = 0
    backward_replays: int = 0
    cap_rejections: int = 0
    resident_bytes: int = 0
    peak_bytes: int = 0
    consumed: bool = False


class _Primal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, u, primal, mode, stats, ac_enabled=False, ac_dtype=None):
        ctx.save_for_backward(v, u)
        ctx.mode, ctx.stats = mode, stats
        ctx.dev_type = v.device.type
        ctx.ac_enabled = bool(ac_enabled)
        ctx.ac_dtype = ac_dtype
        return primal

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        v, u = ctx.saved_tensors
        ctx.stats.backward_nodes += 1
        nv, nu = ctx.needs_input_grad[:2]
        if ctx.mode == 'residual':
            g = (-g).T
        ac_ctx = (
            torch.amp.autocast(ctx.dev_type, enabled=True, dtype=ctx.ac_dtype)
            if ctx.ac_enabled else nullcontext()
        )
        with ac_ctx:
            if ctx.mode == 'replay':
                ctx.stats.backward_replays += 1
                with torch.enable_grad():
                    a = v.detach().requires_grad_(nv)
                    b = u.detach().requires_grad_(nu)
                    z = a @ b
                    grads = iter(torch.autograd.grad(z, tuple(t for t in (a,b) if t.requires_grad), g))
                    gv = next(grads) if nv else None
                    gu = next(grads) if nu else None
            else:
                gv = g @ u.T if nv else None
                gu = v.T @ g if nu else None
        if gv is not None and gv.dtype != v.dtype:
            gv = gv.to(v.dtype)
        if gu is not None and gu.dtype != u.dtype:
            gu = gu.to(u.dtype)
        return gv, gu, None, None, None, None, None


class _SharedCastPrimal(torch.autograd.Function):
    """FUSE-13: Independent first-order adjoint for shared compute-dtype residual payload.

    Takes compute-dtype factor tensors (vc, uc) obtained through PyTorch's native
    autocast cast path so ToCopyBackward0 sharing matches native autograd bit-for-bit.
    """

    @staticmethod
    def forward(ctx, vc, uc, payload, stats):
        ctx.save_for_backward(vc, uc)
        ctx.stats = stats
        ctx.dev_type = vc.device.type
        return payload

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        vc, uc = ctx.saved_tensors
        ctx.stats.backward_nodes += 1
        nv, nu = ctx.needs_input_grad[:2]
        with torch.autocast(ctx.dev_type, enabled=False):
            dz = (-g.float().T).to(vc.dtype)
            gv = dz @ uc.T if nv else None
            gu = vc.T @ dz if nu else None
        return gv, gu, None, None


def _resolve_autocast_factors(v: torch.Tensor, u: torch.Tensor, compute_dtype: torch.dtype):
    """Obtain (vc, uc) in compute_dtype sharing native autocast ToCopyBackward0 nodes (0 FLOPs)."""
    if v.dtype == compute_dtype:
        vc = v
    elif v.requires_grad:
        zv = torch.mm(
            torch.empty((0, v.shape[0]), device=v.device, dtype=v.dtype, requires_grad=True),
            v,
        )
        vc = zv.grad_fn._saved_mat2
    else:
        vc = v.to(compute_dtype)

    if u.dtype == compute_dtype:
        uc = u
    elif u.requires_grad:
        zu = torch.mm(
            u,
            torch.empty((u.shape[1], 0), device=u.device, dtype=u.dtype, requires_grad=True),
        )
        uc = zu.grad_fn._saved_self
    else:
        uc = u.to(compute_dtype)
    return vc, uc


_MISSING = object()


def _key(w):
    ts = (w.W_target, w.V, w.U)
    return (
        tuple((id(t), t._version, tuple(t.shape), t.stride(), t.dtype, t.device,
               t.requires_grad) for t in ts),
        runtime_precision_state(w.W_target.device.type),
    )


@contextmanager
def independent_residuals(
    wrappers,
    *,
    variant='product',
    max_bytes=16*1024**2,
    allow_cuda: bool = False,
    precision_mode: str = 'fp32',
):
    """Single fixed-weight, first-order forward graph, backward after exit.

    CPU float32 by default; experimental CUDA requires allow_cuda=True and
    single sequential stream ownership. No checkpointing, reentry, concurrent
    calls, .data writes, external mutation of returned tensors or graph reuse.
    Numeric cached primal payload cap excludes all other memory.
    """
    validate_cuda_options(allow_cuda, precision_mode)
    if variant not in ('residual','product','replay') or type(max_bytes) is not int or max_bytes<0:
        raise ValueError('invalid cache contract')
    ws=list(wrappers)
    if len({id(w) for w in ws})!=len(ws):raise ValueError('duplicate wrapper')
    if any(getattr(w,'_residual_cache_active',False) for w in ws):raise RuntimeError('nested residual scope')
    if any(getattr(w,'use_fused_residual',False) for w in ws):raise ValueError('disable fusion')
    stats=Stats();entries={};originals=[];active=True;owner_stream=None
    def consumed(g):
        if active:stats.consumed=True
        return g
    def build(w,original):
        def get(self):
            nonlocal owner_stream
            stats.calls+=1
            if stats.consumed:raise RuntimeError('backward consumed scope')
            if getattr(self,'use_fused_residual',False):raise RuntimeError('fusion enabled')
            def bypass():
                stats.bypasses+=1;stats.reconstructions+=1
                return original()
            ts=(self.W_target,self.V,self.U);old=entries.get(id(self))
            if old is not None and (any(t.is_inference() for t in ts) or old[0]!=_key(self) or old[2]!=old[1]._version):
                stats.consumed=True
                raise RuntimeError('parameter or cached primal mutated')
            if old is not None and self.W_target.device.type == 'cuda':
                tok = current_stream_token(self.W_target.device)
                if owner_stream is not None and tok != owner_stream:
                    stats.consumed=True
                    raise RuntimeError('CUDA stream ownership violation')
            if (not torch.is_grad_enabled() or torch.is_inference_mode_enabled()
                or self.W_target.requires_grad
                or not device_and_precision_supported(
                    ts, allow_cuda=allow_cuda, precision_mode=precision_mode)):
                return bypass()
            if self.W_target.device.type == 'cuda':
                tok = current_stream_token(self.W_target.device)
                if owner_stream is None:
                    owner_stream = tok
                elif tok != owner_stream:
                    stats.consumed=True
                    raise RuntimeError('CUDA stream ownership violation')
            if old is None:
                n=self.W_target.numel()*4
                if stats.resident_bytes+n>max_bytes:
                    stats.cap_rejections+=1;return bypass()
                with torch.no_grad():p=original() if variant=='residual' else self.V@self.U
                expected_n = p.numel()*p.element_size()
                if variant=='residual' and (p.dtype != torch.float32 or expected_n != n):
                    raise RuntimeError('unexpected primal size')
                if variant!='residual' and expected_n != self.W_target.numel()*p.element_size():
                    raise RuntimeError('unexpected primal size')
                entries[id(self)]=(_key(self),p,p._version)
                stats.misses+=1;stats.reconstructions+=1;stats.resident_bytes+=expected_n;stats.peak_bytes=max(stats.peak_bytes,stats.resident_bytes)
            else:
                stats.hits+=1;p=old[1]
            dev_type = self.V.device.type
            ac_enabled = bool(torch.is_autocast_enabled(dev_type))
            ac_dtype = torch.get_autocast_dtype(dev_type) if ac_enabled else None
            value=_Primal.apply(self.V,self.U,p,variant,stats,ac_enabled,ac_dtype)
            if value.requires_grad:value.register_hook(consumed)
            return value if variant=='residual' else self.W_target.float()-value.T
        return MethodType(get,w)
    try:
        for w in ws:
            orig=w.weight_delta;originals.append((w,w.__dict__.get('weight_delta',_MISSING),w.__dict__.get('_residual_cache_active',_MISSING)))
            w._residual_cache_active=True;w.weight_delta=build(w,orig)
        yield stats
    finally:
        active=False;entries.clear();stats.resident_bytes=0
        for w,method,flag in reversed(originals):
            if method is _MISSING:del w.weight_delta
            else:w.weight_delta=method
            if flag is _MISSING:del w._residual_cache_active
            else:w._residual_cache_active=flag
