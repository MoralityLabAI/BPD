"""FUSE-05: experimental cached primal, independent first-order adjoints."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
import torch
from torch.autograd.function import once_differentiable


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
    def forward(ctx, v, u, primal, mode, stats):
        ctx.save_for_backward(v, u)
        ctx.mode, ctx.stats = mode, stats
        return primal

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        v, u = ctx.saved_tensors
        ctx.stats.backward_nodes += 1
        nv, nu = ctx.needs_input_grad[:2]
        if ctx.mode == 'residual':
            g = (-g).T
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
        return gv, gu, None, None, None


_MISSING = object()


def _key(w):
    return tuple((id(t), t._version, tuple(t.shape), t.stride(), t.dtype, t.device,
                  t.requires_grad) for t in (w.W_target, w.V, w.U))


@contextmanager
def independent_residuals(wrappers, *, variant='product', max_bytes=16*1024**2):
    """Single fixed-weight, first-order forward graph, backward after exit.

    CPU float32 only. No checkpointing, reentry, concurrent calls, .data writes,
    external mutation of returned tensors or reuse of autograd graphs.
    Numeric cached primal payload cap excludes all other memory.
    """
    if variant not in ('residual','product','replay') or type(max_bytes) is not int or max_bytes<0:
        raise ValueError('invalid cache contract')
    ws=list(wrappers)
    if len({id(w) for w in ws})!=len(ws):raise ValueError('duplicate wrapper')
    if any(getattr(w,'_residual_cache_active',False) for w in ws):raise RuntimeError('nested residual scope')
    if any(getattr(w,'use_fused_residual',False) for w in ws):raise ValueError('disable fusion')
    stats=Stats();entries={};originals=[];active=True
    def consumed(g):
        if active:stats.consumed=True
        return g
    def build(w,original):
        def get(self):
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
            if (not torch.is_grad_enabled() or torch.is_inference_mode_enabled() or torch.is_autocast_enabled('cpu')
                or torch.get_float32_matmul_precision()!='highest' or self.W_target.requires_grad
                or not all(t.dtype==torch.float32 and t.device.type=='cpu' and t.ndim==2 and t.is_contiguous() and not t.is_inference() for t in ts)):
                return bypass()
            if old is None:
                n=self.W_target.numel()*self.W_target.element_size()
                if stats.resident_bytes+n>max_bytes:
                    stats.cap_rejections+=1;return bypass()
                with torch.no_grad():p=original() if variant=='residual' else self.V@self.U
                if p.numel()*p.element_size()!=n:raise RuntimeError('unexpected primal size')
                entries[id(self)]=(_key(self),p,p._version)
                stats.misses+=1;stats.reconstructions+=1;stats.resident_bytes+=n;stats.peak_bytes=max(stats.peak_bytes,stats.resident_bytes)
            else:
                stats.hits+=1;p=old[1]
            value=_Primal.apply(self.V,self.U,p,variant,stats)
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
