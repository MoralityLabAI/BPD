"""CURATE-03B transactional custody for native ComponentTRM training state.

One snapshot covers everything a feedback evaluation, candidate gradient or
trial minibatch step can mutate:

* parameter values and existing ``.grad`` tensors (restored in place, so the
  optimizer keeps referencing the same Parameter objects);
* every registered buffer, persistent or not, by slot, so a callback that
  replaces or resizes a buffer (``TiedComponentLinear._ensure_pass_capacity``)
  is undone, and buffers added by the callback are removed;
* every public module attribute (``mode``, ``schedule``, ``calls``,
  ``pass_offset``, ``recorded_inputs``, ``max_passes``, ``training`` ...),
  including deletion of attributes the callback created;
* AdamW per-parameter state (moments, ``step``, AMSGrad max) and every
  parameter-group hyperparameter;
* Torch CPU, Python ``random`` and legacy NumPy global RNG state, plus any
  explicitly passed ``torch.Generator`` / ``numpy.random.Generator``.

``isolated`` always restores. ``transaction`` restores unless ``commit()`` was
called and no exception escaped. ``custody_digest`` hashes the same surface so
tests can check exact restoration rather than trusting the code path.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import copy
import dataclasses
import hashlib
import random
import time
from typing import Any, Iterable

import numpy as np
import torch

_TORCH_INTERNAL = frozenset({
    '_parameters', '_buffers', '_modules', '_non_persistent_buffers_set',
    '_backward_pre_hooks', '_backward_hooks', '_is_full_backward_hook',
    '_forward_hooks', '_forward_hooks_with_kwargs', '_forward_hooks_always_called',
    '_forward_pre_hooks', '_forward_pre_hooks_with_kwargs', '_state_dict_hooks',
    '_state_dict_pre_hooks', '_load_state_dict_pre_hooks',
    '_load_state_dict_post_hooks', '_compiled_call_impl',
})


def _freeze(value: Any) -> Any:
    """Detached deep copy that tolerates tensors carrying autograd history."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [_freeze(v) for v in value]
    if isinstance(value, tuple) and not hasattr(value, '_fields'):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return {k: _freeze(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {f.name: _freeze(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return dataclasses.replace(value, **fields) if fields else copy.copy(value)
    if isinstance(value, (torch.nn.Module, torch.Generator)):
        raise TypeError('module/generator held as a plain attribute is outside custody')
    return copy.deepcopy(value)


def _module_attrs(mod: torch.nn.Module) -> dict[str, Any]:
    return {k: v for k, v in mod.__dict__.items() if k not in _TORCH_INTERNAL}


@dataclasses.dataclass
class Snapshot:
    params: list[torch.Tensor]
    grads: list[torch.Tensor | None]
    buffers: list[dict[str, torch.Tensor | None]]
    attrs: list[dict[str, Any]]
    opt_state: dict[int, dict[str, Any]] | None
    opt_groups: list[dict[str, Any]] | None
    rng: tuple
    torch_generators: list[torch.Tensor]
    numpy_generators: list[dict]


class Custody:
    """Capture/restore the full mutable training surface of one model."""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None,
                 generators: Iterable[Any] = ()) -> None:
        self.model = model
        self.optimizer = optimizer
        self.modules = list(model.modules())
        self.params = list(model.parameters())
        if any(p.device.type != 'cpu' for p in self.params):
            raise ValueError('CPU custody only')
        gens = tuple(generators)
        self.torch_generators = [g for g in gens if isinstance(g, torch.Generator)]
        self.numpy_generators = [g for g in gens if isinstance(g, np.random.Generator)]
        if len(self.torch_generators) + len(self.numpy_generators) != len(gens):
            raise TypeError('generators must be torch.Generator or numpy Generator')
        if optimizer is not None:
            opt_params = [p for g in optimizer.param_groups for p in g['params']]
            known = {id(p) for p in self.params}
            if any(id(p) not in known for p in opt_params):
                raise ValueError('optimizer owns parameters outside the model')
            self.opt_params = opt_params
        self.stats = defaultdict(float)

    def capture(self) -> Snapshot:
        t0 = time.process_time()
        opt_state = opt_groups = None
        if self.optimizer is not None:
            opt_state = {i: _freeze(dict(self.optimizer.state[p]))
                         for i, p in enumerate(self.opt_params) if p in self.optimizer.state}
            opt_groups = [{k: _freeze(v) for k, v in g.items() if k != 'params'}
                          for g in self.optimizer.param_groups]
        snap = Snapshot(
            params=[p.detach().clone() for p in self.params],
            grads=[None if p.grad is None else p.grad.detach().clone() for p in self.params],
            buffers=[{k: (None if b is None else b.detach().clone()) for k, b in m._buffers.items()}
                     for m in self.modules],
            attrs=[{k: _freeze(v) for k, v in _module_attrs(m).items()} for m in self.modules],
            opt_state=opt_state,
            opt_groups=opt_groups,
            rng=(torch.get_rng_state(), random.getstate(), np.random.get_state()),
            torch_generators=[g.get_state().clone() for g in self.torch_generators],
            numpy_generators=[copy.deepcopy(g.bit_generator.state) for g in self.numpy_generators],
        )
        self.stats['snapshots'] += 1
        self.stats['snapshot_cpu_seconds'] += time.process_time() - t0
        return snap

    @torch.no_grad()
    def restore(self, snap: Snapshot) -> None:
        t0 = time.process_time()
        for p, v, g in zip(self.params, snap.params, snap.grads, strict=True):
            if p.shape != v.shape:
                p.data = v.clone()
            else:
                p.copy_(v)
            p.grad = None if g is None else g.clone()
        for m, bufs, attrs in zip(self.modules, snap.buffers, snap.attrs, strict=True):
            for k in [k for k in m._buffers if k not in bufs]:
                del m._buffers[k]
                m._non_persistent_buffers_set.discard(k)
            for k, b in bufs.items():
                m._buffers[k] = None if b is None else b.clone()
            current = _module_attrs(m)
            for k in [k for k in current if k not in attrs]:
                delattr(m, k)
            for k, v in attrs.items():
                object.__setattr__(m, k, _freeze(v))
        if self.optimizer is not None:
            self.optimizer.state.clear()
            for i, st in snap.opt_state.items():
                self.optimizer.state[self.opt_params[i]] = _freeze(st)
            for g, saved in zip(self.optimizer.param_groups, snap.opt_groups, strict=True):
                for k in [k for k in g if k != 'params' and k not in saved]:
                    del g[k]
                for k, v in saved.items():
                    g[k] = _freeze(v)
        torch.set_rng_state(snap.rng[0])
        random.setstate(snap.rng[1])
        np.random.set_state(snap.rng[2])
        for g, st in zip(self.torch_generators, snap.torch_generators, strict=True):
            g.set_state(st)
        for g, st in zip(self.numpy_generators, snap.numpy_generators, strict=True):
            g.bit_generator.state = copy.deepcopy(st)
        self.stats['restores'] += 1
        self.stats['restore_cpu_seconds'] += time.process_time() - t0

    @contextmanager
    def isolated(self):
        """Run a block against the live model; restore everything afterwards."""
        snap = self.capture()
        try:
            yield snap
        finally:
            self.restore(snap)

    @contextmanager
    def transaction(self):
        """Commit-or-restore block. Call ``txn.commit()`` to keep mutations."""
        txn = _Transaction(self, self.capture())
        try:
            yield txn
        except BaseException:
            self.restore(txn.snapshot)
            self.stats['exception_restores'] += 1
            raise
        if not txn.committed:
            self.restore(txn.snapshot)
            self.stats['rejected'] += 1
        else:
            self.stats['committed'] += 1

    def rng_point(self):
        return (torch.get_rng_state(), random.getstate(), np.random.get_state(),
                [g.get_state().clone() for g in self.torch_generators],
                [copy.deepcopy(g.bit_generator.state) for g in self.numpy_generators])

    def replay_rng(self, point) -> None:
        """Rewind global and explicit RNG to a point so a re-evaluation sees matched draws."""
        torch.set_rng_state(point[0])
        random.setstate(point[1])
        np.random.set_state(point[2])
        for g, st in zip(self.torch_generators, point[3], strict=True):
            g.set_state(st)
        for g, st in zip(self.numpy_generators, point[4], strict=True):
            g.bit_generator.state = copy.deepcopy(st)


class _Transaction:
    def __init__(self, custody: Custody, snapshot: Snapshot) -> None:
        self.custody, self.snapshot, self.committed = custody, snapshot, False

    def commit(self) -> None:
        self.committed = True


def _hash_value(h, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().contiguous()
        h.update(f'T{t.dtype}{tuple(t.shape)}'.encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b'')
    elif isinstance(value, (list, tuple)):
        h.update(f'L{len(value)}'.encode())
        for v in value:
            _hash_value(h, v)
    elif isinstance(value, dict):
        h.update(f'D{len(value)}'.encode())
        for k in sorted(value, key=repr):
            h.update(repr(k).encode())
            _hash_value(h, value[k])
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        h.update(type(value).__name__.encode())
        for f in dataclasses.fields(value):
            h.update(f.name.encode())
            _hash_value(h, getattr(value, f.name))
    elif isinstance(value, np.ndarray):
        h.update(f'N{value.dtype}{value.shape}'.encode())
        h.update(np.ascontiguousarray(value).tobytes())
    else:
        h.update(repr(value).encode())


def custody_digest(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None,
                   generators: Iterable[Any] = (), *, include_rng: bool = True) -> str:
    """SHA-256 over the exact surface ``Custody`` captures."""
    h = hashlib.sha256()
    params = list(model.parameters())
    for p in params:
        _hash_value(h, p)
        _hash_value(h, p.grad if p.grad is not None else 'nograd')
    for m in model.modules():
        _hash_value(h, {k: ('None' if b is None else b) for k, b in m._buffers.items()})
        _hash_value(h, sorted(m._non_persistent_buffers_set))
        _hash_value(h, {k: v for k, v in _module_attrs(m).items()})
    if optimizer is not None:
        index = {id(p): i for i, p in enumerate(params)}
        for g in optimizer.param_groups:
            _hash_value(h, {k: v for k, v in g.items() if k != 'params'})
            for p in g['params']:
                h.update(str(index[id(p)]).encode())
                _hash_value(h, dict(optimizer.state[p]) if p in optimizer.state else 'nostate')
    if include_rng:
        _hash_value(h, torch.get_rng_state())
        _hash_value(h, random.getstate())
        _hash_value(h, list(np.random.get_state()))
        for g in generators:
            if isinstance(g, torch.Generator):
                _hash_value(h, g.get_state())
            else:
                _hash_value(h, g.bit_generator.state)
    return h.hexdigest()
