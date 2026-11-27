"""CPU-only one-step optimizer probes with rollback of tensor, native BPD and RNG state."""
from __future__ import annotations

import copy
import random
import time
from typing import Iterable

import numpy as np
import torch


def _snapshot_native_state(model) -> list[tuple[object, dict]]:
    """Capture mutable Python/BPD pass-tracking attributes on model modules."""
    attrs = (
        'mode', 'schedule', 'intervention', 'pass_offset', 'calls',
        'record_inputs', 'detach_recorded_inputs', 'update_running_means',
        'recorded_inputs',
    )
    snap = []
    for mod in model.modules():
        saved = {}
        for name in attrs:
            if hasattr(mod, name):
                val = getattr(mod, name)
                if name == 'recorded_inputs' and isinstance(val, list):
                    saved[name] = [t.detach().clone() if isinstance(t, torch.Tensor) else copy.deepcopy(t) for t in val]
                else:
                    saved[name] = val
        if saved:
            snap.append((mod, saved))
    return snap


def _restore_native_state(snap: list[tuple[object, dict]]) -> None:
    for mod, saved in snap:
        for name, val in saved.items():
            setattr(mod, name, val)


def probe_update(model, optimizer, train_loss, calibration_loss, *,
                 clip: float = 1.0,
                 generators: Iterable[torch.Generator] = (),
                 cached_cal_grad: tuple[torch.Tensor | None, ...] | None = None,
                 cached_cal_loss0: float | None = None):
    """Evaluate one actual candidate Adam/SGD update; always restore parent.

    Loss callbacks return a scalar Tensor, use only train data and may consume
    global CPU RNG or any explicitly passed ``generators``. Model parameters,
    buffers, native BPD mutable module attributes, optimizer state, existing
    grads, training flags and RNG states are completely restored. No candidate
    is committed by this function.
    """
    params = tuple(model.parameters())
    if not params or any(p.device.type != 'cpu' for p in params):
        raise ValueError('CPU model required')
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError('invalid clip')
    if (cached_cal_grad is None) != (cached_cal_loss0 is None):
        raise ValueError('cached_cal_grad and cached_cal_loss0 must both be provided or both None')

    gens = tuple(generators)
    start = time.process_time()
    initial = copy.deepcopy(model.state_dict())
    opt = copy.deepcopy(optimizer.state_dict())
    grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
    flags = [(m, m.training) for m in model.modules()]
    native_snap = _snapshot_native_state(model)
    rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    gen_states = [g.get_state().clone() for g in gens]
    try:
        model.eval()
        cal_rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
        cal_gen_states = [g.get_state().clone() for g in gens]
        if cached_cal_grad is not None:
            g = cached_cal_grad
            loss0_val = float(cached_cal_loss0)
        else:
            loss0 = calibration_loss()
            g = torch.autograd.grad(loss0, params, allow_unused=True)
            loss0_val = float(loss0.detach())
        old = [p.detach().clone() for p in params]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            predicted = -sum(
                float((a.detach().double() * (p.detach() - b).double()).sum())
                for a, p, b in zip(g, params, old) if a is not None
            )
            step_sq = sum(
                float((p.detach() - b).double().square().sum())
                for p, b in zip(params, old)
            )
            # Replay exact RNG draws used for calibration_before.
            torch.set_rng_state(cal_rng[0])
            random.setstate(cal_rng[1])
            np.random.set_state(cal_rng[2])
            for gen, st in zip(gens, cal_gen_states):
                gen.set_state(st)
            model.eval()
            loss1 = calibration_loss()
            loss1_val = float(loss1.detach())
            result = dict(
                predicted_gain=predicted,
                realized_gain=float(loss0_val - loss1_val),
                calibration_before=loss0_val,
                calibration_after=loss1_val,
                train_loss=float(loss.detach()),
                update_norm=step_sq ** 0.5,
            )
        if not all(np.isfinite(v) for v in result.values()):
            raise ValueError('nonfinite probe')
    finally:
        model.load_state_dict(initial)
        optimizer.load_state_dict(opt)
        for p, grad in zip(params, grads):
            p.grad = grad
        for module, flag in flags:
            module.training = flag
        _restore_native_state(native_snap)
        torch.set_rng_state(rng[0])
        random.setstate(rng[1])
        np.random.set_state(rng[2])
        for gen, st in zip(gens, gen_states):
            gen.set_state(st)
    result['cpu_seconds'] = max(time.process_time() - start, 1e-9)
    return result
