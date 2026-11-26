"""CPU-only one-step optimizer probes with rollback of tensor and RNG state."""
import copy
import random
import time

import numpy as np
import torch


def probe_update(model, optimizer, train_loss, calibration_loss, *, clip=1.0):
    """Evaluate one actual candidate Adam/SGD update; always restore parent.

    Loss callbacks return a scalar Tensor, use only train data and may consume
    global CPU RNG. Model parameters/buffers and optimizer state are restored;
    callers must not mutate external caches, sampler generators or Python model
    attributes in callbacks. No candidate is committed by this function.
    """
    params = tuple(model.parameters())
    if not params or any(p.device.type != 'cpu' for p in params):
        raise ValueError('CPU model required')
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError('invalid clip')
    start = time.process_time()
    initial = copy.deepcopy(model.state_dict())
    opt = copy.deepcopy(optimizer.state_dict())
    grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
    flags = [(m, m.training) for m in model.modules()]
    rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    try:
        model.eval()
        cal_rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
        loss0 = calibration_loss()
        g = torch.autograd.grad(loss0, params, allow_unused=True)
        old = [p.detach().clone() for p in params]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            predicted = -sum(float((a.detach().double()*(p.detach()-b).double()).sum())
                             for a, p, b in zip(g, params, old) if a is not None)
            step_sq = sum(float((p.detach()-b).double().square().sum()) for p, b in zip(params, old))
            # Replay the exact global random draws used for calibration before.
            torch.set_rng_state(cal_rng[0]); random.setstate(cal_rng[1]); np.random.set_state(cal_rng[2])
            model.eval()
            loss1 = calibration_loss()
            result = dict(predicted_gain=predicted, realized_gain=float(loss0.detach()-loss1),
                          calibration_before=float(loss0.detach()), calibration_after=float(loss1),
                          train_loss=float(loss.detach()), update_norm=step_sq**.5)
        if not all(np.isfinite(v) for v in result.values()):
            raise ValueError('nonfinite probe')
    finally:
        model.load_state_dict(initial)
        optimizer.load_state_dict(opt)
        for p, grad in zip(params, grads):
            p.grad = grad
        for module, flag in flags:
            module.training = flag
        torch.set_rng_state(rng[0]); random.setstate(rng[1]); np.random.set_state(rng[2])
    result['cpu_seconds'] = time.process_time()-start
    return result
