"""Opt-in exact-algebra BPD component write; pure CPU pilot, no native patch."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import statistics
import time

import torch
import torch.nn.functional as F


def component_write(x, W, V, U, mask, delta=None, bias=None, mu_c=None,
                    mu_d=None, mean_rms=None, route=None, eps=1e-6,
                    *, fused=True):
    """Inputs use one common floating dtype; means are fixed for this pass.

    delta/route have shape x.shape[:-1]+(1,); mask broadcasts to x@V.
    mu_c/mu_d/mean_rms None means no calibrated mean/sphere for this pass.
    Caller handles the wrapper's call counter, observation and output cast.
    """
    a = x @ V
    d = 1.0 if delta is None else delta
    if fused:
        y = F.linear(x, W)
        z = (mask - d) * a
        if mu_c is not None:
            z = z + (1.0 - mask) * mu_c
        out = d * y + z @ U
        if delta is not None and mu_d is not None:
            out = out + (1.0 - delta) * mu_d
    else:
        z = mask * a
        if mu_c is not None:
            z = z + (1.0 - mask) * mu_c
        raw = F.linear(x, W - (V @ U).T)
        residual = d * raw
        if delta is not None and mu_d is not None:
            residual = residual + (1.0 - delta) * mu_d
        out = z @ U + residual
    if mean_rms is not None:
        frac = (1.0 - mask.clamp(0.0, 1.0)).mean(dim=-1, keepdim=True)
        ms = out.square().mean(dim=-1, keepdim=True)
        rms = (ms + eps).sqrt()
        desired = (1.0 - frac) * rms + frac * mean_rms.clamp_min(eps)
        out = out * torch.where(ms > eps, desired / rms, torch.ones_like(rms))
    if bias is not None:
        out = out + bias
    if route is not None:
        target = y if fused else F.linear(x, W)
        if bias is not None:
            target = target + bias
        out = torch.where(route, out, target)
    return out


def fixture(seed, n, din, dout, c, dtype, mask_kind, calibrated, residual,
            bias_on, sphere, routing):
    gen = torch.Generator().manual_seed(seed)
    def randn(*shape):
        return torch.randn(*shape, dtype=dtype, generator=gen)
    x = randn(n, din)
    W = randn(dout, din) / math.sqrt(din)
    V = randn(din, c) / math.sqrt(din)
    U = randn(c, dout) / math.sqrt(c)
    m = torch.rand(n, c, dtype=dtype, generator=gen)
    if mask_kind == 'binary':
        m = (m > .5).to(dtype)
    elif mask_kind == 'zero':
        m.zero_()
    elif mask_kind == 'one':
        m.fill_(1)
    d = torch.rand(n, 1, dtype=dtype, generator=gen) if residual else None
    if residual and mask_kind == 'zero':
        d.zero_()
    if residual and mask_kind == 'one':
        d.fill_(1)
    bias = randn(dout) / 10 if bias_on else None
    mc = randn(c) / 10 if calibrated else None
    md = randn(dout) / 10 if calibrated else None
    mr = torch.tensor(.8, dtype=dtype) if calibrated and sphere else None
    route = (torch.arange(n).reshape(n, 1) % 3 != 0) if routing else None
    cot = randn(n, dout) / math.sqrt(n*dout)
    return (x, W, V, U, m, d, bias, mc, md, mr, route), cot


def evaluate(args, cot, fused, backward):
    args = list(args)
    indices = [i for i in (0, 2, 3, 4, 5) if args[i] is not None]
    if backward:
        for i in indices:
            args[i] = args[i].detach().requires_grad_(True)
        y = component_write(*args, fused=fused)
        grads = torch.autograd.grad((y*cot).sum(), [args[i] for i in indices])
        return y.detach(), tuple(g.detach() for g in grads)
    with torch.no_grad():
        return component_write(*args, fused=fused), ()


def parity():
    checks = 0
    max_errors = {}
    for dtype in (torch.float64, torch.float32):
        errors = {'output': 0., 'gradient': 0.}
        atol, rtol = (1e-10, 1e-9) if dtype == torch.float64 else (5e-5, 5e-4)
        for j, (mask, cal, residual, bias, sphere, route) in enumerate(itertools.product(
                ('soft', 'binary', 'zero', 'one'), (False, True), (False, True),
                (False, True), (False, True), (False, True))):
            args, cot = fixture(20262800+j, 9, 17, 23, 7, dtype, mask,
                                cal, residual, bias, sphere, route)
            a, ag = evaluate(args, cot, False, True)
            b, bg = evaluate(args, cot, True, True)
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
            errors['output'] = max(errors['output'], float((a-b).abs().max()))
            checks += 1
            for u, v in zip(ag, bg):
                torch.testing.assert_close(u, v, atol=atol, rtol=rtol)
                errors['gradient'] = max(errors['gradient'], float((u-v).abs().max()))
                checks += 1
        max_errors[str(dtype)] = errors
    return {'checks': checks, 'max_abs_error': max_errors}


def bench():
    rows = []
    for din, dout, c in ((128,128,32),(256,256,64),(512,512,64),
                          (1024,1024,128),(1024,256,64)):
        n = 64
        args, cot = fixture(20262900+din+dout+c, n, din, dout, c,
                            torch.float32, 'soft', True, True, True, False, False)
        for backward in (False, True):
            for _ in range(3):
                for f in (False, True):
                    evaluate(args, cot, f, backward)
            timings = {False: [], True: []}
            for j in range(11):
                for f in ((False, True) if j % 2 == 0 else (True, False)):
                    start = time.perf_counter_ns()
                    evaluate(args, cot, f, backward)
                    timings[f].append((time.perf_counter_ns()-start)/1e6)
            ref, opt = statistics.median(timings[False]), statistics.median(timings[True])
            mac_fused = n*(din*dout + din*c + c*dout)
            mac_reference = mac_fused + din*c*dout
            rows.append({'tokens': n, 'din': din, 'dout': dout, 'components': c,
                         'backward': backward, 'reference_ms': ref, 'fused_ms': opt,
                         'reference_min_ms': min(timings[False]), 'fused_min_ms': min(timings[True]),
                         'speedup': ref/opt, 'forward_reference_macs': mac_reference,
                         'forward_fused_macs': mac_fused})
    return rows


def native_parity():
    """Optional in-repository check of frozen native TiedComponentLinear.forward."""
    from mechinterp.bpd.components import TiedComponentLinear, PassMaskSchedule
    from mechinterp.bpd.config import AblationConfig
    checks = 0
    for mode, sphere, residual, routing in itertools.product(
            ('zero', 'pass_mean'), (False, True), (False, True), (False, True)):
        torch.manual_seed(20263000+checks)
        lin = torch.nn.Linear(17, 23)
        mod = TiedComponentLinear(lin, 7, max_passes=2,
                                  ablation=AblationConfig(mode=mode, sphere_rescale=sphere),
                                  init_svd=False)
        mod.mode = 'component'
        x = torch.randn(2, 3, 17, requires_grad=True)
        mask = torch.rand(2, 1, 3, 7, requires_grad=True)
        d = torch.rand(2, 1, 3, requires_grad=True) if residual else None
        route = torch.tensor([[[True, False, True]], [[False, True, True]]]) if routing else None
        mod.schedule = PassMaskSchedule(mask, d, route)
        mod.calibrated_passes[0] = True
        mod.mean_comp_acts[0].normal_(0,.1)
        mod.mean_delta_out[0].normal_(0,.1)
        mod.mean_write_rms[0] = .8
        cal = mode == 'pass_mean'
        a = mod(x)
        b = component_write(x, mod.W_target.float(), mod.V, mod.U, mask[:,0],
                            None if d is None else d[:,0].unsqueeze(-1), mod.bias,
                            mod.mean_comp_acts[0] if cal else None,
                            mod.mean_delta_out[0] if cal else None,
                            mod.mean_write_rms[0] if cal and sphere else None,
                            None if route is None else route[:,0].unsqueeze(-1),
                            mod.ablation.eps, fused=True)
        torch.testing.assert_close(a,b,atol=5e-5,rtol=5e-4)
        cot = torch.randn_like(a)
        vars = (x, mod.V, mod.U, mask) + (() if d is None else (d,))
        ga = torch.autograd.grad((a*cot).sum(), vars)
        gb = torch.autograd.grad((b*cot).sum(), vars)
        for u,v in zip(ga,gb):
            torch.testing.assert_close(u,v,atol=5e-5,rtol=5e-4)
        checks += 1+len(vars)
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps({'status': 'PASS', 'torch': torch.__version__,
                      'platform': platform.platform(), 'threads': torch.get_num_threads(),
                      'scope': 'isolated CPU kernels; no native model or GPU',
                      'parity': parity(), 'benchmark': bench(),
                      'native_checks': native_parity() if args.native else None}, indent=2))
