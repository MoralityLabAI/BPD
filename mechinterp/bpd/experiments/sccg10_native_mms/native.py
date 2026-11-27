"""Exact native response maps, finite verification and analytic AdamW previews.

Two separate action spaces on the native ComponentTRM:

* ``state``: an additive edit of the post-write joint carried state [z_H, z_L]
  (64 coordinates) immediately before the first distractor. Metric: identity.
* ``write``: an additive native factor write at the first suffix low visit,
  ``up_out += dc @ U[S]`` for a component support S of the shared
  TiedComponentLinear factors. Metric: the physical write Gram U_S U_S^T.

Responses are signed relative logits m_{d,q,v} = y[q,v]-y[q,ref_q], v != ref_q,
for every query q in 0..3 and every panel delay d, computed by exact suffix
continuation from the post-write state (``rollout_suffix``). These are full
recurrent task-output derivatives, not local reader proxies. Row order matches
trm_curation.observer.observe_edge.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

N_VALUES = 4


@dataclass
class Panel:
    """One semantic single-slot edit expanded into four queries."""
    batch: Any
    slot: int
    old: int
    new: int
    refs: tuple[int, ...]


def make_panel(batch) -> Panel:
    slot = int(batch.edit_slot[0])
    return Panel(batch, slot, int(batch.labels_r[slot]), int(batch.labels_h[slot]),
                 tuple(int(v) for v in batch.labels_r.tolist()))


def row_layout(panel: Panel, delays) -> dict[str, Any]:
    """Indices of target / protected rows in the stacked margin vector."""
    idx, prot, target = 0, [], None
    meta = []
    for d in delays:
        for q in range(4):
            for v in range(N_VALUES):
                if v == panel.refs[q]:
                    continue
                meta.append((d, q, v))
                if q != panel.slot:
                    prot.append(idx)
                if d == delays[-1] and q == panel.slot and v == panel.new:
                    target = idx
                idx += 1
    return {'target': target, 'protected': prot, 'rows': meta, 'n': idx}


def _suffix_inputs(x, d):
    seq = torch.cat((x[:, :4 + d], x[:, -1:]), dim=1)
    return seq[:, 4:]


def post_write_state(model, panel: Panel) -> torch.Tensor:
    with torch.no_grad():
        return model.prefix_state(panel.batch.receiver[:1]).detach().reshape(-1)


def hybrid_state(model, panel: Panel) -> torch.Tensor:
    with torch.no_grad():
        return model.prefix_state(panel.batch.hybrid[:1]).detach().reshape(-1)


def margins_from_logits(y: torch.Tensor, panel: Panel) -> torch.Tensor:
    out = []
    for q in range(4):
        for v in range(N_VALUES):
            if v != panel.refs[q]:
                out.append(y[q, v] - y[q, panel.refs[q]])
    return torch.stack(out)


def suffix_logits(model, panel: Panel, d: int, state: torch.Tensor, *,
                  write_patch: torch.Tensor | None = None, write_site: str = 'low'):
    x = panel.batch.receiver
    xs = _suffix_inputs(x, d)
    init = state.reshape(1, -1).expand(4, -1) if state.ndim == 1 else state
    wp = None if write_patch is None else write_patch.reshape(1, -1).expand(4, -1).contiguous()
    return model.rollout_suffix(xs, init, prefix_steps=4, write_patch=wp, write_site=write_site,
                                patch_step=0)['logits']


@torch.no_grad()
def margins(model, panel: Panel, delays, *, state_delta=None, write_delta=None,
            write_site: str = 'low', state=None) -> torch.Tensor:
    """Stacked native margins (all delays) after an optional finite edit."""
    s = post_write_state(model, panel) if state is None else state
    if state_delta is not None:
        s = s + state_delta
    out = []
    for d in delays:
        y = suffix_logits(model, panel, d, s, write_patch=write_delta, write_site=write_site)
        out.append(margins_from_logits(y, panel))
    return torch.cat(out)


@torch.no_grad()
def predictions(model, panel: Panel, delays, *, state_delta=None, write_delta=None,
                write_site: str = 'low') -> torch.Tensor:
    s = post_write_state(model, panel)
    if state_delta is not None:
        s = s + state_delta
    return torch.stack([suffix_logits(model, panel, d, s, write_patch=write_delta,
                                      write_site=write_site).argmax(-1) for d in delays])


def state_jacobian(model, panel: Panel, delays) -> torch.Tensor:
    """Exact d margins / d post-write state, [rows, 64].

    Each query gets its own leaf copy of the shared state; logits of query q
    depend only on leaf q (row-independent recurrence), so for each value v a
    single backward of sum_q m_{q,v} yields all four queries' rows. Four
    backward passes per delay instead of twelve; exact, parity-tested against
    observe_edge.
    """
    s = post_write_state(model, panel)
    rows = []
    with torch.enable_grad():
        for d in delays:
            leaf = s.reshape(1, -1).expand(4, -1).clone().requires_grad_(True)
            y = suffix_logits(model, panel, d, leaf)
            refs = torch.tensor(panel.refs)
            ref_logit = y.gather(1, refs[:, None]).squeeze(1)
            per_v = []
            for v in range(N_VALUES):
                g = torch.autograd.grad((y[:, v] - ref_logit).sum(), leaf, retain_graph=v < N_VALUES - 1)[0]
                per_v.append(g.detach())
            for q in range(4):
                for v in range(N_VALUES):
                    if v != panel.refs[q]:
                        rows.append(per_v[v][q])
    return torch.stack(rows)


def write_jacobian(model, panel: Panel, delays, *, write_site: str = 'low') -> torch.Tensor:
    """Exact d margins / d up_out write patch at the first suffix visit, [rows, 2w]."""
    s = post_write_state(model, panel)
    w = 2 * model.width
    rows = []
    with torch.enable_grad():
        for d in delays:
            leaf = torch.zeros((4, w), dtype=s.dtype, requires_grad=True)
            x = panel.batch.receiver
            y = model.rollout_suffix(_suffix_inputs(x, d), s.reshape(1, -1).expand(4, -1),
                                     prefix_steps=4, write_patch=leaf, write_site=write_site,
                                     patch_step=0)['logits']
            refs = torch.tensor(panel.refs)
            ref_logit = y.gather(1, refs[:, None]).squeeze(1)
            per_v = [torch.autograd.grad((y[:, v] - ref_logit).sum(), leaf,
                                         retain_graph=v < N_VALUES - 1)[0].detach()
                     for v in range(N_VALUES)]
            for q in range(4):
                for v in range(N_VALUES):
                    if v != panel.refs[q]:
                        rows.append(per_v[v][q])
    return torch.stack(rows)


def state_jvp(model, panel: Panel, delays, direction: torch.Tensor) -> torch.Tensor:
    """Exact directional derivative J u by forward-mode AD (one dual pass per delay)."""
    import torch.autograd.forward_ad as fwAD
    s = post_write_state(model, panel)
    out = []
    with torch.no_grad(), fwAD.dual_level():
        for d in delays:
            dual = fwAD.make_dual(s.reshape(1, -1).expand(4, -1).contiguous(),
                                  direction.reshape(1, -1).expand(4, -1).contiguous())
            y = suffix_logits(model, panel, d, dual)
            out.append(margins_from_logits(fwAD.unpack_dual(y).tangent, panel))
    return torch.cat(out)


@dataclass
class Observation:
    a: np.ndarray          # target row, float64
    C: np.ndarray          # protected rows, float64
    rows: np.ndarray       # full stacked rows
    clean: np.ndarray      # clean margins
    hybrid_dir: np.ndarray  # unit hybrid-minus-receiver direction (state space)
    natural_norm: float
    layout: dict
    cost_cpu: float


def observe_state(model, panel: Panel, delays) -> Observation:
    t0 = time.process_time()
    J = state_jacobian(model, panel, delays).double().numpy()
    lay = row_layout(panel, delays)
    clean = margins(model, panel, delays).double().numpy()
    raw = (hybrid_state(model, panel) - post_write_state(model, panel)).double().numpy()
    n = float(np.linalg.norm(raw))
    return Observation(J[lay['target']].copy(), J[lay['protected']].copy(), J, clean,
                       raw / n if n > 1e-12 else np.zeros_like(raw), n, lay,
                       time.process_time() - t0)


def write_frame(model) -> np.ndarray:
    """Physical write frame U [components, 2w] of the shared factors (float64)."""
    return model.up.U.detach().double().numpy().copy()


@torch.no_grad()
def panel_logits(model, panel: Panel, delays, *, state_delta=None, write_delta=None,
                 write_site: str = 'low') -> list[torch.Tensor]:
    s = post_write_state(model, panel)
    if state_delta is not None:
        s = s + state_delta
    return [suffix_logits(model, panel, d, s, write_patch=write_delta, write_site=write_site)
            for d in delays]


def finite_outcome(model, panel: Panel, delays, *, state_delta=None, write_delta=None,
                   write_site: str = 'low', clean=None) -> dict[str, Any]:
    """Native finite verification of one edit on this panel (exact continuation).

    ``clean`` may carry the clean per-delay logits from ``panel_logits`` so a
    verification costs one edited continuation per delay.
    """
    lay = row_layout(panel, delays)
    y0 = panel_logits(model, panel, delays) if clean is None or not isinstance(clean, list) else clean
    sd = None if state_delta is None else torch.as_tensor(state_delta, dtype=torch.float32)
    wd = None if write_delta is None else torch.as_tensor(write_delta, dtype=torch.float32)
    y1 = panel_logits(model, panel, delays, state_delta=sd, write_delta=wd, write_site=write_site)
    m0 = torch.cat([margins_from_logits(y, panel) for y in y0])
    m1 = torch.cat([margins_from_logits(y, panel) for y in y1])
    dm = (m1 - m0).double().numpy()
    p0 = torch.stack([y.argmax(-1) for y in y0])
    p1 = torch.stack([y.argmax(-1) for y in y1])
    refs = torch.tensor(panel.refs)
    unaffected = torch.tensor([q != panel.slot for q in range(4)])
    corr0 = (p0 == refs[None, :]) & unaffected[None, :]
    corrupted = corr0 & (p1 != refs[None, :])
    return {'target_gain': float(dm[lay['target']]),
            'protected_change_norm': float(np.linalg.norm(dm[lay['protected']])),
            'protected_change_max': float(np.abs(dm[lay['protected']]).max()),
            'target_flip': bool(p1[-1, panel.slot] == panel.new),
            'target_initially_new': bool(p0[-1, panel.slot] == panel.new),
            'corrupted': int(corrupted.sum()), 'eligible_unaffected_correct': int(corr0.sum()),
            'delta_margins': dm}


# ---------------------------------------------------------------- AdamW preview

def clipped_grads(params, grads, clip: float):
    total = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
    coef = torch.clamp(clip / (total + 1e-6), max=1.0)
    return [g * coef for g in grads], float(total)


def adamw_delta(optimizer: torch.optim.AdamW, params, grads) -> list[torch.Tensor]:
    """Exact parameter change of one torch AdamW step with the given (clipped) grads.

    Mirrors torch.optim.adamw single-tensor math for amsgrad=False,
    maximize=False: decoupled decay first, then bias-corrected Adam step.
    Parity with an actual cloned step is tested (test_native.py).
    """
    group = optimizer.param_groups[0]
    lr, (b1, b2), eps, wd = group['lr'], group['betas'], group['eps'], group['weight_decay']
    if group.get('amsgrad') or group.get('maximize'):
        raise ValueError('preview supports amsgrad=False, maximize=False')
    out = []
    for p, g in zip(params, grads):
        st = optimizer.state.get(p, {})
        if st:
            step = float(st['step']) + 1
            m, v = st['exp_avg'], st['exp_avg_sq']
        else:
            step, m, v = 1.0, torch.zeros_like(p), torch.zeros_like(p)
        theta = p.detach() * (1 - lr * wd)
        m1 = m.lerp(g, 1 - b1)
        v1 = (v * b2).addcmul(g, g, value=1 - b2)
        bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
        denom = (v1.sqrt() / math.sqrt(bc2)) + eps
        new = theta - (lr / bc1) * (m1 / denom)
        out.append(new - p.detach())
    return out


def candidate_gradient(model, batch) -> tuple[list[torch.Tensor], float]:
    params = [p for p in model.parameters()]
    model.train()
    x = torch.cat((batch.receiver, batch.hybrid), 0)
    y = torch.cat((batch.labels_r, batch.labels_h), 0)
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params)
    return [g.detach() for g in grads], float(loss.detach())


def preview_gain(model, optimizer, cal_grad, batch, clip: float) -> dict[str, float]:
    """First-order calibration gain -<g_cal, dtheta> of the exact AdamW step on ``batch``."""
    params = [p for p in model.parameters()]
    grads, loss = candidate_gradient(model, batch)
    grads, gnorm = clipped_grads(params, grads, clip)
    delta = adamw_delta(optimizer, params, grads)
    pred = -sum(float((a.double() * d.double()).sum()) for a, d in zip(cal_grad, delta) if a is not None)
    step = math.sqrt(sum(float(d.double().square().sum()) for d in delta))
    return {'predicted_gain': pred, 'train_loss': loss, 'update_norm': step, 'grad_norm': gnorm}
