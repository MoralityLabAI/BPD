"""Signed readout Jacobians at the actual joint post-write state."""
from __future__ import annotations

import torch


def _continuation_logits(model, x, d, s, *, use_suffix: bool = True):
    """Compute logits at delay ``d`` from post-write joint state ``s``."""
    seq = torch.cat((x[:, :4 + d], x[:, -1:]), dim=1)
    if use_suffix and hasattr(model, 'rollout_suffix'):
        x_suffix = seq[:, 4:]
        return model.rollout_suffix(
            x_suffix, s[None, :].expand(4, -1), prefix_steps=4
        )['logits']
    return model.rollout(
        seq, state_replacement=s[None, :].expand(4, -1), patch_step=4
    )['logits']


def observe_edge(model, batch, delays, *, dose: float = .75,
                 use_suffix: bool = True, cache=None,
                 cache_key=None):
    """One edge expanded to four address queries, one fixed nuisance draw.

    Each earlier query is a declared counterfactual query schedule, not an
    observed native intermediate output. State replacement is before the
    first distractor. All Jacobians are with respect to that same joint state.
    When ``use_suffix=True`` and ``model`` implements ``rollout_suffix``,
    continuations reuse the post-write joint state directly.
    """
    if cache is not None and cache_key is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit

    x, h = batch.receiver, batch.hybrid
    delays = tuple(delays)
    if (x.ndim != 3 or x.shape != h.shape or len(x) != 4 or
            batch.query.tolist() != list(range(4)) or len(set(batch.edit_slot.tolist())) != 1 or
            not delays or tuple(sorted(set(delays))) != delays or
            any(type(d) is not int or d < 0 or d > x.shape[1] - 5 for d in delays) or
            not torch.isfinite(torch.tensor(dose)) or dose <= 0):
        raise ValueError('invalid observer schedule')
    slot = int(batch.edit_slot[0])
    old = int(batch.labels_r[slot])
    new = int(batch.labels_h[slot])
    if (old == new or not torch.equal(x[:, 4:], h[:, 4:]) or
            not torch.equal(batch.labels_r != batch.labels_h, batch.query == slot) or
            not torch.equal(x[:, :4], x[:1, :4].expand_as(x[:, :4])) or
            not torch.equal(h[:, :4], h[:1, :4].expand_as(h[:, :4]))):
        raise ValueError('invalid single-edge/query alignment')

    with torch.no_grad():
        state = model.prefix_state(x[:1]).detach().reshape(-1)
        hybrid = model.prefix_state(h[:1]).detach().reshape(-1)
    rows, protected, logits, a = [], [], [], None
    with torch.enable_grad():
        state_leaf = state.clone().requires_grad_(True)
        for d in delays:
            y = _continuation_logits(model, x, d, state_leaf, use_suffix=use_suffix)
            logits.append(y.detach())
            for q in range(4):
                ref = int(batch.labels_r[q])
                for v in range(y.shape[1]):
                    if v == ref:
                        continue
                    row = torch.autograd.grad(y[q, v] - y[q, ref], state_leaf, retain_graph=True)[0].detach()
                    rows.append(row)
                    if q != slot:
                        protected.append(row)
                    if d == delays[-1] and q == slot and v == new:
                        a = row
    if a is None:
        raise ValueError('missing target margin')
    raw = hybrid - state
    norm = raw.norm()
    valid = bool(norm > 1e-9)
    delta = raw * (dose / norm) if valid else torch.zeros_like(raw)
    with torch.no_grad():
        actual = _continuation_logits(model, x, delays[-1], state + delta, use_suffix=use_suffix)
    clean = logits[-1]
    diff = actual - clean
    prot_stack = torch.stack(protected)
    obs_stack = torch.stack(rows)
    target_linear = float(a @ delta)
    target_actual = float(diff[slot, new] - diff[slot, old])
    out = dict(
        target=a.detach().clone(),
        protected=prot_stack.detach().clone(),
        observer=obs_stack.detach().clone(),
        clean_logits=clean.detach().clone(),
        actual_logits=actual.detach().clone(),
        hybrid_direction=delta.detach().clone(),
        post_write_state=state.detach().clone(),
        hybrid_state=hybrid.detach().clone(),
        raw_hybrid_displacement=raw.detach().clone(),
        valid_dose=valid,
        target_linear=target_linear,
        target_actual=target_actual,
        finite_dose_error=abs(target_actual - target_linear),
        natural_norm=float(norm),
        edit_slot=slot,
        old_value=old,
        new_value=new,
        delays=delays,
        dose=float(dose),
    )
    if cache is not None and cache_key is not None:
        cache.put(cache_key, out)
    return out


@torch.no_grad()
def finite_difference_observer(model, batch, delays, *, eps: float = 1e-3,
                               use_suffix: bool = True) -> dict[str, torch.Tensor]:
    """Independent central finite-difference Jacobian verification at post-write state."""
    x = batch.receiver
    delays = tuple(delays)
    slot = int(batch.edit_slot[0])
    new = int(batch.labels_h[slot])
    state = model.prefix_state(x[:1]).detach().reshape(-1)
    dim = state.numel()
    # Evaluate all margin scalars for a given state vector
    def margins(s: torch.Tensor) -> torch.Tensor:
        vals = []
        for d in delays:
            y = _continuation_logits(model, x, d, s, use_suffix=use_suffix)
            for q in range(4):
                ref = int(batch.labels_r[q])
                for v in range(y.shape[1]):
                    if v == ref:
                        continue
                    vals.append(y[q, v] - y[q, ref])
        return torch.stack(vals)

    n_rows = 4 * 3 * len(delays)
    jac = torch.zeros((n_rows, dim), dtype=torch.float64)
    for k in range(dim):
        step = torch.zeros_like(state)
        step[k] = eps
        f_plus = margins(state + step).double()
        f_minus = margins(state - step).double()
        jac[:, k] = (f_plus - f_minus) / (2.0 * eps)

    # Identify protected rows and target row matching observe_edge order
    prot_idx = []
    target_idx = None
    idx = 0
    for d in delays:
        for q in range(4):
            ref = int(batch.labels_r[q])
            for v in range(4):
                if v == ref:
                    continue
                if q != slot:
                    prot_idx.append(idx)
                if d == delays[-1] and q == slot and v == new:
                    target_idx = idx
                idx += 1
    assert target_idx is not None
    return {
        'observer': jac.float(),
        'protected': jac[prot_idx].float(),
        'target': jac[target_idx].float(),
    }
