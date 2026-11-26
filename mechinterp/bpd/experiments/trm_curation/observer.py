"""Signed readout Jacobians at the actual joint post-write state."""
import torch


def observe_edge(model, batch, delays, *, dose=.75):
    """One edge expanded to four address queries, one fixed nuisance draw.

    Each earlier query is a declared counterfactual query schedule, not an
    observed native intermediate output. State replacement is before the
    first distractor. All Jacobians are with respect to that same joint state.
    """
    x, h = batch.receiver, batch.hybrid
    delays = tuple(delays)
    if (x.ndim != 3 or x.shape != h.shape or len(x) != 4 or
            batch.query.tolist() != list(range(4)) or len(set(batch.edit_slot.tolist())) != 1 or
            not delays or tuple(sorted(set(delays))) != delays or
            any(type(d) is not int or d < 0 or d > x.shape[1]-5 for d in delays) or
            not torch.isfinite(torch.tensor(dose)) or dose <= 0):
        raise ValueError('invalid observer schedule')
    slot = int(batch.edit_slot[0]); old = int(batch.labels_r[slot]); new = int(batch.labels_h[slot])
    if (old == new or not torch.equal(x[:,4:], h[:,4:]) or
            not torch.equal(batch.labels_r != batch.labels_h, batch.query == slot) or
            not torch.equal(x[:,:4],x[:1,:4].expand_as(x[:,:4])) or
            not torch.equal(h[:,:4],h[:1,:4].expand_as(h[:,:4]))):
        raise ValueError('invalid single-edge/query alignment')
    with torch.no_grad():
        state = model.prefix_state(x[:1]).detach().reshape(-1)
        hybrid = model.prefix_state(h[:1]).detach().reshape(-1)
    state.requires_grad_(True)
    rows, protected, logits, a = [], [], [], None
    def continuation(d, s):
        seq = torch.cat((x[:,:4+d], x[:,-1:]), dim=1)
        return model.rollout(seq, state_replacement=s[None,:].expand(4,-1), patch_step=4)['logits']
    for d in delays:
        y = continuation(d,state)
        logits.append(y.detach())
        for q in range(4):
            ref = int(batch.labels_r[q])
            for v in range(y.shape[1]):
                if v == ref:
                    continue
                row = torch.autograd.grad(y[q,v]-y[q,ref],state,retain_graph=True)[0].detach()
                rows.append(row)
                if q != slot:
                    protected.append(row)
                if d == delays[-1] and q == slot and v == new:
                    a = row
    if a is None:
        raise ValueError('missing target margin')
    raw = hybrid-state.detach(); norm = raw.norm()
    valid = bool(norm > 1e-9)
    delta = raw*(dose/norm) if valid else torch.zeros_like(raw)
    with torch.no_grad():
        actual = continuation(delays[-1],state.detach()+delta)
    clean = logits[-1]
    diff = actual-clean
    return dict(target=a, protected=torch.stack(protected), observer=torch.stack(rows),
                clean_logits=clean, hybrid_direction=delta, valid_dose=valid,
                target_linear=float(a@delta), target_actual=float(diff[slot,new]-diff[slot,old]),
                natural_norm=float(norm), actual_logits=actual)
