"""Feedback-cohort mechanism: SCCG-01 sparse selective directions and the
mechanistic calibration objective that consumes them.

For each TRAIN feedback edge i the mechanism is a physical carried-state edit
delta_i (k-sparse support, dose <= 0.75, linear collateral <= eta) chosen by
SCCG-01 from the exact full-suffix derivative at the feedback delay. The
calibration objective is CURATE-02's mechanistic feedback loss with these
per-edge directions in place of the warmup-fixed slot projectors. It scores
curriculum candidates and gates commits; it is never a training loss (the
causal auxiliary cross stays closed while behavioral admission is failed).

The same forward pass yields, for every edge, the native finite outcome of its
direction at the current checkpoint (target margin gain and protected margin
change at the feedback delay): the L4 verification readout.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import native
from .support import beam_support


def slice_edge(batch, i: int):
    sl = slice(4 * i, 4 * i + 4)
    kw = {}
    for f in dataclasses.fields(batch):
        v = getattr(batch, f.name)
        kw[f.name] = v[sl] if isinstance(v, torch.Tensor) else v
    return dataclasses.replace(batch, **kw)


def refresh_edge(model, panel, delays, dose, eta, k, beam, ledger) -> dict:
    """Full mechanism refresh for one edge: exact observation + support search."""
    with ledger.charge('mechanism_observe'):
        obs = native.observe_state(model, panel, delays)
    ledger.count('backward_passes', 4 * len(delays))
    with ledger.charge('support_search'):
        bs = beam_support(obs.a, obs.C, np.eye(len(obs.a)), dose, eta, k, beam)
    ledger.count('solver_calls', bs['solves'])
    support, sol = bs['best_by_size'][k]
    delta = sol['delta'] if sol['gain'] > 0 else np.zeros_like(obs.a)
    with ledger.charge('verification'):
        fo = native.finite_outcome(model, panel, delays, state_delta=delta)
    sup = list(support)
    Cs = obs.C[:, sup]
    _, sv, vt = np.linalg.svd(Cs, full_matrices=True)
    r = int(np.count_nonzero(sv > 1e-10 * sv[0])) if len(sv) and sv[0] > 0 else 0
    return {'support': tuple(support), 'delta': delta, 'a_support': obs.a[sup].copy(),
            'C_support': Cs.copy(), 'svals': sv.copy(), 'null_basis': vt[r:].T.copy(),
            'lin_gain': float(sol['gain']), 'source_gain': fo['target_gain'],
            'source_protected': fo['protected_change_norm'], 'obs_cpu': obs.cost_cpu}


def mech_loss(model, fb, deltas: torch.Tensor, *, w_ordinary=1.0, w_target=0.5, w_preserve=0.5):
    """Mechanistic calibration loss with per-edge detached directions [n_edges, 64].

    Returns (loss, components dict, per-edge readout list). Rows of edges with a
    zero direction contribute only the ordinary term (ordinary fallback).
    """
    n = len(fb.labels_r)
    out = model.rollout(torch.cat((fb.receiver, fb.hybrid), 0))
    y = torch.cat((fb.labels_r, fb.labels_h), 0)
    loss_ord = F.cross_entropy(out['logits'], y)
    sr = out['states'][:n, 3]
    drow = deltas.to(sr.dtype).repeat_interleave(4, 0)
    active = drow.abs().sum(-1) > 0
    edit = model.rollout_suffix(fb.receiver[:, 4:], sr + drow, prefix_steps=4)['logits']
    clean = out['logits'][:n].detach()
    targ = fb.query == fb.edit_slot
    unaff = ~targ
    t_rows, u_rows = targ & active, unaff & active
    zero = loss_ord * 0
    loss_t = F.cross_entropy(edit[t_rows], fb.labels_h[t_rows]) if bool(t_rows.any()) else zero
    if bool(u_rows.any()):
        kl = F.kl_div(F.log_softmax(edit[u_rows], -1), F.softmax(clean[u_rows], -1), reduction='batchmean')
        loss_p = 0.5 * (kl + F.cross_entropy(edit[u_rows], fb.labels_r[u_rows]))
    else:
        loss_p = zero
    loss = w_ordinary * loss_ord + w_target * loss_t + w_preserve * loss_p
    # native finite readout per edge at this checkpoint
    with torch.no_grad():
        e = edit.detach()
        readout = []
        for i in range(n // 4):
            rows = range(4 * i, 4 * i + 4)
            slot = int(fb.edit_slot[4 * i])
            old, new = int(fb.labels_r[4 * i + slot]), int(fb.labels_h[4 * i + slot])
            r = 4 * i + slot
            tg = float((e[r, new] - e[r, old]) - (clean[r, new] - clean[r, old]))
            prot = []
            for q in rows:
                if q == r:
                    continue
                ref = int(fb.labels_r[q])
                for v in range(e.shape[1]):
                    if v != ref:
                        prot.append(float((e[q, v] - e[q, ref]) - (clean[q, v] - clean[q, ref])))
            readout.append({'target_gain': tg, 'protected_norm': float(np.linalg.norm(prot))})
    comps = {'ordinary_ce': float(loss_ord.detach()), 'target': float(loss_t.detach()),
             'preserve': float(loss_p.detach())}
    return loss, comps, readout


def ordinary_loss(model, fb):
    return F.cross_entropy(model(torch.cat((fb.receiver, fb.hybrid), 0)),
                           torch.cat((fb.labels_r, fb.labels_h), 0))
