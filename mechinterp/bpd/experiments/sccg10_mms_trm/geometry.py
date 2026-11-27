"""Native margin maps, Jacobians, finite outcomes and drift geometry for ComponentTRM.

Two separate action spaces at the same chronological point (the post-write
joint state entering the first suffix token):

* ``state``: additive edit of the carried joint state [z_H, z_L] (64 coords,
  Euclidean metric).
* ``write``: native per-component factor write at the low visit of the first
  suffix token. Coefficients alpha in R^29 produce the pre-activation write
  alpha @ U (U is the live TiedComponentLinear factor, shape [29, 64]); the
  physical metric is G = U U^T so ||alpha||_G = ||alpha @ U||_2.

Margin rows follow ``observer.observe_edge`` exactly: for each delay, query
and wrong value v != labels_r[q], the row is y[q, v] - y[q, labels_r[q]].
Target row is (longest delay, edited slot, hybrid value); protected rows are
every row with query != edited slot.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from mechinterp.bpd.experiments.trm_curation.reference import budgeted_selective_direction


@dataclass(frozen=True)
class RowIndex:
    target: int
    protected: tuple[int, ...]
    slot: int
    old: int
    new: int
    delays: tuple[int, ...]


def row_index(batch, delays) -> RowIndex:
    slot = int(batch.edit_slot[0])
    old, new = int(batch.labels_r[slot]), int(batch.labels_h[slot])
    prot, tgt, i = [], None, 0
    for d in delays:
        for q in range(4):
            ref = int(batch.labels_r[q])
            for v in range(4):
                if v == ref:
                    continue
                if q != slot:
                    prot.append(i)
                if d == delays[-1] and q == slot and v == new:
                    tgt = i
                i += 1
    return RowIndex(tgt, tuple(prot), slot, old, new, tuple(delays))


def _suffix_seq(x, d):
    return torch.cat((x[:, 4:4 + d], x[:, -1:]), dim=1)


def logits_at(model, x, d, state, *, write=None):
    kw = {}
    if write is not None:
        kw = dict(write_patch=write[None, :].expand(4, -1), patch_step=0, write_site='low')
    return model.rollout_suffix(_suffix_seq(x, d), state[None, :].expand(4, -1), prefix_steps=4, **kw)['logits']


def margins_from_logits(logits_by_delay, labels_r):
    out = []
    for y in logits_by_delay:
        for q in range(4):
            ref = int(labels_r[q])
            for v in range(y.shape[1]):
                if v != ref:
                    out.append(y[q, v] - y[q, ref])
    return torch.stack(out)


def native_logits(model, batch, delays, state, *, delta_state=None, write=None):
    s = state if delta_state is None else state + delta_state
    return [logits_at(model, batch.receiver, d, s, write=write) for d in delays]


def post_write_state(model, batch):
    with torch.no_grad():
        return model.prefix_state(batch.receiver[:1]).detach().reshape(-1)


def hybrid_state(model, batch):
    with torch.no_grad():
        return model.prefix_state(batch.hybrid[:1]).detach().reshape(-1)


def jacobian(fn, z):
    """Exact reverse-mode Jacobian of a vector function at z (rows by autograd)."""
    leaf = z.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        m = fn(leaf)
        rows = [torch.autograd.grad(m[i], leaf, retain_graph=i < len(m) - 1)[0].detach()
                for i in range(len(m))]
    return m.detach(), torch.stack(rows)


def state_jacobian(model, batch, delays, state):
    return jacobian(lambda s: margins_from_logits(native_logits(model, batch, delays, s), batch.labels_r), state)


def write_jacobian(model, batch, delays, state):
    """Jacobian w.r.t. the 64-dim pre-activation write; coefficient map is J_w @ U^T."""
    zero = torch.zeros(2 * model.width)
    return jacobian(lambda w: margins_from_logits(native_logits(model, batch, delays, state, write=w),
                                                  batch.labels_r), zero)


def directional_derivative(model, batch, delays, state, *, delta_state=None, write=None):
    """Exact live linear response along one direction via forward-mode AD (one JVP)."""
    import torch.autograd.forward_ad as fwAD
    if (delta_state is None) == (write is None):
        raise ValueError('exactly one direction')
    with torch.no_grad(), fwAD.dual_level():
        if delta_state is not None:
            out = margins_from_logits(native_logits(model, batch, delays, fwAD.make_dual(state, delta_state)),
                                      batch.labels_r)
        else:
            w = fwAD.make_dual(torch.zeros(2 * model.width), write)
            out = margins_from_logits(native_logits(model, batch, delays, state, write=w), batch.labels_r)
        t = fwAD.unpack_dual(out).tangent
    return t.detach().clone()


def finite_outcome(model, batch, delays, state, idx: RowIndex, clean_logits, *, delta_state=None,
                   write=None, linear_target=None, linear_protected=None):
    """Actual native finite intervention outcome with explicit denominators."""
    with torch.no_grad():
        edit = native_logits(model, batch, delays, state, delta_state=delta_state, write=write)
    m0 = margins_from_logits(clean_logits, batch.labels_r).double()
    m1 = margins_from_logits(edit, batch.labels_r).double()
    dm = m1 - m0
    prot = list(idx.protected)
    dg = dm[prot]
    y0, y1 = clean_logits[-1][idx.slot], edit[-1][idx.slot]
    eligible = int(y0.argmax()) != idx.new
    flipped = eligible and int(y1.argmax()) == idx.new
    corr_den = corr_num = 0
    for k, _ in enumerate(delays):
        for q in range(4):
            if q == idx.slot:
                continue
            ref = int(batch.labels_r[q])
            if int(clean_logits[k][q].argmax()) == ref:
                corr_den += 1
                corr_num += int(int(edit[k][q].argmax()) != ref)
    out = dict(
        target_gain=float(dm[idx.target]),
        protected_rms=float(dg.square().mean().sqrt()),
        protected_l2=float(dg.norm()),
        protected_max=float(dg.abs().max()),
        target_eligible=bool(eligible),
        target_flipped=bool(flipped),
        corruption_numerator=corr_num,
        corruption_denominator=corr_den,
    )
    if linear_target is not None:
        out['target_remainder'] = abs(out['target_gain'] - float(linear_target))
    if linear_protected is not None:
        out['protected_remainder_l2'] = float((dg - torch.as_tensor(linear_protected, dtype=torch.float64)).norm())
    return out


# ---------------------------------------------------------------- geometry

def _sv(A):
    return np.linalg.svd(A, compute_uv=False) if A.size else np.zeros(0)


def numerical_rank(s, rtol):
    return int(np.count_nonzero(s > rtol * s[0])) if len(s) and s[0] > 0 else 0


def principal_sin(Q0, Q1):
    """sin of the largest principal angle between column spaces (orthonormal inputs)."""
    if Q0.shape[1] == 0 and Q1.shape[1] == 0:
        return 0.0
    if Q0.shape[1] != Q1.shape[1]:
        return 1.0
    s = np.linalg.svd(Q0.T @ Q1, compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - float(s.min()) ** 2)))


def row_and_null_space(C, rtol):
    _, s, vh = np.linalg.svd(C, full_matrices=True)
    r = numerical_rank(s, rtol)
    return vh[:r].T, vh[r:].T, s, r


def drift_metrics(a0, C0, a1, C1, *, Ginv0=None, rtols=(1e-3, 1e-6)):
    """Measured derivative drift. With Ginv0 the target drift is the G-dual norm.

    These are observations of the live derivative, not bounds obtained
    without computing it.
    """
    da, dC = a1 - a0, C1 - C0
    out = {}
    if Ginv0 is None:
        out['alpha'] = float(np.linalg.norm(da))
        out['gamma'] = float(_sv(dC)[0]) if dC.size else 0.0
        out['a_norm'] = float(np.linalg.norm(a0))
        out['C_norm'] = float(_sv(C0)[0])
    else:
        L = np.linalg.cholesky(Ginv0)  # Ginv0 = L L^T; dual norm ||L^T da||
        out['alpha'] = float(np.linalg.norm(L.T @ da))
        out['gamma'] = float(_sv(dC @ L)[0])
        out['a_norm'] = float(np.linalg.norm(L.T @ a0))
        out['C_norm'] = float(_sv(C0 @ L)[0])
    out['alpha_rel'] = out['alpha'] / max(out['a_norm'], 1e-30)
    out['gamma_rel'] = out['gamma'] / max(out['C_norm'], 1e-30)
    out['target_cosine'] = float(a0 @ a1 / max(np.linalg.norm(a0) * np.linalg.norm(a1), 1e-30))
    s_dC = _sv(dC)
    e = np.cumsum(s_dC ** 2) / max(float((s_dC ** 2).sum()), 1e-300)
    out['dC_rank90'] = int(np.searchsorted(e, .90) + 1) if len(e) else 0
    out['dC_rank99'] = int(np.searchsorted(e, .99) + 1) if len(e) else 0
    for rt in rtols:
        R0, N0, s0, r0 = row_and_null_space(C0, rt)
        R1, N1, s1, r1 = row_and_null_space(C1, rt)
        out[f'rank0_{rt:g}'] = r0
        out[f'rank1_{rt:g}'] = r1
        out[f'rowspace_sin_{rt:g}'] = principal_sin(R0, R1)
        out[f'nullspace_sin_{rt:g}'] = principal_sin(N0, N1)
    s1 = _sv(C1)
    out['C1_singular'] = [float(v) for v in s1]
    out['C1_condition'] = float(s1[0] / s1[-1]) if len(s1) and s1[-1] > 0 else float('inf')
    return out


def lowrank_residual(dC_ref, dC, r):
    """Residual of projecting dC onto the top-r right singular subspace of dC_ref."""
    if not dC.size or not np.any(dC):
        return 0.0
    _, _, vh = np.linalg.svd(dC_ref, full_matrices=False)
    P = vh[:r].T @ vh[:r]
    return float(np.linalg.norm(dC - dC @ P) / np.linalg.norm(dC))


# ---------------------------------------------------------------- proposals

def greedy_support(a, C, B, dose, *, rank, budget, beam=4):
    """Deterministic single-column beam search over supports of span(B).

    Cheaper than CURATE-02's single/pair expansion; no optimality claim.
    Returns (support, solver result, solver calls).
    """
    n = B.shape[1]
    layer = [()]
    best = None
    calls = 0
    for _ in range(rank):
        cand = {}
        for old in layer:
            for j in range(n):
                if j in old:
                    continue
                sup = tuple(sorted(old + (j,)))
                if sup in cand:
                    continue
                cand[sup] = budgeted_selective_direction(a, C, B[:, list(sup)], dose, collateral_budget=budget)
                calls += 1
        layer = sorted(cand, key=lambda k: (-cand[k]['gain'], k))[:beam]
        top = layer[0]
        if best is None or cand[top]['gain'] > best[1]['gain']:
            best = (top, cand[top])
    return best[0], best[1], calls


def solve_proposal(a, C, B, dose, budget, support):
    """Re-solve on a fixed support (columns of B) or the full span when support is None."""
    Bs = B if support is None else B[:, list(support)]
    return budgeted_selective_direction(a, C, Bs, dose, collateral_budget=budget)


def sym_sqrt(G):
    w, V = np.linalg.eigh(G)
    if w.min() <= 0:
        raise ValueError('metric not positive definite')
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


def timed(fn, *args, **kw):
    t0 = time.process_time()
    out = fn(*args, **kw)
    return out, time.process_time() - t0
