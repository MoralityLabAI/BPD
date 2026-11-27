"""SCCG-10 Workstream B: native checkpoint-drift instrument.

From a common warmup checkpoint, commit ordinary uniform AdamW updates
(curation lr) and inspect checkpoints after 0,1,2,4,8,16,32,64 updates.
On a fixed TRAIN feedback-view panel (24 single-slot edges, balanced over
slot and unordered value pair, orientations alternating; one nuisance draw
expanded into all four queries; delays 2/7/31) measure for two separate
action spaces (carried joint state; native factor write):

* live target/protected full-suffix derivatives and their drift from the
  checkpoint-0 observer (target G-dual norm, protected operator norm,
  spectrum, rank, row/null-space principal angles, low-rank residuals);
* physical write-metric and factor-frame drift;
* actual finite outcomes of cached, support-restricted and fresh proposals
  (target gain, protected change, flips, corruption with denominators);
* the privileged full-hybrid diagnostic (matched dose and natural dose);
* exact per-action CPU cost of screening, restricted refresh, full refresh
  and native verification.

Nothing here trains with steering or changes the ordinary update stream.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from mechinterp.bpd.experiments.trm_curation.data import edge_batch
from mechinterp.bpd.experiments.trm_curation.memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap
from mechinterp.bpd.experiments.trm_curation.observer import observe_edge
from mechinterp.bpd.experiments.trm_curation.scheduler import edge_views

from . import geometry as geo
from .common import (CHECKPOINT_UPDATES, ENGINEERING_ROOTS, HISTORICAL_ENGINEERING_ROOTS, PANEL_DELAYS,
                     accounting_dict, dump_json, historical_warmup_path, load_admission, load_checkpoint,
                     model_state_sha256, recipe, sccg_seed, source_hashes)

DOSE = 0.75
BUDGET = 0.25
SUPPORT_RANKS = (4, 8)
DOSE_GRID = (0.25, 0.75, 1.5, 3.0)
GAIN_FLOOR = Fraction(1, 10)


def drift_panel(feedback_edges, n_per_slot=6):
    """24 edges: per slot one edge per unordered value pair, orientation alternating."""
    panel = []
    for s in range(4):
        pairs = sorted({(min(r[s], h[s]), max(r[s], h[s])) for r, h, sl in feedback_edges if sl == s})
        for k, (lo, hi) in enumerate(pairs[:n_per_slot]):
            want = (lo, hi) if k % 2 == 0 else (hi, lo)
            cands = sorted(e for e in feedback_edges if e[2] == s and (e[0][s], e[1][s]) == want)
            if not cands:
                cands = sorted(e for e in feedback_edges if e[2] == s and {e[0][s], e[1][s]} == {lo, hi})
            panel.append(cands[sccg_seed(0, f'panel_pick_{s}_{lo}{hi}') % len(cands)])
    return tuple(panel)


def ordinary_update(model, opt, fit, root, step, rc):
    g = torch.Generator().manual_seed(sccg_seed(root, 'drift_train_edges', step))
    idx = torch.randint(0, len(fit), (rc.selected_per_window,), generator=g).tolist()
    delay = rc.train_delays[step % len(rc.train_delays)]
    b = edge_batch(tuple(fit[i] for i in idx), root=root, seed=sccg_seed(root, 'drift_train_nuisance', step),
                   delay=delay, split='train', noise=rc.noise)
    model.train()
    x = torch.cat((b.receiver, b.hybrid)); y = torch.cat((b.labels_r, b.labels_h))
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), rc.gradient_clip)
    opt.step()
    model.eval()
    return float(loss.detach())


def _np(t):
    return t.detach().double().cpu().numpy()


def frac(x):
    return Fraction(float(x))


def admission_on_support(adm, G_S, a0, C0, a1, C1, c, *, remainder_t, remainder_p):
    """Run the unchanged exact SCCG-10 reference on a reduced support.

    Drift (alpha, gamma) is the measured live-minus-cached derivative on the
    support; remainders are the sampled finite residuals of this one
    candidate. Neither is a uniform bound obtained without refresh, so the
    verdict is reported with scope HEURISTIC_MEASURED_INPUTS.
    """
    Ginv = np.linalg.inv(G_S)
    L = np.linalg.cholesky(Ginv)
    alpha = float(np.linalg.norm(L.T @ (a1 - a0)))
    gamma = float(np.linalg.svd((C1 - C0) @ L, compute_uv=False)[0])
    dose = float(np.sqrt(c @ G_S @ c)) * (1 + 1e-9) + 1e-12
    r = adm.assess([[frac(v) for v in row] for row in G_S], [frac(v) for v in a0],
                   [[frac(v) for v in row] for row in C0], [frac(v) for v in c],
                   dose=frac(dose), collateral=frac(BUDGET), alpha=frac(alpha), gamma=frac(gamma),
                   target_remainder=frac(remainder_t), protected_remainder=frac(remainder_p),
                   min_gain=GAIN_FLOOR)
    return dict(verdict=r['verdict'], safe=r['safe'], target_lower=float(Fraction(r['target_lower'])),
                target_upper=float(Fraction(r['target_upper'])), alpha=alpha, gamma=gamma,
                scope='HEURISTIC_MEASURED_INPUTS')


class EdgeState:
    """Checkpoint-0 cached observer and proposals for one panel edge (detached)."""


def analyse_edge(model, batch, delays, cache0: dict | None, adm, *, with_admission: bool):
    idx = geo.row_index(batch, delays)
    cost = {}
    # Native observer (carried state): exact full-suffix VJP rows + hybrid finite transport.
    obs, cost['observe_state'] = geo.timed(observe_edge, model, batch, delays, dose=DOSE)
    J_s = _np(obs['observer'])
    s = obs['post_write_state']
    with torch.no_grad():
        clean = geo.native_logits(model, batch, delays, s)
    # Native factor write Jacobian; coefficient map through live U.
    (_, J_w), cost['observe_write'] = geo.timed(geo.write_jacobian, model, batch, delays, s)
    J_w = _np(J_w)
    U = _np(model.up.U)                      # [29, 64]
    G = U @ U.T
    Gh, Gih = geo.sym_sqrt(G)
    Q = Gih @ U                              # orthonormal rows: whitened write frame
    J_alpha = J_w @ U.T
    spaces = {
        'state': dict(J=J_s, B=np.eye(J_s.shape[1]), cols=J_s.shape[1]),
        'write': dict(J=J_w @ Q.T, B=Gh, cols=G.shape[0]),  # whitened coefficients beta = G^{1/2} alpha
    }
    tgt, prot = idx.target, list(idx.protected)
    rec = dict(slot=idx.slot, old=idx.old, new=idx.new, natural_norm=obs['natural_norm'],
               hybrid_matched_target_linear=obs['target_linear'],
               hybrid_matched_target_actual=obs['target_actual'], spaces={}, cost=cost)
    # Privileged full-hybrid diagnostics.
    hyb = geo.hybrid_state(model, batch)
    rec['hybrid_natural'] = geo.finite_outcome(model, batch, delays, s, idx, clean, delta_state=hyb - s)
    rec['hybrid_matched'] = geo.finite_outcome(model, batch, delays, s, idx, clean,
                                               delta_state=obs['hybrid_direction'])
    new_cache = dict(s=s, U=U, G=G, spaces={})
    for name, sp in spaces.items():
        J, B = sp['J'], sp['B']
        a, C = J[tgt], J[prot]
        srec = {}
        live_props = {}
        t0 = time.process_time()
        full = geo.solve_proposal(a, C, B, DOSE, BUDGET, None)
        live_props['full'] = (None, full)
        for k in SUPPORT_RANKS:
            sup, res, calls = geo.greedy_support(a, C, B, DOSE, rank=k, budget=BUDGET)
            live_props[f'k{k}'] = (sup, res)
            srec[f'k{k}_solver_calls'] = calls
        cost[f'{name}_support_search'] = time.process_time() - t0
        srec['live_linear_headroom_full'] = full['gain']
        for key, (sup, res) in live_props.items():
            srec[f'live_{key}'] = _apply(model, batch, delays, s, idx, clean, name, res['delta'], Q, a, C,
                                         cost, f'{name}_verify')
            srec[f'live_{key}']['linear_gain'] = res['gain']
            srec[f'live_{key}']['support'] = list(sup) if sup is not None else None
        # Remainder scaling along the live full-span direction.
        d_unit = full['delta'] / max(np.linalg.norm(full['delta']), 1e-30)
        srec['dose_grid'] = {str(e): _apply(model, batch, delays, s, idx, clean, name, d_unit * e, Q, a, C)
                             for e in DOSE_GRID}
        new_cache['spaces'][name] = dict(a=a, C=C, J=J, props={k: (sup, res['delta']) for k, (sup, res) in live_props.items()})
        if cache0 is not None:
            c0 = cache0['spaces'][name]
            a0, C0 = c0['a'], c0['C']
            if name == 'state':
                srec['drift'] = geo.drift_metrics(a0, C0, a, C)
            else:
                # Cached and live whitened frames differ; compare in coefficient (alpha) coordinates
                # under the cached metric, and separately report metric/frame drift.
                Ja0 = cache0['J_alpha']
                srec['drift'] = geo.drift_metrics(Ja0[tgt], Ja0[prot], J_alpha[tgt], J_alpha[prot],
                                                  Ginv0=np.linalg.inv(cache0['G']))
                srec['metric_rel_change'] = float(np.linalg.norm(G - cache0['G'], 2) / np.linalg.norm(cache0['G'], 2))
                ev0, ev1 = np.linalg.eigvalsh(cache0['G']), np.linalg.eigvalsh(G)
                srec['metric_condition'] = [float(ev0[-1] / ev0[0]), float(ev1[-1] / ev1[0])]
                P1 = U.T @ np.linalg.solve(G, U)  # projector onto live rowspace(U)
                srec['frame_rowspace_sin'] = geo.principal_sin(np.linalg.qr(cache0['U'].T)[0], np.linalg.qr(U.T)[0])
            for key, (sup, delta0) in c0['props'].items():
                if name == 'state':
                    srec[f'cached_{key}'] = _apply(model, batch, delays, s, idx, clean, name, delta0, Q, a, C,
                                                   cost, f'{name}_verify_cached')
                else:
                    # delta0 is whitened in the cached frame: physical write w0 = delta0 @ Q0.
                    w0 = delta0 @ cache0['Q']
                    alpha0 = delta0 @ cache0['Gih']
                    w_comp = alpha0 @ U              # component-identity transport
                    w_proj = w0 @ P1                 # physical write projected into live factor rowspace
                    srec[f'cached_{key}'] = _apply_write(model, batch, delays, s, idx, clean, w_comp, J_w,
                                                         cost, 'write_verify_cached')
                    srec[f'cached_{key}']['transport'] = 'component_identity'
                    srec[f'cached_{key}_physical'] = _apply_write(model, batch, delays, s, idx, clean, w_proj, J_w)
                    srec[f'cached_{key}_physical']['frame_residual'] = float(
                        np.linalg.norm(w0 - w_proj) / max(np.linalg.norm(w0), 1e-30))
                    srec[f'cached_{key}']['dose_after_transport'] = float(np.linalg.norm(w_comp))
                if sup is not None:
                    # Support-restricted refresh: exact live J restricted to k columns (k JVPs).
                    t1 = time.process_time()
                    Jr = _restricted_jacobian(model, batch, delays, s, name, B[:, list(sup)], Q)
                    # Solve in support coefficients c (metric B_S^T B_S) after whitening.
                    GS = B[:, list(sup)].T @ B[:, list(sup)]
                    GSh, GSih = geo.sym_sqrt(GS)
                    resw = geo.solve_proposal(Jr[tgt] @ GSih, Jr[prot] @ GSih, np.eye(len(sup)), DOSE, BUDGET, None)
                    c = GSih @ resw['delta']
                    delta_r = B[:, list(sup)] @ c
                    cost[f'{name}_restricted_refresh_{key}'] = time.process_time() - t1
                    srec[f'restricted_{key}'] = _apply(model, batch, delays, s, idx, clean, name, delta_r, Q, a, C)
                    srec[f'restricted_{key}']['linear_gain'] = resw['gain']
                    if with_admission and key == 'k4':
                        a0S, C0S = a0 @ B[:, list(sup)], C0 @ B[:, list(sup)]
                        if name == 'write':
                            # cached coefficients live in the cached whitened frame
                            B0 = cache0['spaces'][name]['B']
                            a0S, C0S = a0 @ B0[:, list(sup)], C0 @ B0[:, list(sup)]
                        c0 = np.linalg.lstsq(B[:, list(sup)] if name == 'state' else cache0['spaces'][name]['B'][:, list(sup)],
                                             delta0, rcond=None)[0]
                        GS0 = (np.eye(len(sup)) if name == 'state'
                               else cache0['spaces'][name]['B'][:, list(sup)].T @ cache0['spaces'][name]['B'][:, list(sup)])
                        out = srec[f'cached_{key}']
                        try:
                            srec['admission_k4'] = admission_on_support(
                                adm, GS0, a0S, C0S, Jr[tgt], Jr[prot], c0,
                                remainder_t=out.get('target_remainder', 0.0),
                                remainder_p=out.get('protected_remainder_l2', 0.0))
                        except (ValueError, np.linalg.LinAlgError) as exc:
                            srec['admission_k4'] = dict(verdict='ADMISSION_INPUT_ERROR', error=str(exc))
        if name == 'write':
            new_cache['spaces'][name]['B'] = B
        rec['spaces'][name] = srec
    new_cache.update(J_alpha=J_alpha, Q=Q, Gih=Gih)
    return rec, new_cache, J_s


def _restricted_jacobian(model, batch, delays, s, name, BS, Q):
    cols = []
    for j in range(BS.shape[1]):
        v = BS[:, j]
        if name == 'state':
            t = geo.directional_derivative(model, batch, delays, s, delta_state=torch.as_tensor(v, dtype=torch.float32))
        else:
            # whitened direction v -> physical write v @ Q
            t = geo.directional_derivative(model, batch, delays, s, write=torch.as_tensor(v @ Q, dtype=torch.float32))
        cols.append(_np(t))
    return np.stack(cols, 1)


def _apply(model, batch, delays, s, idx, clean, name, delta, Q, a, C, cost=None, cost_key=None):
    t0 = time.process_time()
    if name == 'state':
        d = torch.as_tensor(delta, dtype=torch.float32)
        out = geo.finite_outcome(model, batch, delays, s, idx, clean, delta_state=d,
                                 linear_target=float(a @ delta), linear_protected=C @ delta)
    else:
        w = torch.as_tensor(delta @ Q, dtype=torch.float32)
        out = geo.finite_outcome(model, batch, delays, s, idx, clean, write=w,
                                 linear_target=float(a @ delta), linear_protected=C @ delta)
    out['dose'] = float(np.linalg.norm(delta))
    if cost is not None:
        cost[cost_key] = cost.get(cost_key, 0.0) + time.process_time() - t0
    return out


def _apply_write(model, batch, delays, s, idx, clean, w, J_w, cost=None, cost_key=None):
    t0 = time.process_time()
    lt = J_w @ w
    out = geo.finite_outcome(model, batch, delays, s, idx, clean, write=torch.as_tensor(w, dtype=torch.float32),
                             linear_target=float(lt[idx.target]), linear_protected=lt[list(idx.protected)])
    out['dose'] = float(np.linalg.norm(w))
    if cost is not None:
        cost[cost_key] = cost.get(cost_key, 0.0) + time.process_time() - t0
    return out


def run_root(root: int, warm_path: Path, out: Path, *, n_edges: int = 24, updates=CHECKPOINT_UPDATES,
             save_checkpoints: bool = True) -> dict:
    rc = recipe()
    torch.set_num_threads(rc.threads)
    torch.use_deterministic_algorithms(True)
    views = edge_views(rc.split_seed)
    panel = drift_panel(views['feedback'])[:n_edges]
    batches = [edge_batch((e,), root=root, seed=sccg_seed(root, 'panel_nuisance', i), delay=max(PANEL_DELAYS),
                          split='train', noise=rc.noise) for i, e in enumerate(panel)]
    adm = load_admission()
    model = load_checkpoint(warm_path)
    model.eval()
    theta0 = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
    opt = torch.optim.AdamW(model.parameters(), lr=rc.curation_learning_rate, weight_decay=rc.weight_decay)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(root=root, warm_checkpoint=str(warm_path.name), warm_state_sha256=model_state_sha256(model),
                  panel=[list(map(list, e[:2])) + [e[2]] for e in panel], delays=list(PANEL_DELAYS),
                  dose=DOSE, collateral_budget=BUDGET, support_ranks=list(SUPPORT_RANKS),
                  checkpoints={}, update_losses=[], source_hashes=source_hashes())
    caches: list[dict | None] = [None] * len(panel)
    step = 0
    parity_checked = False
    for t in updates:
        while step < t:
            report['update_losses'].append(ordinary_update(model, opt, views['fit'], root, step, rc))
            step += 1
        model.eval()
        theta = torch.cat([p.detach().flatten() for p in model.parameters()])
        ck = dict(update=t, state_sha256=model_state_sha256(model), factor_frame=model.factor_frame_digest(),
                  param_distance=float((theta - theta0).norm()), edges=[])
        if save_checkpoints:
            torch.save(model.state_dict(), out / f'ckpt_u{t:03d}.pt')
        t_ck = time.process_time()
        for i, b in enumerate(batches):
            rec, cache, J_s = analyse_edge(model, b, PANEL_DELAYS, caches[i], adm, with_admission=t > 0)
            if not parity_checked:
                # Row order of the instrument equals observe_edge's signed observer stack.
                s = geo.post_write_state(model, b)
                _, J_mine = geo.state_jacobian(model, b, PANEL_DELAYS, s)
                ck['row_parity_max_abs'] = float(np.abs(_np(J_mine) - J_s).max())
                parity_checked = True
            if caches[i] is None:
                caches[i] = cache
            ck['edges'].append(rec)
        ck['cpu_seconds'] = time.process_time() - t_ck
        report['checkpoints'][str(t)] = ck
        print(json.dumps({'root': root, 'update': t, 'cpu_s': round(ck['cpu_seconds'], 1),
                          'dist': round(ck['param_distance'], 5)}), flush=True)
    report['summary'] = summarize(report)
    dump_json(out / 'drift.json', report)
    return report


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def summarize(report: dict) -> dict:
    summ = {}
    for t, ck in report['checkpoints'].items():
        E = ck['edges']
        row = dict(param_distance=ck['param_distance'])
        for sp in ('state', 'write'):
            S = [e['spaces'][sp] for e in E]
            if 'drift' in S[0]:
                for k in ('alpha_rel', 'gamma_rel', 'target_cosine', 'rowspace_sin_0.001', 'nullspace_sin_0.001',
                          'dC_rank90', 'rank1_0.001'):
                    row[f'{sp}_{k}'] = _mean([s['drift'][k] for s in S])
            for prop in ('full', 'k4', 'k8'):
                for kind in ('live', 'cached', 'restricted', 'cached_physical'):
                    key = f'{kind}_{prop}' if kind != 'cached_physical' else f'cached_{prop}_physical'
                    if key not in S[0]:
                        continue
                    row[f'{sp}_{key}_gain'] = _mean([s[key]['target_gain'] for s in S])
                    row[f'{sp}_{key}_prot_l2'] = _mean([s[key]['protected_l2'] for s in S])
                    num = sum(s[key]['corruption_numerator'] for s in S)
                    den = sum(s[key]['corruption_denominator'] for s in S)
                    row[f'{sp}_{key}_corruption'] = [num, den]
                    fl = sum(s[key]['target_flipped'] for s in S)
                    el = sum(s[key]['target_eligible'] for s in S)
                    row[f'{sp}_{key}_flips'] = [fl, el]
            if 'admission_k4' in S[0]:
                row[f'{sp}_admission_k4'] = dict(Counter(s['admission_k4']['verdict'] for s in S))
        for hk in ('hybrid_natural', 'hybrid_matched'):
            row[f'{hk}_gain'] = _mean([e[hk]['target_gain'] for e in E])
            row[f'{hk}_corruption'] = [sum(e[hk]['corruption_numerator'] for e in E),
                                       sum(e[hk]['corruption_denominator'] for e in E)]
            row[f'{hk}_flips'] = [sum(e[hk]['target_flipped'] for e in E), sum(e[hk]['target_eligible'] for e in E)]
        costs = Counter()
        for e in E:
            costs.update(e['cost'])
        row['cost_cpu_s_per_edge'] = {k: v / len(E) for k, v in costs.items()}
        summ[t] = row
    return summ


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=int, required=True)
    p.add_argument('--warm', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--edges', type=int, default=24)
    p.add_argument('--updates', type=str, default=','.join(map(str, CHECKPOINT_UPDATES)))
    a = p.parse_args()
    if a.root in HISTORICAL_ENGINEERING_ROOTS:
        warm = a.warm or historical_warmup_path(a.root)
    elif a.root in ENGINEERING_ROOTS:
        if a.warm is None:
            raise SystemExit('fresh roots require --warm')
        warm = a.warm
    else:
        raise SystemExit('drift runs are restricted to engineering and historical engineering roots')
    cap = ProcessTreeMemoryCap(DEFAULT_CAP_BYTES)
    with cap:
        rep = run_root(a.root, warm, a.out, n_edges=a.edges, updates=tuple(int(u) for u in a.updates.split(',')))
        rep['memory'] = accounting_dict(cap)
        dump_json(a.out / 'drift.json', rep)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
