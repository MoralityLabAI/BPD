"""Workstream B: native checkpoint drift instrument.

From a common warmup checkpoint, run the ordinary committed update stream at the
curation learning rate and inspect checkpoints after 0,1,2,4,8,16,32,64 committed
updates. A fixed TRAIN-fit panel of semantic single-slot edits (one nuisance draw
per edge, four queries, delays 2/7/31) is observed at every checkpoint in two
separate action spaces (carried state; native low-visit factor write). Full
hybrid replacement is a privileged diagnostic only.

Every number is an actual measurement on that panel at that checkpoint. Operator
norms of derivative differences are exact for the measured checkpoint pair; they
are not uniform bounds over other inputs or other checkpoints.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..trm_curation.data import edge_batch
from .admission import assess
from .common import Recipe, dump_json, make_trm, seed10, state_sha256, views
from . import native
from .support import beam_support, resolve_fixed, solve_on

CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64)


def panel_edges(root: int, recipe: Recipe) -> tuple:
    """Six edges per edit slot, distinct (old,new) transitions within slot, TRAIN fit only."""
    fit = views(recipe)['fit']
    per = recipe.panel_edges // 4
    out = []
    for s in range(4):
        cands = sorted((e for e in fit if e[2] == s),
                       key=lambda e: hashlib.sha256(repr((seed10(root, 'drift_panel'), e)).encode()).digest())
        seen = set()
        for e in cands:
            t = (e[0][s], e[1][s])
            if t in seen:
                continue
            seen.add(t)
            out.append(e)
            if len(seen) == per:
                break
    return tuple(out)


def panel_batches(root: int, recipe: Recipe, edges) -> list:
    dmax = max(recipe.panel_delays)
    return [edge_batch((e,), root=root, seed=seed10(root, 'drift_panel_nuisance', i), delay=dmax,
                       split='train', noise=recipe.noise) for i, e in enumerate(edges)]


def stream_batch(root: int, step: int, recipe: Recipe):
    fit = views(recipe)['fit']
    delay = recipe.train_delays[step % len(recipe.train_delays)]
    g = torch.Generator().manual_seed(seed10(root, 'drift_stream', step))
    idx = torch.randint(0, len(fit), (recipe.selected_per_window,), generator=g).tolist()
    b = edge_batch(tuple(fit[i] for i in idx), root=root, seed=seed10(root, 'drift_stream_nuisance', step),
                   delay=delay, split='train', noise=recipe.noise)
    return torch.cat((b.receiver, b.hybrid)), torch.cat((b.labels_r, b.labels_h))


def _angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angle sines between column spaces of orthonormal A, B."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.zeros(0)
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.sqrt(np.clip(1 - np.clip(s, 0, 1) ** 2, 0, 1))


def _rowspace(M: np.ndarray, rtol=1e-9) -> np.ndarray:
    _, s, vt = np.linalg.svd(M, full_matrices=False)
    r = int(np.count_nonzero(s > rtol * s[0])) if len(s) and s[0] > 0 else 0
    return vt[:r].T


def _families(obs, Jw, U, recipe: Recipe) -> dict[str, dict]:
    """Live SCCG directions in each action space. Returns physical directions."""
    dose, eta = recipe.dose, recipe.collateral_budget
    lay = obs.layout
    aw, Cw = Jw[lay['target']], Jw[lay['protected']]
    I = np.eye(len(obs.a))
    fam = {}
    t0 = time.process_time()
    r = solve_on(obs.a, obs.C, I, dose, eta)
    fam['state_full'] = {'support': tuple(range(len(obs.a))), 'delta': r['delta'], 'lin_gain': r['gain'],
                         'cpu': time.process_time() - t0}
    t0 = time.process_time()
    r0 = solve_on(obs.a, obs.C, I, dose, 0.0)
    fam['state_null'] = {'support': tuple(range(len(obs.a))), 'delta': r0['delta'], 'lin_gain': r0['gain'],
                         'null_rank': r0['null_rank'], 'cpu': time.process_time() - t0}
    bs = beam_support(obs.a, obs.C, I, dose, eta, recipe.support_k, recipe.support_beam)
    for k in (4, recipe.support_k):
        s, rr = bs['best_by_size'][k]
        fam[f'state_k{k}'] = {'support': s, 'delta': rr['delta'], 'lin_gain': rr['gain'],
                              'cpu': bs['cpu_seconds'], 'solves': bs['solves']}
    B = U.T
    t0 = time.process_time()
    r = solve_on(aw, Cw, B, dose, eta)
    fam['write_full'] = {'support': tuple(range(U.shape[0])), 'delta': r['delta'], 'lin_gain': r['gain'],
                         'coef': np.linalg.lstsq(B, r['delta'], rcond=None)[0], 'cpu': time.process_time() - t0}
    bw = beam_support(aw, Cw, B, dose, eta, recipe.support_k, recipe.support_beam)
    for k in (4, recipe.support_k):
        s, rr = bw['best_by_size'][k]
        coef = np.zeros(U.shape[0])
        coef[list(s)] = np.linalg.lstsq(B[:, list(s)], rr['delta'], rcond=None)[0]
        fam[f'write_k{k}'] = {'support': s, 'delta': rr['delta'], 'lin_gain': rr['gain'], 'coef': coef,
                              'cpu': bw['cpu_seconds'], 'solves': bw['solves']}
    return fam


def _apply(model, panel, delays, fam_name, delta, clean=None):
    if fam_name.startswith('state'):
        return native.finite_outcome(model, panel, delays, state_delta=delta, clean=clean)
    return native.finite_outcome(model, panel, delays, write_delta=delta, write_site='low', clean=clean)


def _strip(o):
    return {k: v for k, v in o.items() if k != 'delta_margins'}


def _flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).double().numpy()


def run_drift(root: int, warm_path, out_dir: Path, recipe: Recipe, *, checkpoints=CHECKPOINTS,
              max_edges: int | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    model = make_trm(recipe)
    model.load_state_dict(torch.load(warm_path, weights_only=True))
    warm_sha = state_sha256(model)
    opt = torch.optim.AdamW(model.parameters(), lr=recipe.curation_learning_rate,
                            weight_decay=recipe.weight_decay)
    edges = panel_edges(root, recipe)[:max_edges]
    batches = panel_batches(root, recipe, edges)
    panels = [native.make_panel(b) for b in batches]
    delays = recipe.panel_delays
    theta0 = _flat_params(model)
    ref: dict[int, dict] = {}
    records = []
    step = 0
    t_all = time.process_time()
    for ck in checkpoints:
        while step < ck:
            model.train()
            x, y = stream_batch(root, step, recipe)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
            opt.step()
            step += 1
        model.eval()
        theta = _flat_params(model)
        U = native.write_frame(model)
        ck_sha = state_sha256(model)
        for i, (e, p) in enumerate(zip(edges, panels)):
            t0 = time.process_time()
            obs = native.observe_state(model, p, delays)
            cost_state_obs = time.process_time() - t0
            t0 = time.process_time()
            Jw = native.write_jacobian(model, p, delays).double().numpy()
            cost_write_obs = time.process_time() - t0
            fam = _families(obs, Jw, U, recipe)
            lay = obs.layout
            clean_t = native.panel_logits(model, p, delays)
            rec: dict[str, Any] = {'checkpoint': ck, 'checkpoint_sha256': ck_sha, 'edge_index': i,
                                   'edge': [list(e[0]), list(e[1]), e[2]],
                                   'param_displacement': float(np.linalg.norm(theta - theta0)),
                                   'param_rel_displacement': float(np.linalg.norm(theta - theta0) / np.linalg.norm(theta0)),
                                   'cost': {'state_observe': cost_state_obs, 'write_observe': cost_write_obs},
                                   'clean_target_margin': float(obs.clean[lay['target']]),
                                   'sv_state_C': np.linalg.svd(obs.C, compute_uv=False).tolist(),
                                   'sv_write_reduced': np.linalg.svd(Jw[lay['protected']] @ U.T, compute_uv=False).tolist(),
                                   'families': {}}
            # privileged full hybrid replacement
            hy = obs.hybrid_dir
            rec['hybrid_matched'] = _strip(native.finite_outcome(model, p, delays, state_delta=hy * recipe.dose, clean=clean_t))
            rec['hybrid_natural'] = _strip(native.finite_outcome(model, p, delays, state_delta=hy * obs.natural_norm, clean=clean_t))
            rec['hybrid_natural']['natural_norm'] = obs.natural_norm
            for name, f in fam.items():
                live = _apply(model, p, delays, name, f['delta'], clean_t)
                lin_pred = (obs.rows if name.startswith('state') else Jw) @ f['delta']
                frec = {'live': _strip(live), 'live_lin_gain': f['lin_gain'], 'support': list(f['support'])[:16],
                        'support_size': len(f['support']), 'search_cpu': f['cpu'],
                        'live_finite_error_target': abs(live['target_gain'] - lin_pred[lay['target']]),
                        'live_finite_error_protected': float(np.linalg.norm(live['delta_margins'][lay['protected']] - lin_pred[lay['protected']]))}
                if ck == 0:
                    ref.setdefault(i, {})[name] = {'delta': f['delta'].copy(), 'support': f['support'],
                                                   'coef': f.get('coef'), 'a': (obs.a if name.startswith('state') else Jw[lay['target']]).copy(),
                                                   'C': (obs.C if name.startswith('state') else Jw[lay['protected']]).copy(),
                                                   'U': U.copy(), 'gain': live['target_gain']}
                r0 = ref[i][name]
                aL = obs.a if name.startswith('state') else Jw[lay['target']]
                CL = obs.C if name.startswith('state') else Jw[lay['protected']]
                d0 = r0['delta']
                # cheap drift observation of the cached direction: JVP + finite verification
                t0 = time.process_time()
                if name.startswith('state'):
                    jv = native.state_jvp(model, p, delays, torch.as_tensor(d0, dtype=torch.float32)).double().numpy()
                else:
                    jv = Jw @ d0  # write-space JVP equivalent from measured Jacobian (cost not charged here)
                cost_jvp = time.process_time() - t0
                t0 = time.process_time()
                cached = _apply(model, p, delays, name, d0, clean_t)
                cost_verify = time.process_time() - t0
                frec['cached_physical'] = _strip(cached)
                frec['cost_cached_jvp'] = cost_jvp
                frec['cost_cached_verify'] = cost_verify
                frec['alpha_u'] = float(abs((aL - r0['a']) @ d0))
                frec['gamma_u'] = float(np.linalg.norm((CL - r0['C']) @ d0))
                frec['alpha_op'] = float(np.linalg.norm(aL - r0['a']))
                frec['gamma_op'] = float(np.linalg.norm(CL - r0['C'], 2))
                frec['a_rel_drift'] = float(np.linalg.norm(aL - r0['a']) / max(np.linalg.norm(r0['a']), 1e-30))
                frec['C_rel_drift'] = float(np.linalg.norm(CL - r0['C']) / max(np.linalg.norm(r0['C']), 1e-30))
                frec['C_drift_rank_95'] = int(np.searchsorted(np.cumsum(np.linalg.svd(CL - r0['C'], compute_uv=False) ** 2)
                                                              / max(np.sum((CL - r0['C']) ** 2), 1e-300), 0.95) + 1) if ck else 0
                rs0, rsk = _rowspace(r0['C']), _rowspace(CL)
                frec['rowspace_max_sin'] = float(_angles(rs0, rsk).max()) if rs0.size and rsk.size else 0.0
                s0 = np.linalg.svd(r0['C'], compute_uv=False)
                frec['wedin_bound'] = float(frec['gamma_op'] / max(s0[-1] - frec['gamma_op'], 1e-30)) if s0[-1] > frec['gamma_op'] else None
                frec['weyl_ok'] = bool(np.all(np.abs(np.linalg.svd(CL, compute_uv=False) - s0) <= frec['gamma_op'] * (1 + 1e-9) + 1e-12))
                frec['jvp_vs_verify_target'] = float(abs(jv[lay['target']] - cached['target_gain']))
                # support reuse: re-solve on stored support with live derivative (needs live observation)
                if name not in ('state_full', 'state_null', 'write_full'):
                    Bm = np.eye(len(aL)) if name.startswith('state') else U.T
                    rr = resolve_fixed(aL, CL, Bm, r0['support'], recipe.dose, recipe.collateral_budget)
                    frec['support_resolve'] = _strip(_apply(model, p, delays, name, rr['delta'], clean_t))
                    frec['support_resolve_lin_gain'] = rr['gain']
                if name.startswith('write'):
                    # factor-frame transport: keep coefficients, use the current frame
                    coef = r0['coef']
                    dT = coef @ U
                    nT = np.linalg.norm(dT)
                    if nT > 0:
                        dT = dT * (recipe.dose / nT)
                    frec['cached_factor_transport'] = _strip(_apply(model, p, delays, name, dT, clean_t))
                    G0, Gk = r0['U'] @ r0['U'].T, U @ U.T
                    ev = np.linalg.eigvals(np.linalg.solve(G0, Gk)).real
                    frec['metric_gen_eig_min'] = float(ev.min())
                    frec['metric_gen_eig_max'] = float(ev.max())
                    frec['frame_rowspace_max_sin'] = float(_angles(_rowspace(r0['U']), _rowspace(U)).max())
                # SCCG-10 conditional admission with exact checkpoint-pair operator norms and the
                # sampled finite remainder at d0 (not a uniform remainder: label SAMPLED).
                rf = abs(cached['target_gain'] - jv[lay['target']])
                rp = float(np.linalg.norm(cached['delta_margins'][lay['protected']] - jv[lay['protected']]))
                Gm = np.eye(len(aL)) if name.startswith('state') else None
                if Gm is not None:
                    from fractions import Fraction as Fr
                    def fr(x):
                        return Fr(float(x)).limit_denominator(10**9)
                    sup = list(r0['support'])
                    a_red = [fr(v) for v in r0['a'][sup]]
                    C_red = [[fr(v) for v in row[sup]] for row in r0['C']]
                    u_red = [fr(v) for v in d0[sup]]
                    n2 = sum(v * v for v in u_red)
                    dose_q = max(fr(recipe.dose), Fr(0))
                    if n2 > dose_q * dose_q:
                        u_red = [v * dose_q * dose_q / n2 for v in u_red]  # rounding guard
                    try:
                        res = assess([[Fr(int(i == j)) for j in range(len(sup))] for i in range(len(sup))],
                                     a_red, C_red, u_red, dose=dose_q,
                                     collateral=fr(recipe.collateral_budget), alpha=fr(frec['alpha_op']),
                                     gamma=fr(frec['gamma_op']), target_remainder=fr(rf),
                                     protected_remainder=fr(rp), min_gain=Fr(1, 100)) if len(sup) <= 8 else None
                    except ValueError as exc:
                        res = {'verdict': f'INPUT_REJECTED:{exc}'}
                    if res is not None:
                        frec['sccg10'] = {'verdict': res['verdict'], 'scope': 'SAMPLED_REMAINDER_NOT_UNIFORM',
                                          'target_lower': float(Fr(res['target_lower'])) if 'target_lower' in res else None,
                                          'target_upper': float(Fr(res['target_upper'])) if 'target_upper' in res else None}
                rec['families'][name] = frec
            records.append(rec)
            print(json.dumps({'root': root, 'ck': ck, 'edge': i,
                              'state_full_live': round(rec['families']['state_full']['live']['target_gain'], 4),
                              'state_full_cached': round(rec['families']['state_full']['cached_physical']['target_gain'], 4),
                              'C_rel': round(rec['families']['state_full']['C_rel_drift'], 4),
                              'cpu': round(time.process_time() - t_all, 1)}), flush=True)
    result = {'root': root, 'warm_sha256': warm_sha, 'recipe': recipe.as_dict(), 'checkpoints': list(checkpoints),
              'panel_edges': [[list(e[0]), list(e[1]), e[2]] for e in edges],
              'panel_fingerprints': [b.fingerprint() for b in batches],
              'records': records, 'cpu_seconds': time.process_time() - t_all,
              'scope': 'TRAIN fit panel; exact native measurements at listed checkpoints; no uniform bound'}
    dump_json(out_dir / 'drift.json', result)
    dump_json(out_dir / 'drift_summary.json', summarize(result))
    return result


def summarize(result: dict) -> dict:
    recs = result['records']
    fams = sorted(recs[0]['families'])
    out = {'root': result['root'], 'checkpoints': {}}
    for ck in result['checkpoints']:
        rs = [r for r in recs if r['checkpoint'] == ck]
        row = {'n_edges': len(rs), 'param_displacement': float(np.mean([r['param_displacement'] for r in rs])),
               'cost_state_observe': float(np.mean([r['cost']['state_observe'] for r in rs])),
               'cost_write_observe': float(np.mean([r['cost']['write_observe'] for r in rs])),
               'clean_target_margin_mean': float(np.mean([r['clean_target_margin'] for r in rs])),
               'hybrid_matched_target_gain': float(np.mean([r['hybrid_matched']['target_gain'] for r in rs])),
               'hybrid_matched_flip_rate': float(np.mean([r['hybrid_matched']['target_flip'] for r in rs])),
               'hybrid_natural_flip_rate': float(np.mean([r['hybrid_natural']['target_flip'] for r in rs])),
               'hybrid_natural_corrupted': int(sum(r['hybrid_natural']['corrupted'] for r in rs)),
               'families': {}}
        for f in fams:
            fr = [r['families'][f] for r in rs]
            m = lambda key, sub=None: float(np.mean([(x[key][sub] if sub else x[key]) for x in fr]))
            d = {'live_target_gain': m('live', 'target_gain'), 'cached_target_gain': m('cached_physical', 'target_gain'),
                 'live_protected_norm': m('live', 'protected_change_norm'),
                 'cached_protected_norm': m('cached_physical', 'protected_change_norm'),
                 'cached_over_live_gain': float(np.mean([x['cached_physical']['target_gain'] / x['live']['target_gain']
                                                         for x in fr if abs(x['live']['target_gain']) > 1e-9] or [float('nan')])),
                 'live_flip_rate': float(np.mean([x['live']['target_flip'] for x in fr])),
                 'cached_flip_rate': float(np.mean([x['cached_physical']['target_flip'] for x in fr])),
                 'live_corrupted': int(sum(x['live']['corrupted'] for x in fr)),
                 'cached_corrupted': int(sum(x['cached_physical']['corrupted'] for x in fr)),
                 'eligible_unaffected': int(sum(x['live']['eligible_unaffected_correct'] for x in fr)),
                 'alpha_u': m('alpha_u'), 'gamma_u': m('gamma_u'), 'alpha_op': m('alpha_op'), 'gamma_op': m('gamma_op'),
                 'a_rel_drift': m('a_rel_drift'), 'C_rel_drift': m('C_rel_drift'),
                 'rowspace_max_sin': m('rowspace_max_sin'),
                 'weyl_all_ok': bool(all(x['weyl_ok'] for x in fr)),
                 'cached_protected_over_budget_rate': float(np.mean([x['cached_physical']['protected_change_norm'] > result['recipe']['collateral_budget'] for x in fr])),
                 'live_protected_over_budget_rate': float(np.mean([x['live']['protected_change_norm'] > result['recipe']['collateral_budget'] for x in fr])),
                 'live_finite_error_target': m('live_finite_error_target'),
                 'live_finite_error_protected': m('live_finite_error_protected'),
                 'search_cpu': m('search_cpu'), 'cost_cached_jvp': m('cost_cached_jvp'),
                 'cost_cached_verify': m('cost_cached_verify')}
            if 'support_resolve' in fr[0]:
                d['support_resolve_target_gain'] = float(np.mean([x['support_resolve']['target_gain'] for x in fr]))
            if 'cached_factor_transport' in fr[0]:
                d['factor_transport_target_gain'] = float(np.mean([x['cached_factor_transport']['target_gain'] for x in fr]))
                d['metric_gen_eig_min'] = float(np.min([x['metric_gen_eig_min'] for x in fr]))
                d['metric_gen_eig_max'] = float(np.max([x['metric_gen_eig_max'] for x in fr]))
                d['frame_rowspace_max_sin'] = float(np.max([x['frame_rowspace_max_sin'] for x in fr]))
            v = [x['sccg10']['verdict'] for x in fr if 'sccg10' in x]
            if v:
                d['sccg10_verdicts'] = {k: v.count(k) for k in sorted(set(v))}
                acc = [x for x in fr if x.get('sccg10', {}).get('verdict') == 'CONDITIONAL_REUSE_CANDIDATE']
                d['sccg10_accept_then_violation'] = int(sum(x['cached_physical']['protected_change_norm'] > result['recipe']['collateral_budget'] for x in acc))
            row['families'][f] = d
        out['checkpoints'][str(ck)] = row
    return out
