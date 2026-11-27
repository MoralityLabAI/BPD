"""Aggregate receipts: drift tables, a-priori bounds, economics contrasts, registered family, preflight.

``--stage engineering`` reads results/engineering (drift + economics on 29200/29201).
``--stage development`` reads results/development (economics on 29301..29308 under DEV_LOCK.json)
and evaluates the frozen family. Writes <stage>/summary.json.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

from mechinterp.bpd.experiments.training_steering_stability.stats import _bootstrap, _holm, exact_sign_flip

from .common import CHECKPOINT_UPDATES, DEVELOPMENT_ROOTS, ENGINEERING_ROOTS, HERE, REPO, dump_json, recipe

ENG = HERE / 'results' / 'engineering'
DRIFT_DIRS = {28000: 'drift_historical_28000', 28001: 'drift_historical_28001',
              29200: 'drift_29200', 29201: 'drift_29201'}


def adamw_kappa(t, b1=.9, b2=.999):
    rho = b1 * b1 / b2
    S = (1 - rho ** t) / (1 - rho)
    return (1 - b1) * math.sqrt(S * (1 - b2 ** t)) / ((1 - b1 ** t) * math.sqrt(1 - b2))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def apriori_vs_measured(root, ddir: Path) -> dict:
    """AdamW displacement bound and factor-frame bound against saved checkpoints."""
    rc = recipe()
    states = {t: torch.load(ddir / f'ckpt_u{t:03d}.pt', map_location='cpu', weights_only=True)
              for t in CHECKPOINT_UPDATES if (ddir / f'ckpt_u{t:03d}.pt').is_file()}
    if 0 not in states:
        return {}
    flat = {t: torch.cat([v.flatten().double() for k, v in sorted(s.items())
                          if k in ('input.weight', 'input.bias', 'up.V', 'up.U', 'down.weight', 'norm.weight',
                                   'output.weight', 'output.bias')]) for t, s in states.items()}
    U0 = states[0]['up.U'].double().numpy()
    smin = float(np.linalg.svd(U0, compute_uv=False).min())
    d = flat[0].numel()
    out = {'sigma_min_U0': smin, 'parameters': d, 'rows': []}
    bound = 0.0
    prev = 0
    theta0_norm = float(flat[0].norm())
    for t in sorted(states):
        for step in range(prev + 1, t + 1):
            # ||theta_{step-1}|| <= ||theta_0|| + displacement bound so far (triangle inequality)
            bound += rc.curation_learning_rate * (adamw_kappa(step) * math.sqrt(d)
                                                  + rc.weight_decay * (theta0_norm + bound))
        prev = t
        dU = states[t]['up.U'].double().numpy() - U0
        nU = float(np.linalg.norm(dU, 2))
        out['rows'].append(dict(update=t, measured_param_distance=float((flat[t] - flat[0]).norm()),
                                adamw_apriori_bound=bound, measured_dU=nU,
                                frame_residual_apriori_bound_measured_dU=nU / smin,
                                frame_residual_apriori_bound_adamw=bound / smin))
    return out


def drift_summary() -> dict:
    out = {}
    for root, name in DRIFT_DIRS.items():
        p = ENG / name / 'drift.json'
        if not p.is_file():
            continue
        r = json.loads(p.read_text(encoding='utf-8'))
        out[str(root)] = dict(summary=r['summary'], row_parity=r['checkpoints']['0'].get('row_parity_max_abs'),
                              memory=r.get('memory'), apriori=apriori_vs_measured(root, ENG / name))
        # frame residual measured (write space, cached full proposal, physical transport)
        fr = {}
        for t, ck in r['checkpoints'].items():
            vals = [e['spaces']['write'].get('cached_full_physical', {}).get('frame_residual') for e in ck['edges']]
            vals = [v for v in vals if v is not None]
            if vals:
                fr[t] = dict(mean=float(np.mean(vals)), max=float(np.max(vals)))
        out[str(root)]['measured_frame_residual'] = fr
        # remainder scaling exponent from dose grid at update 0
        ex = {}
        for sp in ('state', 'write'):
            xs, ys = [], []
            for e in r['checkpoints']['0']['edges']:
                g = e['spaces'][sp]['dose_grid']
                for dose, o in g.items():
                    if o.get('target_remainder', 0) > 0:
                        xs.append(math.log(float(dose))); ys.append(math.log(o['target_remainder']))
            if len(xs) > 2:
                ex[sp] = float(np.polyfit(xs, ys, 1)[0])
        out[str(root)]['remainder_dose_exponent_fit'] = ex
    return out


def pooled_drift(ds: dict) -> dict:
    keys = None
    pooled = {}
    for t in map(str, CHECKPOINT_UPDATES):
        rows = [ds[r]['summary'][t] for r in ds if t in ds[r]['summary']]
        if not rows:
            continue
        keys = rows[0].keys()
        agg = {}
        for k in keys:
            vals = [row.get(k) for row in rows]
            if all(isinstance(v, (int, float)) and v is not None for v in vals):
                agg[k] = float(np.mean(vals))
            elif all(isinstance(v, list) and len(v) == 2 for v in vals):
                agg[k] = [sum(v[0] for v in vals), sum(v[1] for v in vals)]
            elif all(isinstance(v, dict) for v in vals) and k.endswith('admission_k4'):
                c = {}
                for v in vals:
                    for kk, n in v.items():
                        c[kk] = c.get(kk, 0) + n
                agg[k] = c
            elif k == 'cost_cpu_s_per_edge':
                cc = {}
                for v in vals:
                    for kk, x in v.items():
                        cc.setdefault(kk, []).append(x)
                agg[k] = {kk: float(np.mean(x)) for kk, x in cc.items()}
        pooled[t] = agg
    return pooled


def econ_summary(base: Path = ENG, roots=ENGINEERING_ROOTS, prefix: str = 'economics') -> dict:
    out = {}
    for root in roots:
        p = base / f'{prefix}_{root}' / 'economics.json'
        if not p.is_file():
            continue
        r = json.loads(p.read_text(encoding='utf-8'))
        arms = {}
        for tag, res in r['results'].items():
            e = res['eval']
            logs = res['logs']
            arms[tag] = dict(windows=res['windows'], cpu=res['cpu_seconds'], overrun=res['overrun_cpu'],
                             counters=res['counters'], sha=res['final_state_sha256'],
                             atlas=res.get('atlas'), cache_hits=res['cache']['hits'],
                             acc={d: e['delays'][d]['accuracy'] for d in e['delays']},
                             correct={d: e['delays'][d]['correct_count'] for d in e['delays']},
                             n=e['delays']['31']['n'],
                             ce31=e['delays']['31']['cross_entropy'],
                             bind={d: e['binding'][d]['accuracy'] for d in e['binding']},
                             feedback_ordinary_loss=e['feedback_ordinary_loss'],
                             feedback_mechanistic_loss=e['feedback_mechanistic_loss'],
                             panel=e['panel_hybrid_matched'],
                             audit_windows=sum(1 for l in logs if l['audit']),
                             atlas_reuse=sum(1 for l in logs if l.get('observer') == 'atlas_reuse'),
                             refresh=sum(1 for l in logs if l.get('observer') == 'refresh'),
                             trace_residual_mean=(float(np.mean([l['trace_residual'] for l in logs if 'trace_residual' in l]))
                                                  if any('trace_residual' in l for l in logs) else None),
                             horizon=horizon_test(logs))
        out[str(root)] = dict(arms=arms, mech_setup_cpu=r['mech_setup_cpu'], memory=r.get('memory'))
    return out


def horizon_test(logs, horizons=(1, 4, 8)) -> dict | None:
    """Does summed multi-window predicted gain track longer-horizon realized feedback change
    better than the one-step predicted gain? Uses consecutive audited windows only."""
    seq = [l for l in logs if l.get('audit') and 'cal_before' in l and 'pred_commit' in l]
    if len(seq) < 12:
        return None
    out = {}
    cb = [l['cal_before'] for l in seq]
    pc = [l['pred_commit'] if l['accepted'] else 0.0 for l in seq]
    for h in horizons:
        tgt = [cb[i] - cb[i + h] for i in range(len(seq) - h)]
        one = pc[:len(tgt)]
        multi = [sum(pc[i:i + h]) for i in range(len(tgt))]
        out[str(h)] = dict(n=len(tgt), spearman_one_step=spearman(one, tgt), spearman_multi_window=spearman(multi, tgt))
    out['one_step_pred_vs_realized'] = spearman([l['pred_commit'] for l in seq], [l['realized_commit'] for l in seq])
    return out


def contrasts(econ: dict) -> dict:
    out = {}
    pairs = [('D_atlas_mms16', 'A_ordinary'), ('D_atlas_mms16', 'C_mech_fresh'), ('D_atlas_mms16', 'B_ord_grad'),
             ('D_atlas_mms16', 'E_shuffled_mms16'), ('C_mech_fresh', 'A_ordinary'), ('B_ord_grad', 'A_ordinary'),
             ('F_obs_only_mms16', 'A_ordinary')]
    for budget in ('updates', 'cpu'):
        for a, b in pairs:
            per_root = {}
            for root, r in econ.items():
                ta, tb = f'{budget}_{a}', f'{budget}_{b}'
                if ta in r['arms'] and tb in r['arms']:
                    per_root[root] = {d: r['arms'][ta]['acc'][d] - r['arms'][tb]['acc'][d] for d in ('31', '63', '127')}
            if per_root:
                out[f'{budget}:{a}-{b}'] = dict(per_root=per_root,
                                                 mean_d31=float(np.mean([v['31'] for v in per_root.values()])))
    return out


def parity(econ: dict) -> dict:
    out = {}
    for root, r in econ.items():
        A = r['arms']
        out[root] = dict(
            C_preview_equals_C_probe=A.get('updates_C_mech_fresh', {}).get('sha') == A.get('updates_C_probe', {}).get('sha'),
            F_obs_only_equals_A=A.get('updates_F_obs_only_mms16', {}).get('sha') == A.get('updates_A_ordinary', {}).get('sha'),
            C_probe_cpu=A.get('updates_C_probe', {}).get('cpu'), C_preview_cpu=A.get('updates_C_mech_fresh', {}).get('cpu'))
    return out


def preflight(econ: dict, experiment_id: str = 'BPD-SCCG10-MMS-TRM-engineering') -> dict | None:
    rows = []
    for root, r in econ.items():
        A = r['arms']
        if not all(t in A for t in ('cpu_D_atlas_mms16', 'cpu_A_ordinary', 'cpu_E_shuffled_mms16')):
            continue
        n = A['cpu_A_ordinary']['n']
        ca, cd, ce = (A[t]['correct']['31'] for t in ('cpu_A_ordinary', 'cpu_D_atlas_mms16', 'cpu_E_shuffled_mms16'))
        rows.append({'unit_id': root, 'paired_effect': f'{cd - ca}/{n}', 'sham_effects': [f'{ce - ca}/{n}']})
    if not rows:
        return None
    spec = {'schema_version': 1, 'experiment_id': experiment_id,
            'independent_unit': 'independently initialized TRM model root with paired descendants',
            'alpha': '0.05', 'scientific_min_effect': '0.05', 'target_effect': '0.10',
            'measurement_resolution': f'1/{n}', 'sham_quantile': '0.95', 'sham_confidence': '0.95',
            'planned_comparisons': 4, 'confirm_units_planned': 8, 'pilot': rows,
            'pilot_from_development_only': True, 'heldout_sealed': False, 'units_independent': True}
    pf = REPO / 'mechinterp/bpd/experiments/repair_portfolio/evidence/preflight.py'
    spec_l = importlib.util.spec_from_file_location('sccg10_preflight', pf)
    mod = importlib.util.module_from_spec(spec_l)
    spec_l.loader.exec_module(mod)
    return dict(input=spec, result=mod.preflight(spec))


FAMILY = (('cpu_D_atlas_mms16', 'cpu_A_ordinary', 'primary'),
          ('cpu_D_atlas_mms16', 'cpu_C_mech_fresh', 'secondary'),
          ('cpu_D_atlas_mms16', 'cpu_B_ord_grad', 'secondary'),
          ('cpu_D_atlas_mms16', 'cpu_E_shuffled_mms16', 'secondary'))


def registered_family(econ: dict, delay: str = '31') -> list[dict]:
    """Frozen DEV_LOCK family: exact root sign-flip, Holm, seeded root bootstrap."""
    rows = []
    for a, b, role in FAMILY:
        eff, per = [], {}
        for root, r in econ.items():
            A = r['arms']
            if a in A and b in A:
                n = A[a]['n']
                e = Fraction(A[a]['correct'][delay] - A[b]['correct'][delay], n)
                eff.append(e); per[root] = float(e)
        if not eff:
            continue
        sf = exact_sign_flip(eff)
        rows.append(dict(contrast=f'{a} - {b}', role=role, n_roots=len(eff), per_root=per,
                         macro_effect=float(sum(eff) / len(eff)), positive=sum(e > 0 for e in eff),
                         negative=sum(e < 0 for e in eff), sign_flip=sf,
                         bootstrap=_bootstrap(eff, 20000, 20261127) if len(eff) > 1 else None,
                         p_fraction=Fraction(sf['p_fraction'])))
    adj = _holm([r['p_fraction'] for r in rows]) if rows else []
    for r, q in zip(rows, adj):
        r['holm_p'] = float(q)
        r['meets_planning_floor_5pp'] = r['macro_effect'] >= 0.05
        r.pop('p_fraction')
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('engineering', 'development'), default='engineering')
    a = ap.parse_args()
    if a.stage == 'development':
        base = HERE / 'results' / 'development'
        econ = econ_summary(base, DEVELOPMENT_ROOTS)
        rep = dict(economics=econ, contrasts=contrasts(econ), parity=parity(econ),
                   registered_family=registered_family(econ),
                   evalmath_preflight=preflight(econ, 'BPD-SCCG10-MMS-TRM-development'),
                   evidence_class='DEVELOPMENT_ONLY', lock='DEV_LOCK.json')
        dump_json(base / 'summary.json', rep)
        print(json.dumps(dict(family=[(r['contrast'], round(r['macro_effect'], 4), r['holm_p'])
                                      for r in rep['registered_family']],
                              preflight=(rep['evalmath_preflight'] or {}).get('result', {}).get('decision')), indent=1))
        return 0
    ds = drift_summary()
    econ = econ_summary()
    econ_b = econ_summary(prefix='economics_batched')
    rep = dict(drift_per_root=ds, drift_pooled=pooled_drift(ds), economics=econ, contrasts=contrasts(econ),
               registered_family=registered_family(econ),
               parity=parity(econ), evalmath_preflight=preflight(econ),
               economics_batched=econ_b, contrasts_batched=contrasts(econ_b), parity_batched=parity(econ_b),
               registered_family_batched=registered_family(econ_b),
               evidence_class='ENGINEERING_ONLY',
               note='Engineering roots 29200/29201 and historical replay roots 28000/28001. No development roots scored.')
    dump_json(ENG / 'summary.json', rep)
    print(json.dumps(dict(parity=rep['parity'], parity_batched=rep['parity_batched'],
                          contrasts_batched={k: v['mean_d31'] for k, v in rep['contrasts_batched'].items()},
                          contrasts={k: v['mean_d31'] for k, v in rep['contrasts'].items()},
                          preflight=(rep['evalmath_preflight'] or {}).get('result', {}).get('decision')), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
