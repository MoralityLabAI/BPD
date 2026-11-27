"""Summaries for SCCG-10 engineering and development runs.

Engineering: TRAIN-tune accuracy per arm, cost ledgers, atlas statistics.
Development: per-root paired contrasts on ordinary steering-off final-development
accuracy (exact two-sided root sign-flip, Holm over the frozen family, seeded root
bootstrap) and the pinned EvalMath strict mechinterp-power preflight. Independent
model roots are the only units.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import numpy as np

from ..training_steering_stability.stats import _bootstrap, _holm, exact_sign_flip
from .common import REPO, dump_json


def _arm_rows(res: dict, key: str) -> dict:
    out = {}
    for name, a in res['arms'].items():
        ev = a.get('eval_tune') or a['eval']['delays']
        row = {'windows': a['windows'], 'accepted': a['accepted'], 'rejected': a['rejected'],
               'curated': a['curated_windows'], 'cpu_used': round(a['cpu_used'], 2),
               'refreshes': a['refreshes'], 'audits': a['audits'],
               'ladder': a['ladder_outcomes'],
               'atlas_peak_bytes': (a['atlas'] or {}).get('peak_bytes'),
               'cpu_by_category': {k: round(v, 2) for k, v in a['ledger']['cpu'].items()}}
        for d, r in ev.items():
            row[f'acc_d{d}'] = r['accuracy']
            row[f'ce_d{d}'] = r['cross_entropy']
            if 'correct_count' in r:
                row[f'correct_d{d}'] = r['correct_count']
        if 'eval' in a:
            for d, r in a['eval']['binding'].items():
                row[f'bind_d{d}'] = r['accuracy']
        out[name] = row
    return out


def engineering(eng_dir: Path) -> dict:
    out = {}
    for res_p in sorted(eng_dir.glob('root_*/*/arms_result.json')):
        res = json.loads(res_p.read_text(encoding='utf-8'))
        warm_rec = json.loads((res_p.parent.parent / 'warmup_receipt.json').read_text(encoding='utf-8'))
        out[f'{res_p.parent.parent.name}/{res_p.parent.name}'] = {
            'mode': res['mode'], 'warm_tune_d31': warm_rec['curve'][-1]['tune_d31'],
            'arms': _arm_rows(res, 'eval_tune')}
    return out


def _preflight(spec):
    p = REPO / 'mechinterp/bpd/experiments/repair_portfolio/evidence/preflight.py'
    sp = importlib.util.spec_from_file_location('sccg10_preflight', p)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod.preflight(spec)


def development(dev_dir: Path, freeze: dict) -> dict:
    fam = freeze['statistics']['family']
    primary_delay = str(freeze['recipe']['primary_delay'])
    out = {'evidence_class': 'DEVELOPMENT_ONLY', 'modes': {}}
    for mode in ('fixed_cpu', 'fixed_update'):
        roots, per = [], {}
        for rd in sorted(dev_dir.glob('root_*')):
            p = rd / mode / 'arms_result.json'
            if not p.is_file():
                continue
            res = json.loads(p.read_text(encoding='utf-8'))
            roots.append(res['root'])
            per[res['root']] = _arm_rows(res, 'eval')
        if not roots:
            continue
        n = freeze['recipe']['eval_n']
        contrasts = {}
        for c in fam:
            a, b = c.split('-')
            eff = [Fraction(per[r][a][f'correct_d{primary_delay}'] - per[r][b][f'correct_d{primary_delay}'], n) for r in roots]
            sf = exact_sign_flip(eff)
            contrasts[c] = {'per_root_pp': [round(100 * float(e), 3) for e in eff],
                            'mean_pp': round(100 * float(sum(eff) / len(eff)), 3),
                            'positive_roots': sum(e > 0 for e in eff), 'negative_roots': sum(e < 0 for e in eff),
                            'sign_flip_p': sf['p_value'], 'sign_flip_fraction': sf['p_fraction'],
                            'bootstrap_95_pp': [round(100 * x, 3) for x in (lambda bb: (bb['lower'], bb['upper']))(_bootstrap(eff, 20000, 20261127))],
                            'meets_floor': float(sum(eff) / len(eff)) >= freeze['statistics']['practical_floor']}
        ps = [Fraction(contrasts[c]['sign_flip_fraction']) for c in fam]
        for c, h in zip(fam, _holm(ps)):
            contrasts[c]['holm_p'] = float(h)
        means = {arm: {k: float(np.mean([per[r][arm][k] for r in roots]))
                       for k in per[roots[0]][arm] if k.startswith(('acc_', 'bind_', 'ce_', 'cpu_used', 'windows', 'accepted', 'refreshes'))
                       and isinstance(per[roots[0]][arm][k], (int, float))}
                 for arm in per[roots[0]]}
        out['modes'][mode] = {'roots': roots, 'contrasts': contrasts, 'arm_means': means, 'per_root': per}
        if mode == 'fixed_cpu' and len(roots) >= 2:
            prim = fam[0]
            a, b = prim.split('-')
            sham_arm = freeze['statistics']['sham_arm']
            spec = {'schema_version': 1, 'experiment_id': 'SCCG10-MMS-TRM-development',
                    'independent_unit': 'independently initialized TRM model root with paired arm descendants',
                    'alpha': '0.05', 'scientific_min_effect': str(Fraction(freeze['statistics']['practical_floor']).limit_denominator(1000)),
                    'target_effect': '0.10', 'measurement_resolution': f'1/{n}',
                    'sham_quantile': '0.95', 'sham_confidence': '0.95',
                    'planned_comparisons': len(fam), 'confirm_units_planned': len(roots),
                    'pilot_from_development_only': True, 'heldout_sealed': False, 'units_independent': True,
                    'pilot': [{'unit_id': str(r),
                               'paired_effect': f"{per[r][a][f'correct_d{primary_delay}'] - per[r][b][f'correct_d{primary_delay}']}/{n}",
                               'sham_effects': [f"{per[r][sham_arm][f'correct_d{primary_delay}'] - per[r][b][f'correct_d{primary_delay}']}/{n}"]}
                              for r in roots]}
            out['evalmath_preflight_input'] = spec
            out['evalmath_preflight'] = _preflight(spec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('kind', choices=('engineering', 'development'))
    ap.add_argument('dir', type=Path)
    ap.add_argument('--freeze', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.kind == 'engineering':
        res = engineering(a.dir)
    else:
        res = development(a.dir, json.loads(a.freeze.read_text(encoding='utf-8')))
    dump_json(a.out, res)
    print(json.dumps(res, indent=1)[:6000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
