"""Recompute the six NC13 contrasts from immutable per-arm root scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any
import numpy as np

ROOTS = tuple(range(2601300, 2601316))
PROTOCOL_COMMIT = '069b355e8f37ec780dc6c46b82176b6f32f8e39b'
ENDPOINTS = (
    'context_gain_E', 'context_gain_D', 'shared_gap_reduction_D',
    'target_minus_shared_D', 'median_sham_minus_shared_D', 'shared_D_minus_clean',
)
REQUIRED = {
    'context_gain_E': ('mixed', 'shared_E'),
    'context_gain_D': ('mixed', 'shared_D'),
    'shared_gap_reduction_D': ('shared_E', 'target_E', 'shared_D', 'target_D'),
    'target_minus_shared_D': ('target_D', 'shared_D'),
    'median_sham_minus_shared_D': ('shared_D',) + tuple(f'sham_D_{i}' for i in range(8)),
    'shared_D_minus_clean': ('shared_D',),
}
FLOORS = (.01, .01, .01, .01, .02, .05)
BOOTSTRAPS = 16384
ALPHA = .05
FAMILY = 6
BUDGETS = {'steps': 384, 'train_A': 2048, 'train_B': 2048,
           'source_pool': 4096, 'source_per_condition': 1024,
           'calibration_pool': 512, 'activation_calibration': 128,
           'target_gradient_union': 8, 'dose': 1024, 'evaluation': 1024,
           'sham_proposals': 2048, 'shams': 8,
           'search_grid': 17, 'search_bisection': 24}


def _nll(rec: dict, arm: str) -> float:
    raw = rec['scores'][arm]['by_context']['shared']['final_nll_nats']
    if raw is None or not np.isfinite(float(raw)):
        if rec.get('admissions', {}).get(arm, {}).get('admitted') is False:
            raise KeyError(f'unavailable ineligible score: {arm}')
        raise ValueError(f'nonfinite shared task NLL: root={rec["root"]}, arm={arm}')
    return float(raw)


def recompute_effects(rec: dict) -> dict[str, float | None]:
    f = lambda arm: _nll(rec, arm)
    formulas = {
        'context_gain_E': lambda: f('mixed') - f('shared_E'),
        'context_gain_D': lambda: f('mixed') - f('shared_D'),
        'shared_gap_reduction_D': lambda: f('shared_E') - f('target_E') - f('shared_D') + f('target_D'),
        'target_minus_shared_D': lambda: f('target_D') - f('shared_D'),
        'median_sham_minus_shared_D': lambda: float(np.median([f(f'sham_D_{i}') for i in range(8)])) - f('shared_D'),
        'shared_D_minus_clean': lambda: f('shared_D') - f('B_clean'),
    }
    result = {}
    for name, formula in formulas.items():
        try:
            result[name] = formula()
        except KeyError:
            result[name] = None
    return result


def _reasons(rec: dict, endpoint: str) -> list[str]:
    reasons = list(rec.get('global_failures', []))
    if not rec.get('complete', False):
        reasons.append('root_execution_incomplete')
    for arm in REQUIRED[endpoint]:
        obj = rec.get('admissions', {}).get(arm, {})
        if obj.get('admitted') is not True:
            reasons.append(f'{arm}:ineligible')
            reasons.extend(f'{arm}:{x}' for x in obj.get('reasons', []))
    return reasons


def bootstrap(values: list[float], endpoint_index: int) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (16,) or not np.isfinite(x).all():
        raise ValueError('bootstrap requires all sixteen admitted paired roots')
    seed = 26130013 + 101 * endpoint_index
    rng = np.random.default_rng(seed)
    estimates = x[rng.integers(0, len(x), (BOOTSTRAPS, len(x)))].mean(1)
    tail = ALPHA / (2 * FAMILY)
    interval = np.quantile(estimates, [tail, 1 - tail])
    return {'mean_nats': float(x.mean()), 'nominal_interval': interval.tolist(),
            'nominal_cell_coverage': 1 - ALPHA / FAMILY, 'resamples': BOOTSTRAPS,
            'seed': seed, 'n_independent_model_pairs': 16}


def aggregate(receipts: list[dict]) -> dict:
    seen = {}
    source_locks = set()
    for rec in receipts:
        root = rec.get('root')
        if root not in ROOTS or root in seen:
            raise ValueError(f'unregistered or duplicated development root: {root}')
        if rec.get('study') != 'BPD-NC-13' or rec.get('protocol_commit') != PROTOCOL_COMMIT:
            raise ValueError(f'study/protocol mismatch at root {root}')
        if rec.get('phase') != 'DEVELOPMENT_ENGINEERING':
            raise ValueError(f'incorrect phase at root {root}')
        if rec.get('budgets') != BUDGETS:
            raise ValueError(f'registered budget mismatch at root {root}')
        lock = rec.get('source_lock_sha256', '')
        if not isinstance(lock, str) or not re.fullmatch(r'[0-9a-f]{64}', lock):
            raise ValueError(f'invalid source lock at root {root}')
        source_locks.add(lock)
        seen[root] = rec
    if len(source_locks) > 1:
        raise ValueError('mixed source locks in registered cohort')
    ordered = [seen[r] for r in ROOTS if r in seen]
    effects = {r['root']: recompute_effects(r) for r in ordered}
    endpoints = {}
    for j, name in enumerate(ENDPOINTS):
        rows = []
        for root in ROOTS:
            if root not in seen:
                rows.append({'root': root, 'effect_nats': None, 'admitted': False,
                             'reasons': ['missing_root']})
                continue
            rec = seen[root]
            value = effects[root][name]
            reasons = _reasons(rec, name)
            if value is None:
                reasons.append('unavailable_arm_score')
            rows.append({'root': root, 'effect_nats': value, 'admitted': not reasons,
                         'reasons': reasons})
        all_admitted = all(row['admitted'] for row in rows)
        available = [row['effect_nats'] for row in rows if row['effect_nats'] is not None]
        stats = bootstrap([row['effect_nats'] for row in rows], j) if all_admitted else None
        passed = bool(stats is not None and (
            stats['nominal_interval'][1] <= FLOORS[j] if j == 5
            else stats['nominal_interval'][0] > FLOORS[j]))
        endpoints[name] = {'status': 'ADMITTED' if all_admitted else 'INELIGIBLE',
                           'threshold_nats': FLOORS[j],
                           'direction': 'upper_at_most' if j == 5 else 'lower_exceeds',
                           'passed': passed, 'inference': stats, 'root_effects': rows,
                           'attempted_mean_nats': float(np.mean(available)) if available else None,
                           'n_available_effects': len(available),
                           'n_admitted_pairs': sum(row['admitted'] for row in rows)}
    recovery = all(endpoints[name]['passed'] for name in ENDPOINTS[3:])
    arm_names = sorted({k for r in ordered for k in r.get('scores', {})})
    means = {}
    for arm in arm_names:
        vals = []
        for rec in ordered:
            try:
                vals.append(_nll(rec, arm))
            except KeyError:
                pass
        means[arm] = {'mean_shared_nll': float(np.mean(vals)) if vals else None,
                      'n_scored_roots': len(vals)}
    return {'schema': 'BPD/NC13-original-aggregate/v1', 'study': 'BPD-NC-13',
            'protocol_commit': PROTOCOL_COMMIT, 'scientific_status': 'ENGINEERING_ONLY',
            'source_lock_sha256': next(iter(source_locks), None),
            'verdict': 'ENGINEERING_TRANSFER_RECOVERED' if recovery else 'ENGINEERING_NO_RECOVERY_OR_INELIGIBLE',
            'registered_roots': list(ROOTS), 'observed_roots': list(seen),
            'missing_roots': [r for r in ROOTS if r not in seen], 'comparison_family': FAMILY,
            'endpoints': endpoints, 'mean_shared_nll_by_arm': means,
            'diagnostic_flags': {name: endpoints[name]['passed'] for name in ENDPOINTS[:3]},
            'scope': 'New development pairs. Nominal percentile-bootstrap intervals; no calibrated sham-tail bound or confirmation release.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.receipts.rglob('root_*.json'))
    files = [p for p in files if p.stem.removeprefix('root_').isdigit()]
    receipts = [json.loads(p.read_text()) for p in files]
    result = aggregate(receipts)
    result['receipt_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result['failure_sidecars'] = {
        p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                 'receipt': json.loads(p.read_text())}
        for p in sorted(args.receipts.rglob('root_*.failure.json'))}
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'verdict': result['verdict'], 'n_pairs': len(receipts),
                      'flags': result['diagnostic_flags']}))


if __name__ == '__main__':
    main()
