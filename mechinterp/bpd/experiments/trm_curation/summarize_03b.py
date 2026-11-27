"""CURATE-03B receipts: combine engineering admission runs and summarize the paired pilot."""
from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np

from ..training_steering_stability.stats import _bootstrap, _holm, exact_sign_flip
from .campaign import dump_json, file_sha256
from .engineering_03b import summarize as summarize_engineering
from .memory_cap import DEFAULT_CAP_BYTES
from .pilot import ARMS, AUDIT_PROBABILITY, REPO, source_hashes

PRIMARY = ('D_analytic_mech', 'E_full16_gate', 'F_early12_gate', 'G_joint_full16', 'H_joint_early12')
SECONDARY = (('D_analytic_mech', 'B_transactional'), ('C_analytic_ord', 'D_analytic_mech'),
             ('E_full16_gate', 'D_analytic_mech'), ('F_early12_gate', 'E_full16_gate'),
             ('G_joint_full16', 'E_full16_gate'), ('H_joint_early12', 'F_early12_gate'),
             ('D_analytic_mech', 'I_shuffled'), ('J_observer_only', 'A_uniform_reinvest'))
BASE = 'A_uniform_reinvest'
FLOOR = 0.05


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def combine_engineering(run_dirs, out: Path, *, tests_passed: bool, tests_summary: str) -> dict:
    per_root = {}
    memory = []
    for d in run_dirs:
        summ = json.loads((d / 'engineering_admission.json').read_text(encoding='utf-8'))
        mem_path = d / 'memory_accounting_engineering.json'
        if mem_path.is_file():
            memory.append(json.loads(mem_path.read_text(encoding='utf-8')))
        for root in summ['roots']:
            rd = d / f'root_{root}'
            joint = _jsonl(rd / 'joint_profile.jsonl')
            for w in joint:
                w['rounds'] = {int(k): v for k, v in w['rounds'].items()}
            per_root[root] = {
                'scores': {'rows': _jsonl(rd / 'score_rows.jsonl'), 'windows': _jsonl(rd / 'score_windows.jsonl')},
                'fixed_bd': summ['fixed_update'][str(root)]['B_vs_D'],
                'fixed_aj': summ['fixed_update'][str(root)]['A_vs_J'],
                'joint': {'windows': joint},
                'order': summ['order'][str(root)],
            }
    res = summarize_engineering(per_root)
    peak = max((m.get('peak_job_commit_bytes') or 0) for m in memory) if memory else None
    res['gates']['AG1_tests'] = bool(tests_passed)
    res['gates']['AG7_memory'] = bool(memory) and all(m['enforced'] for m in memory) and peak < DEFAULT_CAP_BYTES
    res['all_gates_pass'] = all(res['gates'].values())
    res['tests'] = tests_summary
    res['memory'] = {'peak_job_commit_bytes': peak, 'cap_bytes': DEFAULT_CAP_BYTES, 'runs': memory}
    res['roots'] = sorted(per_root)
    res['evidence_class'] = 'ENGINEERING_SPENT_ROOTS'
    res['source_sha256'] = source_hashes()
    res['decision'] = 'FRESH_ROSTER_AUTHORIZED' if res['all_gates_pass'] else 'FRESH_ROSTER_NOT_AUTHORIZED'
    dump_json(out, res)
    return res


def _contrast(results, arm, base, key):
    eff = []
    for r in results:
        a = r['arms'][arm]['eval']
        b = r['arms'][base]['eval']
        ca, n = _metric_count(a, key)
        cb, _ = _metric_count(b, key)
        eff.append(Fraction(ca - cb, n))
    sf = exact_sign_flip(eff)
    boot = _bootstrap(eff, 10000, 20261127)
    mean = float(sum(eff) / len(eff))
    return {'arm': arm, 'baseline': base, 'endpoint': key, 'per_root_effect': [float(x) for x in eff],
            'mean_effect': mean, 'positive_roots': sum(x > 0 for x in eff), 'zero_roots': sum(x == 0 for x in eff),
            'negative_roots': sum(x < 0 for x in eff), 'sign_flip': sf, 'bootstrap_95': boot,
            'exceeds_5pp_floor': mean >= FLOOR, '_p': Fraction(sf['p_fraction'])}


def _metric_count(ev, key):
    if key.startswith('d'):
        d = ev['delays'][key[1:]]
        return d['correct_count'], d['n']
    if key == 'bind31':
        d = ev['binding']['31']
        return d['correct_count'], d['n']
    if key == 'transport':
        t = ev['selective_transport']
        return t['target_hits'], t['n_edges']
    raise KeyError(key)


def summarize_pilot(dev_dir: Path, roots, out: Path) -> dict:
    results = [json.loads((dev_dir / f'root_{r}' / 'result.json').read_text(encoding='utf-8')) for r in roots]
    arms = [a for a in ARMS if all(a in r['arms'] for r in results)]
    table = {}
    for a in arms:
        ev = [r['arms'][a]['eval'] for r in results]
        runs = [r['arms'][a] for r in results]
        table[a] = {
            'd31': statistics.mean(e['delays']['31']['accuracy'] for e in ev),
            'd63': statistics.mean(e['delays']['63']['accuracy'] for e in ev),
            'd127': statistics.mean(e['delays']['127']['accuracy'] for e in ev),
            'd2': statistics.mean(e['delays']['2']['accuracy'] for e in ev),
            'bind31': statistics.mean(e['binding']['31']['accuracy'] for e in ev),
            'transport': statistics.mean(e['selective_transport']['transport_accuracy'] for e in ev),
            'protected_corruption': statistics.mean(e['selective_transport']['protected_corruption_rate'] for e in ev),
            'windows': statistics.mean(x['windows'] for x in runs),
            'committed_updates': statistics.mean(x['committed_updates'] for x in runs),
            'abstained_windows': statistics.mean(x['abstained_windows'] for x in runs),
            'rejected_windows': statistics.mean(x['rejected_windows'] for x in runs),
            'candidates_acquired': statistics.mean(x['candidates_acquired'] for x in runs),
            'cpu_seconds': statistics.mean(x['cpu_seconds'] for x in runs),
            'max_overshoot_cpu_seconds': max(x['overshoot_cpu_seconds'] for x in runs),
            'peak_tree_rss_mib': max(x['peak_tree_rss_bytes'] for x in runs) / 2**20,
            'cpu_by_category': _mean_dicts([x['cpu_by_category'] for x in runs]),
            'audit_cpu_inclusive': statistics.mean(x['cpu_inclusive_by_category'].get('audit', 0.0) for x in runs),
            'exposure_resets': statistics.mean(x['exposure_resets'] for x in runs),
        }
    primary = [_contrast(results, a, BASE, 'd31') for a in PRIMARY if a in arms]
    adj = _holm([c['_p'] for c in primary])
    for c, p in zip(primary, adj):
        c['holm_p'] = float(p)
    secondary = [_contrast(results, a, b, 'd31') for a, b in SECONDARY if a in arms and b in arms]
    extra = {k: [_contrast(results, a, BASE, k) for a in PRIMARY if a in arms]
             for k in ('d63', 'd127', 'bind31', 'transport')}
    for group in [primary, secondary] + list(extra.values()):
        for c in group:
            c.pop('_p', None)
    gates = gate_statistics(dev_dir, roots, arms)
    joint = joint_statistics(dev_dir, roots, arms)
    parity = {str(r['root']): r['observer_fixed_update_parity'] for r in results}
    preflight = run_preflight(results)
    summary = {
        'experiment': 'BPD-TRM-CURATE-03B', 'evidence_class': 'DEVELOPMENT_ONLY',
        'roots': list(roots), 'budget_cpu_seconds': results[0]['budget_cpu_seconds'],
        'joint_rounds': results[0]['joint_rounds'], 'arm_means': table,
        'primary_contrasts_d31_vs_A_holm': primary, 'secondary_contrasts_d31': secondary,
        'secondary_endpoints_vs_A': extra, 'gates': gates, 'joint': joint,
        'observer_fixed_update_parity': parity,
        'observer_parity_all_identical': all(p and p['state_identical'] for p in parity.values()),
        'evalmath_preflight': preflight, 'decision': 'REDESIGN_OR_PILOT_ONLY (preserved)',
        'warmup_sha256': {str(r['root']): r['shared_warmup_sha256'] for r in results},
        'result_sha256': {str(r): file_sha256(dev_dir / f'root_{r}' / 'result.json') for r in roots},
        'source_sha256': results[0]['source_sha256'],
    }
    dump_json(out, summary)
    return summary


def _mean_dicts(ds):
    keys = sorted(set().union(*ds))
    return {k: statistics.mean(d.get(k, 0.0) for d in ds) for k in keys}


def gate_statistics(dev_dir, roots, arms) -> dict:
    out = {}
    for a in arms:
        n = abst = audited = would = 0
        audited_pos_gain = 0.0
        acq_stage = {}
        for r in roots:
            for w in _jsonl(dev_dir / f'root_{r}' / f'{a}_windows.jsonl'):
                n += 1
                acq_stage[w.get('acquired', 0)] = acq_stage.get(w.get('acquired', 0), 0) + 1
                if w.get('decision') == 'ABSTAIN':
                    abst += 1
                    if w.get('audit'):
                        audited += 1
                        if w['audit'].get('would_accept'):
                            would += 1
                            audited_pos_gain += max(w['audit']['actual_gain'], 0.0)
        out[a] = {
            'windows': n, 'abstained': abst, 'audited': audited, 'audited_would_accept': would,
            'acquired_count_histogram': {str(k): v for k, v in sorted(acq_stage.items())},
            'ht_estimate_missed_accepts': would / AUDIT_PROBABILITY,
            'ht_estimate_missed_positive_gain': audited_pos_gain / AUDIT_PROBABILITY,
            'unaudited_abstentions_missing': abst - audited,
        }
    return out


def joint_statistics(dev_dir, roots, arms) -> dict:
    out = {}
    for a in arms:
        rows = [w for r in roots for w in _jsonl(dev_dir / f'root_{r}' / f'{a}_windows.jsonl')
                if w.get('parity') is not None]
        if not rows:
            continue
        att = [w for w in rows if w.get('actual_gain') is not None]
        d = {'verified_windows': len(rows), 'parity_ok_fraction': float(np.mean([w['parity']['ok'] for w in rows])),
             'max_rel_l2': max(w['parity']['rel_l2_error'] for w in rows)}
        if att:
            pred = [w['predicted_joint_gain_native'] for w in att]
            act = [w['actual_gain'] for w in att]
            d['mean_predicted_joint_gain'] = float(np.mean(pred))
            d['mean_actual_gain'] = float(np.mean(act))
            d['accepted_fraction'] = float(np.mean([w['accepted'] for w in att]))
            if np.std(pred) > 0 and np.std(act) > 0:
                from scipy.stats import spearmanr
                d['spearman_predicted_actual'] = float(spearmanr(pred, act).statistic)
        sw = [w for w in rows if 'swaps' in w]
        if sw:
            d['swap_windows_fraction'] = float(np.mean([bool(w['swaps']) for w in sw]))
            d['mean_predicted_improvement'] = float(np.mean([w['predicted_joint_gain_final'] - w['predicted_joint_gain_initial'] for w in sw]))
        out[a] = d
    return out


def run_preflight(results) -> dict:
    n = results[0]['arms'][BASE]['eval']['delays']['31']['n']
    rows = []
    for r in results:
        ca = r['arms'][BASE]['eval']['delays']['31']['correct_count']
        cg = r['arms']['G_joint_full16']['eval']['delays']['31']['correct_count']
        ci = r['arms']['I_shuffled']['eval']['delays']['31']['correct_count']
        rows.append({'unit_id': str(r['root']), 'paired_effect': f'{cg - ca}/{n}', 'sham_effects': [f'{ci - ca}/{n}']})
    spec = {'schema_version': 1, 'experiment_id': 'BPD-TRM-CURATE-03B-development',
            'independent_unit': 'independently initialized TRM model root with paired curation descendants',
            'alpha': '0.05', 'scientific_min_effect': '0.05', 'target_effect': '0.10',
            'measurement_resolution': f'1/{n}', 'sham_quantile': '0.95', 'sham_confidence': '0.95',
            'planned_comparisons': len(PRIMARY), 'confirm_units_planned': len(results), 'pilot': rows,
            'pilot_from_development_only': True, 'heldout_sealed': False, 'units_independent': True}
    pf = REPO / 'mechinterp/bpd/experiments/repair_portfolio/evidence/preflight.py'
    spec_ = importlib.util.spec_from_file_location('c03b_preflight', pf)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return {'input': spec, 'result': mod.preflight(spec), 'preflight_sha256': file_sha256(pf)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('engineering')
    e.add_argument('--runs', type=Path, nargs='+', required=True)
    e.add_argument('--out', type=Path, required=True)
    e.add_argument('--tests-passed', action='store_true')
    e.add_argument('--tests-summary', default='')
    p = sub.add_parser('pilot')
    p.add_argument('--dev', type=Path, required=True)
    p.add_argument('--roots', type=int, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == 'engineering':
        r = combine_engineering(a.runs, a.out, tests_passed=a.tests_passed, tests_summary=a.tests_summary)
        print(json.dumps({'gates': r['gates'], 'decision': r['decision']}, indent=1))
    else:
        s = summarize_pilot(a.dev, a.roots, a.out)
        print(json.dumps({k: s['arm_means'][k]['d31'] for k in s['arm_means']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
