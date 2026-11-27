"""CURATE-03B DataGoblin training export V2 (TRAIN-only, explicit state/action/outcome).

Three label kinds stay in separate fields and are never mixed:
  individual  per-candidate predicted gain (analytic or transactional) and, only
              where measured, the transactional realized one-step gain
  joint       predicted joint minibatch gain (aggregated and native) and the
              committed or audited minibatch's actual feedback gain
  horizon     the arm's end-of-budget EVAL accuracy, as a separate file keyed
              by root/arm, never attached to a decision row as a label

Missing counterfactuals are null with an explicit ``label_status``. Rows carry
the action vocabulary ACQUIRE_4/8/12/16, ABSTAIN, SELECT_BATCH, SWAP_EDGE,
VERIFY_BATCH, REINVEST_CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .campaign import dump_json, file_sha256

SCHEMA = 'bpd-trm-curate03b-datagoblin-v2'
ACTIONS = ('ACQUIRE_4', 'ACQUIRE_8', 'ACQUIRE_12', 'ACQUIRE_16', 'ABSTAIN', 'SELECT_BATCH',
           'SWAP_EDGE', 'VERIFY_BATCH', 'REINVEST_CPU')


def _jsonl(path: Path):
    from .summarize_03b import read_jsonl
    return read_jsonl(path)


def _actions(w: dict, arm_spec: dict) -> list[str]:
    if w.get('decision') == 'REINVEST_CPU':
        return ['REINVEST_CPU']
    acts = [f'ACQUIRE_{s["stage"]}' for s in w.get('stages', [])]
    if w.get('decision') == 'ABSTAIN':
        return acts + ['ABSTAIN']
    acts.append('SELECT_BATCH')
    acts += ['SWAP_EDGE'] * len(w.get('swaps', []))
    if arm_spec.get('accept'):
        acts.append('VERIFY_BATCH')
    return acts


def build(dev_dir: Path, roots, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / 'trajectories_v2.jsonl'
    stage_path = out_dir / 'acquisition_stages_v2.jsonl'
    horizon_path = out_dir / 'horizon_outcomes_v2.jsonl'
    counts = {'windows': 0, 'stage_decisions': 0, 'candidates': 0, 'measured_individual_labels': 0,
              'audited_abstentions': 0, 'unaudited_abstentions': 0}
    action_counts = {a: 0 for a in ACTIONS}
    with traj_path.open('w', encoding='utf-8', newline='\n') as ft, \
            stage_path.open('w', encoding='utf-8', newline='\n') as fs, \
            horizon_path.open('w', encoding='utf-8', newline='\n') as fh:
        for root in roots:
            rdir = dev_dir / f'root_{root}'
            res = json.loads((rdir / 'result.json').read_text(encoding='utf-8'))
            for arm, summ in res['arms'].items():
                fh.write(_dump({'root': root, 'arm': arm, 'evidence_class': 'DEVELOPMENT_ONLY',
                                'label_kind': 'horizon', 'split_evaluated': 'eval',
                                'decision_rows_attached': False,
                                'committed_updates': summ['committed_updates'], 'windows': summ['windows'],
                                'cpu_seconds': summ['cpu_seconds'],
                                'eval_d31_accuracy': summ['eval']['delays']['31']['accuracy'],
                                'eval_d63_accuracy': summ['eval']['delays']['63']['accuracy'],
                                'eval_d127_accuracy': summ['eval']['delays']['127']['accuracy'],
                                'eval_bind31_accuracy': summ['eval']['binding']['31']['accuracy'],
                                'transport_accuracy': summ['eval']['selective_transport']['transport_accuracy'],
                                'protected_corruption_rate': summ['eval']['selective_transport']['protected_corruption_rate']}))
                windows = _jsonl(rdir / f'{arm}_windows.jsonl')
                cands = {}
                for c in _jsonl(rdir / f'{arm}_candidates.jsonl'):
                    cands.setdefault(c['window'], []).append(c)
                spec = summ['spec']
                for w in windows:
                    pool = sorted(cands.get(w['window'], []), key=lambda c: c['acquisition_position'])
                    acts = _actions(w, spec)
                    for a in acts:
                        action_counts[a] += 1
                    abstain = w.get('decision') == 'ABSTAIN'
                    audit = w.get('audit')
                    if abstain:
                        counts['audited_abstentions' if audit else 'unaudited_abstentions'] += 1
                    individual = []
                    for c in pool:
                        status = c.get('label_source') or 'not_acquired'
                        if c.get('realized_gain') is not None:
                            counts['measured_individual_labels'] += 1
                        individual.append({
                            'edge': c['edge'], 'slot': c['slot'], 'acquisition_position': c['acquisition_position'],
                            'stage': c['stage'], 'acquired': c['acquired'],
                            'acquired_for_audit': c.get('acquired_for_audit', False),
                            'features_after_acquisition': None if not (c['acquired'] or c.get('acquired_for_audit')) else {
                                'predicted_gain': c.get('predicted_gain'), 'update_norm': c.get('update_norm'),
                                'gradient_norm': c.get('gradient_norm'), 'clip_scale': c.get('clip_scale'),
                                'train_loss': c.get('train_loss')},
                            'realized_individual_gain': c.get('realized_gain'),
                            'individual_label_status': 'measured_transactional' if c.get('realized_gain') is not None
                            else ('missing_unaudited' if c['acquired'] or c.get('acquired_for_audit') else 'not_acquired'),
                            'in_final_batch': c.get('in_final_batch', False),
                        })
                        counts['candidates'] += 1
                    joint = None
                    if w.get('attempted') and w.get('decision') != 'REINVEST_CPU':
                        joint = {'status': 'committed' if w.get('committed') else 'rolled_back',
                                 'predicted_initial': w.get('predicted_joint_gain_initial'),
                                 'predicted_final': w.get('predicted_joint_gain_final'),
                                 'predicted_native': w.get('predicted_joint_gain_native'),
                                 'predicted_aggregated': w.get('predicted_joint_gain_aggregated'),
                                 'actual_gain': w.get('actual_gain'),
                                 'cal_before': w.get('cal_before'), 'cal_after': w.get('cal_after'),
                                 'parity': w.get('parity')}
                    elif abstain:
                        joint = {'status': 'audited_counterfactual' if audit else 'missing_unaudited',
                                 'predicted_native': audit.get('predicted_joint_gain_native') if audit else None,
                                 'actual_gain': audit.get('actual_gain') if audit else None,
                                 'would_accept': audit.get('would_accept') if audit else None,
                                 'audit_probability': w.get('audit_probability'),
                                 'audit_draw_u': w.get('audit_draw_u')}
                    row = {
                        'schema': SCHEMA, 'split': 'train', 'training_eligible': True, 'evaluation_leakage': False,
                        'evidence_class': 'DEVELOPMENT_ONLY',
                        'record_id': hashlib.sha256(repr((root, arm, w['window'])).encode()).hexdigest()[:24],
                        'provenance': {'root': root, 'arm': arm, 'window': w['window'],
                                       'parent_checkpoint_sha256': w.get('checkpoint_sha256'),
                                       'optimizer_state_digest': w.get('optimizer_digest'),
                                       'shared_warmup_sha256': res['shared_warmup_sha256'],
                                       'source_sha256': res['source_sha256']},
                        'state': {'delay': w['delay'], 'stages': w.get('stages'),
                                  'feedback': w.get('feedback'),
                                  'cumulative_cpu_seconds_before': w['cumulative_cpu_seconds'] - w['cpu_seconds'],
                                  'selection_exposure_total': w.get('selection_exposure_total')},
                        'candidate_pool': individual,
                        'acquisition_order': w['acquisition_order'],
                        'actions': acts,
                        'decision': w.get('decision'),
                        'selection': {'initial': w.get('initial_selected'), 'reasons': w.get('selection_reasons'),
                                      'final': w.get('selected'), 'swaps': w.get('swaps'),
                                      'scored_batches': w.get('scored_batches')},
                        'joint_outcome': joint,
                        'committed': bool(w.get('committed')),
                        'cost': {'window_cpu_seconds': w['cpu_seconds'], 'tree_rss_bytes': w.get('tree_rss_bytes')},
                    }
                    ft.write(_dump(row))
                    counts['windows'] += 1
                    # Stage-level decision points with features available before the next acquisition.
                    prefix = [c for c in pool if c['acquired']]
                    for st in w.get('stages', []):
                        k = st['stage']
                        seen = [c.get('predicted_gain') for c in prefix[:k]]
                        if len(seen) < k:
                            break
                        fs.write(_dump({
                            'schema': SCHEMA, 'split': 'train', 'root': root, 'arm': arm, 'window': w['window'],
                            'stage': k, 'features_before_next_acquisition': {
                                'max_score': max(seen), 'n_positive': sum(s > 0 for s in seen),
                                'sum_positive': sum(max(s, 0) for s in seen), 'mean_score': sum(seen) / k,
                                'max_update_norm': max(c.get('update_norm') or 0 for c in prefix[:k]),
                                'delay': w['delay'], 'feedback_loss0': (w.get('feedback') or {}).get('loss0'),
                                'window_fraction_of_budget': (w['cumulative_cpu_seconds'] - w['cpu_seconds'])
                                / max(res['budget_cpu_seconds'], 1e-9)},
                            'policy_action': (f'ACQUIRE_{k + 4}' if k < w['stages'][-1]['stage']
                                              else ('ABSTAIN' if abstain else 'SELECT_BATCH')),
                            'window_attempted': bool(w.get('attempted')),
                            'window_committed': bool(w.get('committed')),
                            'window_actual_gain': (w.get('actual_gain') if w.get('attempted') else
                                                   (audit.get('actual_gain') if audit else None)),
                            'window_outcome_status': ('attempted' if w.get('attempted') else
                                                      ('audited' if audit else 'missing_unaudited')),
                        }))
                        counts['stage_decisions'] += 1
    unit = unit_costs(dev_dir, roots)
    dump_json(out_dir / 'unit_costs.json', unit)
    manifest = {
        'schema': SCHEMA, 'evidence_class': 'DEVELOPMENT_ONLY', 'split': 'train', 'roots': list(roots),
        'unit_costs_sha256': file_sha256(out_dir / 'unit_costs.json'),
        'action_vocabulary': list(ACTIONS), 'action_counts': action_counts, 'counts': counts,
        'files': {p.name: {'sha256': file_sha256(p), 'bytes': p.stat().st_size}
                  for p in (traj_path, stage_path, horizon_path)},
        'label_kinds': {
            'individual': 'per-candidate predicted gain; realized gain only for B_transactional probes',
            'joint': 'predicted joint minibatch gain and committed/rolled-back/audited actual feedback gain',
            'horizon': 'end-of-budget EVAL accuracy per root/arm in horizon_outcomes_v2.jsonl, never a decision label'},
        'missingness': 'null + status; unaudited abstentions and unacquired candidates are missing, never zero',
        'controller_model': 'Q(s,a) = E[dU + gamma V(s\') | s,a] - lambda C(s,a), resource-bounded sequential acquisition',
    }
    dump_json(out_dir / 'manifest.json', manifest)
    return manifest


def unit_costs(dev_dir: Path, roots, arm: str = 'D_analytic_mech') -> dict:
    """Measured mean CPU per candidate, per attempt and per window base, from the arm's ledger."""
    cand = att = base = 0.0
    n_c = n_a = n_w = 0
    for r in roots:
        s = json.loads((dev_dir / f'root_{r}' / 'result.json').read_text(encoding='utf-8'))['arms'][arm]
        c = s['cpu_by_category']
        cand += c.get('candidate_gradient', 0) + c.get('analytic_scoring', 0)
        att += c.get('acceptance_step', 0) + c.get('feedback_eval', 0) + c.get('verification', 0)
        base += s['cpu_seconds'] - (c.get('candidate_gradient', 0) + c.get('analytic_scoring', 0)
                                    + c.get('acceptance_step', 0) + c.get('feedback_eval', 0) + c.get('verification', 0))
        n_c += s['candidates_acquired']
        n_a += s['windows'] - s['abstained_windows']
        n_w += s['windows']
    return {'arm': arm, 'roots': list(roots), 'per_candidate': cand / max(n_c, 1),
            'per_attempt': att / max(n_a, 1), 'per_window_base': base / max(n_w, 1),
            'definition': 'per_candidate = candidate_gradient + analytic_scoring; per_attempt = acceptance_step + '
                          'feedback_eval + verification; base = everything else in the arm CPU (feedback gradient, '
                          'proposal, materialization, controller, provenance, custody, unattributed)'}


def _dump(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dev', type=Path, required=True)
    ap.add_argument('--roots', type=int, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    m = build(a.dev, a.roots, a.out)
    print(json.dumps({'counts': m['counts'], 'actions': m['action_counts']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
