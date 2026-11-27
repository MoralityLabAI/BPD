"""CURATE-03B native engineering admission on spent roots 28000/28001.

Measurements (CURATE_03B_PILOT_PROTOCOL.md, "Engineering admission"):
  scores   analytical vs original transactional probe at identical parents:
           cold/warm optimizer x ordinary/mechanistic x clip {1.0, 0.05, 1e6}
  fixed    fixed-update B (transactional) vs D (analytical) selections/checkpoints
  observer fixed-update A (uniform) vs J (observer-only) checkpoint identity
  joint    aggregated vs native minibatch gradient; 0/1/2 search rounds with
           real (isolated) acceptance outcomes for every distinct batch
  order    prospective acquisition order: determinism and gate replay on the
           full-16 D trajectory
Writes engineering_admission.json with every gate, pass or fail.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import time

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from .batch_screen import mean_gradient
from .campaign import CampaignRecipe, dump_json, _sample_window_proposals
from .memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap
from .native_screen import (
    STAGES, NativeScreen, acquisition_order, edge_xy, exposure_legal, gate_decision,
    score_agreement,
)
from .pilot import ArmRunner, HERE, prepare_root, source_hashes
from .scheduler import Candidate, pair_key, select_candidates
from .signal_bank import cached_edge_batch, seed_for
from .transaction import probe_update

CLIPS = (1.0, 0.05, 1e6)
OBJECTIVES = ('ordinary', 'mechanistic')
SCORE_WINDOWS = 3
WARM_STEPS = 30
FIXED_WINDOWS = 40
JOINT_WINDOWS = 30

GATES = {
    'AG2_observer_parity': 'fixed-update A vs J final state identical on both roots',
    'AG3_score_agreement': '>=99% of candidate comparisons within SCORE_TOLERANCE and top-4 '
                           'ranked set identical in >=95% of comparison windows',
    'AG4_fixed_update_selection': 'B vs D per-window selections identical in >=95% of windows on both roots',
    'AG5_joint_parity': '100% of verified windows within JOINT_TOLERANCE',
    'AG6_cpu': 'median analytical per-candidate CPU <= 0.5 x median transactional probe CPU',
    'AG8_order_determinism': 'acquisition order identical on recomputation for every window',
}


def _cands(prep, recipe, root, win, model_screen, cache=None):
    delay = recipe.train_delays[win % len(recipe.train_delays)]
    props = _sample_window_proposals(prep['fit_edges'], root, win, recipe.proposals_per_window,
                                     Counter(), recipe.pair_exposure_cap)
    order = acquisition_order(props, split_seed=recipe.split_seed, root=root, window=win)
    nv = win % 4
    import hashlib
    batches = {e: cached_edge_batch(
        (e,), root=root, seed=seed_for(root, f'cand_v{nv}_d{delay}',
                                       int.from_bytes(hashlib.sha256(repr(e).encode()).digest()[:4], 'little')),
        delay=delay, split='train', noise=recipe.noise, cache=cache) for e in order}
    return order, batches


def score_comparison(root, prep, recipe) -> dict:
    rows, windows, custody = [], [], []
    for state in ('cold', 'warm'):
        base = copy.deepcopy(prep['warm'])
        base_opt = torch.optim.AdamW(base.parameters(), lr=recipe.curation_learning_rate,
                                     weight_decay=recipe.weight_decay)
        if state == 'warm':
            rn = ArmRunner('A_uniform_reinvest', root, prep['warm'], recipe, prep['projectors'],
                           prep['fb_edges'], prep['fit_edges'], mms_mib=0, budget_cpu=None,
                           fixed_windows=WARM_STEPS, record_rows=False)
            rn.run()
            base, base_opt = rn.model, rn.opt
        fb_batch = cached_edge_batch(prep['fb_edges'], root=root,
                                     seed=seed_for(root, f'fb_cohort_d{recipe.feedback_delay}', 0),
                                     delay=recipe.feedback_delay, split='train', noise=recipe.noise)
        for clip in CLIPS:
            rcp = replace(recipe, gradient_clip=clip)
            for objective in OBJECTIVES:
                screen = NativeScreen(base, base_opt, rcp, prep['projectors'])
                fb = screen.feedback_gradient(fb_batch, objective)
                cal_norm = sum(float(g.double().square().sum()) for g in fb.grads if g is not None) ** .5
                for win in range(1000, 1000 + SCORE_WINDOWS):
                    order, batches = _cands(prep, recipe, root, win, screen)
                    a_scores, t_scores, realized = [], [], []
                    for e in order:
                        c0 = time.process_time()
                        g, _ = screen.candidate_gradient(batches[e])
                        a = screen.analytic_score(g, fb)
                        a_cpu = time.process_time() - c0
                        b = batches[e]
                        c0 = time.process_time()
                        p = probe_update(base, base_opt, lambda b=b: F.cross_entropy(base(edge_xy(b)[0]), edge_xy(b)[1]),
                                         lambda: screen.objective_loss(fb_batch, objective), clip=clip,
                                         cached_cal_grad=fb.grads, cached_cal_loss0=fb.loss0)
                        t_cpu = time.process_time() - c0
                        ag = score_agreement(a['predicted_gain'], p['predicted_gain'], cal_norm, p['update_norm'])
                        rows.append({'root': root, 'state': state, 'clip': clip, 'objective': objective,
                                     'window': win, 'analytic': a['predicted_gain'],
                                     'transactional': p['predicted_gain'], 'realized': p['realized_gain'],
                                     'abs_error': ag['abs_error'], 'ok': ag['ok'], 'clip_scale': a['clip_scale'],
                                     'analytic_cpu': a_cpu, 'transactional_cpu': t_cpu})
                        a_scores.append(a['predicted_gain'])
                        t_scores.append(p['predicted_gain'])
                        realized.append(p['realized_gain'])
                    top_a = set(np.argsort(-np.array(a_scores), kind='stable')[:4].tolist())
                    top_t = set(np.argsort(-np.array(t_scores), kind='stable')[:4].tolist())
                    windows.append({
                        'root': root, 'state': state, 'clip': clip, 'objective': objective, 'window': win,
                        'feedback_cpu': fb.cpu_seconds,
                        'spearman_analytic_transactional': float(spearmanr(a_scores, t_scores).statistic),
                        'spearman_analytic_realized': float(spearmanr(a_scores, realized).statistic),
                        'top4_identical': top_a == top_t,
                        'ties_analytic': len(a_scores) - len(set(a_scores)),
                        'ties_transactional': len(t_scores) - len(set(t_scores)),
                        'sign_agreement_realized': float(np.mean((np.array(a_scores) > 0) == (np.array(realized) > 0))),
                        'positive_analytic': int(sum(s > 0 for s in a_scores)),
                    })
                custody.append({'state': state, 'clip': clip, 'objective': objective,
                                **dict(screen.custody.stats)})
    return {'rows': rows, 'windows': windows, 'analytic_custody_stats': custody}


def fixed_update_selection(root, prep, recipe, arm_a, arm_b, windows) -> dict:
    out = {}
    for arm in (arm_a, arm_b):
        rn = ArmRunner(arm, root, prep['warm'], recipe, prep['projectors'], prep['fb_edges'],
                       prep['fit_edges'], mms_mib=16, budget_cpu=None, fixed_windows=windows)
        s = rn.run()
        out[arm] = {'summary': {k: v for k, v in s.items() if k not in ('spec',)},
                    'windows': rn.windows, 'rows': rn.rows}
    sa = [w.get('selected') for w in out[arm_a]['windows']]
    sb = [w.get('selected') for w in out[arm_b]['windows']]
    first_div = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), None)
    return {
        'arms': [arm_a, arm_b], 'windows': windows,
        'selections_identical_fraction': float(np.mean([x == y for x, y in zip(sa, sb)])),
        'first_divergent_window': first_div,
        'state_identical': out[arm_a]['summary']['final_state_sha256'] == out[arm_b]['summary']['final_state_sha256'],
        'cpu_seconds': {a: out[a]['summary']['cpu_seconds'] for a in out},
        'cpu_by_category': {a: out[a]['summary']['cpu_by_category'] for a in out},
        'committed': {a: out[a]['summary']['committed_updates'] for a in out},
        '_detail': out,
    }


def joint_profile(root, prep, recipe) -> dict:
    """D-like trajectory; at every window score 0/1/2 rounds and realize each distinct batch."""
    rn = ArmRunner('G_joint_full16', root, prep['warm'], recipe, prep['projectors'], prep['fb_edges'],
                   prep['fit_edges'], mms_mib=16, budget_cpu=None, fixed_windows=1, record_rows=False)
    screen, model, opt = rn.screen, rn.model, rn.opt
    out = []
    exposures = Counter()
    fb_batch = cached_edge_batch(prep['fb_edges'], root=root,
                                 seed=seed_for(root, f'fb_cohort_d{recipe.feedback_delay}', 0),
                                 delay=recipe.feedback_delay, split='train', noise=recipe.noise)
    for win in range(JOINT_WINDOWS):
        order, batches = _cands(prep, recipe, root, win, screen)
        fb = screen.feedback_gradient(fb_batch, 'mechanistic')
        c0 = time.process_time()
        grads, scores = {}, {}
        for e in order:
            grads[e], _ = screen.candidate_gradient(batches[e])
            scores[e] = screen.analytic_score(grads[e], fb)['predicted_gain']
        acq_cpu = time.process_time() - c0
        epoch = f'jp{win}'
        cands = [Candidate(e, 1.0, scores[e], 1e-3, epoch) for e in order]
        sel, _ = select_candidates(cands, fit_edges=prep['fit_edges'], epoch=epoch, mode='mechanistic',
                                   count=4, seed=seed_for(root, 'sched', 0), window=win, exposures=exposures,
                                   cap=recipe.pair_exposure_cap, exploration=recipe.exploration)
        idx = {e: i for i, e in enumerate(order)}
        init = [idx[e] for e, _ in sel]
        pins = [idx[e] for e, why in sel if why == 'explore']
        legal_e = exposure_legal(exposures, recipe.pair_exposure_cap)
        res = {}
        for rounds in (0, 1, 2):
            c0 = time.process_time()
            r = screen.joint_search([grads[e] for e in order], fb, init, pinned_idx=pins,
                                    legal=lambda b: legal_e([order[i] for i in b]), rounds=rounds)
            res[rounds] = dict(r, search_cpu=time.process_time() - c0)
        realized = {}
        for rounds, r in res.items():
            key = r['batch']
            if key not in realized:
                edges = [order[i] for i in key]
                agg = mean_gradient([grads[e] for e in edges])
                info = screen.step_minibatch([batches[e] for e in edges], fb=fb, fb_batch=fb_batch,
                                             commit_allowed=False, aggregated=agg)
                realized[key] = info
        rows = {rounds: {'batch': list(r['batch']), 'predicted': r['predicted_gain'], 'scored_batches': r['scored_batches'],
                         'search_cpu': r['search_cpu'], 'actual_gain': realized[r['batch']]['actual_gain'],
                         'would_accept': realized[r['batch']]['would_accept'],
                         'parity_ok': realized[r['batch']]['parity']['ok'],
                         'parity_rel_l2': realized[r['batch']]['parity']['rel_l2_error'],
                         'predicted_native': realized[r['batch']]['predicted_joint_gain_native']}
                for rounds, r in res.items()}
        out.append({'window': win, 'acquisition_cpu': acq_cpu, 'rounds': rows})
        # Advance along the one-round trajectory (commit only if accepted, as arm G would).
        edges = [order[i] for i in res[1]['batch']]
        info = screen.step_minibatch([batches[e] for e in edges], fb=fb, fb_batch=fb_batch,
                                     aggregated=mean_gradient([grads[e] for e in edges]))
        exposures += Counter(pair_key(e) for e in edges)
        if sum(1 for e in prep['fit_edges'] if exposures[pair_key(e)] < recipe.pair_exposure_cap) < 16:
            exposures = Counter()
    return {'windows': out}


def order_audit(fixed_d_windows, fixed_d_rows, root, recipe) -> dict:
    """Prospective order determinism + gate replay on the full-16 analytical trajectory."""
    deterministic = True
    replay = {g: Counter() for g in ('always', 'full16', 'early4', 'early8', 'early12')}
    by_win = {}
    for row in fixed_d_rows:
        by_win.setdefault(row['window'], []).append(row)
    for w in fixed_d_windows:
        order = [tuple(map(tuple, e[:2])) + (e[2],) for e in w['acquisition_order']]
        if tuple(order) != acquisition_order(order[::-1], split_seed=recipe.split_seed, root=root,
                                             window=w['window']):
            deterministic = False
        rows = sorted(by_win[w['window']], key=lambda r: r['acquisition_position'])
        s = [r['predicted_gain'] for r in rows]
        gain = w.get('actual_gain', 0.0)
        acc = bool(w.get('committed'))
        for name, k in (('always', None), ('full16', 16), ('early4', 4), ('early8', 8), ('early12', 12)):
            if name == 'always':
                acquired, attempt = 16, True
            elif not any(x > 0 for x in s[:k]):
                acquired, attempt = k, False
            else:
                acquired, attempt = 16, any(x > 0 for x in s)
            c = replay[name]
            c['windows'] += 1
            c['acquired'] += acquired
            c['attempts'] += int(attempt)
            c['accepted'] += int(acc)
            c['accepted_missed'] += int(acc and not attempt)
            c['positive_gain'] += max(gain, 0) if acc else 0
            c['missed_positive_gain'] += max(gain, 0) if (acc and not attempt) else 0
    out = {}
    for k, c in replay.items():
        d = dict(c)
        d['retained_fraction'] = 1 - d['missed_positive_gain'] / d['positive_gain'] if d['positive_gain'] else 1.0
        out[k] = d
    return {'deterministic': deterministic, 'gate_replay': out,
            'scope': 'replay of the recorded full-16 analytical trajectory; skipped windows would have changed later checkpoints'}


def run(out_dir: Path, recipe: CampaignRecipe, roots) -> dict:
    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_root = {}
    for root in roots:
        wp = HERE / 'results' / 'engineering' / 'phase_a' / f'root_{root}' / 'TRM_w32_r29.pt'
        prep = prepare_root(root, recipe, out_dir / f'root_{root}', warm_path=wp)
        t0 = time.process_time()
        sc = score_comparison(root, prep, recipe)
        print(json.dumps({'root': root, 'stage': 'scores', 'cpu': round(time.process_time() - t0, 1)}), flush=True)
        bd = fixed_update_selection(root, prep, recipe, 'B_transactional', 'D_analytic_mech', FIXED_WINDOWS)
        print(json.dumps({'root': root, 'stage': 'fixed_bd', 'sel': bd['selections_identical_fraction']}), flush=True)
        aj = fixed_update_selection(root, prep, recipe, 'A_uniform_reinvest', 'J_observer_only', FIXED_WINDOWS)
        print(json.dumps({'root': root, 'stage': 'fixed_aj', 'state_identical': aj['state_identical']}), flush=True)
        jp = joint_profile(root, prep, recipe)
        print(json.dumps({'root': root, 'stage': 'joint'}), flush=True)
        d_detail = bd['_detail']['D_analytic_mech']
        oa = order_audit(d_detail['windows'], d_detail['rows'], root, recipe)
        rdir = out_dir / f'root_{root}'
        _jsonl(rdir / 'score_rows.jsonl', sc['rows'])
        _jsonl(rdir / 'score_windows.jsonl', sc['windows'])
        _jsonl(rdir / 'joint_profile.jsonl', jp['windows'])
        for tag, res in (('fixed_bd', bd), ('fixed_aj', aj)):
            for arm, det in res['_detail'].items():
                _jsonl(rdir / f'{tag}_{arm}_windows.jsonl', det['windows'])
                _jsonl(rdir / f'{tag}_{arm}_candidates.jsonl', det['rows'])
        per_root[root] = {'scores': sc, 'fixed_bd': _strip(bd), 'fixed_aj': _strip(aj), 'joint': jp,
                          'order': oa, 'warm_sha256': prep['warm_sha256']}
    summary = summarize(per_root)
    summary['source_sha256'] = source_hashes()
    summary['roots'] = list(roots)
    summary['evidence_class'] = 'ENGINEERING_SPENT_ROOTS'
    dump_json(out_dir / 'engineering_admission.json', summary)
    return summary


def summarize(per_root) -> dict:
    rows = [r for pr in per_root.values() for r in pr['scores']['rows']]
    wins = [w for pr in per_root.values() for w in pr['scores']['windows']]
    ok_frac = float(np.mean([r['ok'] for r in rows]))
    top4 = float(np.mean([w['top4_identical'] for w in wins]))
    a_cpu = statistics.median(r['analytic_cpu'] for r in rows)
    t_cpu = statistics.median(r['transactional_cpu'] for r in rows)
    strata = {}
    for r in rows:
        k = f"{r['state']}|{r['objective']}|clip={r['clip']}"
        strata.setdefault(k, []).append(r)
    strata_out = {k: {'n': len(v), 'within_tolerance': float(np.mean([x['ok'] for x in v])),
                      'max_abs_error': max(x['abs_error'] for x in v),
                      'spearman_analytic_realized': float(spearmanr([x['analytic'] for x in v],
                                                                    [x['realized'] for x in v]).statistic),
                      'sign_agreement_realized': float(np.mean([(x['analytic'] > 0) == (x['realized'] > 0) for x in v])),
                      'mean_abs_realized_minus_predicted': float(np.mean([abs(x['realized'] - x['analytic']) for x in v])),
                      'clip_active_fraction': float(np.mean([x['clip_scale'] < 1 for x in v]))}
                  for k, v in sorted(strata.items())}
    jw = [w for pr in per_root.values() for w in pr['joint']['windows']]
    parity_all = all(w['rounds'][k]['parity_ok'] for w in jw for k in w['rounds'])
    imp1 = float(np.mean([w['rounds'][1]['predicted'] - w['rounds'][0]['predicted'] for w in jw]))
    imp2 = float(np.mean([w['rounds'][2]['predicted'] - w['rounds'][1]['predicted'] for w in jw]))
    s2_cpu = float(np.mean([w['rounds'][2]['search_cpu'] for w in jw]))
    acq_cpu = float(np.mean([w['acquisition_cpu'] for w in jw]))
    two_rounds = imp2 >= 0.1 * imp1 and s2_cpu <= 0.25 * acq_cpu
    joint = {
        'windows': len(jw),
        'mean_predicted_improvement_round1_over_0': imp1,
        'mean_predicted_improvement_round2_over_1': imp2,
        'mean_actual_gain': {k: float(np.mean([w['rounds'][k]['actual_gain'] for w in jw])) for k in (0, 1, 2)},
        'would_accept_fraction': {k: float(np.mean([w['rounds'][k]['would_accept'] for w in jw])) for k in (0, 1, 2)},
        'mean_scored_batches': {k: float(np.mean([w['rounds'][k]['scored_batches'] for w in jw])) for k in (0, 1, 2)},
        'mean_search_cpu': {k: float(np.mean([w['rounds'][k]['search_cpu'] for w in jw])) for k in (0, 1, 2)},
        'mean_acquisition_cpu': acq_cpu,
        'max_parity_rel_l2': max(w['rounds'][k]['parity_rel_l2'] for w in jw for k in w['rounds']),
        'round1_changed_batch_fraction': float(np.mean([w['rounds'][1]['batch'] != w['rounds'][0]['batch'] for w in jw])),
        'round2_changed_batch_fraction': float(np.mean([w['rounds'][2]['batch'] != w['rounds'][1]['batch'] for w in jw])),
        'spearman_predicted_actual_round1': float(spearmanr([w['rounds'][1]['predicted_native'] for w in jw],
                                                            [w['rounds'][1]['actual_gain'] for w in jw]).statistic),
        'decision_rounds': 2 if two_rounds else 1,
        'decision_rule': 'two rounds iff mean round-2 predicted improvement >= 0.1 x round-1 improvement '
                         'and mean round-2 search CPU <= 0.25 x mean candidate acquisition CPU',
    }
    gates = {
        'AG2_observer_parity': all(pr['fixed_aj']['state_identical'] for pr in per_root.values()),
        'AG3_score_agreement': ok_frac >= 0.99 and top4 >= 0.95,
        'AG4_fixed_update_selection': all(pr['fixed_bd']['selections_identical_fraction'] >= 0.95
                                          for pr in per_root.values()),
        'AG5_joint_parity': parity_all,
        'AG6_cpu': a_cpu <= 0.5 * t_cpu,
        'AG8_order_determinism': all(pr['order']['deterministic'] for pr in per_root.values()),
    }
    return {
        'gates': gates, 'gate_definitions': GATES, 'all_measured_gates_pass': all(gates.values()),
        'score_agreement': {'n': len(rows), 'within_tolerance_fraction': ok_frac,
                            'top4_identical_fraction': top4,
                            'max_abs_error': max(r['abs_error'] for r in rows),
                            'median_analytic_cpu_per_candidate': a_cpu,
                            'median_transactional_cpu_per_candidate': t_cpu,
                            'cpu_ratio': a_cpu / t_cpu,
                            'strata': strata_out},
        'fixed_update': {str(r): {'B_vs_D': pr['fixed_bd'], 'A_vs_J': pr['fixed_aj']} for r, pr in per_root.items()},
        'joint': joint,
        'order': {str(r): pr['order'] for r, pr in per_root.items()},
    }


def _strip(res):
    return {k: v for k, v in res.items() if k != '_detail'}


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, separators=(',', ':'), default=_default) + '\n')


def _default(o):
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--roots', type=int, nargs='+', default=[28000, 28001])
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--no-cap', action='store_true')
    args = ap.parse_args()
    recipe = CampaignRecipe(threads=args.threads)
    cap = None if args.no_cap else ProcessTreeMemoryCap(DEFAULT_CAP_BYTES)
    if cap is not None:
        cap.__enter__()
    try:
        run(args.out, recipe, tuple(args.roots))
    finally:
        if cap is not None:
            dump_json(args.out / 'memory_accounting_engineering.json', asdict(cap.accounting()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
