"""CURATE-03B exploratory TRM gym expansion: E29, E31, E27, E28 (development only).

Three stages per task, each on exploratory roots 29100/29101 (never curation roots):

1. Competence/headroom: uniform ComponentTRM training (CURATE-02 shape, input 11,
   4 values) for WARM_STEPS updates; frozen rule: basic competence when TRAIN
   accuracy >= 0.40, measurable headroom when 0.35 <= EVAL accuracy <= 0.95.
2. Observational signal study at checkpoints along the uniform trajectory: for
   16 candidate micro-batches per checkpoint, compute mechanistic signals and
   the realized one-step feedback gain of each candidate under custody. No
   training decision depends on it.
3. Matched-CPU hill climbing (only when stage 1 passes): uniform reinvestment vs
   analytical screen + full-16 gate + one-round joint search + transactional
   acceptance, same budget. The previously failed selective causal-auxiliary
   intervention is not used.

Signals (per candidate, at the parent checkpoint, feedback = TRAIN-split batch):
  predicted_gain        -<g_fb, delta_AdamW> (exact read-only AdamW displacement)
  grad_cosine           cos(g_candidate, g_fb)
  state_pr / state_norm participation ratio and mean norm of final joint states
  observer_signed       J_obs . ds, J_obs = d mean correct margin / d final state,
                        ds = final-state change under the candidate displacement
  target_transport      mean margin change on feedback examples wrong at parent
  protected_interference mean margin change on examples right at parent
  finite_dose_error     |realized gain - predicted gain|
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from .adamw_screen import screen_adamw
from .batch_screen import mean_gradient, search_batch
from .campaign import dump_json
from .custody import Custody
from .eval_suite_36 import _gen_micro_task
from .memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap
from .model import ComponentTRM
from .native_screen import ACCEPT_TOL, audit_draw
from .signal_bank import seed_for

TASKS = ('E29_dyck1_bounded_depth', 'E31_2hop_pointer_chase', 'E27_copy_retrieve_2slot',
         'E28_dynamic_overwrite_register')
ROLES = {'E29_dyck1_bounded_depth': 'length-extrapolation target', 'E31_2hop_pointer_chase': 'iterative target',
         'E27_copy_retrieve_2slot': 'curriculum failure control', 'E28_dynamic_overwrite_register': 'distribution-shift control'}
ROOTS = (29100, 29101)
WARM_STEPS = 450
BATCH = 64
CAND_N = 16
FB_N = 128
CHECKPOINTS = (150, 300, 450)
BUDGET_CPU = 30.0
LR, WD, CLIP = 0.003, 1e-4, 1.0
CURATION_LR = 0.001


def _model(root, task):
    torch.manual_seed(seed_for(root, f'gym03b_{task}_init', 0))
    return ComponentTRM(width=32, components=29, low_cycles=1, input_dim=11, values=4)


def _acc(model, x, y):
    model.eval()
    with torch.no_grad():
        return float((model(x).argmax(-1) == y).float().mean())


def _train_step(model, opt, x, y):
    model.train()
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step()
    return float(loss.detach())


def _grad(model, custody, x, y, params):
    with custody.isolated():
        model.train()
        loss = F.cross_entropy(model(x), y)
        return tuple(g.detach().clone() for g in torch.autograd.grad(loss, params))


def _flat(gs):
    return torch.cat([g.reshape(-1).double() for g in gs])


def _margins(logits, y):
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.clone()
    other[torch.arange(len(y)), y] = -1e9
    return true - other.max(-1).values


def checkpoint_signals(model, opt, task, root, step, fb) -> list[dict]:
    """Observational per-candidate signals + realized gain at one checkpoint."""
    params = tuple(model.parameters())
    custody = Custody(model, opt)
    xf, yf = fb
    with custody.isolated():
        model.eval()
        out = model.rollout(xf)
        s0 = out['states'][:, -1]
        loss0 = F.cross_entropy(out['logits'], yf)
        g_fb = tuple(g.detach().clone() for g in torch.autograd.grad(loss0, params, retain_graph=True))
        s_req = s0.detach().clone().requires_grad_(True)
        # Observer: margin as a function of the final state through the output head.
        m_obs = _margins(model.output(s_req[:, :model.width]), yf).mean()
        j_obs = torch.autograd.grad(m_obs, s_req)[0].detach()
        m0 = _margins(out['logits'].detach(), yf)
        st = s0.detach().double()
        cov = torch.cov(st.T)
        ev = torch.linalg.eigvalsh(cov).clamp_min(0)
        pr = float(ev.sum() ** 2 / ev.square().sum().clamp_min(1e-30))
        snorm = float(st.norm(dim=1).mean())
    l0 = float(loss0.detach())
    wrong, right = m0 <= 0, m0 > 0
    rows = []
    for c in range(CAND_N):
        x, y, _ = _gen_micro_task(task, seed_for(root, f'gym03b_{task}_cand_s{step}', c), CAND_N, 'train')
        g = _grad(model, custody, x, y, params)
        scr = screen_adamw(opt, g, g_fb, CLIP)
        cos = float(torch.nn.functional.cosine_similarity(_flat(g), _flat(g_fb), dim=0))
        with custody.isolated():
            with torch.no_grad():
                for p, d in zip(params, scr['deltas']):
                    p.add_(d)
                model.eval()
                o1 = model.rollout(xf)
                l1 = float(F.cross_entropy(o1['logits'], yf))
                ds = o1['states'][:, -1] - s0.detach()
                m1 = _margins(o1['logits'], yf)
        dm = m1 - m0
        rows.append({
            'task': task, 'root': root, 'step': step, 'candidate': c,
            'predicted_gain': scr['predicted_gain'], 'realized_gain': l0 - l1,
            'finite_dose_error': abs((l0 - l1) - scr['predicted_gain']),
            'grad_cosine': cos, 'update_norm': scr['update_norm'], 'gradient_norm': scr['gradient_norm'],
            'state_pr': pr, 'state_norm': snorm,
            'observer_signed': float((j_obs * ds).sum(-1).mean()),
            'observer_actual_margin_change': float(dm.mean()),
            'target_transport': float(dm[wrong].mean()) if bool(wrong.any()) else 0.0,
            'protected_interference': float(dm[right].mean()) if bool(right.any()) else 0.0,
        })
    return rows


def competence_and_signals(task, root) -> dict:
    model = _model(root, task)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    x_ev, y_ev, _ = _gen_micro_task(task, seed_for(root, f'gym03b_{task}_ev', 0), 512, 'eval')
    x_tr, y_tr, _ = _gen_micro_task(task, seed_for(root, f'gym03b_{task}_trev', 0), 256, 'train')
    fb = _gen_micro_task(task, seed_for(root, f'gym03b_{task}_fb', 0), FB_N, 'train')[:2]
    signals, curve = [], []
    t0 = time.process_time()
    for step in range(1, WARM_STEPS + 1):
        x, y, _ = _gen_micro_task(task, seed_for(root, f'gym03b_{task}_tr', step), BATCH, 'train')
        _train_step(model, opt, x, y)
        if step in CHECKPOINTS:
            curve.append({'step': step, 'train_acc': _acc(model, x_tr, y_tr), 'eval_acc': _acc(model, x_ev, y_ev)})
            signals += checkpoint_signals(model, opt, task, root, step, fb)
    tr, ev = _acc(model, x_tr, y_tr), _acc(model, x_ev, y_ev)
    return {'task': task, 'root': root, 'train_acc': tr, 'eval_acc': ev, 'curve': curve,
            'competent': tr >= 0.40, 'headroom': 0.35 <= ev <= 0.95,
            'warm_cpu_seconds': time.process_time() - t0, 'model': model, 'signals': signals}


def hill_climb(task, root, warm) -> dict:
    """Matched-CPU uniform reinvestment vs analytical joint screen with full-16 gate."""
    x_ev, y_ev, _ = _gen_micro_task(task, seed_for(root, f'gym03b_{task}_ev', 0), 512, 'eval')
    fb = _gen_micro_task(task, seed_for(root, f'gym03b_{task}_fb2', 0), FB_N, 'train')[:2]
    out = {}
    for arm in ('uniform', 'screen'):
        model = copy.deepcopy(warm)
        opt = torch.optim.AdamW(model.parameters(), lr=CURATION_LR, weight_decay=WD)
        params = tuple(model.parameters())
        custody = Custody(model, opt)
        t0 = time.process_time()
        win = committed = abstained = 0
        gains = []
        while time.process_time() - t0 < BUDGET_CPU:
            if arm == 'uniform':
                x, y, _ = _gen_micro_task(task, seed_for(root, f'gym03b_{task}_hc_u', win), BATCH, 'train')
                _train_step(model, opt, x, y)
                committed += 1
            else:
                cands = [_gen_micro_task(task, seed_for(root, f'gym03b_{task}_hc_c{win}', c), CAND_N, 'train')[:2]
                         for c in range(16)]
                with custody.isolated():
                    model.eval()
                    l0t = F.cross_entropy(model(fb[0]), fb[1])
                    g_fb = tuple(g.detach().clone() for g in torch.autograd.grad(l0t, params))
                l0 = float(l0t.detach())
                grads = [_grad(model, custody, x, y, params) for x, y in cands]
                scores = [screen_adamw(opt, g, g_fb, CLIP)['predicted_gain'] for g in grads]
                if not any(s > 0 for s in scores):
                    abstained += 1
                    win += 1
                    continue
                init = tuple(sorted(np.argsort(-np.array(scores), kind='stable')[:4].tolist()))
                res = search_batch(opt, grads, g_fb, init, clip=CLIP, rounds=1)
                x = torch.cat([cands[i][0] for i in res['batch']])
                y = torch.cat([cands[i][1] for i in res['batch']])
                with custody.transaction() as txn:
                    _train_step(model, opt, x, y)
                    with custody.isolated():
                        model.eval()
                        with torch.no_grad():
                            l1 = float(F.cross_entropy(model(fb[0]), fb[1]))
                    if l1 <= l0 + ACCEPT_TOL:
                        txn.commit()
                        committed += 1
                        gains.append(l0 - l1)
            win += 1
        out[arm] = {'windows': win, 'committed': committed, 'abstained': abstained,
                    'cpu_seconds': time.process_time() - t0, 'eval_acc': _acc(model, x_ev, y_ev),
                    'mean_committed_gain': statistics.mean(gains) if gains else None}
    out['screen_minus_uniform_pp'] = 100 * (out['screen']['eval_acc'] - out['uniform']['eval_acc'])
    return out


SIGNALS = ('predicted_gain', 'grad_cosine', 'update_norm', 'gradient_norm', 'observer_signed',
           'target_transport', 'protected_interference', 'state_pr', 'state_norm')


def signal_validity(rows) -> dict:
    """Spearman of each signal with realized gain, per task and per (task, checkpoint)."""
    out = {}
    tasks = sorted({r['task'] for r in rows})
    for t in tasks:
        tr = [r for r in rows if r['task'] == t]
        per = {}
        for s in SIGNALS:
            a = [r[s] for r in tr]
            b = [r['realized_gain'] for r in tr]
            per[s] = float(spearmanr(a, b).statistic) if np.std(a) > 0 and np.std(b) > 0 else None
        within = {}
        for s in SIGNALS:
            vals = []
            for key in sorted({(r['root'], r['step']) for r in tr}):
                sub = [r for r in tr if (r['root'], r['step']) == key]
                a = [r[s] for r in sub]
                b = [r['realized_gain'] for r in sub]
                if np.std(a) > 0 and np.std(b) > 0:
                    vals.append(float(spearmanr(a, b).statistic))
            within[s] = statistics.mean(vals) if vals else None
        out[t] = {'pooled_spearman_with_realized_gain': per,
                  'mean_within_checkpoint_spearman': within,
                  'mean_finite_dose_error': statistics.mean(r['finite_dose_error'] for r in tr),
                  'sign_accuracy_predicted_gain': float(np.mean([(r['predicted_gain'] > 0) == (r['realized_gain'] > 0) for r in tr]))}
    return out


def run(out_dir: Path) -> dict:
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {'experiment': 'BPD-TRM-CURATE-03B-gym', 'evidence_class': 'EXPLORATORY_DEVELOPMENT',
               'roots': list(ROOTS), 'tasks': {}, 'frozen_rule': 'competent: TRAIN acc >= 0.40; headroom: 0.35 <= EVAL acc <= 0.95; hill climbing only when both pass on every root',
               'causal_auxiliary_intervention': 'not activated (CURATE-02 selective-control admission failed)'}
    all_rows = []
    for task in TASKS:
        per_root = []
        for root in ROOTS:
            c = competence_and_signals(task, root)
            all_rows += c['signals']
            per_root.append(c)
            print(json.dumps({'task': task, 'root': root, 'train': c['train_acc'], 'eval': c['eval_acc']}), flush=True)
        admitted = all(c['competent'] and c['headroom'] for c in per_root)
        hc = {}
        if admitted:
            for c in per_root:
                hc[str(c['root'])] = hill_climb(task, c['root'], c['model'])
                print(json.dumps({'task': task, 'root': c['root'], 'hill_climb': hc[str(c['root'])]['screen_minus_uniform_pp']}), flush=True)
        summary['tasks'][task] = {
            'role': ROLES[task],
            'competence': [{k: v for k, v in c.items() if k not in ('model', 'signals')} for c in per_root],
            'admitted_for_hill_climbing': admitted,
            'hill_climbing': hc or None,
        }
    with (out_dir / 'gym_signal_rows.jsonl').open('w', encoding='utf-8', newline='\n') as f:
        for r in all_rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    summary['signal_validity'] = signal_validity(all_rows)
    dump_json(out_dir / 'gym_summary.json', summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--no-cap', action='store_true')
    a = ap.parse_args()
    cap = None if a.no_cap else ProcessTreeMemoryCap(DEFAULT_CAP_BYTES)
    if cap is not None:
        cap.__enter__()
    try:
        run(a.out)
    finally:
        if cap is not None:
            dump_json(a.out / 'memory_accounting_gym.json', asdict(cap.accounting()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
