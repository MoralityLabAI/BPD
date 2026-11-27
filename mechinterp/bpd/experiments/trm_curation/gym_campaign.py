"""BPD-TRM-GYM-01 (CURATE-03B Expansion): Paired Multi-Task TRM Gym Campaign.

Runs a matched-CPU, 8-root paired curriculum hill-climbing campaign across 5
verified native ComponentTRM (w=32, r=29, low_cycles=1, 5,380 params) gym tasks:
  - E29_dyck1_bounded_depth        (length-extrapolation target, warmup=250)
  - E31_2hop_pointer_chase         (iterative multi-hop target, warmup=250)
  - E28_dynamic_overwrite_register (distribution-shift control, warmup=250)
  - E27_copy_retrieve_2slot        (extrapolation-collapse control, warmup=250)
  - G05_3slot_copy_retrieve        (3-slot associative memory bridge, warmup=200)

Six paired arms from identical warmup checkpoints on 8 fresh development roots
(29201..29208) under a matched process-CPU budget (default 25.0 CPU-s/arm) and
a hard 6 GiB process-tree memory cap (Win32 Job Object spanning all workers):
  A_uniform_reinvest    : 100% CPU reinvested into plain uniform updates
  B_loss_pool_2x        : forward-only 2x pool hard-example (80%) + explore (20%)
  C_analytic_full16     : 16-cand analytical AdamW screen + transactional accept
  D_joint_early12       : prospective early-12 gate (p=0.25 audits) + 1-round joint
                          minibatch hill climbing + transactional accept
  E_fast_analytic_k4    : low-overhead K=4 analytical AdamW screen, feedback grad
                          refreshed every 4 steps, top-2 aligned micro-batches,
                          no transactional rollback tax
  F_shuffled_k4         : compute-matched sham control for E (shuffled scores)

Preserves CURATE-02 frozen files, CURATE-03B frozen pilot artifacts, failed
selective-control admission (causal auxiliary stays disabled), and EvalMath
REDESIGN_OR_PILOT_ONLY. Evidence class: DEVELOPMENT_ONLY.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import numpy as np
import psutil
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from ..training_steering_stability.stats import _bootstrap, _holm, exact_sign_flip
from .adamw_screen import screen_adamw
from .batch_screen import mean_gradient, search_batch
from .campaign import file_sha256
from .custody import Custody, custody_digest


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8'))
from .eval_suite_36 import _gen_micro_task
from .memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap
from .model import ComponentTRM
from .native_screen import ACCEPT_TOL, Ledger, audit_draw, gradient_parity
from .signal_bank import model_state_sha256, seed_for

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
EXPERIMENT = 'BPD-TRM-GYM-01'
FRESH_GYM_ROOTS = tuple(range(29201, 29209))

TASK_SPECS: dict[str, dict[str, Any]] = {
    'E29_dyck1_bounded_depth': {
        'role': 'length-extrapolation target',
        'warm_steps': 250,
    },
    'E31_2hop_pointer_chase': {
        'role': 'iterative multi-hop target',
        'warm_steps': 250,
    },
    'E28_dynamic_overwrite_register': {
        'role': 'distribution-shift control',
        'warm_steps': 250,
    },
    'E27_copy_retrieve_2slot': {
        'role': 'extrapolation-collapse control',
        'warm_steps': 250,
    },
    'G05_3slot_copy_retrieve': {
        'role': '3-slot associative memory bridge',
        'warm_steps': 200,
    },
}
TASKS = tuple(TASK_SPECS)

ARMS = (
    'A_uniform_reinvest',
    'B_loss_pool_2x',
    'C_analytic_full16',
    'D_joint_early12',
    'E_fast_analytic_k4',
    'F_shuffled_k4',
)
PRIMARY_CONTRASTS = (
    'B_loss_pool_2x',
    'C_analytic_full16',
    'D_joint_early12',
    'E_fast_analytic_k4',
)
SECONDARY_CONTRASTS = (
    ('E_fast_analytic_k4', 'F_shuffled_k4'),
    ('E_fast_analytic_k4', 'C_analytic_full16'),
    ('D_joint_early12', 'C_analytic_full16'),
    ('F_shuffled_k4', 'A_uniform_reinvest'),
)

BATCH = 64
MICRO_N = 16       # 4 micro-batches = 64 examples (matches BATCH)
FAST_MICRO_N = 32  # 2 micro-batches = 64 examples (matches BATCH)
FB_N = 128
LR_WARM = 0.003
LR_CURATE = 0.001
WD = 1e-4
CLIP = 1.0
AUDIT_PROB = 0.25
BUDGET_CPU = 25.0
FAST_FB_REFRESH = 4
EVAL_N = 512
TRAIN_EVAL_N = 256
CHECKPOINT_FRACS = (0.25, 0.50, 0.75, 1.0)
FLOOR = 0.05

ACTIONS_V3 = (
    'ACQUIRE_4', 'ACQUIRE_8', 'ACQUIRE_12', 'ACQUIRE_16',
    'ABSTAIN', 'SELECT_BATCH', 'SWAP_EDGE', 'VERIFY_BATCH',
    'LOSS_POOL_2X', 'FAST_SCREEN_K4', 'REINVEST_CPU',
)


def gen_gym_task(task_id: str, seed: int, n: int, split: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Generate (x [n, T, 11], y [n], n_classes) with strictly disjoint train/eval."""
    if task_id == 'G05_3slot_copy_retrieve':
        g = torch.Generator(device='cpu').manual_seed(seed % (2**63 - 1))
        is_eval = (split == 'eval')
        delay = 10 if is_eval else 5
        T = 3 + delay + 1
        mems = torch.randint(0, 4, (n, 3), generator=g)
        for _ in range(32):
            bad = ((mems.sum(dim=1) % 3 == 0) != is_eval)
            if not bool(bad.any()):
                break
            mems[bad] = torch.randint(0, 4, (int(bad.sum()), 3), generator=g)
        if bool((((mems.sum(dim=1) % 3 == 0) != is_eval)).any()):
            raise RuntimeError('failed to enforce disjoint G05 split')
        q = torch.randint(0, 3, (n,), generator=g)
        x = torch.zeros((n, T, 11))
        x[:, :3, 0] = 1.0
        x[:, 3:-1, 1] = 1.0
        x[:, -1, 2] = 1.0
        row = torch.arange(n)
        for s in range(3):
            x[row, s, 3 + s] = 1.0
            x[row, s, 7 + mems[:, s]] = 1.0
        for t in range(3, 3 + delay):
            da = torch.randint(0, 3, (n,), generator=g)
            dv = torch.randint(0, 4, (n,), generator=g)
            x[row, t, 3 + da] = 1.0
            x[row, t, 7 + dv] = 1.0
        x[row, -1, 3 + q] = 1.0
        return x, mems[row, q], 4
    return _gen_micro_task(task_id, seed, n, split)


def make_model(root: int, task: str) -> ComponentTRM:
    torch.manual_seed(seed_for(root, f'gym01_{task}_init', 0))
    return ComponentTRM(width=32, components=29, low_cycles=1, input_dim=11, values=4)


def _rss_tree() -> int:
    p = psutil.Process()
    total = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            total += c.memory_info().rss
        except psutil.Error:
            pass
    return total


@torch.no_grad()
def eval_metrics(model: ComponentTRM, x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    model.eval()
    logits = model(x)
    preds = logits.argmax(-1)
    corr = int((preds == y).sum())
    n = int(len(y))
    return {
        'accuracy': corr / n,
        'correct_count': corr,
        'n': n,
        'loss': float(F.cross_entropy(logits, y)),
    }


def _margins(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.clone()
    other[torch.arange(len(y)), y] = -1e9
    return true - other.max(-1).values


def _flat(gs) -> torch.Tensor:
    return torch.cat([g.reshape(-1).double() for g in gs if g is not None])


def _cand_grad(model, custody: Custody, x: torch.Tensor, y: torch.Tensor, params) -> tuple[tuple, float]:
    with custody.isolated():
        model.train()
        loss = F.cross_entropy(model(x), y)
        grads = tuple(g.detach().clone() for g in torch.autograd.grad(loss, params))
        return grads, float(loss.detach())


def _fb_grad(model, custody: Custody, x_fb: torch.Tensor, y_fb: torch.Tensor, params) -> tuple[tuple, float, tuple]:
    with custody.isolated():
        point = custody.rng_point()
        model.eval()
        loss = F.cross_entropy(model(x_fb), y_fb)
        grads = tuple(g.detach().clone() for g in torch.autograd.grad(loss, params))
        return grads, float(loss.detach()), point


@torch.no_grad()
def _fb_eval(model, custody: Custody, x_fb: torch.Tensor, y_fb: torch.Tensor, rng_point: tuple) -> float:
    with custody.isolated():
        custody.replay_rng(rng_point)
        model.eval()
        return float(F.cross_entropy(model(x_fb), y_fb))


def _prospective_order(task: str, root: int, win: int, k: int) -> tuple[int, ...]:
    """Deterministic pre-acquisition ordering of candidate indices 0..k-1."""
    items = list(range(k))
    items.sort(key=lambda c: hashlib.sha256(
        repr(('BPD-TRM-GYM-01:acq:v1', task, root, win, c)).encode()).digest())
    return tuple(items)


def _train_step_raw(model, opt, x: torch.Tensor, y: torch.Tensor) -> tuple[float, tuple]:
    model.train()
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    native = tuple(None if p.grad is None else p.grad.detach().clone() for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=True)
    opt.step()
    return float(loss.detach()), native


def compute_signal_bank(model: ComponentTRM, opt: torch.optim.AdamW, task: str, root: int,
                        fb: tuple[torch.Tensor, torch.Tensor], tag: str) -> dict[str, Any]:
    """Observational mechanistic signals + realized one-step gain at one checkpoint."""
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
    for c in range(16):
        x, y, _ = gen_gym_task(task, seed_for(root, f'gym01_{task}_sig_{tag}', c), MICRO_N, 'train')
        g, _ = _cand_grad(model, custody, x, y, params)
        scr = screen_adamw(opt, g, g_fb, CLIP)
        cos = float(F.cosine_similarity(_flat(g), _flat(g_fb), dim=0))
        with custody.isolated():
            with torch.no_grad():
                for p, d in zip(params, scr['deltas'], strict=True):
                    p.add_(d)
                model.eval()
                o1 = model.rollout(xf)
                l1 = float(F.cross_entropy(o1['logits'], yf))
                ds = o1['states'][:, -1] - s0.detach()
                m1 = _margins(o1['logits'], yf)
        dm = m1 - m0
        rows.append({
            'predicted_gain': scr['predicted_gain'],
            'realized_gain': l0 - l1,
            'finite_dose_error': abs((l0 - l1) - scr['predicted_gain']),
            'grad_cosine': cos,
            'observer_signed': float((j_obs * ds).sum(-1).mean()),
            'target_transport': float(dm[wrong].mean()) if bool(wrong.any()) else 0.0,
            'protected_interference': float(dm[right].mean()) if bool(right.any()) else 0.0,
            'state_pr': pr,
            'state_norm': snorm,
        })
    real = [r['realized_gain'] for r in rows]

    def _sp(key: str) -> float | None:
        vals = [r[key] for r in rows]
        if np.std(vals) <= 0 or np.std(real) <= 0:
            return None
        return float(spearmanr(vals, real).statistic)

    return {
        'tag': tag,
        'n_candidates': len(rows),
        'state_pr': pr,
        'state_norm': snorm,
        'mean_finite_dose_error': float(np.mean([r['finite_dose_error'] for r in rows])),
        'sign_accuracy_predicted_gain': float(np.mean([(r['predicted_gain'] > 0) == (r['realized_gain'] > 0) for r in rows])),
        'spearman_predicted_gain': _sp('predicted_gain'),
        'spearman_grad_cosine': _sp('grad_cosine'),
        'spearman_observer_signed': _sp('observer_signed'),
        'spearman_target_transport': _sp('target_transport'),
        'spearman_protected_interference': _sp('protected_interference'),
    }


def observer_parity_check(warm: ComponentTRM, task: str, root: int,
                          fb: tuple[torch.Tensor, torch.Tensor], steps: int = 16) -> dict[str, Any]:
    """Fixed-update uninstrumented vs custody-isolated observer-only must match byte-for-byte."""
    digests = {}
    for instrumented in (False, True):
        m = copy.deepcopy(warm)
        opt = torch.optim.AdamW(m.parameters(), lr=LR_CURATE, weight_decay=WD)
        params = tuple(m.parameters())
        custody = Custody(m, opt)
        torch.manual_seed(seed_for(root, f'gym01_{task}_obs_rng', 0))
        for step in range(steps):
            if instrumented:
                g_fb, _, _ = _fb_grad(m, custody, fb[0], fb[1], params)
                for c in range(4):
                    xc, yc, _ = gen_gym_task(task, seed_for(root, f'gym01_{task}_obs_c{step}', c), MICRO_N, 'train')
                    gc, _ = _cand_grad(m, custody, xc, yc, params)
                    screen_adamw(opt, gc, g_fb, CLIP)
            x, y, _ = gen_gym_task(task, seed_for(root, f'gym01_{task}_obs_tr', step), BATCH, 'train')
            _train_step_raw(m, opt, x, y)
        digests[instrumented] = {
            'state_sha256': model_state_sha256(m),
            'custody_digest': custody_digest(m, opt, include_rng=False),
        }
    return {
        'steps': steps,
        'state_identical': digests[False]['state_sha256'] == digests[True]['state_sha256'],
        'custody_identical': digests[False]['custody_digest'] == digests[True]['custody_digest'],
        'state_sha256': digests[False]['state_sha256'],
    }


def run_arm(arm: str, task: str, root: int, warm: ComponentTRM,
            fb: tuple[torch.Tensor, torch.Tensor],
            ev_xy: tuple[torch.Tensor, torch.Tensor],
            tr_xy: tuple[torch.Tensor, torch.Tensor],
            *, budget_cpu: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one paired arm from shared warmup under strictly matched process CPU."""
    model = copy.deepcopy(warm)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_CURATE, weight_decay=WD)
    params = tuple(model.parameters())
    custody = Custody(model, opt)
    ledger = Ledger()

    xf, yf = fb
    x_ev, y_ev = ev_xy
    x_tr, y_tr = tr_xy

    t0 = time.process_time()
    w0 = time.perf_counter()
    eval_cpu_excluded = 0.0
    peak_rss = _rss_tree()
    win = 0
    committed = 0
    abstained = 0
    rejected = 0
    cands_acquired = 0
    audits = 0
    audits_would_accept = 0
    parity_checks = 0
    parity_ok = 0
    max_rel_l2 = 0.0
    swaps_total = 0
    next_cp_idx = 0
    curve: list[dict[str, Any]] = []
    window_logs: list[dict[str, Any]] = []

    # Cached feedback state for E/F fast arms (refreshed every FAST_FB_REFRESH updates)
    cached_fb_g = None
    cached_fb_l0 = None
    cached_fb_ckpt = None
    steps_since_fb = FAST_FB_REFRESH

    def charged_cpu() -> float:
        return (time.process_time() - t0) - eval_cpu_excluded

    def maybe_eval_checkpoint(force: bool = False) -> None:
        nonlocal eval_cpu_excluded, next_cp_idx
        spent = charged_cpu()
        while next_cp_idx < len(CHECKPOINT_FRACS) and (
            force or spent >= CHECKPOINT_FRACS[next_cp_idx] * budget_cpu
        ):
            te0 = time.process_time()
            ev_m = eval_metrics(model, x_ev, y_ev)
            tr_m = eval_metrics(model, x_tr, y_tr)
            eval_cpu_excluded += time.process_time() - te0
            curve.append({
                'fraction': CHECKPOINT_FRACS[next_cp_idx],
                'charged_cpu_seconds': spent,
                'window': win,
                'committed': committed,
                'eval_acc': ev_m['accuracy'],
                'eval_loss': ev_m['loss'],
                'train_acc': tr_m['accuracy'],
            })
            next_cp_idx += 1
            if force and next_cp_idx >= len(CHECKPOINT_FRACS):
                break

    while win < 50000:
        spent = charged_cpu()
        if spent >= budget_cpu:
            break
        c0 = charged_cpu()
        rec: dict[str, Any] = {'window': win, 'charged_cpu_before': c0}

        if arm == 'A_uniform_reinvest':
            with ledger.timed('data_gen'):
                x, y, _ = gen_gym_task(task, seed_for(root, f'gym01_{task}_u', win), BATCH, 'train')
            with ledger.timed('acceptance_step'):
                loss, _ = _train_step_raw(model, opt, x, y)
            committed += 1
            rec.update(decision='REINVEST_CPU', actions=['REINVEST_CPU'],
                       acquired=0, attempted=True, committed=True, train_loss=loss)

        elif arm == 'B_loss_pool_2x':
            with ledger.timed('data_gen'):
                xp, yp, _ = gen_gym_task(task, seed_for(root, f'gym01_{task}_lp', win), BATCH * 2, 'train')
            with ledger.timed('forward_screen'):
                with custody.isolated():
                    model.eval()
                    with torch.no_grad():
                        losses = F.cross_entropy(model(xp), yp, reduction='none')
            with ledger.timed('controller'):
                n_exp = max(1, int(0.20 * BATCH))
                n_hard = BATCH - n_exp
                order = torch.argsort(losses, descending=True, stable=True)
                hard_idx = order[:n_hard]
                rem_idx = order[n_hard:]
                # Deterministic exploration draw from remainder
                g_exp = torch.Generator(device='cpu').manual_seed(seed_for(root, f'gym01_{task}_lp_exp', win) % (2**63 - 1))
                perm = torch.randperm(len(rem_idx), generator=g_exp)[:n_exp]
                sel = torch.cat((hard_idx, rem_idx[perm]))
                x, y = xp[sel], yp[sel]
            with ledger.timed('acceptance_step'):
                loss, _ = _train_step_raw(model, opt, x, y)
            committed += 1
            rec.update(decision='LOSS_POOL_2X', actions=['LOSS_POOL_2X', 'SELECT_BATCH'],
                       acquired=BATCH * 2, attempted=True, committed=True, train_loss=loss,
                       pool_mean_loss=float(losses.mean()), selected_mean_loss=float(losses[sel].mean()))

        elif arm in ('C_analytic_full16', 'D_joint_early12'):
            use_early12 = (arm == 'D_joint_early12')
            use_joint = (arm == 'D_joint_early12')
            with ledger.timed('provenance'):
                ckpt = model_state_sha256(model)
            with ledger.timed('feedback_gradient'):
                if cached_fb_ckpt == ckpt and cached_fb_g is not None:
                    g_fb, l0, rng_pt = cached_fb_g, cached_fb_l0, cached_fb_rng
                    fb_hit = True
                else:
                    g_fb, l0, rng_pt = _fb_grad(model, custody, xf, yf, params)
                    cached_fb_ckpt, cached_fb_g, cached_fb_l0, cached_fb_rng = ckpt, g_fb, l0, rng_pt
                    fb_hit = False

            order = _prospective_order(task, root, win, 16)
            with ledger.timed('data_gen'):
                cands = {
                    c: gen_gym_task(task, seed_for(root, f'gym01_{task}_c{win}', c), MICRO_N, 'train')[:2]
                    for c in order
                }

            grads: dict[int, tuple] = {}
            scores: dict[int, float] = {}
            update_norms: dict[int, float] = {}
            acquired_list: list[int] = []
            stages_log: list[dict[str, Any]] = []
            abstain = False

            def acquire_one(c: int) -> None:
                with ledger.timed('candidate_gradient'):
                    g, _ = _cand_grad(model, custody, cands[c][0], cands[c][1], params)
                with ledger.timed('analytic_scoring'):
                    sc = screen_adamw(opt, g, g_fb, CLIP)
                grads[c] = g
                scores[c] = sc['predicted_gain']
                update_norms[c] = sc['update_norm']

            for stage in (4, 8, 12, 16):
                for c in order[len(acquired_list):stage]:
                    acquire_one(c)
                    acquired_list.append(c)
                pref = [scores[c] for c in acquired_list]
                stages_log.append({
                    'stage': stage,
                    'max_score': max(pref),
                    'n_positive': sum(s > 0 for s in pref),
                    'mean_score': float(np.mean(pref)),
                })
                if use_early12 and stage == 12 and not any(s > 0 for s in pref):
                    abstain = True
                    break

            if use_early12 and not abstain and not any(scores[c] > 0 for c in acquired_list):
                abstain = True

            cands_acquired += len(acquired_list)
            actions = [f'ACQUIRE_{st["stage"]}' for st in stages_log]

            def eval_and_step(commit_allowed: bool) -> dict[str, Any]:
                nonlocal parity_checks, parity_ok, max_rel_l2, swaps_total
                with ledger.timed('controller'):
                    ranked = sorted(order, key=lambda c: (-scores[c], c))
                    # 3 top-scored + 1 deterministic exploration micro-batch (preserving 25% explore)
                    top3 = ranked[:3]
                    rem = [c for c in order if c not in top3]
                    exp_c = rem[int(hashlib.sha256(repr((task, root, win, 'exp')).encode()).digest()[0]) % len(rem)]
                    init_c = tuple(sorted(top3 + [exp_c]))
                    idx_map = {c: i for i, c in enumerate(order)}
                    init_idx = [idx_map[c] for c in init_c]
                    pin_idx = [idx_map[exp_c]]

                chosen_c = init_c
                pred_init = pred_final = None
                swapped = False
                if use_joint:
                    with ledger.timed('joint_search'):
                        res = search_batch(opt, [grads[c] for c in order], g_fb, init_idx,
                                           pinned=pin_idx, clip=CLIP, rounds=1)
                    chosen_c = tuple(order[i] for i in res['batch'])
                    pred_init = res['initial_gain']
                    pred_final = res['predicted_gain']
                    swapped = (set(chosen_c) != set(init_c))
                    if swapped and commit_allowed:
                        swaps_total += 1

                agg = mean_gradient([grads[c] for c in chosen_c])
                x_b = torch.cat([cands[c][0] for c in chosen_c], dim=0)
                y_b = torch.cat([cands[c][1] for c in chosen_c], dim=0)

                step_info: dict[str, Any] = {
                    'swapped': swapped,
                    'predicted_joint_initial': pred_init,
                    'predicted_joint_final': pred_final,
                }
                with custody.transaction() as txn:
                    with ledger.timed('acceptance_step'):
                        loss, native = _train_step_raw(model, opt, x_b, y_b)
                    with ledger.timed('verification'):
                        par = gradient_parity(agg, native)
                        parity_checks += 1
                        if par['ok']:
                            parity_ok += 1
                        max_rel_l2 = max(max_rel_l2, par['rel_l2_error'])
                        s_nat = screen_adamw(opt, native, g_fb, CLIP)
                        step_info['predicted_joint_native'] = s_nat['predicted_gain']
                        step_info['parity_ok'] = par['ok']
                        step_info['parity_rel_l2'] = par['rel_l2_error']
                    with ledger.timed('feedback_eval'):
                        l1 = _fb_eval(model, custody, xf, yf, rng_pt)
                    actual_gain = l0 - l1
                    would_acc = l1 <= l0 + ACCEPT_TOL
                    acc = would_acc and commit_allowed
                    if acc:
                        txn.commit()
                    step_info.update(
                        train_loss=loss, cal_before=l0, cal_after=l1,
                        actual_gain=actual_gain, would_accept=would_acc, accepted=acc,
                    )
                return step_info

            if abstain:
                abstained += 1
                actions.append('ABSTAIN')
                do_audit, u = audit_draw(root, f'{task}:{arm}', win, AUDIT_PROB)
                audit_info = None
                if do_audit:
                    audits += 1
                    with ledger.timed('audit'):
                        for c in order[len(acquired_list):]:
                            acquire_one(c)
                        audit_info = eval_and_step(commit_allowed=False)
                    if audit_info['would_accept']:
                        audits_would_accept += 1
                rec.update(
                    decision='ABSTAIN', actions=actions, acquired=len(acquired_list),
                    attempted=False, committed=False, stages=stages_log,
                    feedback_loss0=l0, feedback_cache_hit=fb_hit,
                    scores_prefix=[scores[c] for c in acquired_list],
                    update_norms_prefix=[update_norms[c] for c in acquired_list],
                    audit_probability=AUDIT_PROB, audit_draw_u=u,
                    audit_outcome=(audit_info if audit_info is not None else None),
                    outcome_status=('audited_counterfactual' if audit_info is not None else 'missing_unaudited'),
                )
            else:
                res = eval_and_step(commit_allowed=True)
                actions.append('SELECT_BATCH')
                if res['swapped']:
                    actions.append('SWAP_EDGE')
                actions.append('VERIFY_BATCH')
                if res['accepted']:
                    committed += 1
                else:
                    rejected += 1
                rec.update(
                    decision=f'ACQUIRE_{len(acquired_list)}', actions=actions,
                    acquired=len(acquired_list), attempted=True, committed=res['accepted'],
                    stages=stages_log, feedback_loss0=l0, feedback_cache_hit=fb_hit,
                    scores_prefix=[scores[c] for c in acquired_list],
                    update_norms_prefix=[update_norms[c] for c in acquired_list],
                    outcome_status=('committed' if res['accepted'] else 'rolled_back'),
                    **res,
                )

        elif arm in ('E_fast_analytic_k4', 'F_shuffled_k4'):
            is_shuffled = (arm == 'F_shuffled_k4')
            if steps_since_fb >= FAST_FB_REFRESH or cached_fb_g is None:
                with ledger.timed('feedback_gradient'):
                    cached_fb_g, cached_fb_l0, _ = _fb_grad(model, custody, xf, yf, params)
                steps_since_fb = 0
                fb_hit = False
            else:
                fb_hit = True

            with ledger.timed('data_gen'):
                cands = [
                    gen_gym_task(task, seed_for(root, f'gym01_{task}_fk4_{win}', c), FAST_MICRO_N, 'train')[:2]
                    for c in range(4)
                ]
            scores_k4 = []
            for c in range(4):
                with ledger.timed('candidate_gradient'):
                    g, _ = _cand_grad(model, custody, cands[c][0], cands[c][1], params)
                with ledger.timed('analytic_scoring'):
                    sc = screen_adamw(opt, g, cached_fb_g, CLIP)
                scores_k4.append(sc['predicted_gain'])
            cands_acquired += 4

            with ledger.timed('controller'):
                if is_shuffled:
                    rng_s = np.random.default_rng(seed_for(root, f'gym01_{task}_shuf', win))
                    perm = rng_s.permutation(4).tolist()
                    pick = sorted(perm[:2])
                else:
                    ranked = sorted(range(4), key=lambda i: (-scores_k4[i], i))
                    pick = sorted(ranked[:2])
            x_b = torch.cat([cands[i][0] for i in pick], dim=0)
            y_b = torch.cat([cands[i][1] for i in pick], dim=0)
            with ledger.timed('acceptance_step'):
                loss, _ = _train_step_raw(model, opt, x_b, y_b)
            committed += 1
            steps_since_fb += 1
            rec.update(
                decision='FAST_SCREEN_K4',
                actions=['ACQUIRE_4', 'FAST_SCREEN_K4', 'SELECT_BATCH'],
                acquired=4, attempted=True, committed=True, train_loss=loss,
                feedback_loss0=cached_fb_l0, feedback_cache_hit=fb_hit,
                scores_prefix=scores_k4, selected_indices=pick,
                outcome_status='committed_no_verify',
            )
        else:
            raise ValueError(f'unknown arm {arm}')

        rec['window_cpu_seconds'] = charged_cpu() - c0
        window_logs.append(rec)
        if win % 16 == 0:
            peak_rss = max(peak_rss, _rss_tree())
        win += 1
        maybe_eval_checkpoint(force=False)

    total_charged = charged_cpu()
    maybe_eval_checkpoint(force=True)
    final_ev = eval_metrics(model, x_ev, y_ev)
    final_tr = eval_metrics(model, x_tr, y_tr)
    peak_rss = max(peak_rss, _rss_tree())

    summary = {
        'arm': arm,
        'task': task,
        'root': root,
        'windows': win,
        'committed_updates': committed,
        'abstained_windows': abstained,
        'rejected_windows': rejected,
        'candidates_acquired': cands_acquired,
        'audits': audits,
        'audits_would_accept': audits_would_accept,
        'ht_est_missed_accepts': (audits_would_accept / AUDIT_PROB) if audits else 0.0,
        'parity_checks': parity_checks,
        'parity_ok': parity_ok,
        'max_parity_rel_l2': max_rel_l2,
        'swaps_total': swaps_total,
        'charged_cpu_seconds': total_charged,
        'budget_cpu_seconds': budget_cpu,
        'overshoot_cpu_seconds': total_charged - budget_cpu,
        'eval_cpu_excluded_seconds': eval_cpu_excluded,
        'wall_seconds': time.perf_counter() - w0,
        'cpu_by_category': dict(ledger.cpu),
        'cpu_inclusive_by_category': dict(ledger.inclusive),
        'custody_stats': dict(custody.stats),
        'peak_tree_rss_bytes': peak_rss,
        'eval': final_ev,
        'train_eval': final_tr,
        'curve': curve,
        'final_state_sha256': model_state_sha256(model),
    }
    return summary, window_logs


def run_task_root(task: str, root: int, out_dir: Path, *,
                  budget_cpu: float = BUDGET_CPU, threads: int = 2,
                  arms: tuple[str, ...] = ARMS,
                  warm_steps_override: int | None = None) -> dict[str, Any]:
    """Run shared warmup, signals, observer parity, and all paired arms for one (task, root)."""
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    pair_dir = out_dir / task / f'root_{root}'
    pair_dir.mkdir(parents=True, exist_ok=True)
    res_path = pair_dir / 'result.json'
    if res_path.is_file():
        return json.loads(res_path.read_text(encoding='utf-8'))

    warm_steps = warm_steps_override if warm_steps_override is not None else TASK_SPECS[task]['warm_steps']
    x_ev, y_ev, _ = gen_gym_task(task, seed_for(root, f'gym01_{task}_ev', 0), EVAL_N, 'eval')
    x_tr, y_tr, _ = gen_gym_task(task, seed_for(root, f'gym01_{task}_trev', 0), TRAIN_EVAL_N, 'train')
    fb = gen_gym_task(task, seed_for(root, f'gym01_{task}_fb', 0), FB_N, 'train')[:2]

    warm = make_model(root, task)
    opt_w = torch.optim.AdamW(warm.parameters(), lr=LR_WARM, weight_decay=WD)
    tw0 = time.process_time()
    for step in range(1, warm_steps + 1):
        x, y, _ = gen_gym_task(task, seed_for(root, f'gym01_{task}_warm', step), BATCH, 'train')
        _train_step_raw(warm, opt_w, x, y)
    warm_cpu = time.process_time() - tw0
    warm_ev = eval_metrics(warm, x_ev, y_ev)
    warm_tr = eval_metrics(warm, x_tr, y_tr)
    warm_sha = model_state_sha256(warm)

    sig_warm = compute_signal_bank(warm, opt_w, task, root, fb, 'warmup')
    obs_parity = observer_parity_check(warm, task, root, fb, steps=12)

    arm_summaries: dict[str, Any] = {}
    for arm in arms:
        summ, logs = run_arm(arm, task, root, warm, fb, (x_ev, y_ev), (x_tr, y_tr), budget_cpu=budget_cpu)
        arm_summaries[arm] = summ
        # Save compact per-arm window trace for DataGoblin V3 export (gzip -n for determinism)
        _write_jsonl_gz(pair_dir / f'{arm}_windows.jsonl.gz', logs)

    result = {
        'experiment': EXPERIMENT,
        'evidence_class': 'DEVELOPMENT_ONLY',
        'task': task,
        'role': TASK_SPECS[task]['role'],
        'root': root,
        'warm_steps': warm_steps,
        'warm_cpu_seconds': warm_cpu,
        'warm_sha256': warm_sha,
        'warm_eval': warm_ev,
        'warm_train': warm_tr,
        'competent': warm_tr['accuracy'] >= 0.40,
        'headroom': 0.35 <= warm_ev['accuracy'] <= 0.95,
        'warmup_signals': sig_warm,
        'observer_parity': obs_parity,
        'budget_cpu_seconds': budget_cpu,
        'arms': arm_summaries,
        'torch': torch.__version__,
        'python': sys.version.split()[0],
        'platform': platform.platform(),
    }
    dump_json(res_path, result)
    print(json.dumps({
        'task': task, 'root': root, 'warm_ev': round(warm_ev['accuracy'], 3),
        'A': round(arm_summaries['A_uniform_reinvest']['eval']['accuracy'], 3) if 'A_uniform_reinvest' in arm_summaries else None,
        'B': round(arm_summaries['B_loss_pool_2x']['eval']['accuracy'], 3) if 'B_loss_pool_2x' in arm_summaries else None,
        'C': round(arm_summaries['C_analytic_full16']['eval']['accuracy'], 3) if 'C_analytic_full16' in arm_summaries else None,
        'D': round(arm_summaries['D_joint_early12']['eval']['accuracy'], 3) if 'D_joint_early12' in arm_summaries else None,
        'E': round(arm_summaries['E_fast_analytic_k4']['eval']['accuracy'], 3) if 'E_fast_analytic_k4' in arm_summaries else None,
        'F': round(arm_summaries['F_shuffled_k4']['eval']['accuracy'], 3) if 'F_shuffled_k4' in arm_summaries else None,
        'obs_ok': obs_parity['state_identical'],
    }), flush=True)
    return result


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = ''.join(
        json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
        for r in rows
    ).encode('utf-8')
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            gz.write(payload)


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _contrast_from_results(results: list[dict[str, Any]], arm: str, base: str) -> dict[str, Any]:
    eff = []
    for r in results:
        a = r['arms'][arm]['eval']
        b = r['arms'][base]['eval']
        eff.append(Fraction(a['correct_count'] - b['correct_count'], a['n']))
    sf = exact_sign_flip(eff)
    boot = _bootstrap(eff, 10000, 20261127)
    mean = float(sum(eff) / len(eff))
    return {
        'arm': arm,
        'baseline': base,
        'per_root_effect': [float(x) for x in eff],
        'mean_effect': mean,
        'mean_effect_pp': 100.0 * mean,
        'positive_roots': sum(x > 0 for x in eff),
        'zero_roots': sum(x == 0 for x in eff),
        'negative_roots': sum(x < 0 for x in eff),
        'sign_flip': sf,
        'bootstrap_95': boot,
        'exceeds_5pp_floor': mean >= FLOOR,
        '_p': Fraction(sf['p_fraction']),
    }


def run_preflight_for_task(task: str, results: list[dict[str, Any]], arm: str = 'E_fast_analytic_k4',
                           sham: str = 'F_shuffled_k4') -> dict[str, Any]:
    n = results[0]['arms']['A_uniform_reinvest']['eval']['n']
    rows = []
    for r in results:
        ca = r['arms']['A_uniform_reinvest']['eval']['correct_count']
        ce = r['arms'][arm]['eval']['correct_count']
        cf = r['arms'][sham]['eval']['correct_count']
        rows.append({
            'unit_id': str(r['root']),
            'paired_effect': f'{ce - ca}/{n}',
            'sham_effects': [f'{cf - ca}/{n}'],
        })
    spec = {
        'schema_version': 1,
        'experiment_id': f'BPD-TRM-GYM-01-{task}',
        'independent_unit': 'independently initialized TRM root on gym task with paired descendants',
        'alpha': '0.05',
        'scientific_min_effect': '0.05',
        'target_effect': '0.10',
        'measurement_resolution': f'1/{n}',
        'sham_quantile': '0.95',
        'sham_confidence': '0.95',
        'planned_comparisons': len(PRIMARY_CONTRASTS),
        'confirm_units_planned': len(results),
        'pilot': rows,
        'pilot_from_development_only': True,
        'heldout_sealed': False,
        'units_independent': True,
    }
    pf = REPO / 'mechinterp/bpd/experiments/repair_portfolio/evidence/preflight.py'
    spec_ = importlib.util.spec_from_file_location('gym01_preflight', pf)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return {'input': spec, 'result': mod.preflight(spec)}


def summarize_campaign(out_dir: Path, tasks: tuple[str, ...], roots: tuple[int, ...]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_task[task] = [
            json.loads((out_dir / task / f'root_{r}' / 'result.json').read_text(encoding='utf-8'))
            for r in roots
        ]
    arms = [a for a in ARMS if all(a in r['arms'] for rs in by_task.values() for r in rs)]
    task_summaries: dict[str, Any] = {}

    for task, rs in by_task.items():
        arm_table = {}
        for a in arms:
            runs = [r['arms'][a] for r in rs]
            arm_table[a] = {
                'eval_acc': statistics.mean(x['eval']['accuracy'] for x in runs),
                'eval_loss': statistics.mean(x['eval']['loss'] for x in runs),
                'train_acc': statistics.mean(x['train_eval']['accuracy'] for x in runs),
                'windows': statistics.mean(x['windows'] for x in runs),
                'committed_updates': statistics.mean(x['committed_updates'] for x in runs),
                'abstained_windows': statistics.mean(x['abstained_windows'] for x in runs),
                'rejected_windows': statistics.mean(x['rejected_windows'] for x in runs),
                'candidates_acquired': statistics.mean(x['candidates_acquired'] for x in runs),
                'audits': sum(x['audits'] for x in runs),
                'audits_would_accept': sum(x['audits_would_accept'] for x in runs),
                'parity_ok_fraction': (
                    sum(x['parity_ok'] for x in runs) / max(sum(x['parity_checks'] for x in runs), 1)
                    if sum(x['parity_checks'] for x in runs) else None
                ),
                'max_parity_rel_l2': max(x['max_parity_rel_l2'] for x in runs),
                'charged_cpu_seconds': statistics.mean(x['charged_cpu_seconds'] for x in runs),
                'max_overshoot_cpu_seconds': max(x['overshoot_cpu_seconds'] for x in runs),
                'peak_tree_rss_mib': max(x['peak_tree_rss_bytes'] for x in runs) / 2**20,
            }
        primary = [_contrast_from_results(rs, a, 'A_uniform_reinvest') for a in PRIMARY_CONTRASTS if a in arms]
        adj = _holm([c['_p'] for c in primary])
        for c, p in zip(primary, adj, strict=True):
            c['holm_p'] = float(p)
            c.pop('_p', None)
        secondary = []
        for a, b in SECONDARY_CONTRASTS:
            if a in arms and b in arms:
                c = _contrast_from_results(rs, a, b)
                c.pop('_p', None)
                secondary.append(c)

        sigs = [r['warmup_signals'] for r in rs]
        sig_means = {}
        for k in ('state_pr', 'state_norm', 'mean_finite_dose_error', 'sign_accuracy_predicted_gain',
                  'spearman_predicted_gain', 'spearman_grad_cosine', 'spearman_observer_signed',
                  'spearman_target_transport', 'spearman_protected_interference'):
            vals = [s[k] for s in sigs if s.get(k) is not None]
            sig_means[k] = statistics.mean(vals) if vals else None

        preflight = (
            run_preflight_for_task(task, rs)
            if len(rs) >= 2 and {'A_uniform_reinvest', 'E_fast_analytic_k4', 'F_shuffled_k4'} <= set(arms)
            else None
        )
        task_summaries[task] = {
            'role': TASK_SPECS[task]['role'],
            'warm_steps': rs[0]['warm_steps'],
            'warm_eval_acc_mean': statistics.mean(r['warm_eval']['accuracy'] for r in rs),
            'warm_train_acc_mean': statistics.mean(r['warm_train']['accuracy'] for r in rs),
            'competent_roots': sum(int(r['competent']) for r in rs),
            'headroom_roots': sum(int(r['headroom']) for r in rs),
            'observer_parity_all_identical': all(r['observer_parity']['state_identical'] for r in rs),
            'warmup_signals_mean': sig_means,
            'arm_means': arm_table,
            'primary_contrasts_vs_A_holm': primary,
            'secondary_contrasts': secondary,
            'evalmath_preflight_E_vs_A': preflight,
        }

    # Pooled across all tasks per root (independent unit = root, averaged across tasks)
    pooled_arms: dict[str, float] = {}
    for a in arms:
        pooled_arms[a] = statistics.mean(task_summaries[t]['arm_means'][a]['eval_acc'] for t in tasks)

    pooled_primary = []
    for a in PRIMARY_CONTRASTS:
        if a not in arms:
            continue
        eff = []
        for idx in range(len(roots)):
            num = sum(by_task[t][idx]['arms'][a]['eval']['correct_count'] -
                      by_task[t][idx]['arms']['A_uniform_reinvest']['eval']['correct_count'] for t in tasks)
            den = sum(by_task[t][idx]['arms'][a]['eval']['n'] for t in tasks)
            eff.append(Fraction(num, den))
        sf = exact_sign_flip(eff)
        boot = _bootstrap(eff, 10000, 20261127)
        mean = float(sum(eff) / len(eff))
        pooled_primary.append({
            'arm': a, 'baseline': 'A_uniform_reinvest',
            'per_root_effect': [float(x) for x in eff],
            'mean_effect': mean, 'mean_effect_pp': 100.0 * mean,
            'positive_roots': sum(x > 0 for x in eff),
            'negative_roots': sum(x < 0 for x in eff),
            'sign_flip': sf, 'bootstrap_95': boot,
            'exceeds_5pp_floor': mean >= FLOOR,
            '_p': Fraction(sf['p_fraction']),
        })
    padj = _holm([c['_p'] for c in pooled_primary])
    for c, p in zip(pooled_primary, padj, strict=True):
        c['holm_p'] = float(p)
        c.pop('_p', None)

    summary = {
        'experiment': EXPERIMENT,
        'evidence_class': 'DEVELOPMENT_ONLY',
        'decision': 'REDESIGN_OR_PILOT_ONLY (preserved; development gym evidence only)',
        'causal_auxiliary_intervention': 'disabled (CURATE-02 selective-control admission failed)',
        'tasks': list(tasks),
        'roots': list(roots),
        'arms': arms,
        'budget_cpu_seconds': by_task[tasks[0]][0]['budget_cpu_seconds'],
        'observer_parity_all_identical': all(task_summaries[t]['observer_parity_all_identical'] for t in tasks),
        'pooled_arm_eval_acc': pooled_arms,
        'pooled_primary_contrasts_vs_A_holm': pooled_primary,
        'per_task': task_summaries,
    }
    dump_json(out_dir / 'gym_campaign_summary.json', summary)
    return summary


def export_datagoblin_gym_v3(out_dir: Path, tasks: tuple[str, ...], roots: tuple[int, ...],
                             export_dir: Path) -> dict[str, Any]:
    """Export strictly TRAIN-only multi-task gym trajectories for DataGoblin."""
    export_dir.mkdir(parents=True, exist_ok=True)
    traj_path = export_dir / 'trajectories_gym_v3.jsonl.gz'
    horizon_path = export_dir / 'horizon_outcomes_gym_v3.jsonl'
    unit_path = export_dir / 'unit_costs_gym_v3.json'

    counts: Counter = Counter()
    action_counts: Counter = Counter({a: 0 for a in ACTIONS_V3})
    unit_acc: dict[str, dict[str, float]] = {}

    traj_chunks: list[bytes] = []
    horizon_lines: list[str] = []

    for task in tasks:
        unit_acc[task] = {'cand_cpu': 0.0, 'att_cpu': 0.0, 'base_cpu': 0.0, 'n_c': 0.0, 'n_a': 0.0, 'n_w': 0.0}
        for root in roots:
            pdir = out_dir / task / f'root_{root}'
            res = json.loads((pdir / 'result.json').read_text(encoding='utf-8'))
            for arm, summ in res['arms'].items():
                horizon_lines.append(json.dumps({
                    'schema': 'bpd-trm-gym01-datagoblin-v3',
                    'evidence_class': 'DEVELOPMENT_ONLY',
                    'label_kind': 'horizon',
                    'split_evaluated': 'eval',
                    'decision_rows_attached': False,
                    'task': task, 'root': root, 'arm': arm,
                    'warm_eval_acc': res['warm_eval']['accuracy'],
                    'final_eval_acc': summ['eval']['accuracy'],
                    'delta_eval_acc': summ['eval']['accuracy'] - res['warm_eval']['accuracy'],
                    'committed_updates': summ['committed_updates'],
                    'windows': summ['windows'],
                    'charged_cpu_seconds': summ['charged_cpu_seconds'],
                }, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')

                if arm == 'C_analytic_full16':
                    c = summ['cpu_by_category']
                    cand_c = c.get('candidate_gradient', 0.0) + c.get('analytic_scoring', 0.0)
                    att_c = c.get('acceptance_step', 0.0) + c.get('feedback_eval', 0.0) + c.get('verification', 0.0)
                    base_c = max(summ['charged_cpu_seconds'] - cand_c - att_c, 0.0)
                    u = unit_acc[task]
                    u['cand_cpu'] += cand_c
                    u['att_cpu'] += att_c
                    u['base_cpu'] += base_c
                    u['n_c'] += summ['candidates_acquired']
                    u['n_a'] += summ['windows'] - summ['abstained_windows']
                    u['n_w'] += summ['windows']

                wfile = pdir / f'{arm}_windows.jsonl.gz'
                if not wfile.is_file():
                    continue
                for w in _read_jsonl_gz(wfile):
                    for a in w.get('actions', []):
                        action_counts[a] += 1
                    counts['windows'] += 1
                    if w.get('decision') == 'ABSTAIN':
                        counts['audited_abstentions' if w.get('audit_outcome') else 'unaudited_abstentions'] += 1
                    # Keep TRAIN-only decision traces for scored/screened arms (and compact uniform markers)
                    if arm in ('C_analytic_full16', 'D_joint_early12', 'E_fast_analytic_k4', 'F_shuffled_k4'):
                        row = {
                            'schema': 'bpd-trm-gym01-datagoblin-v3',
                            'split': 'train',
                            'training_eligible': True,
                            'evaluation_leakage': False,
                            'evidence_class': 'DEVELOPMENT_ONLY',
                            'record_id': hashlib.sha256(repr((task, root, arm, w['window'])).encode()).hexdigest()[:24],
                            'provenance': {
                                'task': task, 'root': root, 'arm': arm, 'window': w['window'],
                                'warm_sha256': res['warm_sha256'],
                            },
                            'state': {
                                'charged_cpu_before': w['charged_cpu_before'],
                                'feedback_loss0': w.get('feedback_loss0'),
                                'feedback_cache_hit': w.get('feedback_cache_hit'),
                                'stages': w.get('stages'),
                            },
                            'scores_prefix': w.get('scores_prefix'),
                            'update_norms_prefix': w.get('update_norms_prefix'),
                            'actions': w.get('actions'),
                            'decision': w.get('decision'),
                            'acquired': w.get('acquired'),
                            'attempted': bool(w.get('attempted')),
                            'committed': bool(w.get('committed')),
                            'joint_outcome': {
                                'status': w.get('outcome_status'),
                                'predicted_joint_native': w.get('predicted_joint_native'),
                                'predicted_joint_initial': w.get('predicted_joint_initial'),
                                'predicted_joint_final': w.get('predicted_joint_final'),
                                'actual_gain': w.get('actual_gain'),
                                'cal_before': w.get('cal_before'),
                                'cal_after': w.get('cal_after'),
                                'audit_probability': w.get('audit_probability'),
                                'audit_outcome': w.get('audit_outcome'),
                            },
                            'cost': {'window_cpu_seconds': w['window_cpu_seconds']},
                        }
                        traj_chunks.append((json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8'))
                        counts['exported_decision_rows'] += 1

    raw_traj = b''.join(traj_chunks)
    with traj_path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            gz.write(raw_traj)

    horizon_bytes = ''.join(horizon_lines).encode('utf-8')
    horizon_path.write_bytes(horizon_bytes)

    unit_costs = {}
    for task, u in unit_acc.items():
        unit_costs[task] = {
            'per_candidate': u['cand_cpu'] / max(u['n_c'], 1.0),
            'per_attempt': u['att_cpu'] / max(u['n_a'], 1.0),
            'per_window_base': u['base_cpu'] / max(u['n_w'], 1.0),
        }
    unit_bytes = (json.dumps(unit_costs, indent=2, sort_keys=True) + '\n').encode('utf-8')
    unit_path.write_bytes(unit_bytes)

    manifest = {
        'schema': 'bpd-trm-gym01-datagoblin-v3',
        'experiment': EXPERIMENT,
        'evidence_class': 'DEVELOPMENT_ONLY',
        'split': 'train',
        'tasks': list(tasks),
        'roots': list(roots),
        'action_vocabulary': list(ACTIONS_V3),
        'action_counts': dict(action_counts),
        'counts': dict(counts),
        'files': {
            'trajectories_gym_v3.jsonl': {
                'stored_as': 'trajectories_gym_v3.jsonl.gz',
                'sha256': hashlib.sha256(raw_traj).hexdigest(),
                'gz_sha256': file_sha256(traj_path),
                'bytes': len(raw_traj),
            },
            'horizon_outcomes_gym_v3.jsonl': {
                'stored_as': 'horizon_outcomes_gym_v3.jsonl',
                'sha256': hashlib.sha256(horizon_bytes).hexdigest(),
                'bytes': len(horizon_bytes),
            },
            'unit_costs_gym_v3.json': {
                'stored_as': 'unit_costs_gym_v3.json',
                'sha256': hashlib.sha256(unit_bytes).hexdigest(),
                'bytes': len(unit_bytes),
            },
        },
        'missingness': 'null + status; unaudited abstentions stay missing_unaudited, never zero',
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode('utf-8')
    (export_dir / 'manifest.json').write_bytes(manifest_bytes)
    return manifest


def _worker_entry(args_tuple):
    task, root, out_dir_str, budget_cpu, threads, arms, warm_override = args_tuple
    return run_task_root(
        task, root, Path(out_dir_str),
        budget_cpu=budget_cpu, threads=threads, arms=arms,
        warm_steps_override=warm_override,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--roots', type=int, nargs='+', default=list(FRESH_GYM_ROOTS))
    ap.add_argument('--arms', nargs='+', default=list(ARMS))
    ap.add_argument('--budget-cpu', type=float, default=BUDGET_CPU)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--warm-steps', type=int, default=None)
    ap.add_argument('--export-dir', type=Path, default=None)
    ap.add_argument('--no-cap', action='store_true')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tasks = tuple(args.tasks)
    roots = tuple(args.roots)
    arms = tuple(args.arms)

    cap = None if args.no_cap else ProcessTreeMemoryCap(DEFAULT_CAP_BYTES)
    if cap is not None:
        cap.__enter__()
    try:
        jobs = [
            (task, root, str(args.out), args.budget_cpu, args.threads, arms, args.warm_steps)
            for task in tasks for root in roots
        ]
        if args.workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                _worker_entry(j)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(_worker_entry, j) for j in jobs]
                for fut in as_completed(futs):
                    fut.result()

        summary = summarize_campaign(args.out, tasks, roots)
        exp_dir = args.export_dir or (args.out / 'datagoblin_export_v3')
        manifest = export_datagoblin_gym_v3(args.out, tasks, roots, exp_dir)
        print(json.dumps({
            'pooled_arm_eval_acc': summary['pooled_arm_eval_acc'],
            'observer_parity_all_identical': summary['observer_parity_all_identical'],
            'export_counts': manifest['counts'],
        }, indent=2), flush=True)
    finally:
        if cap is not None:
            acc = cap.accounting()
            dump_json(args.out / 'memory_accounting_gym_campaign.json', asdict(acc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
