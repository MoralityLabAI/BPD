"""MMS-TRM Workstreams D/E: transactional compute-reinvestment trainer.

Paired arms from one common warmup (weights, fresh AdamW at curation lr,
identical proposal streams and nuisance draws):

  A_ordinary     uniform ordinary training; reinvests all compute.
  B_ord_grad     ordinary feedback-gradient curation (exact same-checkpoint
                 cache, prediction-only transactional preview, acceptance check).
  C_mech_fresh   mechanistic feedback-gradient curation, fresh calibration
                 gradient at every checkpoint (exact cache), preview probes.
  C_probe        C with CURATE-02's full ``probe_update`` (also realizes the
                 calibration loss per candidate). Selections are identical to
                 C by construction; the pair isolates that exact economy.
  D_atlas        C plus the cross-checkpoint atlas: the mechanistic calibration
                 gradient is a detached observer reused across checkpoints
                 under the screen/verify/refresh hierarchy, plus a no-audit
                 action chosen by a first-order value-of-audit rule.
  E_shuffled     D with shuffled scores (same audit work, verification, policy).
  F_obs_only     D's audit work and schedule with uniform selection, no gating.

Budgets: ``updates`` (fixed number of windows) or ``cpu`` (fixed process CPU
including setup, cache/atlas traffic, probes, rollbacks and discarded
updates; a window starts only if its estimated cost fits).
Endpoints are ordinary, steering-disabled accuracies. Engineering only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from mechinterp.bpd.experiments.trm_curation.cache import CacheKey, MechanisticCache
from mechinterp.bpd.experiments.trm_curation.campaign import (_get_cached_cal_grad, _sample_window_proposals,
                                                             eval_model_ordinary)
from mechinterp.bpd.experiments.trm_curation.memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap
from mechinterp.bpd.experiments.trm_curation.observer import observe_edge
from mechinterp.bpd.experiments.trm_curation.scheduler import Candidate, edge_views, pair_key, select_candidates
from mechinterp.bpd.experiments.trm_curation.signal_bank import (cached_edge_batch, discover_slot_projectors,
                                                                 mechanistic_feedback_loss, ordinary_feedback_loss,
                                                                 stratified_edge_sample)
from mechinterp.bpd.experiments.trm_curation.transaction import _restore_native_state, _snapshot_native_state, probe_update

from . import geometry as geo
from .atlas import AtlasKey, MechanismAtlas, Policy, decide
from .common import (ENGINEERING_ROOTS, PANEL_DELAYS, accounting_dict, dump_json, load_checkpoint,
                     model_state_sha256, recipe, sccg_seed, source_hashes)
from .drift import drift_panel
from mechinterp.bpd.experiments.trm_curation.data import edge_batch

ARMS = ('A_ordinary', 'B_ord_grad', 'C_mech_fresh', 'C_probe', 'D_atlas', 'E_shuffled', 'F_obs_only')
ATLAS_ARMS = ('D_atlas', 'E_shuffled', 'F_obs_only')
MECH_ARMS = ('C_mech_fresh', 'C_probe') + ATLAS_ARMS
NO_AUDIT_WINDOWS = 4


def preview_update(model, optimizer, train_loss, cal_grad, *, clip=1.0):
    """Prediction-only transactional preview: the clipped AdamW step and -g.dtheta.

    Same candidate step and the same predicted gain as ``probe_update`` with a
    cached calibration gradient, without realizing the calibration loss.
    Parameters, buffers, optimizer state, grads, training flags, native BPD
    module state and global RNG are restored.
    """
    params = tuple(model.parameters())
    t0 = time.process_time()
    initial = copy.deepcopy(model.state_dict())
    opt = copy.deepcopy(optimizer.state_dict())
    grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
    flags = [(m, m.training) for m in model.modules()]
    native = _snapshot_native_state(model)
    rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    try:
        old = [p.detach().clone() for p in params]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            predicted = -sum(float((a.detach().double() * (p.detach() - b).double()).sum())
                             for a, p, b in zip(cal_grad, params, old) if a is not None)
            step_sq = sum(float((p.detach() - b).double().square().sum()) for p, b in zip(params, old))
        out = dict(predicted_gain=predicted, train_loss=float(loss.detach()), update_norm=step_sq ** .5)
    finally:
        model.load_state_dict(initial)
        optimizer.load_state_dict(opt)
        for p, g in zip(params, grads):
            p.grad = g
        for m, f in flags:
            m.training = f
        _restore_native_state(native)
        torch.set_rng_state(rng[0]); random.setstate(rng[1]); np.random.set_state(rng[2])
    out['cpu_seconds'] = max(time.process_time() - t0, 1e-9)
    return out


class Setup:
    """Root-level shared inputs. Mechanistic setup cost is charged to mechanistic arms."""

    def __init__(self, root: int, warm_path: Path):
        rc = self.rc = recipe()
        self.root = root
        views = edge_views(rc.split_seed)
        self.fit = views['fit']
        self.fb_edges = stratified_edge_sample(views['feedback'], rc.feedback_batch_edges,
                                               seed=sccg_seed(root, 'fb_edges'))
        self.warm_path = warm_path
        self.warm = load_checkpoint(warm_path)
        self.warm_sha = model_state_sha256(self.warm)
        t0 = time.process_time()
        self.proj, self.proj_info = discover_slot_projectors(self.warm, root, self.fit, delay=2, dose=rc.dose,
                                                             n_edges=32, cache=None)
        self.mech_setup_cpu = time.process_time() - t0
        self.fb_batch = edge_batch(self.fb_edges, root=root, seed=sccg_seed(root, 'fb_cohort'),
                                   delay=rc.feedback_delay, split='train', noise=rc.noise)
        self.panel = drift_panel(views['feedback'])
        self.panel_batches = [edge_batch((e,), root=root, seed=sccg_seed(root, 'panel_nuisance', i),
                                         delay=max(PANEL_DELAYS), split='train', noise=rc.noise)
                              for i, e in enumerate(self.panel)]

    def cal_loss(self, model, mode):
        rc = self.rc
        if mode == 'ordinary':
            return ordinary_feedback_loss(model, self.fb_batch)
        return mechanistic_feedback_loss(model, self.fb_batch, self.proj, dose=rc.dose, w_ordinary=rc.w_ordinary,
                                         w_target=rc.w_target, w_preserve=rc.w_preserve)


def _edge_seed(e):
    return int.from_bytes(hashlib.sha256(repr(e).encode()).digest()[:4], 'little')


def _batch_xy(b):
    return torch.cat((b.receiver, b.hybrid)), torch.cat((b.labels_r, b.labels_h))


def fresh_cal_grad(model, setup, mode):
    t0 = time.process_time()
    model.eval()
    loss = setup.cal_loss(model, mode)
    g = tuple(None if x is None else x.detach().clone()
              for x in torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True))
    return g, float(loss.detach()), time.process_time() - t0


def run_arm(arm: str, setup: Setup, *, budget: str, limit: float, mms_mib: float = 16.0,
            policy: Policy = Policy(), decisions: list | None = None) -> tuple[dict, torch.nn.Module]:
    rc, root = setup.rc, setup.root
    torch.set_num_threads(rc.threads)
    torch.use_deterministic_algorithms(True)
    model = copy.deepcopy(setup.warm)
    opt = torch.optim.AdamW(model.parameters(), lr=rc.curation_learning_rate, weight_decay=rc.weight_decay)
    cache = MechanisticCache(mms_mib=16.0)
    atlas = MechanismAtlas(mms_mib) if arm in ATLAS_ARMS else None
    mode = 'ordinary' if arm == 'B_ord_grad' else 'mechanistic'
    exposures: Counter = Counter()
    t0 = time.process_time()
    charged_setup = setup.mech_setup_cpu if arm in MECH_ARMS else 0.0
    counters = Counter()
    logs = []
    window_cost = {'audit': 0.0, 'plain': 0.0}
    no_audit_left = 0
    ord_update_cpu = None
    theta_src = None
    w = 0
    overrun = 0.0

    def elapsed():
        return time.process_time() - t0 + charged_setup

    while True:
        if budget == 'updates' and w >= limit:
            break
        audit = arm != 'A_ordinary' and not (arm in ATLAS_ARMS and no_audit_left > 0)
        if budget == 'cpu':
            est = window_cost['audit' if audit else 'plain']
            if elapsed() + est > limit:
                if not audit or arm == 'A_ordinary':
                    break
                # cannot afford an audit: take the affordable ordinary action instead
                audit = False
                counters['unaffordable_audit_fallbacks'] += 1
                if elapsed() + window_cost['plain'] > limit:
                    break
        tw = time.process_time()
        delay = rc.train_delays[w % len(rc.train_delays)]
        if sum(1 for e in setup.fit if exposures[pair_key(e)] < rc.pair_exposure_cap) < rc.proposals_per_window:
            exposures = Counter()
        props = _sample_window_proposals(setup.fit, root, w, rc.proposals_per_window, exposures, rc.pair_exposure_cap)
        ckpt = model_state_sha256(model)
        epoch = f'{arm}:w{w}:{ckpt[:16]}'
        batches = {e: cached_edge_batch((e,), root=root, seed=sccg_seed(root, f'cand_v{w % 4}_d{delay}', _edge_seed(e)),
                                        delay=delay, split='train', noise=rc.noise, cache=cache) for e in props}
        det_cost = float((delay + 5) * (rc.low_cycles + 1)) * 1e-4
        log = dict(window=w, audit=audit, delay=delay)
        g = None
        if not audit:
            rows = [Candidate(e, 1.0, 0.0, det_cost, epoch) for e in props]
            sched = 'uniform'
        else:
            # ---- calibration observer
            if arm in ATLAS_ARMS:
                theta = torch.cat([p.detach().flatten() for p in model.parameters()])
                fp = setup.fb_batch.fingerprint()
                entry = atlas.find(lambda k: k.module == 'ComponentTRM.all_params' and k.objective == 'cal_mechanistic'
                                   and k.input_fingerprint == fp)
                age = counters['updates_committed'] - entry.created_update if entry else 0
                dist = float((theta - theta_src).norm()) if (entry is not None and theta_src is not None) else 0.0
                dec = decide(entry, age=age, param_distance=dist, policy=policy, refresh_value=1.0, refresh_cost=0.0)
                log['atlas_decision'] = dict(action=dec.action, tier=dec.tier, scope=dec.scope, reason=dec.reason)
                if dec.action == 'reuse_pending_verification':
                    g = tuple(torch.as_tensor(entry.payload[f'g{i}'], dtype=torch.float32) for i in range(len(entry.payload) - 1))
                    log['observer'] = 'atlas_reuse'
                    log['observer_age'] = age
                    counters['atlas_reuse'] += 1
                else:
                    g, l0, gcpu = fresh_cal_grad(model, setup, mode)
                    payload = {f'g{i}': x for i, x in enumerate(g)}
                    payload['loss0'] = np.array([l0])
                    key = AtlasKey(source_checkpoint=ckpt, module='ComponentTRM.all_params', site='parameters',
                                   phase='calibration', input_fingerprint=fp, support=('all',),
                                   metric_id='euclidean_parameters', factor_frame=model.factor_frame_digest(),
                                   objective='cal_mechanistic', dtype='float32', dose=str(rc.dose))
                    entry = atlas.insert(key, payload, build_cpu_seconds=gcpu, created_update=counters['updates_committed'])
                    theta_src = theta.clone()
                    log['observer'] = 'refresh'
                    counters['atlas_refresh'] += 1
            else:
                g, g_loss0, _ = _get_cached_cal_grad(model, setup.fb_batch, setup.proj, rc, mode, cache)
                log['observer'] = 'fresh_exact'
            # ---- candidate previews
            rows, meta = [], {}
            for e in props:
                b = batches[e]

                def tr(b=b):
                    x, y = _batch_xy(b)
                    return F.cross_entropy(model(x), y)

                if arm == 'C_probe':
                    pr = probe_update(model, opt, tr, lambda: setup.cal_loss(model, mode), clip=rc.gradient_clip,
                                      cached_cal_grad=g, cached_cal_loss0=g_loss0)
                else:
                    pr = preview_update(model, opt, tr, g, clip=rc.gradient_clip)
                rows.append(Candidate(e, pr['train_loss'], pr['predicted_gain'], det_cost, epoch))
                meta[e] = pr
                counters['previews'] += 1
            sched = {'B_ord_grad': 'mechanistic', 'C_mech_fresh': 'mechanistic', 'C_probe': 'mechanistic',
                     'D_atlas': 'mechanistic', 'E_shuffled': 'shuffled', 'F_obs_only': 'uniform'}[arm]
            preds = sorted((r.predicted_gain for r in rows), reverse=True)
            log['pred_top4_mean'] = float(np.mean(preds[:rc.selected_per_window]))
            log['pred_all_mean'] = float(np.mean(preds))
        selected, new_exp = select_candidates(rows, fit_edges=setup.fit, epoch=epoch, mode=sched,
                                              count=rc.selected_per_window, seed=sccg_seed(root, 'sched'), window=w,
                                              exposures=exposures, cap=rc.pair_exposure_cap, exploration=rc.exploration)
        exposures = Counter(new_exp)
        sel = tuple(e for e, _ in selected)
        if not sel:
            w += 1
            continue
        xs, ys = zip(*(_batch_xy(batches[e]) for e in sel))
        x_mb, y_mb = torch.cat(xs), torch.cat(ys)
        gate = audit and arm != 'F_obs_only'
        verify = audit  # F does the verification work but never acts on it
        snap = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())) if gate else None
        old = [p.detach().clone() for p in model.parameters()] if verify else None
        if verify:
            model.eval()
            with torch.no_grad():
                cal_before = float(setup.cal_loss(model, mode))
        tc = time.process_time()
        model.train()
        loss = F.cross_entropy(model(x_mb), y_mb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), rc.gradient_clip)
        opt.step()
        commit_cpu = time.process_time() - tc
        accepted = True
        if verify:
            model.eval()
            with torch.no_grad():
                cal_after = float(setup.cal_loss(model, mode))
                pred_commit = -sum(float((a.double() * (p.detach() - b).double()).sum())
                                   for a, p, b in zip(g, model.parameters(), old) if a is not None)
            realized = cal_before - cal_after
            log.update(cal_before=cal_before, cal_after=cal_after, pred_commit=pred_commit, realized_commit=realized)
            if arm in ATLAS_ARMS and entry is not None:
                resid = abs(pred_commit - realized) / max(abs(pred_commit), abs(realized), 1e-3)
                entry.trace_residuals.append(resid)
                atlas.record_validation(entry, dict(destination_checkpoint=ckpt, trace_residual=resid,
                                                    realized=realized, predicted=pred_commit))
                log['trace_residual'] = resid
            if gate and cal_after > cal_before + 1e-6:
                model.load_state_dict(snap[0]); opt.load_state_dict(snap[1])
                accepted = False
        counters['accepted' if accepted else 'rejected'] += 1
        if accepted:
            counters['updates_committed'] += 1
        log.update(accepted=accepted, train_loss=float(loss.detach()), selected=[repr(e) for e in sel])
        # ---- value-of-audit rule (atlas arms): first-order, assumes additive per-window gains.
        # The alternative is priced as full plain windows (proposal, batches, update).
        wcpu = time.process_time() - tw
        kind = 'audit' if audit else 'plain'
        window_cost[kind] = max(window_cost[kind] * .5, wcpu)
        if not audit:
            counters['no_audit_windows'] += 1
            ord_update_cpu = wcpu if ord_update_cpu is None else .8 * ord_update_cpu + .2 * wcpu
        if audit and arm in ATLAS_ARMS:
            ord_cpu = ord_update_cpu if ord_update_cpu is not None else commit_cpu
            n_extra = wcpu / max(ord_cpu, 1e-6) - 1.0
            benefit = log['pred_top4_mean'] - log['pred_all_mean']
            alt = n_extra * max(log['pred_all_mean'], 0.0)
            log.update(audit_benefit=benefit, audit_alternative=alt, audit_window_cpu=wcpu, ordinary_cpu=ord_cpu)
            if benefit <= alt:
                no_audit_left = NO_AUDIT_WINDOWS
                counters['no_audit_triggers'] += 1
        elif no_audit_left > 0:
            no_audit_left -= 1
        if decisions is not None and audit and arm in ('C_mech_fresh', 'D_atlas'):
            sel_map = dict(selected)
            for e in props:
                pm = meta[e]
                decisions.append(dict(
                    root=root, arm=arm, window=w, delay=delay, checkpoint_sha256=ckpt, split='train',
                    edge=[list(e[0]), list(e[1]), e[2]], selected=e in sel_map,
                    reason=sel_map.get(e, 'not_selected'), window_accepted=accepted,
                    predicted_gain=pm['predicted_gain'], update_norm=pm['update_norm'],
                    preview_cpu_seconds=pm['cpu_seconds'], observer=log.get('observer'),
                    observer_age=log.get('observer_age', 0), atlas_decision=log.get('atlas_decision'),
                    window_realized_gain=log.get('realized_commit'), window_predicted_gain=log.get('pred_commit'),
                    scope=('HEURISTIC_PROPOSAL' if log.get('observer') == 'atlas_reuse' else 'EXACT_CURRENT_OBSERVER'),
                    verifier='committed_update_acceptance', dose=rc.dose))
        log['cpu'] = elapsed()
        logs.append(log)
        w += 1
    total = elapsed()
    if budget == 'cpu' and total > limit:
        overrun = total - limit
    res = dict(arm=arm, budget=budget, limit=limit, windows=w, cpu_seconds=total, charged_setup_cpu=charged_setup,
               overrun_cpu=overrun, counters=dict(counters), final_state_sha256=model_state_sha256(model),
               cache=cache.stats(), atlas=atlas.stats() if atlas else None, logs=logs)
    return res, model


def evaluate(model, setup: Setup) -> dict:
    rc, root = setup.rc, setup.root
    ev = eval_model_ordinary(model, root, rc, split='eval', domain_prefix='sccg10_eng_eval')
    summary = {'delays': {k: {f: v[f] for f in ('accuracy', 'correct_count', 'n', 'cross_entropy')}
                          for k, v in ev['delays'].items()},
               'binding': {k: {f: v[f] for f in ('accuracy', 'correct_count', 'n')} for k, v in ev['binding'].items()}}
    model.eval()
    with torch.no_grad():
        summary['feedback_ordinary_loss'] = float(setup.cal_loss(model, 'ordinary'))
        summary['feedback_mechanistic_loss'] = float(setup.cal_loss(model, 'mechanistic'))
    # Privileged mechanism diagnostic on the TRAIN panel (matched-dose hybrid transport).
    flips = elig = cnum = cden = 0
    ferr = []
    for b in setup.panel_batches:
        obs = observe_edge(model, b, PANEL_DELAYS, dose=rc.dose)
        idx = geo.row_index(b, PANEL_DELAYS)
        with torch.no_grad():
            clean = geo.native_logits(model, b, PANEL_DELAYS, obs['post_write_state'])
        out = geo.finite_outcome(model, b, PANEL_DELAYS, obs['post_write_state'], idx, clean,
                                 delta_state=obs['hybrid_direction'])
        flips += out['target_flipped']; elig += out['target_eligible']
        cnum += out['corruption_numerator']; cden += out['corruption_denominator']
        ferr.append(obs['finite_dose_error'])
    summary['panel_hybrid_matched'] = dict(flips=[flips, elig], corruption=[cnum, cden],
                                           finite_error_mean=float(np.mean(ferr)), dose=rc.dose)
    return summary


def run_root(root: int, warm: Path, out: Path, *, arms=ARMS, cpu_limit=100.0, windows=80,
             atlas_tiers=(0.0, 4.0, 16.0, 64.0)) -> dict:
    setup = Setup(root, warm)
    out.mkdir(parents=True, exist_ok=True)
    rep = dict(root=root, warm_state_sha256=setup.warm_sha, mech_setup_cpu=setup.mech_setup_cpu,
               cpu_limit=cpu_limit, windows=windows, results={}, source_hashes=source_hashes())
    decisions: list = []
    jobs = [('updates', windows, a, 16.0) for a in arms]
    jobs += [('cpu', cpu_limit, a, 16.0) for a in arms if a != 'C_probe']
    jobs += [('cpu', cpu_limit, 'D_atlas', t) for t in atlas_tiers if t != 16.0]
    for budget, limit, arm, mib in jobs:
        tag = f'{budget}_{arm}' + (f'_mms{int(mib)}' if arm in ATLAS_ARMS else '')
        res, model = run_arm(arm, setup, budget=budget, limit=limit, mms_mib=mib,
                             decisions=decisions if budget == 'cpu' and mib == 16.0 else None)
        res['eval'] = evaluate(model, setup)
        res['mms_mib'] = mib
        torch.save(model.state_dict(), out / f'{tag}.pt')
        rep['results'][tag] = res
        e = res['eval']
        print(json.dumps({'root': root, 'tag': tag, 'windows': res['windows'], 'cpu': round(res['cpu_seconds'], 1),
                          'acc': res['counters'].get('accepted', 0), 'd31': e['delays']['31']['accuracy'],
                          'd63': e['delays']['63']['accuracy'], 'sha': res['final_state_sha256'][:12]}), flush=True)
        dump_json(out / 'economics.json', rep)
    with open(out / 'train_decisions.jsonl', 'w', encoding='utf-8', newline='\n') as fh:
        for d in decisions:
            fh.write(json.dumps(d, sort_keys=True) + '\n')
    return rep


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=int, required=True)
    p.add_argument('--warm', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cpu-limit', type=float, default=100.0)
    p.add_argument('--windows', type=int, default=80)
    p.add_argument('--arms', type=str, default=','.join(ARMS))
    a = p.parse_args()
    if a.root not in ENGINEERING_ROOTS:
        raise SystemExit('economics runs before the freeze are restricted to engineering roots')
    cap = ProcessTreeMemoryCap(DEFAULT_CAP_BYTES)
    with cap:
        rep = run_root(a.root, a.warm, a.out, arms=tuple(a.arms.split(',')), cpu_limit=a.cpu_limit, windows=a.windows)
        rep['memory'] = accounting_dict(cap)
        dump_json(a.out / 'economics.json', rep)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
