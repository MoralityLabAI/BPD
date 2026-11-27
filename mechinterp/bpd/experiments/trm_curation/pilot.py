"""CURATE-03B paired native pilot: analytical screening, paid acquisition, joint hill climbing.

Arms (CURATE_03B_PILOT_PROTOCOL.md):
  A uniform CPU reinvestment          F early-12 acquisition gate
  B original transactional selector   G joint hill climbing + full-16 gate
  C analytical ordinary selector      H joint hill climbing + early-12 gate
  D analytical mechanistic selector   I matched shuffled control (D with shuffled scores)
  E analytical + full-16 abstention   J observer-only control (D scoring, uniform pick)

Budget: each arm starts from the identical warmup checkpoint with a fresh
AdamW (curation lr) and its own exposure ledger and MMS cache, and runs
windows until its process CPU (all threads, ``time.process_time``) reaches the
budget. The check happens before each window; a started window finishes and
the overshoot is recorded. Evaluation is not charged. Everything else is.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from .batch_screen import mean_gradient
from .cache import MechanisticCache
from .campaign import (
    CampaignRecipe, _make_trm, _sample_window_proposals, dump_json, eval_model_ordinary,
    file_sha256, train_warmup_model,
)
from .custody import custody_digest
from .data import edge_batch
from .memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap
from .native_screen import (
    STAGES, Ledger, NativeScreen, acquisition_order, audit_draw, exposure_legal, gate_decision,
)
from .reference import semantic_edges
from .scheduler import Candidate, edge_views, pair_key, select_candidates
from .signal_bank import (
    cached_edge_batch, discover_slot_projectors, eval_state_direction, model_state_sha256,
    seed_for, stratified_edge_sample,
)
from .transaction import probe_update

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
EXPERIMENT = 'BPD-TRM-CURATE-03B'
ENGINEERING_ROOTS = (28000, 28001)
FRESH_DEVELOPMENT_ROOTS = tuple(range(28301, 28309))
AUDIT_PROBABILITY = 0.25
BUDGET_CPU_SECONDS = 100.0
MAX_WINDOWS = 20000

_D = dict(kind='scored', score='analytic', objective='mechanistic', gate=None, joint_rounds=0,
          accept=True, sched='mechanistic')
ARM_SPECS: dict[str, dict] = {
    'A_uniform_reinvest': dict(kind='uniform', accept=False, sched='uniform'),
    'B_transactional': {**_D, 'score': 'transactional'},
    'C_analytic_ord': {**_D, 'objective': 'ordinary'},
    'D_analytic_mech': dict(_D),
    'E_full16_gate': {**_D, 'gate': 'full16'},
    'F_early12_gate': {**_D, 'gate': 'early12'},
    'G_joint_full16': {**_D, 'gate': 'full16', 'joint_rounds': 1},
    'H_joint_early12': {**_D, 'gate': 'early12', 'joint_rounds': 1},
    'I_shuffled': {**_D, 'sched': 'shuffled'},
    'J_observer_only': {**_D, 'accept': False, 'sched': 'uniform'},
}
ARMS = tuple(ARM_SPECS)


def source_hashes() -> dict[str, str]:
    names = ('adamw_screen.py', 'batch_screen.py', 'custody.py', 'native_screen.py', 'pilot.py',
             'campaign.py', 'scheduler.py', 'signal_bank.py', 'transaction.py', 'model.py',
             'data.py', 'cache.py', 'memory_cap.py', 'reference.py')
    return {n: file_sha256(HERE / n) for n in names}


def _rss_tree() -> int:
    p = psutil.Process()
    total = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            total += c.memory_info().rss
        except psutil.Error:
            pass
    return total


def selective_transport_eval(model, root: int, recipe: CampaignRecipe, n_edges: int = 48) -> dict:
    """Oracle hybrid-displacement transport at delay 31 on EVAL edges, steering otherwise off."""
    model.eval()
    sample = stratified_edge_sample(semantic_edges('eval'), n_edges, seed=seed_for(root, 'c03b_transport', 0))
    hits = corrupted = init_ok = 0
    with torch.no_grad():
        for i, e in enumerate(sample):
            b = edge_batch((e,), root=root, seed=seed_for(root, 'c03b_transport_b', i),
                           delay=recipe.primary_delay, split='eval', noise=recipe.noise)
            sr = model.prefix_state(b.receiver[:1]).double().reshape(-1)
            sh = model.prefix_state(b.hybrid[:1]).double().reshape(-1)
            ev = eval_state_direction(model, b, (sh - sr).numpy(), recipe.primary_delay)
            hits += int(ev['target_hit'])
            corrupted += ev['prot_corrupted']
            init_ok += ev['prot_init_correct']
    return {'n_edges': len(sample), 'target_hits': hits, 'transport_accuracy': hits / len(sample),
            'protected_initially_correct': init_ok, 'protected_corrupted': corrupted,
            'protected_corruption_rate': corrupted / max(init_ok, 1)}


def evaluate_arm(model, root: int, recipe: CampaignRecipe) -> dict:
    ev = eval_model_ordinary(model, root, recipe, split='eval', delays=recipe.eval_delays,
                             eval_n=recipe.eval_n, binding_n=recipe.binding_n, domain_prefix='dev_eval')
    for d in ev['delays'].values():
        for k in ('predictions', 'labels', 'correct'):
            d.pop(k, None)
    ev['selective_transport'] = selective_transport_eval(model, root, recipe)
    return ev


class ArmRunner:
    """One arm on one root from the shared warmup checkpoint."""

    def __init__(self, arm: str, root: int, warm, recipe: CampaignRecipe, slot_projectors,
                 fb_edges, fit_edges, *, mms_mib: float, budget_cpu: float | None,
                 fixed_windows: int | None, joint_rounds: int | None = None,
                 record_rows: bool = True):
        self.arm, self.root, self.recipe = arm, root, recipe
        self.spec = dict(ARM_SPECS[arm])
        if joint_rounds is not None and self.spec.get('joint_rounds'):
            self.spec['joint_rounds'] = joint_rounds
        self.model = copy.deepcopy(warm)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=recipe.curation_learning_rate,
                                     weight_decay=recipe.weight_decay)
        self.cache = MechanisticCache(mms_mib=mms_mib)
        self.ledger = Ledger()
        self.screen = NativeScreen(self.model, self.opt, recipe, slot_projectors,
                                   cache=self.cache, ledger=self.ledger)
        self.fb_edges, self.fit_edges = fb_edges, fit_edges
        self.budget_cpu, self.fixed_windows = budget_cpu, fixed_windows
        if (budget_cpu is None) == (fixed_windows is None):
            raise ValueError('exactly one of budget_cpu / fixed_windows')
        self.selection_exposure: Counter = Counter()
        self.committed_exposure: Counter = Counter()
        self.proposal_exposure: Counter = Counter()
        self.exposure_resets = 0
        self.rows: list[dict] = []
        self.windows: list[dict] = []
        self.record_rows = record_rows

    # ------------------------------------------------------------------
    def run(self) -> dict:
        torch.set_num_threads(self.recipe.threads)
        t0, w0 = time.process_time(), time.perf_counter()
        win = 0
        peak_rss = _rss_tree()
        while win < MAX_WINDOWS:
            spent = time.process_time() - t0
            if self.budget_cpu is not None and spent >= self.budget_cpu:
                break
            if self.fixed_windows is not None and win >= self.fixed_windows:
                break
            c0 = time.process_time()
            rec = self._window(win)
            rec['cpu_seconds'] = time.process_time() - c0
            rec['cumulative_cpu_seconds'] = time.process_time() - t0
            if win % 16 == 0:
                peak_rss = max(peak_rss, _rss_tree())
                rec['tree_rss_bytes'] = peak_rss
            self.windows.append(rec)
            win += 1
        total = time.process_time() - t0
        peak_rss = max(peak_rss, _rss_tree())
        acc = sum(1 for w in self.windows if w.get('committed'))
        cats = dict(self.ledger.cpu)
        out = {
            'arm': self.arm, 'spec': self.spec, 'root': self.root,
            'windows': len(self.windows), 'committed_updates': acc,
            'abstained_windows': sum(1 for w in self.windows if w.get('decision') == 'ABSTAIN'),
            'rejected_windows': sum(1 for w in self.windows
                                    if w.get('attempted') and not w.get('committed')),
            'candidates_acquired': sum(w.get('acquired', 0) for w in self.windows),
            'audits': sum(1 for w in self.windows if w.get('audit')),
            'cpu_seconds': total, 'wall_seconds': time.perf_counter() - w0,
            'budget_cpu_seconds': self.budget_cpu,
            'overshoot_cpu_seconds': (total - self.budget_cpu) if self.budget_cpu else None,
            'fixed_windows': self.fixed_windows,
            'cpu_by_category': cats,
            'cpu_inclusive_by_category': dict(self.ledger.inclusive),
            'cpu_unattributed': total - sum(cats.values()),
            'counts_by_category': dict(self.ledger.counts),
            'custody_stats': dict(self.screen.custody.stats),
            'cache_stats': self.cache.stats(),
            'peak_tree_rss_bytes': peak_rss,
            'exposure_resets': self.exposure_resets,
            'distinct_selected_pairs': len(self.selection_exposure),
            'distinct_committed_pairs': len(self.committed_exposure),
            'final_state_sha256': model_state_sha256(self.model),
            'final_custody_digest': custody_digest(self.model, self.opt, include_rng=False),
        }
        return out

    # ------------------------------------------------------------------
    def _materialize(self, edges, delay, win):
        nv = win % 4
        with self.ledger.timed('batch_materialization'):
            return {e: cached_edge_batch(
                (e,), root=self.root,
                seed=seed_for(self.root, f'cand_v{nv}_d{delay}',
                              int.from_bytes(hashlib.sha256(repr(e).encode()).digest()[:4], 'little')),
                delay=delay, split='train', noise=self.recipe.noise, cache=self.cache) for e in edges}

    def _window(self, win: int) -> dict:
        r = self.recipe
        delay = r.train_delays[win % len(r.train_delays)]
        with self.ledger.timed('proposal'):
            if sum(1 for e in self.fit_edges
                   if self.selection_exposure[pair_key(e)] < r.pair_exposure_cap) < r.proposals_per_window:
                self.selection_exposure = Counter()
                self.exposure_resets += 1
            props = _sample_window_proposals(self.fit_edges, self.root, win, r.proposals_per_window,
                                             self.selection_exposure, r.pair_exposure_cap)
            order = acquisition_order(props, split_seed=r.split_seed, root=self.root, window=win)
            for e in props:
                self.proposal_exposure[pair_key(e)] += 1
        exposures_before = Counter(self.selection_exposure)
        rec: dict[str, Any] = {'window': win, 'delay': delay,
                               'acquisition_order': [_edge_json(e) for e in order]}
        if self.spec['kind'] == 'uniform':
            # The uniform policy reads no checkpoint; skip provenance hashing it would not pay for.
            epoch = f'w{win}'
        else:
            with self.ledger.timed('provenance'):
                ckpt = model_state_sha256(self.model)
                epoch = f'w{win}:{ckpt[:16]}'
                rec['checkpoint_sha256'] = ckpt
                if self.record_rows:
                    rec['optimizer_digest'] = custody_digest(self.model, self.opt, include_rng=False)
                    rec['selection_exposure_total'] = sum(self.selection_exposure.values())
        det_cost = float((delay + 5) * (r.low_cycles + 1)) * 1e-4
        sched_seed = seed_for(self.root, 'sched', 0)

        if self.spec['kind'] == 'uniform':
            with self.ledger.timed('controller'):
                cands = [Candidate(e, 1.0, 0.0, det_cost, epoch) for e in props]
                sel, new_exp = select_candidates(cands, fit_edges=self.fit_edges, epoch=epoch, mode='uniform',
                                                 count=r.selected_per_window, seed=sched_seed, window=win,
                                                 exposures=self.selection_exposure, cap=r.pair_exposure_cap,
                                                 exploration=r.exploration)
            edges = tuple(e for e, _ in sel)
            batches = self._materialize(edges, delay, win)
            info = self.screen.step_minibatch([batches[e] for e in edges], accept_test=False)
            self.selection_exposure = Counter(new_exp)
            self._commit_exposure(edges, info['accepted'])
            rec.update(decision='REINVEST_CPU', acquired=0, attempted=True, committed=info['accepted'],
                       selected=[_edge_json(e) for e in edges], train_loss=info['train_loss'])
            return rec

        spec = self.spec
        batches = self._materialize(props, delay, win)
        fb_batch = cached_edge_batch(self.fb_edges, root=self.root,
                                     seed=seed_for(self.root, f'fb_cohort_d{r.feedback_delay}', 0),
                                     delay=r.feedback_delay, split='train', noise=r.noise, cache=self.cache)
        fb = self.screen.feedback_gradient(fb_batch, spec['objective'])
        rec['feedback'] = {'objective': spec['objective'], 'loss0': fb.loss0, 'cache_hit': fb.cache_hit,
                           'cpu_seconds': fb.cpu_seconds}

        scores: dict = {}
        cand_info: dict = {}
        grads: dict = {}

        def acquire(e):
            if spec['score'] == 'transactional':
                b = batches[e]
                with self.ledger.timed('transactional_probe'):
                    pr = probe_update(self.model, self.opt,
                                      lambda b=b: F.cross_entropy(self.model(_x(b)), _y(b)),
                                      lambda: self.screen.objective_loss(fb_batch, spec['objective']),
                                      clip=r.gradient_clip, cached_cal_grad=fb.grads,
                                      cached_cal_loss0=fb.loss0)
                scores[e] = pr['predicted_gain']
                cand_info[e] = {'predicted_gain': pr['predicted_gain'], 'realized_gain': pr['realized_gain'],
                                'train_loss': pr['train_loss'], 'update_norm': pr['update_norm'],
                                'label_source': 'transactional_probe'}
            else:
                g, loss = self.screen.candidate_gradient(batches[e])
                s = self.screen.analytic_score(g, fb)
                grads[e] = g
                scores[e] = s['predicted_gain']
                cand_info[e] = {'predicted_gain': s['predicted_gain'], 'realized_gain': None,
                                'train_loss': loss, 'update_norm': s['update_norm'],
                                'gradient_norm': s['gradient_norm'], 'clip_scale': s['clip_scale'],
                                'label_source': 'missing_unaudited'}

        acquired: list = []
        abstain = False
        stage_log = []
        for stage in STAGES:
            for e in order[len(acquired):stage]:
                acquire(e)
                acquired.append(e)
            prefix_scores = [scores[e] for e in acquired]
            stage_log.append({'stage': stage, 'max_score': max(prefix_scores),
                              'n_positive': sum(s > 0 for s in prefix_scores)})
            with self.ledger.timed('controller'):
                if spec['gate'] == 'early12' and stage == 12 and not any(s > 0 for s in prefix_scores):
                    abstain = True
                    break
        with self.ledger.timed('controller'):
            if not abstain and spec['gate'] in ('full16', 'early12'):
                abstain = gate_decision([scores[e] for e in acquired], 'full16')[1]
        rec['stages'] = stage_log
        rec['acquired'] = len(acquired)
        rec['decision'] = 'ABSTAIN' if abstain else f'ACQUIRE_{len(acquired)}'

        audit = None
        if abstain:
            do_audit, u = audit_draw(self.root, self.arm, win, AUDIT_PROBABILITY)
            rec['audit_draw_u'] = u
            rec['audit_probability'] = AUDIT_PROBABILITY
            if do_audit:
                with self.ledger.timed('audit'):
                    for e in order[len(acquired):]:
                        acquire(e)
                    audit = self._select_and_step(order, scores, grads, batches, fb, fb_batch, win, epoch,
                                                  det_cost, exposures_before, commit_allowed=False)
                rec['audit'] = {k: v for k, v in audit.items() if k not in ('edges',)}
                rec['audit']['selected'] = [_edge_json(e) for e in audit['edges']]
            rec.update(attempted=False, committed=False)
        else:
            res = self._select_and_step(order, scores, grads, batches, fb, fb_batch, win, epoch,
                                        det_cost, exposures_before, commit_allowed=True)
            edges = res.pop('edges')
            rec.update(res)
            rec['selected'] = [_edge_json(e) for e in edges]
            rec['attempted'] = True
            rec['committed'] = res['accepted']
            self._commit_exposure(edges, res['accepted'])

        if self.record_rows:
            final = set(map(tuple, map(_edge_key, rec.get('selected', []))))
            for pos, e in enumerate(order):
                info = cand_info.get(e)
                self.rows.append({
                    'window': win, 'edge': _edge_json(e), 'slot': e[2], 'acquisition_position': pos,
                    'stage': STAGES[pos // 4], 'acquired': e in acquired,
                    'acquired_for_audit': info is not None and e not in acquired,
                    **({} if info is None else info),
                    'in_final_batch': _edge_key(_edge_json(e)) in final,
                })
        return rec

    def _select_and_step(self, order, scores, grads, batches, fb, fb_batch, win, epoch, det_cost,
                         exposures_before, *, commit_allowed: bool) -> dict:
        r, spec = self.recipe, self.spec
        with self.ledger.timed('controller'):
            rows = [Candidate(e, 1.0, scores[e], det_cost, epoch) for e in order]
            sel, new_exp = select_candidates(rows, fit_edges=self.fit_edges, epoch=epoch, mode=spec['sched'],
                                             count=r.selected_per_window, seed=seed_for(self.root, 'sched', 0),
                                             window=win, exposures=exposures_before, cap=r.pair_exposure_cap,
                                             exploration=r.exploration)
        edges = tuple(e for e, _ in sel)
        reasons = {e: why for e, why in sel}
        out: dict[str, Any] = {'initial_selected': [_edge_json(e) for e in edges],
                               'selection_reasons': [reasons[e] for e in edges]}
        aggregated = None
        if spec['joint_rounds'] and grads:
            idx = {e: i for i, e in enumerate(order)}
            legal_edges = exposure_legal(exposures_before, r.pair_exposure_cap)
            res = self.screen.joint_search(
                [grads[e] for e in order], fb, [idx[e] for e in edges],
                pinned_idx=[idx[e] for e in edges if reasons[e] == 'explore'],
                legal=lambda batch: legal_edges([order[i] for i in batch]),
                rounds=spec['joint_rounds'])
            new_edges = tuple(order[i] for i in res['batch'])
            out.update(predicted_joint_gain_initial=res['initial_gain'],
                       predicted_joint_gain_final=res['predicted_gain'],
                       scored_batches=res['scored_batches'],
                       swaps=sorted(set(map(_edge_key, map(_edge_json, new_edges))) -
                                    set(map(_edge_key, map(_edge_json, edges)))))
            edges = new_edges
            new_exp = exposures_before + Counter(pair_key(e) for e in edges)
        if grads and all(e in grads for e in edges):
            with self.ledger.timed('verification'):
                aggregated = mean_gradient([grads[e] for e in edges])
        info = self.screen.step_minibatch(
            [batches[e] for e in edges], fb=fb if spec['accept'] or aggregated is not None else None,
            fb_batch=fb_batch, accept_test=spec['accept'], commit_allowed=commit_allowed,
            aggregated=aggregated)
        if commit_allowed:
            self.selection_exposure = Counter(new_exp)
        out.update(info)
        out['edges'] = edges
        return out

    def _commit_exposure(self, edges, committed: bool) -> None:
        if committed:
            for e in edges:
                self.committed_exposure[pair_key(e)] += 1


def _x(b):
    return torch.cat((b.receiver, b.hybrid), dim=0)


def _y(b):
    return torch.cat((b.labels_r, b.labels_h), dim=0)


def _edge_json(e):
    return [list(e[0]), list(e[1]), e[2]]


def _edge_key(ej):
    return (tuple(ej[0]), tuple(ej[1]), ej[2])


# ----------------------------------------------------------------------
def prepare_root(root: int, recipe: CampaignRecipe, out_dir: Path, *, warm_path: Path | None = None) -> dict:
    """Shared warmup checkpoint, slot projectors and feedback edges for one root."""
    views = edge_views(recipe.split_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    warm = _make_trm(recipe)
    warm_info: dict[str, Any] = {}
    target = out_dir / 'shared_warmup.pt'
    if warm_path is not None:
        warm.load_state_dict(torch.load(warm_path, map_location='cpu', weights_only=True))
        warm_info = {'source': str(warm_path.relative_to(REPO)).replace('\\', '/'),
                     'source_sha256': file_sha256(warm_path)}
    elif target.is_file():
        warm.load_state_dict(torch.load(target, map_location='cpu', weights_only=True))
        warm_info = {'source': 'reused_existing_shared_warmup'}
    else:
        torch.manual_seed(seed_for(root, 'init_TRM_w32_r29', 0))
        warm = _make_trm(recipe)
        warm_info = train_warmup_model(warm, 'TRM_w32_r29', root, views['fit'], recipe,
                                       log_every=recipe.warmup_updates)
        torch.save(warm.state_dict(), target)
    if not target.is_file():
        torch.save(warm.state_dict(), target)
    projectors, proj_info = discover_slot_projectors(warm, root, views['fit'], delay=2, dose=recipe.dose,
                                                     n_edges=32, cache=None)
    fb_edges = stratified_edge_sample(views['feedback'], recipe.feedback_batch_edges,
                                      seed=seed_for(root, 'fb_edges', 0))
    return {'warm': warm, 'warm_sha256': model_state_sha256(warm), 'warm_info': warm_info,
            'projectors': projectors, 'projector_info': proj_info, 'fb_edges': fb_edges,
            'fit_edges': views['fit']}


def run_root(root: int, recipe: CampaignRecipe, out_dir: Path, *, arms=ARMS, mms_mib: float,
             budget_cpu: float, warm_path: Path | None, joint_rounds: int,
             observer_parity_windows: int = 40) -> dict:
    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    prep = prepare_root(root, recipe, out_dir, warm_path=warm_path)
    warm_eval = evaluate_arm(copy.deepcopy(prep['warm']), root, recipe)
    results: dict[str, Any] = {}
    for arm in arms:
        runner = ArmRunner(arm, root, prep['warm'], recipe, prep['projectors'], prep['fb_edges'],
                           prep['fit_edges'], mms_mib=mms_mib, budget_cpu=budget_cpu, fixed_windows=None,
                           joint_rounds=joint_rounds)
        summary = runner.run()
        summary['eval'] = evaluate_arm(runner.model, root, recipe)
        torch.save(runner.model.state_dict(), out_dir / f'{arm}.pt')
        _write_jsonl(out_dir / f'{arm}_windows.jsonl', runner.windows)
        _write_jsonl(out_dir / f'{arm}_candidates.jsonl', runner.rows)
        results[arm] = summary
        ev = summary['eval']
        print(json.dumps({'root': root, 'arm': arm, 'windows': summary['windows'],
                          'committed': summary['committed_updates'], 'abstained': summary['abstained_windows'],
                          'cpu': round(summary['cpu_seconds'], 1),
                          'd31': ev['delays'][str(recipe.primary_delay)]['accuracy'],
                          'd127': ev['delays']['127']['accuracy']}), flush=True)
    parity = None
    if observer_parity_windows and {'A_uniform_reinvest', 'J_observer_only'} <= set(arms):
        parity = fixed_update_pair(root, prep, recipe, 'A_uniform_reinvest', 'J_observer_only',
                                   observer_parity_windows, mms_mib)
    result = {
        'experiment': EXPERIMENT, 'evidence_class': 'DEVELOPMENT_ONLY', 'root': root,
        'shared_warmup_sha256': prep['warm_sha256'], 'warmup': _strip_curve(prep['warm_info']),
        'warmup_eval': warm_eval, 'projector_info': prep['projector_info'],
        'budget_cpu_seconds': budget_cpu, 'mms_mib': mms_mib, 'joint_rounds': joint_rounds,
        'arms': results, 'observer_fixed_update_parity': parity,
        'source_sha256': source_hashes(),
        'torch': torch.__version__, 'python': sys.version.split()[0], 'platform': platform.platform(),
    }
    dump_json(out_dir / 'result.json', result)
    dump_json(out_dir / 'artifact_hashes.json',
              {p.name: file_sha256(p) for p in sorted(out_dir.iterdir())
               if p.is_file() and p.name != 'artifact_hashes.json'})
    return result


def fixed_update_pair(root, prep, recipe, arm_a, arm_b, windows, mms_mib) -> dict:
    """Fixed-window runs whose checkpoints must be byte-identical (observer-only invariance)."""
    out = {}
    for arm in (arm_a, arm_b):
        rn = ArmRunner(arm, root, prep['warm'], recipe, prep['projectors'], prep['fb_edges'],
                       prep['fit_edges'], mms_mib=mms_mib, budget_cpu=None, fixed_windows=windows,
                       record_rows=False)
        s = rn.run()
        out[arm] = {'final_state_sha256': s['final_state_sha256'],
                    'final_custody_digest': s['final_custody_digest'],
                    'selections': [w.get('selected') for w in rn.windows], 'cpu_seconds': s['cpu_seconds']}
    a, b = out[arm_a], out[arm_b]
    return {'windows': windows, 'arms': [arm_a, arm_b],
            'state_identical': a['final_state_sha256'] == b['final_state_sha256'],
            'custody_identical': a['final_custody_digest'] == b['final_custody_digest'],
            'selections_identical': a['selections'] == b['selections'],
            'final_state_sha256': {arm_a: a['final_state_sha256'], arm_b: b['final_state_sha256']},
            'cpu_seconds': {arm_a: a['cpu_seconds'], arm_b: b['cpu_seconds']}}


def _strip_curve(info: dict) -> dict:
    return {k: v for k, v in info.items() if k not in ('curve',)}


def _write_jsonl(path: Path, rows) -> None:
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--roots', type=int, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--arms', nargs='+', default=list(ARMS))
    ap.add_argument('--budget-cpu', type=float, default=BUDGET_CPU_SECONDS)
    ap.add_argument('--mms-mib', type=float, default=16.0)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--joint-rounds', type=int, default=1)
    ap.add_argument('--observer-parity-windows', type=int, default=40)
    ap.add_argument('--engineering-warmup', action='store_true',
                    help='Load CURATE-02 phase_a TRM_w32_r29 checkpoints (spent engineering roots)')
    ap.add_argument('--warmup-only', action='store_true',
                    help='Train and save shared warmup checkpoints, run no arm')
    ap.add_argument('--no-cap', action='store_true')
    args = ap.parse_args()
    recipe = CampaignRecipe(threads=args.threads, mms_mib=args.mms_mib)
    cap = None if args.no_cap else ProcessTreeMemoryCap(DEFAULT_CAP_BYTES)
    if cap is not None:
        cap.__enter__()
    try:
        for root in args.roots:
            wp = None
            if args.engineering_warmup:
                wp = HERE / 'results' / 'engineering' / 'phase_a' / f'root_{root}' / 'TRM_w32_r29.pt'
            if args.warmup_only:
                torch.set_num_threads(recipe.threads)
                torch.use_deterministic_algorithms(True)
                prep = prepare_root(root, recipe, args.out / f'root_{root}', warm_path=wp)
                dump_json(args.out / f'root_{root}' / 'warmup.json',
                          {'root': root, 'shared_warmup_sha256': prep['warm_sha256'],
                           'warmup': _strip_curve(prep['warm_info'])})
                continue
            run_root(root, recipe, args.out / f'root_{root}', arms=tuple(args.arms), mms_mib=args.mms_mib,
                     budget_cpu=args.budget_cpu, warm_path=wp, joint_rounds=args.joint_rounds,
                     observer_parity_windows=args.observer_parity_windows)
    finally:
        if cap is not None:
            acc = cap.accounting()
            dump_json(args.out / f'memory_accounting_{"_".join(map(str, args.roots))}.json', asdict(acc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
