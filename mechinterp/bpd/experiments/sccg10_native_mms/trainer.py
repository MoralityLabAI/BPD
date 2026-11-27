"""Workstreams D/E: transactional compute-reinvestment trainer, paired arms A-F.

A  ordinary      uniform proposals, every window commits (full reinvestment).
B  ord_curation  ordinary TRAIN-feedback gradient, analytic AdamW previews,
                 transactional acceptance; same rate controller as C/D/E.
C  mech_fresh    mechanism refreshed (observation + support search) at every
                 curated window; only exact same-checkpoint memoization.
D  mech_atlas    mechanism from the cross-checkpoint atlas through the admission
                 ladder; refresh on failure; same controller.
E  mech_shuffled D with directions deranged across feedback edges of different
                 slots; same atlas, ladder, verification and controller.
F  obs_only      D's full mechanism maintenance and scoring work, uniform
                 decisions, always commit (no controller).

Arms of one root advance interleaved (least-CPU-first) inside one process so
they share host load. Fixed-update mode stops each arm after N windows;
fixed-CPU mode stops before a window whose predicted cost would exceed the cap.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..trm_curation.cache import CacheKey, MechanisticCache
from ..trm_curation.data import edge_batch
from ..trm_curation.scheduler import Candidate, pair_key, select_candidates
from ..trm_curation.signal_bank import stratified_edge_sample
from ..trm_curation.transaction import _restore_native_state, _snapshot_native_state
from .atlas import AtlasEntry, AtlasKey, MMSAtlas, metric_id
from .common import Recipe, dump_json, eval_final_development, eval_tune, make_trm, seed10, state_sha256, views
from .economics import RateController, ResourceLedger
from .ladder import LadderConfig, certificate, screen, verify
from . import mechanism, native

KINDS = {'A': 'ordinary', 'G': 'ordinary_guarded', 'B': 'ord_curation', 'C': 'mech_fresh', 'D': 'mech_atlas',
         'E': 'mech_shuffled', 'F': 'obs_only'}


def _flat(model) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy().astype(np.float32)


class Context:
    """Per-root shared, immutable inputs (identical for every arm)."""

    def __init__(self, root: int, recipe: Recipe):
        self.root, self.recipe = root, recipe
        v = views(recipe)
        self.fit = v['fit']
        self.fb_edges = stratified_edge_sample(v['feedback'], recipe.feedback_batch_edges,
                                               seed=seed10(root, 'fb_edges'))
        self.fb = edge_batch(self.fb_edges, root=root, seed=seed10(root, 'fb_cohort'),
                             delay=recipe.feedback_delay, split='train', noise=recipe.noise)
        self.n_fb = len(self.fb_edges)
        self.panels = [native.make_panel(mechanism.slice_edge(self.fb, i)) for i in range(self.n_fb)]
        self.panel_fp = [p.batch.fingerprint() for p in self.panels]
        self.fb_fp = self.fb.fingerprint()
        self.delays = (recipe.feedback_delay,)
        # derangement across slots: feedback edges are stratified per slot in blocks
        per = self.n_fb // 4
        self.perm = [(i + per) % self.n_fb for i in range(self.n_fb)]
        assert all(self.fb_edges[i][2] != self.fb_edges[self.perm[i]][2] for i in range(self.n_fb))

    def proposals(self, window: int, exposures: Counter) -> tuple:
        r = self.recipe
        eligible = [e for e in self.fit if exposures[pair_key(e)] < r.pair_exposure_cap]
        per = r.proposals_per_window // 4
        chosen = []
        for s in range(4):
            cand = sorted((e for e in eligible if e[2] == s),
                          key=lambda e: hashlib.sha256(repr((seed10(self.root, 'prop'), window, e)).encode()).digest())
            chosen.extend(cand[:per])
        return tuple(chosen)

    def cand_batch(self, e, window: int, delay: int):
        h = int.from_bytes(hashlib.sha256(repr(e).encode()).digest()[:4], 'little')
        return edge_batch((e,), root=self.root, seed=seed10(self.root, f'cand_v{window % 4}_d{delay}', h),
                          delay=delay, split='train', noise=self.recipe.noise)


class Arm:
    def __init__(self, name: str, kind: str, warm: dict, ctx: Context, cfg: dict):
        self.name, self.kind, self.ctx, self.cfg = name, kind, ctx, cfg
        r = ctx.recipe
        self.model = make_trm(r)
        self.model.load_state_dict(copy.deepcopy(warm))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=r.curation_learning_rate,
                                     weight_decay=r.weight_decay)
        self.ledger = ResourceLedger()
        self.cache = MechanisticCache(mms_mib=float(cfg.get('cache_mib', 16.0)))
        self.atlas = MMSAtlas(float(cfg.get('atlas_mib', 16.0))) if kind in ('mech_atlas', 'mech_shuffled', 'obs_only') else None
        mode = 'never' if kind in ('ordinary', 'ordinary_guarded') else ('always' if kind == 'obs_only' else cfg.get('controller', 'rate'))
        self.controller = RateController(block=int(cfg.get('block', 8)), explore_every=int(cfg.get('explore_every', 5)),
                                         mode=mode)
        self.ladder = LadderConfig(**cfg.get('ladder', {}))
        self.refresh_cap = int(cfg.get('refresh_cap', ctx.n_fb))
        self.window = 0
        self.accepted = self.rejected = self.no_audit = 0
        self.exposures: Counter = Counter()
        self.cpu_used = 0.0
        self.window_cost = {'ordinary': None, 'curated': None}
        self.overruns = []
        self.logs, self.decisions, self.mech_records = [], [], []
        self.readouts: dict[str, dict[int, tuple]] = {}   # sha -> edge -> (entry_id, readout)
        self.block_action, self.block_left = None, 0
        self.block_cpu0 = 0.0
        self.block_ce0 = None
        self.entry_ids: dict[int, str] = {}
        self.refreshes = self.audits = 0
        self.ladder_counts: Counter = Counter()

    # ------------------------------------------------------------ utilities
    def sha(self) -> str:
        with self.ledger.charge('cache_ops'):
            h = state_sha256(self.model)
            if h not in self.readouts and len(self.readouts) >= 4:
                self.readouts.pop(next(iter(self.readouts)))
            return h

    def ce_ordinary(self) -> float:
        self.model.eval()
        with torch.no_grad():
            return float(mechanism.ordinary_loss(self.model, self.ctx.fb))

    def _key(self, i: int) -> AtlasKey:
        r = self.ctx.recipe
        return AtlasKey(module='ComponentTRM.joint_state', site='post_write_prefix4',
                        phase='suffix_step0_pre_low', input_fingerprint=self.ctx.panel_fp[i],
                        objective=f'sccg01_budgeted:dose={r.dose}:eta={r.collateral_budget}:delays={self.ctx.delays}:k={r.support_k}:beam={r.support_beam}',
                        dtype='float64', dose=r.dose)

    def _refresh(self, i: int, sha: str, theta: np.ndarray) -> dict:
        r = self.ctx.recipe
        m = mechanism.refresh_edge(self.model, self.ctx.panels[i], self.ctx.delays, r.dose,
                                   r.collateral_budget, r.support_k, r.support_beam, self.ledger)
        self.refreshes += 1
        return m

    def _entry_from(self, i, m, sha, theta) -> AtlasEntry:
        payload = {'delta': m['delta'], 'a_support': m['a_support'], 'C_support': m['C_support'],
                   'svals': m['svals'], 'null_basis': m['null_basis'], 'theta_source': theta.copy(),
                   'metric': np.eye(1)}
        return AtlasEntry(key=self._key(i), source_checkpoint=sha,
                          source_factor_frame=self.model.factor_frame_digest(),
                          metric_id='identity64', support=m['support'], payload=payload,
                          source_gain=m['source_gain'], source_protected=m['source_protected'],
                          created_window=self.window)

    # ------------------------------------------------------- mechanism state
    def mechanism_deltas(self, sha: str) -> tuple[torch.Tensor, dict]:
        """Directions for every feedback edge at the current checkpoint, verified."""
        ctx, r = self.ctx, self.ctx.recipe
        n = ctx.n_fb
        info = {'refresh': 0, 'audit': 0, 'verified': 0, 'fallback': 0, 'reasons': Counter()}
        deltas = np.zeros((n, 2 * r.width))
        if self.kind == 'mech_fresh':
            for i in range(n):
                with self.ledger.charge('cache_ops'):
                    key = CacheKey('mech_fresh', sha, 'ComponentTRM.joint_state', f'edge={i}',
                                   ctx.panel_fp[i], 'sccg01_budgeted', 'n/a')
                    hit = self.cache.get(key)
                if hit is None:
                    m = self._refresh(i, sha, None)
                    with self.ledger.charge('cache_ops'):
                        self.cache.put(key, {'delta': torch.as_tensor(m['delta'])})
                    deltas[i] = m['delta']
                    info['refresh'] += 1
                    info['reasons']['FRESH_REFRESH'] += 1
                else:
                    deltas[i] = hit['delta'].numpy()
                    info['reasons']['EXACT_SAME_CHECKPOINT_HIT'] += 1
            return torch.as_tensor(deltas, dtype=torch.float32), info
        theta = _flat(self.model)
        known = self.readouts.get(sha, {})
        # entries passing the screen whose readout at this checkpoint is unknown need a verification forward
        entries, need = {}, []
        with self.ledger.charge('atlas_ops'):
            for i in range(n):
                e = self.atlas.lookup(self._key(i), self.window)
                ok, why, rel = screen(e, self._key(i), theta, self.window, self.ladder)
                entries[i] = (e, ok, why, rel)
                if ok and (i not in known or known[i][0] != self.entry_ids.get(i)):
                    need.append(i)
        if need:
            d = np.zeros((n, 2 * r.width))
            for i in need:
                d[i] = entries[i][0].payload['delta']
            with self.ledger.charge('verification'):
                self.model.eval()
                with torch.no_grad():
                    _, _, rd = mechanism.mech_loss(self.model, ctx.fb, torch.as_tensor(d, dtype=torch.float32))
            for i in need:
                known[i] = (self.entry_ids.get(i), rd[i])
            self.readouts[sha] = known
        refreshed = 0
        for i in range(n):
            e, ok, why, rel = entries[i]
            rec = {'window': self.window, 'edge': i, 'checkpoint_sha256': sha, 'screen': why,
                   'rel_displacement': rel, 'certificate': None, 'verify': None, 'outcome': None,
                   'scope': None}
            if ok:
                with self.ledger.charge('certificate'):
                    cert = certificate(e, theta, r.collateral_budget, self.ladder)
                rec['certificate'] = cert['verdict']
                if cert['verdict'] in ('NO_HEADROOM_IN_DECLARED_SPAN', 'REFRESH_OR_ORDINARY') and self.ladder.mode == 'heuristic':
                    ok = False
                    why = 'L2_HEURISTIC_REFUSE'
            if ok:
                rd = known[i][1]
                v, ratio = verify(e, rd, r.collateral_budget, self.ladder)
                rec['verify'], rec['gain_ratio'] = v, ratio
                rec['estimated_gain'], rec['realized_gain'] = e.source_gain, rd['target_gain']
                if v == 'L4_VERIFIED':
                    deltas[i] = e.payload['delta']
                    e.validations.append((sha, self.window, v))
                    e.uses += 1
                    info['verified'] += 1
                    rec['outcome'], rec['scope'] = 'REUSE_VERIFIED', 'NATIVE_FINITE_VERIFIED_ON_PANEL_AT_CHECKPOINT'
                elif v == 'L4_BORDERLINE' and self.ladder.audit:
                    # L3 exact audit: live observation on the cached support, exact re-solve
                    with self.ledger.charge('mechanism_observe'):
                        obs = native.observe_state(self.model, ctx.panels[i], ctx.delays)
                    sup = list(e.support)
                    with self.ledger.charge('certificate'):
                        from .support import resolve_fixed
                        sol = resolve_fixed(obs.a, obs.C, np.eye(len(obs.a)), sup, r.dose, r.collateral_budget)
                    with self.ledger.charge('verification'):
                        fo = native.finite_outcome(self.model, ctx.panels[i], ctx.delays, state_delta=sol['delta'])
                    self.audits += 1
                    info['audit'] += 1
                    if fo['target_gain'] > 0 and fo['protected_change_norm'] <= max(r.collateral_budget, e.source_protected) * (1 + self.ladder.protected_slack):
                        m = {'support': e.support, 'delta': sol['delta'], 'a_support': obs.a[sup], 'C_support': obs.C[:, sup],
                             'svals': np.linalg.svd(obs.C[:, sup], compute_uv=False), 'null_basis': np.zeros((len(sup), 0)),
                             'source_gain': fo['target_gain'], 'source_protected': fo['protected_change_norm']}
                        ne = self._entry_from(i, m, sha, theta)
                        with self.ledger.charge('atlas_ops'):
                            self.atlas.insert(ne)
                        self.entry_ids[i] = f'{sha[:12]}:{self.window}:audit'
                        deltas[i] = sol['delta']
                        rec['outcome'], rec['scope'] = 'L3_AUDIT_RESOLVED', 'NATIVE_FINITE_VERIFIED_ON_PANEL_AT_CHECKPOINT'
                    else:
                        ok = False
                        why = 'L3_AUDIT_FAILED'
                else:
                    ok = False
                    why = v
            if not ok:
                if refreshed < self.refresh_cap:
                    m = self._refresh(i, sha, theta)
                    refreshed += 1
                    info['refresh'] += 1
                    if m['source_gain'] > 0:
                        ne = self._entry_from(i, m, sha, theta)
                        with self.ledger.charge('atlas_ops'):
                            stored = self.atlas.insert(ne)
                        self.entry_ids[i] = f'{sha[:12]}:{self.window}:refresh'
                        deltas[i] = m['delta']
                        rec['outcome'] = 'REFRESH' if stored else 'REFRESH_NOT_STORED'
                        rec['scope'] = 'NATIVE_FINITE_VERIFIED_AT_SOURCE'
                    else:
                        info['fallback'] += 1
                        rec['outcome'], rec['scope'] = 'ORDINARY_FALLBACK_NO_POSITIVE_GAIN', 'NONE'
                else:
                    info['fallback'] += 1
                    rec['outcome'], rec['scope'] = 'ORDINARY_FALLBACK_REFRESH_CAP', 'NONE'
                rec['reason'] = why
            info['reasons'][rec['outcome']] += 1
            self.ladder_counts[rec['outcome']] += 1
            rec['physical_dose'] = float(np.linalg.norm(deltas[i]))
            self.mech_records.append(rec)
        return torch.as_tensor(deltas, dtype=torch.float32), info

    # ---------------------------------------------------------------- window
    def _select(self, props, scores, epoch, mode, delay):
        r = self.ctx.recipe
        det = float((delay + 5) * 2) * 1e-4
        rows = [Candidate(e, scores.get(e, (1.0, 0.0))[0], scores.get(e, (1.0, 0.0))[1], det, epoch) for e in props]
        with self.ledger.charge('selection'):
            sel, used = select_candidates(rows, fit_edges=self.ctx.fit, epoch=epoch, mode=mode,
                                          count=r.selected_per_window, seed=seed10(self.ctx.root, 'sched'),
                                          window=self.window, exposures=self.exposures,
                                          cap=r.pair_exposure_cap, exploration=r.exploration)
        return sel, used

    def _minibatch_step(self, batches) -> float:
        x = torch.cat([torch.cat((b.receiver, b.hybrid)) for b in batches])
        y = torch.cat([torch.cat((b.labels_r, b.labels_h)) for b in batches])
        self.model.train()
        loss = F.cross_entropy(self.model(x), y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ctx.recipe.gradient_clip)
        self.opt.step()
        return float(loss.detach())

    def run_window(self) -> dict:
        ctx, r = self.ctx, self.ctx.recipe
        if self.block_left == 0:
            self._close_block()
            self.block_action = self.controller.choose()
            self.block_left = self.controller.block
            self.block_cpu0 = self.cpu_used
            if self.block_action == 'ordinary' and self.kind != 'ordinary':
                with self.ledger.charge('controller'):
                    self.block_ce0 = self.ce_ordinary()
            else:
                self.block_ce0 = None
            self.block_ce_drop = 0.0
        action = self.block_action
        w = self.window
        delay = r.train_delays[w % len(r.train_delays)]
        with self.ledger.charge('selection'):
            if sum(1 for e in ctx.fit if self.exposures[pair_key(e)] < r.pair_exposure_cap) < r.proposals_per_window:
                self.exposures = Counter()
            props = ctx.proposals(w, self.exposures)
        with self.ledger.charge('data'):
            batches = {e: ctx.cand_batch(e, w, delay) for e in props}
        log = {'window': w, 'action': action, 'delay': delay}
        if action == 'ordinary' and self.kind == 'ordinary_guarded':
            # strongest cheap ordinary competitor: uniform proposals, transactional
            # acceptance on ordinary TRAIN-feedback CE (no candidate scoring).
            sha = self.sha()
            with self.ledger.charge('cache_ops'):
                key = CacheKey('ce_ord', sha, 'ComponentTRM', 'ordinary', ctx.fb_fp, 'ce_ordinary', 'n/a')
                hit = self.cache.get(key)
            if hit is None:
                with self.ledger.charge('acceptance'):
                    ce0 = self.ce_ordinary()
            else:
                ce0 = float(hit[0])
            sel, used = self._select(props, {}, f'{self.name}:w{w}', 'uniform', delay)
            self.exposures = Counter(used)
            with self.ledger.charge('transaction'):
                snap = (copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.opt.state_dict()),
                        _snapshot_native_state(self.model))
            with self.ledger.charge('train_step'):
                log['train_loss'] = self._minibatch_step([batches[e] for e, _ in sel])
            with self.ledger.charge('acceptance'):
                ce1 = self.ce_ordinary()
            accepted = ce1 <= ce0 + 1e-6
            if accepted:
                self.accepted += 1
                new_sha = self.sha()
                with self.ledger.charge('cache_ops'):
                    self.cache.put(CacheKey('ce_ord', new_sha, 'ComponentTRM', 'ordinary', ctx.fb_fp, 'ce_ordinary', 'n/a'),
                                   (torch.tensor(ce1),))
            else:
                with self.ledger.charge('transaction'):
                    self.model.load_state_dict(snap[0])
                    self.opt.load_state_dict(snap[1])
                    _restore_native_state(snap[2])
                self.rejected += 1
                with self.ledger.charge('cache_ops'):
                    self.cache.put(key, (torch.tensor(ce0),))
            log.update({'accepted': accepted, 'audited': True, 'ce_before': ce0, 'ce_after': ce1})
        elif action == 'ordinary':
            sel, used = self._select(props, {}, f'{self.name}:w{w}', 'uniform', delay)
            self.exposures = Counter(used)
            with self.ledger.charge('train_step'):
                log['train_loss'] = self._minibatch_step([batches[e] for e, _ in sel])
            self.accepted += 1
            log['accepted'] = True
        else:
            sha = self.sha()
            params = list(self.model.parameters())
            self.model.eval()
            info = {}
            if self.kind == 'ord_curation':
                with self.ledger.charge('cache_ops'):
                    key = CacheKey('cal_grad', sha, 'ComponentTRM.all_params', 'ordinary', ctx.fb_fp, 'cal_ordinary', 'n/a')
                    hit = self.cache.get(key)
                if hit is None:
                    with self.ledger.charge('cal_grad'):
                        loss0 = mechanism.ordinary_loss(self.model, ctx.fb)
                        g = [t.detach() for t in torch.autograd.grad(loss0, params)]
                        cal0, ce0 = float(loss0.detach()), float(loss0.detach())
                    with self.ledger.charge('cache_ops'):
                        self.cache.put(key, (tuple(g), cal0))
                else:
                    g, cal0 = list(hit[0]), hit[1]
                    ce0 = cal0
            else:
                deltas, info = self.mechanism_deltas(sha)
                use = deltas[ctx.perm] if self.kind == 'mech_shuffled' else deltas
                with self.ledger.charge('cache_ops'):
                    dkey = hashlib.sha256(use.numpy().tobytes()).hexdigest()
                    key = CacheKey('cal_grad', sha, 'ComponentTRM.all_params', f'mech:{dkey}', ctx.fb_fp, 'cal_mech', 'n/a')
                    hit = self.cache.get(key)
                if hit is None:
                    with self.ledger.charge('cal_grad'):
                        loss0, comps, rd = mechanism.mech_loss(self.model, ctx.fb, use, w_ordinary=r.w_ordinary,
                                                               w_target=r.w_target, w_preserve=r.w_preserve)
                        g = [t.detach() for t in torch.autograd.grad(loss0, params)]
                        cal0, ce0 = float(loss0.detach()), comps['ordinary_ce']
                    with self.ledger.charge('cache_ops'):
                        self.cache.put(key, (tuple(g), cal0, ce0))
                    if self.kind != 'mech_shuffled':
                        self.readouts.setdefault(sha, {}).update({i: (self.entry_ids.get(i), rd[i]) for i in range(ctx.n_fb)})
                else:
                    g, cal0, ce0 = list(hit[0]), hit[1], hit[2]
            # analytic previews of every candidate's exact AdamW step
            scores, preds = {}, []
            with self.ledger.charge('candidate_preview'):
                for e in props:
                    pv = native.preview_gain(self.model, self.opt, g, batches[e], r.gradient_clip)
                    scores[e] = (pv['train_loss'], pv['predicted_gain'])
                    preds.append(pv['predicted_gain'])
            self.ledger.count('candidate_backward', len(props))
            mode = 'uniform' if self.kind == 'obs_only' else 'mechanistic'
            sel, used = self._select(props, scores, f'{self.name}:w{w}:{sha[:12]}', mode, delay)
            self.exposures = Counter(used)
            sel_edges = [e for e, _ in sel]
            top = sorted(preds, reverse=True)[:r.selected_per_window]
            log.update({'pred_pool_mean': float(np.mean(preds)), 'pred_top_mean': float(np.mean(top)),
                        'pred_selected': [scores[e][1] for e in sel_edges], 'mech': {k: (dict(v) if isinstance(v, Counter) else v) for k, v in info.items()}})
            audit = self.kind != 'obs_only' and max(preds) > 0
            if self.kind != 'obs_only' and not audit:
                self.no_audit += 1
            with self.ledger.charge('transaction'):
                snap = (copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.opt.state_dict()),
                        _snapshot_native_state(self.model)) if audit else None
            with self.ledger.charge('train_step'):
                log['train_loss'] = self._minibatch_step([batches[e] for e in sel_edges])
            cal1 = ce1 = None
            if audit or self.kind == 'obs_only':
                new_sha = self.sha()
                with self.ledger.charge('acceptance'):
                    self.model.eval()
                    with torch.no_grad():
                        if self.kind == 'ord_curation':
                            cal1 = float(mechanism.ordinary_loss(self.model, ctx.fb))
                            ce1 = cal1
                        else:
                            l1, c1, rd1 = mechanism.mech_loss(self.model, ctx.fb, use, w_ordinary=r.w_ordinary,
                                                              w_target=r.w_target, w_preserve=r.w_preserve)
                            cal1, ce1 = float(l1), c1['ordinary_ce']
                            if self.kind != 'mech_shuffled':
                                self.readouts.setdefault(new_sha, {}).update(
                                    {i: (self.entry_ids.get(i), rd1[i]) for i in range(ctx.n_fb)})
            accepted = True
            if audit and cal1 > cal0 + 1e-6:
                with self.ledger.charge('transaction'):
                    self.model.load_state_dict(snap[0])
                    self.opt.load_state_dict(snap[1])
                    _restore_native_state(snap[2])
                accepted = False
                ce1 = ce0
            if accepted:
                self.accepted += 1
            else:
                self.rejected += 1
            log.update({'accepted': accepted, 'audited': audit, 'cal_before': cal0, 'cal_after': cal1,
                        'ce_before': ce0, 'ce_after': ce1})
            if ce1 is not None:
                self.block_ce_drop += ce0 - ce1
            for e in props:
                self.decisions.append({'root': ctx.root, 'arm': self.name, 'window': w, 'delay': delay,
                                       'checkpoint_sha256': sha, 'edge': [list(e[0]), list(e[1]), e[2]],
                                       'predicted_gain': scores[e][1], 'train_loss': scores[e][0],
                                       'selected': e in sel_edges,
                                       'selection_reason': dict(sel).get(e, 'not_selected'),
                                       'window_accepted': accepted, 'window_audited': audit,
                                       'window_realized_cal_gain': (cal0 - cal1) if (cal1 is not None and accepted) else (0.0 if cal1 is not None else None),
                                       'mechanism_outcomes': {k: v for k, v in (dict(info.get('reasons', {}))).items()},
                                       'scope': 'HEURISTIC_PROPOSAL_FIRST_ORDER_SCORE_VERIFIED_BY_TRANSACTION' if audit else 'HEURISTIC_PROPOSAL_UNAUDITED',
                                       'split': 'train', 'view': 'fit', 'training_eligible': True})
        self.window += 1
        self.block_left -= 1
        return log

    def _close_block(self) -> None:
        if self.block_action is None:
            return
        cpu = self.cpu_used - self.block_cpu0
        if self.block_action == 'ordinary':
            if self.kind in ('ordinary', 'ordinary_guarded'):
                return
            t0 = time.process_time()
            with self.ledger.charge('controller'):
                ce1 = self.ce_ordinary()
            self.cpu_used += time.process_time() - t0
            drop = (self.block_ce0 - ce1) if self.block_ce0 is not None else 0.0
        else:
            drop = self.block_ce_drop
        self.controller.update(self.block_action, drop, cpu)

    def step(self) -> dict:
        t0 = time.process_time()
        log = self.run_window()
        dt = time.process_time() - t0
        self.cpu_used += dt
        a = log['action']
        old = self.window_cost[a]
        self.window_cost[a] = dt if old is None else 0.8 * old + 0.2 * dt
        log['cpu'] = dt
        log['cpu_used'] = self.cpu_used
        self.logs.append(log)
        return log

    def predicted_next_cost(self) -> float:
        a = self.block_action if self.block_left > 0 else self.controller.choose()
        c = self.window_cost.get(a)
        return 0.0 if c is None else c

    def summary(self) -> dict:
        return {'arm': self.name, 'kind': self.kind, 'windows': self.window, 'accepted': self.accepted,
                'rejected': self.rejected, 'no_audit_windows': self.no_audit,
                'curated_windows': sum(1 for l in self.logs if l['action'] == 'curated'),
                'ordinary_windows': sum(1 for l in self.logs if l['action'] == 'ordinary'),
                'cpu_used': self.cpu_used, 'ledger': self.ledger.as_dict(),
                'uncategorized_cpu': self.cpu_used - self.ledger.total_cpu,
                'refreshes': self.refreshes, 'audits': self.audits,
                'ladder_outcomes': dict(self.ladder_counts),
                'controller_history': self.controller.history, 'controller_rates': self.controller.rates,
                'cache': self.cache.stats(), 'atlas': self.atlas.stats() if self.atlas else None,
                'overruns': self.overruns, 'final_state_sha256': state_sha256(self.model)}


def run_interleaved(arms: list[Arm], *, mode: str, n_windows: int | None = None,
                    cpu_cap: float | None = None, safety_cpu: float = 900.0) -> None:
    active = list(arms)
    while active:
        arm = min(active, key=lambda a: (a.cpu_used, a.name))
        if mode == 'fixed_update':
            if arm.window >= n_windows or arm.cpu_used >= safety_cpu:
                if arm.window < n_windows:
                    arm.overruns.append({'stopped_by_safety_cpu': safety_cpu, 'windows': arm.window})
                active.remove(arm)
                continue
        else:
            if arm.cpu_used + arm.predicted_next_cost() > cpu_cap:
                active.remove(arm)
                continue
        before = arm.cpu_used
        arm.step()
        if mode == 'fixed_cpu' and arm.cpu_used > cpu_cap:
            arm.overruns.append({'window': arm.window - 1, 'cpu_before': before, 'cpu_after': arm.cpu_used})
    for a in arms:
        a._close_block()


def run_arms(root: int, warm_path, out_dir: Path, recipe: Recipe, cfg: dict) -> dict:
    """cfg: {'arms': {name: kind-letter or {'kind':..,...}}, 'mode': 'fixed_cpu'|'fixed_update',
    'cpu_cap': 100, 'n_windows': 80, 'eval': 'tune'|'final_development', defaults...}"""
    torch.manual_seed(seed10(root, 'arms_global'))
    random.seed(seed10(root, 'arms_py'))
    np.random.seed(seed10(root, 'arms_np') % 2**32)
    out_dir = Path(out_dir)
    warm = torch.load(warm_path, weights_only=True)
    ctx = Context(root, recipe)
    arms = []
    for name, spec in cfg['arms'].items():
        spec = {'kind': spec} if isinstance(spec, str) else dict(spec)
        letter = spec.pop('kind')
        merged = {**cfg.get('defaults', {}), **spec}
        arms.append(Arm(name, KINDS[letter], warm, ctx, merged))
    t0 = time.process_time()
    run_interleaved(arms, mode=cfg['mode'], n_windows=cfg.get('n_windows'), cpu_cap=cfg.get('cpu_cap'),
                    safety_cpu=cfg.get('safety_cpu', 900.0))
    results = {'root': root, 'mode': cfg['mode'], 'config': cfg,
               'feedback_fingerprint': ctx.fb_fp, 'arms': {}, 'process_cpu_total': time.process_time() - t0}
    warm_model = make_trm(recipe)
    warm_model.load_state_dict(warm)
    results['warm_sha256'] = state_sha256(warm_model)
    for a in arms:
        s = a.summary()
        if cfg.get('eval', 'tune') == 'final_development':
            s['eval'] = eval_final_development(a.model, root, recipe)
        else:
            s['eval_tune'] = eval_tune(a.model, root, recipe, delays=(2, 7, 31))
        torch.save(a.model.state_dict(), out_dir / f'{a.name}.pt')
        s['checkpoint_file'] = f'{a.name}.pt'
        results['arms'][a.name] = s
        with (out_dir / f'{a.name}_windows.jsonl').open('w', encoding='utf-8', newline='\n') as f:
            for l in a.logs:
                f.write(json.dumps(l, sort_keys=True, default=float) + '\n')
        with (out_dir / f'{a.name}_decisions.jsonl').open('w', encoding='utf-8', newline='\n') as f:
            for d in a.decisions:
                f.write(json.dumps(d, sort_keys=True, default=float) + '\n')
        with (out_dir / f'{a.name}_mechanism.jsonl').open('w', encoding='utf-8', newline='\n') as f:
            for d in a.mech_records:
                f.write(json.dumps(d, sort_keys=True, default=float) + '\n')
        ev = s.get('eval_tune') or s['eval']['delays']
        print(json.dumps({'root': root, 'arm': a.name, 'windows': a.window, 'acc': a.accepted, 'rej': a.rejected,
                          'cpu': round(a.cpu_used, 1), 'refresh': a.refreshes,
                          'd31': round(ev['31']['accuracy'], 4)}), flush=True)
    dump_json(out_dir / 'arms_result.json', results)
    return results
