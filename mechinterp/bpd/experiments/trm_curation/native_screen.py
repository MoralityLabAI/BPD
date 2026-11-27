"""CURATE-03B native ComponentTRM integration of analytical AdamW screening.

Every forward/backward that is not the committed training step runs inside
``Custody.isolated``. The committed step runs inside ``Custody.transaction``
and is kept only when the matched-RNG feedback loss does not rise by more than
``ACCEPT_TOL`` (the CURATE-02 acceptance rule).

Frozen before any CURATE-03B native measurement (see CURATE_03B_PILOT_PROTOCOL.md):
``JOINT_TOLERANCE`` (aggregated-vs-native minibatch gradient parity),
``SCORE_TOLERANCE`` (analytical-vs-transactional predicted gain), the
acquisition order key, stages (4, 8, 12, 16) and the two gates.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import math
import time
from typing import Callable

import torch
import torch.nn.functional as F

from .adamw_screen import screen_adamw
from .batch_screen import mean_gradient, search_batch
from .cache import CacheKey, MechanisticCache
from .custody import Custody
from .scheduler import pair_key
from .signal_bank import mechanistic_feedback_loss, model_state_sha256, ordinary_feedback_loss

ACCEPT_TOL = 1e-6
STAGES = (4, 8, 12, 16)
ACQ_DOMAIN = 'BPD-TRM-CURATE-03B:acquisition-order:v1'
# Elementwise: |g_agg - g_nat| <= abs_scale * max(1, max|g_nat|); global: rel L2 <= rel_l2.
JOINT_TOLERANCE = {
    torch.float32: {'rel_l2': 1e-5, 'abs_scale': 1e-5, 'score_rel': 1e-4},
    torch.float64: {'rel_l2': 1e-12, 'abs_scale': 1e-12, 'score_rel': 1e-10},
}
# |s_a - s_t| <= score_rel * (|s_t| + ||g_cal|| * ||delta||)
SCORE_TOLERANCE = {torch.float32: 1e-4, torch.float64: 1e-10}


def edge_xy(batch):
    return (torch.cat((batch.receiver, batch.hybrid), dim=0),
            torch.cat((batch.labels_r, batch.labels_h), dim=0))


def minibatch_xy(batches):
    xs, ys = zip(*(edge_xy(b) for b in batches))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def acquisition_order(edges, *, split_seed: int, root: int, window: int) -> tuple:
    """Prospective slot-balanced order from semantic edge keys only.

    Within each edit slot, edges sort by SHA-256 of (domain, split seed, root,
    window, edge); slots are then visited round-robin 0..3. Uses no score,
    gradient, label outcome or record identifier.
    """
    by_slot = {s: [] for s in range(4)}
    for e in edges:
        by_slot[e[2]].append(e)
    for s in by_slot:
        by_slot[s].sort(key=lambda e: (hashlib.sha256(
            repr((ACQ_DOMAIN, split_seed, root, window, e)).encode()).digest(), e))
    out = []
    depth = max((len(v) for v in by_slot.values()), default=0)
    for j in range(depth):
        for s in range(4):
            if j < len(by_slot[s]):
                out.append(by_slot[s][j])
    if sorted(out) != sorted(edges):
        raise AssertionError('acquisition order lost an edge')
    return tuple(out)


def gate_decision(scores_in_order, gate: str | None):
    """Return (acquired_count, abstain) for a full ordered score list.

    ``scores_in_order`` must hold the analytic scores of the candidates in
    acquisition order; only the prefix the policy would have paid for is used.
    """
    n = len(scores_in_order)
    if gate is None:
        return n, False
    if gate == 'full16':
        return n, not any(s > 0 for s in scores_in_order)
    if gate == 'early12':
        k = min(12, n)
        if not any(s > 0 for s in scores_in_order[:k]):
            return k, True
        return n, not any(s > 0 for s in scores_in_order)
    raise ValueError(f'unknown gate {gate}')


def audit_draw(root: int, arm: str, window: int, probability: float) -> tuple[bool, float]:
    """Known-probability shadow audit draw, independent of any score."""
    u = int.from_bytes(hashlib.sha256(
        repr(('BPD-TRM-CURATE-03B:audit:v1', root, arm, window)).encode()).digest()[:8], 'little') / 2**64
    return u < probability, u


@dataclass
class FeedbackGradient:
    grads: tuple
    loss0: float
    rng_point: tuple
    checkpoint_sha256: str
    objective: str
    cpu_seconds: float
    cache_hit: bool


@dataclass
class Ledger:
    """Exclusive CPU attribution by category (nested blocks credit only their own time).

    ``inclusive`` keeps the full elapsed time of every category, so e.g. the
    whole shadow-audit cost is ``inclusive['audit']`` while its candidate
    gradients still appear under ``candidate_gradient``.
    """
    cpu: Counter = field(default_factory=Counter)
    inclusive: Counter = field(default_factory=Counter)
    counts: Counter = field(default_factory=Counter)
    _stack: list = field(default_factory=list)

    def timed(self, category: str):
        ledger = self

        class _T:
            def __enter__(self):
                ledger._stack.append([category, time.process_time(), 0.0])

            def __exit__(self, *exc):
                cat, t0, child = ledger._stack.pop()
                elapsed = time.process_time() - t0
                ledger.cpu[cat] += elapsed - child
                ledger.inclusive[cat] += elapsed
                ledger.counts[cat] += 1
                if ledger._stack:
                    ledger._stack[-1][2] += elapsed
        return _T()


class NativeScreen:
    def __init__(self, model, optimizer, recipe, slot_projectors, *,
                 cache: MechanisticCache | None = None, generators=(), ledger: Ledger | None = None):
        self.model, self.opt, self.recipe = model, optimizer, recipe
        self.slot_projectors = slot_projectors
        self.cache = cache
        self.custody = Custody(model, optimizer, generators)
        self.params = tuple(model.parameters())
        opt_params = [p for g in optimizer.param_groups for p in g['params']]
        if [id(p) for p in opt_params] != [id(p) for p in self.params]:
            raise ValueError('optimizer parameter order must match model.parameters()')
        self.ledger = ledger or Ledger()

    # ----- objectives -------------------------------------------------
    def objective_loss(self, fb_batch, objective: str):
        if objective == 'ordinary':
            return ordinary_feedback_loss(self.model, fb_batch)
        if objective == 'mechanistic':
            r = self.recipe
            return mechanistic_feedback_loss(
                self.model, fb_batch, self.slot_projectors, dose=r.dose,
                w_ordinary=r.w_ordinary, w_target=r.w_target, w_preserve=r.w_preserve)
        raise ValueError(objective)

    # ----- acquisition ------------------------------------------------
    def feedback_gradient(self, fb_batch, objective: str) -> FeedbackGradient:
        """TRAIN feedback gradient once per checkpoint/objective/input recipe; detached cache."""
        with self.ledger.timed('feedback_gradient'):
            ckpt = model_state_sha256(self.model)
            key = None
            if self.cache is not None and self.cache.max_bytes > 0:
                fframe = self.model.factor_frame_digest()
                self.cache.bind_checkpoint(ckpt, fframe)
                key = CacheKey(
                    namespace='cal_grad_03b', checkpoint_sha256=ckpt,
                    module_id='ComponentTRM.all_params', visit_spec=f'dose={self.recipe.dose}',
                    input_fingerprint=fb_batch.fingerprint(), objective_id=f'cal_{objective}',
                    factor_frame_sha256=fframe)
                hit = self.cache.get(key)
                if hit is not None:
                    grads, loss0, rng_state = hit
                    return FeedbackGradient(grads, loss0, self._rng_from_cache(rng_state), ckpt,
                                            objective, 0.0, True)
            t0 = time.process_time()
            with self.custody.isolated():
                point = self.custody.rng_point()
                self.model.eval()
                loss = self.objective_loss(fb_batch, objective)
                grads = tuple(None if g is None else g.detach().clone() for g in
                              torch.autograd.grad(loss, self.params, allow_unused=True))
                loss0 = float(loss.detach())
            cost = time.process_time() - t0
            if key is not None:
                self.cache.put(key, (grads, loss0, point[0]), compute_cpu_seconds=cost)
            return FeedbackGradient(grads, loss0, point, ckpt, objective, cost, False)

    def _rng_from_cache(self, torch_state):
        # Feedback objectives here consume only torch CPU RNG (none in this
        # model); explicit generators and Python/NumPy draws replay from now.
        p = self.custody.rng_point()
        return (torch_state.clone(),) + p[1:]

    def candidate_gradient(self, batch):
        """Raw unclipped TRAIN gradient of one candidate at the exact parent."""
        with self.ledger.timed('candidate_gradient'):
            with self.custody.isolated():
                self.model.train()
                x, y = edge_xy(batch)
                loss = F.cross_entropy(self.model(x), y)
                grads = tuple(None if g is None else g.detach().clone() for g in
                              torch.autograd.grad(loss, self.params, allow_unused=True))
            return grads, float(loss.detach())

    def analytic_score(self, grads, fb: FeedbackGradient) -> dict:
        with self.ledger.timed('analytic_scoring'):
            out = screen_adamw(self.opt, grads, fb.grads, self.recipe.gradient_clip)
            return {k: v for k, v in out.items() if k != 'deltas'}

    def native_batch_gradient(self, batches):
        with self.ledger.timed('native_batch_gradient'):
            with self.custody.isolated():
                self.model.train()
                x, y = minibatch_xy(batches)
                loss = F.cross_entropy(self.model(x), y)
                return tuple(None if g is None else g.detach().clone() for g in
                             torch.autograd.grad(loss, self.params, allow_unused=True))

    # ----- joint minibatch --------------------------------------------
    def joint_search(self, candidate_grads, fb: FeedbackGradient, initial_idx, *, pinned_idx,
                     legal: Callable, rounds: int):
        with self.ledger.timed('joint_search'):
            return search_batch(self.opt, candidate_grads, fb.grads, initial_idx,
                                pinned=pinned_idx, allowed=legal,
                                clip=self.recipe.gradient_clip, rounds=rounds)

    # ----- feedback evaluation ----------------------------------------
    @torch.no_grad()
    def feedback_after(self, fb_batch, fb: FeedbackGradient) -> float:
        """Matched-RNG feedback loss at the current (trial) parameters; side-effect free."""
        with self.ledger.timed('feedback_eval'):
            with self.custody.isolated():
                self.custody.replay_rng(fb.rng_point)
                self.model.eval()
                return float(self.objective_loss(fb_batch, fb.objective))

    # ----- committed step ---------------------------------------------
    def step_minibatch(self, batches, *, fb: FeedbackGradient | None = None, fb_batch=None,
                       accept_test: bool = True, commit_allowed: bool = True,
                       aggregated: tuple | None = None) -> dict:
        """Real native minibatch step under transactional custody.

        With ``accept_test`` the step is kept iff matched-RNG feedback loss does
        not increase by more than ACCEPT_TOL and ``commit_allowed``. Without it
        the step is committed unconditionally (uniform/observer arms).
        ``aggregated`` (mean raw per-edge gradient) is compared with the native
        concatenated gradient captured before clipping.
        """
        info: dict = {}
        with self.custody.transaction() as txn:
            with self.ledger.timed('acceptance_step'):
                self.model.train()
                x, y = minibatch_xy(batches)
                loss = F.cross_entropy(self.model(x), y)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                native = tuple(None if p.grad is None else p.grad.detach().clone() for p in self.params)
            if aggregated is not None or fb is not None:
                with self.ledger.timed('verification'):
                    if aggregated is not None:
                        info['parity'] = gradient_parity(aggregated, native)
                    if fb is not None:
                        s_nat = screen_adamw(self.opt, native, fb.grads, self.recipe.gradient_clip)
                        info['predicted_joint_gain_native'] = s_nat['predicted_gain']
                        if aggregated is not None:
                            s_agg = screen_adamw(self.opt, aggregated, fb.grads, self.recipe.gradient_clip)
                            info['predicted_joint_gain_aggregated'] = s_agg['predicted_gain']
                            info['parity']['score_abs_error'] = abs(
                                s_agg['predicted_gain'] - s_nat['predicted_gain'])
                            scale = abs(s_nat['predicted_gain']) + _norm(fb.grads) * s_nat['update_norm']
                            tol = JOINT_TOLERANCE[self.params[0].dtype]['score_rel'] * max(scale, 1e-30)
                            info['parity']['score_ok'] = info['parity']['score_abs_error'] <= tol
                            info['parity']['ok'] = info['parity']['ok'] and info['parity']['score_ok']
            with self.ledger.timed('acceptance_step'):
                torch.nn.utils.clip_grad_norm_(self.params, self.recipe.gradient_clip,
                                               error_if_nonfinite=True)
                self.opt.step()
            info['train_loss'] = float(loss.detach())
            if accept_test:
                after = self.feedback_after(fb_batch, fb)
                info['cal_before'] = fb.loss0
                info['cal_after'] = after
                info['actual_gain'] = fb.loss0 - after
                info['would_accept'] = after <= fb.loss0 + ACCEPT_TOL
                info['accepted'] = info['would_accept'] and commit_allowed
            else:
                info['accepted'] = commit_allowed
            if info['accepted']:
                txn.commit()
        return info


def _norm(grads) -> float:
    return math.sqrt(sum(float(g.double().square().sum()) for g in grads if g is not None))


def gradient_parity(aggregated, native) -> dict:
    """Frozen dtype-aware comparison of mean per-edge vs native concatenated gradient."""
    max_abs, err_sq, ref_sq, max_ref = 0.0, 0.0, 0.0, 0.0
    dtype = None
    for a, n in zip(aggregated, native, strict=True):
        if a is None and n is None:
            continue
        a = torch.zeros_like(n) if a is None else a
        n = torch.zeros_like(a) if n is None else n
        dtype = n.dtype
        d = (a.double() - n.double())
        max_abs = max(max_abs, float(d.abs().max()))
        err_sq += float(d.square().sum())
        ref_sq += float(n.double().square().sum())
        max_ref = max(max_ref, float(n.double().abs().max()))
    tol = JOINT_TOLERANCE[dtype or torch.float32]
    rel = math.sqrt(err_sq) / max(math.sqrt(ref_sq), 1e-30)
    ok = rel <= tol['rel_l2'] and max_abs <= tol['abs_scale'] * max(1.0, max_ref)
    return {'max_abs_error': max_abs, 'rel_l2_error': rel, 'ok': bool(ok),
            'dtype': str(dtype), 'tolerance': dict(tol)}


def score_agreement(analytic: float, transactional: float, cal_norm: float, update_norm: float,
                    dtype=torch.float32) -> dict:
    err = abs(analytic - transactional)
    scale = abs(transactional) + cal_norm * update_norm
    return {'abs_error': err, 'scale': scale,
            'ok': err <= SCORE_TOLERANCE[dtype] * max(scale, 1e-30)}


def exposure_legal(exposures: Counter, cap: int):
    """Legality for a whole candidate batch: every pair key stays within cap."""
    def legal(edges) -> bool:
        used = Counter(pair_key(e) for e in edges)
        return all(exposures[k] + n <= cap for k, n in used.items())
    return legal
