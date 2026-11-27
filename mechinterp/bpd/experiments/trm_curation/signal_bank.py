"""Mechanistic signal bank, SCCG-01 selective-control frontier, and feedback objectives.

All discovery, calibration, feedback and scheduler-tuning operate strictly on
TRAIN semantic edges partitioned by ``edge_views(split_seed)`` (624 fit,
144 feedback, 144 tune, both orientations grouped together).
"""
from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr, pearsonr
import torch
from torch import Tensor
import torch.nn.functional as F

from .cache import CacheKey, MechanisticCache
from .data import edge_batch
from .model import ComponentTRM
from .observer import observe_edge
from .reference import (
    budgeted_selective_direction,
    joint_support_search,
    sccg_frontier,
    selective_direction,
    signed_observer_matched_sham,
)
from .transaction import probe_update


def seed_for(root: int, domain: str, index: int = 0) -> int:
    raw = f"BPD-TRM-CURATE-02:{root}:{domain}:{index}".encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little') % (2**63 - 1)


def model_state_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode('ascii'))
        h.update(str(tuple(value.shape)).encode('ascii'))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cached_edge_batch(edges, *, root: int, seed: int, delay: int,
                      split: str = 'train', noise: float = 0.03,
                      cache: MechanisticCache | None = None):
    """Build or retrieve an immutable feature template for ``edges``."""
    if cache is not None and cache.max_bytes > 0:
        inp_fp = hashlib.sha256(
            repr((tuple(edges), root, seed, delay, split, noise)).encode('utf-8')
        ).hexdigest()
        key = CacheKey(
            namespace='template',
            checkpoint_sha256='immutable',
            module_id='data.edge_batch',
            visit_spec=f'delay={delay}',
            input_fingerprint=inp_fp,
            objective_id=split,
            factor_frame_sha256='none',
        )
        hit = cache.get(key)
        if hit is not None:
            return hit
        t0 = time.process_time()
        batch = edge_batch(edges, root=root, seed=seed, delay=delay, split=split, noise=noise)
        cost = time.process_time() - t0
        cache.put(key, batch, compute_cpu_seconds=cost)
        return batch
    return edge_batch(edges, root=root, seed=seed, delay=delay, split=split, noise=noise)


def cached_observe_edge(model: ComponentTRM, batch, delays, *,
                        dose: float = 0.75,
                        ckpt_sha: str | None = None,
                        factor_sha: str | None = None,
                        cache: MechanisticCache | None = None,
                        objective_id: str = 'signed_observer'):
    """Observe an edge with exact checkpoint/factor/input provenance cache binding."""
    if cache is not None and cache.max_bytes > 0:
        ckpt = ckpt_sha or model_state_sha256(model)
        fframe = factor_sha or model.factor_frame_digest()
        key = CacheKey(
            namespace='observer',
            checkpoint_sha256=ckpt,
            module_id='ComponentTRM.joint_state',
            visit_spec=f'delays={tuple(delays)};dose={dose}',
            input_fingerprint=batch.fingerprint(),
            objective_id=objective_id,
            factor_frame_sha256=fframe,
        )
        hit = cache.get(key)
        if hit is not None:
            return hit
        t0 = time.process_time()
        obs = observe_edge(model, batch, delays, dose=dose, use_suffix=True)
        cost = time.process_time() - t0
        cache.put(key, obs, compute_cpu_seconds=cost)
        return obs
    return observe_edge(model, batch, delays, dose=dose, use_suffix=True)


def stratified_edge_sample(edges: tuple, n: int, *, seed: int) -> tuple:
    """Deterministically sample ``n`` edges balanced across the 4 edited slots."""
    by_slot = {s: [] for s in range(4)}
    for e in edges:
        by_slot[e[2]].append(e)
    per_slot = max(1, n // 4)
    chosen = []
    for s in range(4):
        ordered = sorted(
            by_slot[s],
            key=lambda e: hashlib.sha256(repr((seed, 'strat', s, e)).encode()).digest(),
        )
        chosen.extend(ordered[:per_slot])
    if len(chosen) < n:
        rem = [e for e in edges if e not in set(chosen)]
        rem_ord = sorted(rem, key=lambda e: hashlib.sha256(repr((seed, 'rem', e)).encode()).digest())
        chosen.extend(rem_ord[:n - len(chosen)])
    return tuple(chosen[:n])


@torch.no_grad()
def eval_state_direction(model: ComponentTRM, batch, delta: np.ndarray, delay: int) -> dict[str, Any]:
    """Evaluate a physical joint-state edit ``delta`` via exact suffix continuation."""
    x = batch.receiver
    slot = int(batch.edit_slot[0])
    old_v = int(batch.labels_r[slot])
    new_v = int(batch.labels_h[slot])
    seq = torch.cat((x[:, :4 + delay], x[:, -1:]), dim=1)
    x_suf = seq[:, 4:]
    s0 = model.prefix_state(x[:1]).detach()  # [1, 2*w]
    d_t = torch.as_tensor(delta, dtype=x.dtype, device=x.device).reshape(1, -1)
    clean_logits = model.rollout_suffix(x_suf, s0.expand(4, -1), prefix_steps=4)['logits']
    edit_logits = model.rollout_suffix(x_suf, (s0 + d_t).expand(4, -1), prefix_steps=4)['logits']

    clean_pred = clean_logits.argmax(-1)
    edit_pred = edit_logits.argmax(-1)
    target_hit = bool(int(edit_pred[slot]) == new_v)
    target_margin_delta = float(
        (edit_logits[slot, new_v] - edit_logits[slot, old_v])
        - (clean_logits[slot, new_v] - clean_logits[slot, old_v])
    )
    prot_init_correct = 0
    prot_corrupted = 0
    prot_preserved = 0
    prot_recovered = 0
    for q in range(4):
        if q == slot:
            continue
        lbl = int(batch.labels_r[q])
        c_ok = bool(int(clean_pred[q]) == lbl)
        e_ok = bool(int(edit_pred[q]) == lbl)
        if c_ok:
            prot_init_correct += 1
            if not e_ok:
                prot_corrupted += 1
        elif e_ok:
            prot_recovered += 1
        if e_ok:
            prot_preserved += 1
    return {
        'target_hit': target_hit,
        'target_margin_delta': target_margin_delta,
        'prot_init_correct': prot_init_correct,
        'prot_corrupted': prot_corrupted,
        'prot_preserved': prot_preserved,
        'prot_recovered': prot_recovered,
        'dose': float(np.linalg.norm(delta)),
    }


@torch.no_grad()
def component_cooperation_metrics(model: ComponentTRM, delta_state: np.ndarray,
                                  group_indices: tuple[int, ...]) -> dict[str, float | None]:
    """Compute physical outside-group write leakage, cosine, and cancellation K.

    Evaluates the native BPD operator W = V @ U on both z_H and z_L half-displacements
    (and their sum entering the low block).
    """
    V = model.up.V.detach().double().cpu().numpy()  # [w, C]
    U = model.up.U.detach().double().cpu().numpy()  # [C, 2w]
    w = model.width
    d_h = delta_state[:w]
    d_l = delta_state[w:]
    # Effective state entering the first low-cycle block is z_L + z_H + e_t
    d_eff = d_h + d_l
    reads = d_eff @ V  # [C]
    mask = np.zeros(V.shape[1], dtype=np.float64)
    mask[list(group_indices)] = 1.0
    g_write = (reads * mask) @ U
    o_write = (reads * (1.0 - mask)) @ U
    gn = float(np.linalg.norm(g_write))
    on = float(np.linalg.norm(o_write))
    tot_n = float(np.linalg.norm(g_write + o_write))
    denom_sq = gn * gn + on * on
    l_outside = (on * on / denom_sq) if denom_sq > 1e-24 else None
    cos_go = (float(g_write @ o_write) / (gn * on)) if (gn > 1e-12 and on > 1e-12) else None
    cancel_k = (1.0 - tot_n / (gn + on)) if (gn + on) > 1e-12 else None
    return {
        'group_write_norm': gn,
        'outside_write_norm': on,
        'total_write_norm': tot_n,
        'outside_write_fraction': l_outside,
        'group_outside_cosine': cos_go,
        'cancellation_K': cancel_k,
    }


def build_signal_bank(model: ComponentTRM, root: int, views: dict[str, tuple], *,
                      delays: tuple[int, ...] = (2, 7, 31),
                      dose: float = 0.75,
                      dose_grid: tuple[float, ...] = (0.25, 0.75, 1.5, 3.0),
                      budgets: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
                      n_edges: int = 24,
                      n_svd_edges: int = 48,
                      cache: MechanisticCache | None = None,
                      tag: str = 'warmup') -> dict[str, Any]:
    """Build complete mechanistic signal bank and SCCG-01 frontier on TRAIN edges."""
    t_start = time.process_time()
    model.eval()
    ckpt_sha = model_state_sha256(model)
    factor_sha = model.factor_frame_digest()
    if cache is not None:
        cache.bind_checkpoint(ckpt_sha, factor_sha)

    max_delay = max(delays)
    w = model.width
    C_comp = model.components
    V_np = model.up.V.detach().double().cpu().numpy()  # [w, C]
    U_np = model.up.U.detach().double().cpu().numpy()  # [C, 2w]

    # 1. Build TRAIN-only hybrid-difference SVD basis from `fit` edges (separate from feedback probe edges)
    svd_edges = stratified_edge_sample(views['fit'], n_svd_edges, seed=seed_for(root, f'svd_{tag}', 0))
    svd_diffs = []
    with torch.no_grad():
        for i, edge in enumerate(svd_edges):
            b = cached_edge_batch(
                (edge,), root=root, seed=seed_for(root, f'svd_batch_{tag}', i),
                delay=max_delay, split='train', cache=cache,
            )
            sr = model.prefix_state(b.receiver[:1]).detach().double().cpu().numpy().reshape(-1)
            sh = model.prefix_state(b.hybrid[:1]).detach().double().cpu().numpy().reshape(-1)
            svd_diffs.append(sh - sr)
    D_mat = np.stack(svd_diffs, axis=1)  # [2w, n_svd_edges]
    svd_u, svd_s, _ = np.linalg.svd(D_mat, full_matrices=False)

    # Physical support bases in 2w-dimensional joint state [z_H, z_L]:
    # - High-only: [I_w; 0], Low-only: [0; I_w], Joint: I_{2w}
    B_high = np.vstack([np.eye(w), np.zeros((w, w))])
    B_low = np.vstack([np.zeros((w, w)), np.eye(w)])
    B_joint = np.eye(2 * w)
    # - Contiguous component groups from V (groups of 4; last group trimmed if C not multiple of 4)
    contig_groups = [tuple(range(i, min(i + 4, C_comp))) for i in range(0, C_comp, 4) if i + 4 <= C_comp]
    # Lift V to high-state basis [V; 0] and low-state basis [0; V] and joint paired basis
    B_V_high = np.vstack([V_np, np.zeros_like(V_np)])
    B_V_low = np.vstack([np.zeros_like(V_np), V_np])
    B_V_joint = np.hstack([B_V_high, B_V_low])  # [2w, 2*C]

    # 2. Sample feedback edges and evaluate full signed observer + SCCG-01 frontier
    probe_edges = stratified_edge_sample(views['feedback'], n_edges, seed=seed_for(root, f'bank_{tag}', 0))
    rng_np = np.random.default_rng(seed_for(root, f'bank_rand_{tag}', 0))

    edge_records = []
    # Aggregate accumulators for clean summary
    site_caps = {k: [] for k in ('high_zero', 'low_zero', 'joint_zero', 'high_b025', 'low_b025', 'joint_b025')}
    family_stats = {
        name: {'linear_gains': [], 'collaterals': [], 'feasible': 0, 'total': 0,
               'target_hits': 0, 'prot_init_correct': 0, 'prot_corrupted': 0}
        for name in (
            'contiguous_v_high_r4', 'contiguous_v_low_r4',
            'joint_pair_search_r4', 'joint_pair_search_r8',
            'svd_r2', 'svd_r4', 'svd_r8',
            'random_r4', 'random_r8',
            'joint_zero_col', 'joint_budget_025', 'joint_budget_050',
            'hybrid_full_dose', 'hybrid_natural_oracle',
            'signed_observer_sham', 'write_site_u_low', 'write_site_rand_low',
        )
    }
    frontier_agg = {str(b): {'gains': [], 'collaterals': [], 'feasible': 0, 'target_hits': 0,
                             'prot_init_correct': 0, 'prot_corrupted': 0}
                    for b in budgets}
    dose_agg = {str(d): {'target_hits': 0, 'prot_init_correct': 0, 'prot_corrupted': 0,
                         'linear_errors': [], 'total': 0}
                for d in dose_grid}

    # Projection residuals & cooperation
    proj_res_v_high, proj_res_svd4, proj_res_svd8 = [], [], []
    finite_dose_errs, linear_preds, actual_deltas = [], [], []
    coop_outside_fracs, coop_cosines, coop_cancels = [], [], []

    def _record_family(name: str, res: dict, ev: dict | None):
        st = family_stats[name]
        st['total'] += 1
        if res.get('feasible', True) and res.get('dose', 0.0) > 0:
            st['feasible'] += 1
        st['linear_gains'].append(float(res.get('gain', 0.0)))
        st['collaterals'].append(float(res.get('collateral', 0.0)))
        if ev is not None:
            st['target_hits'] += int(ev['target_hit'])
            st['prot_init_correct'] += int(ev['prot_init_correct'])
            st['prot_corrupted'] += int(ev['prot_corrupted'])

    # Precompute orthonormal projections for residual checks
    q_vh, _, _ = np.linalg.svd(B_V_high, full_matrices=False)
    q_svd4 = svd_u[:, :min(4, svd_u.shape[1])]
    q_svd8 = svd_u[:, :min(8, svd_u.shape[1])]

    for i, edge in enumerate(probe_edges):
        batch = cached_edge_batch(
            (edge,), root=root, seed=seed_for(root, f'bank_batch_{tag}', i),
            delay=max_delay, split='train', cache=cache,
        )
        obs = cached_observe_edge(
            model, batch, delays, dose=dose, ckpt_sha=ckpt_sha,
            factor_sha=factor_sha, cache=cache,
        )
        a = obs['target'].double().cpu().numpy()
        C_mat = obs['protected'].double().cpu().numpy()
        J_obs = obs['observer'].double().cpu().numpy()
        raw_h = obs['raw_hybrid_displacement'].double().cpu().numpy()
        raw_h_norm = float(np.linalg.norm(raw_h))

        # Projection residuals of raw hybrid displacement
        if raw_h_norm > 1e-12:
            proj_res_v_high.append(float(np.linalg.norm(raw_h - q_vh @ (q_vh.T @ raw_h)) / raw_h_norm))
            proj_res_svd4.append(float(np.linalg.norm(raw_h - q_svd4 @ (q_svd4.T @ raw_h)) / raw_h_norm))
            proj_res_svd8.append(float(np.linalg.norm(raw_h - q_svd8 @ (q_svd8.T @ raw_h)) / raw_h_norm))

        finite_dose_errs.append(float(obs['finite_dose_error']))
        linear_preds.append(float(obs['target_linear']))
        actual_deltas.append(float(obs['target_actual']))

        # Site capacities: high, low, joint (zero-collateral & budgeted 0.25)
        r_hz = selective_direction(a, C_mat, B_high, dose)
        r_lz = selective_direction(a, C_mat, B_low, dose)
        r_jz = selective_direction(a, C_mat, B_joint, dose)
        r_hb = budgeted_selective_direction(a, C_mat, B_high, dose, collateral_budget=0.25)
        r_lb = budgeted_selective_direction(a, C_mat, B_low, dose, collateral_budget=0.25)
        r_jb = budgeted_selective_direction(a, C_mat, B_joint, dose, collateral_budget=0.25)
        r_jb5 = budgeted_selective_direction(a, C_mat, B_joint, dose, collateral_budget=0.50)

        site_caps['high_zero'].append(r_hz['gain'])
        site_caps['low_zero'].append(r_lz['gain'])
        site_caps['joint_zero'].append(r_jz['gain'])
        site_caps['high_b025'].append(r_hb['gain'])
        site_caps['low_b025'].append(r_lb['gain'])
        site_caps['joint_b025'].append(r_jb['gain'])

        _record_family('joint_zero_col', r_jz, eval_state_direction(model, batch, r_jz['delta'], max_delay))
        _record_family('joint_budget_025', r_jb, eval_state_direction(model, batch, r_jb['delta'], max_delay))
        _record_family('joint_budget_050', r_jb5, eval_state_direction(model, batch, r_jb5['delta'], max_delay))

        # SCCG-01 frontier on full joint state
        front = sccg_frontier(a, C_mat, B_joint, dose, budgets=budgets)
        for b_val, fr in zip(budgets, front):
            fb = frontier_agg[str(b_val)]
            fb['gains'].append(fr['gain'])
            fb['collaterals'].append(fr['collateral'])
            fb['feasible'] += int(fr['feasible'])
            ev_f = eval_state_direction(model, batch, fr['delta'], max_delay)
            fb['target_hits'] += int(ev_f['target_hit'])
            fb['prot_init_correct'] += int(ev_f['prot_init_correct'])
            fb['prot_corrupted'] += int(ev_f['prot_corrupted'])

        # Support family comparisons (use collateral_budget=0.25 when zero-collateral is infeasible
        # for low-rank supports, and record exact zero-collateral feasibility separately)
        # 1) Best contiguous V group (high & low sites) at budget 0.25
        best_ch = max(
            (budgeted_selective_direction(a, C_mat, B_V_high[:, list(g)], dose, 0.25) for g in contig_groups),
            key=lambda r: r['gain'],
        )
        best_cl = max(
            (budgeted_selective_direction(a, C_mat, B_V_low[:, list(g)], dose, 0.25) for g in contig_groups),
            key=lambda r: r['gain'],
        )
        _record_family('contiguous_v_high_r4', best_ch, eval_state_direction(model, batch, best_ch['delta'], max_delay))
        _record_family('contiguous_v_low_r4', best_cl, eval_state_direction(model, batch, best_cl['delta'], max_delay))

        # 2) Joint pair beam search over V columns (high+low pooled columns, rank 4 & 8)
        # Keep search fast by using top-16 columns by |a^T b| plus pair expansion
        col_scores = np.abs(a @ B_V_joint)
        top_cols = np.argsort(-col_scores)[:min(16, B_V_joint.shape[1])]
        B_sub = B_V_joint[:, top_cols]
        _, r_jp4, _ = joint_support_search(a, C_mat, B_sub, dose, rank=4, beam=4, collateral_budget=0.25)
        _, r_jp8, _ = joint_support_search(a, C_mat, B_sub, dose, rank=min(8, B_sub.shape[1]), beam=4, collateral_budget=0.25)
        _record_family('joint_pair_search_r4', r_jp4, eval_state_direction(model, batch, r_jp4['delta'], max_delay))
        _record_family('joint_pair_search_r8', r_jp8, eval_state_direction(model, batch, r_jp8['delta'], max_delay))

        # 3) SVD supports (rank 2, 4, 8)
        for r_k, fname in ((2, 'svd_r2'), (4, 'svd_r4'), (8, 'svd_r8')):
            B_svd = svd_u[:, :min(r_k, svd_u.shape[1])]
            res_s = budgeted_selective_direction(a, C_mat, B_svd, dose, collateral_budget=0.25)
            _record_family(fname, res_s, eval_state_direction(model, batch, res_s['delta'], max_delay))

        # 4) Physically matched random supports (rank 4, 8)
        for r_k, fname in ((4, 'random_r4'), (8, 'random_r8')):
            B_rnd = rng_np.standard_normal((2 * w, r_k))
            res_r = budgeted_selective_direction(a, C_mat, B_rnd, dose, collateral_budget=0.25)
            _record_family(fname, res_r, eval_state_direction(model, batch, res_r['delta'], max_delay))

        # 5) Full hybrid direction (equal dose 0.75) & natural hybrid replacement (unmatched dose)
        hd = obs['hybrid_direction'].double().cpu().numpy()
        _record_family(
            'hybrid_full_dose',
            {'gain': float(a @ hd), 'collateral': float(np.linalg.norm(C_mat @ hd)),
             'dose': float(np.linalg.norm(hd)), 'feasible': bool(obs['valid_dose'])},
            eval_state_direction(model, batch, hd, max_delay),
        )
        _record_family(
            'hybrid_natural_oracle',
            {'gain': float(a @ raw_h), 'collateral': float(np.linalg.norm(C_mat @ raw_h)),
             'dose': raw_h_norm, 'feasible': raw_h_norm > 1e-9},
            eval_state_direction(model, batch, raw_h, max_delay),
        )

        # 6) Signed-observer-matched sham (MOLT-06) matching joint_budget_025
        sham_res = signed_observer_matched_sham(
            J_obs, r_jb['delta'], r_jb['dose'], seed=seed_for(root, f'sham_{tag}', i)
        )
        ev_sham = (
            eval_state_direction(model, batch, sham_res['delta'], max_delay)
            if sham_res['feasible'] else None
        )
        _record_family(
            'signed_observer_sham',
            {'gain': float(a @ sham_res['delta']) if sham_res['feasible'] else 0.0,
             'collateral': float(np.linalg.norm(C_mat @ sham_res['delta'])) if sham_res['feasible'] else 0.0,
             'dose': sham_res['dose'], 'feasible': sham_res['feasible']},
            ev_sham,
        )

        # 7) Native write-site intervention (up.U write space at low visit) vs matched random write
        # Compute native hybrid write difference at step 4 low visit: (h_hybrid - h_recv) @ V @ U
        x_seq = torch.cat((batch.receiver[:, :4 + max_delay], batch.receiver[:, -1:]), dim=1)
        x_suf = x_seq[:, 4:]
        s0 = obs['post_write_state'].reshape(1, -1)
        raw_d_eff = (raw_h[:w] + raw_h[w:])  # change in z_H + z_L entering low block
        w_nat = (raw_d_eff @ V_np) @ U_np  # [2w] in write space
        wn = float(np.linalg.norm(w_nat))
        w_dir = (w_nat * (dose / wn)) if wn > 1e-9 else np.zeros_like(w_nat)
        w_rand = rng_np.standard_normal(2 * w)
        w_rand = w_rand * (dose / max(float(np.linalg.norm(w_rand)), 1e-12))

        def _eval_write(wp_vec: np.ndarray) -> dict[str, Any]:
            with torch.no_grad():
                wp_t = torch.as_tensor(wp_vec, dtype=x_seq.dtype).reshape(1, -1).expand(4, -1)
                cl = model.rollout_suffix(x_suf, s0.expand(4, -1), prefix_steps=4)['logits']
                el = model.rollout_suffix(
                    x_suf, s0.expand(4, -1), prefix_steps=4, write_patch=wp_t, patch_step=0, write_site='low'
                )['logits']
                slot_i = int(batch.edit_slot[0])
                new_v = int(batch.labels_h[slot_i])
                c_pred, e_pred = cl.argmax(-1), el.argmax(-1)
                pic = sum(int(c_pred[q] == batch.labels_r[q]) for q in range(4) if q != slot_i)
                pcor = sum(
                    int(c_pred[q] == batch.labels_r[q] and e_pred[q] != batch.labels_r[q])
                    for q in range(4) if q != slot_i
                )
                return {
                    'target_hit': bool(int(e_pred[slot_i]) == new_v),
                    'prot_init_correct': pic,
                    'prot_corrupted': pcor,
                }

        _record_family(
            'write_site_u_low',
            {'gain': 0.0, 'collateral': 0.0, 'dose': float(np.linalg.norm(w_dir)), 'feasible': wn > 1e-9},
            _eval_write(w_dir),
        )
        _record_family(
            'write_site_rand_low',
            {'gain': 0.0, 'collateral': 0.0, 'dose': float(np.linalg.norm(w_rand)), 'feasible': True},
            _eval_write(w_rand),
        )

        # 8) Dose grid on hybrid_full direction
        for d_val in dose_grid:
            dd = (raw_h * (d_val / raw_h_norm)) if raw_h_norm > 1e-9 else np.zeros_like(raw_h)
            ev_d = eval_state_direction(model, batch, dd, max_delay)
            dg = dose_agg[str(d_val)]
            dg['total'] += 1
            dg['target_hits'] += int(ev_d['target_hit'])
            dg['prot_init_correct'] += int(ev_d['prot_init_correct'])
            dg['prot_corrupted'] += int(ev_d['prot_corrupted'])
            dg['linear_errors'].append(abs(ev_d['target_margin_delta'] - float(a @ dd)))

        # 9) Component cooperation & cancellation on best contiguous group
        coop = component_cooperation_metrics(model, r_jb['delta'], contig_groups[0])
        if coop['outside_write_fraction'] is not None:
            coop_outside_fracs.append(coop['outside_write_fraction'])
        if coop['group_outside_cosine'] is not None:
            coop_cosines.append(coop['group_outside_cosine'])
        if coop['cancellation_K'] is not None:
            coop_cancels.append(coop['cancellation_K'])

        edge_records.append({
            'edge': [list(edge[0]), list(edge[1]), edge[2]],
            'natural_norm': raw_h_norm,
            'target_linear': float(obs['target_linear']),
            'target_actual': float(obs['target_actual']),
            'finite_dose_error': float(obs['finite_dose_error']),
            'high_zero_gain': r_hz['gain'],
            'low_zero_gain': r_lz['gain'],
            'joint_zero_gain': r_jz['gain'],
            'joint_b025_gain': r_jb['gain'],
            'sham_feasible': sham_res['feasible'],
            'sham_null_dim': sham_res['null_dim'],
        })

    # Summarize support families
    family_summary = {}
    for name, st in family_stats.items():
        n_tot = max(st['total'], 1)
        pic = st['prot_init_correct']
        family_summary[name] = {
            'total': st['total'],
            'feasible': st['feasible'],
            'feasible_rate': st['feasible'] / n_tot,
            'mean_linear_gain': float(np.mean(st['linear_gains'])) if st['linear_gains'] else 0.0,
            'mean_collateral_norm': float(np.mean(st['collaterals'])) if st['collaterals'] else 0.0,
            'target_hits': st['target_hits'],
            'target_accuracy': st['target_hits'] / n_tot,
            'prot_init_correct': pic,
            'prot_corrupted': st['prot_corrupted'],
            'collateral_corruption_rate': (st['prot_corrupted'] / pic) if pic > 0 else 1.0,
        }

    frontier_summary = {}
    for b_key, fb in frontier_agg.items():
        n_tot = max(len(fb['gains']), 1)
        pic = fb['prot_init_correct']
        frontier_summary[b_key] = {
            'feasible_rate': fb['feasible'] / n_tot,
            'mean_linear_gain': float(np.mean(fb['gains'])),
            'mean_collateral_norm': float(np.mean(fb['collaterals'])),
            'target_accuracy': fb['target_hits'] / n_tot,
            'collateral_corruption_rate': (fb['prot_corrupted'] / pic) if pic > 0 else 1.0,
        }

    dose_summary = {}
    for d_key, dg in dose_agg.items():
        n_tot = max(dg['total'], 1)
        pic = dg['prot_init_correct']
        dose_summary[d_key] = {
            'target_accuracy': dg['target_hits'] / n_tot,
            'collateral_corruption_rate': (dg['prot_corrupted'] / pic) if pic > 0 else 1.0,
            'mean_finite_dose_error': float(np.mean(dg['linear_errors'])) if dg['linear_errors'] else 0.0,
        }

    # Engineering admission check per PROTOCOL.md Phase B:
    # Target improvement over random with collateral corruption <= 10% of initially correct
    # protected queries, plus improvement over RP03-style contiguous V donor span at equal dose.
    best_sel = family_summary['joint_budget_025']
    rand_ref = family_summary['random_r8']
    contig_ref = family_summary['contiguous_v_high_r4']
    zero_sel = family_summary['joint_zero_col']
    # Check both zero-collateral and budgeted-0.25 joint selective control
    admitted_zero = (
        zero_sel['target_accuracy'] > rand_ref['target_accuracy']
        and zero_sel['collateral_corruption_rate'] <= 0.10
        and zero_sel['target_accuracy'] > contig_ref['target_accuracy']
    )
    admitted_budget = (
        best_sel['target_accuracy'] > rand_ref['target_accuracy']
        and best_sel['collateral_corruption_rate'] <= 0.10
        and best_sel['target_accuracy'] > contig_ref['target_accuracy']
    )
    admission_passed = bool(admitted_zero or admitted_budget)

    lin_corr = None
    if len(linear_preds) >= 3 and np.std(linear_preds) > 1e-12 and np.std(actual_deltas) > 1e-12:
        lin_corr = float(pearsonr(linear_preds, actual_deltas)[0])

    return {
        'root': root,
        'tag': tag,
        'checkpoint_sha256': ckpt_sha,
        'factor_frame_sha256': factor_sha,
        'delays': list(delays),
        'dose': dose,
        'n_edges': n_edges,
        'n_svd_edges': n_svd_edges,
        'svd_top8_singular_values': svd_s[:8].tolist(),
        'site_capacity_means': {k: float(np.mean(v)) for k, v in site_caps.items()},
        'support_families': family_summary,
        'sccg_frontier': frontier_summary,
        'dose_grid': dose_summary,
        'projection_residuals': {
            'v_high_mean': float(np.mean(proj_res_v_high)) if proj_res_v_high else None,
            'svd_r4_mean': float(np.mean(proj_res_svd4)) if proj_res_svd4 else None,
            'svd_r8_mean': float(np.mean(proj_res_svd8)) if proj_res_svd8 else None,
        },
        'finite_dose_fidelity': {
            'mean_abs_error': float(np.mean(finite_dose_errs)) if finite_dose_errs else None,
            'pearson_linear_vs_actual': lin_corr,
        },
        'component_cooperation': {
            'mean_outside_write_fraction': float(np.mean(coop_outside_fracs)) if coop_outside_fracs else None,
            'mean_group_outside_cosine': float(np.mean(coop_cosines)) if coop_cosines else None,
            'mean_cancellation_K': float(np.mean(coop_cancels)) if coop_cancels else None,
        },
        'admission': {
            'passed': admission_passed,
            'zero_collateral_passed': admitted_zero,
            'budget_025_passed': admitted_budget,
            'status': 'SELECTIVE_CONTROL_ADMISSION_PASSED' if admission_passed else 'SELECTIVE_CONTROL_ADMISSION_FAILED',
        },
        'cpu_seconds': time.process_time() - t_start,
        'edges': edge_records,
    }


@torch.no_grad()
def discover_slot_projectors(model: ComponentTRM, root: int, fit_edges: tuple, *,
                             delay: int = 2, dose: float = 0.75,
                             n_edges: int = 32,
                             cache: MechanisticCache | None = None) -> tuple[Tensor, dict[str, Any]]:
    """Discover per-slot rank-4 joint-state projectors from TRAIN-only fit edges.

    Uses TRAIN-only hybrid displacements per slot (top-4 SVD basis per edited slot),
    detached from parameters so the projector defines a fixed physical subspace
    for selective transport and causal auxiliary / feedback objectives.
    """
    sample = stratified_edge_sample(fit_edges, n_edges, seed=seed_for(root, 'slot_proj', 0))
    by_slot = {s: [] for s in range(4)}
    for i, edge in enumerate(sample):
        b = cached_edge_batch(
            (edge,), root=root, seed=seed_for(root, 'slot_proj_b', i),
            delay=delay, split='train', cache=cache,
        )
        sr = model.prefix_state(b.receiver[:1]).detach().double().reshape(-1)
        sh = model.prefix_state(b.hybrid[:1]).detach().double().reshape(-1)
        by_slot[edge[2]].append((sh - sr).cpu().numpy())

    dim = 2 * model.width
    projs = []
    ranks = []
    for s in range(4):
        mat = np.stack(by_slot[s], axis=1)  # [2w, k]
        u, sv, _ = np.linalg.svd(mat, full_matrices=False)
        k = min(4, int(np.count_nonzero(sv > 1e-10 * sv[0]))) if len(sv) and sv[0] > 0 else 0
        ranks.append(k)
        if k > 0:
            q = u[:, :k]
            P = q @ q.T
        else:
            P = np.zeros((dim, dim), dtype=np.float64)
        projs.append(torch.as_tensor(P, dtype=torch.float32))
    proj_tensor = torch.stack(projs, dim=0).detach()  # [4, 2w, 2w]
    return proj_tensor, {'slot_ranks': ranks, 'n_edges': len(sample), 'delay': delay}


def ordinary_feedback_loss(model: ComponentTRM, fb_batch) -> Tensor:
    """Ordinary task cross-entropy on TRAIN feedback cohort (receiver + hybrid)."""
    x = torch.cat((fb_batch.receiver, fb_batch.hybrid), dim=0)
    y = torch.cat((fb_batch.labels_r, fb_batch.labels_h), dim=0)
    return F.cross_entropy(model(x), y)


def mechanistic_feedback_loss(model: ComponentTRM, fb_batch, slot_projectors: Tensor, *,
                              dose: float = 0.75,
                              w_ordinary: float = 1.0,
                              w_target: float = 0.5,
                              w_preserve: float = 0.5) -> Tensor:
    """Frozen mechanistic feedback objective combining ordinary CE, targeted transport,
    and unaffected-answer preservation under fixed-dose joint-state intervention.

    All inputs come strictly from the TRAIN feedback edge view.
    - Ordinary CE on receiver & hybrid
    - Targeted transport: fixed-dose projected hybrid-minus-receiver state edit
      continued via exact suffix rollout, scoring cross-entropy toward ``labels_h``
      on the edited slot (``query == edit_slot``).
    - Unaffected preservation: on unaffected addresses (``query != edit_slot``),
      KL divergence from clean receiver probabilities PLUS cross-entropy toward
      the clean receiver label ``labels_r``.
    """
    n = len(fb_batch.labels_r)
    x_cat = torch.cat((fb_batch.receiver, fb_batch.hybrid), dim=0)
    y_cat = torch.cat((fb_batch.labels_r, fb_batch.labels_h), dim=0)
    out = model.rollout(x_cat)
    loss_ord = F.cross_entropy(out['logits'], y_cat)

    # Post-write joint states (step 3 = after 4 writes)
    sr = out['states'][:n, 3]
    sh = out['states'][n:, 3].detach()
    proj = slot_projectors.detach()[fb_batch.edit_slot]  # [n, 2w, 2w]
    raw = torch.bmm((sh - sr)[:, None, :], proj).squeeze(1)
    raw_norm = torch.linalg.vector_norm(raw, dim=-1)
    valid = raw_norm > 1e-9
    scale = torch.where(valid, dose / raw_norm.clamp_min(1e-9), torch.zeros_like(raw_norm))
    delta = raw * scale[:, None]

    # Exact suffix-only continuation from edited post-write state
    x_suffix = fb_batch.receiver[:, 4:]
    edit_logits = model.rollout_suffix(x_suffix, sr + delta, prefix_steps=4)['logits']
    clean_logits = out['logits'][:n].detach()

    targeted = fb_batch.query == fb_batch.edit_slot
    unaffected = ~targeted
    loss_target = F.cross_entropy(edit_logits[targeted], fb_batch.labels_h[targeted])
    # Preservation: KL(clean || edited) + CE(edited, labels_r) on unaffected slots
    p_clean = F.softmax(clean_logits[unaffected], dim=-1)
    log_p_edit = F.log_softmax(edit_logits[unaffected], dim=-1)
    kl_preserve = F.kl_div(log_p_edit, p_clean, reduction='batchmean')
    ce_preserve = F.cross_entropy(edit_logits[unaffected], fb_batch.labels_r[unaffected])
    loss_preserve = 0.5 * (kl_preserve + ce_preserve)

    return w_ordinary * loss_ord + w_target * loss_target + w_preserve * loss_preserve


def evaluate_candidate_prediction_quality(
    model: ComponentTRM,
    optimizer: torch.optim.Optimizer,
    candidate_batches: list,
    fb_batch,
    slot_projectors: Tensor,
    *,
    clip: float = 1.0,
    dose: float = 0.75,
    mode: str = 'mechanistic',
    cache: MechanisticCache | None = None,
) -> dict[str, Any]:
    """Compare predicted_gain vs realized_gain across candidate mini-batches via probe_update."""
    if mode not in ('ordinary', 'mechanistic'):
        raise ValueError('mode must be ordinary or mechanistic')

    def cal_fn():
        if mode == 'ordinary':
            return ordinary_feedback_loss(model, fb_batch)
        return mechanistic_feedback_loss(model, fb_batch, slot_projectors, dose=dose)

    # Optionally compute & cache calibration gradient once for this exact checkpoint/objective/batch
    cached_grad = None
    cached_loss0 = None
    cal_build_cpu = 0.0
    params = tuple(model.parameters())
    if cache is not None and cache.max_bytes > 0:
        ckpt = model_state_sha256(model)
        fframe = model.factor_frame_digest()
        cache.bind_checkpoint(ckpt, fframe)
        ckey = CacheKey(
            namespace='cal_grad',
            checkpoint_sha256=ckpt,
            module_id='ComponentTRM.all_params',
            visit_spec=f'dose={dose}',
            input_fingerprint=fb_batch.fingerprint(),
            objective_id=f'cal_{mode}',
            factor_frame_sha256=fframe,
        )
        hit = cache.get(ckey)
        if hit is not None:
            cached_grad, cached_loss0 = hit
        else:
            t0 = time.process_time()
            model.eval()
            loss0 = cal_fn()
            g = tuple(
                None if gi is None else gi.detach().clone()
                for gi in torch.autograd.grad(loss0, params, allow_unused=True)
            )
            l0_val = float(loss0.detach())
            cal_build_cpu = time.process_time() - t0
            cache.put(ckey, (g, l0_val), compute_cpu_seconds=cal_build_cpu)
            cached_grad, cached_loss0 = g, l0_val

    rows = []
    for cb in candidate_batches:
        def train_fn(b=cb):
            x = torch.cat((b.receiver, b.hybrid), dim=0)
            y = torch.cat((b.labels_r, b.labels_h), dim=0)
            return F.cross_entropy(model(x), y)

        res = probe_update(
            model, optimizer, train_fn, cal_fn, clip=clip,
            cached_cal_grad=cached_grad, cached_cal_loss0=cached_loss0,
        )
        res['utility'] = res['predicted_gain'] / max(res['cpu_seconds'], 1e-9)
        rows.append(res)

    preds = np.array([r['predicted_gain'] for r in rows], dtype=np.float64)
    reals = np.array([r['realized_gain'] for r in rows], dtype=np.float64)
    utils = np.array([r['utility'] for r in rows], dtype=np.float64)
    rho = float(spearmanr(preds, reals).statistic) if np.std(preds) > 1e-12 and np.std(reals) > 1e-12 else 0.0
    pr = float(pearsonr(preds, reals).statistic) if np.std(preds) > 1e-12 and np.std(reals) > 1e-12 else 0.0
    sign_acc = float(np.mean((preds > 0) == (reals > 0)))
    top4 = np.argsort(-utils)[:min(4, len(rows))]
    return {
        'mode': mode,
        'n_candidates': len(rows),
        'spearman_rho': rho,
        'pearson_r': pr,
        'sign_accuracy': sign_acc,
        'mean_predicted_gain': float(np.mean(preds)),
        'mean_realized_gain': float(np.mean(reals)),
        'top4_mean_realized_gain': float(np.mean(reals[top4])),
        'top4_positive_fraction': float(np.mean(reals[top4] > 0)),
        'total_probe_cpu_seconds': float(sum(r['cpu_seconds'] for r in rows) + cal_build_cpu),
        'rows': rows,
    }
