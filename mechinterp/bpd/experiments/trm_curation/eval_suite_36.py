"""36 Low-Difficulty TRM Evaluations Suite (BPD-TRM-CURATE-02 Overnight Exploration).

Six families x six evaluations = 36 distinct low-difficulty evaluations tailored
for compact weight-tied TRMs (5,380-param ComponentTRM):

  Family 1 (E01-E06): Delayed-Memory Horizon Ladder (d=0, 2, 7, 15, 31, 63/127)
  Family 2 (E07-E12): Address Binding, Noise Robustness & Write-Order Invariance
  Family 3 (E13-E18): Distractor Stress, Slot Collisions, Calibration & Transition Shift
  Family 4 (E19-E24): Causal Selectivity, High/Low Sites, SCCG-01 & Cancellation
  Family 5 (E25-E30): 6 Low-Difficulty Standalone Algorithmic State-Tracking Benchmarks
  Family 6 (E31-E36): 6 Low-Difficulty Iterative-Refinement / Structured TRM Benchmarks

Zero access to any sealed/protected evaluation cohort (e.g., ARG-03-N). Every
benchmark uses strictly disjoint synthetic train/eval partitions and deterministic seeds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..training_steering.data import make_pairs, root_encoding
from ..training_steering_stability.data import binding_batch
from ..training_steering_stability.model import ComponentStateMemory
from .campaign import CampaignRecipe, DEVELOPMENT_ROOTS, dump_json
from .data import edge_batch
from .memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap
from .model import ComponentTRM
from .observer import observe_edge
from .reference import budgeted_selective_direction, selective_direction, semantic_edges
from .scheduler import edge_views
from .signal_bank import component_cooperation_metrics, eval_state_direction, seed_for, stratified_edge_sample


# ---------------------------------------------------------------------------
# Families 1-4 (E01-E24): Evaluating trained delayed-memory TRM checkpoints
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_custom_memory_batch(model: ComponentTRM, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    logits = model(x)
    preds = logits.argmax(-1)
    probs = F.softmax(logits, dim=-1)
    conf, _ = probs.max(dim=-1)
    top2 = logits.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).mean()
    # Expected Calibration Error (10 bins)
    acc_bits = (preds == y).float()
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        mask = (conf >= lo) & (conf < hi if b < 9 else conf <= hi)
        if bool(mask.any()):
            ece += float(mask.float().mean()) * abs(float(acc_bits[mask].mean()) - float(conf[mask].mean()))
    return {
        'accuracy': float(acc_bits.mean()),
        'cross_entropy': float(F.cross_entropy(logits, y)),
        'mean_margin': float(margin),
        'ece': float(ece),
    }


@torch.no_grad()
def evaluate_families_1_to_4(dev_dir: Path, recipe: CampaignRecipe,
                             arms: tuple[str, ...] = ('A_uniform', 'B_loss_mined', 'C_ord_grad', 'D_mech_grad', 'E_shuffled', 'G_cpu_reinvest'),
                             roots: tuple[int, ...] = DEVELOPMENT_ROOTS,
                             n_eval: int = 256) -> dict[str, Any]:
    """Run E01-E24 across all available trained roots and arms."""
    eval_edges = semantic_edges('eval')
    train_trans = {(s, r[s], h[s]) for r, h, s in semantic_edges('train')}
    eval_trans = {(s, r[s], h[s]) for r, h, s in eval_edges}

    per_eval: dict[str, dict[str, list[float]]] = {
        f'E{i:02d}': {a: [] for a in arms} for i in range(1, 25)
    }
    eval_meta = {
        'E01': ('mem_delay_0', 'Family 1: Horizon Ladder', 'accuracy'),
        'E02': ('mem_delay_2', 'Family 1: Horizon Ladder', 'accuracy'),
        'E03': ('mem_delay_7', 'Family 1: Horizon Ladder', 'accuracy'),
        'E04': ('mem_delay_15', 'Family 1: Horizon Ladder', 'accuracy'),
        'E05': ('mem_delay_31', 'Family 1: Horizon Ladder', 'accuracy'),
        'E06': ('mem_delay_63_127_mean', 'Family 1: Horizon Ladder', 'accuracy'),
        'E07': ('binding_all_distinct_d7', 'Family 2: Binding & Nuisance', 'accuracy'),
        'E08': ('binding_all_distinct_d31', 'Family 2: Binding & Nuisance', 'accuracy'),
        'E09': ('noise_clean_000_d31', 'Family 2: Binding & Nuisance', 'accuracy'),
        'E10': ('noise_elevated_010_d31', 'Family 2: Binding & Nuisance', 'accuracy'),
        'E11': ('noise_high_020_d31', 'Family 2: Binding & Nuisance', 'accuracy'),
        'E12': ('write_order_invariance_1_minus_gap', 'Family 2: Binding & Nuisance', 'score'),
        'E13': ('distractor_adversarial_same_addr_d31', 'Family 3: Distractor & Stress', 'accuracy'),
        'E14': ('distractor_repeated_value_flood_d31', 'Family 3: Distractor & Stress', 'accuracy'),
        'E15': ('distractor_zero_silent_d31', 'Family 3: Distractor & Stress', 'accuracy'),
        'E16': ('slot_collision_3of4_minority_d15', 'Family 3: Distractor & Stress', 'accuracy'),
        'E17': ('calibration_1_minus_ece_d31', 'Family 3: Distractor & Stress', 'score'),
        'E18': ('transition_shift_eval_edge_acc_d31', 'Family 3: Distractor & Stress', 'accuracy'),
        'E19': ('causal_full_hybrid_oracle_d31', 'Family 4: Causal Selectivity', 'target_acc'),
        'E20': ('causal_high_minus_low_capacity_ratio', 'Family 4: Causal Selectivity', 'score'),
        'E21': ('causal_sccg_zero_col_target_d31', 'Family 4: Causal Selectivity', 'target_acc'),
        'E22': ('causal_sccg_budget_025_net_selectivity', 'Family 4: Causal Selectivity', 'target_minus_cor'),
        'E23': ('causal_linear_vs_nonlinear_fidelity', 'Family 4: Causal Selectivity', 'score'),
        'E24': ('causal_component_non_cancellation_1_minus_K', 'Family 4: Causal Selectivity', 'score'),
    }

    used_roots = []
    for root in roots:
        r_dir = dev_dir / f'root_{root}'
        if not r_dir.is_dir():
            continue
        used_roots.append(root)
        addr_enc, val_enc = root_encoding(root, 4, 4)

        # Build shared root-specific eval batches once so all arms are strictly paired
        b_delays = {
            d: make_pairs(seed_for(root, f'e36_d{d}', 0), n_eval, root_seed=root,
                          delay=d, split='eval', noise=recipe.noise)
            for d in (0, 2, 7, 15, 31, 63, 127)
        }
        bind_d7 = binding_batch(root, 'e36_bind7', 128, 7, 'eval', recipe)
        bind_d31 = binding_batch(root, 'e36_bind31', 128, 31, 'eval', recipe)
        b_n00 = make_pairs(seed_for(root, 'e36_n00', 0), n_eval, root_seed=root, delay=31, split='eval', noise=0.0)
        b_n10 = make_pairs(seed_for(root, 'e36_n10', 0), n_eval, root_seed=root, delay=31, split='eval', noise=0.10)
        b_n20 = make_pairs(seed_for(root, 'e36_n20', 0), n_eval, root_seed=root, delay=31, split='eval', noise=0.20)

        # Distractor stress variants derived from b_delays[31]
        base31 = b_delays[31]
        # E13: Adversarial same-address wrong-value distractors
        x_adv = base31.receiver.clone()
        x_adv[:, 4:-1, 3:] = 0.0
        for i in range(n_eval):
            q_slot = int(base31.query[i])
            wrong_v = (int(base31.labels_r[i]) + 1) % 4
            x_adv[i, 4:-1, 3 + addr_enc[q_slot]] = 1.0
            x_adv[i, 4:-1, 7 + val_enc[wrong_v]] = 1.0
        # E14: Repeated single-value flood across distractors
        x_flood = base31.receiver.clone()
        x_flood[:, 4:-1, 3:] = 0.0
        x_flood[:, 4:-1, 3 + addr_enc[0]] = 1.0
        x_flood[:, 4:-1, 7 + val_enc[0]] = 1.0
        # E15: Silent zero distractors (only NOOP bit active)
        x_zero = base31.receiver.clone()
        x_zero[:, 4:-1, 3:] = 0.0

        # E16: Minority 3-of-4 collision memories at d=15
        b15_pool = make_pairs(seed_for(root, 'e36_col15', 0), 1024, root_seed=root, delay=15, split='eval', noise=recipe.noise)
        col_idx = []
        for i in range(1024):
            mem = b15_pool.values_r[i].tolist()
            q_v = int(b15_pool.labels_r[i])
            if mem.count(q_v) == 1 and max(mem.count(v) for v in range(4)) == 3:
                col_idx.append(i)
        col_idx = col_idx[:128]

        # E18 + E19-E24: Eval-edge cohort (12 stratified eval edges for causal probes)
        sample_e = stratified_edge_sample(eval_edges, 12, seed=seed_for(root, 'e36_causal', 0))
        causal_batches = [
            edge_batch((e,), root=root, seed=seed_for(root, 'e36_cb', i), delay=31, split='eval', noise=recipe.noise)
            for i, e in enumerate(sample_e)
        ]

        for arm in arms:
            ckpt = r_dir / f'{arm}.pt'
            if not ckpt.is_file():
                continue
            model = ComponentTRM(width=recipe.width, components=recipe.components, low_cycles=recipe.low_cycles)
            model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
            model.eval()

            # Family 1
            for code, d in (('E01', 0), ('E02', 2), ('E03', 7), ('E04', 15), ('E05', 31)):
                per_eval[code][arm].append(_eval_custom_memory_batch(model, b_delays[d].receiver, b_delays[d].labels_r)['accuracy'])
            a63 = _eval_custom_memory_batch(model, b_delays[63].receiver, b_delays[63].labels_r)['accuracy']
            a127 = _eval_custom_memory_batch(model, b_delays[127].receiver, b_delays[127].labels_r)['accuracy']
            per_eval['E06'][arm].append(0.5 * (a63 + a127))

            # Family 2
            per_eval['E07'][arm].append(_eval_custom_memory_batch(model, bind_d7[0], bind_d7[1])['accuracy'])
            per_eval['E08'][arm].append(_eval_custom_memory_batch(model, bind_d31[0], bind_d31[1])['accuracy'])
            per_eval['E09'][arm].append(_eval_custom_memory_batch(model, b_n00.receiver, b_n00.labels_r)['accuracy'])
            per_eval['E10'][arm].append(_eval_custom_memory_batch(model, b_n10.receiver, b_n10.labels_r)['accuracy'])
            per_eval['E11'][arm].append(_eval_custom_memory_batch(model, b_n20.receiver, b_n20.labels_r)['accuracy'])

            # E12: Write-order primacy/recency gap on d=31
            x31 = base31.receiver
            addr_inv = addr_enc.argsort()
            first_addr = addr_inv[x31[:, 0, 3:7].argmax(-1)]
            last_addr = addr_inv[x31[:, 3, 3:7].argmax(-1)]
            preds31 = model(x31).argmax(-1)
            corr31 = (preds31 == base31.labels_r).float()
            m_first = base31.query == first_addr
            m_last = base31.query == last_addr
            acc_f = float(corr31[m_first].mean()) if bool(m_first.any()) else 0.0
            acc_l = float(corr31[m_last].mean()) if bool(m_last.any()) else 0.0
            per_eval['E12'][arm].append(1.0 - abs(acc_f - acc_l))

            # Family 3
            per_eval['E13'][arm].append(_eval_custom_memory_batch(model, x_adv, base31.labels_r)['accuracy'])
            per_eval['E14'][arm].append(_eval_custom_memory_batch(model, x_flood, base31.labels_r)['accuracy'])
            per_eval['E15'][arm].append(_eval_custom_memory_batch(model, x_zero, base31.labels_r)['accuracy'])
            if col_idx:
                per_eval['E16'][arm].append(
                    _eval_custom_memory_batch(model, b15_pool.receiver[col_idx], b15_pool.labels_r[col_idx])['accuracy']
                )
            else:
                per_eval['E16'][arm].append(0.0)
            cal31 = _eval_custom_memory_batch(model, base31.receiver, base31.labels_r)
            per_eval['E17'][arm].append(1.0 - cal31['ece'])

            # E18: Eval-edge all-4-query accuracy under transition-support shift
            e18_hits = []
            for cb in causal_batches:
                p_c = model(cb.receiver).argmax(-1)
                e18_hits.append(float((p_c == cb.labels_r).float().mean()))
            per_eval['E18'][arm].append(float(np.mean(e18_hits)))

            # Family 4: Causal Selectivity (E19-E24)
            w = model.width
            B_h = np.vstack([np.eye(w), np.zeros((w, w))])
            B_l = np.vstack([np.zeros((w, w)), np.eye(w)])
            B_j = np.eye(2 * w)
            oracle_hits, zero_hits, b25_hits, b25_pic, b25_cor = 0, 0, 0, 0, 0
            hl_ratios, lin_fids, non_cancels = [], [], []
            for cb in causal_batches:
                with torch.enable_grad():
                    obs = observe_edge(model, cb, (7, 31), dose=recipe.dose, use_suffix=True)
                a_v = obs['target'].double().cpu().numpy()
                C_m = obs['protected'].double().cpu().numpy()
                raw_h = obs['raw_hybrid_displacement'].double().cpu().numpy()
                ev_or = eval_state_direction(model, cb, raw_h, 31)
                oracle_hits += int(ev_or['target_hit'])

                rh = budgeted_selective_direction(a_v, C_m, B_h, recipe.dose, 0.25)
                rl = budgeted_selective_direction(a_v, C_m, B_l, recipe.dose, 0.25)
                rz = selective_direction(a_v, C_m, B_j, recipe.dose)
                rb = budgeted_selective_direction(a_v, C_m, B_j, recipe.dose, 0.25)
                hl_ratios.append(rh['gain'] / max(rh['gain'] + rl['gain'], 1e-9))

                ev_z = eval_state_direction(model, cb, rz['delta'], 31)
                ev_b = eval_state_direction(model, cb, rb['delta'], 31)
                zero_hits += int(ev_z['target_hit'])
                b25_hits += int(ev_b['target_hit'])
                b25_pic += int(ev_b['prot_init_correct'])
                b25_cor += int(ev_b['prot_corrupted'])

                # Linear fidelity: 1 / (1 + |actual - linear| / max(|actual|, 0.5))
                err_l = abs(obs['target_actual'] - obs['target_linear'])
                lin_fids.append(1.0 / (1.0 + err_l / max(abs(obs['target_actual']), 0.5)))

                coop = component_cooperation_metrics(model, rb['delta'], (0, 1, 2, 3))
                if coop['cancellation_K'] is not None:
                    non_cancels.append(1.0 - coop['cancellation_K'])

            n_cb = max(len(causal_batches), 1)
            per_eval['E19'][arm].append(oracle_hits / n_cb)
            per_eval['E20'][arm].append(float(np.mean(hl_ratios)))
            per_eval['E21'][arm].append(zero_hits / n_cb)
            cor_rate = (b25_cor / b25_pic) if b25_pic > 0 else 1.0
            per_eval['E22'][arm].append((b25_hits / n_cb) - cor_rate)
            per_eval['E23'][arm].append(float(np.mean(lin_fids)))
            per_eval['E24'][arm].append(float(np.mean(non_cancels)) if non_cancels else 0.0)

    summary_rows = []
    for code in sorted(eval_meta.keys()):
        name, fam, metric = eval_meta[code]
        arm_means = {
            a: (float(np.mean(per_eval[code][a])) if per_eval[code][a] else None)
            for a in arms
        }
        d_minus_a = (
            (arm_means['D_mech_grad'] - arm_means['A_uniform'])
            if (arm_means.get('D_mech_grad') is not None and arm_means.get('A_uniform') is not None)
            else None
        )
        summary_rows.append({
            'id': code, 'name': name, 'family': fam, 'metric': metric,
            'arm_means': arm_means, 'D_minus_A': d_minus_a,
        })
    return {
        'roots_evaluated': used_roots,
        'train_transition_types': len(train_trans),
        'eval_transition_types': len(eval_trans),
        'evals': summary_rows,
    }


# ---------------------------------------------------------------------------
# Families 5-6 (E25-E36): 12 Standalone Low-Difficulty TRM Micro-Benchmarks
# ---------------------------------------------------------------------------

def _gen_micro_task(task_id: str, seed: int, n: int, split: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Generate (x [n, T, 11], y [n], n_classes) for 12 low-difficulty TRM micro-tasks.

    All tasks map cleanly onto the exact 11-dim feature interface (input_dim=11,
    values=4) so the exact same 5,380-param ComponentTRM and matched 5,380-param
    GRU can be compared directly across all 12 tasks!
    Train/eval split uses deterministic parity/hash partition of canonical problem IDs.
    """
    g = torch.Generator(device='cpu').manual_seed(seed % (2**63 - 1))
    is_eval = (split == 'eval')

    if task_id == 'E25_parity_mod2':
        # Running binary parity: train len 8, eval len 16 (extrapolation!)
        T = 16 if is_eval else 8
        bits = torch.randint(0, 2, (n, T), generator=g)
        x = torch.zeros((n, T, 11))
        x[:, :, 0] = 1.0
        x[torch.arange(n)[:, None], torch.arange(T)[None, :], 3 + bits] = 1.0
        y = bits.sum(dim=1) % 2
        return x, y, 2

    if task_id == 'E26_mod4_accumulator':
        # Running mod-4 sum: train len 8, eval len 16
        T = 16 if is_eval else 8
        vals = torch.randint(0, 4, (n, T), generator=g)
        x = torch.zeros((n, T, 11))
        x[:, :, 0] = 1.0
        x[torch.arange(n)[:, None], torch.arange(T)[None, :], 7 + vals] = 1.0
        y = vals.sum(dim=1) % 4
        return x, y, 4

    if task_id == 'E27_copy_retrieve_2slot':
        # Easy 2-slot associative memory: 2 writes + 6 (train) / 14 (eval) distractors + query
        delay = 14 if is_eval else 6
        T = 2 + delay + 1
        mems = torch.randint(0, 4, (n, 2), generator=g)
        # Disjoint split on sum mod 3 (eval: 0, train: 1,2)
        for _ in range(8):
            bad = ((mems[:, 0] + mems[:, 1]) % 3 == 0) != is_eval
            if not bool(bad.any()):
                break
            mems[bad] = torch.randint(0, 4, (int(bad.sum()), 2), generator=g)
        q = torch.randint(0, 2, (n,), generator=g)
        x = torch.zeros((n, T, 11))
        x[:, :2, 0] = 1.0
        x[:, 2:-1, 1] = 1.0
        x[:, -1, 2] = 1.0
        row = torch.arange(n)
        x[row, 0, 3 + 0] = 1.0; x[row, 0, 7 + mems[:, 0]] = 1.0
        x[row, 1, 3 + 1] = 1.0; x[row, 1, 7 + mems[:, 1]] = 1.0
        for t in range(2, 2 + delay):
            da = torch.randint(0, 2, (n,), generator=g)
            dv = torch.randint(0, 4, (n,), generator=g)
            x[row, t, 3 + da] = 1.0; x[row, t, 7 + dv] = 1.0
        x[row, -1, 3 + q] = 1.0
        y = mems[row, q]
        return x, y, 4

    if task_id == 'E28_dynamic_overwrite_register':
        # 4-slot register with 6 (train) / 10 (eval) write/overwrite ops, last-write-wins
        writes = 10 if is_eval else 6
        T = writes + 1
        state = torch.zeros((n, 4), dtype=torch.long)
        x = torch.zeros((n, T, 11))
        x[:, :writes, 0] = 1.0
        x[:, -1, 2] = 1.0
        row = torch.arange(n)
        for t in range(writes):
            a = torch.randint(0, 4, (n,), generator=g)
            v = torch.randint(0, 4, (n,), generator=g)
            x[row, t, 3 + a] = 1.0
            x[row, t, 7 + v] = 1.0
            state[row, a] = v
        q = torch.randint(0, 4, (n,), generator=g)
        x[row, -1, 3 + q] = 1.0
        y = state[row, q]
        return x, y, 4

    if task_id == 'E29_dyck1_bounded_depth':
        # Bounded Dyck-1 depth tracking in {0,1,2,3}: train len 8, eval len 16
        T = 16 if is_eval else 8
        x = torch.zeros((n, T, 11))
        x[:, :, 0] = 1.0
        depth = torch.zeros(n, dtype=torch.long)
        row = torch.arange(n)
        for t in range(T):
            step = torch.randint(0, 2, (n,), generator=g)  # 0 = '(', 1 = ')'
            # Clamp to valid bounded depth [0, 3]
            inc = (step == 0) & (depth < 3)
            dec = (step == 1) & (depth > 0)
            act = torch.where(inc, torch.zeros_like(step), torch.where(dec, torch.ones_like(step), torch.full_like(step, 2)))
            depth = depth + inc.long() - dec.long()
            x[row, t, 3 + act] = 1.0
        return x, depth, 4

    if task_id == 'E30_majority_vote_4sym':
        # Majority symbol among 4 classes over length 9 (train) / 13 (eval) with unique strict winner
        T = 13 if is_eval else 9
        seqs = torch.empty((n, T), dtype=torch.long)
        y = torch.empty(n, dtype=torch.long)
        for i in range(n):
            winner = int(torch.randint(0, 4, (1,), generator=g))
            # Put winner in >= (T//2 + 1) positions so majority is well-defined and low-difficulty
            k_win = T // 2 + 1
            others = [v for v in range(4) if v != winner]
            arr = [winner] * k_win + [others[int(torch.randint(0, 3, (1,), generator=g))] for _ in range(T - k_win)]
            perm = torch.randperm(T, generator=g).tolist()
            seqs[i] = torch.tensor([arr[p] for p in perm])
            y[i] = winner
        x = torch.zeros((n, T, 11))
        x[:, :, 0] = 1.0
        x[torch.arange(n)[:, None], torch.arange(T)[None, :], 7 + seqs] = 1.0
        return x, y, 4

    if task_id == 'E31_2hop_pointer_chase':
        # 4-register 2-hop pointer chase: write reg[0..3] in {0..3}, query q -> return reg[reg[q]]!
        # Add 2 blank thinking tokens so weight-tied TRM can iterate!
        T = 4 + 1 + 2
        regs = torch.randint(0, 4, (n, 4), generator=g)
        # Disjoint train/eval split by base-4 code mod 5
        for _ in range(16):
            code = regs[:, 0] + 4 * regs[:, 1] + 16 * regs[:, 2] + 64 * regs[:, 3]
            bad = ((code % 5 <= 1) != is_eval)
            if not bool(bad.any()):
                break
            regs[bad] = torch.randint(0, 4, (int(bad.sum()), 4), generator=g)
        q = torch.randint(0, 4, (n,), generator=g)
        x = torch.zeros((n, T, 11))
        x[:, :4, 0] = 1.0
        x[:, 4, 2] = 1.0
        x[:, 5:, 1] = 1.0  # 2 scratchpad/think tokens
        row = torch.arange(n)
        for s in range(4):
            x[row, s, 3 + s] = 1.0
            x[row, s, 7 + regs[:, s]] = 1.0
        x[row, 4, 3 + q] = 1.0
        hop1 = regs[row, q]
        y = regs[row, hop1]
        return x, y, 4

    if task_id == 'E32_latin4_1blank':
        # 4x4 Latin-square single-blank deduction (low-difficulty mini-Sudoku!)
        # Present 3 filled cells of a row/col + query blank -> deduce missing symbol in {0,1,2,3}
        T = 4 + 2  # 3 clues + 1 query + 2 refine steps
        perms = torch.stack([torch.randperm(4, generator=g) for _ in range(n)])
        q_pos = torch.randint(0, 4, (n,), generator=g)
        x = torch.zeros((n, T, 11))
        row = torch.arange(n)
        for i in range(n):
            qp = int(q_pos[i])
            clues = [c for c in range(4) if c != qp]
            for t, c in enumerate(clues):
                x[i, t, 0] = 1.0
                x[i, t, 3 + c] = 1.0
                x[i, t, 7 + int(perms[i, c])] = 1.0
            x[i, 3, 2] = 1.0
            x[i, 3, 3 + qp] = 1.0
            x[i, 4:, 1] = 1.0
        y = perms[row, q_pos]
        return x, y, 4

    if task_id == 'E33_latin4_2blank_coupled':
        # 4x4 mini-Sudoku 2-blank coupled constraint deduction:
        # Row has 2 known clues + 2 blanks; Column of queried blank supplies 1 extra disambiguating clue!
        T = 6  # 2 row clues + 1 col clue + 1 query + 2 refine tokens
        x = torch.zeros((n, T, 11))
        y = torch.empty(n, dtype=torch.long)
        for i in range(n):
            p = torch.randperm(4, generator=g).tolist()
            # Row clues at pos 0,1; blanks at pos 2,3 with true values p[2], p[3].
            # Col clue tells which of {p[2], p[3]} is FORBIDDEN at queried blank (so answer is the other!)
            x[i, 0, 0] = 1.0; x[i, 0, 3 + 0] = 1.0; x[i, 0, 7 + p[0]] = 1.0
            x[i, 1, 0] = 1.0; x[i, 1, 3 + 1] = 1.0; x[i, 1, 7 + p[1]] = 1.0
            # Col constraint (op bit 1 + pos 2 + forbidden val p[3]) -> pos 2 must be p[2]!
            q_idx = int(torch.randint(0, 2, (1,), generator=g))
            target_v = p[2] if q_idx == 0 else p[3]
            forbid_v = p[3] if q_idx == 0 else p[2]
            x[i, 2, 1] = 1.0; x[i, 2, 3 + (2 + q_idx)] = 1.0; x[i, 2, 7 + forbid_v] = 1.0
            x[i, 3, 2] = 1.0; x[i, 3, 3 + (2 + q_idx)] = 1.0
            x[i, 4:, 1] = 1.0
            y[i] = target_v
        return x, y, 4

    if task_id == 'E34_argmin4_index':
        # 4 distinct values in {0..3} written to slots 0..3 -> return index of slot holding 0 (argmin!)
        T = 6
        x = torch.zeros((n, T, 11))
        y = torch.empty(n, dtype=torch.long)
        for i in range(n):
            p = torch.randperm(4, generator=g).tolist()
            for s in range(4):
                x[i, s, 0] = 1.0; x[i, s, 3 + s] = 1.0; x[i, s, 7 + p[s]] = 1.0
            x[i, 4, 2] = 1.0
            x[i, 5, 1] = 1.0
            y[i] = p.index(0)
        return x, y, 4

    if task_id == 'E35_cellular_rule90_2step':
        # 4-cell periodic binary ring under 2 steps of Rule 90 (cell_next[i] = cell[i-1] XOR cell[i+1])
        # Query cell q -> output binary state after 2 steps (requires multi-step spatial XOR!)
        T = 7
        bits = torch.randint(0, 2, (n, 4), generator=g)
        q = torch.randint(0, 4, (n,), generator=g)
        x = torch.zeros((n, T, 11))
        row = torch.arange(n)
        for s in range(4):
            x[row, s, 0] = 1.0; x[row, s, 3 + s] = 1.0; x[row, s, 7 + bits[:, s]] = 1.0
        x[row, 4, 2] = 1.0; x[row, 4, 3 + q] = 1.0
        x[:, 5:, 1] = 1.0
        # Step 1 & Step 2 Rule 90
        b1 = bits.roll(1, dims=1) ^ bits.roll(-1, dims=1)
        b2 = b1.roll(1, dims=1) ^ b1.roll(-1, dims=1)
        y = b2[row, q]
        return x, y, 2

    if task_id == 'E36_dfa_4state_extrap':
        # 4-state DFA with 3 input symbols: s' = (s + a + 1) % 4 if a<2 else (3 - s)
        # Train len 8, eval len 20 (long-horizon state machine extrapolation!)
        T = 20 if is_eval else 8
        syms = torch.randint(0, 3, (n, T), generator=g)
        x = torch.zeros((n, T, 11))
        x[:, :, 0] = 1.0
        row = torch.arange(n)
        st = torch.zeros(n, dtype=torch.long)
        for t in range(T):
            a = syms[:, t]
            x[row, t, 3 + a] = 1.0
            st = torch.where(a < 2, (st + a + 1) % 4, 3 - st)
        return x, st, 4

    raise ValueError(f'unknown micro task {task_id}')


def run_micro_benchmark(task_id: str, roots: tuple[int, ...] = (29000, 29001), *,
                        steps: int = 450, batch_size: int = 64,
                        threads: int = 2) -> dict[str, Any]:
    """Train matched GRU-32, TRM-32(low_cycles=1), TRM-32(low_cycles=2), and Loss-Curated TRM-32."""
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    models_spec = [
        ('GRU_w32', lambda: ComponentStateMemory(width=32, components=32, input_dim=11, values=4), 'uniform'),
        ('TRM_c1_uniform', lambda: ComponentTRM(width=32, components=29, low_cycles=1, input_dim=11, values=4), 'uniform'),
        ('TRM_c2_uniform', lambda: ComponentTRM(width=32, components=29, low_cycles=2, input_dim=11, values=4), 'uniform'),
        ('TRM_c1_curated', lambda: ComponentTRM(width=32, components=29, low_cycles=1, input_dim=11, values=4), 'loss_curated'),
    ]
    agg = {name: {'eval_acc': [], 'train_acc': [], 'cpu_s': []} for name, _, _ in models_spec}

    for root in roots:
        x_ev, y_ev, n_cls = _gen_micro_task(task_id, seed_for(root, f'{task_id}_ev', 0), 512, 'eval')
        x_tr_eval, y_tr_eval, _ = _gen_micro_task(task_id, seed_for(root, f'{task_id}_trev', 0), 256, 'train')

        for name, ctor, mode in models_spec:
            torch.manual_seed(seed_for(root, f'micro_{task_id}_{name}', 0))
            m = ctor()
            opt = torch.optim.AdamW(m.parameters(), lr=0.003, weight_decay=0.0001)
            t0 = time.process_time()
            for step in range(steps):
                if mode == 'uniform':
                    x, y, _ = _gen_micro_task(task_id, seed_for(root, f'{task_id}_tr', step), batch_size, 'train')
                else:
                    # Low-cost hard-example + 20% exploration pool curation (2x pool -> select 1x)
                    xp, yp, _ = _gen_micro_task(task_id, seed_for(root, f'{task_id}_tr', step), batch_size * 2, 'train')
                    m.eval()
                    with torch.no_grad():
                        losses = F.cross_entropy(m(xp)[:, :n_cls], yp, reduction='none')
                    n_exp = max(1, int(0.2 * batch_size))
                    n_hard = batch_size - n_exp
                    top_idx = losses.argsort(descending=True)[:n_hard]
                    rem_mask = torch.ones(len(yp), dtype=torch.bool)
                    rem_mask[top_idx] = False
                    rem_idx = torch.where(rem_mask)[0][:n_exp]
                    sel = torch.cat((top_idx, rem_idx))
                    x, y = xp[sel], yp[sel]
                m.train()
                logits = m(x)[:, :n_cls]
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            cpu_s = time.process_time() - t0
            m.eval()
            with torch.no_grad():
                ev_acc = float((m(x_ev)[:, :n_cls].argmax(-1) == y_ev).float().mean())
                tr_acc = float((m(x_tr_eval)[:, :n_cls].argmax(-1) == y_tr_eval).float().mean())
            agg[name]['eval_acc'].append(ev_acc)
            agg[name]['train_acc'].append(tr_acc)
            agg[name]['cpu_s'].append(cpu_s)

    summary = {
        name: {
            'eval_acc': float(np.mean(v['eval_acc'])),
            'train_acc': float(np.mean(v['train_acc'])),
            'cpu_seconds': float(np.mean(v['cpu_s'])),
        }
        for name, v in agg.items()
    }
    trm_c1 = summary['TRM_c1_uniform']['eval_acc']
    trm_c2 = summary['TRM_c2_uniform']['eval_acc']
    gru = summary['GRU_w32']['eval_acc']
    cur = summary['TRM_c1_curated']['eval_acc']
    return {
        'id': task_id[:3],
        'name': task_id[4:],
        'family': 'Family 5: Algorithmic State-Tracking' if int(task_id[1:3]) <= 30 else 'Family 6: Iterative TRM Reasoning',
        'steps': steps,
        'roots': list(roots),
        'models': summary,
        'TRM_c1_minus_GRU_pp': (trm_c1 - gru) * 100.0,
        'TRM_c2_minus_c1_pp': (trm_c2 - trm_c1) * 100.0,
        'curated_minus_uniform_pp': (cur - trm_c1) * 100.0,
        'trm_appropriate_headroom': bool(0.35 <= max(trm_c1, trm_c2) <= 0.98),
    }


def run_all_36_evals(dev_dir: Path, out_dir: Path, recipe: CampaignRecipe, *,
                     micro_steps: int = 450, threads: int = 2) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    f14 = evaluate_families_1_to_4(dev_dir, recipe)
    micro_tasks = [
        'E25_parity_mod2',
        'E26_mod4_accumulator',
        'E27_copy_retrieve_2slot',
        'E28_dynamic_overwrite_register',
        'E29_dyck1_bounded_depth',
        'E30_majority_vote_4sym',
        'E31_2hop_pointer_chase',
        'E32_latin4_1blank',
        'E33_latin4_2blank_coupled',
        'E34_argmin4_index',
        'E35_cellular_rule90_2step',
        'E36_dfa_4state_extrap',
    ]
    f56 = []
    for tid in micro_tasks:
        res = run_micro_benchmark(tid, steps=micro_steps, threads=threads)
        f56.append(res)
        print(json.dumps({
            'eval': res['id'], 'name': res['name'],
            'GRU': round(res['models']['GRU_w32']['eval_acc'], 4),
            'TRM_c1': round(res['models']['TRM_c1_uniform']['eval_acc'], 4),
            'TRM_c2': round(res['models']['TRM_c2_uniform']['eval_acc'], 4),
            'TRM_cur': round(res['models']['TRM_c1_curated']['eval_acc'], 4),
        }), flush=True)

    full = {
        'schema_version': 1,
        'suite': 'BPD-TRM-CURATE-02-36-low-difficulty-evals',
        'total_evals': len(f14['evals']) + len(f56),
        'sealed_cohorts_touched': False,
        'elapsed_seconds': time.perf_counter() - t0,
        'families_1_to_4': f14,
        'families_5_to_6': f56,
    }
    dump_json(out_dir / 'eval_suite_36_results.json', full)
    return full


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--micro-steps', type=int, default=450)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()

    recipe = CampaignRecipe(threads=args.threads)
    with ProcessTreeMemoryCap(DEFAULT_CAP_BYTES) as cap:
        run_all_36_evals(args.dev_dir, args.out, recipe, micro_steps=args.micro_steps, threads=args.threads)
        dump_json(args.out / 'memory_accounting_eval36.json', asdict(cap.accounting()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
