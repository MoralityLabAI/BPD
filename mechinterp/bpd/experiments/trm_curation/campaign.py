"""BPD-TRM-CURATE-02: Sequential CPU research campaign runner with hard 6 GiB cap.

Implements:
- Phase A: Competence & matched architecture (RP-03 GRU vs TRM-32 vs TRM-16)
- Phase B: Mechanistic signal bank & SCCG-01 constrained selective-control frontier
- Phase C Pilot + RAM Scaling (--mms-mib 0/4/16/64/256)
- Source Freeze + 8-Root Paired Development Campaign (Arms A-G, fixed-update & fixed-CPU)
- Optional 2x2 Crossed Causal-Auxiliary Campaign (gated on selective-control admission)
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..training_steering.data import make_pairs
from ..training_steering.experiment import combine_gradients
from ..training_steering_stability.data import binding_batch
from ..training_steering_stability.model import ComponentStateMemory
from ..training_steering_stability.stats import exact_sign_flip, _holm, _bootstrap

from .cache import CacheKey, MechanisticCache
from .data import edge_batch
from .memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap
from .model import ComponentTRM
from .reference import semantic_edges
from .scheduler import Candidate, edge_views, pair_key, select_candidates
from .signal_bank import (
    build_signal_bank,
    cached_edge_batch,
    discover_slot_projectors,
    evaluate_candidate_prediction_quality,
    mechanistic_feedback_loss,
    model_state_sha256,
    ordinary_feedback_loss,
    seed_for,
    stratified_edge_sample,
)
from .transaction import probe_update

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

ENGINEERING_ROOTS = (28000, 28001)
DEVELOPMENT_ROOTS = tuple(range(28101, 28109))
ARMS = (
    'A_uniform',
    'B_loss_mined',
    'C_ord_grad',
    'D_mech_grad',
    'E_shuffled',
    'F_obs_only',
    'G_cpu_reinvest',
)
CROSSED_ARMS = ('U_ord', 'U_causal', 'M_ord', 'M_causal')


@dataclass(frozen=True)
class CampaignRecipe:
    width: int = 32
    components: int = 29
    low_cycles: int = 1
    ctrl_width: int = 16
    ctrl_components: int = 16
    slots: int = 4
    values: int = 4
    noise: float = 0.03
    threads: int = 2
    split_seed: int = 20261127
    warmup_updates: int = 6000
    warmup_batch_edges: int = 12  # 12 edges * 4 queries * 2 (r+h) = 96 seqs (comparable to RP-03 48*3=144)
    learning_rate: float = 0.003
    curation_learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    train_delays: tuple[int, ...] = (1, 2, 3)
    feedback_delay: int = 15
    eval_delays: tuple[int, ...] = (2, 7, 31, 63, 127)
    primary_delay: int = 31
    dose: float = 0.75
    dose_grid: tuple[float, ...] = (0.25, 0.75, 1.5, 3.0)
    auxiliary_gradient_ratio: float = 0.25
    # Phase C curation settings (Section 4)
    curation_windows: int = 80
    proposals_per_window: int = 16
    selected_per_window: int = 4
    exploration: float = 0.20
    pair_exposure_cap: int = 4
    feedback_batch_edges: int = 12  # 12 feedback edges * 4 queries = 48 rows
    # Mechanistic feedback objective weights (frozen before dev)
    w_ordinary: float = 1.0
    w_target: float = 0.5
    w_preserve: float = 0.5
    # Evaluation cohort sizes
    eval_n: int = 512
    binding_n: int = 256
    # Default MMS cache MiB for development
    mms_mib: float = 16.0


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    files = [
        HERE / name for name in (
            '__init__.py', 'PROTOCOL.md', 'V2_RUNBOOK.md', 'data.py', 'model.py',
            'observer.py', 'reference.py', 'scheduler.py', 'transaction.py',
            'memory_cap.py', 'cache.py', 'signal_bank.py', 'campaign.py',
            'eval_suite_36.py', 'export_datagoblin.py', 'audit.py',
        )
    ]
    files += [REPO / 'mechinterp/bpd' / name for name in ('__init__.py', 'components.py', 'config.py')]
    files += [REPO / 'mechinterp/bpd/experiments/repair_portfolio/evidence/preflight.py']
    return {
        str(p.relative_to(REPO)).replace('\\', '/'): file_sha256(p)
        for p in files if p.is_file()
    }


def _make_trm(recipe: CampaignRecipe, *, control16: bool = False) -> ComponentTRM:
    if control16:
        return ComponentTRM(
            width=recipe.ctrl_width, components=recipe.ctrl_components,
            low_cycles=recipe.low_cycles, input_dim=3 + recipe.slots + recipe.values,
            values=recipe.values,
        )
    return ComponentTRM(
        width=recipe.width, components=recipe.components,
        low_cycles=recipe.low_cycles, input_dim=3 + recipe.slots + recipe.values,
        values=recipe.values,
    )


def _make_gru(recipe: CampaignRecipe) -> ComponentStateMemory:
    return ComponentStateMemory(
        width=recipe.width, components=recipe.width,
        input_dim=3 + recipe.slots + recipe.values,
        values=recipe.values,
    )


@torch.no_grad()
def eval_model_ordinary(model: torch.nn.Module, root: int, recipe: CampaignRecipe, *,
                        split: str = 'eval', delays: tuple[int, ...] | None = None,
                        eval_n: int | None = None, binding_n: int | None = None,
                        domain_prefix: str = 'eval') -> dict[str, Any]:
    """Evaluate ordinary steering-off accuracy across delays + all-distinct address binding."""
    model.eval()
    delays = delays or recipe.eval_delays
    n_ev = eval_n or recipe.eval_n
    n_bind = binding_n or recipe.binding_n
    out: dict[str, Any] = {'split': split, 'steering_enabled': False, 'delays': {}}
    for d in delays:
        batch = make_pairs(
            seed_for(root, f'{domain_prefix}_d{d}', 0), n_ev, root_seed=root,
            delay=d, split=split, slots=recipe.slots, values=recipe.values, noise=recipe.noise,
        )
        logits = model(batch.receiver)
        preds = logits.argmax(-1)
        correct = (preds == batch.labels_r).int()
        out['delays'][str(d)] = {
            'delay': d,
            'n': n_ev,
            'correct_count': int(correct.sum()),
            'accuracy': float(correct.float().mean()),
            'cross_entropy': float(F.cross_entropy(logits, batch.labels_r)),
            'batch_sha256': batch.fingerprint(),
            'predictions': preds.tolist(),
            'labels': batch.labels_r.tolist(),
            'correct': correct.tolist(),
        }
    # All-distinct address binding at delays 2, 7, 31
    out['binding'] = {}
    for bd in (2, 7, recipe.primary_delay):
        x_b, y_b, meta = binding_batch(
            root, f'{domain_prefix}_bind_d{bd}', n_bind, bd, split, recipe
        )
        logits_b = model(x_b)
        preds_b = logits_b.argmax(-1)
        corr_b = (preds_b == y_b).int()
        out['binding'][str(bd)] = {
            'delay': bd,
            'n': n_bind,
            'correct_count': int(corr_b.sum()),
            'accuracy': float(corr_b.float().mean()),
            'cross_entropy': float(F.cross_entropy(logits_b, y_b)),
            'batch_sha256': meta['batch_sha256'],
        }
    return out


def _warmup_batch(fit_edges: tuple, root: int, step: int, recipe: CampaignRecipe):
    """Deterministic matched warmup mini-batch from TRAIN fit edges."""
    delay = recipe.train_delays[step % len(recipe.train_delays)]
    rng_seed = seed_for(root, 'warmup_edges', step)
    g = torch.Generator(device='cpu').manual_seed(rng_seed)
    idx = torch.randint(0, len(fit_edges), (recipe.warmup_batch_edges,), generator=g).tolist()
    chosen = tuple(fit_edges[i] for i in idx)
    batch = edge_batch(
        chosen, root=root, seed=seed_for(root, 'warmup_nuisance', step),
        delay=delay, split='train', noise=recipe.noise,
    )
    x = torch.cat((batch.receiver, batch.hybrid), dim=0)
    y = torch.cat((batch.labels_r, batch.labels_h), dim=0)
    return x, y, delay


def train_warmup_model(model: torch.nn.Module, arch_name: str, root: int,
                       fit_edges: tuple, recipe: CampaignRecipe, *,
                       log_every: int = 500,
                       cpu_checkpoints: tuple[float, ...] = ()) -> dict[str, Any]:
    """Train a model on the exact shared warmup stream, retaining full learning curves."""
    opt = torch.optim.AdamW(
        model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay
    )
    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()
    curve = []
    block_visits = 0
    total_examples = 0
    cpu_snaps = {}
    pending_cpu = list(sorted(cpu_checkpoints))

    for step in range(1, recipe.warmup_updates + 1):
        model.train()
        x, y, delay = _warmup_batch(fit_edges, root, step - 1, recipe)
        out = model.rollout(x)
        loss = F.cross_entropy(out['logits'], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
        opt.step()

        seq_len = x.shape[1]
        visits_per_tok = (model.low_cycles + 1) if isinstance(model, ComponentTRM) else 1
        block_visits += seq_len * visits_per_tok
        total_examples += len(y)
        elapsed_cpu = time.process_time() - t_cpu0

        while pending_cpu and elapsed_cpu >= pending_cpu[0]:
            target_c = pending_cpu.pop(0)
            ev_snap = eval_model_ordinary(
                model, root, recipe, split='train', delays=(2, 7, 31),
                eval_n=256, binding_n=128, domain_prefix=f'cpu_snap_{int(target_c)}'
            )
            cpu_snaps[str(target_c)] = {
                'step': step,
                'cpu_seconds': elapsed_cpu,
                'd2_acc': ev_snap['delays']['2']['accuracy'],
                'd7_acc': ev_snap['delays']['7']['accuracy'],
                'd31_acc': ev_snap['delays']['31']['accuracy'],
                'bind_d7_acc': ev_snap['binding']['7']['accuracy'],
            }

        if step == 1 or step % log_every == 0 or step == recipe.warmup_updates:
            ev_tr = eval_model_ordinary(
                model, root, recipe, split='train', delays=(2, 7, 31),
                eval_n=256, binding_n=128, domain_prefix='warmup_curve'
            )
            curve.append({
                'step': step,
                'train_ce': float(loss.detach()),
                'cpu_seconds': elapsed_cpu,
                'wall_seconds': time.perf_counter() - t_wall0,
                'block_visits': block_visits,
                'train_d2_acc': ev_tr['delays']['2']['accuracy'],
                'train_d7_acc': ev_tr['delays']['7']['accuracy'],
                'train_d31_acc': ev_tr['delays']['31']['accuracy'],
                'train_bind_d7_acc': ev_tr['binding']['7']['accuracy'],
            })

    return {
        'arch': arch_name,
        'root': root,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'carried_coordinates': (2 * model.width) if isinstance(model, ComponentTRM) else model.width,
        'visits_per_token': (model.low_cycles + 1) if isinstance(model, ComponentTRM) else 1,
        'updates': recipe.warmup_updates,
        'total_examples': total_examples,
        'total_block_visits': block_visits,
        'cpu_seconds': time.process_time() - t_cpu0,
        'wall_seconds': time.perf_counter() - t_wall0,
        'state_sha256': model_state_sha256(model),
        'curve': curve,
        'cpu_snaps': cpu_snaps,
    }


def run_phase_a(out_dir: Path, recipe: CampaignRecipe, roots: tuple[int, ...] = ENGINEERING_ROOTS) -> dict[str, Any]:
    """Phase A: Establish model competence & matched architecture comparison on engineering roots."""
    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    views = edge_views(recipe.split_seed)
    fit_edges = views['fit']

    phase_a: dict[str, Any] = {
        'phase': 'Phase_A_competence',
        'roots': list(roots),
        'recipe': asdict(recipe),
        'per_root': {},
    }

    for root in roots:
        r_dir = out_dir / f'root_{root}'
        r_dir.mkdir(parents=True, exist_ok=True)
        root_rec: dict[str, Any] = {}
        # Run GRU-32, TRM-32, TRM-16 on exact same warmup data stream
        # First train GRU to measure its CPU budget for fixed-CPU comparison
        arch_specs = [
            ('GRU_w32', lambda: _make_gru(recipe)),
            ('TRM_w32_r29', lambda: _make_trm(recipe, control16=False)),
            ('TRM_w16_r16', lambda: _make_trm(recipe, control16=True)),
        ]
        gru_cpu = None
        for arch_name, ctor in arch_specs:
            torch.manual_seed(seed_for(root, f'init_{arch_name}', 0))
            model = ctor()
            snaps = (gru_cpu,) if (gru_cpu is not None and arch_name != 'GRU_w32') else ()
            tr_info = train_warmup_model(
                model, arch_name, root, fit_edges, recipe,
                log_every=1000, cpu_checkpoints=snaps,
            )
            if arch_name == 'GRU_w32':
                gru_cpu = tr_info['cpu_seconds']
            # Evaluate on TRAIN (engineering selection) and EVAL (descriptive competence)
            ev_train = eval_model_ordinary(
                model, root, recipe, split='train', delays=recipe.eval_delays,
                eval_n=recipe.eval_n, binding_n=recipe.binding_n, domain_prefix='phase_a_train'
            )
            ev_eval = eval_model_ordinary(
                model, root, recipe, split='eval', delays=recipe.eval_delays,
                eval_n=recipe.eval_n, binding_n=recipe.binding_n, domain_prefix='phase_a_eval'
            )
            torch.save(model.state_dict(), r_dir / f'{arch_name}.pt')
            # Strip raw prediction vectors from summary json to keep size clean, save full in per-arch file
            summary_train = {
                'delays': {k: {fk: fv for fk, fv in v.items() if fk not in ('predictions', 'labels', 'correct')}
                           for k, v in ev_train['delays'].items()},
                'binding': ev_train['binding'],
            }
            summary_eval = {
                'delays': {k: {fk: fv for fk, fv in v.items() if fk not in ('predictions', 'labels', 'correct')}
                           for k, v in ev_eval['delays'].items()},
                'binding': ev_eval['binding'],
            }
            comp_passed = (
                summary_train['delays']['2']['accuracy'] >= 0.80
                and summary_train['delays']['7']['accuracy'] >= 0.80
                and summary_train['binding']['2']['accuracy'] >= 0.75
                and summary_train['binding']['7']['accuracy'] >= 0.75
            )
            root_rec[arch_name] = {
                **tr_info,
                'train_eval': summary_train,
                'held_eval': summary_eval,
                'competence_gate_passed': comp_passed,
            }
            print(json.dumps({
                'phase': 'A', 'root': root, 'arch': arch_name,
                'params': tr_info['parameter_count'],
                'cpu_s': round(tr_info['cpu_seconds'], 2),
                'tr_d2': round(summary_train['delays']['2']['accuracy'], 4),
                'tr_d7': round(summary_train['delays']['7']['accuracy'], 4),
                'tr_d31': round(summary_train['delays']['31']['accuracy'], 4),
                'ev_d31': round(summary_eval['delays']['31']['accuracy'], 4),
                'bind_d7': round(summary_train['binding']['7']['accuracy'], 4),
                'gate': comp_passed,
            }), flush=True)
        phase_a['per_root'][str(root)] = root_rec

    dump_json(out_dir / 'phase_a_summary.json', phase_a)
    return phase_a


def run_phase_b(phase_a_dir: Path, out_dir: Path, recipe: CampaignRecipe,
                roots: tuple[int, ...] = ENGINEERING_ROOTS) -> dict[str, Any]:
    """Phase B: Build mechanistic signal bank & SCCG-01 frontier on engineering TRM-32 warmups."""
    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    views = edge_views(recipe.split_seed)

    results = {'phase': 'Phase_B_signal_bank', 'roots': list(roots), 'per_root': {}}
    admitted_all = True
    for root in roots:
        model = _make_trm(recipe, control16=False)
        ckpt_path = phase_a_dir / f'root_{root}' / 'TRM_w32_r29.pt'
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        cache = MechanisticCache(mms_mib=recipe.mms_mib)
        bank = build_signal_bank(
            model, root, views, delays=(2, 7, recipe.primary_delay),
            dose=recipe.dose, dose_grid=recipe.dose_grid,
            n_edges=24, n_svd_edges=48, cache=cache, tag='warmup',
        )
        bank['cache_stats'] = cache.stats()
        dump_json(out_dir / f'root_{root}_signal_bank.json', bank)
        admitted_all = admitted_all and bool(bank['admission']['passed'])
        # Compact summary
        results['per_root'][str(root)] = {
            k: v for k, v in bank.items() if k != 'edges'
        }
        print(json.dumps({
            'phase': 'B', 'root': root,
            'admission': bank['admission']['status'],
            'joint_zero_acc': round(bank['support_families']['joint_zero_col']['target_accuracy'], 4),
            'joint_b025_acc': round(bank['support_families']['joint_budget_025']['target_accuracy'], 4),
            'joint_b025_cor': round(bank['support_families']['joint_budget_025']['collateral_corruption_rate'], 4),
            'hybrid_oracle_acc': round(bank['support_families']['hybrid_natural_oracle']['target_accuracy'], 4),
            'rand_r8_acc': round(bank['support_families']['random_r8']['target_accuracy'], 4),
            'cpu_s': round(bank['cpu_seconds'], 2),
        }), flush=True)

    results['overall_admission_passed'] = admitted_all
    results['overall_admission_status'] = (
        'SELECTIVE_CONTROL_ADMISSION_PASSED' if admitted_all else 'SELECTIVE_CONTROL_ADMISSION_FAILED'
    )
    dump_json(out_dir / 'phase_b_summary.json', results)
    return results


def _sample_window_proposals(fit_edges: tuple, root: int, window: int,
                             n_proposals: int, exposures: Counter, cap: int) -> tuple:
    """Sample ``n_proposals`` unique eligible edges balanced across the 4 slots."""
    eligible = [e for e in fit_edges if exposures[pair_key(e)] < cap]
    by_slot = {s: [] for s in range(4)}
    for e in eligible:
        by_slot[e[2]].append(e)
    per_slot = max(1, n_proposals // 4)
    chosen = []
    for s in range(4):
        ordered = sorted(
            by_slot[s],
            key=lambda e: hashlib.sha256(repr((root, 'prop', window, s, e)).encode()).digest(),
        )
        chosen.extend(ordered[:per_slot])
    if len(chosen) < n_proposals:
        chosen_set = set(chosen)
        rem = [e for e in eligible if e not in chosen_set]
        rem_ord = sorted(
            rem, key=lambda e: hashlib.sha256(repr((root, 'prop_rem', window, e)).encode()).digest()
        )
        chosen.extend(rem_ord[:n_proposals - len(chosen)])
    return tuple(chosen[:n_proposals])


def _get_cached_cal_grad(model: ComponentTRM, fb_batch, slot_projectors, recipe: CampaignRecipe,
                         mode: str, cache: MechanisticCache | None) -> tuple[tuple, float, float]:
    """Compute or retrieve detached calibration gradient for exact checkpoint/objective/batch."""
    params = tuple(model.parameters())
    ckpt = model_state_sha256(model)
    fframe = model.factor_frame_digest()
    if cache is not None and cache.max_bytes > 0:
        cache.bind_checkpoint(ckpt, fframe)
        ckey = CacheKey(
            namespace='cal_grad',
            checkpoint_sha256=ckpt,
            module_id='ComponentTRM.all_params',
            visit_spec=f'dose={recipe.dose}',
            input_fingerprint=fb_batch.fingerprint(),
            objective_id=f'cal_{mode}',
            factor_frame_sha256=fframe,
        )
        hit = cache.get(ckey)
        if hit is not None:
            return hit[0], hit[1], 0.0
    t0 = time.process_time()
    model.eval()
    if mode == 'ordinary':
        loss0 = ordinary_feedback_loss(model, fb_batch)
    else:
        loss0 = mechanistic_feedback_loss(
            model, fb_batch, slot_projectors, dose=recipe.dose,
            w_ordinary=recipe.w_ordinary, w_target=recipe.w_target, w_preserve=recipe.w_preserve,
        )
    g = tuple(
        None if gi is None else gi.detach().clone()
        for gi in torch.autograd.grad(loss0, params, allow_unused=True)
    )
    l0 = float(loss0.detach())
    cpu_cost = time.process_time() - t0
    if cache is not None and cache.max_bytes > 0:
        cache.put(ckey, (g, l0), compute_cpu_seconds=cpu_cost)
    return g, l0, cpu_cost


def run_paired_curation_root(
    root: int,
    warm_state: dict[str, torch.Tensor],
    out_dir: Path,
    recipe: CampaignRecipe,
    *,
    mms_mib: float | None = None,
    arms: tuple[str, ...] = ARMS,
    include_crossed: bool = False,
) -> dict[str, Any]:
    """Run all paired TRM curation arms (A-G, plus optional 2x2 crossed causal aux) on one root."""
    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    views = edge_views(recipe.split_seed)
    fit_edges = views['fit']
    fb_edges = stratified_edge_sample(
        views['feedback'], recipe.feedback_batch_edges, seed=seed_for(root, 'fb_edges', 0)
    )

    cache_mib = recipe.mms_mib if mms_mib is None else float(mms_mib)
    cache = MechanisticCache(mms_mib=cache_mib)

    # Load common warmup checkpoint
    warm = _make_trm(recipe, control16=False)
    warm.load_state_dict(copy.deepcopy(warm_state))
    warm_sha = model_state_sha256(warm)
    torch.save(warm.state_dict(), out_dir / 'shared_warmup.pt')

    # Discover fixed slot projectors from TRAIN fit edges on warmup checkpoint
    slot_projectors, proj_info = discover_slot_projectors(
        warm, root, fit_edges, delay=2, dose=recipe.dose, n_edges=32, cache=cache,
    )

    # Pre-build fixed TRAIN feedback cohort at feedback_delay (strictly TRAIN partition)
    # Using non-saturated TRAIN feedback_delay (e.g. 15) gives high-signal gradient alignment
    # for long-horizon memory retention while all training updates use train_delays (1, 2, 3).
    fb_batch_fixed = cached_edge_batch(
        fb_edges, root=root, seed=seed_for(root, f'fb_cohort_d{recipe.feedback_delay}', 0),
        delay=recipe.feedback_delay, split='train', noise=recipe.noise, cache=cache,
    )

    # Measure candidate prediction quality at warmup (ordinary vs mechanistic calibration)
    probe_model = copy.deepcopy(warm)
    probe_opt = torch.optim.AdamW(
        probe_model.parameters(), lr=recipe.curation_learning_rate, weight_decay=recipe.weight_decay
    )
    probe_props = _sample_window_proposals(fit_edges, root, 0, recipe.proposals_per_window, Counter(), recipe.pair_exposure_cap)
    probe_batches = [
        cached_edge_batch((e,), root=root, seed=seed_for(root, 'probe_q', i), delay=2, split='train', noise=recipe.noise, cache=cache)
        for i, e in enumerate(probe_props)
    ]
    pred_quality_ord = evaluate_candidate_prediction_quality(
        probe_model, probe_opt, probe_batches, fb_batch_fixed, slot_projectors,
        clip=recipe.gradient_clip, dose=recipe.dose, mode='ordinary', cache=cache,
    )
    pred_quality_mech = evaluate_candidate_prediction_quality(
        probe_model, probe_opt, probe_batches, fb_batch_fixed, slot_projectors,
        clip=recipe.gradient_clip, dose=recipe.dose, mode='mechanistic', cache=cache,
    )

    # Run arms A..F sequentially (G runs last using D's exact measured process CPU budget!)
    arm_results: dict[str, Any] = {}
    decision_records: list[dict[str, Any]] = []
    arm_d_cpu = None

    ordered_arms = [a for a in arms if a != 'G_cpu_reinvest']
    if 'G_cpu_reinvest' in arms:
        ordered_arms.append('G_cpu_reinvest')

    for arm in ordered_arms:
        model = copy.deepcopy(warm)
        opt = torch.optim.AdamW(
            model.parameters(), lr=recipe.curation_learning_rate, weight_decay=recipe.weight_decay
        )
        arm_cache = MechanisticCache(mms_mib=cache_mib)
        exposures: Counter = Counter()
        t_wall0 = time.perf_counter()
        t_cpu0 = time.process_time()

        accepted_steps = 0
        rejected_steps = 0
        fallback_selections = 0
        total_probes = 0
        window_logs = []

        if arm == 'G_cpu_reinvest':
            target_cpu = arm_d_cpu if (arm_d_cpu is not None and arm_d_cpu > 0) else 1.0
            win = 0
            while (time.process_time() - t_cpu0) < target_cpu:
                delay = recipe.train_delays[win % len(recipe.train_delays)]
                # Reset exposure cap cycle if exhausted during long reinvestment
                if sum(1 for e in fit_edges if exposures[pair_key(e)] < recipe.pair_exposure_cap) < recipe.selected_per_window:
                    exposures = Counter()
                props = _sample_window_proposals(
                    fit_edges, root, win, recipe.proposals_per_window, exposures, recipe.pair_exposure_cap
                )
                epoch_id = f'{arm}:w{win}:{model_state_sha256(model)[:12]}'
                cands = [Candidate(e, 1.0, 0.0, 1e-3, epoch_id) for e in props]
                sel, new_exp = select_candidates(
                    cands, fit_edges=fit_edges, epoch=epoch_id, mode='uniform',
                    count=recipe.selected_per_window, seed=seed_for(root, 'sched', 0),
                    window=win, exposures=exposures, cap=recipe.pair_exposure_cap,
                    exploration=recipe.exploration,
                )
                exposures = Counter(new_exp)
                sel_edges = tuple(e for e, _ in sel)
                if not sel_edges:
                    win += 1
                    continue
                mb = cached_edge_batch(
                    sel_edges, root=root, seed=seed_for(root, f'mb_reinvest_{win % 4}', 0),
                    delay=delay, split='train', noise=recipe.noise, cache=arm_cache,
                )
                model.train()
                x = torch.cat((mb.receiver, mb.hybrid), dim=0)
                y = torch.cat((mb.labels_r, mb.labels_h), dim=0)
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
                opt.step()
                accepted_steps += 1
                win += 1
            total_windows_done = win
        else:
            for win in range(recipe.curation_windows):
                delay = recipe.train_delays[win % len(recipe.train_delays)]
                if sum(1 for e in fit_edges if exposures[pair_key(e)] < recipe.pair_exposure_cap) < recipe.proposals_per_window:
                    exposures = Counter()
                props = _sample_window_proposals(
                    fit_edges, root, win, recipe.proposals_per_window, exposures, recipe.pair_exposure_cap
                )
                ckpt_sha = model_state_sha256(model)
                epoch_id = f'w{win}:{ckpt_sha[:16]}'
                # Retrieve/cache fixed TRAIN feedback cohort in arm_cache
                fb_batch = cached_edge_batch(
                    fb_edges, root=root, seed=seed_for(root, f'fb_cohort_d{recipe.feedback_delay}', 0),
                    delay=recipe.feedback_delay, split='train', noise=recipe.noise, cache=arm_cache,
                )

                # Materialize identical candidate proposal batches using bounded controlled nuisance variants (4 variants/edge)
                nuisance_variant = win % 4
                prop_batches = {
                    e: cached_edge_batch(
                        (e,), root=root,
                        seed=seed_for(root, f'cand_v{nuisance_variant}_d{delay}',
                                      int.from_bytes(hashlib.sha256(repr(e).encode()).digest()[:4], 'little')),
                        delay=delay, split='train', noise=recipe.noise, cache=arm_cache,
                    )
                    for e in props
                }
                # Deterministic equal-work cost per candidate avoids OS timer quantization noise
                # breaking cache-tier parity while full real CPU is charged & reported separately.
                det_work_cost = float((delay + 5) * (recipe.low_cycles + 1)) * 1e-4

                rows: list[Candidate] = []
                probe_meta: dict[tuple, dict] = {}
                if arm == 'A_uniform':
                    for e in props:
                        rows.append(Candidate(e, 1.0, 0.0, det_work_cost, epoch_id))
                    sched_mode = 'uniform'
                elif arm == 'B_loss_mined':
                    model.eval()
                    with torch.no_grad():
                        for e in props:
                            t_p0 = time.process_time()
                            b_e = prop_batches[e]
                            x_e = torch.cat((b_e.receiver, b_e.hybrid), dim=0)
                            y_e = torch.cat((b_e.labels_r, b_e.labels_h), dim=0)
                            l_val = float(F.cross_entropy(model(x_e), y_e))
                            c_s = max(time.process_time() - t_p0, 1e-6)
                            rows.append(Candidate(e, l_val, 0.0, det_work_cost, epoch_id))
                            probe_meta[e] = {'train_loss': l_val, 'cpu_seconds': c_s}
                    sched_mode = 'loss'
                else:
                    # Arms C, D, E, F all probe candidates with exact transaction rollback
                    cal_mode = 'ordinary' if arm == 'C_ord_grad' else 'mechanistic'

                    def cal_fn():
                        if cal_mode == 'ordinary':
                            return ordinary_feedback_loss(model, fb_batch)
                        return mechanistic_feedback_loss(
                            model, fb_batch, slot_projectors, dose=recipe.dose,
                            w_ordinary=recipe.w_ordinary, w_target=recipe.w_target,
                            w_preserve=recipe.w_preserve,
                        )

                    for e in props:
                        b_e = prop_batches[e]

                        def tr_fn(b=b_e):
                            x_e = torch.cat((b.receiver, b.hybrid), dim=0)
                            y_e = torch.cat((b.labels_r, b.labels_h), dim=0)
                            return F.cross_entropy(model(x_e), y_e)

                        # Query cached cal_grad per candidate so mms_mib=0 pays full recompute
                        # while mms_mib>0 reuses the detached checkpoint-bound gradient!
                        cg_use, cl0_use, cal_cpu_i = _get_cached_cal_grad(
                            model, fb_batch, slot_projectors, recipe, cal_mode, arm_cache
                        )
                        pr = probe_update(
                            model, opt, tr_fn, cal_fn, clip=recipe.gradient_clip,
                            cached_cal_grad=cg_use, cached_cal_loss0=cl0_use,
                        )
                        charged_cpu = max(pr['cpu_seconds'] + cal_cpu_i, 1e-6)
                        rows.append(Candidate(e, pr['train_loss'], pr['predicted_gain'], det_work_cost, epoch_id))
                        probe_meta[e] = {**pr, 'charged_cpu_seconds': charged_cpu, 'det_work_cost': det_work_cost}
                        total_probes += 1

                    if arm in ('C_ord_grad', 'D_mech_grad'):
                        sched_mode = 'mechanistic'
                    elif arm == 'E_shuffled':
                        sched_mode = 'shuffled'
                    else:  # F_obs_only
                        sched_mode = 'uniform'

                selected, new_exp = select_candidates(
                    rows, fit_edges=fit_edges, epoch=epoch_id, mode=sched_mode,
                    count=recipe.selected_per_window, seed=seed_for(root, 'sched', 0),
                    window=win, exposures=exposures, cap=recipe.pair_exposure_cap,
                    exploration=recipe.exploration,
                )
                exposures = Counter(new_exp)
                sel_edges = tuple(e for e, _ in selected)
                fallback_selections += sum(1 for _, rsn in selected if rsn == 'ordinary_fallback')

                # Build combined mini-batch of the 4 selected edges (identical ordinary labels)
                sel_batches = [prop_batches[e] for e in sel_edges]
                x_mb = torch.cat([torch.cat((b.receiver, b.hybrid), dim=0) for b in sel_batches], dim=0)
                y_mb = torch.cat([torch.cat((b.labels_r, b.labels_h), dim=0) for b in sel_batches], dim=0)

                # For C and D, check calibration acceptance with full rollback on negative gain
                snap_model = copy.deepcopy(model.state_dict()) if arm in ('C_ord_grad', 'D_mech_grad') else None
                snap_opt = copy.deepcopy(opt.state_dict()) if arm in ('C_ord_grad', 'D_mech_grad') else None
                cal_before = None
                if arm in ('C_ord_grad', 'D_mech_grad'):
                    model.eval()
                    with torch.no_grad():
                        cal_before = float(
                            ordinary_feedback_loss(model, fb_batch)
                            if arm == 'C_ord_grad'
                            else mechanistic_feedback_loss(
                                model, fb_batch, slot_projectors, dose=recipe.dose,
                                w_ordinary=recipe.w_ordinary, w_target=recipe.w_target,
                                w_preserve=recipe.w_preserve,
                            )
                        )

                model.train()
                loss_mb = F.cross_entropy(model(x_mb), y_mb)
                opt.zero_grad(set_to_none=True)
                loss_mb.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
                opt.step()

                accepted = True
                cal_after = None
                if arm in ('C_ord_grad', 'D_mech_grad'):
                    model.eval()
                    with torch.no_grad():
                        cal_after = float(
                            ordinary_feedback_loss(model, fb_batch)
                            if arm == 'C_ord_grad'
                            else mechanistic_feedback_loss(
                                model, fb_batch, slot_projectors, dose=recipe.dose,
                                w_ordinary=recipe.w_ordinary, w_target=recipe.w_target,
                                w_preserve=recipe.w_preserve,
                            )
                        )
                    # Reject if calibration loss degrades beyond tiny numerical tolerance
                    if cal_after > cal_before + 1e-6:
                        model.load_state_dict(snap_model)
                        opt.load_state_dict(snap_opt)
                        accepted = False
                        rejected_steps += 1
                    else:
                        accepted_steps += 1
                else:
                    accepted_steps += 1

                # Record DataGoblin-exportable train-only curation decisions for Arm D
                if arm == 'D_mech_grad':
                    sel_map = {e: rsn for e, rsn in selected}
                    for e in props:
                        pm = probe_meta[e]
                        decision_records.append({
                            'root': root,
                            'arm': arm,
                            'window': win,
                            'delay': delay,
                            'epoch': epoch_id,
                            'checkpoint_sha256': ckpt_sha,
                            'edge': [list(e[0]), list(e[1]), e[2]],
                            'selected': e in sel_map,
                            'selection_reason': sel_map.get(e, 'rejected_pool'),
                            'window_accepted': accepted,
                            'train_loss': pm['train_loss'],
                            'predicted_gain': pm['predicted_gain'],
                            'realized_gain': pm['realized_gain'],
                            'utility': pm['predicted_gain'] / pm['charged_cpu_seconds'],
                            'update_norm': pm['update_norm'],
                            'charged_cpu_seconds': pm['charged_cpu_seconds'],
                            'window_cal_before': cal_before,
                            'window_cal_after': cal_after,
                            'split': 'train',
                            'training_eligible': True,
                        })

                if (win + 1) % 20 == 0 or win == recipe.curation_windows - 1:
                    window_logs.append({
                        'window': win + 1,
                        'train_loss': float(loss_mb.detach()),
                        'accepted': accepted,
                        'accepted_steps': accepted_steps,
                        'rejected_steps': rejected_steps,
                        'cpu_seconds': time.process_time() - t_cpu0,
                    })
            total_windows_done = recipe.curation_windows

        arm_cpu = time.process_time() - t_cpu0
        arm_wall = time.perf_counter() - t_wall0
        if arm == 'D_mech_grad':
            arm_d_cpu = arm_cpu

        # Evaluate trained arm (steering disabled!) on evaluation partition
        ev = eval_model_ordinary(
            model, root, recipe, split='eval', delays=recipe.eval_delays,
            eval_n=recipe.eval_n, binding_n=recipe.binding_n, domain_prefix='dev_eval'
        )
        torch.save(model.state_dict(), out_dir / f'{arm}.pt')
        arm_results[arm] = {
            'arm': arm,
            'final_state_sha256': model_state_sha256(model),
            'windows': total_windows_done,
            'accepted_steps': accepted_steps,
            'rejected_steps': rejected_steps,
            'fallback_selections': fallback_selections,
            'total_probes': total_probes,
            'cpu_seconds': arm_cpu,
            'wall_seconds': arm_wall,
            'cache_stats': arm_cache.stats(),
            'window_logs': window_logs,
            'eval': ev,
        }
        print(json.dumps({
            'root': root, 'arm': arm,
            'acc_steps': accepted_steps, 'rej_steps': rejected_steps,
            'cpu_s': round(arm_cpu, 2),
            'd2': round(ev['delays']['2']['accuracy'], 4),
            'd31': round(ev['delays']['31']['accuracy'], 4),
            'd127': round(ev['delays']['127']['accuracy'], 4),
            'bind_d31': round(ev['binding']['31']['accuracy'], 4),
        }), flush=True)

    # Optional 2x2 crossed causal-auxiliary experiment (Section 5)
    crossed_results = {}
    if include_crossed:
        for carm in CROSSED_ARMS:
            use_mech = carm.startswith('M_')
            use_aux = carm.endswith('_causal')
            model = copy.deepcopy(warm)
            opt = torch.optim.AdamW(
                model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay
            )
            arm_cache = MechanisticCache(mms_mib=cache_mib)
            exposures = Counter()
            t_cpu0 = time.process_time()
            params = tuple(model.parameters())
            for win in range(recipe.curation_windows):
                delay = recipe.train_delays[win % len(recipe.train_delays)]
                if sum(1 for e in fit_edges if exposures[pair_key(e)] < recipe.pair_exposure_cap) < recipe.proposals_per_window:
                    exposures = Counter()
                props = _sample_window_proposals(
                    fit_edges, root, win, recipe.proposals_per_window, exposures, recipe.pair_exposure_cap
                )
                ckpt_sha = model_state_sha256(model)
                epoch_id = f'{carm}:w{win}:{ckpt_sha[:12]}'
                fb_batch = fb_batch_fixed
                det_work_cost = float((delay + 5) * (recipe.low_cycles + 1)) * 1e-4
                prop_batches = {
                    e: cached_edge_batch(
                        (e,), root=root, seed=seed_for(root, f'xcand_w{win}', idx),
                        delay=delay, split='train', noise=recipe.noise, cache=arm_cache,
                    )
                    for idx, e in enumerate(props)
                }
                if not use_mech:
                    rows = [Candidate(e, 1.0, 0.0, det_work_cost, epoch_id) for e in props]
                    sched_mode = 'uniform'
                else:
                    cal_grad, cal_l0, cal_build_cpu = _get_cached_cal_grad(
                        model, fb_batch, slot_projectors, recipe, 'mechanistic', arm_cache
                    )
                    rows = []
                    for e in props:
                        b_e = prop_batches[e]

                        def tr_fn(b=b_e):
                            return F.cross_entropy(
                                model(torch.cat((b.receiver, b.hybrid), dim=0)),
                                torch.cat((b.labels_r, b.labels_h), dim=0),
                            )

                        def cal_fn():
                            return mechanistic_feedback_loss(
                                model, fb_batch, slot_projectors, dose=recipe.dose,
                                w_ordinary=recipe.w_ordinary, w_target=recipe.w_target,
                                w_preserve=recipe.w_preserve,
                            )

                        pr = probe_update(
                            model, opt, tr_fn, cal_fn, clip=recipe.gradient_clip,
                            cached_cal_grad=cal_grad, cached_cal_loss0=cal_l0,
                        )
                        rows.append(Candidate(e, pr['train_loss'], pr['predicted_gain'], det_work_cost, epoch_id))
                    sched_mode = 'mechanistic'

                selected, new_exp = select_candidates(
                    rows, fit_edges=fit_edges, epoch=epoch_id, mode=sched_mode,
                    count=recipe.selected_per_window, seed=seed_for(root, 'xsched', 0),
                    window=win, exposures=exposures, cap=recipe.pair_exposure_cap,
                    exploration=recipe.exploration,
                )
                exposures = Counter(new_exp)
                sel_batches = [prop_batches[e] for e, _ in selected]
                # Combine selected batches
                x_r = torch.cat([b.receiver for b in sel_batches], dim=0)
                x_h = torch.cat([b.hybrid for b in sel_batches], dim=0)
                y_r = torch.cat([b.labels_r for b in sel_batches], dim=0)
                y_h = torch.cat([b.labels_h for b in sel_batches], dim=0)
                edit_slot = torch.cat([b.edit_slot for b in sel_batches], dim=0)
                query = torch.cat([b.query for b in sel_batches], dim=0)

                model.train()
                out_cat = model.rollout(torch.cat((x_r, x_h), dim=0))
                loss_task = F.cross_entropy(out_cat['logits'], torch.cat((y_r, y_h), dim=0))
                task_g = torch.autograd.grad(loss_task, params, retain_graph=use_aux)

                if use_aux:
                    n_s = len(y_r)
                    sr = out_cat['states'][:n_s, 3]
                    sh = out_cat['states'][n_s:, 3].detach()
                    proj = slot_projectors.detach()[edit_slot]
                    raw = torch.bmm((sh - sr)[:, None, :], proj).squeeze(1)
                    rn = torch.linalg.vector_norm(raw, dim=-1)
                    sc = torch.where(rn > 1e-9, recipe.dose / rn.clamp_min(1e-9), torch.zeros_like(rn))
                    delta = raw * sc[:, None]
                    ed_logits = model.rollout_suffix(x_r[:, 4:], sr + delta, prefix_steps=4)['logits']
                    targ = query == edit_slot
                    unaff = ~targ
                    loss_aux = (
                        0.5 * F.cross_entropy(ed_logits[targ], y_h[targ])
                        + 0.5 * F.cross_entropy(ed_logits[unaff], y_r[unaff])
                    )
                    aux_g = torch.autograd.grad(loss_aux, params)
                    comb_g, _ = combine_gradients(task_g, aux_g, recipe.auxiliary_gradient_ratio, True)
                else:
                    comb_g = task_g

                opt.zero_grad(set_to_none=True)
                for p, g in zip(params, comb_g):
                    p.grad = g.detach().clone()
                torch.nn.utils.clip_grad_norm_(params, recipe.gradient_clip)
                opt.step()

            ev_c = eval_model_ordinary(
                model, root, recipe, split='eval', delays=recipe.eval_delays,
                eval_n=recipe.eval_n, binding_n=recipe.binding_n, domain_prefix='dev_eval'
            )
            torch.save(model.state_dict(), out_dir / f'crossed_{carm}.pt')
            crossed_results[carm] = {
                'arm': carm,
                'final_state_sha256': model_state_sha256(model),
                'cpu_seconds': time.process_time() - t_cpu0,
                'eval': ev_c,
            }

    # Save train-only curation decisions JSONL for DataGoblin export
    dec_path = out_dir / 'train_curation_decisions.jsonl'
    with dec_path.open('w', encoding='utf-8', newline='\n') as f:
        for rec in decision_records:
            f.write(json.dumps(rec, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')

    result = {
        'root': root,
        'evidence_class': 'DEVELOPMENT_ONLY',
        'shared_warmup_sha256': warm_sha,
        'slot_projector_info': proj_info,
        'candidate_prediction_quality': {
            'ordinary_calibration': {k: v for k, v in pred_quality_ord.items() if k != 'rows'},
            'mechanistic_calibration': {k: v for k, v in pred_quality_mech.items() if k != 'rows'},
        },
        'arms': arm_results,
        'crossed_arms': crossed_results,
        'train_decisions_count': len(decision_records),
        'train_decisions_sha256': file_sha256(dec_path),
    }
    dump_json(out_dir / 'result.json', result)
    dump_json(
        out_dir / 'artifact_hashes.json',
        {p.name: file_sha256(p) for p in sorted(out_dir.iterdir())
         if p.is_file() and p.name != 'artifact_hashes.json'},
    )
    return result


def run_ram_scaling_benchmark(phase_a_dir: Path, out_dir: Path, recipe: CampaignRecipe,
                              root: int = 28000,
                              tiers_mib: tuple[float, ...] = (0.0, 4.0, 16.0, 64.0, 256.0),
                              windows: int = 16) -> dict[str, Any]:
    """Benchmark --mms-mib across 0/4/16/64/256 MiB with exact parity verification."""
    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    warm_state = torch.load(phase_a_dir / f'root_{root}' / 'TRM_w32_r29.pt', map_location='cpu', weights_only=True)
    sub_recipe = CampaignRecipe(**{**asdict(recipe), 'curation_windows': windows, 'eval_n': 256, 'binding_n': 128})

    tier_rows = []
    ref_sha = None
    ref_d31 = None
    for mib in tiers_mib:
        t_dir = out_dir / f'mms_{int(mib)}mib'
        res = run_paired_curation_root(
            root, warm_state, t_dir, sub_recipe,
            mms_mib=mib, arms=('A_uniform', 'D_mech_grad'), include_crossed=False,
        )
        d_arm = res['arms']['D_mech_grad']
        a_arm = res['arms']['A_uniform']
        sha = d_arm['final_state_sha256']
        d31_d = d_arm['eval']['delays']['31']['accuracy']
        d31_a = a_arm['eval']['delays']['31']['accuracy']
        if ref_sha is None:
            ref_sha = sha
            ref_d31 = d31_d
        exact_parity = (sha == ref_sha)
        cs = d_arm['cache_stats']
        cpu_hours = max(d_arm['cpu_seconds'] / 3600.0, 1e-9)
        tier_rows.append({
            'mms_mib': mib,
            'exact_checkpoint_parity': exact_parity,
            'final_state_sha256': sha,
            'd31_accuracy_D': d31_d,
            'd31_accuracy_A': d31_a,
            'delta_d31_pp': (d31_d - d31_a) * 100.0,
            'total_cpu_seconds': d_arm['cpu_seconds'],
            'total_wall_seconds': d_arm['wall_seconds'],
            'improvement_pp_per_cpu_hour': ((d31_d - d31_a) * 100.0) / cpu_hours,
            'cache': cs,
        })
        print(json.dumps({
            'ram_tier_mib': mib, 'parity': exact_parity,
            'cpu_s': round(d_arm['cpu_seconds'], 3),
            'hit_rate': round(cs['hit_rate'], 4),
            'peak_kb': round(cs['peak_total_bytes'] / 1024.0, 2),
            'recomp_avoided': cs['recomputations_avoided'],
            'saved_cpu_est': round(cs['saved_cpu_seconds_est'], 3),
        }), flush=True)

    # Check whether measured evidence justifies expanding beyond 256 MiB
    peak_256_mb = tier_rows[-1]['cache']['peak_total_bytes'] / (1024 * 1024)
    expand_justified = peak_256_mb > 200.0 and tier_rows[-1]['total_cpu_seconds'] < tier_rows[-2]['total_cpu_seconds']
    summary = {
        'root': root,
        'windows': windows,
        'tiers': tier_rows,
        'all_tiers_exact_parity': all(r['exact_checkpoint_parity'] for r in tier_rows),
        'peak_occupancy_at_256mib_mb': peak_256_mb,
        'expand_beyond_256mib_justified': expand_justified,
    }
    dump_json(out_dir / 'ram_scaling_summary.json', summary)
    return summary


def summarize_paired_campaign(results: list[dict], out_dir: Path, *,
                              primary_delay: int = 31) -> dict[str, Any]:
    """Compute exact root sign-flip, Holm adjustment, bootstrap & EvalMath preflight."""
    d_key = str(primary_delay)
    arms_present = list(results[0]['arms'].keys())
    per_arm_acc = {}
    for arm in arms_present:
        per_root = []
        for r in results:
            rec = r['arms'][arm]['eval']['delays'][d_key]
            per_root.append({
                'root': r['root'],
                'correct': rec['correct_count'],
                'n': rec['n'],
                'accuracy': rec['accuracy'],
                'cpu_seconds': r['arms'][arm]['cpu_seconds'],
            })
        tot_c = sum(x['correct'] for x in per_root)
        tot_n = sum(x['n'] for x in per_root)
        per_arm_acc[arm] = {
            'macro_accuracy': float(np.mean([x['accuracy'] for x in per_root])),
            'pooled_accuracy': tot_c / tot_n,
            'correct': tot_c,
            'n': tot_n,
            'mean_cpu_seconds': float(np.mean([x['cpu_seconds'] for x in per_root])),
            'per_root': per_root,
        }

    # Registered contrast family for D_mech_grad (fixed-update & fixed-CPU)
    contrast_pairs = [
        ('D_mech_grad', 'A_uniform', 'primary_fixed_update'),
        ('D_mech_grad', 'B_loss_mined', 'vs_loss_mining'),
        ('D_mech_grad', 'E_shuffled', 'vs_shuffled_sham'),
        ('D_mech_grad', 'G_cpu_reinvest', 'primary_fixed_cpu'),
        ('C_ord_grad', 'A_uniform', 'ordinary_grad_vs_uniform'),
    ]
    contrasts = []
    p_fracs = []
    for cand, base, role in contrast_pairs:
        if cand not in arms_present or base not in arms_present:
            continue
        effects = []
        rows = []
        tot_rec, tot_cor, tot_n = 0, 0, 0
        for r in results:
            a_bits = r['arms'][base]['eval']['delays'][d_key]['correct']
            b_bits = r['arms'][cand]['eval']['delays'][d_key]['correct']
            rec_i = sum(1 for x, y in zip(a_bits, b_bits) if x == 0 and y == 1)
            cor_i = sum(1 for x, y in zip(a_bits, b_bits) if x == 1 and y == 0)
            eff = Fraction(rec_i - cor_i, len(a_bits))
            effects.append(eff)
            tot_rec += rec_i
            tot_cor += cor_i
            tot_n += len(a_bits)
            rows.append({
                'root': r['root'], 'recovery': rec_i, 'corruption': cor_i,
                'delta': float(eff), 'delta_fraction': str(eff),
            })
        sf = exact_sign_flip(effects)
        p_fracs.append(Fraction(sf['p_fraction']))
        macro_eff = sum(effects, Fraction(0)) / len(effects)
        contrasts.append({
            'id': f'{cand}_minus_{base}_d{primary_delay}',
            'role': role,
            'candidate': cand,
            'baseline': base,
            'macro_effect': float(macro_eff),
            'macro_effect_pp': float(macro_eff) * 100.0,
            'macro_effect_fraction': str(macro_eff),
            'recovery': tot_rec,
            'corruption': tot_cor,
            'net_recovery': tot_rec - tot_cor,
            'per_root': rows,
            'sign_flip': sf,
            'bootstrap_95': _bootstrap(effects, 10000, 20261126),
        })

    for c_obj, adj_p in zip(contrasts, _holm(p_fracs)):
        c_obj['holm_adjusted_p'] = float(adj_p)
        c_obj['holm_adjusted_p_fraction'] = str(adj_p)
        c_obj['exceeds_5pp_floor'] = c_obj['macro_effect'] >= 0.05

    # Run pinned EvalMath preflight on D_mech_grad - A_uniform with E_shuffled - A_uniform as sham
    preflight_verdict = None
    if 'D_mech_grad' in arms_present and 'A_uniform' in arms_present and 'E_shuffled' in arms_present:
        pilot_rows = []
        n_ev = results[0]['arms']['A_uniform']['eval']['delays'][d_key]['n']
        for r in results:
            ca = r['arms']['A_uniform']['eval']['delays'][d_key]['correct_count']
            cd = r['arms']['D_mech_grad']['eval']['delays'][d_key]['correct_count']
            ce = r['arms']['E_shuffled']['eval']['delays'][d_key]['correct_count']
            pilot_rows.append({
                'unit_id': str(r['root']),
                'paired_effect': f'{cd - ca}/{n_ev}',
                'sham_effects': [f'{ce - ca}/{n_ev}'],
            })
        spec = {
            'schema_version': 1,
            'experiment_id': 'BPD-TRM-CURATE-02-development',
            'independent_unit': 'independently initialized TRM model root with paired curation descendants',
            'alpha': '0.05',
            'scientific_min_effect': '0.05',
            'target_effect': '0.10',
            'measurement_resolution': f'1/{n_ev}',
            'sham_quantile': '0.95',
            'sham_confidence': '0.95',
            'planned_comparisons': len(contrasts),
            'confirm_units_planned': len(DEVELOPMENT_ROOTS),
            'pilot': pilot_rows,
            'pilot_from_development_only': True,
            'heldout_sealed': False,
            'units_independent': True,
        }
        pf_path = REPO / 'mechinterp/bpd/experiments/repair_portfolio/evidence/preflight.py'
        loader = importlib.util.spec_from_file_location('trm_curate_preflight', pf_path)
        mod = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(mod)
        preflight_verdict = mod.preflight(spec)
        dump_json(out_dir / 'preflight_input.json', spec)
        dump_json(out_dir / 'preflight_result.json', preflight_verdict)

    summary = {
        'schema_version': 1,
        'experiment': 'BPD-TRM-CURATE-02',
        'evidence_class': 'DEVELOPMENT_ONLY',
        'primary_delay': primary_delay,
        'n_roots': len(results),
        'roots': [r['root'] for r in results],
        'arms': per_arm_acc,
        'contrasts': contrasts,
        'evalmath_preflight': preflight_verdict,
    }
    dump_json(out_dir / 'summary.json', summary)
    return summary


def write_source_freeze(path: Path, recipe: CampaignRecipe, *,
                        parent_commit: str = '154a3a1bde93a84015c687dd9464077df6631015',
                        selective_control_admitted: bool = False) -> dict[str, Any]:
    """Freeze source hashes, recipe, split seed, feedback objective, and dev roster."""
    views = edge_views(recipe.split_seed)
    split_hashes = {
        k: hashlib.sha256(repr(v).encode('utf-8')).hexdigest()
        for k, v in sorted(views.items())
    }
    lock = {
        'schema_version': 1,
        'experiment': 'BPD-TRM-CURATE-02',
        'parent_commit': parent_commit,
        'frozen_unix': time.time(),
        'recipe': asdict(recipe),
        'split_counts': {k: len(v) for k, v in views.items()},
        'split_hashes': split_hashes,
        'engineering_roots': list(ENGINEERING_ROOTS),
        'development_roots': list(DEVELOPMENT_ROOTS),
        'selective_control_admitted': selective_control_admitted,
        'feedback_objective': {
            'name': 'mechanistic_feedback_loss_v1',
            'w_ordinary': recipe.w_ordinary,
            'w_target': recipe.w_target,
            'w_preserve': recipe.w_preserve,
            'dose': recipe.dose,
            'preserve_terms': '0.5 * KL(p_clean || p_edit) + 0.5 * CE(edit, labels_r) on unaffected slots',
        },
        'source_hashes': source_hashes(),
    }
    dump_json(path, lock)
    return lock


def run_development_campaign(out_dir: Path, lock_path: Path, recipe: CampaignRecipe, *,
                             roots: tuple[int, ...] = DEVELOPMENT_ROOTS,
                             include_crossed: bool | None = None) -> dict[str, Any]:
    """Execute 8 fresh paired development roots under source lock and 6 GiB cap."""
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    if lock.get('source_hashes') != source_hashes():
        raise ValueError('Source freeze mismatch! Re-freeze before running development.')
    if lock.get('recipe') != json.loads(json.dumps(asdict(recipe))):
        raise ValueError('Recipe differs from SOURCE_FREEZE.json')
    use_crossed = bool(lock.get('selective_control_admitted', False)) if include_crossed is None else include_crossed

    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    views = edge_views(recipe.split_seed)
    fit_edges = views['fit']

    dump_json(out_dir / 'run_manifest.json', {
        'phase': 'development',
        'roots': list(roots),
        'evidence_class': 'DEVELOPMENT_ONLY',
        'recipe': asdict(recipe),
        'source_lock_sha256': file_sha256(lock_path),
        'include_crossed': use_crossed,
        'python': platform.python_version(),
        'torch': torch.__version__,
        'started_unix': time.time(),
    })

    all_results = []
    t0 = time.perf_counter()
    for root in roots:
        r_dir = out_dir / f'root_{root}'
        # Resumable: if result.json already exists, load it cleanly
        if (r_dir / 'result.json').is_file():
            print(json.dumps({'dev_root': root, 'status': 'loaded_existing'}), flush=True)
            all_results.append(json.loads((r_dir / 'result.json').read_text(encoding='utf-8')))
            continue
        # Step 1: Common 6,000-update TRM-32 warmup
        torch.manual_seed(seed_for(root, 'init_TRM_w32_r29', 0))
        warm = _make_trm(recipe, control16=False)
        w_info = train_warmup_model(
            warm, 'TRM_w32_r29', root, fit_edges, recipe, log_every=2000,
        )
        warm_state = copy.deepcopy(warm.state_dict())
        # Step 2: Run paired curation arms A-G (+ optional crossed)
        res = run_paired_curation_root(
            root, warm_state, r_dir, recipe,
            mms_mib=recipe.mms_mib, arms=ARMS, include_crossed=use_crossed,
        )
        res['warmup_info'] = w_info
        dump_json(r_dir / 'result.json', res)
        dump_json(
            r_dir / 'artifact_hashes.json',
            {p.name: file_sha256(p) for p in sorted(r_dir.iterdir())
             if p.is_file() and p.name != 'artifact_hashes.json'},
        )
        all_results.append(res)

    summary = summarize_paired_campaign(all_results, out_dir, primary_delay=recipe.primary_delay)
    dump_json(out_dir / 'completion.json', {
        'completed': True,
        'roots': list(roots),
        'elapsed_seconds': time.perf_counter() - t0,
        'result_sha256': {
            str(r): file_sha256(out_dir / f'root_{r}' / 'result.json') for r in roots
        },
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--stage',
        choices=('phase_a', 'phase_b', 'ram_scaling', 'eng_smoke', 'all_engineering', 'freeze', 'development'),
        required=True,
    )
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mms-mib', type=float, default=16.0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--warmup-updates', type=int, default=6000)
    parser.add_argument('--curation-windows', type=int, default=80)
    parser.add_argument('--source-lock', type=Path)
    parser.add_argument('--include-crossed', action='store_true')
    parser.add_argument('--no-cap', action='store_true', help='Disable hard cap only for unit testing')
    args = parser.parse_args()

    recipe = CampaignRecipe(
        mms_mib=args.mms_mib,
        threads=args.threads,
        warmup_updates=args.warmup_updates,
        curation_windows=args.curation_windows,
    )

    cap_ctx = ProcessTreeMemoryCap(DEFAULT_CAP_BYTES) if not args.no_cap else None
    if cap_ctx is not None:
        cap_ctx.__enter__()
    try:
        if args.stage == 'phase_a':
            run_phase_a(args.out / 'phase_a', recipe)
        elif args.stage == 'phase_b':
            run_phase_b(args.out / 'phase_a', args.out / 'phase_b', recipe)
        elif args.stage == 'ram_scaling':
            run_ram_scaling_benchmark(args.out / 'phase_a', args.out / 'ram_scaling', recipe)
        elif args.stage == 'eng_smoke':
            smoke_results = []
            for r in ENGINEERING_ROOTS:
                ws = torch.load(args.out / 'phase_a' / f'root_{r}' / 'TRM_w32_r29.pt', map_location='cpu', weights_only=True)
                smoke_results.append(
                    run_paired_curation_root(r, ws, args.out / 'eng_smoke' / f'root_{r}', recipe,
                                             include_crossed=args.include_crossed)
                )
            summarize_paired_campaign(smoke_results, args.out / 'eng_smoke', primary_delay=recipe.primary_delay)
        elif args.stage == 'all_engineering':
            if not (args.out / 'phase_a' / 'phase_a_summary.json').is_file():
                run_phase_a(args.out / 'phase_a', recipe)
            else:
                print(json.dumps({'phase': 'A', 'status': 'reusing_existing'}), flush=True)
            if not (args.out / 'phase_b' / 'phase_b_summary.json').is_file():
                pb = run_phase_b(args.out / 'phase_a', args.out / 'phase_b', recipe)
            else:
                print(json.dumps({'phase': 'B', 'status': 'reusing_existing'}), flush=True)
                pb = json.loads((args.out / 'phase_b' / 'phase_b_summary.json').read_text(encoding='utf-8'))
            run_ram_scaling_benchmark(args.out / 'phase_a', args.out / 'ram_scaling', recipe)
            smoke_results = []
            admitted = bool(pb['overall_admission_passed'])
            for r in ENGINEERING_ROOTS:
                ws = torch.load(args.out / 'phase_a' / f'root_{r}' / 'TRM_w32_r29.pt', map_location='cpu', weights_only=True)
                smoke_results.append(
                    run_paired_curation_root(
                        r, ws, args.out / 'eng_smoke' / f'root_{r}', recipe,
                        include_crossed=(admitted or args.include_crossed),
                    )
                )
            summarize_paired_campaign(smoke_results, args.out / 'eng_smoke', primary_delay=recipe.primary_delay)
            write_source_freeze(
                HERE / 'SOURCE_FREEZE.json', recipe,
                selective_control_admitted=admitted,
            )
        elif args.stage == 'freeze':
            admitted = False
            pb_path = args.out / 'phase_b' / 'phase_b_summary.json'
            if pb_path.is_file():
                admitted = bool(json.loads(pb_path.read_text(encoding='utf-8')).get('overall_admission_passed', False))
            write_source_freeze(HERE / 'SOURCE_FREEZE.json', recipe, selective_control_admitted=admitted)
        elif args.stage == 'development':
            lock_p = args.source_lock or (HERE / 'SOURCE_FREEZE.json')
            run_development_campaign(
                args.out / 'development', lock_p, recipe,
                include_crossed=(True if args.include_crossed else None),
            )
    finally:
        if cap_ctx is not None:
            acc = cap_ctx.accounting()
            dump_json(args.out / f'memory_accounting_{args.stage}.json', asdict(acc))
            print(json.dumps({
                'memory_cap': acc.enforcement_mechanism,
                'cap_gib': round(acc.cap_bytes / (1024**3), 2),
                'peak_job_mib': round((acc.peak_job_commit_bytes or 0) / (1024**2), 2),
                'peak_tree_rss_mib': round(acc.peak_tree_rss_bytes / (1024**2), 2),
                'host_avail_min_gib': round(acc.host_available_bytes_min / (1024**3), 2),
                'wall_s': round(acc.wall_seconds, 2),
                'cpu_s': round(acc.process_cpu_seconds, 2),
            }), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
