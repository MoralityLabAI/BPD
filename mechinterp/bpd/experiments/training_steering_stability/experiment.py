"""BPD-RP-03: bounded paired training after a hidden-state intervention redesign."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import resource
import time

import torch
import torch.nn.functional as F

from ..training_steering.experiment import (
    batch_hash, combine_gradients, correctness, full_task_batch, state_hash,
    source_hashes as rp02_source_hashes,
)
from .data import binding_batch, make_batch, seed_for
from .model import ComponentStateMemory, projectors, state_patch
from .stats import summarize

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ARMS = ("A_task", "B_causal_fixed", "C_rotated_fixed", "D_causal_live")
SMOKE_ROOTS = (27010, 27011)
DEVELOPMENT_ROOTS = tuple(range(27101, 27109))


@dataclass(frozen=True)
class Recipe:
    width: int = 32
    components: int = 32
    slots: int = 4
    values: int = 4
    warmup_steps: int = 6000
    training_steps: int = 240
    batch_size: int = 48
    learning_rate: float = .003
    weight_decay: float = .0001
    auxiliary_gradient_ratio: float = .25
    gradient_clip: float = 1.
    intervention_l2: float = .75
    calibration_n: int = 512
    group_size: int = 4
    anchor_n: int = 128
    anchor_steps: tuple[int, ...] = (0, 80, 160, 240)
    train_delays: tuple[int, ...] = (1, 2, 3)
    iid_delay: int = 2
    long_delay: int = 31
    evaluation_n: int = 512
    causal_n: int = 256
    binding_n: int = 256
    noise: float = .03
    threads: int = 1


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def new_model(recipe: Recipe) -> ComponentStateMemory:
    return ComponentStateMemory(width=recipe.width, components=recipe.components,
                                input_dim=3 + recipe.slots + recipe.values,
                                values=recipe.values)


def rotation_for(root: int, width: int):
    generator = torch.Generator().manual_seed(seed_for(root, 'physical_sham'))
    return (torch.randperm(width, generator=generator),
            2 * torch.randint(0, 2, (width,), generator=generator).float() - 1)


def rotate(delta, rotation):
    return delta[:, rotation[0]] * rotation[1]


@torch.no_grad()
def discover_groups(model, root: int, recipe: Recipe):
    batch = make_batch(root, 'attribution_train', 0, recipe.calibration_n,
                       recipe.iid_delay, 'train', recipe)
    n = len(batch.query)
    clean = model.rollout(torch.cat((batch.receiver, batch.donor)))
    receiver = clean['states'][:n, batch.prefix_steps - 1]
    donor = clean['states'][n:, batch.prefix_steps - 1]
    candidates = torch.arange(recipe.components).reshape(-1, recipe.group_size)
    all_projectors = projectors(model, candidates)
    losses, targeted_accuracy, invalid = [], [], []
    for p in all_projectors:
        delta, info = state_patch(receiver, donor, p.expand(recipe.slots, -1, -1),
                                   batch.edit_slot, recipe.intervention_l2)
        logits = model.rollout(batch.receiver, patch=delta,
                               patch_step=batch.prefix_steps)['logits']
        ce = F.cross_entropy(logits, batch.labels_h, reduction='none')
        correct = logits.argmax(-1).eq(batch.labels_h)
        losses.append([float(ce[batch.edit_slot == slot].mean()) for slot in range(recipe.slots)])
        targeted_accuracy.append([float(correct[(batch.edit_slot == slot) &
                                                (batch.query == slot)].float().mean())
                                  for slot in range(recipe.slots)])
        invalid.append(int((~info['valid']).sum()))
    chosen = torch.tensor(losses).argmin(dim=0)
    groups = candidates[chosen].clone()
    fixed = all_projectors[chosen].clone()
    return groups, fixed, {
        'batch_sha256': batch_hash(batch), 'candidate_groups': candidates.tolist(),
        'candidate_ce_by_slot': losses, 'candidate_targeted_accuracy_by_slot': targeted_accuracy,
        'selected_candidate': chosen.tolist(), 'groups': groups.tolist(),
        'invalid_by_candidate': invalid,
        'selection_scope': 'training-only mean hybrid CE per edited slot; eight candidates; overlap allowed',
        'clean_calibration_accuracy': float(clean['logits'][:n].argmax(-1).eq(batch.labels_r).float().mean()),
    }


def training_step(model, optimizer, batch, groups, fixed, rotation, recipe: Recipe, arm: str):
    if arm not in ARMS:
        raise ValueError('Unknown treatment arm')
    model.train()
    n = len(batch.query)
    features, labels = full_task_batch(batch)
    ordinary = model.rollout(features)
    task_loss = F.cross_entropy(ordinary['logits'], labels)
    receiver = ordinary['states'][:n, batch.prefix_steps - 1]
    donor = ordinary['states'][n:2*n, batch.prefix_steps - 1]
    live = projectors(model, groups)
    fixed_delta, fixed_info = state_patch(receiver, donor, fixed, batch.edit_slot,
                                         recipe.intervention_l2)
    live_delta, live_info = state_patch(receiver, donor, live, batch.edit_slot,
                                       recipe.intervention_l2)
    sham_delta = rotate(fixed_delta, rotation)
    deltas = (fixed_delta, sham_delta, live_delta)
    losses = [F.cross_entropy(model.rollout(batch.receiver, patch=delta,
                   patch_step=batch.prefix_steps)['logits'], batch.labels_h) for delta in deltas]
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    task_gradient = torch.autograd.grad(task_loss, parameters, retain_graph=True)
    auxiliary = [torch.autograd.grad(loss, parameters, retain_graph=(i < 2))
                 for i, loss in enumerate(losses)]
    selected = {'A_task': 0, 'B_causal_fixed': 0, 'C_rotated_fixed': 1, 'D_causal_live': 2}[arm]
    gradient, accounting = combine_gradients(task_gradient, auxiliary[selected],
                                             recipe.auxiliary_gradient_ratio, arm != 'A_task')
    before = [p.detach().clone() for p in parameters]
    optimizer.zero_grad(set_to_none=True)
    for parameter, grad in zip(parameters, gradient, strict=True):
        if not torch.isfinite(grad).all():
            raise ValueError('Nonfinite training gradient')
        parameter.grad = grad.detach().clone()
    preclip = float(torch.nn.utils.clip_grad_norm_(parameters, recipe.gradient_clip))
    optimizer.step()
    update = math.sqrt(sum(float((p.detach().double() - old.double()).square().sum())
                            for p, old in zip(parameters, before, strict=True)))
    return {
        'task_ce': float(task_loss.detach()), 'fixed_ce': float(losses[0].detach()),
        'rotated_ce': float(losses[1].detach()), 'live_ce': float(losses[2].detach()),
        **accounting, 'preclip_combined_gradient_l2': preclip, 'parameter_update_l2': update,
        'fixed_physical_l2_min': float(fixed_info['delivered_norm'].detach().min()),
        'fixed_physical_l2_max': float(fixed_info['delivered_norm'].detach().max()),
        'live_physical_l2_min': float(live_info['delivered_norm'].detach().min()),
        'live_physical_l2_max': float(live_info['delivered_norm'].detach().max()),
        'sham_norm_max_error': float((fixed_delta.detach().norm(dim=-1) - sham_delta.detach().norm(dim=-1)).abs().max()),
        'fixed_invalid': int((~fixed_info['valid']).sum()), 'live_invalid': int((~live_info['valid']).sum()),
        'projectors_detached': not fixed.requires_grad and not live.requires_grad,
        'trajectories': 6*n, 'gradient_evaluations': 4,
        'ordinary_examples': 3*n, 'auxiliary_examples': 3*n,
    }


@torch.no_grad()
def causal_readout(model, batch, groups, fixed, rotation, recipe):
    n, step = len(batch.query), batch.prefix_steps
    ordinary = model.rollout(torch.cat((batch.receiver, batch.donor, batch.hybrid)))
    receiver, donor, hybrid = (ordinary['states'][i*n:(i+1)*n, step-1] for i in range(3))
    live = projectors(model, groups)
    fixed_delta, fi = state_patch(receiver, donor, fixed, batch.edit_slot, recipe.intervention_l2)
    live_delta, li = state_patch(receiver, donor, live, batch.edit_slot, recipe.intervention_l2)
    deltas = {'causal': fixed_delta, 'causal_sham': rotate(fixed_delta, rotation), 'causal_live': live_delta}
    target = batch.query.eq(batch.edit_slot)
    clean_correct = ordinary['logits'][:n].argmax(-1).eq(batch.labels_r)
    records, physical = {}, {}
    for name, delta in deltas.items():
        run = model.rollout(batch.receiver, patch=delta, patch_step=step)
        predicted_original = run['logits'].argmax(-1).eq(batch.labels_r)
        records[name] = {**correctness(run['logits'], batch.labels_h),
                         'targeted': target.tolist(), 'batch_sha256': batch_hash(batch)}
        physical[name] = {
            'delivered_l2_min': float(delta.norm(dim=-1).min()),
            'delivered_l2_max': float(delta.norm(dim=-1).max()),
            'incoming_state_delta_max_error': float((run['incoming_states'][:, step] - receiver - delta).abs().max()),
            'prefix_state_max_error': float((run['states'][:, :step] - ordinary['states'][:n, :step]).abs().max()),
            'recipient_state_rms': float(receiver.square().mean().sqrt()),
            'edited_incoming_state_rms': float(run['incoming_states'][:, step].square().mean().sqrt()),
            'terminal_state_distance_mean': float((run['states'][:, -1] - ordinary['states'][:n, -1]).norm(dim=-1).mean()),
            'all_head_max_change': float((model.output(run['states']) - model.output(ordinary['states'][:n])).abs().max()),
            'untargeted_original_recoveries': int((~target & ~clean_correct & predicted_original).sum()),
            'untargeted_original_corruptions': int((~target & clean_correct & ~predicted_original).sum()),
        }
    oracle = model.rollout(batch.receiver, state_replacement=hybrid, patch_step=step)
    records['hybrid_state_oracle'] = {
        **correctness(oracle['logits'], batch.labels_h), 'targeted': target.tolist(),
        'native_hybrid_logits_max_error': float((oracle['logits'] - ordinary['logits'][2*n:]).abs().max()),
        'physical_l2_mean': float((hybrid - receiver).norm(dim=-1).mean()),
        'scope': 'structural continuation ceiling; unmatched dose; full hybrid input available',
    }
    chordal = ((live - fixed).double().square().sum(dim=(-1, -2)) / (2 * recipe.group_size)).sqrt()
    records['geometry'] = {
        'projector_chordal_by_slot': chordal.tolist(),
        'fixed_live_delta_cosine_mean': float(F.cosine_similarity(fixed_delta, live_delta, dim=-1).mean()),
        'fixed_live_delta_distance_mean': float((fixed_delta - live_delta).norm(dim=-1).mean()),
        'fixed_invalid': int((~fi['valid']).sum()), 'live_invalid': int((~li['valid']).sum()),
        'fixed_sham_norm_max_error': float((fixed_delta.norm(dim=-1) - deltas['causal_sham'].norm(dim=-1)).abs().max()),
    }
    records['physical_audit'] = physical
    records['ordinary_on_causal'] = correctness(ordinary['logits'][:n], batch.labels_r)
    return records


@torch.no_grad()
def anchor_readout(model, batch, groups, fixed, rotation, recipe, step):
    result = causal_readout(model, batch, groups, fixed, rotation, recipe)
    result.update({'step': step, 'scope': 'fixed training-partition anchor; never used for updates or selection'})
    return result


@torch.no_grad()
def evaluate(model, batches, binding, groups, fixed, rotation, recipe):
    model.eval()
    result = {'eval': {}}
    for name in ('iid', 'long'):
        batch = batches[name]
        result['eval'][name] = {**correctness(model(batch.receiver), batch.labels_r),
                                'batch_sha256': batch_hash(batch), 'steering_enabled': False}
    result.update(causal_readout(model, batches['causal'], groups, fixed, rotation, recipe))
    x, labels, metadata = binding
    result['binding'] = {**correctness(model(x), labels), **metadata, 'steering_enabled': False}
    return result


def run_root(root: int, out: Path, recipe: Recipe):
    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed_for(root, 'initialization'))
    out.mkdir(parents=True, exist_ok=False)
    start, cpu_start = time.time(), time.process_time()
    warm = new_model(recipe)
    optimizer = torch.optim.AdamW(warm.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
    warm_trace = []
    warm_stream = hashlib.sha256()
    for step in range(recipe.warmup_steps):
        delay = recipe.train_delays[step % len(recipe.train_delays)]
        batch = make_batch(root, 'warmup', step, recipe.batch_size, delay, 'train', recipe)
        warm_stream.update(batch_hash(batch).encode())
        features, labels = full_task_batch(batch)
        loss = F.cross_entropy(warm(features), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(warm.parameters(), recipe.gradient_clip)
        optimizer.step()
        if (step + 1) % 100 == 0:
            warm_trace.append({'step': step+1, 'task_ce': float(loss.detach())})
        if (step + 1) % 1000 == 0:
            print(json.dumps({'root': root, 'warmup_step': step+1, 'task_ce': float(loss.detach())}), flush=True)
    groups, fixed, attribution = discover_groups(warm, root, recipe)
    rotation = rotation_for(root, recipe.width)
    shared_hash = state_hash(warm)
    torch.save(warm.state_dict(), out / 'shared_warmup.pt')
    torch.save({'groups': groups, 'fixed_projectors': fixed, 'permutation': rotation[0], 'signs': rotation[1]}, out / 'geometry.pt')
    models = {arm: copy.deepcopy(warm) for arm in ARMS}
    initial_hashes = {arm: state_hash(model) for arm, model in models.items()}
    if any(value != shared_hash for value in initial_hashes.values()):
        raise AssertionError('Paired initialization differs')
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
                  for arm, model in models.items()}
    anchor = make_batch(root, 'anchor_train', 0, recipe.anchor_n, recipe.long_delay, 'train', recipe)
    shared_anchor = anchor_readout(warm, anchor, groups, fixed, rotation, recipe, 0)
    anchors = {arm: [shared_anchor] for arm in ARMS}
    traces = {arm: [] for arm in ARMS}
    stream = hashlib.sha256()
    for step in range(recipe.training_steps):
        batch = make_batch(root, 'paired_train', step, recipe.batch_size,
                           recipe.train_delays[step % len(recipe.train_delays)], 'train', recipe)
        digest = batch_hash(batch)
        stream.update(digest.encode())
        for arm in ARMS:
            row = training_step(models[arm], optimizers[arm], batch, groups, fixed, rotation, recipe, arm)
            traces[arm].append({'step': step, 'batch_sha256': digest, **row})
        if step + 1 in recipe.anchor_steps:
            for arm in ARMS:
                anchors[arm].append(anchor_readout(models[arm], anchor, groups, fixed, rotation, recipe, step+1))
            print(json.dumps({'root': root, 'paired_step': step+1,
                              'task_ce': {a: traces[a][-1]['task_ce'] for a in ARMS}}), flush=True)
    # Final scored inputs are first generated after every update has completed.
    batches = {name: make_batch(root, 'evaluation_'+name, 0, n, delay, 'eval', recipe)
               for name, n, delay in (('iid', recipe.evaluation_n, recipe.iid_delay),
                                      ('long', recipe.evaluation_n, recipe.long_delay),
                                      ('causal', recipe.causal_n, recipe.long_delay))}
    binding = binding_batch(root, 'evaluation_binding', recipe.binding_n, recipe.long_delay, 'eval', recipe)
    result = {'root': root, 'evidence_class': 'DEVELOPMENT_ONLY', 'recipe': asdict(recipe),
              'shared_initial_state_sha256': shared_hash, 'arm_initial_state_sha256': initial_hashes,
              'warmup_stream_sha256': warm_stream.hexdigest(), 'training_stream_sha256': stream.hexdigest(),
              'attribution': attribution, 'warmup_trace': warm_trace,
              'sham_permutation': rotation[0].tolist(), 'sham_signs': rotation[1].tolist(),
              'parameter_count': sum(p.numel() for p in warm.parameters()), 'arms': {}}
    for arm, model in models.items():
        torch.save(model.state_dict(), out / f'{arm}.pt')
        result['arms'][arm] = {**evaluate(model, batches, binding, groups, fixed, rotation, recipe),
                               'final_state_sha256': state_hash(model), 'training': traces[arm], 'anchors': anchors[arm]}
    result['elapsed_seconds'] = time.time() - start
    result['process_cpu_seconds'] = time.process_time() - cpu_start
    result['process_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    dump(out / 'result.json', result)
    dump(out / 'artifact_hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({'root': root, 'complete': True,
                      'long_accuracy': {a: result['arms'][a]['eval']['long']['accuracy'] for a in ARMS}}), flush=True)
    return result


def source_hashes():
    hashes = rp02_source_hashes()
    for name in ('__init__.py', 'data.py', 'model.py', 'experiment.py', 'stats.py', 'audit.py',
                 'test_model.py', 'test_stats.py', 'test_training.py', 'PROTOCOL.md', 'requirements.txt'):
        path = HERE / name
        hashes[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def validate_lock(path: Path, recipe: Recipe):
    lock = json.loads(path.read_text())
    if lock.get('source_hashes') != source_hashes():
        raise ValueError('Source freeze mismatch')
    if lock.get('recipe') != json.loads(json.dumps(asdict(recipe))):
        raise ValueError('Recipe differs from source freeze')
    if lock.get('development_roots') != list(DEVELOPMENT_ROOTS):
        raise ValueError('Development roster mismatch')
    commit = lock.get('remote_commit', '')
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Source lock needs a published commit')
    return lock


def preflight_from_results(results, out: Path):
    rows = []
    prospective_development_unopened = all(result['root'] in SMOKE_ROOTS for result in results)
    for result in results:
        counts = {arm: sum(result['arms'][arm]['eval']['long']['correct']) for arm in ARMS}
        n = len(result['arms']['A_task']['eval']['long']['correct'])
        rows.append({'unit_id': str(result['root']),
                     'paired_effect': f"{counts['B_causal_fixed']-counts['A_task']}/{n}",
                     'sham_effects': [f"{counts['C_rotated_fixed']-counts['A_task']}/{n}"]})
    spec = {'schema_version': 1, 'experiment_id': 'BPD-RP-03-hidden-state-steering-development',
            'independent_unit': 'independently trained model root with four paired descendants',
            'alpha': '.05', 'scientific_min_effect': '.05', 'target_effect': '.10',
            'measurement_resolution': f"1/{results[0]['recipe']['evaluation_n']}",
            'sham_quantile': '.95', 'sham_confidence': '.95', 'planned_comparisons': 3,
            'confirm_units_planned': len(DEVELOPMENT_ROOTS), 'pilot': rows,
            'pilot_from_development_only': True, 'heldout_sealed': prospective_development_unopened,
            'units_independent': True,
            'planning_scope': 'Eight bounded development roots only; confirmation outside this protocol',
            'heldout_scope': ('Eight assigned development roots remain unopened' if prospective_development_unopened
                              else 'Declared development readout has been consumed; no confirmation cohort allocated'),
            'sham_scope': 'same-site physical norm and auxiliary gradient ratio matched; small-n population sham tail uncalibrated'}
    path = HERE.parent / 'repair_portfolio/evidence/preflight.py'
    loader = importlib.util.spec_from_file_location('rp03_pinned_preflight', path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    verdict = module.preflight(spec)
    dump(out / 'preflight_input.json', spec)
    dump(out / 'preflight_result.json', verdict)
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'development'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-lock', type=Path)
    parser.add_argument('--roots', nargs='+', type=int)
    args = parser.parse_args()
    recipe = Recipe()
    roster = SMOKE_ROOTS if args.phase == 'smoke' else DEVELOPMENT_ROOTS
    roots = tuple(args.roots) if args.roots else roster
    if len(set(roots)) != len(roots) or any(r not in roster for r in roots):
        parser.error('Roots must be unique members of the declared phase roster')
    lock = None
    if args.phase == 'development':
        if args.source_lock is None:
            parser.error('Development requires --source-lock')
        lock = validate_lock(args.source_lock, recipe)
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    dump(args.out / 'run_manifest.json', {'phase': args.phase, 'roots': list(roots),
         'evidence_class': 'DEVELOPMENT_ONLY', 'recipe': asdict(recipe), 'source_hashes': source_hashes(),
         'started_unix': start, 'python': platform.python_version(), 'torch': torch.__version__, 'source_lock': lock})
    results = [run_root(root, args.out / f'root_{root}', recipe) for root in roots]
    dump(args.out / 'summary.json', summarize(results))
    preflight_from_results(results, args.out)
    dump(args.out / 'completion.json', {'completed': True, 'roots': list(roots),
         'elapsed_seconds': time.time() - start,
         'result_sha256': {str(r): hashlib.sha256((args.out / f'root_{r}' / 'result.json').read_bytes()).hexdigest()
                           for r in roots}})


if __name__ == '__main__':
    main()
