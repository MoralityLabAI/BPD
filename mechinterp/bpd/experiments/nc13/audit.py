"""Audit archived NC13 observations without importing or executing a model.

NumPy float64 softplus checks the archived float32 Torch scores within 1e-6
absolute plus 1e-6 relative tolerance. Validated receipt means then permit exact
replay of the registered arithmetic and PCG64 bootstrap, including rounding.
Scientific ineligibility is reported separately from an archive-integrity error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOTS = tuple(range(2601300, 2601316))
SMOKE = 2601299
PROTOCOL = '069b355e8f37ec780dc6c46b82176b6f32f8e39b'
ENDPOINTS = ('context_gain_E', 'context_gain_D', 'shared_gap_reduction_D',
             'target_minus_shared_D', 'median_sham_minus_shared_D', 'shared_D_minus_clean')
REQUIRED = (('mixed', 'shared_E'), ('mixed', 'shared_D'),
            ('shared_E', 'target_E', 'shared_D', 'target_D'),
            ('target_D', 'shared_D'),
            ('shared_D',) + tuple(f'sham_D_{i}' for i in range(8)), ('shared_D',))
FLOORS = (.01, .01, .01, .01, .02, .05)
BUDGETS = {'steps': 384, 'train_A': 2048, 'train_B': 2048, 'source_pool': 4096,
           'source_per_condition': 1024, 'calibration_pool': 512,
           'activation_calibration': 128, 'target_gradient_union': 8,
           'dose': 1024, 'evaluation': 1024, 'sham_proposals': 2048,
           'shams': 8, 'search_grid': 17, 'search_bisection': 24}
ATOL = RTOL = 1e-6


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def _expect(ok: bool, message: str, errors: list[str]) -> None:
    if not ok:
        errors.append(message)


def _json(path: Path) -> dict:
    def reject_nonfinite(value):
        raise ValueError(f'nonfinite JSON number: {value}')
    result = json.loads(path.read_text(), parse_constant=reject_nonfinite)
    if not isinstance(result, dict):
        raise ValueError(f'JSON object required: {path.name}')
    return result


def _safe_file(directory: Path, filename: str) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError('artifact reference must be a local basename')
    result = directory / filename
    if result.resolve().parent != directory.resolve():
        raise ValueError('artifact reference escapes receipt directory')
    return result


def _nlls(rec: dict) -> dict[str, float]:
    result = {}
    for arm, score in rec.get('scores', {}).items():
        value = score.get('by_context', {}).get('shared', {}).get('final_nll_nats')
        if isinstance(value, (int, float)) and np.isfinite(value):
            result[arm] = float(value)
    return result


def effects(losses: dict[str, float]) -> dict[str, float | None]:
    expressions = (
        lambda: losses['mixed'] - losses['shared_E'],
        lambda: losses['mixed'] - losses['shared_D'],
        lambda: losses['shared_E'] - losses['target_E'] - losses['shared_D'] + losses['target_D'],
        lambda: losses['target_D'] - losses['shared_D'],
        lambda: float(np.median([losses[f'sham_D_{i}'] for i in range(8)])) - losses['shared_D'],
        lambda: losses['shared_D'] - losses['B_clean'],
    )
    result = {}
    for name, expression in zip(ENDPOINTS, expressions):
        try:
            result[name] = expression()
        except KeyError:
            result[name] = None
    return result


def replay_summary(records: dict[int, dict]) -> dict:
    """Independent protocol arithmetic; roots always retain their registered slots."""
    values = {root: effects(_nlls(rec)) for root, rec in records.items()}
    endpoints = {}
    for j, name in enumerate(ENDPOINTS):
        rows = []
        for root in ROOTS:
            if root not in records:
                rows.append({'root': root, 'effect_nats': None, 'admitted': False,
                             'reasons': ['missing_root']})
                continue
            rec = records[root]
            reasons = list(rec.get('global_failures', []))
            if not rec.get('complete', False):
                reasons.append('root_execution_incomplete')
            for arm in REQUIRED[j]:
                admission = rec.get('admissions', {}).get(arm, {})
                if admission.get('admitted') is not True:
                    reasons.append(f'{arm}:ineligible')
                    reasons.extend(f'{arm}:{reason}' for reason in admission.get('reasons', []))
            value = values[root][name]
            if value is None:
                reasons.append('unavailable_arm_score')
            rows.append({'root': root, 'effect_nats': value, 'admitted': not reasons,
                         'reasons': reasons})
        admitted = all(row['admitted'] for row in rows)
        inference = None
        if admitted:
            paired = np.array([row['effect_nats'] for row in rows], dtype=np.float64)
            seed = 26130013 + 101 * j
            rng = np.random.Generator(np.random.PCG64(seed))
            draws = rng.integers(0, 16, size=(16384, 16))
            estimates = paired[draws].mean(axis=1)
            interval = np.quantile(estimates, [.05 / 12, 1 - .05 / 12], method='linear')
            inference = {'mean_nats': float(paired.mean()), 'nominal_interval': interval.tolist(),
                         'nominal_cell_coverage': 1 - .05 / 6, 'resamples': 16384,
                         'seed': seed, 'n_independent_model_pairs': 16}
        passed = bool(inference is not None and (
            inference['nominal_interval'][1] <= FLOORS[j] if j == 5
            else inference['nominal_interval'][0] > FLOORS[j]))
        available = [row['effect_nats'] for row in rows if row['effect_nats'] is not None]
        endpoints[name] = {'status': 'ADMITTED' if admitted else 'INELIGIBLE',
                           'threshold_nats': FLOORS[j],
                           'direction': 'upper_at_most' if j == 5 else 'lower_exceeds',
                           'passed': passed, 'inference': inference, 'root_effects': rows,
                           'attempted_mean_nats': float(np.mean(available)) if available else None,
                           'n_available_effects': len(available),
                           'n_admitted_pairs': sum(row['admitted'] for row in rows)}
    arms = sorted({arm for rec in records.values() for arm in rec.get('scores', {})})
    means = {}
    for arm in arms:
        arm_values = [_nlls(records[root])[arm] for root in ROOTS
                      if root in records and arm in _nlls(records[root])]
        means[arm] = {'mean_shared_nll': float(np.mean(arm_values)) if arm_values else None,
                      'n_scored_roots': len(arm_values)}
    recovered = all(endpoints[name]['passed'] for name in ENDPOINTS[3:])
    locks = {rec.get('source_lock_sha256') for rec in records.values()}
    return {'schema': 'BPD/NC13-original-aggregate/v1', 'study': 'BPD-NC-13',
            'protocol_commit': PROTOCOL, 'scientific_status': 'ENGINEERING_ONLY',
            'source_lock_sha256': next(iter(locks)) if len(locks) == 1 else None,
            'verdict': 'ENGINEERING_TRANSFER_RECOVERED' if recovered else 'ENGINEERING_NO_RECOVERY_OR_INELIGIBLE',
            'registered_roots': list(ROOTS), 'observed_roots': [r for r in ROOTS if r in records],
            'missing_roots': [r for r in ROOTS if r not in records], 'comparison_family': 6,
            'endpoints': endpoints, 'mean_shared_nll_by_arm': means,
            'diagnostic_flags': {name: endpoints[name]['passed'] for name in ENDPOINTS[:3]}}


def _compare(expected, observed, label: str, errors: list[str], differences: list[float]) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            errors.append(f'{label}: expected object')
            return
        for key, value in expected.items():
            if key not in observed:
                errors.append(f'{label}.{key}: missing')
            else:
                _compare(value, observed[key], f'{label}.{key}', errors, differences)
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            errors.append(f'{label}: list length mismatch')
        else:
            for j, (a, b) in enumerate(zip(expected, observed)):
                _compare(a, b, f'{label}[{j}]', errors, differences)
    else:
        if isinstance(expected, float) and isinstance(observed, (float, int)):
            differences.append(abs(expected - observed))
        _expect(expected == observed, f'{label}: replay mismatch', errors)


def _audit_root(path: Path, rec: dict, lock: dict, lock_sha: str, errors: list[str]) -> dict:
    root = rec['root']
    label = f'root_{root}'
    start_errors = len(errors)
    _expect(type(root) is int and root in ROOTS + (SMOKE,), f'{label}: unregistered root', errors)
    _expect(path.name == f'{label}.json', f'{label}: receipt filename mismatch', errors)
    expected_phase = 'EXCLUDED_SMOKE' if root == SMOKE else 'DEVELOPMENT_ENGINEERING'
    for key, expected in {'study': 'BPD-NC-13', 'protocol_commit': PROTOCOL,
                          'source_lock_sha256': lock_sha, 'phase': expected_phase,
                          'budgets': BUDGETS, 'complete': True}.items():
        _expect(rec.get(key) == expected, f'{label}: {key} mismatch', errors)
    source_hashes = rec.get('source_hashes_sha256', {})
    _expect(bool(source_hashes), f'{label}: source hashes missing', errors)
    for source, digest in source_hashes.items():
        _expect(lock.get('files_sha256', {}).get(source) == digest,
                f'{label}: source hash outside execution lock: {source}', errors)
    if root != SMOKE:
        runtime = rec.get('runtime', {})
        recipe = lock.get('runtime_recipe', {})
        for key in ('torch', 'numpy'):
            _expect(runtime.get(key) == recipe.get(key), f'{label}: runtime {key} mismatch', errors)
        _expect(str(runtime.get('python', '')).startswith(str(recipe.get('python_minor', '')) + '.'),
                f'{label}: Python minor mismatch', errors)
        _expect(runtime.get('torch_threads') == 1, f'{label}: Torch thread count mismatch', errors)
    artifacts = {}
    for kind, suffix in (('weights', '_weights.npz'), ('choices', '_choices.npz'),
                         ('frozen', '_frozen.json'), ('eval', '_eval.npz')):
        reference = rec.get('artifacts', {}).get(kind, {})
        artifact = _safe_file(path.parent, reference.get('file'))
        _expect(artifact.name == label + suffix, f'{label}: {kind} filename mismatch', errors)
        digest = file_hash(artifact)
        _expect(digest == reference.get('sha256'), f'{label}: {kind} file hash mismatch', errors)
        artifacts[kind] = artifact
    frozen = _json(artifacts['frozen'])
    for key in ('study', 'root', 'phase', 'protocol_commit', 'source_lock_sha256',
                'budgets', 'source_hashes_sha256', 'choice_array_hashes_sha256', 'choices_frozen_utc'):
        _expect(frozen.get(key) == rec.get(key), f'{label}: frozen {key} mismatch', errors)
    _expect(frozen.get('complete') is False and 'scores' not in frozen and 'contrasts' not in frozen,
            f'{label}: frozen archive contains post-evaluation state', errors)
    array_counts = {}
    for kind, hash_key in (('choices', 'choice_array_hashes_sha256'), ('eval', 'evaluation_array_hashes_sha256')):
        with np.load(artifacts[kind], allow_pickle=False) as archive:
            expected_hashes = rec.get(hash_key, {})
            _expect(set(archive.files) == set(expected_hashes), f'{label}: {kind} array roster mismatch', errors)
            for key in archive.files:
                _expect(array_hash(archive[key]) == expected_hashes.get(key),
                        f'{label}: {kind} array hash mismatch: {key}', errors)
            array_counts[kind] = len(archive.files)
    with np.load(artifacts['weights'], allow_pickle=False) as weights:
        for task in ('A', 'B'):
            training = rec.get('training', {}).get(task, {})
            _expect(training.get('root') == root and training.get('task') == task and
                    training.get('model_seed') == root + (100000 if task == 'B' else 0),
                    f'{label}: {task} training lineage mismatch', errors)
            h = hashlib.sha256()
            names = sorted(key for key in weights.files if key.startswith(task + '__'))
            _expect(bool(names), f'{label}: {task} weights missing', errors)
            for key in names:
                h.update(key.split('__', 1)[1].encode())
                h.update(array_hash(weights[key]).encode())
            _expect(h.hexdigest() == training.get('model_state_sha256'),
                    f'{label}: {task} trained state hash mismatch', errors)
    checked = {}
    max_error = 0.
    with np.load(artifacts['eval'], allow_pickle=False) as archive:
        inputs = archive['inputs']
        if inputs.shape != (1024, 5) or not np.isfinite(inputs).all():
            raise ValueError('evaluation inputs must be finite with shape (1024, 5)')
        _expect(array_hash(inputs) == rec.get('evaluation_input_sha256'),
                f'{label}: evaluation input hash mismatch', errors)
        masks = {'shared': inputs[:, 4] > 0, 'specific': inputs[:, 4] <= 0}
        labels = {}
        for task, column in (('A', 1), ('B', 2)):
            labels[task] = archive[f'labels_{task}']
            expected = np.where(masks['shared'], inputs[:, 0], inputs[:, column]) > 0
            _expect(np.array_equal(labels[task], expected), f'{label}: task {task} labels mismatch', errors)
        archived_arms = {key.removeprefix('logits__') for key in archive.files if key.startswith('logits__')}
        _expect(archived_arms == set(rec.get('scores', {})), f'{label}: logit/score arm roster mismatch', errors)
        for arm in sorted(archived_arms):
            logits = archive[f'logits__{arm}']
            if logits.shape != (1024, 4):
                raise ValueError(f'{arm}: logit shape must be (1024, 4)')
            if not np.isfinite(logits).all():
                _expect(rec.get('admissions', {}).get(arm, {}).get('admitted') is False,
                        f'{label}: nonfinite logits claimed admitted: {arm}', errors)
                checked[arm] = {'finite': False}
                continue
            task = 'A' if arm == 'A_clean' or arm.startswith('A_owned_') else 'B'
            signs = np.where(labels[task], 1., -1.)
            nll = np.logaddexp(0., -2 * signs * logits[:, -1].astype(np.float64))
            contexts = {}
            for context, mask in masks.items():
                if not mask.any():
                    raise ValueError(f'evaluation context absent: {context}')
                computed = float(nll[mask].mean())
                reported = rec['scores'][arm]['by_context'][context]
                old = reported.get('final_nll_nats')
                valid = isinstance(old, (int, float)) and np.isfinite(old)
                error = abs(computed - old) if valid else None
                _expect(valid and np.isclose(computed, old, atol=ATOL, rtol=RTOL),
                        f'{label}: {arm}/{context} raw NLL mismatch', errors)
                _expect(reported.get('n') == int(mask.sum()), f'{label}: {arm}/{context} count mismatch', errors)
                accuracy = float(np.mean((logits[mask, -1] >= 0) == labels[task][mask]))
                _expect(np.isclose(accuracy, reported.get('accuracy', -1), atol=ATOL, rtol=RTOL),
                        f'{label}: {arm}/{context} accuracy mismatch', errors)
                if error is not None:
                    max_error = max(max_error, error)
                contexts[context] = {'n': int(mask.sum()), 'numpy_float64_nll_nats': computed,
                                     'receipt_nll_nats': old, 'absolute_error_nats': error,
                                     'accuracy': accuracy}
            checked[arm] = {'finite': True, 'by_context': contexts}
    replayed_effects = effects(_nlls(rec))
    raw_effects = effects({arm: score['by_context']['shared']['numpy_float64_nll_nats']
                          for arm, score in checked.items() if score['finite']})
    for name, value in replayed_effects.items():
        stored = rec.get('contrasts', {}).get(name, {}).get('value')
        equal = stored is None if value is None else isinstance(stored, (int, float)) and abs(value - stored) <= 1e-14
        _expect(equal, f'{label}: stored contrast mismatch: {name}', errors)
    return {'root': root, 'passed': len(errors) == start_errors, 'phase': expected_phase,
            'receipt_sha256': file_hash(path), 'arrays_checked': array_counts,
            'context_mask_rule': 'shared: inputs[:,4] > 0; specific: inputs[:,4] <= 0',
            'maximum_raw_nll_absolute_error_nats': max_error, 'scores': checked,
            'replayed_effects': replayed_effects, 'numpy_float64_effects': raw_effects,
            'global_failures': rec.get('global_failures', [])}


def audit_receipts(receipts: Path, summary: Path | None = None, *,
                   lock_path: Path | None = None, source_root: Path = REPO) -> dict:
    """Read existing artifacts only. Optional paths support isolated fixture audits."""
    receipts = Path(receipts)
    errors = []
    report = {'schema': 'BPD/NC13-independent-audit/v1', 'study': 'BPD-NC-13',
              'protocol_commit': PROTOCOL, 'passed': False,
              'raw_score_tolerance': {'absolute': ATOL, 'relative': RTOL},
              'bootstrap_comparison_tolerance': 0., 'registered_roots': list(ROOTS),
              'root_audits': [], 'exposure': {'started_sidecars': [], 'failure_sidecars': []},
              'errors': errors}
    try:
        archived_locks = sorted(receipts.rglob('EXECUTE_LOCK.json'))
        if lock_path is None:
            lock_path = archived_locks[0] if archived_locks else HERE / 'EXECUTE_LOCK.json'
        lock_path = Path(lock_path)
        lock_sha = file_hash(lock_path)
        lock = _json(lock_path)
        report['source_lock_sha256'] = lock_sha
        _expect(lock.get('schema') == 'BPD/NC13-execution-lock/v1' and
                lock.get('protocol_commit') == PROTOCOL and lock.get('development_roots') == list(ROOTS),
                'execution lock registration mismatch', errors)
        _expect(bool(lock.get('files_sha256')), 'empty execution lock', errors)
        for archived in archived_locks:
            _expect(file_hash(archived) == lock_sha, 'multiple differing archived execution locks', errors)
        current_differences = []
        current_missing = []
        for filename, digest in lock.get('files_sha256', {}).items():
            parts = PurePosixPath(filename)
            if parts.is_absolute() or '..' in parts.parts:
                raise ValueError('unsafe path in execution lock')
            _expect(isinstance(digest, str) and bool(re.fullmatch(r'[0-9a-f]{64}', digest)),
                    f'invalid locked source digest: {filename}', errors)
            current = Path(source_root) / filename
            if not current.is_file():
                current_missing.append(filename)
            elif file_hash(current) != digest:
                current_differences.append(filename)
        report['source_provenance'] = {
            'execution_lock_file': str(lock_path),
            'receipt_source_hashes_checked_against_archived_lock': True,
            'current_tree_differences_from_lock': current_differences,
            'current_tree_files_unavailable': current_missing,
            'scope': 'Receipt provenance uses its execution lock. Current-tree differences are retrospective context.'}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f'execution lock: {exc}')
        lock, lock_sha = {}, ''
    seen, receipt_hashes = {}, {}
    for path in sorted(receipts.rglob('root_*.json')):
        if not re.fullmatch(r'root_\d+\.json', path.name):
            continue
        try:
            rec = _json(path)
            root = rec['root']
            if type(root) is not int or root not in ROOTS + (SMOKE,) or root in seen:
                raise ValueError(f'unregistered or duplicated root: {root}')
            seen[root] = rec
            receipt_hashes[path.name] = file_hash(path)
            report['root_audits'].append(_audit_root(path, rec, lock, lock_sha, errors))
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            errors.append(f'{path.name}: {type(exc).__name__}: {exc}')
    for kind in ('started', 'failure'):
        exposed = report['exposure'][f'{kind}_sidecars']
        for path in sorted(receipts.rglob(f'root_*.{kind}.json')):
            try:
                sidecar = _json(path)
                root = sidecar.get('root')
                _expect(type(root) is int and root in ROOTS + (SMOKE,), f'{path.name}: unregistered root', errors)
                _expect(sidecar.get('study') == 'BPD-NC-13' and sidecar.get('source_lock_sha256') == lock_sha,
                        f'{path.name}: source lock/study mismatch', errors)
                if kind == 'failure':
                    _expect(root not in seen, f'{path.name}: failure conflicts with completed receipt', errors)
                exposed.append({'file': path.relative_to(receipts).as_posix(),
                                'sha256': file_hash(path), 'receipt': sidecar})
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f'{path.name}: {exc}')
    started = [item['receipt']['root'] for item in report['exposure']['started_sidecars']]
    _expect(len(started) == len(set(started)), 'duplicate started root exposure', errors)
    for root in seen:
        _expect(root in started, f'root_{root}: started sidecar missing', errors)
    report['exposure']['started_without_complete_receipt'] = sorted(set(started) - set(seen))
    report['observed_roots'] = sorted(seen)
    report['receipt_sha256'] = receipt_hashes
    development = {root: seen[root] for root in ROOTS if root in seen}
    smoke_only = set(seen) == {SMOKE} and all(root == SMOKE for root in started)
    report['mode'] = 'EXCLUDED_SMOKE_SCORE_AUDIT' if smoke_only else 'DEVELOPMENT_RECEIPT_AUDIT'
    report['replayed_summary'] = None if smoke_only else replay_summary(development)
    _expect(bool(seen or started or report['exposure']['failure_sidecars']), 'no NC13 exposure artifacts found', errors)
    if summary is not None:
        if smoke_only or SMOKE in seen:
            errors.append('development summary cannot include the excluded smoke')
        else:
            try:
                observed = _json(Path(summary))
                expected = dict(report['replayed_summary'])
                expected['receipt_sha256'] = receipt_hashes
                expected['failure_sidecars'] = {
                    Path(item['file']).name: {'sha256': item['sha256'], 'receipt': item['receipt']}
                    for item in report['exposure']['failure_sidecars']}
                differences = []
                _compare(expected, observed, 'summary', errors, differences)
                report['summary_check'] = {'file': str(summary), 'sha256': file_hash(Path(summary)),
                                           'maximum_numeric_absolute_error': max(differences, default=0.)}
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f'summary: {exc}')
    report['passed'] = not errors
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit_receipts(args.receipts, args.summary)
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'audit': 'PASS' if report['passed'] else 'FAIL',
                      'observed_roots': report['observed_roots'], 'errors': report['errors']}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
