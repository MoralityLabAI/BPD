"""Synthetic archive integrity and independent inference replay; no model runs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from . import audit
from .aggregate import aggregate, recompute_effects


def write_json(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True))


def make_fixture(directory, roots):
    source = directory / 'source.py'
    source.write_text('# synthetic archived source\n')
    lock = {'schema': 'BPD/NC13-execution-lock/v1', 'protocol_commit': audit.PROTOCOL,
            'development_roots': list(audit.ROOTS),
            'files_sha256': {'source.py': audit.file_hash(source)},
            'runtime_recipe': {'python_minor': '3.12', 'torch': '2.5.1+cpu', 'numpy': '2.3.5'}}
    write_json(directory / 'EXECUTE_LOCK.json', lock)
    lock_sha = audit.file_hash(directory / 'EXECUTE_LOCK.json')
    records = []
    for root in roots:
        rng = np.random.default_rng(root)
        inputs = rng.standard_normal((1024, 5)).astype(np.float32)
        shared = inputs[:, 4] > 0
        labels = {task: np.where(shared, inputs[:, 0], inputs[:, col]) > 0
                  for task, col in (('A', 1), ('B', 2))}
        arrays = {'inputs': inputs, 'labels_A': labels['A'], 'labels_B': labels['B']}
        arm_means = {'A_clean': .1, 'B_clean': .1, 'mixed': .25, 'shared_E': .20,
                     'shared_D': .12, 'target_E': .17, 'target_D': .22}
        arm_means.update({f'sham_D_{i}': .21 + i * .01 for i in range(8)})
        scores = {}
        for arm, base in arm_means.items():
            task = 'A' if arm == 'A_clean' else 'B'
            signs = np.where(labels[task], 1., -1.)
            row_nll = base + rng.uniform(0, .015, 1024)
            final = (-.5 * signs * np.log(np.expm1(row_nll))).astype(np.float32)
            arrays[f'logits__{arm}'] = np.repeat(final[:, None], 4, axis=1)
            raw = np.logaddexp(np.float32(0), (-2 * signs * final).astype(np.float32))
            scores[arm] = {'by_context': {}}
            for name, mask in (('shared', shared), ('specific', ~shared)):
                scores[arm]['by_context'][name] = {
                    'n': int(mask.sum()), 'final_nll_nats': float(raw[mask].mean()),
                    'accuracy': float(np.mean((final[mask] >= 0) == labels[task][mask]))}
        prefix = f'root_{root}'
        choices = {'synthetic_choice': np.asarray([root], dtype=np.int64)}
        weights = {f'{task}__weight': np.asarray([root + j], dtype=np.float64)
                   for j, task in enumerate(('A', 'B'))}
        training = {}
        for task in ('A', 'B'):
            h = hashlib.sha256()
            h.update(b'weight')
            h.update(audit.array_hash(weights[f'{task}__weight']).encode())
            training[task] = {'root': root, 'task': task,
                              'model_seed': root + (100000 if task == 'B' else 0),
                              'model_state_sha256': h.hexdigest()}
        for kind, data in (('weights', weights), ('choices', choices), ('eval', arrays)):
            np.savez_compressed(directory / f'{prefix}_{kind}.npz', **data)
        rec = {'study': 'BPD-NC-13', 'root': root, 'protocol_commit': audit.PROTOCOL,
               'source_lock_sha256': lock_sha, 'complete': True,
               'phase': 'EXCLUDED_SMOKE' if root == audit.SMOKE else 'DEVELOPMENT_ENGINEERING',
               'budgets': dict(audit.BUDGETS), 'runtime': {'python': '3.12.12 (fixture)',
                   'torch': '2.5.1+cpu', 'numpy': '2.3.5', 'torch_threads': 1},
               'source_hashes_sha256': lock['files_sha256'], 'training': training,
               'choice_array_hashes_sha256': {k: audit.array_hash(v) for k, v in choices.items()},
               'evaluation_array_hashes_sha256': {k: audit.array_hash(v) for k, v in arrays.items()},
               'evaluation_input_sha256': audit.array_hash(inputs), 'choices_frozen_utc': 'fixture',
               'scores': scores, 'global_failures': [],
               'admissions': {arm: {'admitted': True, 'reasons': []} for arm in scores}}
        rec['contrasts'] = {name: {'value': value} for name, value in recompute_effects(rec).items()}
        frozen = {key: value for key, value in rec.items() if key not in ('scores', 'contrasts')}
        frozen['complete'] = False
        write_json(directory / f'{prefix}_frozen.json', frozen)
        rec['artifacts'] = {kind: {'file': f'{prefix}{suffix}',
                                  'sha256': audit.file_hash(directory / f'{prefix}{suffix}')}
                            for kind, suffix in (('weights', '_weights.npz'), ('choices', '_choices.npz'),
                                                 ('eval', '_eval.npz'), ('frozen', '_frozen.json'))}
        write_json(directory / f'{prefix}.json', rec)
        write_json(directory / f'{prefix}.started.json',
                   {'study': 'BPD-NC-13', 'root': root, 'source_lock_sha256': lock_sha})
        records.append(rec)
    return records


def make_summary(directory, records):
    summary = aggregate(records)
    summary['receipt_sha256'] = {f'root_{rec["root"]}.json': audit.file_hash(directory / f'root_{rec["root"]}.json')
                                 for rec in records}
    summary['failure_sidecars'] = {
        path.name: {'sha256': audit.file_hash(path), 'receipt': json.loads(path.read_text())}
        for path in directory.glob('root_*.failure.json')}
    path = directory / 'SUMMARY.json'
    write_json(path, summary)
    return path


def test_sixteen_roots_raw_scores_and_bootstrap_replay_exactly(tmp_path):
    records = make_fixture(tmp_path, audit.ROOTS)
    summary = make_summary(tmp_path, records)
    result = audit.audit_receipts(tmp_path, summary, source_root=tmp_path)
    assert result['passed'], result['errors']
    assert result['summary_check']['maximum_numeric_absolute_error'] == 0.
    assert len(result['root_audits']) == 16
    assert all(cell['inference']['n_independent_model_pairs'] == 16
               for cell in result['replayed_summary']['endpoints'].values())
    altered = json.loads(summary.read_text())
    altered['endpoints']['context_gain_D']['inference']['nominal_interval'][0] += .001
    write_json(summary, altered)
    result = audit.audit_receipts(tmp_path, summary, source_root=tmp_path)
    assert not result['passed']
    assert any('nominal_interval' in error for error in result['errors'])


def test_excluded_smoke_score_only_and_postrun_source_context(tmp_path):
    make_fixture(tmp_path, [audit.SMOKE])
    (tmp_path / 'source.py').write_text('# later documentation-era source\n')
    result = audit.audit_receipts(tmp_path, source_root=tmp_path)
    assert result['passed'], result['errors']
    assert result['mode'] == 'EXCLUDED_SMOKE_SCORE_AUDIT'
    assert result['replayed_summary'] is None
    assert result['source_provenance']['current_tree_differences_from_lock'] == ['source.py']


@pytest.mark.parametrize('suffix', ['weights.npz', 'choices.npz', 'eval.npz', 'frozen.json'])
def test_artifact_tampering_is_detected(tmp_path, suffix):
    make_fixture(tmp_path, [audit.SMOKE])
    path = tmp_path / f'root_{audit.SMOKE}_{suffix}'
    path.write_bytes(path.read_bytes() + b' ')
    result = audit.audit_receipts(tmp_path, source_root=tmp_path)
    assert not result['passed']
    assert any('file hash mismatch' in error for error in result['errors'])


def test_receipt_score_forgery_detected_from_logits(tmp_path):
    records = make_fixture(tmp_path, [audit.SMOKE])
    records[0]['scores']['shared_D']['by_context']['shared']['final_nll_nats'] += .02
    write_json(tmp_path / f'root_{audit.SMOKE}.json', records[0])
    result = audit.audit_receipts(tmp_path, source_root=tmp_path)
    assert not result['passed']
    assert any('shared_D/shared raw NLL mismatch' in error for error in result['errors'])


def test_fatal_exposure_keeps_missing_pair_and_withholds_every_interval(tmp_path):
    records = make_fixture(tmp_path, audit.ROOTS[:-1])
    root = audit.ROOTS[-1]
    sidecar = {'study': 'BPD-NC-13', 'root': root, 'source_lock_sha256': records[0]['source_lock_sha256'],
               'complete': False, 'exception_type': 'SyntheticFailure', 'exception': 'fixture'}
    write_json(tmp_path / f'root_{root}.failure.json', sidecar)
    write_json(tmp_path / f'root_{root}.started.json', {k: sidecar[k] for k in ('study', 'root', 'source_lock_sha256')})
    summary = make_summary(tmp_path, records)
    result = audit.audit_receipts(tmp_path, summary, source_root=tmp_path)
    assert result['passed'], result['errors']
    assert result['exposure']['failure_sidecars'][0]['receipt'] == sidecar
    assert result['exposure']['started_without_complete_receipt'] == [root]
    assert all(cell['inference'] is None and cell['status'] == 'INELIGIBLE'
               for cell in result['replayed_summary']['endpoints'].values())


def test_receipt_source_lock_mismatch_is_fatal(tmp_path):
    records = make_fixture(tmp_path, [audit.SMOKE])
    records[0]['source_lock_sha256'] = 'a' * 64
    write_json(tmp_path / f'root_{audit.SMOKE}.json', records[0])
    result = audit.audit_receipts(tmp_path, source_root=tmp_path)
    assert not result['passed']
    assert any('source_lock_sha256 mismatch' in error for error in result['errors'])


def test_nonfinite_ineligible_arm_preserves_empty_mean_and_exposure(tmp_path):
    records = make_fixture(tmp_path, [audit.ROOTS[0]])
    rec = records[0]
    arm = 'sham_D_7'
    path = tmp_path / rec['artifacts']['eval']['file']
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays[f'logits__{arm}'][:] = np.nan
    np.savez_compressed(path, **arrays)
    rec['artifacts']['eval']['sha256'] = audit.file_hash(path)
    rec['evaluation_array_hashes_sha256'][f'logits__{arm}'] = audit.array_hash(arrays[f'logits__{arm}'])
    rec['admissions'][arm] = {'admitted': False, 'reasons': ['nonfinite_evaluation']}
    for score in rec['scores'][arm]['by_context'].values():
        score['final_nll_nats'] = None
    rec['contrasts'] = {name: {'value': value} for name, value in recompute_effects(rec).items()}
    write_json(tmp_path / f'root_{rec["root"]}.json', rec)
    summary = make_summary(tmp_path, records)
    result = audit.audit_receipts(tmp_path, summary, source_root=tmp_path)
    assert result['passed'], result['errors']
    assert result['replayed_summary']['mean_shared_nll_by_arm'][arm] == {
        'mean_shared_nll': None, 'n_scored_roots': 0}
    assert result['root_audits'][0]['scores'][arm]['finite'] is False
