"""Synthetic receipt tests for clustering, conditional rates and pairing.

These fixtures contain hand-authored labels and predictions. They instantiate
no model, draw no RP04 inputs and open no checkpoint.
"""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from .common import CHECKPOINTS, CONDITIONS, ROOTS, SMOKE_ROOTS, Recipe
from .stats import scores, summarize, with_rates


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _fixture(folder, phase='smoke', wrong_clean_checkpoints=()):
    """Create parser fixtures with a known 1/16 effect in one contrast."""
    recipe = Recipe()
    roots = SMOKE_ROOTS if phase == 'smoke' else ROOTS
    n = recipe.smoke_base_pairs if phase == 'smoke' else recipe.base_pairs
    _write(folder/'run_manifest.json', {
        'phase': phase, 'roots': list(roots), 'recipe': asdict(recipe),
    })
    query = list(range(4))*n
    edit = [pair % 4 for pair in range(n) for _ in range(4)]
    target = [q == e for q, e in zip(query, edit)]
    labels_r = [(pair+q) % 4 for pair in range(n) for q in range(4)]
    labels_h = [(r+1) % 4 if t else r for r, t in zip(labels_r, target)]
    digest = 'a'*64
    for root in roots:
        directory = folder/f'root_{root}'
        _write(directory/'cases.json', {
            'root': root, 'phase': phase, 'n_base_pairs': n,
            'n_query_cases': 4*n, 'batch_sha256': digest,
            'targeted': target, 'labels_r': labels_r, 'labels_h': labels_h,
            'query': query, 'edited_slot': edit,
        })
        for checkpoint in CHECKPOINTS:
            conditions = {}
            for condition in CONDITIONS:
                predictions = list(labels_r)
                if condition == 'clean' and checkpoint in wrong_clean_checkpoints:
                    predictions = [(v+1) % 4 for v in labels_r]
                if condition == 'hybrid_span':
                    for i in [i for i, t in enumerate(target) if t][:n//16]:
                        predictions[i] = labels_h[i]
                conditions[condition] = {
                    'predictions': predictions,
                    'correct': [int(p == h) for p, h in zip(predictions, labels_h)],
                }
            _write(directory/(checkpoint+'.json'), {
                'root': root, 'checkpoint': checkpoint,
                'batch_sha256': digest, 'conditions': conditions,
            })


class ScoreTests(unittest.TestCase):
    def test_conditional_recovery_and_corruption_denominators(self):
        counts = scores(
            [1, 0, 2, 2, 3, 0], [1, 2, 1, 2, 3, 0],
            [0, 0, 1, 2, 3, 0], [True, True, False, False, False, False],
            [0, 0, 1, 1, 3, 1],
        )
        self.assertEqual(counts['targeted_correct'], 1)
        self.assertEqual(counts['targeted_n'], 2)
        self.assertEqual(counts['untargeted_corrupted'], 1)
        self.assertEqual(counts['untargeted_original_correct'], 2)
        self.assertEqual(counts['untargeted_recovered'], 2)
        self.assertEqual(counts['untargeted_original_wrong'], 2)
        self.assertEqual(counts['untargeted_correct'], 3)
        self.assertEqual(with_rates(counts)['untargeted_corruption_rate'], .5)

    def test_integer_target_mask_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'booleans'):
            scores([1, 0], [1, 0], [0, 0], [1, 0], [0, 0])

    def test_label_semantics_and_duplicate_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Hybrid labels'):
            scores([0, 0], [0, 0], [0, 0], [True, False], [0, 0])
        with self.assertRaisesRegex(ValueError, 'unique'):
            scores([1, 0], [1, 0], [0, 0], [True, False], [0, 0], [0, 0])


class SummaryTests(unittest.TestCase):
    def test_eight_roots_define_sign_flip_and_four_contrast_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _fixture(folder, 'development')
            result = summarize(folder)
        self.assertEqual(result['n_units'], 8)
        self.assertEqual(result['n_conditions'], 26)
        self.assertEqual(result['actual_inferential_comparisons'], 4)
        first = result['contrasts'][0]
        self.assertEqual(first['mean_effect'], 1/16)
        self.assertEqual(first['sign_flip']['total_assignments'], 256)
        self.assertEqual(first['sign_flip']['p_fraction'], '1/128')
        self.assertEqual(first['holm_adjusted_p_fraction'], '1/32')
        self.assertTrue(all(c['confirmation_claim'] is False for c in result['contrasts']))
        self.assertEqual(result['pooled']['B_causal_fixed']['hybrid_span']['targeted_n'], 2048)

    def test_zero_correct_unaffected_baseline_preserves_null_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _fixture(folder, wrong_clean_checkpoints=('shared_warmup',))
            result = summarize(folder)
        for condition in CONDITIONS:
            self.assertIsNone(result['pooled']['shared_warmup'][condition]['untargeted_corruption_rate'])
            self.assertIsNone(result['changes_from_shared_warmup']['B_causal_fixed'][condition]
                              ['untargeted_corruption_rate_change'])

    def test_malformed_edited_slot_inventory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _fixture(folder)
            path = folder/f'root_{SMOKE_ROOTS[0]}'/'cases.json'
            cases = json.loads(path.read_text())
            cases['edited_slot'].pop()
            _write(path, cases)
            with self.assertRaisesRegex(ValueError, 'Edited-slot inventory'):
                summarize(folder)

    def test_same_size_target_mask_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _fixture(folder)
            path = folder/f'root_{SMOKE_ROOTS[0]}'/'cases.json'
            cases = json.loads(path.read_text())
            cases['targeted'][0], cases['targeted'][1] = cases['targeted'][1], cases['targeted'][0]
            _write(path, cases)
            with self.assertRaisesRegex(ValueError, 'Target mask disagrees'):
                summarize(folder)

    def test_saved_correctness_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _fixture(folder)
            path = folder/f'root_{SMOKE_ROOTS[0]}'/'B_causal_fixed.json'
            snapshot = json.loads(path.read_text())
            snapshot['conditions']['hybrid_span']['correct'][0] ^= 1
            _write(path, snapshot)
            with self.assertRaisesRegex(ValueError, 'Saved correctness'):
                summarize(folder)

    def test_incomplete_frozen_roster_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _fixture(folder)
            path = folder/'run_manifest.json'
            manifest = json.loads(path.read_text())
            manifest['roots'].pop()
            _write(path, manifest)
            with self.assertRaisesRegex(ValueError, 'Incomplete frozen root roster'):
                summarize(folder)


if __name__ == '__main__':
    unittest.main()
