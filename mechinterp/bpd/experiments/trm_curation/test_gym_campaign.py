"""Focused unit tests for BPD-TRM-GYM-01 multi-task TRM gym campaign."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from mechinterp.bpd.experiments.trm_curation.gym_campaign import (
    TASKS,
    _prospective_order,
    export_datagoblin_gym_v3,
    gen_gym_task,
    make_model,
    observer_parity_check,
    run_task_root,
    summarize_campaign,
)


class GymCampaignTests(unittest.TestCase):
    def test_g05_disjoint_train_eval_split(self):
        x_tr, _, _ = gen_gym_task('G05_3slot_copy_retrieve', 11, 256, 'train')
        x_ev, _, _ = gen_gym_task('G05_3slot_copy_retrieve', 12, 256, 'eval')
        # First 3 tokens encode the 3 written values in channels 7..10
        v_tr = x_tr[:, :3, 7:11].argmax(-1)
        v_ev = x_ev[:, :3, 7:11].argmax(-1)
        self.assertTrue(bool(((v_tr.sum(dim=1) % 3) != 0).all()))
        self.assertTrue(bool(((v_ev.sum(dim=1) % 3) == 0).all()))
        self.assertEqual(x_tr.shape[1], 9)
        self.assertEqual(x_ev.shape[1], 14)

    def test_all_gym_tasks_shape_and_labels(self):
        for task in TASKS:
            for split in ('train', 'eval'):
                x, y, n_cls = gen_gym_task(task, 7, 16, split)
                self.assertEqual(x.ndim, 3)
                self.assertEqual(x.shape[0], 16)
                self.assertEqual(x.shape[2], 11)
                self.assertEqual(y.shape, (16,))
                self.assertGreaterEqual(int(y.min()), 0)
                self.assertLess(int(y.max()), n_cls)

    def test_prospective_order_deterministic_permutation(self):
        o1 = _prospective_order('E31_2hop_pointer_chase', 29201, 5, 16)
        o2 = _prospective_order('E31_2hop_pointer_chase', 29201, 5, 16)
        o3 = _prospective_order('E31_2hop_pointer_chase', 29201, 6, 16)
        self.assertEqual(o1, o2)
        self.assertEqual(sorted(o1), list(range(16)))
        self.assertNotEqual(o1, o3)

    def test_observer_parity_exact_on_gym_task(self):
        warm = make_model(29000, 'E31_2hop_pointer_chase')
        fb = gen_gym_task('E31_2hop_pointer_chase', 99, 32, 'train')[:2]
        par = observer_parity_check(warm, 'E31_2hop_pointer_chase', 29000, fb, steps=4)
        self.assertTrue(par['state_identical'])
        self.assertTrue(par['custody_identical'])

    def test_end_to_end_smoke_and_datagoblin_v3_export(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'gym'
            res = run_task_root(
                'E31_2hop_pointer_chase', 29000, out,
                budget_cpu=0.4, threads=1, warm_steps_override=4,
            )
            self.assertTrue(res['observer_parity']['state_identical'])
            for arm in ('A_uniform_reinvest', 'B_loss_pool_2x', 'C_analytic_full16',
                        'D_joint_early12', 'E_fast_analytic_k4', 'F_shuffled_k4'):
                self.assertIn(arm, res['arms'])
                self.assertGreaterEqual(res['arms'][arm]['windows'], 1)
                if res['arms'][arm]['parity_checks'] > 0:
                    self.assertEqual(res['arms'][arm]['parity_ok'], res['arms'][arm]['parity_checks'])

            summ = summarize_campaign(out, ('E31_2hop_pointer_chase',), (29000,))
            self.assertTrue(summ['observer_parity_all_identical'])

            exp_dir = Path(td) / 'export_v3'
            man = export_datagoblin_gym_v3(out, ('E31_2hop_pointer_chase',), (29000,), exp_dir)
            raw = gzip.decompress((exp_dir / 'trajectories_gym_v3.jsonl.gz').read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), man['files']['trajectories_gym_v3.jsonl']['sha256'])
            rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line]
            self.assertGreater(len(rows), 0)
            for r in rows:
                self.assertEqual(r['split'], 'train')
                self.assertTrue(r['training_eligible'])
                self.assertFalse(r['evaluation_leakage'])


if __name__ == '__main__':
    unittest.main()
