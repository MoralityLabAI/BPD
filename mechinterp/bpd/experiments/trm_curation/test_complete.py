"""Comprehensive Section 1 verification suite for BPD-TRM-CURATE-02."""
from __future__ import annotations

import copy
import random
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from .cache import CacheKey, MechanisticCache
from .data import edge_batch
from .memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap, run_capped_subprocess
from .model import ComponentTRM
from .observer import finite_difference_observer, observe_edge
from .reference import (
    budgeted_selective_direction,
    sccg_frontier,
    selective_direction,
    semantic_edges,
    signed_observer_matched_sham,
)
from .scheduler import edge_views, pair_key
from .signal_bank import discover_slot_projectors, mechanistic_feedback_loss, ordinary_feedback_loss
from .transaction import probe_update


class Section1VerificationTests(unittest.TestCase):
    def test_parameter_count_and_shared_factors(self):
        torch.manual_seed(101)
        m = ComponentTRM(width=32, components=29, low_cycles=1, input_dim=11, values=4)
        self.assertEqual(m.parameter_count(), 5380)
        # Width-16 carried-state-size control (32 carried coordinates matching GRU)
        m16 = ComponentTRM(width=16, components=16, low_cycles=1, input_dim=11, values=4)
        self.assertEqual(2 * m16.width, 32)
        self.assertGreater(m16.parameter_count(), 0)

        # Shared physical BPD factors across high and low visits + exact reconstruction
        w_recon = m.up.reconstructed_weight()
        torch.testing.assert_close(w_recon, (m.up.V @ m.up.U).T, rtol=0, atol=1e-6)
        torch.testing.assert_close(m.native_component_operator(), m.up.V @ m.up.U, rtol=0, atol=0)

        # W_target invariance: mutating W_target must produce byte-identical outputs
        b = edge_batch(semantic_edges('train')[:2], root=28000, seed=7, delay=5)
        out0 = m.rollout(b.receiver)
        self.assertEqual(out0['block_calls'], 2 * b.receiver.shape[1])
        m.up.W_target.fill_(123.456)
        out1 = m.rollout(b.receiver)
        torch.testing.assert_close(out0['logits'], out1['logits'], rtol=0, atol=0)
        torch.testing.assert_close(out0['states'], out1['states'], rtol=0, atol=0)

    def test_exact_suffix_continuation_logits_and_gradients_parity(self):
        torch.manual_seed(202)
        m = ComponentTRM(width=32, components=29, low_cycles=1)
        b = edge_batch(semantic_edges('train')[:2], root=28000, seed=19, delay=7)
        x = b.receiver

        # 1) Clean state suffix parity (logits, states, block calls, pass_offset)
        full = m.rollout(x)
        pre = m.prefix_state(x, prefix_steps=4)
        suf = m.rollout_suffix(x[:, 4:], pre, prefix_steps=4)
        torch.testing.assert_close(suf['logits'], full['logits'], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(suf['states'], full['states'][:, 4:], rtol=1e-5, atol=1e-6)
        self.assertEqual(suf['block_calls'], 2 * (x.shape[1] - 4))
        self.assertEqual(suf['pass_offset'], 8)
        self.assertEqual(m.up.pass_offset, 8)

        # 2) Full hybrid-state replacement parity
        full_h = m.rollout(b.hybrid)
        suf_h = m.rollout_suffix(x[:, 4:], m.prefix_state(b.hybrid, prefix_steps=4), prefix_steps=4)
        torch.testing.assert_close(suf_h['logits'], full_h['logits'], rtol=1e-5, atol=1e-6)

        # 3) Exact autograd gradient parity w.r.t. post-write state AND all trainable params
        s_oracle = pre.detach().clone().requires_grad_(True)
        s_suffix = pre.detach().clone().requires_grad_(True)
        params = tuple(m.parameters())

        loss_oracle = F.cross_entropy(
            m.rollout(x, state_replacement=s_oracle, patch_step=4)['logits'], b.labels_r
        )
        grads_oracle = torch.autograd.grad(loss_oracle, (s_oracle,) + params)

        loss_suffix = F.cross_entropy(
            m.rollout_suffix(x[:, 4:], s_suffix, prefix_steps=4)['logits'], b.labels_r
        )
        grads_suffix = torch.autograd.grad(loss_suffix, (s_suffix,) + params)

        torch.testing.assert_close(loss_suffix, loss_oracle, rtol=1e-5, atol=1e-6)
        for go, gs in zip(grads_oracle, grads_suffix):
            torch.testing.assert_close(gs, go, rtol=1e-4, atol=1e-5)

        # 4) Observer parity: use_suffix=True vs use_suffix=False
        b1 = edge_batch(semantic_edges('train')[:1], root=28000, seed=31, delay=7)
        obs_full = observe_edge(m, b1, (0, 2, 7), dose=0.75, use_suffix=False)
        obs_suf = observe_edge(m, b1, (0, 2, 7), dose=0.75, use_suffix=True)
        for key in ('target', 'protected', 'observer', 'clean_logits', 'actual_logits', 'hybrid_direction'):
            torch.testing.assert_close(obs_suf[key], obs_full[key], rtol=1e-4, atol=1e-5)
        self.assertAlmostEqual(obs_suf['target_linear'], obs_full['target_linear'], places=5)
        self.assertAlmostEqual(obs_suf['target_actual'], obs_full['target_actual'], places=5)

    def test_independent_finite_difference_observer_jacobians(self):
        torch.manual_seed(303)
        m = ComponentTRM(width=16, components=14, low_cycles=1)
        m.eval()
        b = edge_batch(semantic_edges('train')[:1], root=28001, seed=42, delay=4)
        obs = observe_edge(m, b, (0, 2, 4), dose=0.75, use_suffix=True)
        fd = finite_difference_observer(m, b, (0, 2, 4), eps=5e-4, use_suffix=True)
        torch.testing.assert_close(obs['observer'], fd['observer'], rtol=2e-2, atol=2e-3)
        torch.testing.assert_close(obs['protected'], fd['protected'], rtol=2e-2, atol=2e-3)
        torch.testing.assert_close(obs['target'], fd['target'], rtol=2e-2, atol=2e-3)

    def test_complete_optimizer_and_native_state_rollback(self):
        torch.manual_seed(404)
        m = ComponentTRM(width=16, components=14, low_cycles=1)
        opt = torch.optim.AdamW(m.parameters(), lr=0.003, weight_decay=0.0001)
        b_tr = edge_batch(semantic_edges('train')[:2], root=28000, seed=11, delay=2)
        b_cal = edge_batch(semantic_edges('train')[2:4], root=28000, seed=12, delay=2)

        # Warm one AdamW step so moments exist
        opt.zero_grad()
        F.cross_entropy(m(b_tr.receiver), b_tr.labels_r).backward()
        opt.step()

        # Set distinctive native mutable state and external sampler RNG
        m.up.calls = 17
        m.up.pass_offset = 6
        m.up.recorded_inputs = [torch.ones(1, 16)]
        gen = torch.Generator(device='cpu').manual_seed(999)
        gen_before = gen.get_state().clone()
        weights_before = copy.deepcopy(m.state_dict())
        opt_before = copy.deepcopy(opt.state_dict())
        torch_rng_before = torch.get_rng_state().clone()

        def tr_loss():
            _ = torch.rand(1, generator=gen)
            _ = random.random()
            _ = np.random.rand()
            return F.cross_entropy(m(b_tr.receiver), b_tr.labels_r)

        def cal_loss():
            _ = torch.rand(1, generator=gen)
            return F.cross_entropy(m(b_cal.receiver), b_cal.labels_r)

        res = probe_update(m, opt, tr_loss, cal_loss, generators=(gen,))
        self.assertTrue(np.isfinite(res['predicted_gain']))
        self.assertTrue(np.isfinite(res['realized_gain']))
        self.assertEqual(m.up.calls, 17)
        self.assertEqual(m.up.pass_offset, 6)
        self.assertEqual(len(m.up.recorded_inputs), 1)
        self.assertTrue(torch.equal(gen.get_state(), gen_before))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng_before))
        for k, v in m.state_dict().items():
            torch.testing.assert_close(v, weights_before[k], rtol=0, atol=0)
        for k in opt_before['state']:
            for sk, sv in opt_before['state'][k].items():
                if isinstance(sv, torch.Tensor):
                    torch.testing.assert_close(opt.state_dict()['state'][k][sk], sv, rtol=0, atol=0)

        # Verify cached calibration gradient gives exact parity with uncached probe_update
        m.eval()
        loss0 = cal_loss()
        g0 = tuple(gi.detach().clone() for gi in torch.autograd.grad(loss0, tuple(m.parameters())))
        gen.set_state(gen_before)
        res_cached = probe_update(
            m, opt, tr_loss, cal_loss, generators=(gen,),
            cached_cal_grad=g0, cached_cal_loss0=float(loss0.detach()),
        )
        self.assertAlmostEqual(res_cached['predicted_gain'], res['predicted_gain'], places=6)
        self.assertAlmostEqual(res_cached['realized_gain'], res['realized_gain'], places=6)

    def test_semantic_partition_integrity(self):
        views = edge_views(20261127)
        self.assertEqual(len(views['fit']), 624)
        self.assertEqual(len(views['feedback']), 144)
        self.assertEqual(len(views['tune']), 144)
        all_train = set(semantic_edges('train'))
        all_eval = set(semantic_edges('eval'))
        union = set(views['fit']) | set(views['feedback']) | set(views['tune'])
        self.assertEqual(union, all_train)
        self.assertFalse(union & all_eval)
        # Both orientations stay together
        for name, edges in views.items():
            s = set(edges)
            for r, h, slot in edges:
                self.assertIn((h, r, slot), s, f'reverse missing in {name}')
        # Unordered pairs disjoint across views
        p_fit = {pair_key(e) for e in views['fit']}
        p_fb = {pair_key(e) for e in views['feedback']}
        p_tu = {pair_key(e) for e in views['tune']}
        self.assertFalse(p_fit & p_fb)
        self.assertFalse(p_fit & p_tu)
        self.assertFalse(p_fb & p_tu)

    def test_sccg_positive_budget_solver_and_sham(self):
        rng = np.random.default_rng(77)
        a = rng.standard_normal(12)
        C = rng.standard_normal((14, 12))  # full column rank => zero-collateral infeasible!
        B = np.eye(12)
        r0 = budgeted_selective_direction(a, C, B, 0.75, collateral_budget=0.0)
        self.assertEqual(r0['null_rank'], 0)
        self.assertFalse(r0['feasible'])
        self.assertEqual(r0['gain'], 0.0)

        # Positive collateral budget finds strictly optimal feasible vector
        r_b = budgeted_selective_direction(a, C, B, 0.75, collateral_budget=0.5)
        self.assertTrue(r_b['feasible'])
        self.assertLessEqual(r_b['dose'], 0.75 + 1e-10)
        self.assertLessEqual(r_b['collateral'], 0.5 + 1e-10)
        # Verify optimality against 500 random feasible vectors
        for _ in range(500):
            v = rng.standard_normal(12)
            v = v / np.linalg.norm(v) * 0.75
            cn = np.linalg.norm(C @ v)
            if cn > 0.5:
                v = v * (0.5 / cn)
            self.assertGreaterEqual(r_b['gain'] + 1e-9, float(a @ v))

        # Monotonicity along frontier
        front = sccg_frontier(a, C, B, 0.75, budgets=(0.0, 0.2, 0.5, 1.0, 5.0))
        gains = [f['gain'] for f in front]
        for g1, g2 in zip(gains, gains[1:]):
            self.assertLessEqual(g1, g2 + 1e-10)

        # Signed-observer-matched sham feasibility when nullspace exists
        J_small = rng.standard_normal((4, 12))
        d_ref = rng.standard_normal(12)
        d_ref = d_ref * (0.75 / np.linalg.norm(d_ref))
        sham = signed_observer_matched_sham(J_small, d_ref, 0.75, seed=5)
        self.assertTrue(sham['feasible'])
        self.assertLess(sham['observer_error'], 1e-10)
        self.assertAlmostEqual(sham['dose'], 0.75, places=10)

    def test_cache_equivalence_stale_rejection_and_no_autograd(self):
        cache = MechanisticCache(mms_mib=4.0)
        cache.bind_checkpoint('ckpt_A', 'frame_1')

        # Refuse tensors with requires_grad=True
        bad_key = CacheKey('observer', 'ckpt_A', 'm', 'v', 'in', 'obj', 'frame_1')
        with self.assertRaises(ValueError):
            cache.put(bad_key, {'t': torch.ones(4, requires_grad=True)})

        # Valid detached entry hits cleanly
        ok_val = {'t': torch.ones(4)}
        self.assertTrue(cache.put(bad_key, ok_val, compute_cpu_seconds=0.01))
        self.assertIsNotNone(cache.get(bad_key))
        self.assertEqual(cache.hits, 1)

        # Advancing checkpoint evicts and rejects stale lookups
        evicted = cache.bind_checkpoint('ckpt_B', 'frame_1')
        self.assertEqual(evicted, 1)
        self.assertIsNone(cache.get(bad_key))
        self.assertEqual(cache.stale_rejections, 1)

        # Exact parity between mms_mib=0 and mms_mib=16 on observe_edge
        torch.manual_seed(505)
        m = ComponentTRM(width=16, components=14, low_cycles=1)
        b = edge_batch(semantic_edges('train')[:1], root=28000, seed=13, delay=5)
        c0 = MechanisticCache(mms_mib=0.0)
        c16 = MechanisticCache(mms_mib=16.0)
        from .signal_bank import cached_observe_edge
        o0 = cached_observe_edge(m, b, (0, 2, 5), cache=c0)
        o16_a = cached_observe_edge(m, b, (0, 2, 5), cache=c16)
        o16_b = cached_observe_edge(m, b, (0, 2, 5), cache=c16)
        torch.testing.assert_close(o0['observer'], o16_a['observer'], rtol=0, atol=0)
        torch.testing.assert_close(o0['observer'], o16_b['observer'], rtol=0, atol=0)
        self.assertEqual(c16.hits, 1)
        self.assertEqual(c16.recomputations_avoided, 1)

        # Verify LRU eviction works cleanly under tight byte budget
        tiny = MechanisticCache(mms_mib=0.004)  # ~4 KiB
        tiny.bind_checkpoint('c1', 'f1')
        for idx in range(10):
            k_i = CacheKey('observer', 'c1', 'm', f'v{idx}', f'in{idx}', 'obj', 'f1')
            tiny.put(k_i, {'t': torch.zeros(256, dtype=torch.float32)}, compute_cpu_seconds=0.01)
        self.assertGreater(tiny.evictions, 0)
        self.assertLessEqual(tiny.payload_bytes + tiny.metadata_bytes, tiny.max_bytes)

    def test_mechanistic_feedback_loss_differentiable(self):
        torch.manual_seed(606)
        m = ComponentTRM(width=16, components=14, low_cycles=1)
        views = edge_views(20261127)
        projs, info = discover_slot_projectors(m, 28000, views['fit'], delay=2, n_edges=16)
        self.assertEqual(tuple(projs.shape), (4, 32, 32))
        fb = edge_batch(views['feedback'][:4], root=28000, seed=99, delay=2)
        loss_o = ordinary_feedback_loss(m, fb)
        loss_m = mechanistic_feedback_loss(m, fb, projs, dose=0.75)
        self.assertTrue(torch.isfinite(loss_o))
        self.assertTrue(torch.isfinite(loss_m))
        g = torch.autograd.grad(loss_m, tuple(m.parameters()))
        self.assertTrue(all(torch.isfinite(gi).all() for gi in g))

    def test_hard_memory_cap_subprocess(self):
        if sys.platform != 'win32':
            self.skipTest('Windows Job Object subprocess check')
        res = run_capped_subprocess(
            [sys.executable, '-c', 'b = bytearray(8 * 1024 * 1024); print(len(b))'],
            cap_bytes=DEFAULT_CAP_BYTES,
        )
        self.assertEqual(res['returncode'], 0, msg=res.get('stderr', ''))
        self.assertEqual(res['stdout'].strip(), str(8 * 1024 * 1024))
        self.assertGreater(res['peak_job_commit_bytes'], 8 * 1024 * 1024)
        self.assertLess(res['peak_job_commit_bytes'], DEFAULT_CAP_BYTES)
        # Verify hard OS enforcement: 256 MiB allocation under 64 MiB cap must fail
        over = run_capped_subprocess(
            [sys.executable, '-c', 'b = bytearray(256 * 1024 * 1024); print(len(b))'],
            cap_bytes=64 * 1024 * 1024,
        )
        self.assertNotEqual(over['returncode'], 0)

    def test_independent_manual_trm_rollout_parity(self):
        from .audit import manual_trm_rollout
        torch.manual_seed(707)
        m = ComponentTRM(width=32, components=29, low_cycles=1)
        m.eval()
        b = edge_batch(semantic_edges('train')[:3], root=28000, seed=55, delay=7)
        expected = m(b.receiver)
        actual = manual_trm_rollout(m.state_dict(), b.receiver, low_cycles=1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_datagoblin_export_rejects_eval_leakage(self):
        import json, tempfile
        from pathlib import Path
        from .export_datagoblin import export_datagoblin
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'src'
            r0 = src / 'root_28000'
            r0.mkdir(parents=True)
            e_tr = semantic_edges('train')[0]
            e_ev = semantic_edges('eval')[0]
            good_row = {
                'root': 28000, 'arm': 'D_mech_grad', 'window': 0, 'delay': 2,
                'epoch': 'w0:abc', 'checkpoint_sha256': '0' * 64,
                'edge': [list(e_tr[0]), list(e_tr[1]), e_tr[2]],
                'selected': True, 'selection_reason': 'ranked', 'window_accepted': True,
                'train_loss': 0.5, 'predicted_gain': 0.02, 'realized_gain': 0.015,
                'utility': 1.5, 'update_norm': 0.01, 'charged_cpu_seconds': 0.01,
                'window_cal_before': 0.6, 'window_cal_after': 0.58,
                'split': 'train', 'training_eligible': True,
            }
            (r0 / 'train_curation_decisions.jsonl').write_text(json.dumps(good_row) + '\n', encoding='utf-8')
            m = export_datagoblin(src, Path(td) / 'out_ok')
            self.assertEqual(m['total_candidate_decisions'], 1)
            self.assertEqual(m['evaluation_rows_exported'], 0)
            # Now inject an eval edge and assert fail-closed rejection
            bad_row = dict(good_row, edge=[list(e_ev[0]), list(e_ev[1]), e_ev[2]])
            (r0 / 'train_curation_decisions.jsonl').write_text(json.dumps(bad_row) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                export_datagoblin(src, Path(td) / 'out_bad')

    def test_build_signal_bank_smoke(self):
        from .signal_bank import build_signal_bank
        torch.manual_seed(808)
        m = ComponentTRM(width=16, components=14, low_cycles=1)
        views = edge_views(20261127)
        cache = MechanisticCache(mms_mib=8.0)
        bank = build_signal_bank(
            m, 28000, views, delays=(2, 5), dose=0.75,
            dose_grid=(0.25, 0.75), budgets=(0.0, 0.25),
            n_edges=4, n_svd_edges=8, cache=cache, tag='ut',
        )
        self.assertIn('admission', bank)
        self.assertEqual(bank['n_edges'], 4)
        self.assertIn('joint_budget_025', bank['support_families'])


if __name__ == '__main__':
    unittest.main()
