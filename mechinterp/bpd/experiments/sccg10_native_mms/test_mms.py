"""Unit tests: atlas custody, admission ladder, controller, mechanism objective."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

import numpy as np
import torch

from ..trm_curation.data import edge_batch
from ..trm_curation.transaction import _restore_native_state, _snapshot_native_state
from .atlas import AtlasEntry, AtlasKey, MMSAtlas
from .common import Recipe, make_trm, views
from .economics import RateController, ResourceLedger
from .ladder import LadderConfig, certificate, screen, verify
from . import mechanism

HIST = (Path(__file__).resolve().parents[1] / 'trm_curation/results/development/root_28101/shared_warmup.pt')


def _key(fp='x'):
    return AtlasKey('ComponentTRM.joint_state', 'post_write_prefix4', 'suffix_step0_pre_low', fp,
                    'sccg01', 'float64', 0.75)


def _entry(fp='x', theta=None, gain=1.0, prot=0.2, window=0, n=8):
    theta = np.ones(100, np.float32) if theta is None else theta
    return AtlasEntry(key=_key(fp), source_checkpoint='s', source_factor_frame='f', metric_id='identity64',
                      support=tuple(range(n)),
                      payload={'delta': np.full(64, 0.75 / 8), 'a_support': np.ones(n), 'C_support': np.zeros((9, n)),
                               'theta_source': theta},
                      source_gain=gain, source_protected=prot, created_window=window)


class AtlasTests(unittest.TestCase):
    def test_refuses_non_mechanism_payload(self):
        a = MMSAtlas(1.0)
        e = _entry()
        e.payload['cal_grad'] = np.zeros(3)
        with self.assertRaises(ValueError):
            a.insert(e)
        e = _entry()
        e.payload['delta'] = torch.zeros(64)
        with self.assertRaises(ValueError):
            a.insert(e)

    def test_disabled_and_bounded(self):
        a0 = MMSAtlas(0.0)
        self.assertFalse(a0.insert(_entry()))
        self.assertIsNone(a0.lookup(_key(), 0))
        p, m = _entry().nbytes()
        a = MMSAtlas((2.5 * (p + m)) / 2**20)
        for i in range(4):
            self.assertTrue(a.insert(_entry(fp=str(i))))
        self.assertLessEqual(a.stats()['total_bytes'], a.max_bytes)
        self.assertGreater(a.stats()['evictions'], 0)
        self.assertIsNone(a.lookup(_key('0'), 1))
        self.assertIsNotNone(a.lookup(_key('3'), 1))


class LadderTests(unittest.TestCase):
    def test_screen(self):
        cfg = LadderConfig(max_age_windows=5, max_rel_displacement=0.1)
        th = np.ones(100, np.float32)
        self.assertEqual(screen(None, _key(), th, 0, cfg)[1], 'L1_MISS')
        self.assertEqual(screen(_entry(fp='y'), _key(), th, 0, cfg)[1], 'L1_PROVENANCE_MISMATCH')
        self.assertEqual(screen(_entry(), _key(), th, 6, cfg)[1], 'L1_AGE')
        self.assertEqual(screen(_entry(), _key(), th * 1.2, 1, cfg)[1], 'L1_DISPLACEMENT')
        self.assertTrue(screen(_entry(), _key(), th * 1.05, 1, cfg)[0])

    def test_verify_relative_to_source(self):
        cfg = LadderConfig()
        e = _entry(gain=1.0, prot=0.6)
        self.assertEqual(verify(e, {'target_gain': 0.9, 'protected_norm': 0.7}, 0.25, cfg)[0], 'L4_VERIFIED')
        self.assertEqual(verify(e, {'target_gain': 0.5, 'protected_norm': 0.7}, 0.25, cfg)[0], 'L4_BORDERLINE')
        self.assertEqual(verify(e, {'target_gain': 0.9, 'protected_norm': 0.8}, 0.25, cfg)[0], 'L4_FAILED')
        self.assertEqual(verify(e, {'target_gain': -0.1, 'protected_norm': 0.1}, 0.25, cfg)[0], 'L4_FAILED')

    def test_exact_mode_never_certifies(self):
        r = certificate(_entry(), np.ones(100, np.float32), 0.25, LadderConfig(mode='exact'))
        self.assertEqual(r['verdict'], 'L2_NOT_APPLICABLE_NO_UNIFORM_BOUND')

    def test_heuristic_mode_is_labelled(self):
        r = certificate(_entry(), np.ones(100, np.float32), 0.25,
                        LadderConfig(mode='heuristic', kappa_a=0.1, kappa_c=0.1, remainder=0.01))
        self.assertEqual(r['scope'], 'HEURISTIC_ENVELOPE_NOT_A_CERTIFICATE')


class ControllerTests(unittest.TestCase):
    def test_rate_choice_and_exploration(self):
        c = RateController(block=2, explore_every=3)
        self.assertEqual(c.choose(), 'ordinary')
        c.update('ordinary', 0.01, 1.0)
        self.assertEqual(c.choose(), 'curated')
        c.update('curated', 0.001, 1.0)
        self.assertEqual(c.choose(), 'curated')  # block index 2 is the exploration slot
        c.update('ordinary', 0.01, 1.0)
        self.assertEqual(c.choose(), 'ordinary')
        self.assertEqual(RateController(mode='always').choose(), 'curated')
        self.assertEqual(RateController(mode='never').choose(), 'ordinary')

    def test_ledger_rejects_unknown_category(self):
        L = ResourceLedger()
        with self.assertRaises(ValueError):
            with L.charge('free_lunch'):
                pass


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = make_trm(Recipe())
        if HIST.is_file():
            cls.model.load_state_dict(torch.load(HIST, weights_only=True))
        fb_edges = views(Recipe())['feedback'][:3]
        cls.fb = edge_batch(fb_edges, root=28101, seed=7, delay=15, split='train')

    def test_zero_directions_reduce_to_ordinary(self):
        loss, comps, rd = mechanism.mech_loss(self.model, self.fb, torch.zeros(3, 64))
        torch.testing.assert_close(loss, mechanism.ordinary_loss(self.model, self.fb))
        self.assertTrue(all(abs(r['target_gain']) < 1e-6 and r['protected_norm'] < 1e-6 for r in rd))

    def test_readout_matches_native_finite_outcome(self):
        from . import native
        L = ResourceLedger()
        p = native.make_panel(mechanism.slice_edge(self.fb, 1))
        m = mechanism.refresh_edge(self.model, p, (15,), 0.75, 0.25, 4, 2, L)
        d = torch.zeros(3, 64)
        d[1] = torch.as_tensor(m['delta'], dtype=torch.float32)
        _, _, rd = mechanism.mech_loss(self.model, self.fb, d)
        self.assertAlmostEqual(rd[1]['target_gain'], m['source_gain'], delta=1e-4)
        self.assertAlmostEqual(rd[1]['protected_norm'], m['source_protected'], delta=1e-4)

    def test_transaction_rollback_exact(self):
        model = copy.deepcopy(self.model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.cat((self.fb.receiver, self.fb.hybrid)); y = torch.cat((self.fb.labels_r, self.fb.labels_h))
        for _ in range(2):
            opt.zero_grad(); torch.nn.functional.cross_entropy(model(x), y).backward(); opt.step()
        snap = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), _snapshot_native_state(model))
        opt.zero_grad(); torch.nn.functional.cross_entropy(model(x), y).backward(); opt.step()
        model.load_state_dict(snap[0]); opt.load_state_dict(snap[1]); _restore_native_state(snap[2])
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, snap[0][k]))
        for j, p in enumerate(model.parameters()):
            for k, v in opt.state[p].items():
                self.assertTrue(torch.equal(v, snap[1]['state'][j][k]))


if __name__ == '__main__':
    unittest.main()
