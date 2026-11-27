"""Focused tests for the SCCG-10 / MMS-TRM instrument, atlas and trainer pieces."""
from __future__ import annotations

from fractions import Fraction
import copy
import random
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from mechinterp.bpd.experiments.trm_curation.data import edge_batch
from mechinterp.bpd.experiments.trm_curation.observer import observe_edge
from mechinterp.bpd.experiments.trm_curation.scheduler import edge_views
from mechinterp.bpd.experiments.trm_curation.transaction import probe_update

from . import geometry as geo
from .atlas import AtlasKey, MechanismAtlas, Policy, Premises, decide, quantize
from .common import load_admission, make_model, model_state_sha256, sccg_seed
from .drift import admission_on_support, drift_panel
from .economics import preview_update


def _model(seed=0):
    torch.manual_seed(seed)
    m = make_model()
    m.eval()
    return m


def _batch(edge, delay=7, seed=3):
    return edge_batch((edge,), root=29200, seed=seed, delay=delay, split='train', noise=.03)


VIEWS = edge_views(20261127)
PANEL = drift_panel(VIEWS['feedback'])


class PanelTests(unittest.TestCase):
    def test_panel_balanced_train_feedback(self):
        self.assertEqual(len(PANEL), 24)
        self.assertEqual(len(set(PANEL)), 24)
        self.assertTrue(set(PANEL) <= set(VIEWS['feedback']))
        self.assertEqual(sorted(e[2] for e in PANEL), sorted([0, 1, 2, 3] * 6))
        for s in range(4):
            pairs = {(min(r[s], h[s]), max(r[s], h[s])) for r, h, sl in PANEL if sl == s}
            self.assertEqual(len(pairs), 6)
        up = sum(1 for r, h, s in PANEL if r[s] < h[s])
        self.assertEqual(up, 12)

    def test_domain_separated_seed(self):
        from mechinterp.bpd.experiments.trm_curation.signal_bank import seed_for
        self.assertNotEqual(sccg_seed(29200, 'x'), seed_for(29200, 'x'))
        self.assertNotEqual(sccg_seed(29200, 'x'), sccg_seed(29201, 'x'))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.m = _model()
        self.b = _batch(PANEL[0])
        self.D = (2, 7)
        self.s = geo.post_write_state(self.m, self.b)

    def test_rows_match_native_observer(self):
        obs = observe_edge(self.m, self.b, self.D, dose=.75)
        _, J = geo.state_jacobian(self.m, self.b, self.D, self.s)
        self.assertLess(float((J - obs['observer']).abs().max()), 1e-6)
        idx = geo.row_index(self.b, self.D)
        self.assertLess(float((J[idx.target] - obs['target']).abs().max()), 1e-6)
        self.assertLess(float((J[list(idx.protected)] - obs['protected']).abs().max()), 1e-6)

    def test_forward_mode_equals_reverse_rows(self):
        torch.manual_seed(1)
        v = torch.randn(64)
        _, J = geo.state_jacobian(self.m, self.b, self.D, self.s)
        t = geo.directional_derivative(self.m, self.b, self.D, self.s, delta_state=v)
        self.assertLess(float((J @ v - t).abs().max()), 1e-4)
        w = torch.randn(64)
        _, Jw = geo.write_jacobian(self.m, self.b, self.D, self.s)
        tw = geo.directional_derivative(self.m, self.b, self.D, self.s, write=w)
        self.assertLess(float((Jw @ w - tw).abs().max()), 1e-4)

    def test_zero_write_is_identity_and_outcome_denominators(self):
        idx = geo.row_index(self.b, self.D)
        with torch.no_grad():
            clean = geo.native_logits(self.m, self.b, self.D, self.s)
        out = geo.finite_outcome(self.m, self.b, self.D, self.s, idx, clean, write=torch.zeros(64))
        self.assertEqual(out['target_gain'], 0.0)
        self.assertEqual(out['corruption_numerator'], 0)
        self.assertLessEqual(out['corruption_denominator'], 3 * len(self.D))

    def test_greedy_support_feasible(self):
        rng = np.random.default_rng(0)
        a, C = rng.normal(size=12), rng.normal(size=(5, 12))
        sup, res, calls = geo.greedy_support(a, C, np.eye(12), .75, rank=3, budget=.25)
        self.assertEqual(len(sup), 3)
        self.assertLessEqual(res['dose'], .75 + 1e-9)
        self.assertLessEqual(res['collateral'], .25 + 1e-6)
        self.assertGreater(calls, 0)

    def test_drift_metrics_identity(self):
        rng = np.random.default_rng(1)
        a, C = rng.normal(size=8), rng.normal(size=(4, 8))
        d = geo.drift_metrics(a, C, a, C)
        self.assertEqual(d['alpha'], 0.0)
        self.assertEqual(d['gamma'], 0.0)
        self.assertAlmostEqual(d['rowspace_sin_0.001'], 0.0, places=7)


class AdmissionBridgeTests(unittest.TestCase):
    def test_support_admission_matches_reference(self):
        adm = load_admission()
        G = np.eye(2)
        a0 = np.array([1.0, 0.0]); C0 = np.array([[0.0, 1.0]])
        r = admission_on_support(adm, G, a0, C0, a0, C0, np.array([.5, 0.0]), remainder_t=0.0, remainder_p=0.0)
        ref = adm.assess([[1, 0], [0, 1]], [1, 0], [[0, 1]], [Fraction(1, 2), 0], dose=Fraction(1, 2),
                         collateral=Fraction(1, 4), alpha=0, gamma=0, target_remainder=0,
                         protected_remainder=0, min_gain=Fraction(1, 10))
        self.assertEqual(r['verdict'], ref['verdict'])
        self.assertEqual(r['scope'], 'HEURISTIC_MEASURED_INPUTS')


def _key(ck='c0', frame='f0'):
    return AtlasKey(ck, 'm', 'site', 'phase', 'fp', ('all',), 'metric', frame, 'obj', 'float32', '0.75')


class AtlasTests(unittest.TestCase):
    def test_refuses_autograd_and_bounds_bytes(self):
        at = MechanismAtlas(0.05)
        with self.assertRaises(ValueError):
            at.insert(_key(), {'g': torch.ones(3, requires_grad=True)}, build_cpu_seconds=0, created_update=0)
        for i in range(20):
            at.insert(_key(ck=f'c{i}'), {'g': np.ones(1000)}, build_cpu_seconds=0, created_update=i)
        self.assertLessEqual(at.bytes, at.max_bytes)
        self.assertGreater(at.counters['evictions'], 0)
        self.assertIsNone(at.find(lambda k: k.source_checkpoint == 'c0'))
        self.assertIsNotNone(at.find(lambda k: k.source_checkpoint == 'c19'))

    def test_zero_mib_stores_nothing(self):
        at = MechanismAtlas(0)
        self.assertIsNone(at.insert(_key(), {'g': np.ones(3)}, build_cpu_seconds=0, created_update=0))

    def test_quantization_bound_holds(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(27, 64)) * np.logspace(-9, 2, 64)
        for dt in ('float16', 'float32'):
            q, b = quantize({'X': X}, dt)
            err = np.linalg.norm(q['X'].astype(np.float64) - X, 2)
            self.assertLessEqual(err, b['X'])
        with self.assertRaises(ValueError):
            quantize({'X': np.array([1e6])}, 'float16')

    def test_validation_requires_destination(self):
        at = MechanismAtlas(1)
        e = at.insert(_key(), {'g': np.ones(2)}, build_cpu_seconds=0, created_update=0)
        with self.assertRaises(ValueError):
            at.record_validation(e, {'trace_residual': 0})
        at.record_validation(e, {'destination_checkpoint': 'c1'})

    def test_decision_hierarchy(self):
        at = MechanismAtlas(1)
        e = at.insert(_key(), {'g': np.ones(2)}, build_cpu_seconds=0, created_update=0)
        pol = Policy(max_age=2)
        self.assertEqual(decide(None, age=0, param_distance=0, refresh_value=1).action, 'refresh')
        d = decide(e, age=3, param_distance=0, policy=pol, refresh_value=0, refresh_cost=1)
        self.assertEqual((d.action, d.tier), ('ordinary', 1))
        cert = lambda p: {'verdict': 'CONDITIONAL_REUSE_CANDIDATE'}
        d = decide(e, age=0, param_distance=0, premises=Premises(0, 0, 0, 0, 'uniform'), certificate=cert, policy=pol)
        self.assertEqual((d.action, d.tier, d.scope), ('reuse', 2, 'CERTIFIED_CONDITIONAL_IN_DECLARED_SPAN'))
        # Measured/sampled premises never certify; verification decides.
        good = lambda: {'target_gain': 1.0, 'protected_l2': 0.0}
        d = decide(e, age=0, param_distance=0, premises=Premises(0, 0, 0, 0, 'sampled'), certificate=cert,
                   verify=good, policy=pol)
        self.assertEqual((d.tier, d.scope), (4, 'VERIFIED_FINITE_ON_DECLARED_QUERIES'))
        bad = lambda: {'target_gain': 1.0, 'protected_l2': 9.0}
        d = decide(e, age=0, param_distance=0, verify=bad, policy=pol, refresh_value=2, refresh_cost=1)
        self.assertEqual(d.action, 'refresh')
        self.assertIn('not justified', d.reason)
        e.trace_residuals.append(5.0)
        self.assertEqual(decide(e, age=0, param_distance=0, policy=pol).tier, 1)


class PreviewTests(unittest.TestCase):
    def test_preview_matches_probe_and_restores(self):
        torch.set_num_threads(2)
        m = _model(4)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        b = _batch(PANEL[3], delay=2)
        fb = _batch(PANEL[5], delay=7)

        def tr():
            x = torch.cat((b.receiver, b.hybrid)); y = torch.cat((b.labels_r, b.labels_h))
            return F.cross_entropy(m(x), y)

        def cal():
            x = torch.cat((fb.receiver, fb.hybrid)); y = torch.cat((fb.labels_r, fb.labels_h))
            return F.cross_entropy(m(x), y)

        # one committed step so optimizer moments are nontrivial
        opt.zero_grad(); tr().backward(); opt.step()
        m.eval()
        g = tuple(x.detach().clone() for x in torch.autograd.grad(cal(), tuple(m.parameters())))
        sha = model_state_sha256(m)
        opt_state = copy.deepcopy(opt.state_dict())
        rng = torch.get_rng_state().clone(); py = random.getstate()
        pv = preview_update(m, opt, tr, g)
        self.assertEqual(model_state_sha256(m), sha)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(random.getstate(), py)
        for k, v in opt.state_dict()['state'].items():
            for f in ('exp_avg', 'exp_avg_sq', 'step'):
                self.assertTrue(torch.equal(v[f], opt_state['state'][k][f]))
        pr = probe_update(m, opt, tr, cal, cached_cal_grad=g, cached_cal_loss0=0.0)
        self.assertEqual(pv['predicted_gain'], pr['predicted_gain'])
        self.assertEqual(pv['train_loss'], pr['train_loss'])
        self.assertEqual(model_state_sha256(m), sha)


if __name__ == '__main__':
    unittest.main()
