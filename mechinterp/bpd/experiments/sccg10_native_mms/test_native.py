"""Parity of the SCCG-10 native instruments against CURATE-02 references.

Uses the exposed CURATE-02 development warmup checkpoint of root 28101 (historical,
replay/diagnostic use only) so derivatives are taken at a trained model.
"""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from ..trm_curation.data import edge_batch
from ..trm_curation.observer import observe_edge
from ..trm_curation.transaction import probe_update
from .common import Recipe, make_trm, views
from . import native

HIST = (Path(__file__).resolve().parents[1] / 'trm_curation/results/development/root_28101/shared_warmup.pt')
DELAYS = (2, 7, 31)


def _model():
    m = make_trm(Recipe())
    if HIST.is_file():
        m.load_state_dict(torch.load(HIST, weights_only=True))
    m.eval()
    return m


def _panel(i=0, delay=31):
    fit = views(Recipe())['fit']
    b = edge_batch((fit[37 * i % len(fit)],), root=28101, seed=1234 + i, delay=delay, split='train')
    return b


class NativeParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        cls.model = _model()

    def test_state_jacobian_matches_observe_edge(self):
        for i in range(2):
            b = _panel(i)
            ref = observe_edge(self.model, b, DELAYS, use_suffix=True)
            J = native.state_jacobian(self.model, native.make_panel(b), DELAYS)
            self.assertEqual(J.shape, ref['observer'].shape)
            torch.testing.assert_close(J, ref['observer'], rtol=1e-5, atol=1e-6)
            obs = native.observe_state(self.model, native.make_panel(b), DELAYS)
            np.testing.assert_allclose(obs.a, ref['target'].double().numpy(), rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(obs.C, ref['protected'].double().numpy(), rtol=1e-5, atol=1e-6)

    def test_write_jacobian_matches_finite_difference(self):
        # The native factors compute in float32; use a 4-point stencil at eps=1e-2
        # (rounding ~1e-5, truncation O(eps^4)).
        p = native.make_panel(_panel(1))
        J = native.write_jacobian(self.model, p, DELAYS).double()
        g = torch.Generator().manual_seed(5)
        u = torch.randn(2 * self.model.width, generator=g)
        u = u / u.norm()
        e = 1e-2
        f = lambda t: native.margins(self.model, p, DELAYS, write_delta=t * u).double()
        fd = (8 * (f(e) - f(-e)) - (f(2 * e) - f(-2 * e))) / (12 * e)
        torch.testing.assert_close(J @ u.double(), fd, rtol=1e-2, atol=1e-3)

    def test_jvp_matches_jacobian(self):
        p = native.make_panel(_panel(0))
        J = native.state_jacobian(self.model, p, DELAYS)
        u = torch.linspace(-1, 1, 2 * self.model.width)
        jv = native.state_jvp(self.model, p, DELAYS, u)
        torch.testing.assert_close(jv, J @ u, rtol=1e-4, atol=1e-5)

    def test_preview_matches_transactional_probe(self):
        model = copy.deepcopy(self.model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # Give the optimizer real moments first.
        warm_b = _panel(3, delay=2)
        for _ in range(3):
            x = torch.cat((warm_b.receiver, warm_b.hybrid)); y = torch.cat((warm_b.labels_r, warm_b.labels_h))
            opt.zero_grad(); F.cross_entropy(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        fb = _panel(4, delay=15)
        params = tuple(model.parameters())

        def cal():
            x = torch.cat((fb.receiver, fb.hybrid)); y = torch.cat((fb.labels_r, fb.labels_h))
            return F.cross_entropy(model(x), y)
        model.eval()
        g = [t.detach() for t in torch.autograd.grad(cal(), params)]
        for i in range(5, 8):
            cb = _panel(i, delay=1 + i % 3)

            def tr(b=cb):
                return F.cross_entropy(model(torch.cat((b.receiver, b.hybrid))),
                                       torch.cat((b.labels_r, b.labels_h)))
            before = copy.deepcopy(model.state_dict())
            ref = probe_update(model, opt, tr, cal, clip=1.0, cached_cal_grad=tuple(g),
                               cached_cal_loss0=0.0)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, before[k]))
            pv = native.preview_gain(model, opt, g, cb, 1.0)
            self.assertAlmostEqual(pv['predicted_gain'], ref['predicted_gain'],
                                   delta=1e-5 * max(1.0, abs(ref['predicted_gain'])))
            self.assertAlmostEqual(pv['update_norm'], ref['update_norm'], delta=1e-6)

    def test_finite_outcome_linear_regime(self):
        p = native.make_panel(_panel(2))
        obs = native.observe_state(self.model, p, DELAYS)
        u = obs.hybrid_dir * 1e-3
        out = native.finite_outcome(self.model, p, DELAYS, state_delta=u)
        lin = obs.rows @ u
        np.testing.assert_allclose(out['delta_margins'], lin, rtol=5e-2, atol=5e-5)


if __name__ == '__main__':
    unittest.main()
