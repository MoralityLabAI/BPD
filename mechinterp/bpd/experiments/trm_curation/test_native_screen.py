"""CURATE-03B native ComponentTRM analytical screen, joint parity, gates and acceptance."""
from collections import Counter
from dataclasses import replace
import copy
import itertools
import unittest

import torch
import torch.nn.functional as F

from mechinterp.bpd.experiments.trm_curation.batch_screen import mean_gradient
from mechinterp.bpd.experiments.trm_curation.campaign import CampaignRecipe
from mechinterp.bpd.experiments.trm_curation.custody import custody_digest
from mechinterp.bpd.experiments.trm_curation.data import edge_batch
from mechinterp.bpd.experiments.trm_curation.model import ComponentTRM
from mechinterp.bpd.experiments.trm_curation.native_screen import (
    NativeScreen, acquisition_order, audit_draw, exposure_legal, gate_decision,
    gradient_parity, score_agreement, edge_xy,
)
from mechinterp.bpd.experiments.trm_curation.scheduler import edge_views, pair_key
from mechinterp.bpd.experiments.trm_curation.transaction import probe_update

RECIPE = CampaignRecipe(width=8, components=6)
FIT = edge_views(20261127)['fit']
FB_EDGES = edge_views(20261127)['feedback'][:4]


def _projectors(width):
    eye = torch.eye(2 * width)
    return torch.stack([eye[:, :4] @ eye[:, :4].T] * 4)


def _setup(dtype=torch.float32, warm=0, clip=1.0, seed=0):
    torch.manual_seed(seed)
    model = ComponentTRM(width=8, components=6).to(dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    recipe = replace(RECIPE, gradient_clip=clip)
    batches = [_cast(edge_batch((FIT[i * 7],), root=28000, seed=i, delay=2), dtype) for i in range(8)]
    fb = _cast(edge_batch(FB_EDGES, root=28000, seed=99, delay=3), dtype)
    for b in batches[:warm]:
        model.train()
        x, y = edge_xy(b)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
    screen = NativeScreen(model, opt, recipe, _projectors(8).to(dtype))
    return model, opt, screen, batches, fb


def _cast(b, dtype):
    return replace(b, receiver=b.receiver.to(dtype), donor=b.donor.to(dtype), hybrid=b.hybrid.to(dtype))


class NativeScreenTests(unittest.TestCase):
    def test_analytic_matches_transactional_probe(self):
        """Cold/warm optimizer x clip regimes x both objectives at identical parents."""
        for warm, clip, objective in itertools.product((0, 3), (1.0, 0.02, 1e6), ('ordinary', 'mechanistic')):
            with self.subTest(warm=warm, clip=clip, objective=objective):
                model, opt, screen, batches, fb_batch = _setup(warm=warm, clip=clip)
                before = custody_digest(model, opt)
                fb = screen.feedback_gradient(fb_batch, objective)
                for b in batches[4:]:
                    grads, _ = screen.candidate_gradient(b)
                    a = screen.analytic_score(grads, fb)

                    def tr(b=b):
                        x, y = edge_xy(b)
                        return F.cross_entropy(model(x), y)
                    p = probe_update(model, opt, tr, lambda: screen.objective_loss(fb_batch, objective),
                                     clip=clip, cached_cal_grad=fb.grads, cached_cal_loss0=fb.loss0)
                    cal_norm = sum(float(g.double().square().sum()) for g in fb.grads) ** .5
                    agree = score_agreement(a['predicted_gain'], p['predicted_gain'], cal_norm, p['update_norm'])
                    self.assertTrue(agree['ok'], (a['predicted_gain'], p['predicted_gain'], agree))
                    if clip == 0.02:
                        self.assertLess(a['clip_scale'], 1.0)
                self.assertEqual(custody_digest(model, opt), before)

    def test_joint_gradient_parity_native_float32(self):
        # TiedComponentLinear computes in float32 (x.float() @ V); float64 native
        # ComponentTRM is unsupported, so the float64 tolerance applies to MLP fixtures only.
        for warm in (0, 2):
            model, opt, screen, batches, fb_batch = _setup(warm=warm)
            per_edge = [screen.candidate_gradient(b)[0] for b in batches[2:6]]
            native = screen.native_batch_gradient(batches[2:6])
            parity = gradient_parity(mean_gradient(per_edge), native)
            self.assertTrue(parity['ok'], parity)

    def test_step_minibatch_reject_restores_and_commit_matches_plain(self):
        model, opt, screen, batches, fb_batch = _setup(warm=2)
        ref = copy.deepcopy(model)
        ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-3, weight_decay=1e-4)
        ref_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        before = custody_digest(model, opt)
        fb = screen.feedback_gradient(fb_batch, 'mechanistic')
        per_edge = [screen.candidate_gradient(b)[0] for b in batches[2:6]]
        info = screen.step_minibatch(batches[2:6], fb=fb, fb_batch=fb_batch, commit_allowed=False,
                                     aggregated=mean_gradient(per_edge))
        self.assertFalse(info['accepted'])
        self.assertTrue(info['parity']['ok'], info['parity'])
        self.assertEqual(custody_digest(model, opt), before)
        info = screen.step_minibatch(batches[2:6], accept_test=False)
        self.assertTrue(info['accepted'])
        ref.train()
        x = torch.cat([edge_xy(b)[0] for b in batches[2:6]])
        y = torch.cat([edge_xy(b)[1] for b in batches[2:6]])
        loss = F.cross_entropy(ref(x), y)
        ref_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
        ref_opt.step()
        self.assertEqual(custody_digest(model, opt, include_rng=False),
                         custody_digest(ref, ref_opt, include_rng=False))

    def test_acceptance_gain_matches_probe_realized_gain(self):
        model, opt, screen, batches, fb_batch = _setup(warm=1)
        fb = screen.feedback_gradient(fb_batch, 'ordinary')
        info = screen.step_minibatch(batches[3:4], fb=fb, fb_batch=fb_batch, commit_allowed=False)
        p = probe_update(model, opt, lambda: F.cross_entropy(model(edge_xy(batches[3])[0]), edge_xy(batches[3])[1]),
                         lambda: screen.objective_loss(fb_batch, 'ordinary'),
                         cached_cal_grad=fb.grads, cached_cal_loss0=fb.loss0)
        self.assertAlmostEqual(info['actual_gain'], p['realized_gain'], places=6)

    def test_feedback_gradient_cache_and_side_effects(self):
        from mechinterp.bpd.experiments.trm_curation.cache import MechanisticCache
        model, opt, screen, batches, fb_batch = _setup(warm=1)
        screen.cache = MechanisticCache(mms_mib=4)
        before = custody_digest(model, opt)
        a = screen.feedback_gradient(fb_batch, 'mechanistic')
        b = screen.feedback_gradient(fb_batch, 'mechanistic')
        self.assertFalse(a.cache_hit)
        self.assertTrue(b.cache_hit)
        self.assertEqual(a.loss0, b.loss0)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.grads, b.grads)))
        self.assertEqual(custody_digest(model, opt), before)

    def test_acquisition_order_prospective_and_balanced(self):
        edges = [e for e in FIT if e[2] == 0][:4] + [e for e in FIT if e[2] == 1][:4] + \
                [e for e in FIT if e[2] == 2][:4] + [e for e in FIT if e[2] == 3][:4]
        o1 = acquisition_order(edges, split_seed=1, root=5, window=3)
        o2 = acquisition_order(list(reversed(edges)), split_seed=1, root=5, window=3)
        self.assertEqual(o1, o2)
        self.assertEqual([e[2] for e in o1], [0, 1, 2, 3] * 4)
        self.assertNotEqual(o1, acquisition_order(edges, split_seed=1, root=5, window=4))

    def test_gate_decisions(self):
        neg = [-1.0] * 16
        self.assertEqual(gate_decision(neg, None), (16, False))
        self.assertEqual(gate_decision(neg, 'full16'), (16, True))
        self.assertEqual(gate_decision(neg, 'early12'), (12, True))
        late = [-1.0] * 12 + [0.5, -1, -1, -1]
        self.assertEqual(gate_decision(late, 'full16'), (16, False))
        self.assertEqual(gate_decision(late, 'early12'), (12, True))
        early = [0.1] + [-1.0] * 15
        self.assertEqual(gate_decision(early, 'early12'), (16, False))
        self.assertEqual(gate_decision([0.0] * 16, 'full16'), (16, True))

    def test_audit_draw_known_probability(self):
        draws = [audit_draw(1, 'E', w, 0.25)[0] for w in range(4000)]
        self.assertAlmostEqual(sum(draws) / 4000, 0.25, delta=0.03)
        self.assertEqual(audit_draw(1, 'E', 7, 0.25), audit_draw(1, 'E', 7, 0.25))

    def test_exposure_legality_counts_reverse_edges(self):
        e = FIT[0]
        rev = (e[1], e[0], e[2])
        legal = exposure_legal(Counter({pair_key(e): 3}), cap=4)
        self.assertTrue(legal([e]))
        self.assertFalse(legal([e, rev]))


if __name__ == '__main__':
    unittest.main()
