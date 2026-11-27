"""CURATE-03B custody: exact restoration on native ComponentTRM + AdamW."""
import copy
import random
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from mechinterp.bpd.experiments.trm_curation.custody import Custody, custody_digest
from mechinterp.bpd.experiments.trm_curation.data import edge_batch
from mechinterp.bpd.experiments.trm_curation.model import ComponentTRM
from mechinterp.bpd.experiments.trm_curation.scheduler import edge_views


def _setup(seed=0, warm_steps=2):
    torch.manual_seed(seed)
    model = ComponentTRM(width=8, components=6)
    model.register_buffer('scratch', torch.zeros(3), persistent=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    fit = edge_views(20261127)['fit']
    batches = [edge_batch((fit[i],), root=28000, seed=i, delay=2) for i in range(6)]
    for b in batches[:warm_steps]:
        _step(model, opt, b)
    return model, opt, batches


def _xy(b):
    return torch.cat((b.receiver, b.hybrid)), torch.cat((b.labels_r, b.labels_h))


def _step(model, opt, b, noise_gen=None):
    model.train()
    x, y = _xy(b)
    if noise_gen is not None:
        x = x + 1e-3 * torch.randn(x.shape, generator=noise_gen)
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss


def _mess(model, opt, b):
    """Mutate every custody surface, including buffer resize and new attributes."""
    _step(model, opt, b)
    torch.rand(3)
    random.random()
    np.random.rand()
    model.up.record_inputs = True
    model.rollout(_xy(b)[0][:2])
    model.up._ensure_pass_capacity(model.up.max_passes + 5)
    model.scratch.add_(1.0)
    model.register_buffer('created', torch.ones(1))
    model.up.surprise = [torch.ones(2, requires_grad=True) * 2]
    opt.param_groups[0]['lr'] = 0.5
    for p in model.parameters():
        p.grad = torch.full_like(p, 7.0)
    model.eval()


class CustodyTests(unittest.TestCase):
    def test_isolated_restores_every_surface(self):
        model, opt, batches = _setup()
        gen = torch.Generator().manual_seed(5)
        npg = np.random.default_rng(3)
        model.train()
        before = custody_digest(model, opt, (gen, npg))
        c = Custody(model, opt, (gen, npg))
        with c.isolated():
            _mess(model, opt, batches[3])
            torch.rand(2, generator=gen)
            npg.random()
        self.assertEqual(custody_digest(model, opt, (gen, npg)), before)
        self.assertFalse(hasattr(model.up, 'surprise'))
        self.assertNotIn('created', model._buffers)
        self.assertEqual(model.up.mean_comp_acts.shape[0], model.up.max_passes)

    def test_exception_restores_and_propagates(self):
        model, opt, batches = _setup()
        before = custody_digest(model, opt)
        c = Custody(model, opt)
        for ctx in (c.isolated, c.transaction):
            with self.assertRaises(RuntimeError):
                with ctx():
                    _mess(model, opt, batches[3])
                    raise RuntimeError('callback failure')
            self.assertEqual(custody_digest(model, opt), before)
        self.assertEqual(c.stats['exception_restores'], 1)

    def test_rejected_transaction_restores_and_commit_matches_plain_step(self):
        model, opt, batches = _setup()
        ref_model, ref_opt = copy.deepcopy(model), None
        ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-3, weight_decay=1e-4)
        ref_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        before = custody_digest(model, opt)
        c = Custody(model, opt)
        with c.transaction():
            _step(model, opt, batches[4])
        self.assertEqual(custody_digest(model, opt), before)
        with c.transaction() as txn:
            _step(model, opt, batches[4])
            txn.commit()
        _step(ref_model, ref_opt, batches[4])
        self.assertEqual(custody_digest(model, opt, include_rng=False),
                         custody_digest(ref_model, ref_opt, include_rng=False))

    def test_repeated_execution_is_identical(self):
        model, opt, batches = _setup()
        c = Custody(model, opt)
        before = custody_digest(model, opt)
        outs = []
        for _ in range(3):
            with c.isolated():
                torch.rand(1)
                loss = _step(model, opt, batches[5])
                outs.append((float(loss.detach()), custody_digest(model, opt)))
        self.assertEqual(len(set(outs)), 1)
        self.assertEqual(custody_digest(model, opt), before)

    def test_observer_only_preserves_trajectory(self):
        """Instrumented run (isolated observers between steps) == uninstrumented run, RNG included."""
        torch.manual_seed(11)
        plain, plain_opt, batches = _setup(seed=1, warm_steps=0)
        obs = copy.deepcopy(plain)
        obs_opt = torch.optim.AdamW(obs.parameters(), lr=1e-3, weight_decay=1e-4)
        gp, go = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
        torch.manual_seed(42)
        for b in batches:
            _step(plain, plain_opt, b, noise_gen=gp)
        digest_plain = custody_digest(plain, plain_opt, (gp,))
        torch.manual_seed(42)
        c = Custody(obs, obs_opt, (go,))
        for b in batches:
            for other in batches:
                with c.isolated():
                    _mess(obs, obs_opt, other)
                    torch.rand(4, generator=go)
            _step(obs, obs_opt, b, noise_gen=go)
        self.assertEqual(custody_digest(obs, obs_opt, (go,)), digest_plain)

    def test_stochastic_feedback_matched_rng(self):
        model, opt, batches = _setup()
        gen = torch.Generator().manual_seed(2)
        c = Custody(model, opt, (gen,))
        x, y = _xy(batches[0])

        def noisy_loss():
            z = x + 0.05 * torch.randn(x.shape) + 0.05 * torch.randn(x.shape, generator=gen)
            return float(F.cross_entropy(model(z), y)) + 1e-3 * random.random() + 1e-3 * np.random.rand()

        point = c.rng_point()
        a = noisy_loss()
        unmatched = noisy_loss()
        c.replay_rng(point)
        b = noisy_loss()
        self.assertEqual(a, b)
        self.assertNotEqual(a, unmatched)


if __name__ == '__main__':
    unittest.main()
