import itertools
import unittest
import torch
from mechinterp.bpd.spectral_span import expected_span, spectral_span_penalty


class TestSpan(unittest.TestCase):
    def test_all_hard_masks(self):
        sites = torch.tensor([0., .25, 1., 1.25, 3.], dtype=torch.float64)
        for pattern in itertools.product([0, 1], repeat=5):
            gates = torch.tensor(pattern, dtype=torch.float64)
            selected = sites[gates.bool()]
            want = (selected.max() - selected.min()).item() if len(selected) else 0.
            self.assertAlmostEqual(expected_span(gates, sites).item(), want, places=12)

    def test_soft_against_enumeration(self):
        sites = torch.tensor([0., .25, 1., 3.], dtype=torch.float64)
        gates = torch.tensor([.1, .3, .7, .9], dtype=torch.float64, requires_grad=True)
        want = gates.new_zeros(())
        for pattern in itertools.product([0, 1], repeat=4):
            mask = torch.tensor(pattern, dtype=torch.bool)
            prob = torch.where(mask, gates, 1-gates).prod()
            selected = sites[mask]
            if len(selected):
                want = want + prob * (selected.max() - selected.min())
        got = expected_span(gates, sites)
        self.assertTrue(torch.allclose(got, want, rtol=1e-12, atol=1e-12))
        self.assertTrue(torch.allclose(torch.autograd.grad(got, gates)[0], torch.autograd.grad(want, gates)[0], rtol=1e-12, atol=1e-12))

    def test_bpd_schedule(self):
        mask = torch.ones(2, 4, 3, 5, dtype=torch.float64, requires_grad=True)
        cost = spectral_span_penalty(mask, torch.arange(4.), torch.ones(5))
        self.assertEqual(cost.item(), 15.)
        cost.backward()
        self.assertTrue(torch.isfinite(mask.grad).all())

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            expected_span(torch.ones(2), torch.tensor([1., 0.]))
        with self.assertRaises(ValueError):
            expected_span(torch.tensor([1., 1.5]), torch.tensor([0., 1.]))
        with self.assertRaises(ValueError):
            expected_span(torch.ones(2), torch.tensor([0., 1.], requires_grad=True))


if __name__ == '__main__':
    unittest.main()
