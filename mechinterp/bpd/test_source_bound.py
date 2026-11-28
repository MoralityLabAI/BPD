import itertools
import torch
from mechinterp.bpd.source_bound import anchored_width, source_bound_penalty


def test_hard_masks():
    sites = torch.tensor([0., 1., 3., 7.], dtype=torch.float64)
    for v in itertools.product([0., 1.], repeat=3):
        gates = torch.tensor(v, dtype=sites.dtype)
        assert anchored_width(gates, sites) == (gates * sites[1:]).max()


def test_soft_expectation_gradient():
    sites = torch.tensor([0., 1., 3., 7.], dtype=torch.float64)
    p = torch.tensor([.2, .3, .8], dtype=sites.dtype, requires_grad=True)
    want = p.new_zeros(())
    for v in itertools.product([0., 1.], repeat=3):
        gates = torch.tensor(v, dtype=sites.dtype)
        want += torch.where(gates.bool(), p, 1 - p).prod() * (gates * sites[1:]).max()
    got = anchored_width(p, sites)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(torch.autograd.grad(got, p)[0], torch.autograd.grad(want, p)[0])


def test_layout_and_fixed_budget():
    mask = torch.ones(2, 6, 3, 3, dtype=torch.float64, requires_grad=True)
    cost = source_bound_penalty(mask, torch.arange(7.), torch.ones(3))
    assert cost.item() == 18
    cost.backward()
    assert torch.isfinite(mask.grad).all()


def test_singleton_late_consumer_costs_distance():
    assert anchored_width(torch.tensor([0., 0., 1.]), torch.arange(4.)) == 3


def test_reject_learnable_geometry_and_prices():
    import pytest
    with pytest.raises(ValueError):
        anchored_width(torch.ones(3), torch.arange(4.).requires_grad_())
    with pytest.raises(ValueError):
        source_bound_penalty(torch.ones(1, 3, 1, 2), torch.arange(4.), torch.ones(2, requires_grad=True))
