"""Pure CPU fake-model test for collector mechanics, not native checkpoint evidence."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from mechinterp.bpd.v3.collect_native_dither import (
    collect, dither_updated, rollout_single, substream_seed,
)


class FakeTRM(torch.nn.Module):
    l_cycles = 2
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(11, 4)
        with torch.no_grad():
            self.embed_tokens.weight.fill_(0.025)

    def encode_inputs(self, ids):
        return self.embed_tokens(ids)

    def init_states(self, n):
        z = torch.zeros((n, 3, 4))
        return z.clone(), z.clone()

    def run_l_level(self, s, i):
        return torch.tanh(1.5*s + .2*i + 0.004)

    def digit_logits(self, z):
        x=z.mean(axis=-1, keepdim=True)
        return torch.cat([j*x for j in range(-4, 5)],dim=-1)


def test_dither_reproducible_and_shaped():
    m = FakeTRM().eval()
    inputs = np.array([[1,2,3],[3,2,1]], dtype=np.int64)
    a = collect(m, inputs, k=8, sigma=0.04, seed=13, readouts=(2,5,8))
    b = collect(m, inputs, k=8, sigma=0.04, seed=13, readouts=(2,5,8))
    assert a['target_logits'].shape == (2,8,3,3,9)
    assert a['static_logits'].shape == (2,3,3,9)
    np.testing.assert_array_equal(a['target_logits'], b['target_logits'])
    assert np.max(np.abs(a['target_logits'] - a['static_logits'][:,None])) > 0
    assert a['max_clean_repeat_logit_difference'] == 0


def test_zero_noise_matches_static_and_disallowed_readout():
    m=FakeTRM().eval()
    inp=np.array([[1,2,3]],dtype=np.int64)
    q=collect(m, inp, k=2, sigma=0, readouts=(2,5,8))
    np.testing.assert_array_equal(q['target_logits'][0,0],q['static_logits'][0])
    with pytest.raises(ValueError, match='readout'):
        rollout_single(m, torch.tensor([[1,2,3]]),seed=7,sigma=0.01,readouts=(1,4))


def test_duplicate_and_invalid_roster():
    m=FakeTRM()
    with pytest.raises(ValueError,match='duplicate'):
        collect(m,np.array([[1,2,3],[1,2,3]]),k=2)
    with pytest.raises(ValueError,match='token'):
        collect(m,np.array([[1,2,0]]),k=2)
    with pytest.raises(ValueError,match='sigma'):
        dither_updated(torch.ones(1,3,4),torch.Generator(),-1)
    assert substream_seed(1,0,0)!=substream_seed(1,0,1)
