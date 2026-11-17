"""BASIS-01-MODULES fake-recursive-model tests; no checkpoint execution."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mechinterp.bpd.v3.collect_basis_inputs import (
    DEFAULT_SITES, capture_module_inputs, fixed_positions,
    phase_sparsity, position_seed, write_capture_files,
)


class MiniBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = torch.nn.Linear(6, 6, bias=False)
        self.down_proj = torch.nn.Linear(6, 6, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x + self.down_proj(torch.relu(self.o_proj(x))))


class MiniTRM(torch.nn.Module):
    l_cycles = 2
    puzzle_emb_len = 1
    total_len = 5

    def __init__(self, *, skip_second: bool = False) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(11, 6)
        self.layers = torch.nn.ModuleList([MiniBlock(), MiniBlock()])
        self.skip_second = skip_second
        with torch.no_grad():
            self.embed_tokens.weight.fill_(0.03)

    def encode_inputs(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed_tokens(ids)
        return torch.cat([torch.zeros_like(emb[:, :1]), emb], dim=1)

    def init_states(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = torch.zeros((n, 5, 6))
        return s.clone(), s.clone()

    def run_l_level(self, state: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        z = self.layers[0](state + injection)
        return z if self.skip_second else self.layers[1](z)


def fake_inputs():
    return np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)


def test_capture_shape_phase_and_position_alignment():
    model = MiniTRM().eval()
    a = capture_module_inputs(model, fake_inputs(), seed=17, n_passes=6,
                              positions_per_call=2)
    b = capture_module_inputs(model, fake_inputs(), seed=17, n_passes=6,
                              positions_per_call=2)
    assert set(a) == set(DEFAULT_SITES)
    for site, row in a.items():
        assert row['W'].shape == (6, 6)
        assert row['inputs'].shape == (24, 6)
        assert row['board_idx'].shape == (24,)
        assert row['token_pos'].shape == (24,)
        assert row['pass_idx'].shape == (24,)
        np.testing.assert_array_equal(row['inputs'], b[site]['inputs'])
        np.testing.assert_array_equal(row['token_pos'], b[site]['token_pos'])
        assert set(row['call_type']) == {0, 1, 2}
        assert np.all(row['call_type'] == row['pass_idx'] % 3)
        assert np.min(row['token_pos']) >= 1
        assert np.max(row['token_pos']) < 5
        positions = row['token_pos'].reshape(2, 6, 2)
        assert np.all(positions[..., 0] != positions[..., 1])
    ref = a[DEFAULT_SITES[0]]
    for site in DEFAULT_SITES[1:]:
        np.testing.assert_array_equal(ref['token_pos'], a[site]['token_pos'])
        np.testing.assert_array_equal(ref['board_idx'], a[site]['board_idx'])
        np.testing.assert_array_equal(ref['pass_idx'], a[site]['pass_idx'])


def test_saved_npz_roundtrip_and_sparse_diagnostics(tmp_path):
    a = capture_module_inputs(MiniTRM().eval(), fake_inputs(), n_passes=6,
                              positions_per_call=2)
    manifest = write_capture_files(
        a, tmp_path, checkpoint_key='FAKE_SOURCE_ONLY',
        checkpoint_sha256='a' * 64, roster_sha256='b' * 64,
        seed=20261117, n_boards=2, n_passes=6,
        positions_per_call=2,
    )
    assert manifest['status'] == 'DEVELOPMENT_ONLY'
    assert len(manifest['sites']) == 4
    for site, meta in manifest['sites'].items():
        assert len(meta['npz_sha256']) == 64
        with np.load(tmp_path / (site.replace('.', '_') + '.npz'),
                     allow_pickle=False) as data:
            np.testing.assert_array_equal(a[site]['inputs'], data['inputs'])
            np.testing.assert_array_equal(a[site]['W'], data['W'])
            assert str(data['checkpoint_sha256']) == 'a' * 64
            assert str(data['roster_sha256']) == 'b' * 64
        stats = meta['sparsity']
        assert len(stats['per_call_type']) == 3
        assert all(float(z['fraction_abs_le_rms_scale']['0.0']) >= 0
                   for z in stats['per_call_type'].values())


def test_reject_bad_sites_counts_and_sampling():
    with pytest.raises(ValueError, match='duplicate puzzle'):
        capture_module_inputs(MiniTRM(), np.array([[1, 2, 3, 4]] * 2),
                              n_passes=6)
    with pytest.raises(ValueError, match='missing linear module'):
        capture_module_inputs(MiniTRM(), fake_inputs(), n_passes=6,
                              sites=('layers.7.o_proj',))
    with pytest.raises(ValueError, match='complete recurrence'):
        capture_module_inputs(MiniTRM(), fake_inputs(), n_passes=5)
    with pytest.raises(ValueError, match='board-token positions'):
        capture_module_inputs(MiniTRM(), fake_inputs(), n_passes=6,
                              positions_per_call=5)
    with pytest.raises(RuntimeError, match='hook counts do not close'):
        capture_module_inputs(MiniTRM(skip_second=True), fake_inputs(),
                              n_passes=6)


def test_seed_schedule_and_phase_zero_handling():
    assert position_seed(1, 0, 0) != position_seed(1, 0, 1)
    p, _ = fixed_positions(seed=1, board=0, n_passes=6, prefix_len=1,
                           total_len=5, positions_per_call=2,
                           device=torch.device('cpu'))
    assert p.shape == (6, 2)
    assert all(len(set(row)) == 2 for row in p.tolist())
    zeros = np.zeros((6, 3))
    phase = np.array([0, 1, 2, 0, 1, 2])
    s = phase_sparsity(zeros, phase)
    assert all(data['fraction_abs_le_rms_scale']['0.0'] == 1.0
               for data in s['per_call_type'].values())
