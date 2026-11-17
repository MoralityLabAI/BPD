"""Source-only board-cluster confidence checks."""
import numpy as np
import pytest
from mechinterp.bpd.v3.floor_and_basis import floor_report, toy_logits
from mechinterp.bpd.v3.uncertainty import board_js, bootstrap_board_means


def test_cluster_matches_floor_point():
    z = toy_logits(seed=711, boards=24, k=8)
    b = board_js(z)
    f = floor_report(z)
    assert b.shape == (24, 3)
    assert b.mean() == pytest.approx(f['j_delta_nats_per_cell'])
    out = bootstrap_board_means(b, seed=20261117, repetitions=1000)
    assert out['point'] == pytest.approx(f['j_delta_nats_per_cell'])
    assert out['percentile95'][0] < out['point'] < out['percentile95'][1]
    assert len(out['readout_percentile95']) == 3


def test_bootstrap_reproducible_and_clustered():
    b = board_js(toy_logits(seed=711))
    a = bootstrap_board_means(b, seed=100, repetitions=800)
    c = bootstrap_board_means(b, seed=100, repetitions=800)
    assert a == c
    assert a['independent_boards'] == 24


def test_bad_shapes_fail():
    with pytest.raises(ValueError):
        board_js(np.zeros((4, 1, 2, 2, 2)))
    with pytest.raises(ValueError):
        bootstrap_board_means(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        bootstrap_board_means(np.ones((4, 3)), repetitions=99)
