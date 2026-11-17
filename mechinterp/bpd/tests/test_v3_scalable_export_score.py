"""BASIS-01-SCALE: independent oracle parity and source-admission CPU tests."""
from pathlib import Path
import numpy as np
import pytest

from mechinterp.bpd.v3.score_exported_basis import (
    make_basis, score_basis, score_npz, sha256,
)
from mechinterp.bpd.v3.floor_and_basis import basis_factorization as historical_basis
from mechinterp.bpd.v3.floor_and_basis import budget_report as historical_budget


@pytest.mark.parametrize('seed', [13, 31, 71, 89, 131, 167])
def test_streaming_matches_reference_on_all_six_bases(seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(5, 9))
    x = rng.normal(size=(43, 9)) * (rng.random((43, 9)) > .55)
    labels = np.arange(43, dtype=np.int64) % 7
    boards = np.arange(43, dtype=np.int32) // 14
    svd = np.linalg.svd(w, full_matrices=True)
    for family, kinds in [('input', ('neuron', 'svd', 'random')),
                          ('output', ('coordinate', 'svd', 'random'))]:
        for kind in kinds:
            v, u = make_basis(w, family, kind, seed=seed, svd=svd)
            vr, ur = historical_basis(w, family, kind, seed=seed)
            np.testing.assert_allclose((v @ u).T, w, rtol=0, atol=1e-11)
            np.testing.assert_allclose(v, vr, rtol=0, atol=1e-11)
            np.testing.assert_allclose(u, ur, rtol=0, atol=1e-11)
            old = historical_budget(w, x, v, u)['budgets']
            first = None
            for chunk in (7, 64):
                new = score_basis(w, x, v, u, chunk_size=chunk,
                                  call_type=labels, board_idx=boards)
                assert new['n_vectors'] == 43
                assert new['n_independent_board_roots'] == 4
                assert new['evidence_type'] == 'MODULE_OUTPUT_CERTIFICATE_ONLY'
                for b, olddata in old.items():
                    for key in ('mean_off','mean_active','mean_active_fraction',
                                'max_actual_minus_certificate','max_certificate_minus_budget',
                                'mean_relative_actual','mean_relative_certificate','mean_logit_kl'):
                        assert new['budgets'][b][key] == pytest.approx(olddata[key], abs=1e-10)
                    assert sum(z['n_vectors'] for z in new['budgets'][b]['per_call_type'].values()) == 43
                if first:
                    for b in old:
                        assert new['budgets'][b]['mean_active'] == pytest.approx(first['budgets'][b]['mean_active'], abs=1e-12)
                first = new


def test_numerical_kernel_counts_have_separate_denominator():
    rng = np.random.default_rng(81)
    w = rng.normal(size=(4, 7))
    x = rng.normal(size=(96, 7))
    v, u = make_basis(w, 'input', 'svd')
    out = score_basis(w, x, v, u)
    assert out['near_null_write_directions'] == 3
    assert out['non_null_write_directions'] == 4
    assert out['budgets']['0.03']['mean_active_nonnull_writes'] <= 4
    v, u = make_basis(w, 'input', 'neuron')
    assert score_basis(w, x, v, u)['near_null_write_directions'] == 0


def test_validated_npz_by_root_budget_and_phase(tmp_path: Path):
    rng = np.random.default_rng(111)
    ids = np.repeat(np.arange(4), 14)
    phases = np.tile(np.repeat(np.arange(7), 2), 4)
    filename = tmp_path / 'module.npz'
    np.savez_compressed(filename, W=rng.normal(size=(5, 9)),
                        inputs=rng.normal(size=(56, 9)),
                        board_idx=ids, call_type=phases,
                        pass_idx=phases, checkpoint_sha256='a'*64,
                        roster_sha256='b'*64)
    digest = sha256(filename)
    out = score_npz(filename, chunk_size=7, max_boards=2,
                    expected_sha256=digest, family='input')
    assert set(out) == {'_provenance', 'input/neuron', 'input/svd', 'input/random'}
    assert out['input/neuron']['n_vectors'] == 28
    assert out['input/neuron']['n_independent_board_roots'] == 2
    assert out['_provenance']['status'] == 'DEVELOPMENT_ONLY_REUSED_TRM_ROOTS'
    with pytest.raises(ValueError, match='SHA-256'):
        score_npz(filename, expected_sha256='bad')
    with pytest.raises(ValueError, match='exceeds'):
        score_npz(filename, max_boards=5)


def test_bad_phase_fails_closed(tmp_path: Path):
    filename = tmp_path / 'bad.npz'
    np.savez_compressed(filename, W=np.eye(4), inputs=np.ones((2,4)),
                        board_idx=np.array([0, 0]), call_type=np.array([0,0]),
                        pass_idx=np.array([0,1]))
    with pytest.raises(ValueError, match='phase tags'):
        score_npz(filename)


def test_streaming_spot_check_large_n_without_dense_gramian():
    rng = np.random.default_rng(109)
    w = rng.normal(size=(64, 192))
    x = rng.normal(size=(1700, 192))
    v, u = make_basis(w, 'input', 'neuron')
    result = score_basis(w, x, v, u, chunk_size=47)
    assert result['n_vectors'] == 1700
    assert result['family_components'] == 192
    assert result['relative_full_on_residual'] < 1e-12
    assert all(q['max_actual_minus_certificate'] <= 1e-9
               for q in result['budgets'].values())


def test_psd_seminorm_matches_historical():
    rng=np.random.default_rng(422)
    w=rng.normal(size=(6,8));x=rng.normal(size=(36,8));z=rng.normal(size=(8,6))
    g=z.T@z
    v,u=make_basis(w,'input','neuron')
    old=historical_budget(w,x,v,u,gramian=g)['budgets']
    new=score_basis(w,x,v,u,gramian=g,chunk_size=7)
    assert new['metric']=='psd_seminorm'
    for b in old:
        assert new['budgets'][b]['mean_active'] == pytest.approx(old[b]['mean_active'], abs=1e-10)
        assert new['budgets'][b]['max_actual_minus_certificate'] < 1e-10
