"""CPU-only deterministic BPD v3 checks, with no neural or protected evaluation data."""
import numpy as np
import pytest

from mechinterp.bpd.v3.floor_and_basis import (
    basis_factorization, budget_report, floor_report, kl, phase_pool_check,
    run_smoke, softmax, toy_logits,
)


def test_floor_closed_risk_identity_and_perfect_drive():
    logits = toy_logits(seed=711, boards=24, k=8)
    centre = softmax(logits).mean(axis=1)
    result = floor_report(logits, static=np.log(centre), driven=logits)
    assert 0.1 < result['j_delta_nats_per_cell'] < np.log(8)
    assert abs(result['risk_identity_residual']) < 1e-12
    assert abs(result['noise_independent_risk'] - result['j_delta_nats_per_cell']) < 1e-12
    assert abs(result['barycentre_kl']) < 1e-12
    assert abs(result['paired_driven_risk']) < 1e-12
    assert abs(result['conditional_leak_lower_bound'] - result['j_delta_nats_per_cell']) < 1e-12


def test_point_ensemble_floor_zero():
    logits = np.broadcast_to(np.array([1.0, -2.0]), (3, 8, 2, 4, 2))
    result = floor_report(logits, static=logits[:, 0], driven=logits)
    assert abs(result['j_delta_nats_per_cell']) < 1e-12
    assert abs(result['risk_identity_residual']) < 1e-12


def test_single_reference_kl_below_positive_floor_without_leak():
    # The single realized branch is a possible constant independent surrogate.
    # Matching that branch produces KL=0 against it, even though the ensemble has J>0.
    logits = np.zeros((1, 8, 1, 1, 2))
    logits[0, :4, 0, 0] = [8, -8]
    logits[0, 4:, 0, 0] = [-8, 8]
    result = floor_report(logits, static=logits[:, 0])
    assert result['j_delta_nats_per_cell'] > .69
    assert kl(softmax(logits[0, 0]), softmax(logits[0, 0])) < 1e-12
    assert result['noise_independent_risk'] > result['j_delta_nats_per_cell']


def test_exact_input_and_output_basis_and_all_budget_masks():
    rng = np.random.default_rng(411)
    w = rng.normal(size=(7, 11))
    inputs = rng.normal(size=(80, 11)) * (rng.random((80, 11)) > .8)
    for family, kinds in (('input', ('neuron', 'svd', 'random')),
                          ('output', ('coordinate', 'svd', 'random'))):
        for kind in kinds:
            v, u = basis_factorization(w, family, kind, seed=411)
            result = budget_report(w, inputs, v, u)
            assert result['relative_full_on_residual'] < 1e-11
            active = []
            for budget in ('0.01', '0.03', '0.1'):
                x = result['budgets'][budget]
                assert x['max_actual_minus_certificate'] <= 1e-10
                assert x['max_certificate_minus_budget'] <= 1e-10
                active.append(x['mean_active'])
            assert active == sorted(active, reverse=True)


def test_psd_metric_certificate():
    rng = np.random.default_rng(422)
    w = rng.normal(size=(5, 7))
    x = rng.normal(size=(80, 7))
    a = rng.normal(size=(9, 5))
    g = a.T @ a
    v, u = basis_factorization(w, 'input', 'neuron')
    report = budget_report(w, x, v, u, gramian=g)
    assert all(r['max_actual_minus_certificate'] < 1e-10
               for r in report['budgets'].values())
    with pytest.raises(ValueError, match='positive semidefinite'):
        budget_report(w, x, v, u, gramian=-np.eye(5))


def test_phase_pool_exact_atom_and_full_support_null():
    report = phase_pool_check(seed=331)
    assert report['coordinate_advantage_vs_45deg'] == pytest.approx(2 / 3)
    assert report['full_support_phase_range'] == 0
    assert report['pooled_minimizer_grid_indices'] == [0, 16]


def test_invalid_shapes_fail_closed():
    with pytest.raises(ValueError):
        floor_report(np.zeros((2, 1, 3, 4, 5)))
    with pytest.raises(ValueError):
        floor_report(np.zeros((2, 2, 3, 4, 5)), static=np.zeros((2, 4, 5)))
    with pytest.raises(ValueError):
        basis_factorization(np.zeros((3, 4)), 'input', 'unknown')


def test_smoke_stable_receipt_and_native_pending():
    result = run_smoke(seed=711)
    assert result['label'] == 'SYNTHETIC_SOURCE_ONLY'
    assert result['native_gaoxin492'].startswith('PENDING')
    assert result['native_yagizdevre'].startswith('PENDING')
