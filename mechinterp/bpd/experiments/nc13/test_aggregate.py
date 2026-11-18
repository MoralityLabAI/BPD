from copy import deepcopy
import pytest

from .aggregate import BUDGETS, ENDPOINTS, ROOTS, PROTOCOL_COMMIT, aggregate


def fixture_receipts():
    records = []
    losses = {'B_clean': .1, 'A_clean': .1, 'mixed': .22,
              'shared_E': .18, 'target_E': .15,
              'shared_D': .14, 'target_D': .20,
              'permuted_E': .24, 'permuted_D': .24}
    for i in range(8):
        losses[f'sham_D_{i}'] = .19 + .01 * i
        losses[f'sham_E_{i}'] = .20 + .01 * i
    for root in ROOTS:
        records.append({'study': 'BPD-NC-13', 'root': root,
                        'protocol_commit': PROTOCOL_COMMIT, 'complete': True,
                        'phase': 'DEVELOPMENT_ENGINEERING', 'global_failures': [],
                        'budgets': dict(BUDGETS),
                        'source_lock_sha256': 'a' * 64,
                        'admissions': {k: {'admitted': True, 'reasons': []} for k in losses},
                        'scores': {k: {'by_context': {'shared': {'final_nll_nats': v}}}
                                   for k, v in losses.items()}})
    return records


def test_fixed_roots_and_constant_effects_preserve_all_six_gates():
    result = aggregate(fixture_receipts())
    assert result['verdict'] == 'ENGINEERING_TRANSFER_RECOVERED'
    expected = [.04, .08, .09, .06, .085, .04]
    for name, value in zip(ENDPOINTS, expected):
        cell = result['endpoints'][name]
        assert cell['passed']
        assert cell['inference']['mean_nats'] == pytest.approx(value)
        assert cell['inference']['nominal_interval'] == pytest.approx([value, value])
        assert cell['inference']['n_independent_model_pairs'] == 16


def test_fidelity_blocks_recovery_even_when_relative_advantages_pass():
    records = fixture_receipts()
    for r in records:
        r['scores']['B_clean']['by_context']['shared']['final_nll_nats'] = .08
    result = aggregate(records)
    assert result['endpoints']['target_minus_shared_D']['passed']
    assert not result['endpoints']['shared_D_minus_clean']['passed']
    assert result['verdict'] == 'ENGINEERING_NO_RECOVERY_OR_INELIGIBLE'


def test_failed_sham_admission_preserves_context_endpoints_and_family():
    records = fixture_receipts()
    records[5]['admissions']['sham_D_3'] = {'admitted': False, 'reasons': ['NO_REGISTERED_BRACKET']}
    result = aggregate(records)
    assert result['comparison_family'] == 6
    assert result['endpoints']['context_gain_D']['passed']
    sham = result['endpoints']['median_sham_minus_shared_D']
    assert sham['status'] == 'INELIGIBLE'
    assert sham['inference'] is None
    assert sham['n_admitted_pairs'] == 15
    assert sham['n_available_effects'] == 16
    assert not sham['passed']
    assert result['verdict'] == 'ENGINEERING_NO_RECOVERY_OR_INELIGIBLE'


def test_missing_root_never_becomes_a_smaller_inference_cohort():
    result = aggregate(fixture_receipts()[:-1])
    assert result['missing_roots'] == [ROOTS[-1]]
    assert all(x['status'] == 'INELIGIBLE' and x['inference'] is None
               for x in result['endpoints'].values())


def test_global_competence_failure_blocks_every_endpoint():
    records = fixture_receipts()
    records[3]['global_failures'] = ['model_task_competence']
    result = aggregate(records)
    assert all(x['status'] == 'INELIGIBLE' for x in result['endpoints'].values())
    assert len(result['endpoints']['context_gain_E']['root_effects']) == 16


def test_duplicate_smoke_and_protocol_substitution_rejected():
    records = fixture_receipts()
    with pytest.raises(ValueError, match='duplicated'):
        aggregate(records + [records[0]])
    bad = deepcopy(records)
    bad[0]['root'] = 2601299
    with pytest.raises(ValueError, match='unregistered'):
        aggregate(bad)
    bad = deepcopy(records)
    bad[0]['protocol_commit'] = 'unregistered'
    with pytest.raises(ValueError, match='protocol'):
        aggregate(bad)


def test_nonfinite_scored_effect_is_an_error():
    records = fixture_receipts()
    records[0]['scores']['shared_D']['by_context']['shared']['final_nll_nats'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        aggregate(records)


def test_budget_changes_cannot_enter_registered_inference():
    records = fixture_receipts()
    records[0]['budgets']['target_gradient_union'] = 32
    with pytest.raises(ValueError, match='budget'):
        aggregate(records)


def test_source_lock_missing_or_mixed_rejected():
    records = fixture_receipts()
    del records[0]['source_lock_sha256']
    with pytest.raises(ValueError, match='source lock'):
        aggregate(records)
    records = fixture_receipts()
    records[0]['source_lock_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='mixed source locks'):
        aggregate(records)
    result = aggregate(fixture_receipts())
    assert result['source_lock_sha256'] == 'a' * 64


def test_recorded_nonfinite_failure_stays_ineligible_without_crashing():
    records = fixture_receipts()
    records[0]['scores']['shared_D']['by_context']['shared']['final_nll_nats'] = None
    records[0]['admissions']['shared_D'] = {'admitted': False, 'reasons': ['nonfinite_evaluation']}
    result = aggregate(records)
    assert result['endpoints']['context_gain_D']['status'] == 'INELIGIBLE'
    assert result['endpoints']['context_gain_E']['passed']
    assert result['mean_shared_nll_by_arm']['shared_D']['n_scored_roots'] == 15
