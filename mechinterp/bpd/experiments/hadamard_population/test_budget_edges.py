"""Regression fixtures for recognizing completed states at budget boundaries.

These cases exercise certificate handling and consume at most one search flip.
They are unit checks, not new benchmark populations.
"""
import numpy as np
import pytest

import archive
import near
from had05 import State, certify, starts


SOLVED_M3 = np.array([[1, 1, 1], [1, 1, -1], [1, 1, -1], [1, 1, -1]], dtype=np.int64)
RUNNERS = [
    (archive, "best"),
    (archive, "archive64"),
    (archive, "random64"),
    (near, "best"),
    (near, "exact64"),
    (near, "near64_noinject"),
    (near, "near64"),
    (near, "random_near64"),
]


def invoke(module, arm, *, seed=3, population=1, rounds=0):
    return module.evaluate(seed, 3, population, 1, rounds, arm, 64, 8, 8)


@pytest.mark.parametrize("module,arm", RUNNERS)
@pytest.mark.parametrize("rounds", [0, 1])
def test_known_starting_solution_survives_zero_or_one_flip_budget(module, arm, rounds):
    xs, _, _ = starts(3, 3, 1)
    assert State(xs[0]).energy == 0
    result = invoke(module, arm, rounds=rounds)
    assert result["solved"]
    assert certify(result["certificate"])
    assert result["solution_kind"] == "native"
    assert result["best"] == 0
    assert result["candidate_units"] == result["archive_candidate_units"] == 0
    assert result["search_candidate_scores"] == 0
    assert result["trace"] == []
    if module is near:
        assert result["solution_injection"] == 0
        assert result["injections"] == []


@pytest.mark.parametrize("module,arm", RUNNERS)
def test_solution_in_later_slot_is_detected_before_initial_archive_work(module, arm, monkeypatch):
    xs = np.stack([np.ones((4, 3), dtype=np.int64), SOLVED_M3])

    def fixture_starts(seed, m, p):
        assert m == 3 and p == 2
        return xs.copy(), np.array([11, 12], dtype=np.int64), "fixture_later_slot"

    monkeypatch.setattr(module, "starts", fixture_starts)
    result = invoke(module, arm, population=2, rounds=1)
    assert result["solved"]
    assert result["certificate"] == SOLVED_M3.tolist()
    assert result["candidate_units"] == 0
    assert result["solution_kind"] == "native"


@pytest.mark.parametrize("module,arm", RUNNERS)
def test_zero_budget_with_unsolved_start_has_zero_cost(module, arm, monkeypatch):
    xs = np.ones((1, 4, 3), dtype=np.int64)

    def fixture_starts(seed, m, p):
        return xs.copy(), np.array([11], dtype=np.int64), "fixture_unsolved"

    monkeypatch.setattr(module, "starts", fixture_starts)
    result = invoke(module, arm, rounds=0)
    assert result["solved"] is False
    assert result["certificate"] is None
    assert result["candidate_units"] == 0
    assert result["archive_candidate_units"] == 0
    assert result["search_candidate_scores"] == 0
    assert result["best"] == 144

