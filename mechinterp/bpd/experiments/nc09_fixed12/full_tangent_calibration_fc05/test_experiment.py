"""FC05 guards and source-before-label contract, without training any root."""
import json
from pathlib import Path

import pytest

from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_calibration_fc05 import experiment as e


def test_unregistered_and_reduced_calibration_budgets_rejected():
    for root in (14999, 15000, 15998, 16059, 17000):
        with pytest.raises(ValueError):
            e.validate_budget(root, **e.FULL_BUDGET)
    for field, value in e.FULL_BUDGET.items():
        budget = {**e.FULL_BUDGET, field: value - 1}
        with pytest.raises(ValueError):
            e.validate_budget(16000, **budget)
    e.validate_budget(15999, **{**e.FULL_BUDGET, "steps": 1, "solver_steps": 0})


def test_calibration_training_cannot_start_without_release(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "ROOT", tmp_path)
    monkeypatch.setattr(e, "protect_source_manifest", lambda: {})
    def forbidden_model():
        raise AssertionError("A model was created before checking its execution lock")
    monkeypatch.setattr(e, "DenseLooper", forbidden_model)
    with pytest.raises(FileNotFoundError):
        e.train_model(16000)
    with pytest.raises(ValueError):
        e.train_model(16000, steps=1)


def test_original_artifact_blocks_retraining_before_model_creation(tmp_path, monkeypatch):
    (tmp_path / "nc09_fc05_root_15999_source.npz").write_bytes(b"original")
    monkeypatch.setattr(e, "protect_source_manifest", lambda: {})
    def forbidden_training(*args, **kwargs):
        raise AssertionError("Original evidence caused a repeated training run")
    monkeypatch.setattr(e, "train_model", forbidden_training)
    with pytest.raises(FileExistsError):
        e.evaluate_calibration_root(15999, out_dir=tmp_path)


def test_incomplete_task_effects_remain_unset():
    record = {"root": 16000, "sham_nlls": [], "task_nll_by_arm": {}}
    assert e._effects(record, 8) is None


def test_source_audit_gate_blocks_task_input_access(monkeypatch):
    def forbidden_input(*args, **kwargs):
        raise AssertionError("Task inputs were opened without a source audit")
    monkeypatch.setattr(e, "sample_x", forbidden_input)
    with pytest.raises(AssertionError, match="Source archive and audit"):
        e.task_inputs_after_source_audit({}, 15999, 16)


def test_failed_source_audit_retains_task_outcomes(monkeypatch):
    sentinel = object()
    calls = []
    def inputs(root, split, count):
        calls.append((root, split, count))
        return sentinel
    def labels(value):
        assert value is sentinel
        calls.append("labels")
        return "labels"
    monkeypatch.setattr(e, "sample_x", inputs)
    monkeypatch.setattr(e, "external_labels", labels)
    record = {"source_audit_before_task_label_access": True,
              "source_archive": {}, "independent_source_audit": {"passed": False}}
    assert e.task_inputs_after_source_audit(record, 15999, 16) == (sentinel, "labels")
    assert calls == [(15999, 3, 16), "labels"]


def test_execution_lock_requires_exact_cohort_and_manifest(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(e, "ROOT", tmp_path)
    manifest = b'{"files":{}}\n'
    (tmp_path / "SOURCE_MANIFEST.json").write_bytes(manifest)
    lock = {"study": e.STUDY, "phase": "FIRST_ORIGINAL_59_ROOT_CALIBRATION",
            "model_roots": list(e.ROOTS), "protocol_commit_sha": e.PROTOCOL_COMMIT,
            "source_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "source_commit_sha": "a" * 40}
    (tmp_path / "EXECUTE_LOCK.json").write_text(json.dumps(lock))
    e.require_calibration_release(16000)
    lock["model_roots"] = list(e.ROOTS[:-1])
    (tmp_path / "EXECUTE_LOCK.json").write_text(json.dumps(lock))
    with pytest.raises(AssertionError):
        e.require_calibration_release(16000)
