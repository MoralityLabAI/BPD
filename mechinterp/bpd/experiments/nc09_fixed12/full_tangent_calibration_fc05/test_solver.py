"""Selection identity and independent byte/source/task corruption checks.

Every input here is manufactured.  These tests train no model and cannot
open an FC05 calibration root.
"""
from __future__ import annotations

import base64
import copy
import hashlib

import numpy as np
import torch

from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_fc04 import solver as frozen
from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_calibration_fc05 import audit, solver
from mechinterp.bpd.experiments.nc08.experiment import passwise_source_candidates
from mechinterp.bpd.experiments.nc09_fixed12.experiment import optimal_fixed12, tuple_basis


def manufactured(n=96, seed=614, gradient_scale=.08):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn((n, 4, 20), generator=gen)
    z = torch.randn((n, 4, 20), generator=gen)
    g = torch.randn((n, 4, 20), generator=gen) * gradient_scale
    E = [torch.eye(20, dtype=torch.float64)[:, :r].clone() for r in (4, 3, 3, 2)]
    return a, torch.eye(20, dtype=torch.float64), z, g, E


def assert_original_fields_equal(original, observed):
    if isinstance(original, dict):
        for key, value in original.items():
            assert key in observed
            assert_original_fields_equal(value, observed[key])
    elif isinstance(original, list):
        assert len(original) == len(observed)
        for a, b in zip(original, observed):
            assert_original_fields_equal(a, b)
    else:
        assert original == observed


def test_instrumentation_preserves_executed_sampler_and_replayable_trace():
    args = manufactured()
    old_p, old = frozen.fit_full_tangent_shams(*args, 15999, starts=6, steps=12, want=2)
    new_p, new = solver.fit_full_tangent_shams(*args, 15999, starts=6, steps=12, want=2)
    assert_original_fields_equal(old, new)
    assert len(old_p) == len(new_p)
    assert all(torch.equal(a, b) for a, b in zip(old_p, new_p))
    trace_updates = 0
    for chart in new["charts"]:
        assert chart["terminal_reason"]
        for state in chart["iterations"]:
            y = state["parameters"]
            binary = base64.b64decode(y["b64_le_float64"])
            assert hashlib.sha256(binary).hexdigest() == y["sha256"]
            assert len(binary) == np.prod(y["shape"]) * 8
            assert len(state["normalized_residual"]) == 3
            if state["jacobian"] is not None:
                trace_updates += 1
                assert len(state["jacobian"]["jacobian_singular_values"]) == 3
                assert state["jacobian"]["direction_norm"] <= .5 + 1e-12
                assert 1 <= len(state["trials"]) <= 4
                for trial in state["trials"]:
                    assert trial["advance"] == (trial["penalty"] < state["penalty"]-1e-12)
    assert trace_updates == new["total_jacobian_corrections"]
    assert trace_updates > 0


def test_no_advance_exit_keeps_original_chart_selection(monkeypatch):
    real_step = frozen.minimum_norm_jacobian_step

    def zero_direction(f, y):
        delta, info = real_step(f, y)
        return torch.zeros_like(delta), dict(info, direction_norm=0.)

    monkeypatch.setattr(frozen, "minimum_norm_jacobian_step", zero_direction)
    monkeypatch.setattr(solver, "minimum_norm_jacobian_step", zero_direction)
    args = manufactured()
    _, old = frozen.fit_full_tangent_shams(*args, 15999, starts=2, steps=12, want=2)
    _, new = solver.fit_full_tangent_shams(*args, 15999, starts=2, steps=12, want=2)
    assert_original_fields_equal(old, new)
    assert new["selected"] == 1
    chart = new["charts"][1]
    assert chart["terminal_reason"] == "no_accepted_descent_step"
    assert len(chart["iterations"][0]["trials"]) == 4
    assert new["total_jacobian_corrections"] == 1


def test_numerical_exception_preserves_attempt_and_stops_new_charts(monkeypatch):
    def failure(f, y):
        raise ArithmeticError("manufactured numerical failure")

    monkeypatch.setattr(solver, "minimum_norm_jacobian_step", failure)
    selected, record = solver.fit_full_tangent_shams(
        *manufactured(), 15999, starts=3, steps=12, want=2)
    assert len(selected) == 1 and not record["eligible"]
    assert record["numerical_exception"]["type"] == "ArithmeticError"
    assert record["charts"][1]["iterations"][0]["parameters"]["sha256"]
    assert all(c["status"] == "not_started_after_exception" for c in record["charts"][2:])


def source_fixture(tmp_path):
    a, U, z, g, _ = manufactured(128, seed=70, gradient_scale=.0001)
    candidates = passwise_source_candidates(z, g)
    optimum = optimal_fixed12(candidates)
    E = tuple_basis(candidates, optimum["ranks"])
    selected, admission = solver.fit_full_tangent_shams(a, U, z, g, E, 15999)
    assert len(selected) == 8
    P0 = frozen.projectors_from_bases(E)
    arrays = {"a": a.double().numpy(), "z": z.double().numpy(),
              "g": g.double().numpy(), "U": U.double().numpy(),
              "dp12_projector": P0.double().numpy(),
              "sham_projectors": torch.stack(selected).numpy(),
              "discovery_inputs": np.zeros((len(a), 20)),
              "model__manufactured_state": np.ones(1)}
    for t, e in enumerate(E):
        arrays[f"E_{t}"] = e.double().numpy()
        arrays[f"C_{t}"] = solver.complementary_basis(e).numpy()
    path = tmp_path / "source.npz"
    record = {
        "root": 15999, "source_rank12_optimum": optimum,
        "source_dp12_projector_receipt": frozen.projector_digest(P0),
        "sham_admission": admission,
        "splits": {"derivative_discovery": len(a)},
        "passwise_source_candidates": {
            t: {r: {k: v for k, v in c.items() if k not in ("basis", "projector")}
                for r, c in per.items()} for t, per in candidates.items()
        },
        "source_archive": audit.save_npz(path, arrays),
    }
    return record, path, arrays


def test_independent_numpy_source_audit_detects_scalar_and_tensor_corruption(tmp_path):
    record, path, arrays = source_fixture(tmp_path)
    checked = audit.source_audit(record, path)
    assert checked["passed"] and checked["complete"], checked["errors"]
    changed = copy.deepcopy(record)
    changed["sham_admission"]["projectors"][0]["physical_write_rms"] *= 2
    failed = audit.source_audit(changed, path)
    assert not failed["passed"]
    assert "sham_0_physical_write_rms_mismatch" in failed["errors"]
    # Refreshing the byte receipt cannot conceal corruption of source moments.
    altered_arrays = dict(arrays, a=arrays["a"] * 2.)
    changed = copy.deepcopy(record)
    changed["source_archive"] = audit.save_npz(path, altered_arrays)
    failed = audit.source_audit(changed, path)
    assert not failed["passed"]
    assert "target_source_write_rms_mismatch" in failed["errors"]
    assert changed["sham_admission"]["projectors"] == record["sham_admission"]["projectors"]


def test_missing_optimum_and_byte_corruption_are_structured_failures(tmp_path):
    path = tmp_path / "partial.npz"
    record = {"source_rank12_optimum": None,
              "source_archive": audit.save_npz(path, {"sham_projectors": np.empty((0,4,20,20))})}
    checked = audit.source_audit(record, path)
    assert not checked["passed"] and not checked["complete"]
    assert checked["errors"] == ["source_rank12_optimum_missing"]
    path.write_bytes(path.read_bytes() + b"changed")
    checked = audit.source_audit(record, path)
    assert not checked["passed"]
    assert any("mismatch" in e for e in checked["errors"])


def test_task_numpy_replay_preserves_label_sign_and_detects_flips(tmp_path):
    labels = np.array([0, 1, 1, 0], dtype=np.uint8)
    logits = np.array([-1., 2., 3., -4.], dtype=np.float32)
    # Independent direct reference has unequal signed logits, so changing truth
    # signs cannot pass through a symmetric example by accident.
    nll = float(np.logaddexp(0., -2*np.where(labels, 1., -1.)*logits).mean())
    arms = {name: {"nll_nats": nll} for name in audit.MANDATORY_ARMS}
    arrays = {"labels": labels, "clean_logits": logits}
    arrays.update({f"arm__{name}": logits for name in arms})
    arrays.update({f"sham__{j}": logits for j in range(8)})
    path = tmp_path / "tasks.npz"
    record = {"clean_nll": nll, "task_nll_by_arm": arms,
              "sham_nlls": [nll]*8, "sham_task_metrics": [{"nll_nats": nll} for _ in range(8)],
              "sham_admission": {"selected": 8},
              "task_archive": audit.save_npz(path, arrays)}
    assert audit.task_audit(record, path)["passed"]
    arrays["labels"] = 1-labels
    record["task_archive"] = audit.save_npz(path, arrays)
    failed = audit.task_audit(record, path)
    assert not failed["passed"]
    assert "clean_nll_mismatch" in failed["errors"]
