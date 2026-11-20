"""Independent replay of Spectralucative's stored numerical evidence.

This module deliberately does not import Spectralucative.math or native. It
uses direct parity sums and independently formed matrix products. Successful
replay establishes consistency of recorded engineering data and its sources.
It does not establish a new causal effect or an interval-arithmetic proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

ATOL = 1e-9
RTOL = 2e-7
ZERO_COUNTERS = (
    "training_updates", "checkpoint_downloads", "gpu_runs", "scientific_roots",
    "confirmation_roots", "reserved_development_roots",
)
REQUIRED_LOCAL_SOURCES = {
    "Spectralucative/__init__.py", "Spectralucative/native.py", "Spectralucative/math.py",
    "Spectralucative/verify.py", "Spectralucative/PROTOCOL.json",
}
REQUIRED_BPD_SOURCES = {
    "mechinterp/__init__.py", "mechinterp/bpd/__init__.py",
    "mechinterp/bpd/adapters/__init__.py", "mechinterp/bpd/adapters/trm.py",
    "mechinterp/bpd/adapters/huginn.py", "mechinterp/bpd/adapters/looped_chase.py",
    "mechinterp/bpd/components.py", "mechinterp/bpd/config.py", "mechinterp/bpd/analysis.py",
    "mechinterp/bpd/adversary.py", "mechinterp/bpd/ci.py", "mechinterp/bpd/contrast.py",
    "mechinterp/bpd/losses.py", "mechinterp/bpd/nulls.py", "mechinterp/bpd/trainer.py",
    "mechinterp/loopscope02/scripts/trm_sudoku.py", "mechinterp/loopscope02/scripts/sudoku_prop.py",
    "mechinterp/loopscope01/scripts/loopscope_toy.py",
}
NATIVE_TOLERANCES = {
    "all_ones_absolute": 3e-5, "factor_weight_absolute": 2e-7,
    "delivered_write_absolute": 4e-6, "jvp_absolute": 2e-5, "jvp_relative": 3e-4,
    "central_difference_absolute_l2": 1e-3, "central_difference_relative_l2": 0.02,
}
HISTORICAL_HASHES = {
    "vendor/evalmath_preflight.py": "c2621a7900930f81ed53799ce439ee1c8f2d7006918a04ef57b5381630d15494",
    "evidence/historical_dt05_input.json": "d25121b4c5da0e6596ae0c54f54e61be57d22ff52eaa806b3744a615c8564af6",
    "evidence/historical_dt05_preflight.json": "d87303d07d934d21c69cbff27c7dc906e928cf3d9865d30a451f40407e9ba80c",
}


class VerificationError(ValueError):
    """The receipt fails its structural or independent numerical checks."""


def _require(condition, message):
    if not condition:
        raise VerificationError(message)


def _array(value, name, ndim=None):
    _require(not np.iscomplexobj(value), f"{name}: complex data")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise VerificationError(f"{name}: invalid numeric data") from error
    _require(ndim is None or result.ndim == ndim, f"{name}: wrong number of dimensions")
    _require(np.isfinite(result).all(), f"{name}: nonfinite data")
    return result


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name}: invalid integer")
    return value


def _scalar(value, name):
    _require(type(value) in (int, float) and math.isfinite(value), f"{name}: invalid scalar")
    return float(value)


def _near(actual, expected, name, *, atol=ATOL, rtol=RTOL):
    actual = _array(actual, name)
    expected = _array(expected, name + " reference")
    _require(actual.shape == expected.shape, f"{name}: shape mismatch")
    _require(np.allclose(actual, expected, rtol=rtol, atol=atol), f"{name}: numerical mismatch")


def _check_slack(report, scales, operations):
    expected = 128 * np.finfo(np.float64).eps * max(1, operations) * max(1.0, *map(abs, scales))
    _near(report.get("floating_point_slack"), expected, "floating_point_slack", atol=1e-20)
    return expected


def _load_json(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"Repeated JSON field: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise VerificationError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object,
                      parse_constant=invalid_constant)


def _engineering_scope(payload):
    _require(payload.get("evidence_class") == "engineering_implementation_only", "Wrong evidence class")
    _require(payload.get("model_initialization") == "random", "Wrong model initialization")
    for key in ZERO_COUNTERS:
        _require(type(payload.get(key)) is int and payload[key] == 0,
                 f"Disallowed evidence counter: {key}")
    for key in ("causal_readiness", "scientific_readiness", "new_study_readiness"):
        _require(key not in payload or payload[key] == "not_assessed", "Scientific readiness was asserted")


def _inf(matrix):
    return max((math.fsum(abs(float(x)) for x in row) for row in matrix), default=0.0)


def _product(matrices, initial_dimension):
    result = np.eye(initial_dimension)
    for matrix in matrices:
        result = np.matmul(matrix, result)
    return result


def payload_digest(payload):
    """Canonical digest of all fields except the digest field itself."""
    data = {key: value for key, value in payload.items() if key != "payload_sha256"}
    try:
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise VerificationError("Payload is not finite canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _check_digest(payload):
    _require(payload.get("payload_sha256") == payload_digest(payload), "Payload digest mismatch")


def _source_path(repo_root, relative):
    _require(isinstance(relative, str) and relative and "\\" not in relative,
             "Source path must use a relative POSIX path")
    path = Path(relative)
    _require(not path.is_absolute() and ".." not in path.parts, "Source path escapes repository")
    _require(path.suffix == ".py" or
             (relative.startswith("Spectralucative/") and path.name.startswith("PROTOCOL")
              and path.suffix in (".md", ".json")), "Unsupported source-file kind")
    resolved = (Path(repo_root) / path).resolve()
    _require(resolved.is_relative_to(Path(repo_root).resolve()), "Source symlink escapes repository")
    return resolved


def verify_sources(provenance, repo_root, *, required=()):
    _require(isinstance(provenance, dict), "Missing provenance")
    sources = provenance.get("source_files_sha256")
    _require(isinstance(sources, dict) and sources, "Missing source hashes")
    _require(set(required).issubset(sources), "Required source hashes are missing")
    for relative, expected in sources.items():
        _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                 f"{relative}: invalid SHA-256")
        path = _source_path(repo_root, relative)
        _require(path.is_file(), f"Missing source: {relative}")
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                 f"Source hash mismatch: {relative}")
    return len(sources)


def verify_trace(matrix, report):
    matrix = _array(matrix, "spectral matrix", 2)
    _require(matrix.shape[0] == matrix.shape[1], "Spectral matrix must be square")
    size = matrix.shape[0]
    z = _scalar(report.get("z"), "z")
    terms = _integer(report.get("terms"), "terms")
    _require(terms <= 4096, "Trace expansion exceeds replay work limit")
    norm = _inf(matrix)
    q = abs(z) * norm
    _require(math.isfinite(q), "Scaled matrix norm is nonfinite")
    _require(report.get("dimension") == size, "Spectral dimension mismatch")
    _near(report.get("matrix_inf_norm"), norm, "matrix_inf_norm", atol=0, rtol=1e-12)
    _near(report.get("q"), q, "q", atol=0, rtol=1e-12)
    inside = q < 1.0
    _require(report.get("convergence_condition") is inside, "Wrong convergence condition")
    _require(report.get("status") == ("within_norm_domain" if inside else "outside_norm_domain"),
             "Wrong trace status")
    _require(report.get("roundoff_certified") is False, "Unsupported roundoff certificate")
    _require(report.get("evidence_class") == "floating_point_diagnostic_with_analytic_truncation_bound",
             "Wrong trace evidence class")
    sign, logabs = np.linalg.slogdet(np.eye(size) - z * matrix)
    _near(report.get("logdet_sign"), float(sign), "logdet_sign")
    logdet = -float(logabs) if sign > 0 and math.isfinite(logabs) else None
    if logdet is None:
        _require(report.get("negative_logdet") is None, "Invalid log determinant claim")
    else:
        _near(report.get("negative_logdet"), logdet, "negative_logdet")
    if not inside:
        for key in ("trace_log_partial", "analytic_tail_bound", "absolute_log_error",
                    "floating_point_slack", "floating_point_check_passed"):
            _require(report.get(key) is None, f"{key}: certificate outside norm domain")
        return
    _require(logdet is not None, "Positive determinant required within norm domain")
    partial = math.fsum(float(np.trace(np.linalg.matrix_power(z * matrix, k))) / k
                        for k in range(1, terms + 1))
    tail = size * q ** (terms + 1) / ((terms + 1) * (1 - q))
    error = abs(partial - logdet)
    _near(report.get("trace_log_partial"), partial, "trace_log_partial")
    _near(report.get("analytic_tail_bound"), tail, "analytic_tail_bound", atol=0, rtol=1e-12)
    _near(report.get("absolute_log_error"), error, "absolute_log_error")
    slack = _check_slack(report, (partial, logdet, tail), max(1, size, terms))
    _require(error <= tail + slack,
             "Trace series violates its tail bound")
    _require(_scalar(report.get("absolute_log_error"), "Reported log error") <=
             _scalar(report.get("analytic_tail_bound"), "Reported tail bound") +
             _scalar(report.get("floating_point_slack"), "Reported trace slack"),
             "Reported trace values fail the declared criterion")
    _require(report.get("floating_point_check_passed") is True, "Trace check did not pass")


def verify_transport(A_steps, B_steps, L_steps, report):
    A = [_array(x, "A map", 2) for x in A_steps]
    B = [_array(x, "B map", 2) for x in B_steps]
    L = [_array(x, "L map", 2) for x in L_steps]
    steps = len(A)
    _require(len(B) == steps and len(L) == steps + 1, "Invalid transport horizon")
    for t in range(steps):
        _require(A[t].shape == (L[t + 1].shape[1], L[t].shape[1]), "A shape mismatch")
        _require(B[t].shape == (L[t + 1].shape[0], L[t].shape[0]), "B shape mismatch")
    _require(report.get("n_steps") == steps, "Transport step count mismatch")
    _require(report.get("source_dimensions") == [x.shape[1] for x in L], "Source dimensions mismatch")
    _require(report.get("target_dimensions") == [x.shape[0] for x in L], "Target dimensions mismatch")
    defect = [B[t] @ L[t] - L[t + 1] @ A[t] for t in range(steps)]
    Ap = [_product(A[:t], L[0].shape[1]) for t in range(steps + 1)]
    Bp = [_product(B[:t], L[0].shape[0]) for t in range(steps + 1)]
    differences = [Bp[t] @ L[0] - L[t] @ Ap[t] for t in range(steps + 1)]
    errors = [_inf(x) for x in differences]
    bounds = []
    for depth in range(steps + 1):
        bounds.append(math.fsum(math.prod(_inf(B[k]) for k in range(j + 1, depth)) *
                                _inf(defect[j]) * _inf(Ap[j]) for j in range(depth)))
    terms = [_product(B[t + 1:], L[t + 1].shape[0]) @ defect[t] @ Ap[t]
             for t in range(steps)]
    expanded = sum(terms, np.zeros_like(differences[-1]))
    for key, expected in {
        "local_defect_norms": [_inf(x) for x in defect],
        "source_prefix_norms": [_inf(x) for x in Ap],
        "target_operator_norms": [_inf(x) for x in B],
        "product_error_norms": errors,
        "norm_envelope": bounds,
        "final_error_norm": errors[-1], "final_bound": bounds[-1],
        "final_difference": differences[-1],
        "telescoping_reconstruction": expanded,
        "reconstruction_residual_norm": _inf(expanded - differences[-1]),
    }.items():
        _near(report.get(key), expected, key)
    _near(expanded, differences[-1], "Independent telescoping identity")
    slack = _check_slack(report, errors + bounds, max(1, steps, *L[0].shape))
    _require(_inf(expanded - differences[-1]) <= slack, "Telescoping residual exceeds numerical slack")
    _require(all(e <= b + slack for e, b in zip(errors, bounds)),
             "Transport error exceeds its envelope")
    _require(report["reconstruction_residual_norm"] <= report["floating_point_slack"] and
             all(e <= b + report["floating_point_slack"] for e, b in
                 zip(report["product_error_norms"], report["norm_envelope"])),
             "Reported transport values fail the declared criterion")
    _require(report.get("floating_point_check_passed") is True, "Transport check did not pass")
    _require(report.get("roundoff_certified") is False, "Unsupported transport roundoff certificate")


def verify_walsh(mask_ids, values, report):
    values = _array(values, "Walsh response cube")
    _require(values.ndim in (1, 2) and len(values) > 0, "Invalid response cube")
    count = len(values)
    _require(count & (count - 1) == 0, "Incomplete Boolean cube")
    _require(len(mask_ids) == count and all(type(x) is int for x in mask_ids), "Invalid mask IDs")
    _require(sorted(mask_ids) == list(range(count)), "Missing or repeated mask IDs")
    bits = count.bit_length() - 1
    degree = _integer(report.get("retained_degree"), "retained_degree")
    _require(degree <= bits, "Retained degree exceeds cube dimension")
    ordered = values[np.argsort(mask_ids)]
    scalar = values.ndim == 1
    if scalar:
        ordered = ordered[:, None]
    H = np.array([[(-1.0 if (s & x).bit_count() % 2 else 1.0)
                   for x in range(count)] for s in range(count)])
    coefficients = H @ ordered / count
    energy = [math.fsum(float(v) ** 2 for v in row) for row in coefficients]
    by_degree = [math.fsum(energy[s] for s in range(count) if s.bit_count() == d)
                 for d in range(bits + 1)]
    kept = coefficients.copy()
    for s in range(count):
        if s.bit_count() > degree:
            kept[s] = 0
    reconstructed = H @ kept
    residual = ordered - reconstructed
    response_energy = math.fsum(float(x) ** 2 for x in ordered.flat) / count
    residual_energy = math.fsum(float(x) ** 2 for x in residual.flat) / count
    variance = float(np.mean(np.sum((ordered - np.mean(ordered, axis=0)) ** 2, axis=1)))
    high = math.fsum(energy[s] for s in range(count) if s.bit_count() >= 2)
    discarded = math.fsum(energy[s] for s in range(count) if s.bit_count() > degree)
    _require(report.get("measure") == "uniform_complete_cube", "Wrong Walsh measure")
    _require(report.get("n_masks") == count and report.get("n_bits") == bits, "Walsh cube dimensions mismatch")
    _require(report.get("output_dimension") == ordered.shape[1], "Walsh output dimension mismatch")
    _require(report.get("mask_ids") == list(range(count)), "Wrong coefficient order")
    expected = {
        "coefficients": coefficients[:, 0] if scalar else coefficients,
        "energy_by_degree": by_degree,
        "mean_squared_l2_response": response_energy,
        "spectral_energy": math.fsum(energy),
        "parseval_absolute_error": abs(response_energy - math.fsum(energy)),
        "variance": variance, "high_degree_energy": high,
        "high_degree_fraction_of_variance": high / variance if variance > 0 else 0.0,
        "truncated_response": reconstructed[:, 0] if scalar else reconstructed,
        "residual_mean_squared_l2_response": residual_energy,
        "discarded_coefficient_energy": discarded,
        "residual_identity_absolute_error": abs(residual_energy - discarded),
    }
    for key, value in expected.items():
        _near(report.get(key), value, key)
    _near(response_energy, math.fsum(energy), "Independent Parseval identity")
    _near(residual_energy, discarded, "Independent Walsh residual identity")
    slack = _check_slack(report, (response_energy, math.fsum(energy), residual_energy),
                         max(count, ordered.shape[1]))
    _require(abs(response_energy - math.fsum(energy)) <= slack and
             abs(residual_energy - discarded) <= slack, "Walsh identities exceed numerical slack")
    _require(report["parseval_absolute_error"] <= report["floating_point_slack"] and
             report["residual_identity_absolute_error"] <= report["floating_point_slack"],
             "Reported Walsh values fail the declared criterion")
    _require(report.get("floating_point_check_passed") is True, "Walsh check did not pass")
    _require(report.get("roundoff_certified") is False, "Unsupported Walsh roundoff certificate")


def verify_historical(folder):
    folder = Path(folder)
    provenance = _load_json(folder / "evidence/provenance.json")
    _require(provenance.get("schema") == "Spectralucative/historical-gate-provenance/v1", "Historical provenance schema")
    _require(provenance.get("execution", {}).get("exit_status") == 2, "Historical exit status changed")
    _require(provenance.get("execution", {}).get("expected_exit_status") == 2,
             "Historical expected exit status changed")
    _require(provenance.get("evidence_class") == "historical_development_gate_reproduction",
             "Historical evidence class changed")
    interpretation = provenance.get("interpretation", {})
    _require(interpretation.get("new_study_readiness") == "not_assessed" and
             interpretation.get("historical_verdict_changed") is False,
             "Historical evidence was promoted")
    populations = provenance.get("populations_consumed", {})
    _require(all(type(populations.get(key)) is int and populations[key] == 0 for key in
                 ("confirmation_roots", "new_model_roots", "reserved_dt05_development_roots")),
             "Historical reproduction consumed a population")
    _require(provenance.get("checkpoint_downloads") == 0 and provenance.get("gpu_runs") == 0,
             "Historical reproduction used model resources")
    for relative, expected in HISTORICAL_HASHES.items():
        actual = hashlib.sha256((folder / relative).read_bytes()).hexdigest()
        _require(actual == expected, f"Historical file changed: {relative}")
        _require(provenance.get("artifacts", {}).get(relative, {}).get("sha256") == expected,
                 f"Historical provenance hash changed: {relative}")
    sources = provenance.get("sources", {})
    _require(sources.get("bpd", {}).get("commit") == "420631c8835475c8b7a1f5f55f01a40ed4fb543e", "Historical BPD pin changed")
    _require(sources.get("evalmath", {}).get("commit") == "9708f1edaf1cb25904b88b07c3f23c1899df7cac", "Historical EvalMath pin changed")
    expected_result = _load_json(folder / "evidence/historical_dt05_preflight.json")
    _require(expected_result.get("decision") == "REDESIGN_OR_PILOT_ONLY", "Historical verdict changed")
    with tempfile.TemporaryDirectory(prefix="spectralucative-gate-") as temporary:
        output = Path(temporary) / "preflight.json"
        command = [sys.executable, str(folder / "vendor/evalmath_preflight.py"),
                   "--pilot", str(folder / "evidence/historical_dt05_input.json"),
                   "--out", str(output), "--strict"]
        replay = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False, timeout=30)
        _require(replay.returncode == 2, f"Historical strict replay returned {replay.returncode}")
        _require(output.is_file(), "Historical replay produced no output")
        _require(_load_json(output) == expected_result, "Historical replay changed its result")
    return {"strict_exit_status": 2, "decision": expected_result["decision"], "exact_reproduction": True}


def verify_factor(factor):
    V = _array(factor.get("V"), "factor V", 2)
    U = _array(factor.get("U"), "factor U", 2)
    target = _array(factor.get("target"), "factor target", 2)
    _require(U.shape[0] == V.shape[1] and target.shape == (U.shape[1], V.shape[0]),
             "Rectangular factor dimensions mismatch")
    mask = _array(factor.get("mask"), "factor mask", 1)
    _require(mask.shape == (V.shape[1],) and np.array_equal(mask, np.ones_like(mask)),
             "Factor parity requires the all-ones mask")
    reconstructed = U.T @ V.T
    residual = _array(factor.get("residual"), "factor residual", 2)
    _near(factor.get("reconstruction"), reconstructed, "Factor reconstruction", atol=2e-7)
    _near(residual, target - reconstructed, "Factor residual", atol=2e-7)
    error = float(np.max(np.abs(reconstructed + residual - target), initial=0.0))
    _require(factor.get("all_ones_tolerance") == 2e-7, "Factor parity tolerance changed")
    _near(factor.get("all_ones_max_abs_weight_error"), error, "Factor parity error", atol=2e-7)
    _require(error <= 2e-7, "All-ones factor fails target parity")
    _require(factor.get("residual_retained") is True and factor.get("module") == "layers.0.down_proj",
             "Native factor configuration changed")
    return V.shape[1], U.shape[1]


def verify_cube(cube, walsh, factor, *, component_count, hidden_size, horizon, state_dimension):
    count = 1 << component_count
    _require(cube.get("component_count") == component_count, "Mask/factor component mismatch")
    _require(cube.get("mask_ids") == list(range(count)), "Mask cube is incomplete or reordered")
    _require(walsh.get("mask_ids") == cube["mask_ids"], "Walsh/cube mask binding mismatch")
    bits = [[(mask >> bit) & 1 for bit in range(component_count)] for mask in range(count)]
    _near(cube.get("bits"), bits, "Mask bit order", atol=0, rtol=0)
    logits = _array(cube.get("full_head_logits"), "Mask full-head logits", 3)
    target = _array(cube.get("target_full_head_logits"), "Target full-head logits", 2)
    _require(logits.shape == (count, *target.shape) and target.shape[1] == 11,
             "Full-head logit dimensions mismatch")
    delta = (logits - target[None, :, :]).reshape(count, -1)
    _near(cube.get("full_head_logit_deltas"), delta, "Full-head response deltas")
    _require(cube.get("response_definition") == "flattened_full_11_head_logit_delta_from_target" and
             cube.get("scalar_response_definition") == "mean_squared_full_head_logit_drift_from_target",
             "Mask response definitions changed")
    _require(walsh.get("values_source") == "mask_cube.full_head_logit_deltas",
             "Wrong primary Walsh response")
    _near(walsh.get("values"), delta, "Walsh/raw logit binding")
    _near(cube.get("scalar_responses"), np.mean(delta ** 2, axis=1), "Descriptive mean-square response")
    writes = _array(cube.get("physical_write_displacements"), "Physical writes", 3)
    _require(writes.shape[0] == count and writes.shape[2] == hidden_size and
             2 * writes.shape[1] * hidden_size == state_dimension, "Physical write dimensions mismatch")
    _near(cube.get("physical_write_displacement_l2"), np.linalg.norm(writes.reshape(count, -1), axis=1),
          "Delivered write norms")
    position = _integer(cube.get("patch_position"), "Patch position")
    _require(position < writes.shape[1], "Patch position exceeds state length")
    inputs = _array(cube.get("first_wrapper_input"), "Wrapper input", 2)
    V, U = np.asarray(factor["V"]), np.asarray(factor["U"])
    _require(inputs.shape == (writes.shape[1], V.shape[0]), "Wrapper input dimensions mismatch")
    expected_writes = np.zeros_like(writes)
    for mask in range(count):
        expected_writes[mask, position] = ((inputs[position] @ V) * (np.asarray(bits[mask]) - 1)) @ U
    _near(cube.get("expected_physical_write_displacements"), expected_writes, "Component write formula")
    delivered_errors = np.max(np.abs(writes - expected_writes), axis=(1, 2))
    _near(cube.get("delivered_write_max_abs_errors"), delivered_errors, "Delivered write errors")
    _require(np.all(delivered_errors <= NATIVE_TOLERANCES["delivered_write_absolute"]),
             "Delivered component writes differ from their formula")
    observed = {}
    for prefix, target_name, states_name in (
        ("first_step", "target_first_step_state", "first_step_states"),
        ("final", "target_final_state", "final_states"),
    ):
        target_state = _array(cube.get(target_name), target_name, 1)
        states = _array(cube.get(states_name), states_name, 2)
        _require(target_state.shape == (state_dimension,) and states.shape == (count, state_dimension),
                 "Mask state dimensions mismatch")
        displacement = states - target_state
        observed[prefix] = displacement
        _near(cube.get(prefix + "_state_displacement_l2"), np.linalg.norm(displacement, axis=1),
              prefix + " state dose")
    _require(cube.get("all_ones_mask_id") == count - 1, "Wrong all-ones mask ID")
    error = float(np.max(np.abs(delta[-1]), initial=0.0))
    _near(cube.get("all_ones_max_abs_logit_error"), error, "All-ones logit parity")
    _require(cube.get("all_ones_tolerance") == 3e-5, "Logit parity tolerance changed")
    _require(error <= 3e-5, "All-ones mask fails native target parity")
    state_error = float(np.max(np.abs(observed["final"][-1]), initial=0))
    _near(cube.get("all_ones_max_abs_state_error"), state_error, "All-ones state parity")
    _require(state_error <= 3e-5, "All-ones mask fails full-state target parity")
    _require(cube.get("residual_retained") is True, "Residual weight was removed")
    _require(cube.get("sphere_rescale") is False and cube.get("ablation_mode") == "zero",
             "Mask intervention transformation changed")
    _require(cube.get("scope") == "descriptive_varying_dose_interaction_cube", "Cube causal scope changed")
    _require(cube.get("patch_invocation") == 0 and cube.get("horizon_outer_steps") == horizon,
             "Cube intervention schedule changed")


def verify_native_pushes(pushes, B_steps, A_steps, Q):
    B = [_array(x, "Native Jacobian", 2) for x in B_steps]
    A = [_array(x, "Reduced Jacobian", 2) for x in A_steps]
    expected = [J @ Q for J in B]
    leakage = [pushed - Q @ reduced for pushed, reduced in zip(expected, A)]
    _near(pushes.get("JQ"), expected, "Stored native pushes")
    _near(pushes.get("leakage"), leakage, "Stored native leakage")
    _near(pushes.get("leakage_frobenius"), [np.linalg.norm(x) for x in leakage], "Native leakage norms")
    checks = pushes.get("derivative_checks")
    _require(isinstance(checks, list) and len(checks) == len(B), "Native derivative checks missing")
    native_Q = Q.astype(np.float32).astype(np.float64)
    for index, check in enumerate(checks):
        _require(check.get("step") == index, "Native derivative step order changed")
        _near(check.get("native_basis"), native_Q, "Native float32 basis", atol=0, rtol=0)
        direct = _array(check.get("direct_jvp"), "Direct JVP", 2)
        target = B[index] @ native_Q
        _require(direct.shape == target.shape, "Direct JVP shape mismatch")
        _near(check.get("direct_jvp_max_abs_error"), np.max(np.abs(direct - target)), "Direct JVP error")
        _require(check.get("direct_jvp_passed") is True and
                 np.allclose(direct, target, atol=NATIVE_TOLERANCES["jvp_absolute"],
                             rtol=NATIVE_TOLERANCES["jvp_relative"]), "Direct JVP fails its declared tolerance")
        difference = _array(check.get("central_difference"), "Central difference", 2)
        _require(difference.shape == target.shape, "Central difference shape mismatch")
        _require(check.get("central_difference_epsilon") == 1e-3, "Central difference dose changed")
        errors = np.linalg.norm(difference - target, axis=0)
        limits = NATIVE_TOLERANCES["central_difference_absolute_l2"] + (
            NATIVE_TOLERANCES["central_difference_relative_l2"] * np.linalg.norm(target, axis=0))
        _near(check.get("central_difference_error_l2"), errors, "Central difference errors")
        _near(check.get("central_difference_limits_l2"), limits, "Central difference tolerances")
        _require(check.get("central_difference_passed") is True and np.all(errors <= limits),
                 "Central difference fails its declared tolerance")


def verify_frozen(comparison, A_steps, B_steps, Q):
    A = [_array(x, "Reduced frozen comparison", 2) for x in A_steps]
    B = [_array(x, "Full frozen comparison", 2) for x in B_steps]
    Q = _array(Q, "Frozen comparison inclusion", 2)
    _require(comparison.get("depths") == list(range(1, len(A) + 1)), "Frozen comparison depths mismatch")
    ordered_A = [_product(A[:t], Q.shape[1]) for t in range(1, len(A) + 1)]
    frozen_A = [np.linalg.matrix_power(A[0], t) for t in range(1, len(A) + 1)]
    ordered_B = [_product(B[:t], Q.shape[0]) @ Q for t in range(1, len(B) + 1)]
    frozen_B = [np.linalg.matrix_power(B[0], t) @ Q for t in range(1, len(B) + 1)]
    for key, expected in {
        "reduced_ordered_products": ordered_A, "reduced_frozen_powers": frozen_A,
        "reduced_gap_inf": [_inf(a - b) for a, b in zip(ordered_A, frozen_A)],
        "full_ordered_pushes": ordered_B, "full_frozen_pushes": frozen_B,
        "full_gap_inf": [_inf(a - b) for a, b in zip(ordered_B, frozen_B)],
    }.items():
        _near(comparison.get(key), expected, key)


def verify_finite(finite, B_steps, Q):
    B = [_array(x, "Finite-displacement Jacobian", 2) for x in B_steps]
    Q = _array(Q, "Finite-displacement inclusion", 2)
    direction = _array(finite.get("direction"), "Finite-displacement direction", 1)
    _near(direction, Q[:, 0], "Finite-displacement direction binding")
    state_norm = _scalar(finite.get("state_norm_l2"), "Baseline state norm")
    _require(state_norm > 0, "Baseline state norm must be positive")
    doses = finite.get("relative_doses")
    _require(doses == [1e-4, 1e-3, 1e-2], "Finite-displacement doses changed")
    records = finite.get("records")
    _require(isinstance(records, list) and len(records) == len(doses), "Finite-displacement records missing")
    baseline = _array(finite.get("baseline_states"), "Baseline states", 2)
    _require(baseline.shape == (len(B) + 1, Q.shape[0]), "Baseline state dimensions mismatch")
    _near(state_norm, np.linalg.norm(baseline[0]), "Baseline state norm")
    for dose, record in zip(doses, records):
        _require(record.get("relative_dose") == dose, "Finite-displacement dose order changed")
        initial = _array(record.get("actual_initial_delta"), "Actual initial displacement", 1)
        _require(initial.shape == (Q.shape[0],), "Initial displacement dimensions mismatch")
        _near(record.get("requested_initial_delta"), dose * state_norm * direction,
              "Requested displacement")
        perturbed = _array(record.get("perturbed_states"), "Perturbed states", 2)
        _require(perturbed.shape == baseline.shape, "Perturbed state dimensions mismatch")
        actual = _array(record.get("actual_deltas"), "Actual state displacements", 2)
        _near(actual, perturbed - baseline, "Observed displacement binding")
        _near(initial, actual[0], "Initial displacement binding")
        linear = np.stack([_product(B[:t], Q.shape[0]) @ initial for t in range(len(B) + 1)])
        _near(record.get("linearized_deltas"), linear, "Ordered linearized displacements")
        remainder = np.linalg.norm(actual - linear, axis=1)
        _near(record.get("remainder_l2"), remainder, "Nonlinear remainder")
        _near(record.get("relative_remainder"), remainder / np.maximum(1e-15, np.linalg.norm(actual, axis=1)),
              "Relative nonlinear remainder")


def verify_unit(payload, repo_root):
    _require(isinstance(payload, dict), "Receipt must be an object")
    _check_digest(payload)
    _require(payload.get("schema") == "BPD/Spectralucative/native-unit/v1", "Wrong unit schema")
    _engineering_scope(payload)
    name = payload.get("engineering_initialization_id")
    _require(isinstance(name, str) and name.lower().startswith("spectralucative"), "Wrong initialization namespace")
    _integer(payload.get("seed"), "seed")
    provenance = payload.get("provenance", {})
    sources = verify_sources(provenance, repo_root, required=REQUIRED_LOCAL_SOURCES | REQUIRED_BPD_SOURCES)
    _require(provenance.get("repository") == "MoralityLabAI/BPD" and
             provenance.get("source_base_commit") == "420631c8835475c8b7a1f5f55f01a40ed4fb543e",
             "Native BPD source pin changed")
    _require(isinstance(payload.get("model_state_sha256"), str) and
             re.fullmatch(r"[0-9a-f]{64}", payload["model_state_sha256"]), "Missing model state digest")
    runtime = payload.get("runtime", {})
    _require(runtime.get("device") == "cpu" and runtime.get("native_dtype") == "float32" and
             runtime.get("analysis_dtype") == "float64" and runtime.get("attention_backend") == "math" and
             runtime.get("deterministic_algorithms") is True and runtime.get("roundoff_certified") is False,
             "Native runtime scope changed")
    _require(payload.get("numerical_tolerances") == NATIVE_TOLERANCES, "Native comparison tolerances changed")
    fixture = payload.get("engineering_fixture", {})
    _require(fixture.get("token_ids") == [[1, 2, 4, 7]] and fixture.get("task_performance_endpoint") is None and
             fixture.get("warmup_outer_steps") == 1, "Engineering fixture scope changed")
    spectral = payload.get("spectral")
    _require(isinstance(spectral, list) and spectral, "Missing spectral diagnostics")
    for index, step in enumerate(spectral):
        _require(step.get("step") == index, "Spectral step order mismatch")
        verify_trace(step.get("matrix"), step.get("raw", {}))
        verify_trace(step.get("matrix"), step.get("damped", {}))
        _require(step["raw"]["z"] == 1, "Raw spectral scale changed")
        _near(step["damped"]["z"], 0.5 / max(1, _inf(np.asarray(step["matrix"]))), "Damped spectral scale")
    transport = payload.get("transport", {})
    verify_transport(transport.get("A_steps", []), transport.get("B_steps", []),
                     transport.get("L_steps", []), transport.get("report", {}))
    walsh = payload.get("walsh", {})
    verify_walsh(walsh.get("mask_ids", []), walsh.get("values", []), walsh.get("report", {}))
    _require(len(spectral) == transport["report"]["n_steps"], "Spectral and transport horizons differ")
    for index, step in enumerate(spectral):
        _near(step["matrix"], transport["A_steps"][index], "Spectral/transport matrix binding")
    full = payload.get("full_state_spectral")
    _require(isinstance(full, list) and len(full) == len(spectral), "Missing full-state spectra")
    Q = _array(transport["L_steps"][0], "Reference inclusion", 2)
    _require(Q.shape[1] > 0, "Empty reference subspace")
    _near(Q.T @ Q, np.eye(Q.shape[1]), "Orthonormal inclusion")
    for inclusion in transport["L_steps"]:
        _near(inclusion, Q, "Fixed inclusion binding")
    for index, step in enumerate(full):
        _require(step.get("step") == index and step.get("matrix_source") == f"transport.B_steps[{index}]",
                 "Full-state matrix binding changed")
        J = _array(transport["B_steps"][index], "Full-state Jacobian", 2)
        _near(transport["A_steps"][index], Q.T @ J @ Q, "Projected native Jacobian")
        verify_trace(J, step.get("raw", {}))
        verify_trace(J, step.get("damped", {}))
        _require(step["raw"]["z"] == 1, "Raw full-state spectral scale changed")
        _near(step["damped"]["z"], 0.5 / max(1, _inf(J)), "Damped full-state scale")
    component_count, hidden_size = verify_factor(payload.get("factor", {}))
    verify_cube(payload.get("mask_cube", {}), walsh, payload["factor"], component_count=component_count,
                hidden_size=hidden_size, horizon=len(spectral), state_dimension=Q.shape[0])
    verify_native_pushes(payload.get("native_pushes", {}), transport["B_steps"], transport["A_steps"], Q)
    verify_frozen(payload.get("frozen_operator_comparison", {}), transport["A_steps"],
                  transport["B_steps"], Q)
    verify_finite(payload.get("finite_displacement", {}), transport["B_steps"], Q)
    state = payload.get("state", {})
    _require(state.get("dimension") == Q.shape[0], "Native state dimension changed")
    _near(state.get("trajectory"), payload["finite_displacement"]["baseline_states"], "Baseline trajectory binding")
    _near(payload["mask_cube"]["target_first_step_state"], state["trajectory"][1], "First mask target binding")
    _near(payload["mask_cube"]["target_final_state"], state["trajectory"][-1], "Final mask target binding")
    heads = _array(state.get("full_head_trajectory"), "Full-head trajectory", 3)
    _require(heads.shape[0] == len(spectral) + 1, "Head trajectory horizon changed")
    _near(payload["mask_cube"]["target_full_head_logits"], heads[-1], "Full-head target binding")
    basis = payload.get("basis", {})
    _near(basis.get("matrix"), Q, "Declared basis binding")
    _require(basis.get("rank") == Q.shape[1] and basis.get("component_ids") == list(range(Q.shape[1])),
             "Basis component selection changed")
    _near(basis.get("orthonormality_max_abs_error"), np.max(np.abs(Q.T @ Q - np.eye(Q.shape[1]))),
          "Basis orthonormality error")
    position = payload["mask_cube"]["patch_position"]
    _require(basis.get("patch_position") == position, "Basis/patch position differs")
    start = Q.shape[0] // 2 + position * hidden_size
    local_Q = Q[start:start + hidden_size]
    outside = Q.copy()
    outside[start:start + hidden_size] = 0
    _near(outside, np.zeros_like(outside), "Basis embedding support", atol=0, rtol=0)
    local_writes = np.asarray(payload["factor"]["U"])[:Q.shape[1]].T
    _near(local_Q @ local_Q.T @ local_writes, local_writes, "Basis component-write span")
    checks = payload.get("checks", {})
    expected_checks = {"all_ones_full_head_parity", "all_ones_full_state_parity", "residual_retained_weight_parity",
                       "delivered_write_formula", "autodiff_jvp_parity", "central_difference_jvp_sanity",
                       "transport_identity", "walsh_identities", "spectral_domain_checks", "model_parameters_unchanged"}
    _require(set(checks) == expected_checks and all(value is True for value in checks.values()) and
             payload.get("all_checks_passed") is True, "Native checks include a failure or missing check")
    return {"engineering_initialization_id": name, "source_files_checked": sources,
            "spectral_steps_checked": len(spectral), "transport_checked": True,
            "complete_walsh_cube_checked": True, "full_state_spectra_checked": True,
            "mask_response_binding_checked": True, "factor_parity_checked": True,
            "native_jvp_arithmetic_checked": True, "delivered_component_writes_checked": True,
            "finite_displacement_arithmetic_checked": True, "frozen_comparison_checked": True,
            "evidence_class": payload["evidence_class"]}


def verify_run(run, repo_root):
    run = Path(run).resolve()
    summary = _load_json(run / "summary.json")
    _check_digest(summary)
    _require(summary.get("schema") == "BPD/Spectralucative/native-run/v1", "Wrong run schema")
    _engineering_scope(summary)
    verify_sources(summary.get("provenance", {}), repo_root, required=REQUIRED_LOCAL_SOURCES | REQUIRED_BPD_SOURCES)
    protocol = _load_json(Path(repo_root) / "Spectralucative/PROTOCOL.json")
    _require(protocol.get("schema") == "Spectralucative/protocol/v1" and
             protocol.get("evidence_class") == "engineering_implementation_only", "Wrong protocol scope")
    entries = summary.get("units")
    _require(isinstance(entries, list) and entries, "Run has no units")
    _require(summary.get("status") == "complete" and summary.get("all_checks_passed") is True and
             summary.get("errors") == [] and summary.get("completed_engineering_roots") == len(entries),
             "Run contains an incomplete or failed initialization")
    _require(summary.get("engineering_root_count") == len(entries) == protocol.get("engineering_initializations"),
             "Engineering initialization count differs from protocol")
    _require([entry.get("seed") for entry in entries] == protocol.get("seeds"), "Initialization seeds differ from protocol")
    paths, names, checked = set(), set(), []
    raw_ineligible = 0
    for entry in entries:
        seed = _integer(entry.get("seed"), "Initialization seed")
        filename = entry.get("file")
        _require(filename == f"spectralucative_cpu_init_{seed}.json", "Unexpected receipt filename")
        _require(filename not in paths, "Repeated unit file")
        paths.add(filename)
        path = (run / filename).resolve()
        _require(path.parent == run, "Unit file escapes run directory")
        unit = _load_json(path)
        _require(entry.get("payload_sha256") == unit.get("payload_sha256"), "Summary/unit digest mismatch")
        _require(entry.get("all_checks_passed") is True, "Summary lists a failed initialization")
        name = unit.get("engineering_initialization_id")
        _require(name == entry.get("engineering_initialization_id") and name not in names,
                 "Repeated or mismatched initialization ID")
        names.add(name)
        _require(unit.get("seed") == seed, "Summary/unit seed mismatch")
        _require(unit.get("provenance", {}).get("source_files_sha256") == summary["provenance"]["source_files_sha256"],
                 "Source manifest differs between run and unit")
        checked.append(verify_unit(unit, repo_root))
        design = protocol["design"]
        _require(len(unit["spectral"]) == design["outer_transitions"], "Transition count differs from protocol")
        Q = np.asarray(unit["transport"]["L_steps"][0])
        model = protocol["model"]
        expected_state = 2 * model["hidden_size"] * (model["seq_len"] + model["puzzle_emb_len"])
        _require(Q.shape == (expected_state, design["subspace_rank"]), "State/subspace dimensions differ from protocol")
        _require(unit["mask_cube"]["component_count"] == design["mask_bits"] and
                 unit["mask_cube"]["patch_position"] == model["puzzle_emb_len"], "Mask design differs from protocol")
        recipe = unit.get("model_recipe", {})
        for key in ("hidden_size", "num_heads", "l_layers", "h_cycles", "l_cycles", "puzzle_emb_len", "seq_len"):
            _require(recipe.get(key) == model[key], f"Model recipe differs from protocol: {key}")
        _require(recipe.get("seed") == seed, "Model recipe seed differs from initialization")
        _require(unit["runtime"].get("cpu_threads") == protocol["execution"]["cpu_threads"],
                 "Native CPU threads differ from protocol")
        raw_ineligible += sum(not step["raw"]["convergence_condition"] for step in unit["full_state_spectral"])
    present = {path.name for path in run.glob("spectralucative_cpu_init_*.json")}
    _require(present == paths, "Unindexed engineering unit files")
    _require(summary.get("raw_z1_ineligible_native_steps") == raw_ineligible, "Raw spectral eligibility count changed")
    configuration = summary.get("configuration", {})
    for key, expected in {"roots": len(entries), "steps": protocol["design"]["outer_transitions"],
                          "rank": protocol["design"]["subspace_rank"], "components": protocol["design"]["mask_bits"],
                          "cpu_threads": protocol["execution"]["cpu_threads"]}.items():
        _require(configuration.get(key) == expected, f"Run configuration differs from protocol: {key}")
    return {"engineering_initializations_checked": len(checked), "scientific_roots": 0,
            "confirmation_roots": 0, "units": checked, "evidence_class": "engineering_implementation_only",
            "scope": "Independent arithmetic replay of stored observations; native model execution is recorded separately."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=Path, help="Directory containing summary.json and unit receipts")
    mode.add_argument("--historical-only", action="store_true", help="Replay the preserved strict DT05 gate")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, help="Write the independent verification receipt")
    arguments = parser.parse_args(argv)
    result = {"schema": "BPD/Spectralucative/verification/v1", "passed": False,
              "verification_tolerances": {"absolute": ATOL, "relative": RTOL,
                                          "float32_factor_absolute": 2e-7},
              "roundoff_certified": False}
    try:
        result["historical_gate"] = verify_historical(arguments.repo_root / "Spectralucative")
        if arguments.run is not None:
            result["native_run"] = verify_run(arguments.run, arguments.repo_root)
        result["passed"] = True
    except (VerificationError, ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError,
            np.linalg.LinAlgError, OverflowError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["payload_sha256"] = payload_digest(result)
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(encoded)
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
