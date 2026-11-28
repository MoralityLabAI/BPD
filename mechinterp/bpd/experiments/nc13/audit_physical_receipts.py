#!/usr/bin/env python3
"""Post-lock NC13 physical receipt audit, using NumPy and the standard library.

This report-stage script sits outside the execution lock. It reads archived
vectors and receipts without importing the experiment, loading models, running
rollouts, selecting new gains, or changing scientific admissions.

Usage: python audit_physical_receipts.py --receipts DIR --out FILE
Exit 1 indicates an audit discrepancy or missing receipt. Registered calibration
failures are valid outcomes and do not fail this consistency audit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

RTOL, ATOL = 5e-11, 5e-12
PROJECTOR_TOL, NATIVE_REL_TOL = 1e-8, 1e-4
D_TOL, E_SHAM_TOL = 1e-4, .15
GRID = np.linspace(.5, 1.5, 17)
CONTEXTS = ("shared", "specific")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def finite(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def rms(value, axis=None):
    value = np.asarray(value, dtype=np.float64)
    return np.sqrt(np.mean(value * value, axis=axis))


def safe_json(value):
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class Checks:
    def __init__(self):
        self.count, self.failures = 0, []
        self.max_error = {"absolute": 0., "path": None}

    def require(self, condition, path, observed=None, expected=None):
        self.count += 1
        if not condition:
            self.failures.append({"path": path, "observed": safe_json(observed),
                                  "expected": safe_json(expected)})

    def equal(self, observed, expected, path):
        self.require(observed == expected, path, observed, expected)

    def close(self, observed, expected, path, *, rtol=RTOL, atol=ATOL):
        try:
            left, right = np.asarray(observed, dtype=float), np.asarray(expected, dtype=float)
            valid = left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
            error = float(np.max(np.abs(left - right))) if valid and left.size else 0.
            if valid and error > self.max_error["absolute"]:
                self.max_error = {"absolute": error, "path": path}
            passed = valid and np.allclose(left, right, rtol=rtol, atol=atol)
        except (TypeError, ValueError):
            passed = False
        self.require(bool(passed), path, observed, expected)

    def tree(self, observed, expected, path):
        if isinstance(expected, dict):
            self.require(isinstance(observed, dict), path + ".mapping", type(observed).__name__, "dict")
            if isinstance(observed, dict):
                for key, value in expected.items():
                    self.require(key in observed, path + "." + key + ".present", key in observed, True)
                    if key in observed:
                        self.tree(observed[key], value, path + "." + key)
        elif expected is None or isinstance(expected, (str, bool)):
            self.equal(observed, expected, path)
        else:
            self.close(observed, expected, path)

    def result(self):
        return {"passed": not self.failures, "checks": self.count,
                "failure_count": len(self.failures), "failures": self.failures,
                "maximum_numeric_comparison_error": self.max_error}


def physical_metrics(reads, u, projector, gain, masks):
    clean = reads @ u
    retained = np.einsum("nti,tij->ntj", reads, projector @ u, optimize=True)

    def measure(index):
        c, r = clean[index], retained[index]
        gained = gain * r
        return {
            "gain": gain,
            "retained_rms": float(rms(gained)),
            "clean_displacement_rms": float(rms(gained - c)),
            "clean_write_rms": float(rms(c)),
            "retained_rms_by_pass": rms(gained, axis=(0, 2)).tolist(),
            "clean_displacement_rms_by_pass": rms(gained - c, axis=(0, 2)).tolist(),
            "clean_write_rms_by_pass": rms(c, axis=(0, 2)).tolist(),
            "quadratic_moments": {"a": float(np.mean(r * r)),
                                  "b": float(np.mean(r * c)),
                                  "c": float(np.mean(c * c))},
        }

    answer = measure(slice(None))
    answer["by_context"] = {name: measure(mask) for name, mask in masks.items()}
    return answer, gain * retained


def delivered_metrics(writes, clean_writes, masks):
    delta = np.asarray(writes, dtype=np.float64) - np.asarray(clean_writes, dtype=np.float64)
    return {
        "rms": float(rms(delta)),
        "rms_by_pass": rms(delta, axis=(0, 2)).tolist(),
        "by_context": {
            name: {"n": int(mask.sum()), "rms": float(rms(delta[mask])),
                   "rms_by_pass": rms(delta[mask], axis=(0, 2)).tolist()}
            for name, mask in masks.items()
        },
    }


def projector_diagnostics(projector, checks, path):
    valid = projector.shape == (4, 20, 20) and bool(np.isfinite(projector).all())
    checks.require(valid, path + ".shape_and_finite", list(projector.shape), [4, 20, 20])
    if not valid:
        return {"valid_shape_and_finite": False}
    symmetry = np.max(np.abs(projector - projector.swapaxes(-1, -2)), axis=(1, 2))
    idempotence = np.max(np.abs(projector @ projector - projector), axis=(1, 2))
    spectra = np.linalg.eigvalsh((projector + projector.swapaxes(-1, -2)) / 2)
    ranks = np.linalg.matrix_rank(projector, tol=PROJECTOR_TOL).tolist()
    checks.require(bool(np.all(symmetry <= PROJECTOR_TOL)), path + ".symmetry", symmetry, PROJECTOR_TOL)
    checks.require(bool(np.all(idempotence <= PROJECTOR_TOL)), path + ".idempotence", idempotence, PROJECTOR_TOL)
    checks.equal(ranks, [4] * 4, path + ".rank")
    checks.close(spectra, np.tile([0.] * 16 + [1.] * 4, (4, 1)),
                 path + ".spectrum", rtol=0, atol=PROJECTOR_TOL)
    return {"rank_by_pass": ranks, "symmetry_max_abs_by_pass": symmetry.tolist(),
            "idempotence_max_abs_by_pass": idempotence.tolist(),
            "trace_by_pass": np.trace(projector, axis1=1, axis2=2).tolist(),
            "eigenvalues_by_pass": spectra.tolist()}


def audit_d_search(record, reference, checks, path):
    """Check the registered rule on archived scalars without calling a model."""
    queries = record.get("queries", [])
    checks.tree(record, {"policy": "D", "gain_bounds": [.5, 1.5], "grid_size": 17,
                         "maximum_rollout_calls": 41, "relative_tolerance": D_TOL,
                         "rollout_calls": len(queries)}, path)
    checks.require(len(queries) <= 41, path + ".budget", len(queries), "<=41")
    if not finite(reference) or reference <= 1e-12:
        checks.tree(record, {"admitted": False, "gain": None, "reason": "INVALID_REFERENCE",
                             "rollout_calls": 0}, path)
        return {"recomputed_outcome": "INVALID_REFERENCE", "selected_query_index": None}
    checks.close(record.get("reference_displacement_rms"), reference, path + ".reference")
    checks.require(len(queries) >= 17, path + ".complete_grid", len(queries), ">=17")
    if len(queries) < 17:
        return {"recomputed_outcome": "INCOMPLETE_QUERY_ARCHIVE", "selected_query_index": None}
    valid, errors = [], []
    for i, query in enumerate(queries):
        label = f"{path}.queries[{i}]"
        checks.equal(query.get("index"), i, label + ".index")
        checks.equal(query.get("stage"), "grid" if i < 17 else "bisection", label + ".stage")
        gain, displacement = query.get("gain"), query.get("displacement")
        checks.require(finite(gain) and .5 <= gain <= 1.5, label + ".gain_bounds", gain, [.5, 1.5])
        okay = finite(displacement) and displacement >= 0
        valid.append(okay)
        checks.equal(query.get("finite_nonnegative"), okay, label + ".finite_nonnegative")
        error = abs(displacement - reference) / reference if okay else None
        errors.append(error)
        checks.tree(query.get("relative_error"), error, label + ".relative_error")
        if i < 17:
            checks.close(gain, float(GRID[i]), label + ".grid_gain", rtol=0, atol=0)

    selected_bracket = final_bracket = None
    expected_count, candidates = 17, []
    reason = "INVALID_DISPLACEMENT"
    grid_matches = [i for i in range(17) if valid[i] and errors[i] <= D_TOL]
    if all(valid[:17]):
        if grid_matches:
            candidates, reason = grid_matches, "MATCHED"
        else:
            brackets = [(i, i + 1) for i in range(16)
                        if (queries[i]["displacement"] - reference) *
                           (queries[i + 1]["displacement"] - reference) < 0]
            if not brackets:
                reason = "NO_REGISTERED_BRACKET"
            else:
                lo, hi = min(brackets, key=lambda pair: (
                    abs((float(GRID[pair[0]]) + float(GRID[pair[1]])) / 2 - 1),
                    float(GRID[pair[0]])))
                left, right = dict(queries[lo]), dict(queries[hi])
                selected_bracket = [left["gain"], right["gain"]]
                expected_count = 41
                for i, query in enumerate(queries[17:41], start=17):
                    checks.close(query.get("gain"), (left["gain"] + right["gain"]) / 2,
                                 f"{path}.queries[{i}].midpoint", rtol=0, atol=0)
                    if valid[i]:
                        if ((left["displacement"] - reference) *
                                (query["displacement"] - reference)) <= 0:
                            right = query
                        else:
                            left = query
                final_bracket = [left["gain"], right["gain"]]
                if len(queries) == 41 and all(valid):
                    candidates = [i for i, error in enumerate(errors) if error <= D_TOL]
                    reason = "MATCHED" if candidates else "MATCH_TOLERANCE_FAILED"
    checks.tree(record, {"rollout_calls": expected_count,
                         "bisection_steps": expected_count - 17,
                         "selected_bracket": selected_bracket,
                         "final_bracket": final_bracket,
                         "reason": reason, "admitted": bool(candidates)}, path)
    selected = min(candidates, key=lambda i: (
        abs(queries[i]["gain"] - 1), queries[i]["gain"])) if candidates else None
    if selected is None:
        checks.tree(record, {"gain": None, "matched_displacement_rms": None,
                             "relative_error": None}, path)
        checks.require("selected_query_index" not in record,
                       path + ".no_selected_query", record.get("selected_query_index"), None)
    else:
        checks.tree(record, {"gain": queries[selected]["gain"],
                             "matched_displacement_rms": queries[selected]["displacement"],
                             "relative_error": errors[selected],
                             "selected_query_index": selected}, path)
    grid_ratios = [q["displacement"] / reference for q in queries[:17]
                   if finite(q.get("displacement"))]
    side = ("ALL_GRID_ABOVE_REFERENCE" if grid_ratios and min(grid_ratios) > 1 else
            "ALL_GRID_BELOW_REFERENCE" if grid_ratios and max(grid_ratios) < 1 else
            "GRID_SPANS_REFERENCE")
    return {"recomputed_outcome": reason, "selected_query_index": selected,
            "expected_rollout_calls": expected_count,
            "grid_minimum_displacement_ratio_to_reference": min(grid_ratios) if grid_ratios else None,
            "grid_maximum_displacement_ratio_to_reference": max(grid_ratios) if grid_ratios else None,
            "grid_position_relative_to_reference": side,
            "missing_bracket_is_global_infeasibility": False}


def ratio_detail(measurement, anchor):
    def ratio(a, b):
        return a / b if finite(a) and finite(b) and b > 1e-12 else None
    pooled = ratio(measurement["rms"], anchor["rms"])
    return {
        "pooled_ratio": pooled,
        "pooled_signed_relative_difference": pooled - 1 if pooled is not None else None,
        "pooled_absolute_relative_difference": abs(pooled - 1) if pooled is not None else None,
        "ratio_by_pass": [ratio(a, b) for a, b in zip(measurement["rms_by_pass"], anchor["rms_by_pass"])],
        "by_context": {
            name: {
                "ratio": ratio(measurement["by_context"][name]["rms"], anchor["by_context"][name]["rms"]),
                "ratio_by_pass": [ratio(a, b) for a, b in zip(
                    measurement["by_context"][name]["rms_by_pass"],
                    anchor["by_context"][name]["rms_by_pass"])],
            } for name in CONTEXTS
        },
    }


def base_projector(arrays, arm):
    if f"projector__{arm}" in arrays:
        return arrays[f"projector__{arm}"]
    if arm.startswith("sham_D_"):
        return arrays.get("projector__" + arm.replace("sham_D_", "sham_E_", 1))
    return arrays.get("constructed_projector__" + arm.rsplit("_", 1)[0])


def audit_root(path):
    checks = Checks()
    record = json.loads(path.read_text())
    root = int(record["root"])
    answer = {"root": root, "phase": record.get("phase"), "receipt": str(path),
              "receipt_sha256": file_hash(path), "protocol_commit": record.get("protocol_commit"),
              "source_lock_sha256": record.get("source_lock_sha256"),
              "scientific_global_failures": record.get("global_failures", []),
              "scientific_admissions": record.get("admissions", {}),
              "arms": {}, "calibrations": {}, "table": []}
    checks.equal(record.get("study"), "BPD-NC-13", "receipt.study")
    checks.equal(record.get("complete"), True, "receipt.complete")
    checks.equal(path.name, f"root_{root}.json", "receipt.filename")
    expected_phase = "EXCLUDED_SMOKE" if root == 2601299 else "DEVELOPMENT_ENGINEERING"
    checks.equal(record.get("phase"), expected_phase, "receipt.phase")
    checks.require(2601299 <= root <= 2601315, "receipt.registered_root", root, "2601299..2601315")

    artifacts = record.get("artifacts", {})
    for name, entry in artifacts.items():
        artifact = path.parent / entry["file"]
        checks.require(artifact.is_file(), f"artifacts.{name}.present", artifact.is_file(), True)
        if artifact.is_file():
            checks.equal(file_hash(artifact), entry["sha256"], f"artifacts.{name}.sha256")
    choices_path = path.parent / artifacts.get("choices", {}).get("file", f"root_{root}_choices.npz")
    with np.load(choices_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    hashes = record.get("choice_array_hashes_sha256", {})
    checks.equal(sorted(hashes), sorted(arrays), "choice_arrays.manifest_keys")
    for name, value in arrays.items():
        checks.equal(array_hash(value), hashes.get(name), f"choice_arrays.{name}.sha256")
    frozen_path = path.parent / artifacts.get("frozen", {}).get("file", f"root_{root}_frozen.json")
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text())
        for key in ("calibrations", "dose", "reference", "sham_selection",
                    "choice_array_hashes_sha256", "choices_frozen_utc"):
            checks.equal(record.get(key), frozen.get(key), "frozen." + key)
    else:
        checks.require(False, "frozen.present", False, True)

    reads = np.asarray(arrays["dose_clean_reads"], dtype=np.float64)
    u = np.asarray(arrays["B_physical_U"], dtype=np.float64)
    clean_writes = np.asarray(arrays["dose_clean_writes"], dtype=np.float64)
    inputs = arrays["dose_inputs"]
    for name, value, shape in (("dose_clean_reads", reads, (1024, 4, 20)),
                               ("B_physical_U", u, (20, 20)),
                               ("dose_clean_writes", clean_writes, (1024, 4, 20)),
                               ("dose_inputs", inputs, (1024, 5))):
        checks.equal(value.shape, shape, "choice_arrays." + name + ".shape")
        checks.require(bool(np.isfinite(value).all()), "choice_arrays." + name + ".finite",
                       bool(np.isfinite(value).all()), True)
    masks = {"shared": inputs[:, 4] > 0, "specific": inputs[:, 4] <= 0}
    for name, mask in masks.items():
        checks.require(bool(mask.any()), "dose_context." + name + ".nonempty", int(mask.sum()), ">0")
    answer["context_counts"] = {name: int(mask.sum()) for name, mask in masks.items()}
    closure = float(rms(reads @ u - clean_writes)) / max(float(rms(clean_writes)), 1e-12)
    checks.require(closure <= NATIVE_REL_TOL, "physical_clean.archived_write_closure_relative_rms",
                   closure, NATIVE_REL_TOL)
    answer["native_clean_write_relative_rms_error"] = closure

    projectors = {key.removeprefix("projector__"): value
                  for key, value in arrays.items() if key.startswith("projector__")}
    chosen = set(projectors)
    for prefix in ("gain__", "dose_writes__"):
        checks.equal(sorted(key.removeprefix(prefix) for key in arrays if key.startswith(prefix)),
                     sorted(chosen), "choice_arrays." + prefix + "arm_set")
    checks.equal(sorted(record.get("dose", {})), sorted(chosen), "dose.arm_set")
    if "mixed" not in chosen:
        checks.require(bool(record.get("global_failures")), "anchor_unavailable.global_failure",
                       record.get("global_failures"), "nonempty")
        answer["coverage"] = "COMMON_ANCHOR_UNAVAILABLE"
        answer["checks"] = checks.result()
        return answer

    reference = record["reference"]
    reference_e = reference["retained_rms"]
    reference_d = reference["delivered_displacement_rms"]
    anchor_actual = delivered_metrics(arrays["dose_writes__mixed"], clean_writes, masks)
    anchor_physical, _ = physical_metrics(reads, u, projectors["mixed"], 1., masks)
    checks.tree(reference, {"gain": 1., "retained_rms": anchor_physical["retained_rms"],
                            "delivered_displacement_rms": anchor_actual["rms"]}, "reference")
    answer["reference_recomputed"] = {"retained_rms": anchor_physical["retained_rms"],
                                      "actual_closed_loop_displacement": anchor_actual}

    for arm, projector in sorted(projectors.items()):
        label = "arms." + arm
        gain = float(arrays["gain__" + arm])
        checks.require(math.isfinite(gain) and .5 <= gain <= 1.5,
                       label + ".gain_bounds", gain, [.5, 1.5])
        diagnostics = projector_diagnostics(projector, checks, label + ".projector")
        physical, retained = physical_metrics(reads, u, projector, gain, masks)
        writes = arrays["dose_writes__" + arm]
        checks.equal(writes.shape, clean_writes.shape, label + ".write_shape")
        checks.require(bool(np.isfinite(writes).all()), label + ".finite_writes",
                       bool(np.isfinite(writes).all()), True)
        actual = delivered_metrics(writes, clean_writes, masks)
        checks.tree(record["dose"].get(arm), {**physical, "actual_closed_loop_displacement": actual},
                    label + ".recorded_dose")
        first_error = float(rms(np.asarray(writes, dtype=float)[:, 0] - retained[:, 0]))
        first_relative = first_error / max(float(rms(retained[:, 0])), 1e-12)
        checks.require(first_relative <= NATIVE_REL_TOL,
                       label + ".first_pass_gain_write_relative_error", first_relative, NATIVE_REL_TOL)
        moments = physical["quadratic_moments"]
        squared = gain * gain * moments["a"] - 2 * gain * moments["b"] + moments["c"]
        checks.close(physical["clean_displacement_rms"] ** 2, squared,
                     label + ".clean_displacement_quadratic")
        ratio = ratio_detail(actual, anchor_actual)
        row = {"root": root, "phase": record.get("phase"), "arm": arm, "gain": gain,
               "scientifically_admitted": record.get("admissions", {}).get(arm, {}).get("admitted"),
               "retained_clean_rms": physical["retained_rms"],
               "retained_clean_ratio_to_anchor": physical["retained_rms"] / reference_e if reference_e > 1e-12 else None,
               "clean_displacement_rms": physical["clean_displacement_rms"],
               "actual_displacement_rms": actual["rms"],
               "actual_displacement_ratio_to_anchor": ratio["pooled_ratio"],
               "actual_displacement_relative_error_to_anchor": ratio["pooled_absolute_relative_difference"],
               "actual_displacement_ratio_by_pass": ratio["ratio_by_pass"],
               "actual_displacement_ratio_by_context": {
                   name: ratio["by_context"][name]["ratio"] for name in CONTEXTS}}
        answer["table"].append(row)
        answer["arms"][arm] = {"projector": diagnostics, "gain": gain,
                               "recomputed_clean_metrics": physical,
                               "recomputed_actual_closed_loop_displacement": actual,
                               "actual_displacement_relative_to_anchor": ratio,
                               "first_pass_physical_write_relative_rms_error": first_relative}
        if arm == "mixed":
            checks.close(gain, 1., label + ".anchor_gain", rtol=0, atol=0)
        elif arm.endswith("_D") or arm.startswith("sham_D_"):
            checks.require(ratio["pooled_absolute_relative_difference"] is not None and
                           ratio["pooled_absolute_relative_difference"] <= D_TOL,
                           label + ".archived_selected_D_match",
                           ratio["pooled_absolute_relative_difference"], D_TOL)
        if arm.startswith("sham_D_"):
            e_arm = arm.replace("sham_D_", "sham_E_", 1)
            checks.require(e_arm in projectors, label + ".E_projector_present", e_arm in projectors, True)
            if e_arm in projectors:
                checks.close(projector, projectors[e_arm],
                             label + ".same_sham_under_E_and_D", rtol=0, atol=0)
        else:
            constructed = arrays.get("constructed_projector__" + (arm if arm == "mixed" else arm.rsplit("_", 1)[0]))
            if constructed is not None:
                checks.close(projector, constructed, label + ".constructed_projector", rtol=0, atol=0)

    search_calls = 0
    for arm, calibration in sorted(record.get("calibrations", {}).items()):
        label = "calibrations." + arm
        admitted = calibration.get("admitted") is True
        checks.equal(arm in chosen, admitted, label + ".selected_arrays_present")
        if admitted:
            checks.close(float(arrays["gain__" + arm]), calibration.get("gain"),
                         label + ".archived_gain", rtol=0, atol=0)
        if record.get("admissions", {}).get(arm, {}).get("admitted"):
            checks.require(admitted, label + ".scientific_admission_requires_calibration", admitted, True)
        if calibration.get("policy") == "D":
            answer["calibrations"][arm] = audit_d_search(calibration, reference_d, checks, label)
            search_calls += len(calibration.get("queries", []))
            if admitted:
                actual = answer["arms"][arm]["recomputed_actual_closed_loop_displacement"]["rms"]
                checks.close(calibration.get("matched_displacement_rms"), actual,
                             label + ".selected_query_vs_archived_writes")
        elif arm in ("shared_E", "target_E", "permuted_E"):
            projector = base_projector(arrays, arm)
            checks.require(projector is not None, label + ".projector_available", projector is not None, True)
            if projector is None:
                continue
            raw, _ = physical_metrics(reads, u, projector, 1., masks)
            observed = raw["retained_rms"]
            raw_gain = reference_e / observed if observed > 1e-12 and reference_e > 1e-12 else None
            reason = ("INVALID_REFERENCE" if not finite(reference_e) or reference_e <= 1e-12 else
                      "ZERO_RETAINED_WRITE" if observed <= 1e-12 else
                      "MATCHED" if .5 <= raw_gain <= 1.5 else "GAIN_OUTSIDE_REGISTERED_BOUNDS")
            okay = reason == "MATCHED"
            checks.tree(calibration, {"policy": "E", "admitted": okay, "reason": reason,
                                      "raw_gain": raw_gain, "gain": raw_gain if okay else None,
                                      "observed_retained_rms": observed,
                                      "matched_retained_rms": raw_gain * observed if okay else None,
                                      "reference_retained_rms": reference_e,
                                      "gain_bounds": [.5, 1.5]}, label)
            answer["calibrations"][arm] = {"gain_recomputed": raw_gain,
                                           "admitted_recomputed": okay,
                                           "retained_rms_at_gain_one_recomputed": observed}
        elif arm.startswith("sham_E_"):
            physical = answer["arms"][arm]["recomputed_clean_metrics"]
            checks.tree(calibration, {"admitted": True, "gain": 1.,
                                      "reason": "SELECTED_WITHIN_15_PERCENT_E"}, label)
            residual = abs(physical["retained_rms"] - reference_e) / reference_e
            checks.require(residual <= E_SHAM_TOL, label + ".physical_E_tolerance", residual, E_SHAM_TOL)
            answer["calibrations"][arm] = {"retained_rms_relative_error_recomputed": residual}
        else:
            checks.require(False, label + ".recognized_calibration", calibration, "E, D, or E sham")

    sham = record.get("sham_selection", {})
    if "sham_proposal_retained_rms" in arrays and finite(reference_e) and reference_e > 1e-12:
        proposal_e = arrays["sham_proposal_retained_rms"]
        selected = arrays["sham_selected_indices"].tolist()
        qualified = np.flatnonzero(np.abs(proposal_e - reference_e) / reference_e <= E_SHAM_TOL)
        expected_ids = qualified[:8].tolist()
        checks.equal(list(proposal_e.shape), [2048], "shams.proposal_count")
        checks.equal(selected, expected_ids, "shams.first_eight_qualified")
        checks.tree(sham, {"proposals": 2048, "evaluated_proposals": 2048,
                           "selected_indices": expected_ids, "selected": len(expected_ids),
                           "qualified": len(qualified), "reference_retained_rms": reference_e,
                           "generator_seed": root * 1009 + 13013, "relative_tolerance": .15,
                           "admitted": len(expected_ids) == 8,
                           "reason": "MATCHED" if len(expected_ids) == 8 else "INSUFFICIENT_ENERGY_MATCHED_SHAMS",
                           "rejection_counts": {"outside_energy_tolerance": 2048 - len(qualified)},
                           "proposal_indices": list(range(2048)),
                           "proposal_retained_rms": proposal_e.tolist(),
                           "proposal_angles": arrays["sham_proposal_angles"].tolist()}, "shams.recorded_selection")
        checks.close(arrays["sham_proposal_angles"], np.resize([.45, .65, .85, 1.05], 2048),
                     "shams.registered_angle_cycle", rtol=0, atol=0)
        for j, proposal in enumerate(selected):
            arm = "sham_E_" + str(j)
            checks.require(arm in chosen, f"shams.selected_{j}.projector_archived", arm in chosen, True)
            if arm in chosen:
                observed = answer["arms"][arm]["recomputed_clean_metrics"]["retained_rms"]
                checks.close(proposal_e[proposal], observed,
                             f"shams.selected_{j}.proposal_energy_vs_physical_vectors")
        answer["sham_selection_audit"] = {"selected_indices_recomputed": expected_ids,
                                          "qualified_from_archived_scalar_values": len(qualified),
                                          "unselected_projectors_regenerated": False}

    access = record.get("target_access", {})
    checks.close(access.get("search_rollouts"), search_calls, "target_access.search_rollouts", rtol=0, atol=0)
    forward = access.get("forward_calls", {}).get("D_search", {"calls": 0, "input_rows": 0})
    checks.tree(forward, {"calls": search_calls, "input_rows": search_calls * 1024}, "target_access.D_search")
    checks.require(search_calls <= 451, "target_access.D_search_budget", search_calls, "<=451")
    checks.require(access.get("original_gradient_inputs", 0) <= 8,
                   "target_access.gradient_budget_recorded", access.get("original_gradient_inputs"), "<=8")
    answer["search_rollouts_from_query_rows"] = search_calls
    answer["coverage"] = "ARCHIVED_SELECTED_PHYSICAL_VECTORS"
    answer["checks"] = checks.result()
    return answer


def summarize(roots):
    summary = {}
    for phase in sorted({root.get("phase") for root in roots if root.get("phase")}):
        groups, searches = {}, {}
        for root in roots:
            if root.get("phase") != phase:
                continue
            for arm, detail in root.get("arms", {}).items():
                if arm == "mixed" or arm.endswith("_D") or arm.startswith("sham_D_"):
                    groups.setdefault(arm, []).append((root["root"], detail["actual_displacement_relative_to_anchor"]))
            for arm, detail in root.get("calibrations", {}).items():
                if "recomputed_outcome" in detail:
                    searches.setdefault(arm, []).append({"root": root["root"], **detail})
        phase_result = {"actual_D_ratios_by_arm": {}, "D_searches_by_arm": searches}
        for arm, rows in sorted(groups.items()):
            def describe(values):
                values = np.asarray([v for v in values if finite(v)], dtype=float)
                if not values.size:
                    return {"n": 0}
                return {"n": len(values), "minimum": float(values.min()), "median": float(np.median(values)),
                        "mean": float(values.mean()), "maximum": float(values.max()),
                        "maximum_absolute_deviation_from_one": float(np.max(np.abs(values - 1))),
                        "outside_pooled_tolerance_count": int(np.sum(np.abs(values - 1) > D_TOL))}
            phase_result["actual_D_ratios_by_arm"][arm] = {
                "roots": [root for root, _ in rows],
                "pooled": describe([row["pooled_ratio"] for _, row in rows]),
                "by_context": {name: describe([row["by_context"][name]["ratio"] for _, row in rows]) for name in CONTEXTS},
                "by_pass": [describe([row["ratio_by_pass"][p] for _, row in rows]) for p in range(4)],
            }
        summary[phase] = phase_result
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    directory = args.receipts.resolve()
    paths = sorted(path for path in directory.rglob("root_*.json") if re.fullmatch(r"root_\d+\.json", path.name))
    roots, issues = [], []
    if not paths:
        issues.append({"reason": "NO_COMPLETE_RECEIPT_FILES", "directory": str(directory)})
    for path in paths:
        try:
            roots.append(audit_root(path))
        except Exception as exc:
            issues.append({"reason": "RECEIPT_AUDIT_EXCEPTION", "receipt": str(path),
                           "exception": type(exc).__name__, "message": str(exc)})
    ids = [root["root"] for root in roots]
    duplicates = sorted({root for root in ids if ids.count(root) > 1})
    if duplicates:
        issues.append({"reason": "DUPLICATE_ROOT_RECEIPTS", "roots": duplicates})
    if any(2601300 <= root <= 2601315 for root in ids):
        missing = sorted(set(range(2601300, 2601316)) - set(ids))
        if missing:
            issues.append({"reason": "DEVELOPMENT_ROOT_RECEIPTS_MISSING", "roots": missing})
    partials = sorted(str(path) for path in directory.rglob("root_*.failure.json"))
    if partials:
        issues.append({"reason": "PRESERVED_SOURCE_FAILURE_RECEIPTS", "files": partials})
    passed = not issues and all(root["checks"]["passed"] for root in roots)
    report = {
        "audit": "NC13_POST_LOCK_PHYSICAL_RECEIPT_AUDIT", "version": 1,
        "stage": "POST_LOCK_REPORT_AUDIT", "execution_lock_modified": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_source_sha256": file_hash(Path(__file__)),
        "runtime": {"python": sys.version, "numpy": np.__version__},
        "receipts_directory": str(directory), "passed": passed,
        "complete_receipts_found": len(paths), "roots_audited": len(roots),
        "root_ids": sorted(ids), "issues": issues,
        "numeric_tolerances": {"comparison_rtol": RTOL, "comparison_atol": ATOL,
                               "projector_max_abs": PROJECTOR_TOL,
                               "native_float32_vs_float64_relative_rms": NATIVE_REL_TOL,
                               "registered_D_relative_tolerance": D_TOL},
        "scope": [
            "Recompute selected clean writes from archived reads, projectors, gains and B physical U.",
            "Subtract archived clean writes from archived selected closed-loop writes in float64.",
            "Check E gains and D search decisions against the fixed numerical rules.",
            "Check hashes, projector geometry, sham reuse and selected sham physical energies.",
            "Summarize context and pass ratios to the mixed anchor separately for smoke and development.",
        ],
        "limits": [
            "Unselected D-query write vectors are absent; only their recorded scalar arithmetic and search sequence can be checked.",
            "Unselected sham projectors are absent; their recorded RMS list and selection order are checked without regenerating projectors.",
            "Archived selected writes are treated as recorded physical outputs; no model is executed to certify them.",
            "D admission targets the pooled dose population. Context and pass ratios are descriptive and have no registered match gate.",
            "This post-lock audit creates no new scientific admissions, bootstrap intervals or independent experimental roots.",
        ],
        "summary_by_phase": summarize(roots),
        "table": [row for root in roots for row in root.get("table", [])],
        "roots": roots,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(safe_json(report), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"passed": passed, "roots_audited": len(roots),
                      "failed_roots": [root["root"] for root in roots if not root["checks"]["passed"]],
                      "issues": issues, "out": str(args.out.resolve())}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
