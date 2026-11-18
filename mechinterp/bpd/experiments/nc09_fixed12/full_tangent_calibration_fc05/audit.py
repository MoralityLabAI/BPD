"""Independent NumPy replay of saved FC05 source and terminal task evidence.

This module imports no model, Torch profile, solver or task scoring function.
All mismatches are returned as explicit audit failures with the original
chosen projectors and root identity retained by the caller.
"""
from __future__ import annotations

import base64
import hashlib
import itertools
from pathlib import Path

import numpy as np

SOURCE_ATOL = 1e-8
SOURCE_RTOL = 1e-8
TASK_ATOL = 1e-6
NUMERICAL_ALLOWANCE = 1e-12
MANDATORY_ARMS = ("dp12", "flat3", "flat3_dose", "flat4")


def array_receipt(value) -> dict:
    a = np.ascontiguousarray(np.asarray(value))
    if a.dtype.hasobject:
        raise ValueError("Object arrays cannot be archived as numerical evidence")
    binary = a.tobytes(order="C")
    return {"shape": list(a.shape), "dtype": a.dtype.str,
            "size_bytes": len(binary),
            "sha256": hashlib.sha256(binary).hexdigest()}


def file_receipt(path: str | Path) -> dict:
    path = Path(path)
    binary = path.read_bytes()
    receipt = {"file": path.name, "size_bytes": len(binary),
               "sha256": hashlib.sha256(binary).hexdigest()}
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            receipt["arrays"] = {name: array_receipt(arrays[name])
                                 for name in arrays.files}
    return receipt


def save_npz(path: str | Path, arrays: dict | None = None, **kwargs) -> dict:
    """Write one compressed archive and return its file/array byte manifest."""
    contents = dict(arrays or {})
    if set(contents) & set(kwargs):
        raise ValueError("Duplicate named NPZ array")
    contents.update(kwargs)
    for value in contents.values():
        array_receipt(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **contents)
    return file_receipt(path)


def _load_archive(record, path, key):
    receipt = record.get(key)
    if not isinstance(receipt, dict):
        raise ValueError(f"{key}_receipt_missing")
    path = Path(path)
    actual = file_receipt(path)
    for field in ("size_bytes", "sha256", "arrays"):
        if actual.get(field) != receipt.get(field):
            raise ValueError(f"{key}_{field}_mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{key}_nonfinite_array:{name}")
    return arrays, actual


def _same_number(actual, logged, errors, label, *, atol=SOURCE_ATOL, rtol=SOURCE_RTOL):
    try:
        valid = bool(np.isfinite(logged) and np.isclose(
            actual, logged, atol=atol, rtol=rtol))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        errors.append(f"{label}_mismatch")


def source_profile(a, z, g, U, P) -> dict:
    """Compute directly from saved projectors, including full four-pass sums."""
    removed = z - np.einsum("tij,ntj->nti", P, z)
    d = np.einsum("nti,nti->n", g, removed)
    writes = np.einsum("nti,tij,jk->ntk", a, P, U)
    return {
        "physical_write_rms": float(np.sqrt(np.mean(writes * writes))),
        "signed_first_order_score_mean": float(d.mean()),
        "first_order_score_rms": float(np.sqrt(np.mean(d * d))),
    }


def _decoded_projector(receipt, expected, errors, label):
    try:
        binary = base64.b64decode(receipt["projector_b64_le_float64"], validate=True)
        if hashlib.sha256(binary).hexdigest() != receipt["projector_sha256"]:
            errors.append(f"{label}_projector_digest_mismatch")
        decoded = np.frombuffer(binary, dtype="<f8").reshape((4, 20, 20))
        if binary != np.ascontiguousarray(expected, dtype="<f8").tobytes():
            errors.append(f"{label}_projector_bytes_mismatch")
        return decoded
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"{label}_invalid_projector_receipt:{type(exc).__name__}")
        return expected


def _geometry(P, ranks, errors, label):
    symmetry = float(np.max(np.abs(P - P.transpose(0, 2, 1))))
    idempotence = float(np.max(np.abs(P @ P - P)))
    traces = np.trace(P, axis1=1, axis2=2)
    eigenvalues = np.linalg.eigvalsh(P)
    expected = np.stack([np.r_[np.zeros(20-r), np.ones(r)] for r in ranks])
    eigen_error = float(np.max(np.abs(eigenvalues - expected)))
    rank_counts = (eigenvalues > .5).sum(axis=1).tolist()
    trace_error = float(np.max(np.abs(traces - np.asarray(ranks))))
    if max(symmetry, idempotence, trace_error, eigen_error) > 1e-6:
        errors.append(f"{label}_projector_geometry_failed")
    if rank_counts != ranks:
        errors.append(f"{label}_rank_failed")
    return {
        "symmetry_max": symmetry, "idempotence_max": idempotence,
        "trace_max_error": trace_error, "eigenvalue_max_error": eigen_error,
        "ranks": rank_counts, "traces": traces.tolist(),
    }


def _source_candidates(record, z, g, ranks, P0, errors):
    """Replay the NC08 gradient-covariance candidates without importing NC08."""
    covariance = np.einsum("nti,ntj->tij", g, g) / len(g)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    logged = record.get("passwise_source_candidates")
    if not isinstance(logged, dict):
        errors.append("passwise_source_candidates_missing")
        return {}
    result = {}
    eligible_ranks = {}
    for t in range(4):
        result[str(t)] = {}
        eligible_ranks[t] = {}
        saved_pass = logged.get(str(t), logged.get(t, {}))
        for r in range(2, 7):
            E = vectors[t, :, -r:]
            P = E @ E.T
            er = float(eigenvalues[t, -r])
            en = float(eigenvalues[t, -r-1])
            ratio = er / max(en, 1e-12)
            eligible = er > 1e-12 and ratio >= 1.001
            removed = z[:, t] @ (np.eye(20) - P)
            score = np.sum(g[:, t] * removed, axis=1)
            proxy = float(np.mean(score * score))
            values = {"eigvalue_at_rank": er, "eigvalue_next": en,
                      "eigengap_ratio": ratio,
                      "squared_local_omitted_model_score": proxy,
                      "eligible": bool(eligible)}
            saved = saved_pass.get(str(r), saved_pass.get(r, {}))
            for field in ("eigvalue_at_rank", "eigvalue_next", "eigengap_ratio",
                          "squared_local_omitted_model_score"):
                _same_number(values[field], saved.get(field), errors,
                             f"source_candidate_{t}_{r}_{field}")
            if saved.get("eligible") is not bool(eligible):
                errors.append(f"source_candidate_{t}_{r}_eligibility_mismatch")
            result[str(t)][str(r)] = values
            eligible_ranks[t][r] = bool(eligible)
            if r == ranks[t] and float(np.max(np.abs(P-P0[t]))) > 1e-6:
                errors.append(f"source_dp12_eigenspace_mismatch:{t}")
    required = {"dp12": ranks, "flat3": [3] * 4, "flat4": [4] * 4}
    if (record.get("source_arms", {}).get("reverse") or {}).get("eligible"):
        required["reverse"] = list(reversed(ranks))
    for name, arm_ranks in required.items():
        if not all(eligible_ranks[t][r] for t, r in enumerate(arm_ranks)):
            errors.append(f"source_{name}_eigengap_admission_failed")
    best = None
    for rank_tuple in itertools.product(range(2, 7), repeat=4):
        if sum(rank_tuple) != 12 or not all(
                eligible_ranks[t][r] for t, r in enumerate(rank_tuple)):
            continue
        loss = sum(result[str(t)][str(r)]["squared_local_omitted_model_score"]
                   for t, r in enumerate(rank_tuple))
        if (best is None or loss < best[1]-1e-12 or
                (abs(loss-best[1]) <= 1e-12 and rank_tuple < best[0])):
            best = (rank_tuple, loss)
    if best is None or tuple(ranks) != best[0]:
        errors.append("source_optimal_fixed12_tuple_mismatch")
    if best is not None:
        _same_number(best[1], record["source_rank12_optimum"].get("source_proxy_loss"),
                     errors, "source_optimal_fixed12_proxy")
    return result


def source_audit(record: dict, source_path: str | Path) -> dict:
    result = {"passed": False, "complete": False, "errors": [],
              "source_arrays": {}, "projectors": []}
    errors = result["errors"]
    try:
        arrays, receipt = _load_archive(record, source_path, "source_archive")
        result["source_arrays"] = receipt["arrays"]
        optimum = record.get("source_rank12_optimum")
        if not optimum:
            errors.append("source_rank12_optimum_missing")
            return result
        ranks = list(optimum["ranks"])
        if len(ranks) != 4 or sum(ranks) != 12 or any(r not in range(2, 7) for r in ranks):
            errors.append("source_rank_tuple_invalid")
            return result
        adm = record.get("sham_admission")
        if not isinstance(adm, dict):
            errors.append("sham_admission_missing")
            return result
        a, z, g, U = (np.asarray(arrays[k], dtype=np.float64) for k in ("a", "z", "g", "U"))
        P0 = np.asarray(arrays["dp12_projector"], dtype=np.float64)
        shams = np.asarray(arrays["sham_projectors"], dtype=np.float64)
        if (a.ndim != 3 or a.shape[1:] != (4, 20) or z.shape != a.shape or
                g.shape != a.shape or U.shape != (20, 20) or P0.shape != (4, 20, 20) or
                shams.ndim != 4 or shams.shape[1:] != (4, 20, 20) or len(a) < 1):
            errors.append("source_array_shape_invalid")
            return result
        if len(a) != adm.get("source_observation_count"):
            errors.append("source_observation_count_mismatch")
        if "derivative_discovery" in record.get("splits", {}) and (
                len(a) != record["splits"]["derivative_discovery"]):
            errors.append("source_split_size_mismatch")
        if len(arrays["discovery_inputs"]) != len(a):
            errors.append("discovery_inputs_size_mismatch")
        if not any(name.startswith("model__") for name in arrays):
            errors.append("model_state_arrays_missing")
        result["passwise_source_candidates"] = _source_candidates(
            record, z, g, ranks, P0, errors)
        bases = []
        for t, r in enumerate(ranks):
            E = np.asarray(arrays[f"E_{t}"], dtype=np.float64)
            C = np.asarray(arrays[f"C_{t}"], dtype=np.float64)
            if E.shape != (20, r) or C.shape != (20, 20-r):
                errors.append(f"source_basis_shape_invalid:{t}")
                return result
            if max(float(np.max(np.abs(E.T @ E - np.eye(r)))),
                   float(np.max(np.abs(C.T @ C - np.eye(20-r)))),
                   float(np.max(np.abs(E.T @ C))),
                   float(np.max(np.abs(E @ E.T - P0[t])))) > 1e-6:
                errors.append(f"source_basis_geometry_failed:{t}")
            bases.append(E)
        _decoded_projector(record.get("source_dp12_projector_receipt", {}), P0, errors, "dp12")
        result["dp12_geometry"] = _geometry(P0, ranks, errors, "dp12")
        target = source_profile(a, z, g, U, P0)
        result["target"] = target
        if target["physical_write_rms"] <= 1e-12:
            errors.append("source_target_write_zero")
            return result
        field_map = {
            "physical_write_rms": "target_source_write_rms",
            "signed_first_order_score_mean": "target_source_signed_score_mean",
            "first_order_score_rms": "target_source_score_rms",
        }
        for name, field in field_map.items():
            _same_number(target[name], adm.get(field), errors, field)
        tol_m = max(.005, .25 * abs(target["signed_first_order_score_mean"]))
        tol_v = max(.005, .25 * target["first_order_score_rms"])
        for field, expected in (
            ("source_write_relative_tolerance", .15),
            ("source_signed_score_abs_tolerance", tol_m),
            ("source_score_rms_abs_tolerance", tol_v),
        ):
            _same_number(expected, adm.get(field), errors, field)
        entries = adm.get("projectors", [])
        if len(entries) != len(shams) or adm.get("selected") != len(shams):
            errors.append("source_selected_count_mismatch")
        if adm.get("uses_external_task_labels") is not False:
            errors.append("source_label_isolation_flag_failed")
        if adm.get("numerical_exception") is not None:
            errors.append("source_solver_numerical_exception")
        for j, P in enumerate(shams):
            logged = entries[j] if j < len(entries) else {}
            prefix = f"sham_{j}"
            _decoded_projector(logged, P, errors, prefix)
            geometry = _geometry(P, ranks, errors, prefix)
            profile = source_profile(a, z, g, U, P)
            for field, actual in profile.items():
                _same_number(actual, logged.get(field), errors, f"{prefix}_{field}")
            dW = abs(profile["physical_write_rms"] / target["physical_write_rms"] - 1.)
            dm = abs(profile["signed_first_order_score_mean"] - target["signed_first_order_score_mean"])
            dv = abs(profile["first_order_score_rms"] - target["first_order_score_rms"])
            for field, actual in (("physical_rms_relative_error", dW),
                                  ("signed_score_abs_error", dm), ("score_rms_abs_error", dv)):
                _same_number(actual, logged.get(field), errors, f"{prefix}_{field}")
            dp_distance = float(np.sqrt(np.square(P-P0).sum()/24.))
            pairwise = min((float(np.sqrt(np.square(P-Q).sum()/24.))
                            for Q in shams[:j]), default=None)
            _same_number(dp_distance, logged.get("distance_from_dp12"), errors, f"{prefix}_dp12_distance")
            if pairwise is not None:
                _same_number(pairwise, logged.get("minimum_pairwise_distance"), errors, f"{prefix}_pairwise_distance")
            cosine_error = max(float(np.max(np.abs(
                np.linalg.svd(P[t] @ E, compute_uv=False) - np.cos(.25))))
                for t, E in enumerate(bases))
            if cosine_error > 1e-6:
                errors.append(f"{prefix}_principal_angle_failed")
            if (dW > .15 + NUMERICAL_ALLOWANCE or dm > tol_m + NUMERICAL_ALLOWANCE or
                    dv > tol_v + NUMERICAL_ALLOWANCE):
                errors.append(f"{prefix}_source_inequality_failed")
            if dp_distance < .20 - NUMERICAL_ALLOWANCE or (
                    pairwise is not None and pairwise < .18 - NUMERICAL_ALLOWANCE):
                errors.append(f"{prefix}_separation_failed")
            result["projectors"].append({
                "index": j, "profile": profile, "geometry": geometry,
                "physical_rms_relative_error": dW, "signed_score_abs_error": dm,
                "score_rms_abs_error": dv, "distance_from_dp12": dp_distance,
                "minimum_pairwise_distance": pairwise,
                "principal_cosine_max_error": cosine_error,
                "source_gate_margins": [.15-dW, tol_m-dm, tol_v-dv],
            })
        if len(shams) != 8:
            errors.append("source_sham_set_incomplete")
        result["passed"] = not errors
        result["complete"] = not errors and len(shams) == 8
    except Exception as exc:
        errors.append(f"source_audit_exception:{type(exc).__name__}:{exc}")
    return result


def _nll(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != labels.shape:
        raise ValueError("Task final logits and labels must be equal-length vectors")
    signed = np.where(labels, 1., -1.)
    return float(np.logaddexp(0., -2. * signed * logits).mean())


def task_audit(record: dict, task_path: str | Path) -> dict:
    result = {"passed": False, "complete": False, "errors": [], "recomputed_nll": {}}
    errors = result["errors"]
    try:
        arrays, receipt = _load_archive(record, task_path, "task_archive")
        result["task_arrays"] = receipt["arrays"]
        labels = arrays["labels"]
        if labels.ndim != 1 or not len(labels) or not np.isin(labels, [0, 1]).all():
            errors.append("task_binary_labels_invalid")
            return result
        labels = labels.astype(bool)
        expected_n = record.get("splits", {}).get("labeled_calibration", len(labels))
        if len(labels) != expected_n:
            errors.append("task_split_size_mismatch")
        clean_nll = _nll(arrays["clean_logits"], labels)
        result["recomputed_nll"]["clean"] = clean_nll
        _same_number(clean_nll, record.get("clean_nll"), errors, "clean_nll", atol=TASK_ATOL, rtol=0.)
        arms = record.get("task_nll_by_arm", {})
        for name, metrics in arms.items():
            if metrics is None:
                if name in MANDATORY_ARMS:
                    errors.append(f"mandatory_task_arm_missing:{name}")
                continue
            value = _nll(arrays[f"arm__{name}"], labels)
            result["recomputed_nll"][name] = value
            _same_number(value, metrics.get("nll_nats"), errors,
                         f"task_{name}_nll", atol=TASK_ATOL, rtol=0.)
        for name in MANDATORY_ARMS:
            if name not in arms:
                errors.append(f"mandatory_task_arm_missing:{name}")
        for name in arrays:
            if name.startswith("arm__") and arms.get(name[5:]) is None:
                errors.append(f"task_array_without_metric:{name}")
        sham_metrics = record.get("sham_task_metrics", [])
        sham_nlls = record.get("sham_nlls", [])
        if len(sham_metrics) != len(sham_nlls):
            errors.append("task_sham_metric_count_mismatch")
        selected = (record.get("sham_admission") or {}).get("selected", 0)
        if len(sham_metrics) != selected:
            errors.append("task_selected_count_mismatch")
        values = []
        for j, metrics in enumerate(sham_metrics):
            value = _nll(arrays[f"sham__{j}"], labels)
            values.append(value)
            _same_number(value, metrics.get("nll_nats"), errors,
                         f"task_sham_{j}_nll", atol=TASK_ATOL, rtol=0.)
            _same_number(value, sham_nlls[j] if j < len(sham_nlls) else None,
                         errors, f"task_sham_{j}_scalar_nll", atol=TASK_ATOL, rtol=0.)
        if {n for n in arrays if n.startswith("sham__")} != {
                f"sham__{j}" for j in range(len(sham_metrics))}:
            errors.append("task_sham_array_count_mismatch")
        result["recomputed_nll"]["shams"] = values
        if len(values) != 8:
            errors.append("task_sham_set_incomplete")
        result["passed"] = not errors
        result["complete"] = not errors and len(values) == 8
    except Exception as exc:
        errors.append(f"task_audit_exception:{type(exc).__name__}:{exc}")
    return result
