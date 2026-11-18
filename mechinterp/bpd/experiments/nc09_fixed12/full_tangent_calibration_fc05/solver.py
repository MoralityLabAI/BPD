"""Observe the executed FC04 sampler without changing its selection arithmetic.

FC05 explicitly registers the original early exits and descent guards.  New
receipts never enter a selection condition.  The independent NumPy audit is
performed by the experiment only after this function finishes selection.
"""
from __future__ import annotations

import base64
import hashlib
import math

import numpy as np
import torch
from torch import Tensor

from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_fc04.solver import (
    ANGLE, CHART_SEED_BASE, CHART_STARTS, DP12_MIN_CHORDAL,
    GAUSS_NEWTON_DAMPING, JAC_SINGULAR_TOL, LINE_SEARCH, MAX_UPDATES,
    PAIRWISE_MIN_CHORDAL, PASSES, REQUIRED_SHAMS, SOURCE_WRITE_REL_TOL,
    STEP_NORM_MAX, WIDTH, check_source_candidate, complementary_basis,
    minimum_norm_jacobian_step, normalized_source_residual,
    projector_digest, projectors_from_bases, source_tensor_profile,
    squared_constraint_excess, tangent_chart, target_values,
    unconstrained_parameter_count,
)


def parameter_receipt(y: Tensor) -> dict:
    values = y.detach().cpu().double().numpy().astype("<f8")
    binary = values.tobytes(order="C")
    return {
        "shape": list(values.shape), "dtype": "<f8",
        "sha256": hashlib.sha256(binary).hexdigest(),
        "b64_le_float64": base64.b64encode(binary).decode("ascii"),
    }


def _observation(p, prof, target, tol, dp, accepted):
    """Additional diagnostics; its result cannot affect the old search loop."""
    with torch.no_grad():
        residual = normalized_source_residual(prof, target, tol)
        return {
            "profile": [float(x.detach()) for x in prof],
            "normalized_residual": residual.detach().numpy().tolist(),
            "penalty": float(squared_constraint_excess(residual).detach()),
            "geometry_and_source": check_source_candidate(
                p, prof, target, tol, dp, accepted),
            "projector_sha256": projector_digest(p)["projector_sha256"],
        }


def fit_full_tangent_shams(
    activations: Tensor, physical_decoder: Tensor,
    clean_reads: Tensor, clean_gradients: Tensor,
    source_basis: list[Tensor], root: int,
    *, starts: int = CHART_STARTS, steps: int = MAX_UPDATES,
    want: int = REQUIRED_SHAMS,
):
    if (len(source_basis) != PASSES or
        activations.ndim != 3 or activations.shape[1:] != (PASSES, WIDTH) or
        clean_reads.shape != activations.shape or
        clean_gradients.shape != activations.shape or
        physical_decoder.shape != (WIDTH, WIDTH)
        or starts < 1 or steps < 0 or want < 1):
        raise ValueError("Original four-pass FC04 source record malformed")
    torch.set_num_threads(1)
    z = clean_reads.detach().cpu().double()
    g = clean_gradients.detach().cpu().double()
    a = activations.detach().cpu().double()
    U = physical_decoder.detach().cpu().double()
    E = [q.detach().cpu().double() for q in source_basis]
    if sum(x.shape[1] for x in E) != 12:
        raise ValueError("Only the prospective 12-slot source DP factor ranks allowed")
    C = [complementary_basis(x) for x in E]
    target, tol, moments, gram = target_values(z, g, a, U, E)
    dp = projectors_from_bases(E).detach()
    rng = np.random.default_rng(CHART_SEED_BASE + root)
    accepted = []
    records = []
    charts = []
    all_iterations = 0
    jacobian_rank_census = {0: 0, 1: 0, 2: 0, 3: 0}
    numerical_exception = None
    for chart_idx in range(starts):
        if numerical_exception is not None:
            charts.append({
                "chart": chart_idx, "status": "not_started_after_exception",
                "terminal_reason": "prior_numerical_exception",
                "updates": 0, "source_metric_evaluations": 0, "iterations": [],
            })
            continue
        if len(accepted) == want:
            charts.append({
                "chart": chart_idx, "status": "skipped_after_full",
                "updates": 0, "source_metric_evaluations": 0,
                "terminal_reason": "eight_accepted", "iterations": [],
            })
            continue
        chunks = [
            rng.standard_normal(((WIDTH - e.shape[1]) * e.shape[1])) /
            math.sqrt(WIDTH - e.shape[1]) for e in E
        ]
        y = torch.tensor(np.concatenate(chunks), dtype=torch.float64)
        initial_y = parameter_receipt(y)
        chart_success = False
        last = None
        attempted_updates = 0
        chart_evals = 0
        trace = []
        terminal_reason = "maximum_updates"
        try:
            for k in range(steps + 1):
                y_leaf = y.detach().clone().requires_grad_(True)
                bases = tangent_chart(E, C, y_leaf)
                p = projectors_from_bases(bases)
                prof = source_tensor_profile(z, g, bases, moments, gram)
                rec = check_source_candidate(p, prof, target, tol, dp, accepted)
                chart_evals += 1
                last = rec
                state = {
                    "iteration": k, "parameters": parameter_receipt(y_leaf),
                    **_observation(p, prof, target, tol, dp, accepted),
                    "jacobian": None, "trials": [],
                }
                trace.append(state)
                if rec["eligible"]:
                    record = {
                        **rec, **projector_digest(p.detach()),
                        "chart_index": chart_idx, "correction_updates": k,
                        "angle_from_dp12_radians": ANGLE,
                        "rank_tuple": [int(e.shape[1]) for e in E],
                        "full_tangent_parameter_count": len(y),
                        "source_observation_count": len(z),
                    }
                    accepted.append(p.detach().clone())
                    records.append(record)
                    terminal_reason = "admitted"
                    charts.append({
                        "chart": chart_idx, "status": "admitted",
                        "correction_updates": k,
                        "source_metric_evaluations": chart_evals,
                        "projector_sha256": record["projector_sha256"],
                        "initial_parameters": initial_y, "iterations": trace,
                        "terminal_reason": terminal_reason,
                    })
                    chart_success = True
                    break
                if k == steps:
                    break
                # These statements reproduce FC04's executed selection law.
                f = normalized_source_residual(prof, target, tol)
                current_penalty = float(squared_constraint_excess(f).detach())
                if current_penalty <= 1e-14:
                    terminal_reason = "excess_at_or_below_1e-14"
                    break
                delta, info = minimum_norm_jacobian_step(f, y_leaf)
                jacobian_rank_census[info["jacobian_rank"]] += 1
                all_iterations += 1
                attempted_updates += 1
                state["jacobian"] = dict(info)
                state["direction"] = parameter_receipt(delta)
                advanced = False
                for alpha in LINE_SEARCH:
                    y_new = y.detach() + alpha * delta
                    with torch.no_grad():
                        b_new = tangent_chart(E, C, y_new)
                        new_profile = source_tensor_profile(z, g, b_new, moments, gram)
                        new_f = normalized_source_residual(new_profile, target, tol)
                        new_penalty = float(squared_constraint_excess(new_f))
                    chart_evals += 1
                    advance = new_penalty < current_penalty - 1e-12
                    with torch.no_grad():
                        p_new = projectors_from_bases(b_new)
                        observation = _observation(
                            p_new, new_profile, target, tol, dp, accepted)
                    state["trials"].append({
                        "alpha": alpha, **observation,
                        "advance": advance,
                        "decision": "advance" if advance else "insufficient_descent",
                    })
                    if new_penalty < current_penalty - 1e-12:
                        y = y_new.detach()
                        advanced = True
                        break
                if not advanced:
                    terminal_reason = "no_accepted_descent_step"
                    break
        except Exception as exc:
            # Preserve the original exception as a terminal root failure.
            # No further chart or replacement is sampled after this event.
            numerical_exception = {
                "type": type(exc).__name__, "message": str(exc),
                "chart": chart_idx, "iteration": k,
            }
            terminal_reason = "numerical_exception"
            if not trace or trace[-1]["iteration"] != k:
                trace.append({"iteration": k, "parameters": parameter_receipt(y),
                              "jacobian": None, "trials": []})
            trace[-1]["exception"] = numerical_exception
        if not chart_success:
            charts.append({
                "chart": chart_idx,
                "status": ("numerical_exception" if numerical_exception is not None
                           else "ineligible_after_full_jacobian"),
                "correction_updates": attempted_updates,
                "source_metric_evaluations": chart_evals,
                "last_source_errors": {
                    name: last[name] for name in (
                        "physical_rms_relative_error", "signed_score_abs_error",
                        "score_rms_abs_error", "distance_from_dp12",
                        "minimum_pairwise_distance",
                    )
                } if last is not None else None,
                "initial_parameters": initial_y, "iterations": trace,
                "terminal_reason": terminal_reason,
            })
    result = {
        "eligible": len(accepted) == want and numerical_exception is None,
        "reason": "eight_full_tangent_distinct_matches" if len(accepted) == want
                  else "insufficient_full_tangent_source_feasible_shams",
        "root": root, "selected": len(accepted), "qualified": len(accepted),
        "charts_configured": starts, "max_corrections_per_chart": steps,
        "line_search_scales": list(LINE_SEARCH),
        "minimum_norm_step_limit": STEP_NORM_MAX,
        "jacobian_damping": GAUSS_NEWTON_DAMPING,
        "root_seed": CHART_SEED_BASE + root,
        "fixed_angle_radians": ANGLE,
        "minimum_DP12_chordal": DP12_MIN_CHORDAL,
        "minimum_pairwise_chordal": PAIRWISE_MIN_CHORDAL,
        "full_tangent_parameter_count": unconstrained_parameter_count(E),
        "source_write_relative_tolerance": SOURCE_WRITE_REL_TOL,
        "source_signed_score_abs_tolerance": tol[1],
        "source_score_rms_abs_tolerance": tol[2],
        "target_source_write_rms": target[0],
        "target_source_signed_score_mean": target[1],
        "target_source_score_rms": target[2],
        "source_observation_count": len(z), "projectors": records,
        "charts": charts,
        "charts_actually_started": sum(q["status"] not in (
            "skipped_after_full", "not_started_after_exception") for q in charts),
        "total_jacobian_corrections": all_iterations,
        "jacobian_rank_census": {str(k): v for k, v in jacobian_rank_census.items()},
        "uses_external_task_labels": False,
        "trace_schema": "FC05_EXECUTED_FC04_OBSERVATIONS_V1",
        "numerical_exception": numerical_exception,
    }
    return accepted, result
