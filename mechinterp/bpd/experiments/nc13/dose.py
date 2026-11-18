"""Frozen NC-13 physical-dose controls.

The delivered-displacement search uses target forwards supplied by its caller.
It assumes no global monotonicity and never substitutes an unmatched gain.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from torch import Tensor


GAIN_LOWER = 0.5
GAIN_UPPER = 1.5
GRID_SIZE = 17
BISECTION_STEPS = 24
MATCH_REL_TOL = 1e-4
MIN_REFERENCE = 1e-12
SHAM_PROPOSALS = 2048
SHAMS = 8
SHAM_REL_TOL = 0.15
ANGLES = (0.45, 0.65, 0.85, 1.05)


def _physical_arrays(acts: Tensor, u: Tensor, p: Tensor):
    if acts.ndim != 3 or not len(acts):
        raise ValueError("Expected nonempty [input,pass,read] activations")
    _, passes, width = acts.shape
    if u.ndim != 2 or u.shape[0] != width or not u.shape[1]:
        raise ValueError("Write matrix must have shape [read,output]")
    if p.shape != (passes, width, width):
        raise ValueError("Projectors must have shape [pass,read,read]")
    a, write, project = acts.detach().double(), u.detach().double(), p.detach().double()
    if not all(bool(torch.isfinite(x).all()) for x in (a, write, project)):
        raise ValueError("Physical arrays must be finite")
    return a, write, project


@torch.no_grad()
def dose_metrics(acts: Tensor, u: Tensor, p: Tensor, gain: float = 1.0) -> dict:
    """Exact clean-trajectory physical-write RMS and displacement.

    The returned clean displacement uses clean prewrite activations throughout.
    Actual delivered closed-loop displacement requires a native rollout.
    """
    if not math.isfinite(float(gain)):
        raise ValueError("Gain must be finite")
    a, write, project = _physical_arrays(acts, u, p)
    clean = a @ write
    retained = torch.einsum("nti,tij,jk->ntk", a, project, write)
    gained = float(gain) * retained
    delta = gained - clean
    return {
        "gain": float(gain),
        "retained_rms": float(gained.square().mean().sqrt()),
        "clean_displacement_rms": float(delta.square().mean().sqrt()),
        "clean_write_rms": float(clean.square().mean().sqrt()),
        "retained_rms_by_pass": gained.square().mean((0, 2)).sqrt().tolist(),
        "clean_displacement_rms_by_pass": delta.square().mean((0, 2)).sqrt().tolist(),
        "clean_write_rms_by_pass": clean.square().mean((0, 2)).sqrt().tolist(),
        "quadratic_moments": {
            "a": float(retained.square().mean()),
            "b": float((retained * clean).mean()),
            "c": float(clean.square().mean()),
        },
    }


def energy_gain(reference: float, acts: Tensor, u: Tensor, p: Tensor) -> dict:
    """NC-13 policy E: match retained clean write RMS with a bounded gain."""
    observed = dose_metrics(acts, u, p)["retained_rms"]
    rec = {
        "policy": "E",
        "admitted": False,
        "gain": None,
        "raw_gain": None,
        "reason": None,
        "observed_retained_rms": observed,
        "matched_retained_rms": None,
        "reference_retained_rms": float(reference) if math.isfinite(float(reference)) else None,
        "gain_bounds": [GAIN_LOWER, GAIN_UPPER],
    }
    if not math.isfinite(float(reference)) or reference <= MIN_REFERENCE:
        rec["reason"] = "INVALID_REFERENCE"
        return rec
    if observed <= MIN_REFERENCE:
        rec["reason"] = "ZERO_RETAINED_WRITE"
        return rec
    gain = float(reference) / observed
    rec["raw_gain"] = gain if math.isfinite(gain) else None
    if not math.isfinite(gain) or not GAIN_LOWER <= gain <= GAIN_UPPER:
        rec["reason"] = "GAIN_OUTSIDE_REGISTERED_BOUNDS"
        return rec
    rec.update(admitted=True, gain=gain, reason="MATCHED", matched_retained_rms=gain * observed)
    return rec


def calibrate_closed_displacement(evaluate: Callable[[float], float], reference: float) -> dict:
    """Frozen 17-point grid and, when needed, 24-step sign-bracket search.

    Every invocation of evaluate is recorded. The callback returns actual
    delivered write RMS on the fixed label-free dose pool. A failed search
    supplies gain=None and does not assert global geometric infeasibility.
    """
    rec = {
        "policy": "D",
        "admitted": False,
        "gain": None,
        "reason": None,
        "reference_displacement_rms": float(reference) if math.isfinite(float(reference)) else None,
        "matched_displacement_rms": None,
        "relative_error": None,
        "queries": [],
        "rollout_calls": 0,
        "grid_size": GRID_SIZE,
        "bisection_steps": 0,
        "maximum_rollout_calls": GRID_SIZE + BISECTION_STEPS,
        "selected_bracket": None,
        "final_bracket": None,
        "gain_bounds": [GAIN_LOWER, GAIN_UPPER],
        "relative_tolerance": MATCH_REL_TOL,
    }
    if not math.isfinite(float(reference)) or reference <= MIN_REFERENCE:
        rec["reason"] = "INVALID_REFERENCE"
        return rec

    def query(gain: float, stage: str):
        value = float(evaluate(float(gain)))
        valid = math.isfinite(value) and value >= 0.0
        row = {
            "index": len(rec["queries"]),
            "stage": stage,
            "gain": float(gain),
            "displacement": value if valid else None,
            "relative_error": abs(value - reference) / reference if valid else None,
            "finite_nonnegative": valid,
        }
        if not valid:
            row["invalid_value"] = repr(value)
        rec["queries"].append(row)
        rec["rollout_calls"] += 1
        return row

    def finish_if_matched():
        accepted = [r for r in rec["queries"]
                    if r["finite_nonnegative"] and r["relative_error"] <= MATCH_REL_TOL]
        if not accepted:
            return False
        chosen = min(accepted, key=lambda r: (abs(r["gain"] - 1.0), r["gain"]))
        rec.update(admitted=True, gain=chosen["gain"], reason="MATCHED",
                   matched_displacement_rms=chosen["displacement"],
                   relative_error=chosen["relative_error"],
                   selected_query_index=chosen["index"])
        return True

    grid = [query(float(gain), "grid")
            for gain in np.linspace(GAIN_LOWER, GAIN_UPPER, GRID_SIZE)]
    if not all(r["finite_nonnegative"] for r in grid):
        rec["reason"] = "INVALID_DISPLACEMENT"
        return rec
    if finish_if_matched():
        return rec
    brackets = [(left, right) for left, right in zip(grid[:-1], grid[1:])
                if (left["displacement"] - reference) * (right["displacement"] - reference) < 0]
    if not brackets:
        rec["reason"] = "NO_REGISTERED_BRACKET"
        return rec
    left, right = min(brackets, key=lambda pair: (
        abs((pair[0]["gain"] + pair[1]["gain"]) / 2.0 - 1.0), pair[0]["gain"]))
    rec["selected_bracket"] = [left["gain"], right["gain"]]
    invalid = False
    for _ in range(BISECTION_STEPS):
        middle = query((left["gain"] + right["gain"]) / 2.0, "bisection")
        rec["bisection_steps"] += 1
        if not middle["finite_nonnegative"]:
            invalid = True
            continue
        if ((left["displacement"] - reference) *
                (middle["displacement"] - reference)) <= 0:
            right = middle
        else:
            left = middle
    rec["final_bracket"] = [left["gain"], right["gain"]]
    if invalid:
        rec["reason"] = "INVALID_DISPLACEMENT"
    elif not finish_if_matched():
        rec["reason"] = "MATCH_TOLERANCE_FAILED"
    return rec


def select_shams(acts: Tensor, u: Tensor, basis: Tensor,
                 reference_E: float, root: int) -> tuple[list[Tensor], dict]:
    """First eight common-energy matches from the frozen 2048 local proposals."""
    if basis.shape != (4, 20, 4):
        raise ValueError("NC-13 sham basis must have shape [4,20,4]")
    if type(root) is not int:
        raise ValueError("Sham root must be an integer")
    if acts.ndim != 3 or acts.shape[1:] != (4, 20) or not len(acts):
        raise ValueError("NC-13 dose activations must have shape [N,4,20]")
    if u.shape != (20, 20):
        raise ValueError("NC-13 physical write matrix must have shape [20,20]")
    arrays = [x.detach().double().cpu().numpy() for x in (acts, u, basis)]
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Sham source arrays must be finite")
    activations, write, e = arrays
    if not np.allclose(e.transpose(0, 2, 1) @ e, np.eye(4), atol=1e-8, rtol=0):
        raise ValueError("Sham source basis must be orthonormal")
    rec = {
        "admitted": False,
        "reason": None,
        "proposals": SHAM_PROPOSALS,
        "evaluated_proposals": 0,
        "qualified": 0,
        "selected": 0,
        "selected_indices": [],
        "proposal_indices": [],
        "proposal_retained_rms": [],
        "proposal_angles": [],
        "reference_retained_rms": float(reference_E) if math.isfinite(float(reference_E)) else None,
        "relative_tolerance": SHAM_REL_TOL,
        "generator_seed": root * 1009 + 13013,
        "rejection_counts": {"outside_energy_tolerance": 0},
    }
    if not math.isfinite(float(reference_E)) or reference_E <= MIN_REFERENCE:
        rec["reason"] = "INVALID_REFERENCE"
        return [], rec
    moment = np.einsum("nti,ntj->tij", activations, activations) / len(activations)
    gram = write @ write.T
    rng = np.random.default_rng(rec["generator_seed"])
    chosen = []
    for ix in range(SHAM_PROPOSALS):
        theta = ANGLES[ix % len(ANGLES)]
        perturb = np.empty_like(e)
        for t in range(4):
            z = rng.standard_normal((20, 4))
            z = z - e[t] @ (e[t].T @ z)
            complement, _ = np.linalg.qr(z, mode="reduced")
            perturb[t] = math.cos(theta) * e[t] + math.sin(theta) * complement
        energy = sum(float(np.trace((v.T @ moment[t] @ v) @ (v.T @ gram @ v)))
                     for t, v in enumerate(perturb))
        retained = math.sqrt(max(energy, 0.0) / 80.0)
        rec["proposal_indices"].append(ix)
        rec["proposal_retained_rms"].append(retained)
        rec["proposal_angles"].append(theta)
        rec["evaluated_proposals"] += 1
        if abs(retained - reference_E) / reference_E <= SHAM_REL_TOL:
            rec["qualified"] += 1
            if len(chosen) < SHAMS:
                chosen.append(torch.from_numpy(perturb @ perturb.transpose(0, 2, 1)))
                rec["selected_indices"].append(ix)
        else:
            rec["rejection_counts"]["outside_energy_tolerance"] += 1
    rec["selected"] = len(chosen)
    rec["admitted"] = len(chosen) == SHAMS
    rec["reason"] = "MATCHED" if rec["admitted"] else "INSUFFICIENT_ENERGY_MATCHED_SHAMS"
    return chosen, rec
