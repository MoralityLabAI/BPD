"""Registered six-dimensional tied recurrence with paired-root development."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from controls import best_atom_subset
from transport import delayed_dark_rank, direction, disagreement, dominance_constant, fixed_future_reader, subspace_angles
from weighted_rank import optimal_sample_rank, truncated_svd, uncentered_second_moment


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Gym:
    basis: np.ndarray
    operator: np.ndarray
    background: np.ndarray
    scale: float
    current: np.ndarray
    delayed: np.ndarray
    random_output: np.ndarray

    def step(self, states, edit=None):
        live, stage = states[..., 0, :], states[..., 1, :]
        writes = live @ self.operator.T
        if edit is not None:
            writes = writes + edit
        return np.stack((live @ self.background.T + self.scale * writes, writes, stage), axis=-2)

    def reaches(self, roots, passes):
        states = np.stack((roots, np.zeros_like(roots), np.zeros_like(roots)), axis=-2)
        result = {}
        for occurrence in range(1, max(passes) + 1):
            if occurrence in passes:
                result[occurrence] = states
            states = self.step(states)
        return result

    def future_effect(self, before, edit):
        base = self.step(self.step(before))
        patched = self.step(self.step(before, edit))
        return (patched[..., 2, :] - base[..., 2, :]) @ self.delayed.T


def build(spec: dict) -> Gym:
    rng = np.random.default_rng(spec["geometry_seed"])
    n = len(spec["spectrum"])
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    q *= np.sign(np.diag(r))
    w = (q * spec["spectrum"]) @ q.T
    f = (q * spec["background_diagonal"]) @ q.T
    random, _ = np.linalg.qr(rng.normal(size=(n, spec["rank"])), mode="reduced")
    return Gym(q, w, f, spec["recurrent_write_scale"], q[:, :2].T,
               np.asarray(spec["delayed_reader_mix"]) @ q[:, 2:4].T, random)


def make_roots(spec: dict, gym: Gym, split: int, count: int, scales: list[float]) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([spec["root_seed"], split]))
    common, diff = rng.normal(size=(2, count, len(scales))) * scales
    planted = np.stack((common - diff, common + diff), axis=1)
    return planted @ gym.basis.T


def reader_maps(gym: Gym) -> tuple[np.ndarray, np.ndarray]:
    n = len(gym.operator)
    zero, eye = np.zeros((n, n)), np.eye(n)
    j = np.block([
        [gym.background + gym.scale * gym.operator, zero, zero],
        [gym.operator, zero, zero],
        [zero, eye, zero],
    ])
    b = np.vstack((gym.scale * eye, eye, zero))
    current = np.hstack((np.zeros_like(gym.current), gym.current, np.zeros_like(gym.current)))
    delayed = np.hstack((np.zeros_like(gym.delayed), np.zeros_like(gym.delayed), gym.delayed))
    now = fixed_future_reader(b, (), (current,), (1.,))
    later = fixed_future_reader(b, (j,), (np.zeros((0, 3 * n)), delayed), (0., 1.))
    return now, later


def fit(gym: Gym, training: np.ndarray, now, later, spec: dict):
    w, q, rank = gym.operator, gym.basis, spec["rank"]
    rows = training[..., 0, :].reshape(-1, w.shape[1])
    readouts = np.vstack((now, later)) / np.sqrt(2)
    atom_writes = (rows @ q * spec["spectrum"])[..., None] * q.T
    best = best_atom_subset(atom_writes @ readouts.T, rows @ w.T @ readouts.T, rank)
    bpd_mask = (q[:, best] * np.asarray(spec["spectrum"])[list(best)]) @ q[:, best].T
    reach_fit = optimal_sample_rank(w, rows, rank)
    reader_fit = optimal_sample_rank(w, rows, rank, later)
    random = gym.random_output
    operators = {
        "svd": truncated_svd(w, rank),
        "reach": reach_fit.operator,
        "reader": reader_fit.operator,
        "bpd_static": bpd_mask,
        "random": random @ (random.T @ w),
    }
    return operators, {
        "bpd_selected_atoms_zero_based": list(best),
        "reach_fit_loss": reach_fit.error,
        "reach_spectral_tail": reach_fit.tail_oracle,
        "reader_fit_loss": reader_fit.error,
        "reader_spectral_tail": reader_fit.tail_oracle,
    }


def score_root(gym, candidate, before, now, later, dose_fraction, atol):
    x = before[:, 0, :]
    writes = x @ gym.operator.T
    estimates = x @ candidate.T
    errors = writes - estimates
    truth = writes[1] - writes[0]
    contrast = estimates[1] - estimates[0]
    dose = dose_fraction * np.linalg.norm(truth)
    if dose <= atol:
        raise ValueError("registered synthetic donor has zero physical contrast")
    norm = np.linalg.norm(contrast)
    proposed = contrast * (dose / norm) if norm > atol else np.zeros_like(contrast)
    reference_write = dose_fraction * truth
    ref = gym.future_effect(before[0], reference_write)
    actual = gym.future_effect(before[0], proposed)
    closure = float(max(np.max(np.abs(ref - later @ reference_write)), np.max(np.abs(actual - later @ proposed))))
    measured_write = gym.step(before[0], proposed)[1] - gym.step(before[0])[1]
    dose_realized = float(np.linalg.norm(measured_write))
    physical_status = "matched" if norm > atol else "zero_write"
    if closure > atol or (physical_status == "matched" and abs(dose_realized - dose) > atol):
        raise ValueError("finite continuation or physical dose did not close")
    outcome = direction(actual, ref, zero_atol=atol)
    return {
        "write_mse": float(np.mean(np.sum(errors ** 2, axis=-1))),
        "current_mse": float(np.mean(np.sum((errors @ now.T) ** 2, axis=-1))),
        "future_mse": float(np.mean(np.sum((errors @ later.T) ** 2, axis=-1))),
        "contrast_write_sq_error": float(np.sum((truth - contrast) ** 2)),
        "truth_write_mse": float(np.mean(np.sum(writes ** 2, axis=-1))),
        "truth_future_mse": float(np.mean(np.sum((writes @ later.T) ** 2, axis=-1))),
        "requested_dose": float(dose),
        "physical_dose": dose_realized,
        "physical_status": physical_status,
        "reader_closure_max_abs": closure,
        "future_direction": outcome,
    }


def summarize(records: list[dict], methods: list[str]) -> dict:
    panes = {}
    for panel in sorted({r["panel"] for r in records}):
        passes = {}
        for occurrence in sorted({r["pass"] for r in records}):
            group = [r["methods"] for r in records if r["panel"] == panel and r["pass"] == occurrence]
            comparison = {}
            for method in methods:
                subset = [r[method] for r in group]
                means = {key: fmean(r[key] for r in subset) for key in (
                    "write_mse", "current_mse", "future_mse", "contrast_write_sq_error",
                    "truth_write_mse", "truth_future_mse",
                )}
                means["write_relative_mse"] = means["write_mse"] / means["truth_write_mse"]
                means["future_relative_mse"] = means["future_mse"] / means["truth_future_mse"]
                means["max_reader_closure_abs"] = max(r["reader_closure_max_abs"] for r in subset)
                means["max_matched_dose_error"] = max(
                    (abs(r["requested_dose"] - r["physical_dose"]) for r in subset if r["physical_status"] == "matched"),
                    default=None,
                )
                means["unmatched_writes"] = sum(r["physical_status"] != "matched" for r in subset)
                for field in ("gain", "cosine"):
                    values = [r["future_direction"][field] for r in subset if r["future_direction"][field] is not None]
                    means[f"mean_{field}_when_defined"] = fmean(values) if values else None
                    means[f"n_{field}_defined"] = len(values)
                means["negative_gain_roots"] = sum((r["future_direction"]["gain"] or 0) < 0 for r in subset)
                comparison[method] = means
            passes[str(occurrence)] = {"root_count": len(group), "methods": comparison}
        panes[panel] = passes
    primary = [r["methods"] for r in records if r["panel"] == "aligned" and r["pass"] == 2]
    paired = [r["svd"]["future_mse"] - r["reader"]["future_mse"] for r in primary]
    return {"panels": panes, "primary_paired_svd_minus_reader": {
        "mean": fmean(paired), "min": min(paired), "max": max(paired),
        "positive_roots": sum(x > 0 for x in paired), "roots": len(paired),
    }}


def evalmath_input(records: list[dict], spec: dict) -> dict:
    def as_score(record):
        measured = record["future_direction"]
        if measured["status"] == "zero_reference":
            raise ValueError("reference contrast has no direction for the registered endpoint")
        return 0.0 if measured["status"] == "zero_observed" else measured["cosine"]

    rows = []
    for r in records:
        if r["panel"] == "aligned" and r["pass"] == 2:
            outcomes = r["methods"]
            baseline = as_score(outcomes["svd"])
            rows.append({
                "unit_id": r["root_id"],
                "paired_effect": f"{(as_score(outcomes['reader']) - baseline) / 2:.12f}",
                "sham_effects": [f"{(as_score(outcomes['random']) - baseline) / 2:.12f}"],
            })
    return {
        "schema_version": 1,
        "experiment_id": "RT-BPD-02 synthetic matched-dose direction",
        "independent_unit": "paired development source root",
        "alpha": spec["alpha"],
        "scientific_min_effect": spec["scientific_min_effect"],
        "target_effect": spec["target_effect"],
        "measurement_resolution": spec["measurement_resolution"],
        "sham_quantile": spec["quantile"],
        "sham_confidence": spec["confidence"],
        "planned_comparisons": spec["planned_comparisons"],
        "confirm_units_planned": spec["nominal_planning_roots"],
        "pilot_from_development_only": True,
        "heldout_sealed": spec["heldout_sealed"],
        "units_independent": True,
        "pilot": rows,
    }


def run(registration: dict) -> tuple[dict, list[dict], dict]:
    spec = registration["synthetic"]
    gym = build(spec)
    passes = spec["occurrence_passes"]
    now, later = reader_maps(gym)
    training = make_roots(spec, gym, 0, spec["train_roots"], spec["initial_reach_scale"])
    train = gym.reaches(training, passes)
    policies, training_diagnostics = {}, {}
    for p in passes:
        policies[p], audit = fit(gym, train[p], now, later, spec)
        halves = [fit(gym, x, now, later, spec)[0] for x in np.split(train[p], 2)]
        audit["column_rank"] = {k: int(np.linalg.matrix_rank(v)) for k, v in policies[p].items()}
        if list(policies[p]) != spec["methods"] or any(r != spec["rank"] for r in audit["column_rank"].values()):
            raise ValueError("a registered comparator failed the matched rank")
        audit["half_cal_max_output_angle_deg"] = {
            k: float(np.degrees(subspace_angles(halves[0][k], halves[1][k]).max()))
            for k in policies[p]
        }
        training_diagnostics[str(p)] = audit

    records, decomposition = [], {}
    for panel, scale in zip(spec["branches"], (spec["initial_reach_scale"], spec["shifted_development_scale"]), strict=True):
        roots = make_roots(spec, gym, 1, spec["development_roots"], scale)
        reached = gym.reaches(roots, passes)
        paired = {p: reached[p][:, 1, 0, :] - reached[p][:, 0, 0, :] for p in passes}
        a, b = passes
        moments = {p: uncentered_second_moment(reached[p][..., 0, :].reshape(-1, len(gym.basis))) for p in passes}
        decomposition[panel] = {
            "empirical_pass2_over_pass1_psd_bound": dominance_constant(moments[a], moments[b]),
            "mean_contrast_energies": {},
        }
        for method, ka, kb in [
            ("tied_target", gym.operator, gym.operator),
            *((m, policies[a][m], policies[b][m]) for m in spec["methods"]),
        ]:
            pieces = [disagreement(ka, kb, x, y).energies() for x, y in zip(paired[a], paired[b], strict=True)]
            decomposition[panel]["mean_contrast_energies"][method] = {
                field: fmean(r[field] for r in pieces) for field in pieces[0]
            }
        for p in passes:
            for i, before in enumerate(reached[p]):
                outcomes = {
                    m: score_root(gym, candidate, before, now, later, spec["physical_dose"],
                                  spec["numerical_closure_atol"])
                    for m, candidate in policies[p].items()
                }
                records.append({"panel": panel, "pass": p, "root_id": f"dev-{i:03d}", "methods": outcomes})
    summary = summarize(records, spec["methods"])
    summary.update({
        "schema": "rtbpd02-synthetic/v1",
        "split_counts": {"train": spec["train_roots"], "development": spec["development_roots"]},
        "delayed_dark_rank": delayed_dark_rank(now, later, np.eye(len(gym.basis))),
        "fit_diagnostics": training_diagnostics,
        "pass_transport": decomposition,
    })
    return summary, records, evalmath_input(records, registration["evalmath"])


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registration = (HERE / "EXTENSION_REGISTRATION.json").read_bytes()
    summary, roots, planner = run(json.loads(registration))
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    root_path = output / "synthetic_roots.json"
    write_json(root_path, roots)
    summary["registration_sha256"] = sha256(registration).hexdigest()
    summary["synthetic_roots_sha256"] = sha256(root_path.read_bytes()).hexdigest()
    write_json(output / "synthetic_summary.json", summary)
    write_json(output / "evalmath_input.json", planner)
    print(json.dumps(summary["primary_paired_svd_minus_reader"], sort_keys=True))


if __name__ == "__main__":
    main()
