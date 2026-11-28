"""Whole-root archive compression and transport analysis in small physical write bases."""
from __future__ import annotations

from itertools import combinations
from statistics import fmean

import numpy as np

from archive_inputs import Bank, require
from controls import best_atom_subset
from transport import Disagreement, direction, subspace_angles
from weighted_rank import optimal_sample_rank, truncated_svd


class CompactWrite:
    """Orthogonal output coordinates for the fixed T2 target and reapproximations."""
    def __init__(self, bank):
        self.output, u = np.linalg.qr(bank.u.T, mode="reduced")
        self.operator = u @ bank.v.T

    def lift(self, operator):
        u, s, vt = np.linalg.svd(operator, full_matrices=False)
        rank = int(np.linalg.matrix_rank(operator))
        return Bank(vt[:rank].T * s[:rank], (self.output @ u[:, :rank]).T)


def contrast(rows):
    return rows[:, 1] - rows[:, 0]


def train_candidates(target, banks, rows, rank, rng_basis, floor, fixed_reader=None):
    width = rows.shape[-1]
    x = rows.reshape(-1, width)
    compact = target.operator
    values = x @ compact.T @ target.output.T
    reach = optimal_sample_rank(compact, x, rank, rtol=floor)
    paired = optimal_sample_rank(compact, contrast(rows), rank, rtol=floor)
    z = rng_basis[:, :rank]
    policies = {
        "svd": target.lift(truncated_svd(compact, rank)),
        "reach": target.lift(reach.operator),
        "contrast": target.lift(paired.operator),
        "random": target.lift(z @ (z.T @ compact)),
    }
    for name, bank in banks.items():
        keep = best_atom_subset(bank.atoms(x), values, rank)
        policies[name] = Bank(bank.v[:, keep], bank.u[list(keep)])
    if fixed_reader is not None:
        fixed = optimal_sample_rank(compact, x, rank, fixed_reader, rtol=floor)
        policies["reader"] = target.lift(fixed.operator)
    for method, bank in policies.items():
        require(
            bank.v.shape[1] == bank.u.shape[0] == rank
            and np.linalg.matrix_rank(bank.v) == np.linalg.matrix_rank(bank.u) == rank,
            f"CAL comparator {method} failed the common rank",
        )
    return policies, {
        "reached_rank": reach.input_support_rank,
        "contrast_rank": paired.input_support_rank,
        "reached_spectral_tail_error_abs": abs(reach.error - reach.tail_oracle),
        "contrast_spectral_tail_error_abs": abs(paired.error - paired.tail_oracle),
    }


def heldout_score(x, bank, target, reader, dose_fraction, zero_atol):
    truth = target.apply(x)
    prediction = bank.apply(x)
    delta = truth[1] - truth[0]
    estimate = prediction[1] - prediction[0]
    dose = dose_fraction * np.linalg.norm(delta)
    length = np.linalg.norm(estimate)
    total = float(np.sum(truth ** 2) / len(x))
    errors = truth - prediction
    result = {
        "write_mse": float(np.sum(errors ** 2) / len(x)),
        "truth_write_mse": total,
        "contrast_write_sq_error": float(np.sum((delta - estimate) ** 2)),
    }
    if reader is None:
        return result
    measured = reader(np.column_stack((errors.T, truth.T, delta, estimate)))
    err, ref = measured[:, :2], measured[:, 2:4]
    normalized = measured[:, 5] * (dose / length) if length > zero_atol else np.zeros_like(measured[:, 5])
    result.update({
        "future_mse": float(np.sum(err ** 2) / len(x)),
        "truth_future_mse": float(np.sum(ref ** 2) / len(x)),
        "physical_status": "matched" if length > zero_atol and dose > zero_atol else "zero_write_or_reference",
        "requested_dose": float(dose),
        "constructed_physical_dose": float(np.linalg.norm(estimate * (dose / length))) if length > zero_atol else 0.0,
        "future_direction": direction(normalized, dose_fraction * measured[:, 4], zero_atol=zero_atol),
    })
    return result


def score_mean(rows):
    result = {}
    for field in ("write_mse", "truth_write_mse", "contrast_write_sq_error", "future_mse", "truth_future_mse"):
        if field in rows[0]:
            result[field] = fmean(r[field] for r in rows)
    for field in ("write", "future"):
        ref = result.get(f"truth_{field}_mse")
        if ref:
            result[f"{field}_relative_mse"] = result[f"{field}_mse"] / ref
    if "future_direction" in rows[0]:
        result["max_constructed_dose_error"] = max(
            (abs(r["constructed_physical_dose"] - r["requested_dose"]) for r in rows if r["physical_status"] == "matched"),
            default=None,
        )
        result["unmatched_doses"] = sum(r["physical_status"] != "matched" for r in rows)
        for name in ("cosine", "gain"):
            values = [r["future_direction"][name] for r in rows if r["future_direction"][name] is not None]
            result[f"mean_{name}_when_defined"] = fmean(values) if values else None
            result[f"n_{name}_defined"] = len(values)
        result["negative_gain_roots"] = sum((r["future_direction"]["gain"] or 0) < 0 for r in rows)
    return result


def split_root(occurrence, heldout):
    train = np.arange(len(occurrence.x)) != heldout
    return occurrence.x[train], occurrence.carrier[train].mean(axis=(0, 1), dtype=np.float64)


def calibrate(occurrence, banks, target, k_reader, ranks, spec, check):
    rank_k = len(target.operator)
    rng = np.random.default_rng(spec["archive_random_seed"])
    random = np.linalg.qr(rng.normal(size=(rank_k, max(ranks))), mode="reduced")[0]
    root_results = {rank: [] for rank in ranks}
    fit_diagnostics = {rank: [] for rank in ranks}
    output_bases = {rank: [] for rank in ranks}
    for heldout in range(len(occurrence.x)):
        train, mean_carrier = split_root(occurrence, heldout)

        def measure(writes):
            return k_reader(mean_carrier, writes)

        compressed_reader = None if k_reader is None else measure(target.output)
        for rank in ranks:
            check()
            policies, info = train_candidates(
                target, banks, train, rank, random, spec["support_relative_cutoff"], compressed_reader,
            )
            scored = {name: heldout_score(
                occurrence.x[heldout], bank, banks["t2"], measure if k_reader is not None else None,
                spec["archive_direction_dose_fraction"], spec["direction_zero_atol"],
            ) for name, bank in policies.items()}
            root_results[rank].append(scored)
            fit_diagnostics[rank].append(info)
            output_bases[rank].append({name: bank.u.T for name, bank in policies.items()})
    summaries = {}
    for rank in ranks:
        rows = root_results[rank]
        result = {}
        for method in rows[0]:
            result[method] = score_mean([r[method] for r in rows])
            result[method]["max_pairwise_output_subspace_angle_deg"] = float(max(
                np.degrees(subspace_angles(a[method], b[method]).max())
                for a, b in combinations(output_bases[rank], 2)
            ))
            result[method]["roots_beating_svd_write_error"] = sum(
                r[method]["write_mse"] < r["svd"]["write_mse"] for r in rows
            )
            if k_reader is not None:
                result[method]["roots_beating_svd_future_error"] = sum(
                    r[method]["future_mse"] < r["svd"]["future_mse"] for r in rows
                )
        info = fit_diagnostics[rank]
        summaries[str(rank)] = {
            "roots": len(rows),
            "methods": result,
            "training_reached_support_ranks": sorted({r["reached_rank"] for r in info}),
            "training_contrast_support_ranks": sorted({r["contrast_rank"] for r in info}),
            "max_reached_spectral_tail_error_abs": max(r["reached_spectral_tail_error_abs"] for r in info),
            "max_contrast_spectral_tail_error_abs": max(r["contrast_spectral_tail_error_abs"] for r in info),
        }
    return summaries, root_results


def compact_closure(states, reference, compact):
    report = {}
    for p, occurrence in states.items():
        direct = reference.apply(occurrence.x)
        reduced = (occurrence.x @ compact.operator.T) @ compact.output.T
        atoms = reference.atoms(occurrence.x).sum(axis=-2)
        delta = np.linalg.norm(reduced - direct) / np.linalg.norm(direct)
        atom_delta = np.linalg.norm(atoms - direct) / np.linalg.norm(direct)
        require(max(delta, atom_delta) < 1e-11, "T2 physical factor sum did not close")
        report[str(p)] = {
            "reference_factor_sum_relative_l2": float(atom_delta),
            "compact_recomposition_relative_l2": float(delta),
            "target_over_archived_native_write_energy": float(
                np.sum(direct ** 2) / np.sum(occurrence.native.astype(np.float64) ** 2)
            ),
        }
    return report


def outside_support(source, target, cutoff):
    x, y = source.reshape(-1, source.shape[-1]), target.reshape(-1, target.shape[-1])
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    support = vt[singular > np.sqrt(cutoff) * singular[0]]
    missed = y - (y @ support.T) @ support
    return float(np.sum(missed ** 2) / np.sum(y ** 2))


def transfer(a, b, xa, xb, cutoff):
    aa, ab = a.apply(xa), a.apply(xb)
    ba, bb = b.apply(xa), b.apply(xb)
    op = ((ba + bb) - (aa + ab)) / 2
    state = ((bb + ab) - (ba + aa)) / 2
    residual = op + state - (bb - aa)
    close = np.linalg.norm(residual) / max(np.linalg.norm(bb - aa), np.linalg.norm(bb), 1.)
    require(close < 1e-11, "symmetric operator/state decomposition failed closure")
    energies = [Disagreement(o, d, np.zeros_like(o)).energies() for o, d in zip(op, state, strict=True)]
    return {
        "mean_root_energy": {k: fmean(r[k] for r in energies) for k in energies[0] if "reader" not in k},
        "total_below_sum_due_to_cancellation": sum(r["operator_state_cross"] < 0 for r in energies),
        "input_energy_b_outside_a_empirical_contrast_support": outside_support(xa, xb, cutoff),
        "input_energy_a_outside_b_empirical_contrast_support": outside_support(xb, xa, cutoff),
        "identity_relative_l2": float(close),
    }


def archive_evalmath(archive_roots, planner):
    def score(record):
        outcome = record["future_direction"]
        if outcome["status"] == "zero_reference":
            raise ValueError("matched-dose archive directional reference is unavailable")
        return 0.0 if outcome["status"] == "zero_observed" else outcome["cosine"]

    rows = []
    for i, row in enumerate(archive_roots):
        svd = score(row["svd"])
        rows.append({
            "unit_id": f"cal-{i:03d}",
            "paired_effect": f"{(score(row['reader']) - svd) / 2:.12f}",
            "sham_effects": [f"{(score(row['random']) - svd) / 2:.12f}"],
        })
    return {
        "schema_version": 1,
        "experiment_id": "RT-BPD-02 archive BF16 pass-1 rank-2 matched-direction descriptor",
        "independent_unit": "selection-exposed CAL root, complete A/B pair",
        "alpha": planner["alpha"],
        "scientific_min_effect": planner["scientific_min_effect"],
        "target_effect": planner["target_effect"],
        "measurement_resolution": planner["measurement_resolution"],
        "sham_quantile": planner["quantile"],
        "sham_confidence": planner["confidence"],
        "planned_comparisons": planner["planned_comparisons"],
        "confirm_units_planned": planner["nominal_planning_roots"],
        "pilot_from_development_only": True,
        "heldout_sealed": False,
        "units_independent": False,  # Reused CAL and heavily overlapping training folds.
        "pilot": rows,
    }
