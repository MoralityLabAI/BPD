"""BPD-SUB-03: source-locked strong-null calibration of causal observability.

PROTOCOL.md was committed first on 2026-11-18 before source or 64-root data.
64 fresh models are for development/calibration only; no confirmation phase exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mechinterp.bpd.experiments.sub01.experiment import (
    WIDTH, Looper, sample_x, external_labels, pca_projectors,
    fiber_score, gauge_score, random_orthogonal,
)
from mechinterp.bpd.experiments.sub02.experiment import (
    obs_projectors, future_logit_gradients,
)

STUDY = "BPD-SUB-03"
ROOT = Path(__file__).resolve().parent
CALIBRATION_ROOTS = tuple(range(1400, 1464))
SMOKE = 1399
RANK = 8
SHAMS = 8
STEPS = 256
TRAIN_N = 2048
DISCOVERY_N = 512
CALIBRATION_N = 512
PILOT_N = 1024
ANGLE_GRID = (0.20, 0.40, 0.60, 0.80)
ALPHA_RATIO = .25
MECHANISTIC_FLOOR = .15
TARGET_BOUNDED = .30
PLANNED_CONFIRM_UNITS = 64
EVALMATH_REF = "6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a"
BOOTSTRAP_DRAWS = 2000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounded_signed(x: float) -> float:
    return x / (1.0 + abs(x))


def inverse_positive(x: float) -> float:
    if not (0 <= x < 1):
        raise ValueError("Invalid bounded positive effect")
    return x / (1.0 - x)


def train_source(seed: int, *, steps: int = STEPS):
    if seed not in CALIBRATION_ROOTS and seed != SMOKE:
        raise ValueError("Outside registered SUB03 cohort")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = Looper()
    x = sample_x(seed, 1, TRAIN_N)
    y = external_labels(x)
    rng = torch.Generator().manual_seed(811031 + seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(steps):
        idx = torch.randint(TRAIN_N, (128,), generator=rng)
        logits, _ = model.rollout(x[idx])
        loss = F.binary_cross_entropy_with_logits(logits, y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred, _ = model.rollout(x)
        train_acc = float(((pred >= 0) == y.bool()).float().mean())
    return model, {
        "training_steps": steps,
        "examples": TRAIN_N,
        "batch_size": 128,
        "learning_rate": .01,
        "train_accuracy": train_acc,
    }


def orthogonal(seed: int, size: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(size, size, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(x)
    signs = torch.where(torch.diag(r) < 0, -1.0, 1.0)
    return q * signs[None, :]


@torch.no_grad()
def observable_gramian(model: Looper, x: torch.Tensor):
    gradients, _ = future_logit_gradients(model, x)
    g = gradients.double()
    cov = g.T @ g / len(g)
    values, vectors = torch.linalg.eigh(cov)
    return cov, values, vectors


def local_gramian_rotation(vectors: torch.Tensor, theta: float, seed: int):
    """Rotate four weakest retained directions into the four discarded modes."""
    if theta not in ANGLE_GRID or vectors.shape != (WIDTH, WIDTH):
        raise ValueError("Wrong local sham parameters")
    missing = vectors[:, :4].double()
    weak = vectors[:, 4:8].double()
    mix = missing @ orthogonal(seed, 4)
    c, s = math.cos(theta), math.sin(theta)
    q = (torch.eye(WIDTH, dtype=torch.float64)
         + (c - 1.0) * (weak @ weak.T + mix @ mix.T)
         + s * (mix @ weak.T - weak @ mix.T))
    return q


@torch.no_grad()
def spectrum_preserving_shams(
    root: int, cov: torch.Tensor, eigvals: torch.Tensor,
    eigvecs: torch.Tensor, obs: torch.Tensor,
):
    if obs.shape != (WIDTH, WIDTH):
        raise ValueError("Expected rank-eight physical projector")
    entries = []
    full = obs.double()
    for idx in range(SHAMS):
        if idx < 4:
            kind = "local_spectral_rotation"
            theta = ANGLE_GRID[idx]
            q = local_gramian_rotation(
                eigvecs, theta, 910310 + 17 * root + idx)
        else:
            kind = "global_haar_spectral_rotation"
            theta = None
            q = orthogonal(911310 + 17 * root + idx, WIDTH)
        proj = q @ full @ q.T
        spectral = torch.linalg.eigvalsh(q @ cov @ q.T)
        quality = {
            "orthogonal_max_error": float(
                (q.T @ q - torch.eye(WIDTH, dtype=torch.float64)).abs().max()),
            "eigenvalue_law_max_error": float(
                (spectral - eigvals).abs().max()),
            "idempotence_max_error": float((proj @ proj - proj).abs().max()),
            "symmetry_max_error": float((proj - proj.T).abs().max()),
            "trace_error": abs(float(torch.trace(proj)) - RANK),
            "projection_distance_to_obs_fro": float((proj - full).norm()),
        }
        entries.append({
            "index": idx,
            "name": kind,
            "theta_radians": theta,
            "projector": proj.float(),
            "geometry": quality,
        })
    return entries


def geometry_admitted(entries):
    return all(
        e["geometry"]["orthogonal_max_error"] < 1e-9
        and e["geometry"]["eigenvalue_law_max_error"] < 1e-9
        and e["geometry"]["idempotence_max_error"] < 1e-9
        and e["geometry"]["symmetry_max_error"] < 1e-9
        and e["geometry"]["trace_error"] < 1e-6
        for e in entries
    )


@torch.no_grad()
def score_fiber(model: Looper, x: torch.Tensor, p: torch.Tensor,
                alpha: float, seed: int, split: int):
    score = fiber_score(
        model, x, p, torch.eye(WIDTH) - p,
        alpha, seed, split,
    )
    return score


@torch.no_grad()
def evaluate_root(seed: int, *, steps: int = STEPS):
    model, training = train_source(seed, steps=steps)
    xd = sample_x(seed, 2, DISCOVERY_N)
    xc = sample_x(seed, 3, CALIBRATION_N)
    xp = sample_x(seed, 4, PILOT_N)
    pca_projectors_, alpha, pca_geom = pca_projectors(model, xd)
    obs_projectors_, obs_geom = obs_projectors(model, xd)
    pca, obs = pca_projectors_[RANK], obs_projectors_[RANK]
    cov, eigvals, eigvecs = observable_gramian(model, xd)
    assert (obs.double()-(eigvecs[:,-RANK:]@eigvecs[:,-RANK:].T)).abs().max() < 5e-6
    shams = spectrum_preserving_shams(seed, cov, eigvals, eigvecs, obs)
    calibration_pca = score_fiber(model, xc, pca, alpha, seed, 3)
    calibration_obs = score_fiber(model, xc, obs, alpha, seed, 3)
    score_shams = []
    for arm in shams:
        response = score_fiber(model, xc, arm["projector"], alpha, seed, 3)
        d = (calibration_pca["terminal_logit_rms"]
             - response["terminal_logit_rms"])
        score_shams.append({
            "index": arm["index"],
            "name": arm["name"],
            "theta_radians": arm["theta_radians"],
            "geometry": arm["geometry"],
            "fiber": response,
            "signed_pca_minus_sham_logit_rms": d,
            "positive_false_improvement_logit_rms": max(0.,d),
            "bounded_sham_effect": bounded_signed(max(0.,d)),
        })
    pilot_pca = score_fiber(model, xp, pca, alpha, seed, 4)
    pilot_obs = score_fiber(model, xp, obs, alpha, seed, 4)
    effect = pilot_pca["terminal_logit_rms"] - pilot_obs["terminal_logit_rms"]
    with torch.no_grad():
        source_logits, _ = model.rollout(xp)
        y = external_labels(xp)
        mask = xp[:, 3] >= 0
        assert bool(mask.any() and (~mask).any())
        contexts = {
            "x3_nonnegative": float(
                ((source_logits[mask] >= 0) == y[mask].bool()).float().mean()),
            "x3_negative": float(
                ((source_logits[~mask] >= 0) == y[~mask].bool()).float().mean()),
        }
        gauge = gauge_score(model, xp[:128], seed, alpha)
        source_nll = float(F.binary_cross_entropy_with_logits(source_logits, y))
    all_fiber = [
        calibration_pca, calibration_obs, pilot_pca, pilot_obs,
        *(entry["fiber"] for entry in score_shams),
    ]
    dose_admitted = all(
        f["max_dose_deviation"] <= 5e-6
        and f["max_null_projection"] <= 5e-6
        for f in all_fiber
    )
    gauge_admitted = (
        gauge["max_clean_logit_error"] <= 5e-5
        and gauge["max_transported_impulse_logit_error"] <= 5e-5
        and gauge["max_clean_state_error"] <= 5e-5
        and gauge["max_intervened_state_error"] <= 5e-5
        and gauge["untransported_impulse_logit_rms"] > 1e-5
    )
    geometry_ok = (
        geometry_admitted(shams)
        and pca_geom["nested_max_abs"] <= 1e-5
        and obs_geom["finite_difference_max_abs"] < .02
        and obs_geom["rank_geometry"]["8"]["idempotence_max_abs"] <= 1e-5
        and obs_geom["rank_geometry"]["8"]["trace_error"] <= 1e-4
    )
    competent = min(contexts.values()) >= .80
    eligible = bool(competent and gauge_admitted and dose_admitted and geometry_ok)
    worst_null = max(e["positive_false_improvement_logit_rms"] for e in score_shams)
    return {
        "study": STUDY,
        "phase": "EXCLUDED_SMOKE" if seed == SMOKE else "CALIBRATION_DEVELOPMENT",
        "root": seed,
        "training": training,
        "protocol_sha256": digest(ROOT / "PROTOCOL.md"),
        "source_sha256": digest(Path(__file__)),
        "dependency_sub02_sha256": digest(
            ROOT.parent / "sub02" / "experiment.py"),
        "dependency_sub01_sha256": digest(
            ROOT.parent / "sub01" / "experiment.py"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "data_sizes": {
            "train": TRAIN_N, "discovery": DISCOVERY_N,
            "calibration": CALIBRATION_N, "pilot": PILOT_N,
        },
        "physical_alpha": alpha,
        "source_task_nll": source_nll,
        "contexts": contexts,
        "competent": competent,
        "gauge": gauge,
        "gauge_admitted": gauge_admitted,
        "geometry_admitted": geometry_ok,
        "dose_admitted": dose_admitted,
        "root_eligible": eligible,
        "pca_geometry": pca_geom,
        "obs_geometry": obs_geom,
        "calibration": {
            "pca": calibration_pca,
            "obs": calibration_obs,
            "shams": score_shams,
            "worst_positive_null_improvement_logit_rms": worst_null,
        },
        "pilot": {
            "pca": pilot_pca, "obs": pilot_obs,
            "pca_minus_obs_logit_rms": effect,
            "bounded_effect": bounded_signed(effect),
        },
    }


def upper_null_quantile(vals: list[float], q: float = .95, confidence: float = .95):
    """One-sided nonparametric upper tolerance order statistic."""
    n = len(vals)
    if not vals:
        return None, None
    # Use log-space Binomial CDF via scipy-free stable recurrence.
    for k in range(1, n+1):
        cdf = sum(math.comb(n, j) * q**j * (1-q)**(n-j)
                  for j in range(k))
        if cdf >= confidence:
            return sorted(vals)[k-1], k
    return None, None


def root_bootstrap(values: list[float]):
    gen = np.random.default_rng(140303)
    a = np.asarray(values, dtype=float)
    samples = gen.choice(a, size=(BOOTSTRAP_DRAWS, len(a)), replace=True).mean(1)
    return [float(np.quantile(samples, .025)),
            float(np.quantile(samples, .975))]


def preflight_spec(rows: list[dict]):
    return {
        "schema_version": 1,
        "experiment_id": STUDY,
        "independent_unit": "independently trained source model root",
        "alpha": "0.05",
        "scientific_min_effect": "0.13043478260869565",
        "target_effect": "0.30",
        "measurement_resolution": "0.005",
        "sham_quantile": "0.95",
        "sham_confidence": "0.95",
        "planned_comparisons": 1,
        "confirm_units_planned": PLANNED_CONFIRM_UNITS,
        "pilot_from_development_only": True,
        "heldout_sealed": True,
        "units_independent": True,
        "pilot": [
            {
                "unit_id": f"root-{r['root']}",
                "paired_effect": repr(r["pilot"]["bounded_effect"]),
                "sham_effects": [
                    repr(e["bounded_sham_effect"])
                    for e in r["calibration"]["shams"]
                ],
            }
            for r in rows
        ],
    }


def aggregate(rows: list[dict]):
    if [row["root"] for row in rows] != list(CALIBRATION_ROOTS):
        raise ValueError("Original 64 registered roots must all be retained")
    eligible = [r for r in rows if r.get("root_eligible", False)]
    n = len(eligible)
    physical = [
        r["calibration"]["worst_positive_null_improvement_logit_rms"]
        for r in eligible
    ]
    bounded = [bounded_signed(x) for x in physical]
    tail, rank = upper_null_quantile(bounded)
    effects = [r["pilot"]["pca_minus_obs_logit_rms"] for r in eligible]
    mean = float(np.mean(effects)) if effects else None
    ci = root_bootstrap(effects) if effects else None
    return {
        "study": STUDY,
        "phase": "CALIBRATION_DEVELOPMENT",
        "attempted_roots": len(rows),
        "root_ids": list(CALIBRATION_ROOTS),
        "eligible_roots": n,
        "ineligible_roots": [
            r["root"] for r in rows if not r.get("root_eligible", False)
        ],
        "at_least_59_eligible": n >= 59,
        "root_level_95_95_bounded_sham_tail": tail if n >= 59 else None,
        "order_statistic_rank": rank if n >= 59 else None,
        "root_level_95_95_physical_sham_tail": (
            inverse_positive(tail) if n >= 59 and tail is not None else None
        ),
        "pilot_mean_raw_logit_rms": mean,
        "pilot_root_bootstrap95_raw_logit_rms": ci,
        "pilot_mean_bounded": (
            float(np.mean([bounded_signed(x) for x in effects])) if effects else None
        ),
        "original_sub02_engineering_mean": .43358741279902396,
        "original_sub02_source_roots": list(range(1300, 1308)),
        "evalmath_ref": EVALMATH_REF,
        "planned_confirmation_population": "UNGENERATED_SEALED",
        "status": (
            "CALIBRATION_SUFFICIENT_FOR_PREFLIGHT"
            if n >= 59 else "CALIBRATION_INSUFFICIENT"
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=("smoke", "calibration"))
    p.add_argument("--ack-calibration", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--pilot-out", type=Path)
    args = p.parse_args()
    if args.phase == "calibration" and not args.ack_calibration:
        p.error("Calibration requires --ack-calibration")
    if args.phase == "smoke":
        receipt = evaluate_root(SMOKE, steps=16)
        report = {
            "phase": "EXCLUDED_SMOKE", "root": SMOKE,
            "dose_admitted": receipt["dose_admitted"],
            "geometry_admitted": receipt["geometry_admitted"],
            "gauge_admitted": receipt["gauge_admitted"],
        }
    else:
        rows = []
        for root in CALIBRATION_ROOTS:
            try:
                record = evaluate_root(root)
            except Exception as exc:
                record = {
                    "study": STUDY, "phase": "CALIBRATION_DEVELOPMENT",
                    "root": root, "root_eligible": False,
                    "error_type": type(exc).__name__, "error_message": str(exc),
                }
            rows.append(record)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps({
                    "study": STUDY, "phase": "RUNNING",
                    "attempted": len(rows), "rows": rows,
                }, indent=2, sort_keys=True, allow_nan=False)+"\n")
        summary = aggregate(rows)
        receipt = {"summary": summary, "roots": rows}
        report = summary
        if args.pilot_out and summary["at_least_59_eligible"]:
            spec = preflight_spec(
                [r for r in rows if r.get("root_eligible", False)])
            args.pilot_out.parent.mkdir(parents=True, exist_ok=True)
            args.pilot_out.write_text(json.dumps(
                spec, indent=2, sort_keys=True, allow_nan=False)+"\n")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2,
            sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
