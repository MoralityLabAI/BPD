"""BPD-SUB-02 registered future-observability versus PCA hidden-grain experiment.

See PROTOCOL.md: source/root roster and outcomes were frozen before code.
No SUB-01 training root, protected confirmation or private test is accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.bpd.experiments.sub01.experiment import (
    WIDTH, PASSES, TRAIN_SIZE, DISCOVERY_SIZE, SELECTION_SIZE, HELDOUT_SIZE,
    SHAMS, ALPHA_FRACTION, Looper, sample_x, external_labels,
    pca_projectors, fiber_score, projected_rollout, gauge_score, rms,
    random_orthogonal, bernoulli_kl,
)

STUDY = "BPD-SUB-02"
ROOT = Path(__file__).resolve().parent
DEVELOPMENT_ROOTS = tuple(range(1300, 1308))
SMOKE = 1299
RANKS = (8, 10)
PRIMARY_RANK = 8
STEPS = 256
ROOT_BOOTSTRAPS = 2000
MINIMUM_MEAN_EFFECT = 0.15


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_source(seed: int, *, steps: int = STEPS):
    if seed not in DEVELOPMENT_ROOTS and seed != SMOKE:
        raise ValueError("SUB-02 source seed is outside registered roster")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = Looper()
    x = sample_x(seed, 1, TRAIN_SIZE)
    y = external_labels(x)
    g = torch.Generator().manual_seed(811031 + seed)
    optim = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(steps):
        ix = torch.randint(TRAIN_SIZE, (128,), generator=g)
        logit, _ = model.rollout(x[ix])
        loss = F.binary_cross_entropy_with_logits(logit, y[ix])
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        train_logit, _ = model.rollout(x)
    return model, {
        "train_accuracy": float(((train_logit >= 0) == y.bool()).float().mean()),
        "training_steps": steps, "train_examples": TRAIN_SIZE,
        "batch_size": 128, "lr": .01,
    }


def future_logit_gradients(model: Looper, x: torch.Tensor):
    """Per-input d terminal logit / d post-first-pass hidden state, float32."""
    with torch.enable_grad():
        h0 = x.new_zeros(len(x), WIDTH)
        h1 = model.step(h0, x).detach().requires_grad_(True)
        h = h1
        for _ in range(PASSES - 1):
            h = model.step(h, x)
        logits = model.head(h).squeeze(-1)
        grads = torch.autograd.grad(logits.sum(), h1)[0]
    return grads.detach(), h1.detach()


def future_logit_from_h1(model: Looper, h1: torch.Tensor, x: torch.Tensor):
    h = h1
    for _ in range(PASSES - 1):
        h = model.step(h, x)
    return model.head(h).squeeze(-1)


def finite_diff_check(model: Looper, x: torch.Tensor, gradients: torch.Tensor,
                      h1: torch.Tensor):
    """Smoke/development diagnostic fixed small x slice, no parameter updates."""
    epsilon = 1e-3
    chosen = [0, 3, 6, 9]
    differences = []
    with torch.no_grad():
        for col in chosen:
            unit = torch.zeros_like(h1[:8])
            unit[:, col] = epsilon
            plus = future_logit_from_h1(model, h1[:8] + unit, x[:8])
            minus = future_logit_from_h1(model, h1[:8] - unit, x[:8])
            fd = (plus - minus) / (2 * epsilon)
            differences.append((fd - gradients[:8, col]).abs().max())
    return float(torch.stack(differences).max())


def obs_projectors(model: Looper, discovery: torch.Tensor):
    grads, h1 = future_logit_gradients(model, discovery)
    g = grads.double()
    cov = g.T @ g / len(g)
    values, vectors = torch.linalg.eigh(cov)
    projectors = {
        rank: (vectors[:, -rank:] @ vectors[:, -rank:].T).float()
        for rank in RANKS
    }
    quality = {}
    for rank, p in projectors.items():
        quality[str(rank)] = {
            "idempotence_max_abs": float((p @ p - p).abs().max()),
            "symmetry_max_abs": float((p - p.T).abs().max()),
            "trace_error": abs(float(torch.trace(p)) - rank),
            "lambda_retained_min": float(values[-rank]),
            "lambda_next": float(values[-rank - 1]),
        }
    return projectors, {
        "spectrum": values.tolist(),
        "discovery_examples": len(discovery),
        "gradient_l2_mean": float(g.norm(dim=-1).mean()),
        "finite_difference_max_abs": finite_diff_check(model, discovery, grads, h1),
        "rank_geometry": quality,
    }


def sham_projector(seed: int, rank: int, sham: int):
    if rank not in RANKS or not (0 <= sham < SHAMS):
        raise ValueError("Unregistered sham or rank")
    q = random_orthogonal(381001 + seed * 1009 + rank * 31 + sham)
    p = torch.zeros(WIDTH, WIDTH)
    p[:rank, :rank] = torch.eye(rank)
    return q @ p @ q.T


@torch.no_grad()
def score_arm(model: Looper, x: torch.Tensor, rank: int,
              p: torch.Tensor, alpha: float, seed: int, split: int):
    null_p = torch.eye(WIDTH) - p
    fiber = fiber_score(model, x, p, null_p, alpha, seed, split)
    teacher_logit, _ = model.rollout(x)
    projected_logit, _ = projected_rollout(model, x, p)
    y = external_labels(x)
    return {
        "fiber": fiber,
        "clean_to_projected_kl_nats":
            float(bernoulli_kl(teacher_logit, projected_logit).mean()),
        "projected_task_nll":
            float(F.binary_cross_entropy_with_logits(projected_logit, y)),
        "projected_accuracy":
            float(((projected_logit >= 0) == y.bool()).float().mean()),
    }


@torch.no_grad()
def score_root(model: Looper, seed: int, heldout: torch.Tensor,
               pca: dict, obs: dict, alpha: float):
    arms = {}
    for rank in RANKS:
        a = score_arm(model, heldout, rank, pca[rank], alpha, seed, 4)
        b = score_arm(model, heldout, rank, obs[rank], alpha, seed, 4)
        shams = [
            score_arm(model, heldout, rank,
                      sham_projector(seed, rank, index), alpha, seed, 4)
            ["fiber"]["terminal_logit_rms"]
            for index in range(SHAMS)
        ]
        d = a["fiber"]["terminal_logit_rms"] - b["fiber"]["terminal_logit_rms"]
        arms[str(rank)] = {
            "pca": a, "observability": b,
            "eight_same_dose_random_shams": shams,
            "mean_primary_comparator_pca_minus_obs_logit_rms": d,
            "median_sham_minus_obs_logit_rms":
                float(np.median(shams)-b["fiber"]["terminal_logit_rms"]),
        }
    return arms


def evaluate_root(seed: int, *, steps: int = STEPS):
    model, training = train_source(seed, steps=steps)
    xd = sample_x(seed, 2, DISCOVERY_SIZE)
    xc = sample_x(seed, 3, SELECTION_SIZE)
    xh = sample_x(seed, 4, HELDOUT_SIZE)
    pca_all, alpha, pca_geom = pca_projectors(model, xd)
    obs, obs_geom = obs_projectors(model, xd)
    with torch.no_grad():
        logit, _ = model.rollout(xh)
        y = external_labels(xh)
        contexts = {
            "x3_nonnegative": float(((logit[xh[:,3]>=0]>=0)
                                     == y[xh[:,3]>=0].bool()).float().mean()),
            "x3_negative": float(((logit[xh[:,3]<0]>=0)
                                  == y[xh[:,3]<0].bool()).float().mean()),
        }
        gauge = gauge_score(model, xh[:128], seed, alpha)
        arms = score_root(model, seed, xh, pca_all, obs, alpha)
        # Calibration remains readout-free and cannot update any projectors.
        physical_dose_calibration = {
            str(rank): {
                "pca": fiber_score(model, xc, pca_all[rank],
                                   torch.eye(WIDTH)-pca_all[rank],
                                   alpha, seed, 3)["max_dose_deviation"],
                "obs": fiber_score(model, xc, obs[rank],
                                   torch.eye(WIDTH)-obs[rank],
                                   alpha, seed, 3)["max_dose_deviation"],
            }
            for rank in RANKS
        }
    gauge_ok = (gauge["max_clean_logit_error"] <= 5e-5
                and gauge["max_transported_impulse_logit_error"] <= 5e-5
                and gauge["max_clean_state_error"] <= 5e-5
                and gauge["max_intervened_state_error"] <= 5e-5
                and gauge["untransported_impulse_logit_rms"] > 1e-5)
    geometry_ok = (pca_geom["nested_max_abs"] <= 1e-5
                   and obs_geom["finite_difference_max_abs"] < .02
                   and all(
                       v["symmetry_max_abs"] <= 1e-5
                       and v["idempotence_max_abs"] <= 1e-5
                       and v["trace_error"] <= 1e-4
                       for v in obs_geom["rank_geometry"].values()))
    dose_ok = all(
        arms[str(r)][mode]["fiber"]["max_dose_deviation"] <= 5e-6
        and arms[str(r)][mode]["fiber"]["max_null_projection"] <= 5e-6
        for r in RANKS for mode in ("pca", "observability")
    ) and all(
        d <= 5e-6
        for v in physical_dose_calibration.values()
        for d in v.values()
    )
    competent = bool(min(contexts.values()) >= .80)
    return {
        "study": STUDY, "phase": "EXCLUDED_SMOKE" if seed == SMOKE else "ENGINEERING",
        "root": seed,
        "protocol_sha256": sha(ROOT / "PROTOCOL.md"),
        "source_sha256": sha(Path(__file__)),
        "dependency_sub01_sha256":
            sha(ROOT.parent / "sub01" / "experiment.py"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__,
        },
        "training": training,
        "data_sizes": {
            "train": TRAIN_SIZE, "discovery": DISCOVERY_SIZE,
            "calibration": SELECTION_SIZE, "heldout": HELDOUT_SIZE,
        },
        "source_heldout_nll": float(
            F.binary_cross_entropy_with_logits(logit, y)),
        "contexts": contexts,
        "competent": competent, "gauge": gauge, "gauge_ok": gauge_ok,
        "pca_geometry": pca_geom, "obs_geometry": obs_geom,
        "geometry_ok": geometry_ok, "dose_ok": dose_ok,
        "physical_impulse_alpha": alpha,
        "calibration_dose_checks": physical_dose_calibration,
        "arms": arms,
        "root_eligible": bool(competent and gauge_ok and geometry_ok and dose_ok),
    }


def whole_root_bootstrap(values, *, seed: int = 130202):
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(data, size=(ROOT_BOOTSTRAPS, len(data)), replace=True).mean(1)
    return [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))]


def aggregate(roots):
    if [r["root"] for r in roots] != list(DEVELOPMENT_ROOTS):
        raise ValueError("Expected exactly eight original SUB02 development roots")
    admitted = [r for r in roots if r["root_eligible"]]
    vals = [r["arms"]["8"]["mean_primary_comparator_pca_minus_obs_logit_rms"]
            for r in admitted]
    ci = whole_root_bootstrap(vals) if len(vals) == len(DEVELOPMENT_ROOTS) else None
    positive = sum(v > 0 for v in vals)
    avg = float(np.mean(vals)) if vals else None
    passed = (len(vals) == len(DEVELOPMENT_ROOTS)
              and avg >= MINIMUM_MEAN_EFFECT
              and positive >= 7 and ci[0] > 0)
    return {
        "study": STUDY, "phase": "ENGINEERING",
        "independent_roots": len(roots),
        "root_ids": list(DEVELOPMENT_ROOTS),
        "eligible": len(admitted),
        "primary_rank": PRIMARY_RANK,
        "root_primary_effects": vals,
        "mean_primary_pca_minus_observability_logit_rms": avg,
        "positive_roots": positive,
        "bootstrap95_whole_roots": ci,
        "registered_minimum_mean_effect": MINIMUM_MEAN_EFFECT,
        "strict_evalmath_preflight": "PENDING_INDEPENDENT_SHAM_CALIBRATION",
        "independent_sham_tail_calibrated": False,
        "verdict": "ENGINEERING_SIGNAL_ONLY" if passed
            else "ENGINEERING_NULL_OR_INELIGIBLE",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("smoke", "development"))
    parser.add_argument("--ack-development", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "development" and not args.ack_development:
        parser.error("Development requires --ack-development")
    if args.phase == "smoke":
        receipt = evaluate_root(SMOKE, steps=16)
        brief = {"root": SMOKE, "phase": "EXCLUDED_SMOKE",
                 "eligible": receipt["root_eligible"],
                 "gauge_ok": receipt["gauge_ok"],
                 "geometry_ok": receipt["geometry_ok"],
                 "dose_ok": receipt["dose_ok"]}
    else:
        roots = [evaluate_root(seed) for seed in DEVELOPMENT_ROOTS]
        receipt = {"summary": aggregate(roots), "roots": roots}
        brief = receipt["summary"]
    data = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    print(json.dumps(brief, sort_keys=True))


if __name__ == "__main__":
    main()
