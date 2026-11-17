"""BPD-PD-11: global versus local angular Givens search for unlabelled reads.

The source law, Q rotations, held-out families and physical BPD component
wrapper come from the frozen PD06/PD09 suite. Global and local arms receive
only observed training x, never latent support or true Q during optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import time
from pathlib import Path

import torch

from mechinterp.bpd.theory.pd06.experiment import (
    build as build_pd06, sample_batch, rank4_alignment,
    exact_mcnemar_p, wilson_ci,
)
from mechinterp.bpd.theory.pd09.experiment import (
    qr_orthogonal, smoothed_l0, set_physical_orthogonal, parameter_fingerprint,
    evaluate_family as pd09_evaluate_family,
    train_one as pd09_train_one,
)

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
SEEDS = tuple(CONFIG["seed_list"])
ARMS = tuple(a["key"] for a in CONFIG["arms"])
ARM_SPEC = {a["key"]: a for a in CONFIG["arms"]}
FAMILIES = tuple(CONFIG["source"]["eval_families"])
PAIRS = tuple(tuple(pair) for pair in CONFIG["objective"]["ordered_pairs"])
WIDTH = 4


def angle_grid(key: str, *, n: int | None = None) -> torch.Tensor:
    arm = ARM_SPEC[key]
    if "theta_min_deg" not in arm:
        raise ValueError(f"{key} does not use Givens angle search")
    count = CONFIG["objective"]["angles_per_pair"] if n is None else n
    assert count >= 3 and count % 2 == 1
    return torch.linspace(
        math.radians(arm["theta_min_deg"]),
        math.radians(arm["theta_max_deg"]),
        count,
        dtype=torch.float64,
    )


def initialize_basis(bpd, true_q: torch.Tensor, arm: str) -> torch.Tensor:
    spec = ARM_SPEC[arm]
    if spec["initialization"] == "true_Q_pairmix_44deg_01":
        angle = math.radians(44.0)
        c, s = math.cos(angle), math.sin(angle)
        mix = torch.eye(WIDTH, dtype=torch.float64)
        mix[0, 0], mix[0, 1] = c, -s
        mix[1, 0], mix[1, 1] = s, c
        return true_q.double() @ mix
    if spec["initialization"] in ("matched_random_qr", "PD06_original"):
        return qr_orthogonal(bpd.wrappers["core"].V.detach().double())
    raise ValueError(spec["initialization"])


def gaussian_mass(observed: torch.Tensor, read: torch.Tensor, sigma: float) -> float:
    return float(
        torch.exp(-.5 * ((observed @ read) / sigma).square()).mean()
    )


def best_pair_angle(
    u: torch.Tensor, v: torch.Tensor, angles: torch.Tensor, sigma: float,
) -> tuple[int, float, float]:
    """All candidate angles computed together; includes the unchanged pair."""
    if u.ndim != 1 or v.shape != u.shape:
        raise ValueError("read projections must be matched vectors")
    cs, sn = angles.cos()[:, None], angles.sin()[:, None]
    a = cs * u[None, :] + sn * v[None, :]
    b = -sn * u[None, :] + cs * v[None, :]
    scores = (
        torch.exp(-.5 * (a / sigma).square()).mean(dim=-1)
        + torch.exp(-.5 * (b / sigma).square()).mean(dim=-1)
    )
    # Absolute angle as tie breaker is a declared no-op preference.
    high = float(scores.max())
    tied = [i for i, value in enumerate(scores.tolist())
            if high - value <= 1e-12]
    chosen = min(tied, key=lambda i: (abs(float(angles[i])), float(angles[i])))
    return chosen, float(angles[chosen]), high


def search_basis(
    observed: torch.Tensor, initial: torch.Tensor, *,
    arm: str, sweeps: int | None = None, angles_per_pair: int | None = None,
):
    """Deterministic, unlabelled, global/local pair-rotation coordinate search."""
    if arm not in ARM_SPEC or ARM_SPEC[arm]["kind"] not in ("global", "local"):
        raise ValueError(arm)
    samples = observed.reshape(-1, WIDTH).double()
    read = initial.clone().double()
    angles = angle_grid(arm, n=angles_per_pair)
    sigma = float(CONFIG["objective"]["sigma"])
    if sweeps is None:
        sweeps = CONFIG["objective"]["sweep_count"]
    first = gaussian_mass(samples, read, sigma)
    adjustments = 0
    objective_path = [first]
    total_eval = 0
    moves = []
    # Each pair's Gaussian mass is exactly the varying part of the full
    # four-column score, so maximizing it increases the full score.
    for _ in range(sweeps):
        for i, j in PAIRS:
            projections = samples @ read[:, [i, j]]
            k, theta, new_pair_mass = best_pair_angle(
                projections[:, 0], projections[:, 1], angles, sigma
            )
            total_eval += len(angles)
            if abs(theta) > 1e-12:
                c, s = math.cos(theta), math.sin(theta)
                vi, vj = read[:, i].clone(), read[:, j].clone()
                read[:, i] = c * vi + s * vj
                read[:, j] = -s * vi + c * vj
                adjustments += 1
            if not bool(torch.isfinite(read).all()):
                raise ArithmeticError("nonfinite Givens factor")
            moves.append({"pair": [i, j], "degrees": theta * 180 / math.pi,
                          "pair_score": new_pair_mass, "selected_index": k})
        objective_path.append(gaussian_mass(samples, read, sigma))
    # Tiny rounding drift is possible after 36 Givens updates.
    # Retain the actual searched basis; physical alignment checked separately.
    ortho_error = float((read.T @ read - torch.eye(WIDTH, dtype=read.dtype)).abs().max())
    return read, {
        "initial_gaussian_mass": first,
        "final_gaussian_mass": objective_path[-1],
        "gaussian_mass_by_sweep": objective_path,
        "actual_nonzero_pair_updates": adjustments,
        "number_of_candidate_pair_scores": total_eval,
        "angle_candidates_per_pair": len(angles),
        "pair_updates_considered": len(moves),
        "orthogonality_max_error": ortho_error,
        "moves": moves,
    }


def source_only_training(
    seed: int, arm: str, *, boards: int | None = None,
    sweeps: int | None = None, angles_per_pair: int | None = None,
):
    bpd, _, q, initial_sha, q_sha = build_pd06(seed, frozen_planted=False)
    basis = initialize_basis(bpd, q, arm)
    initial_basis_hash = hashlib.sha256(
        json.dumps(basis.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    if boards is None:
        boards = CONFIG["source"]["train_boards"]
    if ARM_SPEC[arm]["kind"] == "negative":
        moves = {
            "initial_gaussian_mass": 0.0,
            "final_gaussian_mass": 0.0,
            "gaussian_mass_by_sweep": [],
            "actual_nonzero_pair_updates": 0,
            "number_of_candidate_pair_scores": 0,
            "angle_candidates_per_pair": 0,
            "pair_updates_considered": 0,
            "orthogonality_max_error": float(
                (basis.T @ basis - torch.eye(WIDTH, dtype=basis.dtype)).abs().max()),
            "moves": [],
        }
        final = basis
    else:
        gen = torch.Generator().manual_seed(CONFIG["source"]["train_seed_offset"] + seed)
        observed, _ignored_support, _ignored_latent = sample_batch(
            gen, boards, q, family="train"
        )
        final, moves = search_basis(
            observed, basis, arm=arm, sweeps=sweeps, angles_per_pair=angles_per_pair
        )
    set_physical_orthogonal(bpd, final.float())
    meta = {
        **moves,
        "source_rotation_sha256": q_sha,
        "initial_parameters_sha256": initial_sha,
        "initial_basis_sha256": initial_basis_hash,
        "physical_factor_sha256": parameter_fingerprint(bpd),
        "privileged_true_Q_initializer": ARM_SPEC[arm]["uses_true_rotation_in_initialization"],
        "used_cardinality_labels": False,
        "observed_training_boards": boards if ARM_SPEC[arm]["kind"] != "negative" else 0,
        "trained_derivative_free": ARM_SPEC[arm]["kind"] != "negative",
    }
    return bpd, q, meta


def train_arm(seed: int, arm: str, *, boards=None, sweeps=None,
              angles_per_pair=None, positive_steps=None):
    if arm not in ARMS:
        raise ValueError(arm)
    if ARM_SPEC[arm]["kind"] == "supervised":
        bpd, q, src = pd09_train_one(
            seed, "count_supervised_positive",
            steps=(CONFIG["training"]["count_supervised_steps"]
                   if positive_steps is None else positive_steps),
            batch_size=CONFIG["training"]["count_supervised_batch"],
        )
        return bpd, q, {
            "source_rotation_sha256": src["rotation_sha256"],
            "initial_parameters_sha256": src["initial_parameters_sha256"],
            "initial_basis_sha256": None,
            "physical_factor_sha256": src["physical_factor_sha256"],
            "privileged_true_Q_initializer": False,
            "used_cardinality_labels": True,
            "observed_training_boards": 0,
            "actual_nonzero_pair_updates": 0,
            "number_of_candidate_pair_scores": 0,
            "angle_candidates_per_pair": 0,
            "pair_updates_considered": 0,
            "orthogonality_max_error": None,
            "first_factor_gradient_norm": src["first_trainable_gradient_norm"],
            "factor_update_norm": src["trainable_update_norm"],
            "trained_derivative_free": False,
        }
    return source_only_training(
        seed, arm, boards=boards, sweeps=sweeps,
        angles_per_pair=angles_per_pair,
    )


def mechanism_success(evaluated, meta, arm: str):
    cfg = CONFIG["scoring"]
    if evaluated["alignment"]["mean_cosine"] < cfg["mean_read_cos_min"]:
        return False
    if evaluated["overall_mse"] > cfg["overall_binary_MSE_max"]:
        return False
    for family in FAMILIES:
        m = evaluated["per_family"][family]
        unseen = "unseen" in family
        mse_limit = cfg["unseen_family_MSE_max" if unseen else "seen_family_MSE_max"]
        support_floor = cfg[
            "unseen_fourpass_exact_support_min" if unseen
            else "seen_fourpass_exact_support_min"
        ]
        if (m["mse"] > mse_limit or m["support_fraction"] < support_floor
                or not m["mask_exact_binary"]):
            return False
    if ARM_SPEC[arm]["kind"] == "supervised":
        return meta["first_factor_gradient_norm"] > 0 and meta["factor_update_norm"] > 0
    if ARM_SPEC[arm]["kind"] == "negative":
        return True
    return meta["actual_nonzero_pair_updates"] > 0 and meta["orthogonality_max_error"] < 1e-4


def run_arm(seed: int, arm: str, *, boards=None, sweeps=None,
            angles_per_pair=None, positive_steps=None, heldout_boards=None):
    torch.set_num_threads(1)
    start = time.monotonic()
    bpd, q, meta = train_arm(
        seed, arm, boards=boards, sweeps=sweeps,
        angles_per_pair=angles_per_pair, positive_steps=positive_steps,
    )
    align = rank4_alignment(bpd, q)
    heldout_boards = (CONFIG["source"]["heldout_boards_per_family"]
                      if heldout_boards is None else heldout_boards)
    with torch.no_grad():
        family = {
            name: pd09_evaluate_family(
                bpd, q, seed, arm, name,
                align["component_to_latent_permutation"], heldout_boards,
            ) for name in FAMILIES
        }
    measured = {
        "alignment": align,
        "overall_mse": sum(x["mse"] for x in family.values()) / len(FAMILIES),
        "physical_weight_delta_mse": float(
            bpd.wrappers["core"].weight_delta().square().mean()
        ),
        "per_family": family,
    }
    return {
        "seed": seed, "arm": arm, **meta, "heldout": measured,
        "success": mechanism_success(measured, meta, arm),
        "elapsed_seconds": round(time.monotonic() - start, 4),
    }


def summarize(rows, *, registered: bool):
    by = {(r["seed"], r["arm"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    if len(rows) != len(by) or set(by) != {
        (s, a) for s in seeds for a in ARMS
    }:
        raise ValueError("missing or duplicate registered arm")
    counts = {a: sum(by[(s, a)]["success"] for s in seeds) for a in ARMS}
    primary, control = CONFIG["scoring"]["primary_arm"], CONFIG["scoring"]["primary_control"]
    only_a = sum(by[(s, primary)]["success"] and not by[(s, control)]["success"]
                 for s in seeds)
    only_b = sum(by[(s, control)]["success"] and not by[(s, primary)]["success"]
                 for s in seeds)
    p = exact_mcnemar_p(only_a, only_b)
    full = bool(registered and tuple(seeds) == SEEDS)
    c = CONFIG["scoring"]
    return {
        "study_id": CONFIG["study_id"], "registered": full,
        "seed_count": len(seeds), "arm_count": len(rows),
        "success_by_arm": counts,
        "Wilson95_by_arm": {a: wilson_ci(n, len(seeds)) for a, n in counts.items()},
        "global_only": only_a, "local_only": only_b,
        "net_global_seed_wins": only_a - only_b,
        "McNemar_exact_two_sided_p": p,
        "registered_primary_passed": bool(
            full and counts[primary] >= c["primary_successes_min"]
            and only_a - only_b >= c["primary_net_paired_seed_wins_min"]
            and p <= c["primary_two_sided_McNemar_p_max"]
        ),
        "paired_seeds": [{
            "seed": seed,
            "global_success": by[(seed, primary)]["success"],
            "local_success": by[(seed, control)]["success"],
            "global_read_cosine": by[(seed, primary)]["heldout"]["alignment"]["mean_cosine"],
            "local_read_cosine": by[(seed, control)]["heldout"]["alignment"]["mean_cosine"],
            "global_unseen_pair": by[(seed, primary)]["heldout"]["per_family"]["k2_unseen"]["support_fraction"],
            "local_unseen_pair": by[(seed, control)]["heldout"]["per_family"]["k2_unseen"]["support_fraction"],
            "global_unseen_triple": by[(seed, primary)]["heldout"]["per_family"]["k3_unseen"]["support_fraction"],
            "local_unseen_triple": by[(seed, control)]["heldout"]["per_family"]["k3_unseen"]["support_fraction"],
        } for seed in seeds],
    }


def run_registered():
    started = time.monotonic()
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(run_arm(seed, arm))
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            if peak >= CONFIG["execution"]["memory_limit_GiB"]:
                raise MemoryError(f"PD11 peak RSS {peak:.3f} GiB")
        matched = {r["initial_basis_sha256"] for r in rows if r["seed"] == seed
                   and r["arm"] in ("global_givens","local_givens","random_orthogonal")}
        rots = {r["source_rotation_sha256"] for r in rows if r["seed"] == seed}
        if len(matched) != 1 or len(rots) != 1:
            raise AssertionError(f"Within-seed mismatch for seed {seed}")
    return {
        "study_id": CONFIG["study_id"],
        "status": "original_registered_angular_search",
        "source_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                          for f in ("CONFIG.json", "PROTOCOL.md", "experiment.py")},
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "peak_RSS_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "summary": summarize(rows, registered=True),
        "arms": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = run_registered()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
