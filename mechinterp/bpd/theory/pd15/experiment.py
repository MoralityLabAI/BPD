"""BPD-PD-15 prospective norm-weighted clustered 16D identification experiment.

CONFIG and PROTOCOL were frozen before scientific model training.
All four learned arms receive only observed dense x, orthogonal initial V,
the fixed sigma and the fixed lambda. Q, support labels, cardinalities and
heldout observations enter only the source generator, the privileged oracle,
and post-fit evaluation/independent diagnostic.
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

from mechinterp.bpd.theory.pd02.experiment import exact_mcnemar_p, wilson_ci
from mechinterp.bpd.theory.pd12.experiment import (
    angle_grid, basis_alignment, best_pair_angle, evaluate_family,
    initial_random_basis, physical_model, rotation_for_seed, sample_batch
)
from mechinterp.bpd.theory.pd13.posthoc_zero_mass_audit import four_pair_remix

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
CELL = CONFIG["seed_cell"]
SEEDS = tuple(CELL["seeds"])
ARMS = tuple(a["key"] for a in CONFIG["arms"])
ARM_SPEC = {a["key"]: a for a in CONFIG["arms"]}
FAMILIES = tuple(CONFIG["source"]["heldout_families"])
DIM = 16
PAIRS = tuple((i, j) for i in range(DIM) for j in range(i + 1, DIM))
ANGLE_GRID = angle_grid("global37")
assert len(PAIRS) == 120 and len(ANGLE_GRID) == 37


def source_for_seed(seed: int, *, boards: int | None = None):
    """Exactly PD12's clustered strong-margin generator, original seeds/RNG."""
    torch.set_num_threads(1)
    src = CONFIG["source"]
    n = src["train_boards"] if boards is None else boards
    gen = torch.Generator().manual_seed(src["training_rng_offset"] + seed)
    q = rotation_for_seed(seed, DIM)
    x, masks, latent = sample_batch(gen, n, q, cell=CELL, family="train")
    return q, x, hashlib.sha256(x.numpy().tobytes()).hexdigest(), masks, latent


def weights_from_observed(observed: torch.Tensor, lam: float) -> torch.Tensor:
    """Rotation invariant, entirely source-observable per-pass weights."""
    if observed.shape[-1] != DIM:
        raise ValueError("PD15 expects 16D observed data")
    if lam not in (0., 1.):
        raise ValueError("Only registered lambda=0 or 1 allowed")
    flat = observed.reshape(-1, DIM).double()
    if lam == 0.:
        return torch.ones(flat.shape[0], dtype=torch.float64, device=flat.device)
    return torch.exp(-lam * flat.square().sum(dim=-1))


def effective_sample_size(weights: torch.Tensor) -> float:
    return float(weights.sum().square() / weights.square().sum())


def weighted_mass(projected: torch.Tensor, sigma: float, weights: torch.Tensor) -> float:
    """Mean over 16 reads, with fixed normalized per-vector weights."""
    if projected.ndim != 2 or projected.shape[1] != DIM:
        raise ValueError("Expected N x 16 projections")
    if weights.shape != (projected.shape[0],):
        raise ValueError("Per-observation weight shape mismatch")
    mass = torch.exp(-.5 * (projected / sigma).square())
    return float((mass * weights[:, None]).sum() / (weights.sum() * DIM))


def best_pair_angle_weighted(u: torch.Tensor, v: torch.Tensor,
                             angles: torch.Tensor, sigma: float,
                             weights: torch.Tensor) -> tuple[float, float]:
    """The exact registered weighted analogue of PD12 best_pair_angle."""
    cs, ss = angles.cos()[:, None], angles.sin()[:, None]
    a = cs * u[None, :] + ss * v[None, :]
    b = -ss * u[None, :] + cs * v[None, :]
    score = ((torch.exp(-.5 * (a / sigma).square()) +
              torch.exp(-.5 * (b / sigma).square())) @ weights) / weights.sum()
    high = float(score.max())
    tied = [i for i, value in enumerate(score.tolist()) if high-value <= 1e-12]
    best = min(tied, key=lambda i: (abs(float(angles[i])), float(angles[i])))
    return float(angles[best]), high


def fit_basis(observed: torch.Tensor, initial: torch.Tensor,
              sigma: float, lam: float, *, sweeps: int | None = None):
    """No Q, support, seed labels or behavioral endpoints in optimizer API."""
    if observed.ndim != 4 or observed.shape[-3:] != (4, 1, DIM):
        raise ValueError("Expected [boards, 4, 1, 16]")
    if initial.shape != (DIM, DIM) or sigma <= 0:
        raise ValueError("Bad source-independent optimizer parameters")
    sweeps = CONFIG["objective"]["sweeps"] if sweeps is None else sweeps
    if sweeps < 1:
        raise ValueError("At least one sweep required")
    flat = observed.reshape(-1, DIM).double()
    weights = weights_from_observed(observed, lam)
    z = flat @ initial.double()
    read = initial.clone().double()
    start = time.monotonic()
    mass_path = [weighted_mass(z, sigma, weights)]
    count = 0
    nonzero_moves = 0
    for _ in range(sweeps):
        for i, j in PAIRS:
            if lam == 0.:
                theta, _ = best_pair_angle(z[:, i], z[:, j], ANGLE_GRID, sigma)
            else:
                theta, _ = best_pair_angle_weighted(
                    z[:, i], z[:, j], ANGLE_GRID, sigma, weights
                )
            count += len(ANGLE_GRID)
            if abs(theta) <= 1e-12:
                continue
            c, s = math.cos(theta), math.sin(theta)
            vi, vj = read[:, i].clone(), read[:, j].clone()
            zi, zj = z[:, i].clone(), z[:, j].clone()
            read[:, i] = c * vi + s * vj
            read[:, j] = -s * vi + c * vj
            z[:, i] = c * zi + s * zj
            z[:, j] = -s * zi + c * zj
            nonzero_moves += 1
        mass_path.append(weighted_mass(z, sigma, weights))
        if mass_path[-1] < mass_path[-2] - 1e-9:
            raise AssertionError("Unlabelled objective decreased after sweep")
    ortho = float((read.T @ read -
                   torch.eye(DIM, dtype=torch.float64)).abs().max())
    diag = {
        "sigma": sigma, "lambda": lam,
        "training_board_count": int(observed.shape[0]),
        "training_observed_vectors": int(flat.shape[0]),
        "candidate_angle_scores": count,
        "candidate_scalar_projection_evaluations": count * int(flat.shape[0]),
        "training_weight_effective_sample_size": effective_sample_size(weights),
        "training_weight_effective_sample_fraction": effective_sample_size(weights)/len(weights),
        "training_weight_mean": float(weights.mean()),
        "training_weight_min": float(weights.min()),
        "training_weight_max": float(weights.max()),
        "mass_by_sweep": mass_path,
        "initial_mass": mass_path[0],
        "final_mass": mass_path[-1],
        "nonzero_moves": nonzero_moves,
        "orthogonality_max_error": ortho,
        "training_seconds": round(time.monotonic() - start, 6),
        "pair_sweeps": sweeps,
        "extra_weighted_observation_reductions": count * int(flat.shape[0]) if lam != 0 else 0,
    }
    if count != sweeps * 120 * 37 or ortho >= 1e-4:
        raise AssertionError("PD15 registered angle or orthogonality budget")
    return read, diag


def qualifies(row: dict) -> bool:
    sc = CONFIG["scoring"]
    h = row["heldout"]
    if h["alignment"]["mean_cosine"] < sc["mean_read_cos_min"]:
        return False
    if h["overall_mse"] > sc["overall_MSE_max"]:
        return False
    for family in FAMILIES:
        v = h["per_family"][family]
        limit = (sc["unseen_full_fourpass_support_min"]
                 if "unseen" in family else sc["seen_full_fourpass_support_min"])
        if (v["mse"] > sc["per_family_MSE_max"] or
                v["support_fraction"] < limit or not v["binary_mask"]):
            return False
    if row["arm"] == "planted_Q_oracle":
        return True
    t = row["training"]
    return t["nonzero_moves"] > 0 and t["orthogonality_max_error"] < 1e-4


def run_arm(seed: int, arm: str, *, q: torch.Tensor,
            pool: torch.Tensor, pool_sha: str,
            heldout_boards: int | None = None,
            sweeps: int | None = None) -> dict:
    torch.set_num_threads(1)
    t0 = time.monotonic()
    spec = ARM_SPEC[arm]
    initial = initial_random_basis(seed, DIM)
    if spec["kind"] == "oracle":
        read = q.double()
        train = {
            "candidate_angle_scores": 0,
            "candidate_scalar_projection_evaluations": 0,
            "nonzero_moves": 0,
            "training_board_count": 0,
            "orthogonality_max_error": float((read.T@read -
                torch.eye(DIM,dtype=torch.float64)).abs().max()),
        }
    else:
        sigma = CONFIG["widths"][spec["width"]]
        lam = CONFIG["weights"][spec["weight"]]
        read, train = fit_basis(pool, initial, sigma, lam, sweeps=sweeps)
    bpd = physical_model(DIM, read)
    alignment = basis_alignment(read, q)
    n = (CONFIG["source"]["heldout_boards_per_family"]
         if heldout_boards is None else heldout_boards)
    with torch.no_grad():
        per = {
            family: evaluate_family(bpd, read, q, seed, CELL, family, alignment, n)
            for family in FAMILIES
        }
    row = {
        "study_id": CONFIG["study_id"],
        "seed": seed, "cell": CELL["id"], "arm": arm,
        "hidden_rotation_sha256": hashlib.sha256(q.numpy().tobytes()).hexdigest(),
        "initial_random_basis_sha256": hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
        "training_full_pool_sha256": pool_sha,
        "uses_true_Q_for_optimizer": bool(spec["kind"] == "oracle"),
        "training": train,
        "heldout": {
            "alignment": alignment,
            "overall_mse": sum(v["mse"] for v in per.values())/len(per),
            "physical_weight_delta_mse": float(
                bpd.wrappers["core"].weight_delta().square().mean()
            ),
            "per_family": per,
        },
        "walltime_seconds": round(time.monotonic() - t0, 6),
    }
    row["success"] = qualifies(row)
    return row


def independent_objective_diagnostic(seed: int, q: torch.Tensor,
                                     *, boards: int | None = None):
    """Privileged Q is used only to score the fixed off-axis witness."""
    n = (CONFIG["source"]["diagnostic_boards"]
         if boards is None else boards)
    generator = torch.Generator().manual_seed(
        CONFIG["source"]["diagnostic_rng_offset"] + seed
    )
    x, mask, latent = sample_batch(
        generator, n, q, cell=CELL, family="train"
    )
    z = latent.reshape(-1, DIM).double()
    observed_z = x.reshape(-1, DIM).double() @ q.double()
    if not torch.allclose(observed_z, z, atol=2e-6, rtol=0):
        raise AssertionError("Diagnostic Q/latent rotation mismatch")
    witness = four_pair_remix(z)
    sigma = CONFIG["widths"]["wide"]
    cases = {}
    for label, lam in CONFIG["weights"].items():
        w = weights_from_observed(x, lam)
        planted = weighted_mass(z, sigma, w)
        remix = weighted_mass(witness, sigma, w)
        cases[label] = {
            "lambda": lam,
            "planted_mass": planted,
            "four_pair_remix_mass": remix,
            "remix_minus_planted": remix-planted,
            "effective_sample_size": effective_sample_size(w),
            "effective_sample_fraction": effective_sample_size(w)/len(w),
        }
    return {
        "seed": seed, "boards": n,
        "source_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
        "privileged_true_Q_only_for_diagnostic": True,
        "weights": cases,
        "dual_sign": bool(
            cases["none"]["remix_minus_planted"] > 0 and
            cases["norm1"]["remix_minus_planted"] < 0
        )
    }


def summarize(rows: list[dict], diagnostics: list[dict], *,
              registered: bool) -> dict:
    by = {(r["seed"],r["arm"]):r for r in rows}
    diag = {d["seed"]:d for d in diagnostics}
    if (len(rows) != len(by) or
        set(by) != {(seed,arm) for seed in SEEDS for arm in ARMS} or
        len(diag) != len(diagnostics) or set(diag) != set(SEEDS)):
        raise ValueError("Incomplete or duplicate PD15 factorial receipts")
    sc = CONFIG["scoring"]
    primary, control = sc["primary_arm"],sc["primary_control"]
    primary_only = sum(by[(s,primary)]["success"] and
                       not by[(s,control)]["success"] for s in SEEDS)
    control_only = sum(by[(s,control)]["success"] and
                       not by[(s,primary)]["success"] for s in SEEDS)
    p = exact_mcnemar_p(primary_only,control_only)
    counts = {arm:sum(by[(s,arm)]["success"] for s in SEEDS) for arm in ARMS}
    dual = sum(diag[s]["dual_sign"] for s in SEEDS)
    full = bool(registered and len(rows)==160 and len(diagnostics)==32)
    mechanism_pass = bool(
        full and counts[primary] >= sc["primary_min_successes"] and
        primary_only-control_only >= sc["primary_net_paired_wins_min"] and
        p <= sc["primary_exact_mcnemar_p_max"]
    )
    consistency_pass = bool(full and
        dual >= sc["objective_consistency_min_dual_signs"])
    others = [("weighted_narrow","uniform_narrow"),
              ("uniform_narrow","uniform_wide"),
              ("weighted_narrow","weighted_wide")]
    return {
        "study_id":"BPD-PD-15",
        "registered":full,
        "units":len(SEEDS),"arms":len(rows),
        "success_by_arm":counts,
        "Wilson95_by_arm":{
            arm:wilson_ci(counts[arm],len(SEEDS)) for arm in ARMS
        },
        "primary_only":primary_only,"control_only":control_only,
        "net_primary_wins":primary_only-control_only,
        "McNemar_exact_two_sided_p":p,
        "primary_mechanism_gate_passed":mechanism_pass,
        "source_objective_dual_sign_count":dual,
        "objective_consistency_gate_passed":consistency_pass,
        "registered_scientific_passed":mechanism_pass and consistency_pass,
        "secondary_pairs":{
            f"{a}_vs_{b}":{
                "first_only":sum(by[(s,a)]["success"] and not by[(s,b)]["success"] for s in SEEDS),
                "second_only":sum(by[(s,b)]["success"] and not by[(s,a)]["success"] for s in SEEDS),
            } for a,b in others
        },
        "mean_training_effective_fraction_by_arm":{
            arm:(sum(by[(s,arm)]["training"].get("training_weight_effective_sample_fraction",0)
                     for s in SEEDS)/len(SEEDS)) for arm in ARMS
        },
        "per_seed":[{
            "seed":seed,
            "success_by_arm":{arm:by[(seed,arm)]["success"] for arm in ARMS},
            "mean_cos_by_arm":{
                arm:by[(seed,arm)]["heldout"]["alignment"]["mean_cosine"]
                for arm in ARMS
            },
            "k4_unseen_by_arm":{
                arm:by[(seed,arm)]["heldout"]["per_family"]["k4_unseen"]["support_fraction"]
                for arm in ARMS
            },
            "objective_dual_sign":diag[seed]["dual_sign"],
        } for seed in SEEDS]
    }


def run_registered() -> dict:
    start = time.monotonic()
    records, diagnostics = [], []
    for seed in SEEDS:
        q,pool,sha,_,_ = source_for_seed(seed)
        for arm in ARMS:
            records.append(run_arm(seed,arm,q=q,pool=pool,pool_sha=sha))
        diagnostics.append(independent_objective_diagnostic(seed,q))
        rec = [r for r in records if r["seed"]==seed]
        assert len(rec)==5
        for key in ("hidden_rotation_sha256",
                    "initial_random_basis_sha256",
                    "training_full_pool_sha256"):
            if len({r[key] for r in rec}) != 1:
                raise AssertionError(f"Mismatched paired source hash on {seed}: {key}")
        for item in rec:
            if item["arm"]!="planted_Q_oracle":
                t=item["training"]
                if (t["candidate_angle_scores"]!=39960 or
                    t["training_board_count"]!=2048 or
                    t["training_observed_vectors"]!=8192 or
                    t["candidate_scalar_projection_evaluations"]!=327352320 or
                    t["pair_sweeps"]!=9):
                    raise AssertionError("PD15 candidate/sample budget mismatch")
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        if peak > CONFIG["execution"]["max_RSS_GiB"]:
            raise MemoryError(f"PD15 RAM budget exceeded: {peak:.4f} GiB")
    return {
        "study_id":"BPD-PD-15",
        "status":"original_pd15_first_registered_norm_weighted_cohort",
        "source_hashes":{
            path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            for path in ("CONFIG.json","PROTOCOL.md","experiment.py")
        },
        "torch_version":torch.__version__,
        "python_version":platform.python_version(),
        "scipy_version":__import__("scipy").__version__,
        "elapsed_seconds":round(time.monotonic()-start,6),
        "peak_RSS_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        "summary":summarize(records,diagnostics,registered=True),
        "diagnostics":diagnostics,
        "arms":records,
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    outcome=run_registered()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(outcome,indent=2,sort_keys=True)+"\n")
    print(json.dumps(outcome["summary"],indent=2,sort_keys=True))


if __name__=="__main__":
    main()
