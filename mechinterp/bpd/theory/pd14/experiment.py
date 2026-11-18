"""BPD-PD-14 prospective matched-angle smoothing and restart experiment.

Scientific seeds/decision frozen in CONFIG.json and PROTOCOL.md before this
implementation. The fitted optimizer accepts dense observed x, seed, sigma and
a fixed search mode. Ground-truth Q is used only to generate samples and after
training to evaluate actual BPD interventions or privileged diagnostics.
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
from mechinterp.bpd.theory.pd13.posthoc_zero_mass_audit import (
    four_pair_remix, gaussian_mass
)

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
CELLS = tuple(CONFIG["seed_cells"])
ARMS = tuple(a["key"] for a in CONFIG["arms"])
ARM_SPEC = {a["key"]: a for a in CONFIG["arms"]}
FAMILIES = tuple(CONFIG["source"]["heldout_families"])
DIM = 16
PAIRS = tuple((i, j) for i in range(DIM) for j in range(i + 1, DIM))
ANGLE_GRID = angle_grid("global37")
assert len(PAIRS) == 120 and len(ANGLE_GRID) == 37


def source_for_seed(seed: int, cell: dict, *, boards: int | None = None):
    """Exactly the frozen PD12 source law, with one shared full training pool."""
    if cell["dimension"] != 16 or cell["support_distribution"] != "clustered":
        raise ValueError("PD14 source contract requires clustered 16D")
    torch.set_num_threads(1)
    src = CONFIG["source"]
    count = src["train_boards"] if boards is None else boards
    gen = torch.Generator().manual_seed(src["train_rng_offset"] + seed)
    q = rotation_for_seed(seed, DIM)
    x, support, latent = sample_batch(gen, count, q, cell=cell, family="train")
    pool_sha = hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
    return q, x, pool_sha, support, latent


def initial_basis_for_restart(seed: int, restart: int) -> torch.Tensor:
    """The source-independent, seeded orthogonal restart family."""
    if restart not in (0, 1, 2):
        raise ValueError("Only three registered restart indices")
    if restart == 0:
        return initial_random_basis(seed, DIM)
    gen = torch.Generator().manual_seed(610000 + seed + 1000003 * restart)
    raw = torch.randn(DIM, DIM, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(raw)
    signs = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0)
    return q * signs.unsqueeze(0)


def _one_givens_search(observed: torch.Tensor, initial: torch.Tensor,
                       sigma: float, sweeps: int):
    """Unlabelled lexicographic global37 Givens optimizer, same as PD12."""
    n = int(observed.reshape(-1, DIM).shape[0])
    z = observed.reshape(-1, DIM).double() @ initial.double()
    v = initial.clone().double()
    masses = [float(torch.exp(-.5 * (z / sigma).square()).mean())]
    updates = 0
    angles = 0
    start = time.monotonic()
    for _ in range(sweeps):
        for i, j in PAIRS:
            theta, _ = best_pair_angle(z[:, i], z[:, j], ANGLE_GRID, sigma)
            angles += len(ANGLE_GRID)
            if abs(theta) <= 1e-12:
                continue
            c, s = math.cos(theta), math.sin(theta)
            vi, vj = v[:, i].clone(), v[:, j].clone()
            zi, zj = z[:, i].clone(), z[:, j].clone()
            v[:, i] = c * vi + s * vj
            v[:, j] = -s * vi + c * vj
            z[:, i] = c * zi + s * zj
            z[:, j] = -s * zi + c * zj
            updates += 1
        masses.append(float(torch.exp(-.5 * (z / sigma).square()).mean()))
        if masses[-1] < masses[-2] - 1e-9:
            raise AssertionError("Givens objective must be nondecreasing")
    orth_err = float((v.T @ v - torch.eye(DIM, dtype=torch.float64)).abs().max())
    return v, {
        "initial_mass": masses[0],
        "final_mass": masses[-1],
        "mass_by_sweep": masses,
        "nonzero_moves": updates,
        "candidate_angle_scores": angles,
        "candidate_scalar_projection_evaluations": angles * n,
        "orthogonality_max_error": orth_err,
        "search_seconds": round(time.monotonic() - start, 6),
        "sweeps": sweeps,
    }


def fit_basis(observed: torch.Tensor, seed: int, sigma: float,
              search: str, *, sweeps_per_restart: int | None = None):
    """Fit from observed x only; no Q/support/gold/heldout arguments."""
    if search not in ("single_long", "three_restarts"):
        raise ValueError(search)
    if observed.shape[-1] != DIM or observed.ndim != 4:
        raise ValueError("Expected [boards, 4, 1, 16]")
    if not sigma > 0:
        raise ValueError("Positive frozen Gaussian width required")
    base_sweeps = (
        CONFIG["optimizer"]["restart_sweeps"]
        if sweeps_per_restart is None else sweeps_per_restart
    )
    if base_sweeps < 1:
        raise ValueError("At least one sweep per restart")
    restarts = 1 if search == "single_long" else 3
    sweeps_each = 3 * base_sweeps if search == "single_long" else base_sweeps
    candidates = []
    for r in range(restarts):
        initial = initial_basis_for_restart(seed, r)
        read, diag = _one_givens_search(observed, initial, sigma, sweeps_each)
        init_sha = hashlib.sha256(initial.numpy().tobytes()).hexdigest()
        candidates.append((read, diag, init_sha))
    best_score = max(c[1]["final_mass"] for c in candidates)
    idx = min(i for i, c in enumerate(candidates)
              if best_score - c[1]["final_mass"] <= 1e-12)
    chosen, selected_diag, _ = candidates[idx]
    total_scores = sum(c[1]["candidate_angle_scores"] for c in candidates)
    vectors = int(observed.reshape(-1, DIM).shape[0])
    stats = {
        "sigma": sigma,
        "search": search,
        "training_board_count": int(observed.shape[0]),
        "training_observed_vectors": vectors,
        "initial_basis_sha256_by_restart": [c[2] for c in candidates],
        "selected_restart_index": idx,
        "restart_final_masses": [c[1]["final_mass"] for c in candidates],
        "restart_initial_masses": [c[1]["initial_mass"] for c in candidates],
        "mass_paths_by_restart": [c[1]["mass_by_sweep"] for c in candidates],
        "selected_final_mass": selected_diag["final_mass"],
        "selected_nonzero_moves": selected_diag["nonzero_moves"],
        "total_nonzero_moves": sum(c[1]["nonzero_moves"] for c in candidates),
        "candidate_angle_scores": total_scores,
        "candidate_scalar_projection_evaluations": total_scores * vectors,
        "orthogonality_max_error": selected_diag["orthogonality_max_error"],
        "all_orthogonality_errors": [c[1]["orthogonality_max_error"]
                                    for c in candidates],
        "search_seconds": round(sum(c[1]["search_seconds"] for c in candidates), 6),
        "candidate_sweeps_total": sum(c[1]["sweeps"] for c in candidates),
    }
    if total_scores != (3 * base_sweeps * 120 * 37):
        raise AssertionError("Unmatched sample-independent angular score budget")
    if max(stats["all_orthogonality_errors"]) >= 1e-4:
        raise AssertionError("Orthogonal dictionary drift")
    return chosen, stats


def qualifies(row: dict) -> bool:
    sc = CONFIG["scoring"]
    e = row["heldout"]
    if e["alignment"]["mean_cosine"] < sc["mean_read_cos_min"]:
        return False
    if e["overall_mse"] > sc["overall_MSE_max"]:
        return False
    for family in FAMILIES:
        v = e["per_family"][family]
        threshold = (sc["unseen_full_fourpass_support_min"]
                     if "unseen" in family else
                     sc["seen_full_fourpass_support_min"])
        if (v["mse"] > sc["per_family_MSE_max"] or
                v["support_fraction"] < threshold or not v["binary_mask"]):
            return False
    if row["arm"] == "planted_Q_oracle":
        return True
    tr = row["training"]
    return (tr["selected_nonzero_moves"] > 0 and
            tr["orthogonality_max_error"] < 1e-4)


def run_arm(seed: int, cell: dict, arm: str, *, q: torch.Tensor,
            pool: torch.Tensor, pool_sha: str,
            heldout_boards: int | None = None,
            sweeps_per_restart: int | None = None) -> dict:
    torch.set_num_threads(1)
    start = time.monotonic()
    spec = ARM_SPEC[arm]
    if spec["kind"] == "oracle":
        read = q.double()
        stats = {
            "candidate_angle_scores": 0,
            "candidate_scalar_projection_evaluations": 0,
            "selected_nonzero_moves": 0,
            "selected_restart_index": None,
            "search_seconds": 0.0,
            "training_board_count": 0,
            "orthogonality_max_error": float((
                read.T @ read - torch.eye(DIM, dtype=torch.float64)
            ).abs().max()),
        }
    else:
        sigma = CONFIG["widths"][spec["width"]][cell["margin"]]
        read, stats = fit_basis(
            pool, seed, sigma, spec["search"],
            sweeps_per_restart=sweeps_per_restart
        )
    bpd = physical_model(DIM, read)
    matched = basis_alignment(read, q)
    n = (CONFIG["source"]["heldout_boards_per_family"]
         if heldout_boards is None else heldout_boards)
    with torch.no_grad():
        per = {
            family: evaluate_family(bpd, read, q, seed, cell, family, matched, n)
            for family in FAMILIES
        }
    row = {
        "study_id": CONFIG["study_id"], "seed": seed,
        "cell": cell["id"], "margin": cell["margin"], "arm": arm,
        "uses_true_Q_for_optimizer": bool(spec["kind"] == "oracle"),
        "hidden_rotation_sha256": hashlib.sha256(q.numpy().tobytes()).hexdigest(),
        "training_full_pool_sha256": pool_sha,
        "initial_random_basis_sha256": hashlib.sha256(
            initial_basis_for_restart(seed, 0).numpy().tobytes()
        ).hexdigest(),
        "training": stats,
        "heldout": {
            "alignment": matched,
            "overall_mse": sum(v["mse"] for v in per.values()) / len(per),
            "physical_weight_delta_mse": float(
                bpd.wrappers["core"].weight_delta().square().mean()),
            "per_family": per,
        },
        "walltime_seconds": round(time.monotonic() - start, 6),
    }
    row["success"] = qualifies(row)
    return row


def objective_diagnostic(seed: int, cell: dict, q: torch.Tensor,
                         *, boards: int | None = None) -> dict:
    """Privileged post-fit diagnostic, never fed into fit_basis."""
    src = CONFIG["source"]
    n = src["diagnostic_boards"] if boards is None else boards
    generator = torch.Generator().manual_seed(src["diagnostic_rng_offset"] + seed)
    x, supports, latent = sample_batch(generator, n, q, cell=cell, family="train")
    z = (x.reshape(-1, DIM).double() @ q.double())
    # The float Q introduces <1e-6 projection noise; use latent for exact
    # planted-coordinate diagnostic as prescribed by the source contract.
    z_truth = latent.reshape(-1, DIM).double()
    if not torch.allclose(z, z_truth, atol=2e-6, rtol=0):
        raise AssertionError("Diagnostic latent source alignment mismatch")
    vals = {}
    for width in ("wide", "narrow"):
        sigma = CONFIG["widths"][width][cell["margin"]]
        planted = gaussian_mass(z_truth, sigma)
        mixed = gaussian_mass(four_pair_remix(z_truth), sigma)
        vals[width] = {
            "sigma": sigma, "planted_mass": planted,
            "four_pair_remix_mass": mixed,
            "remix_minus_planted": mixed - planted,
        }
    return {
        "seed": seed, "margin": cell["margin"],
        "boards": n,
        "source_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
        "privileged_true_Q_only_for_diagnostic": True,
        "widths": vals,
        "dual_sign_strong": bool(
            cell["margin"] == "strong" and
            vals["wide"]["remix_minus_planted"] > 0 and
            vals["narrow"]["remix_minus_planted"] < 0
        ),
    }


def summarize(rows: list[dict], diagnostics: list[dict], *,
              registered: bool) -> dict:
    seeds = [s for c in CELLS for s in c["seeds"]]
    by = {(r["seed"], r["arm"]): r for r in rows}
    diag = {r["seed"]: r for r in diagnostics}
    if (len(by) != len(rows) or
        set(by) != {(s, a) for s in seeds for a in ARMS} or
        len(diag) != len(seeds) or set(diag) != set(seeds)):
        raise ValueError("Incomplete or duplicate PD14 prospective factorial")
    sc = CONFIG["scoring"]
    strong = CELLS[0]["seeds"]
    target, control = sc["primary_arm"], sc["primary_control"]
    a = sum(by[(s, target)]["success"] and not by[(s, control)]["success"]
            for s in strong)
    b = sum(by[(s, control)]["success"] and not by[(s, target)]["success"]
            for s in strong)
    p = exact_mcnemar_p(a, b)
    strong_success = sum(by[(s, target)]["success"] for s in strong)
    dual = sum(diag[s]["dual_sign_strong"] for s in strong)
    is_full = bool(registered and len(rows) == 160 and len(diagnostics) == 32)
    mechanism_pass = bool(
        is_full and strong_success >= sc["primary_min_narrow_strong_successes"]
        and a - b >= sc["primary_net_paired_wins_min"]
        and p <= sc["primary_exact_mcnemar_p_max"]
    )
    consistency_pass = bool(
        is_full and dual >= sc["objective_consistency_min_dual_signs"]
    )
    paired_secondary = {}
    comparisons = [
        ("wide_restart", "wide_single"),
        ("narrow_restart", "narrow_single"),
        ("narrow_restart", "wide_restart"),
    ]
    for cell in CELLS:
        paired_secondary[cell["id"]] = {}
        for newer, older in comparisons:
            paired_secondary[cell["id"]][f"{newer}_vs_{older}"] = {
                "newer_only": sum(by[(s, newer)]["success"] and not by[(s, older)]["success"]
                                  for s in cell["seeds"]),
                "older_only": sum(by[(s, older)]["success"] and not by[(s, newer)]["success"]
                                  for s in cell["seeds"]),
            }
    return {
        "study_id": CONFIG["study_id"],
        "registered": is_full,
        "units": len(seeds), "arms": len(rows),
        "success_by_arm": {
            a: sum(by[(s, a)]["success"] for s in seeds) for a in ARMS
        },
        "success_by_cell": {
            c["id"]: {
                a: sum(by[(s, a)]["success"] for s in c["seeds"])
                for a in ARMS
            } for c in CELLS
        },
        "Wilson95_by_arm": {
            a: wilson_ci(sum(by[(s, a)]["success"] for s in seeds), len(seeds))
            for a in ARMS
        },
        "strong_narrow_single_successes": strong_success,
        "strong_narrow_only": a,
        "strong_wide_only": b,
        "strong_net_paired_wins": a - b,
        "strong_McNemar_exact_two_sided_p": p,
        "strong_objective_dual_sign_count": dual,
        "objective_consistency_gate_passed": consistency_pass,
        "primary_mechanism_gate_passed": mechanism_pass,
        "registered_scientific_passed": mechanism_pass and consistency_pass,
        "paired_secondary": paired_secondary,
        "per_seed": [{
            "seed": s,
            "cell": next(c["id"] for c in CELLS if s in c["seeds"]),
            "success_by_arm": {a: by[(s, a)]["success"] for a in ARMS},
            "mean_read_cosine_by_arm": {
                a: by[(s, a)]["heldout"]["alignment"]["mean_cosine"] for a in ARMS
            },
            "k4_unseen_exact_by_arm": {
                a: by[(s, a)]["heldout"]["per_family"]["k4_unseen"]["support_fraction"]
                for a in ARMS
            },
        } for s in seeds],
    }


def run_registered() -> dict:
    start = time.monotonic()
    rows, diagnostics = [], []
    for cell in CELLS:
        for seed in cell["seeds"]:
            q, pool, sha, _, _ = source_for_seed(seed, cell)
            for arm in ARMS:
                row = run_arm(seed, cell, arm, q=q, pool=pool, pool_sha=sha)
                rows.append(row)
            diagnostics.append(objective_diagnostic(seed, cell, q))
            same = [r for r in rows if r["seed"] == seed]
            for key in ("hidden_rotation_sha256", "training_full_pool_sha256",
                        "initial_random_basis_sha256"):
                if len({r[key] for r in same}) != 1:
                    raise AssertionError(f"Mismatch on paired source {seed}: {key}")
            for r in same:
                if r["arm"] != "planted_Q_oracle":
                    t = r["training"]
                    if (t["candidate_angle_scores"] != 39960 or
                        t["candidate_scalar_projection_evaluations"] != 327352320 or
                        t["training_board_count"] != 2048 or
                        t["candidate_sweeps_total"] != 9):
                        raise AssertionError("PD14 matched angular budget failure")
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            if peak > CONFIG["execution"]["max_RSS_GiB"]:
                raise MemoryError(f"PD14 RSS {peak:.3f} GiB exceeds memory contract")
    return {
        "study_id": CONFIG["study_id"],
        "status": "original_pd14_first_registered_bandwidth_restart_cohort",
        "source_hashes": {
            f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
            for f in ("CONFIG.json", "PROTOCOL.md", "experiment.py")
        },
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "scipy_version": __import__("scipy").__version__,
        "elapsed_seconds": round(time.monotonic() - start, 6),
        "peak_RSS_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        "summary": summarize(rows, diagnostics, registered=True),
        "diagnostics": diagnostics,
        "arms": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run_registered()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
