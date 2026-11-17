"""BPD v3: source-only Jensen-Shannon floor and exact-basis off-budget controls.

Native data must be passed in explicitly. This module never obtains checkpoints,
assumes availability of a GPU, or calls the protected evaluation dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BUDGETS = (0.01, 0.03, 0.1)


def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Elementwise categorical KL on the last axis; broadcasts all others."""
    p, q = np.broadcast_arrays(np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64))
    if np.any(q <= 0) or np.any(p < 0):
        raise ValueError("Invalid probability: q must be positive, p nonnegative")
    return np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300) / q), 0.0), axis=-1)


def floor_report(target: np.ndarray, static: np.ndarray | None = None,
                 driven: np.ndarray | None = None,
                 independent: np.ndarray | None = None) -> dict:
    """All inputs logits. Target shape [board, dither, readout, cell, class].

    static shape [board, readout, cell, class] is independent of the target dither.
    driven shape [board, dither, readout, cell, class] is matched to the realized dither.
    independent shape [board, dither, readout, cell, class] measures an ensemble only.
    """
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 5 or min(target.shape) < 1 or target.shape[1] < 2:
        raise ValueError("target shape must be [B,K,R,P,V] with K >= 2")
    if not np.isfinite(target).all():
        raise ValueError("nonfinite target logits")
    p = softmax(target)
    mean_p = p.mean(axis=1)
    js = kl(p, mean_p[:, None, ...])  # [B,K,R,P]
    js_by_board_readout = js.mean(axis=(1, 3))
    out = {
        "n_boards": int(target.shape[0]),
        "n_dithers": int(target.shape[1]),
        "n_readouts": int(target.shape[2]),
        "n_cells": int(target.shape[3]),
        "j_delta_nats_per_cell": float(js.mean()),
        "j_by_readout": [float(z) for z in js.mean(axis=(0, 1, 3))],
        "j_by_board_readout": js_by_board_readout.tolist(),
        "j_positive_board_count": int(np.count_nonzero(js_by_board_readout.mean(axis=1) > 1e-8)),
    }
    if static is not None:
        static = np.asarray(static, dtype=np.float64)
        if static.shape != (target.shape[0], *target.shape[2:]):
            raise ValueError("static must have [B,R,P,V] shape")
        qs = softmax(static)
        actual = float(kl(p, qs[:, None, ...]).mean())
        bary = float(kl(mean_p, qs).mean())
        out["noise_independent_risk"] = actual
        out["barycentre_kl"] = bary
        out["risk_identity_residual"] = actual - (float(js.mean()) + bary)
    if driven is not None:
        driven = np.asarray(driven, dtype=np.float64)
        if driven.shape != target.shape:
            raise ValueError("driven must match target shape")
        matched_risk = float(kl(p, softmax(driven)).mean())
        out["paired_driven_risk"] = matched_risk
        out["conditional_leak_lower_bound"] = float(js.mean()) - matched_risk
        out["leak_bound_requires_drive_markov_chain"] = True
    if independent is not None:
        independent = np.asarray(independent, dtype=np.float64)
        if independent.shape != target.shape:
            raise ValueError("independent must match target shape")
        mean_q = softmax(independent).mean(axis=1)
        out["ensemble_mean_kl"] = float(kl(mean_p, mean_q).mean())
    return out


def basis_factorization(w: np.ndarray, family: str, kind: str, seed: int = 0):
    """Return exact V [d_in,C], U [C,d_out] with (V @ U).T = W."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2 or min(w.shape) == 0:
        raise ValueError("W must be a nonempty [d_out,d_in] array")
    dout, din = w.shape
    rng = np.random.default_rng(seed)
    if family == "input":
        if kind == "neuron":
            v = np.eye(din)
        elif kind == "svd":
            v = np.linalg.svd(w, full_matrices=True)[2].T
        elif kind == "random":
            v = np.linalg.qr(rng.normal(size=(din, din)))[0]
        else:
            raise ValueError("input kind: neuron, svd, random")
        u = np.linalg.solve(v, w.T)
    elif family == "output":
        if kind == "coordinate":
            u = np.eye(dout)
        elif kind == "svd":
            u = np.linalg.svd(w, full_matrices=True)[0].T
        elif kind == "random":
            u = np.linalg.qr(rng.normal(size=(dout, dout)))[0].T
        else:
            raise ValueError("output kind: coordinate, svd, random")
        v = np.linalg.solve(u.T, w).T
    else:
        raise ValueError("family: input or output")
    return v, u


def budget_report(w: np.ndarray, inputs: np.ndarray, v: np.ndarray, u: np.ndarray,
                  budgets: tuple[float, ...] = BUDGETS,
                  gramian: np.ndarray | None = None) -> dict:
    """Ascending off-contribution gate, using only the current module input.

    G is a PSD metric on the output; for nonlinear delayed effects the certificate
    is first-order and needs a separate curvature remainder.
    """
    w, inputs = np.asarray(w, np.float64), np.asarray(inputs, np.float64)
    v, u = np.asarray(v, np.float64), np.asarray(u, np.float64)
    dout, din = w.shape
    if inputs.ndim != 2 or inputs.shape[1] != din or v.shape[0] != din or u.shape != (v.shape[1], dout):
        raise ValueError("shape mismatch among W, inputs, V, U")
    g = np.eye(dout) if gramian is None else np.asarray(gramian, np.float64)
    if g.shape != (dout, dout) or np.linalg.eigvalsh((g + g.T) / 2).min() < -1e-9:
        raise ValueError("G must be positive semidefinite")
    norm_g = lambda z: np.sqrt(np.maximum(np.einsum('...i,ij,...j->...', z, g, z), 0.0))
    full = inputs @ w.T
    activ = inputs @ v
    write_norms = norm_g(u)
    iota = np.abs(activ) * write_norms[None, :]
    residual = float(np.linalg.norm((v @ u).T - w) / max(np.linalg.norm(w), 1e-300))
    reports = {}
    for beta in budgets:
        off = np.zeros_like(iota, dtype=bool)
        output_norm = norm_g(full)
        for b in range(inputs.shape[0]):
            bound = float(beta) * output_norm[b]
            total = 0.0
            for c in np.argsort(iota[b], kind="stable"):
                if total + iota[b, c] <= bound + 1e-14:
                    off[b, c] = True
                    total += iota[b, c]
                else:
                    break
        drop = (activ * off) @ u
        actual = norm_g(drop)
        certificate = np.sum(iota * off, axis=1)
        out_ablate = full - drop
        reports[str(beta)] = {
            "mean_off": float(off.sum(axis=1).mean()),
            "mean_active": float(u.shape[0] - off.sum(axis=1).mean()),
            "mean_active_fraction": float(1.0 - off.sum(axis=1).mean() / u.shape[0]),
            "max_actual_minus_certificate": float(np.max(actual - certificate)),
            "max_certificate_minus_budget": float(np.max(certificate - beta * output_norm)),
            "mean_relative_actual": float(np.mean(actual / np.maximum(output_norm, 1e-12))),
            "mean_relative_certificate": float(np.mean(certificate / np.maximum(output_norm, 1e-12))),
            "mean_logit_kl": float(kl(softmax(full), softmax(out_ablate)).mean()),
        }
    return {"family_components": int(u.shape[0]), "relative_full_on_residual": residual,
            "budgets": reports}


def phase_pool_check(seed: int = 331) -> dict:
    rng = np.random.default_rng(seed)
    x_dense = rng.normal(size=(256, 2))
    x_1 = np.column_stack([rng.normal(size=256), np.zeros(256)])
    x_2 = np.column_stack([np.zeros(256), rng.normal(size=256)])
    x = np.vstack([x_dense, x_1, x_2])
    angles = np.arange(0, 17) * np.pi / 32
    active = []
    dense_only = []
    for a in angles:
        c, s = np.cos(a), np.sin(a)
        rot = np.array([[c, -s], [s, c]])
        active.append(float(np.mean(np.sum(np.abs(x @ rot) > 1e-10, axis=1))))
        dense_only.append(float(np.mean(np.sum(np.abs(x_dense @ rot) > 1e-10, axis=1))))
    return {"angles_radians": angles.tolist(), "pooled_active": active,
            "full_support_phase_active": dense_only,
            "coordinate_advantage_vs_45deg": float(active[8] - active[0]),
            "full_support_phase_range": float(max(dense_only) - min(dense_only)),
            "pooled_minimizer_grid_indices": [i for i, x in enumerate(active) if abs(x - min(active)) < 1e-10]}


def toy_logits(seed: int = 711, boards: int = 24, k: int = 8) -> np.ndarray:
    """Bistable 63-step toy with additive per-call dither, read at 20/41/62."""
    rng = np.random.default_rng(seed)
    logits = np.zeros((boards, k, 3, 2, 2), np.float64)
    readout = {20: 0, 41: 1, 62: 2}
    for b in range(boards):
        for draw in range(k):
            z = 0.0 if b < boards * 3 // 4 else (0.15 if b % 2 else -0.15)
            for t in range(63):
                z = float(np.tanh(4 * z + rng.uniform(-0.002, 0.002)))
                if t in readout:
                    j = readout[t]
                    for cell in range(2):
                        logits[b, draw, j, cell] = (6 * z + (cell - 0.5) * 0.03,
                                                   -6 * z - (cell - 0.5) * 0.03)
    return logits


def run_smoke(seed: int = 711) -> dict:
    logits = toy_logits(seed=seed)
    qbar = softmax(logits).mean(axis=1)
    floor = floor_report(logits, static=np.log(qbar), driven=logits,
                         independent=toy_logits(seed=seed + 25))
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(6, 8)) / np.sqrt(8)
    inputs = rng.normal(size=(512, 8))
    mask = rng.random(size=inputs.shape) < .72
    inputs[mask] = 0.0
    matrices = {}
    for family, kinds in (("input", ("neuron", "svd", "random")),
                          ("output", ("coordinate", "svd", "random"))):
        for kind in kinds:
            v, u = basis_factorization(w, family, kind, seed=seed + 1)
            matrices[f"{family}/{kind}"] = budget_report(w, inputs, v, u)
    phase = phase_pool_check(seed=seed)
    assert abs(floor["risk_identity_residual"]) < 1e-9
    assert floor["j_delta_nats_per_cell"] > 0
    assert abs(floor["paired_driven_risk"]) < 1e-10
    assert phase["coordinate_advantage_vs_45deg"] > 0.2
    for result in matrices.values():
        assert result["relative_full_on_residual"] < 1e-11
        assert all(z["max_actual_minus_certificate"] < 1e-10 and
                   z["max_certificate_minus_budget"] < 1e-10
                   for z in result["budgets"].values())
        vals = [z["mean_off"] for z in result["budgets"].values()]
        assert vals == sorted(vals)
    return {"label": "SYNTHETIC_SOURCE_ONLY", "seed": seed, "floor": floor,
            "phase_pool": phase, "basis": matrices,
            "native_gaoxin492": "PENDING_NO_LOCAL_PINNED_CHECKPOINT",
            "native_yagizdevre": "PENDING_NO_LOCAL_PINNED_CHECKPOINT"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--target-npz", type=Path, help="Must contain target_logits [B,K,R,P,V]")
    ap.add_argument("--matrix-npz", type=Path, help="Must contain W [d_out,d_in], inputs [N,d_in]")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if not any((args.smoke, args.target_npz, args.matrix_npz)):
        ap.error("supply --smoke or explicit local .npz input")
    result = {}
    if args.smoke:
        result["synthetic"] = run_smoke()
    if args.target_npz:
        with np.load(args.target_npz, allow_pickle=False) as d:
            result["native_dither_audit"] = floor_report(
                d["target_logits"],
                static=d["static_logits"] if "static_logits" in d.files else None,
                driven=d["driven_logits"] if "driven_logits" in d.files else None,
                independent=d["independent_logits"] if "independent_logits" in d.files else None)
    if args.matrix_npz:
        with np.load(args.matrix_npz, allow_pickle=False) as d:
            w, inputs = d["W"], d["inputs"]
        result["matrix_audit"] = {}
        for family, kinds in (("input", ("neuron", "svd", "random")),
                              ("output", ("coordinate", "svd", "random"))):
            for kind in kinds:
                v, u = basis_factorization(w, family, kind)
                result["matrix_audit"][f"{family}/{kind}"] = budget_report(w, inputs, v, u)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
