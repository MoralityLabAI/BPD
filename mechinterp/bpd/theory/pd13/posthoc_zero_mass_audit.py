"""Post-hoc objective inversion audit for frozen BPD-PD-13, without training.

Usage:
python -m mechinterp.bpd.theory.pd13.posthoc_zero_mass_audit \
  --receipt bpd-pd13-original-first-run.json --csv-out audit.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

from mechinterp.bpd.theory.pd12.experiment import support_banks
from mechinterp.bpd.theory.pd13.experiment import CELLS, DIM, pool_for_seed


def gaussian_mass(z: torch.Tensor, sigma: float) -> float:
    return float(torch.exp(-0.5 * (z.double() / sigma).square()).mean())


def four_pair_remix(z: torch.Tensor) -> torch.Tensor:
    out = z.clone().double()
    for i in (0, 4, 8, 12):
        a, b = z[:, i].double(), z[:, i + 1].double()
        out[:, i] = (a + b) / math.sqrt(2)
        out[:, i + 1] = (b - a) / math.sqrt(2)
    return out


def support_joint_probabilities(pair: tuple[int, int] = (0, 1)) -> dict[str, float]:
    banks = support_banks(DIM)
    p = {"00": 0., "10": 0., "01": 0., "11": 0.}
    for k in range(1, 5):
        arms = ((1., banks["train"][k]),) if k == 1 else (
            (0.85, banks["cluster"][k]), (0.15, banks["train"][k])
        )
        for weight, supports in arms:
            for support in supports:
                key = f"{int(pair[0] in support)}{int(pair[1] in support)}"
                p[key] += 0.25 * weight / len(supports)
    assert abs(sum(p.values()) - 1) < 1e-12
    return p


def population_comparison(margin: str, sigma: float, nodes: int = 256) -> dict:
    """Exact support weights, numerical uniform-amplitude integration.

    Pair score-sum difference:
    D = (p10+p01)*(2h-1-g) + p11*(c-2g),
    g=E exp(-A^2/(2 sigma^2)), h=E exp(-A^2/(4 sigma^2)),
    c=E[exp(-(A+B)^2/(4 sigma^2)) + exp(-(A-B)^2/(4 sigma^2))].
    """
    lo, hi = (.75, 1.25) if margin == "strong" else (.25, 1.25)
    p = support_joint_probabilities()
    t, w = leggauss(nodes)
    a = lo + (t + 1) * (hi - lo) / 2
    w = w / 2
    g = float(w @ np.exp(-.5 * (a / sigma)**2))
    h = float(w @ np.exp(-.25 * (a / sigma)**2))
    plus = np.exp(-.25 * ((a[:, None] + a[None, :]) / sigma)**2)
    minus = np.exp(-.25 * ((a[:, None] - a[None, :]) / sigma)**2)
    c = float(w @ (plus + minus) @ w)
    d = (p["10"] + p["01"]) * (2*h-1-g) + p["11"] * (c-2*g)
    planted = 1 - (2.5 / DIM) * (1 - g)
    return {"margin": margin, "sigma": sigma, "p": p,
            "g": g, "h": h, "c": c, "pair_sum_gain": d,
            "planted_mass": planted, "four_pair_remix_mass": planted + 4*d/DIM,
            "four_pair_gain": 4*d/DIM}


def replay(receipt_path: Path | None) -> list[dict]:
    receipt = json.loads(receipt_path.read_text()) if receipt_path else None
    rows = {(r["seed"], r["arm"]): r for r in receipt["arms"]} if receipt else {}
    if receipt:
        assert receipt["study_id"] == "BPD-PD-13"
        assert receipt["summary"]["registered"] and len(receipt["arms"]) == 160
    out = []
    for cell in CELLS:
        margin = cell["margin"]
        sigma = .15 if margin == "strong" else .05
        for seed in cell["seeds"]:
            q, x, pool_sha, support, latent = pool_for_seed(seed, cell)
            latent = latent.reshape(-1, DIM).double()
            q_sha = hashlib.sha256(q.numpy().tobytes()).hexdigest()
            if receipt:
                origin = rows[(seed, "priority2048")]
                assert pool_sha == origin["training_full_pool_sha256"]
                assert q_sha == origin["hidden_rotation_sha256"]
            planted = gaussian_mass(latent, sigma)
            remixed = gaussian_mass(four_pair_remix(latent), sigma)
            out.append({"seed": seed, "margin": margin, "pool_sha256": pool_sha,
                        "rotation_sha256": q_sha, "planted_mass": planted,
                        "remixed_mass": remixed, "gain": remixed - planted})
    assert len(out) == 32
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--csv-out", type=Path)
    ap.add_argument("--population-only", action="store_true")
    args = ap.parse_args()
    for margin, sigma in (("strong", .15), ("weak", .05)):
        x = population_comparison(margin, sigma)
        x["crossover_sigma"] = float(brentq(
            lambda s: population_comparison(margin, s)["pair_sum_gain"], .01, .2
        ))
        print(json.dumps(x, sort_keys=True))
    if args.population_only:
        return
    rows = replay(args.receipt)
    for margin in ("strong", "weak"):
        gains = [x["gain"] for x in rows if x["margin"] == margin]
        print(json.dumps({"margin": margin, "units": len(gains),
                          "remix_wins": sum(x > 0 for x in gains),
                          "mean_gain": mean(gains), "min_gain": min(gains),
                          "max_gain": max(gains)}))
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
