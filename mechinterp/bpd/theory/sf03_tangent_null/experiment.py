"""BPD-SF03: gradient-matched dark-fiber shams and source-Jacobian profile.

Reuses frozen SF02 nonlinear teacher and factor selection. Prospective roots,
thresholds, and evidence boundaries are pinned in PROTOCOL.md and LOCK.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from mechinterp.bpd.theory.sf02_secret_circuit.experiment import (
    D, C, train_teacher, train_factors, donor_pairs, calibrate,
    factor_writes, patch, init_rng,
)

torch.set_num_threads(1)


def draw_grad_matched(delta, grad, rng, random_sign=False):
    """Randomize the dark orthogonal part with exact norm and tangent matching."""
    grad = grad.clone()
    grad[:, 0] = 0
    gnorm = grad.norm(dim=1, keepdim=True)
    u = grad / gnorm.clamp_min(1e-8)
    rho = delta.norm(dim=1, keepdim=True)
    alpha = (delta * u).sum(dim=1, keepdim=True)
    perp = (rho.square() - alpha.square()).clamp_min(0).sqrt()
    rv = torch.randn(delta.shape, generator=rng)
    rv[:, 0] = 0
    rv = rv - (rv * u).sum(dim=1, keepdim=True) * u
    rv = rv / rv.norm(dim=1, keepdim=True).clamp_min(1e-8)
    if random_sign:
        flip = torch.where(torch.rand((len(delta), 1), generator=rng) < .5, -1., 1.)
    else:
        flip = 1.
    direction = (flip * alpha) * u + perp * rv
    direction[:, 0] = 0
    norm_error = float((direction.norm(dim=1) - rho[:, 0]).abs().max())
    dot_error = float(((grad * direction).sum(1).abs() -
                       (grad * delta).sum(1).abs()).abs().max())
    signed_error = (float(((grad * direction).sum(1) -
                           (grad * delta).sum(1)).abs().max())
                    if not random_sign else None)
    return direction, norm_error, dot_error, signed_error


def layer_gradients(teacher, h, role):
    x = h.detach().requires_grad_(True)
    q1 = teacher(x)[:, 0]
    q2 = teacher(teacher(x))[:, 0]
    g1 = torch.autograd.grad(q1.sum(), x, retain_graph=True)[0]
    g2 = torch.autograd.grad(q2.sum(), x)[0]
    own = torch.where(role == 1, 2, 1)
    other = torch.where(role == 1, 1, 2)
    ar = torch.arange(len(h))
    auth = role != 2
    public = role == 2

    def mean(t):
        return float(t.detach().mean()) if t.numel() else 0.

    return {
        "q1_own_abs_grad_auth": mean(g1[ar, own][auth].abs()),
        "q2_own_abs_grad_auth": mean(g2[ar, own][auth].abs()),
        "q2_other_abs_grad_auth": mean(g2[ar, other][auth].abs()),
        "q2_private_abs_grad_public": mean(g2[ar, own][public].abs()),
    }


def run_seed(seed, n=384, signed_arms=64, unsigned_arms=16):
    teacher = train_teacher(seed, steps=850)
    factors = train_factors(seed, teacher, steps=400)
    h, d, z, role, target = donor_pairs(seed + 710300, n)
    with torch.no_grad():
        h1, h2 = teacher.rollout(h)
        d1, d2 = teacher.rollout(d)
        selected, _ = calibrate(seed, teacher, factors)
        writes = factor_writes(teacher, factors, h, d)
        out, delta = patch(teacher, h1, writes, selected)
        donor, base = d2[:, 0], h2[:, 0]
        eligible = ((role != 2) & ((base - target).abs() <= .08)
                    & ((h1[:, 0] - d1[:, 0]).abs() <= .04)
                    & ((donor - base).abs() >= .20))
    count = int(eligible.sum())
    if count == 0:
        raise RuntimeError(f"no authorized eligible pairs for seed {seed}")
    h1_grad = h1.detach().requires_grad_(True)
    g = torch.autograd.grad(teacher(h1_grad)[:, 0].sum(), h1_grad)[0].detach()
    rng = init_rng(seed + 933333)
    signed, unsigned = [], []
    max_norm = max_abs = max_signed = 0.

    with torch.no_grad():
        def recovery(q):
            return float(((q - donor).abs() <= .10)[eligible].float().mean())

        selected_recovery = recovery(out)
        for _ in range(signed_arms):
            delta_sham, ne, de, se = draw_grad_matched(delta, g, rng)
            max_norm = max(max_norm, ne)
            max_abs = max(max_abs, de)
            max_signed = max(max_signed, se)
            signed.append(recovery(teacher(h1 + delta_sham)[:, 0]))
        for _ in range(unsigned_arms):
            delta_sham, ne, de, _ = draw_grad_matched(delta, g, rng, random_sign=True)
            max_norm = max(max_norm, ne)
            max_abs = max(max_abs, de)
            unsigned.append(recovery(teacher(h1 + delta_sham)[:, 0]))

    profile = layer_gradients(teacher, h, role)
    mean_signed = sum(signed) / signed_arms
    return {
        "seed": seed, "pairs": n, "eligible": count,
        "bpd_recovery": selected_recovery,
        "tangent_signed_sham_mean": mean_signed,
        "tangent_signed_sham_best": max(signed),
        "tangent_unsigned_sham_mean": sum(unsigned) / unsigned_arms,
        "bpd_minus_signed_sham": selected_recovery - mean_signed,
        "pseudo_sham_gap": signed[0] - sum(signed[1:]) / (signed_arms - 1),
        "max_norm_error": max_norm,
        "max_abs_jacobian_error": max_abs,
        "max_signed_jacobian_error": max_signed,
        "factor_weight_mse": float((factors.weight().detach() -
                                    teacher.W2.detach()).square().mean()),
        "teacher_readout2_mae": float((base - target).abs().mean()),
        "layer_gradient": profile,
    }


def run(seeds):
    return {
        "scope": "BPD-SF03 independent-root tangent-matched null experiment",
        "torch_version": torch.__version__,
        "roots": list(seeds),
        "rows": [run_seed(s) for s in seeds],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("calibration", "confirmation", "excluded"), required=True)
    p.add_argument("--out")
    a = p.parse_args()
    if a.phase == "calibration":
        seeds = list(range(4101, 4160))
    elif a.phase == "confirmation":
        seeds = list(range(5101, 5133))
    else:
        seeds = [901, 902]
    result = run(seeds)
    print(json.dumps(result, separators=(",", ":")))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
