"""SF04: trained interleaved-source bindings in a 3-pass shared nonlinear circuit.

Canonical source after two engineering seed pilots (901/902). Data streams
for heldout source-tag flips are disjoint from teacher/factor/selection streams.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

H, WIDTH, IN, COMPONENTS = 20, 40, 9, 20


def rng(seed):
    return torch.Generator().manual_seed(int(seed))


def sample(seed, n):
    g = rng(seed)
    source1 = torch.randint(0, 2, (n,), generator=g)
    source2 = torch.randint(0, 2, (n,), generator=g)
    value1 = torch.rand(n, generator=g) * 1.6 - .8
    value2 = torch.rand(n, generator=g) * 1.6 - .8
    role = torch.randint(0, 3, (n,), generator=g)
    public = torch.rand(n, generator=g) * 1.6 - .8
    first, second, query = (torch.zeros(n, IN) for _ in range(3))
    idx = torch.arange(n)
    first[idx, source1] = 1
    first[:, 2], first[:, 3] = value1, 1
    second[idx, source2] = 1
    second[:, 2], second[:, 3] = value2, 1
    query[idx, 4 + role] = 1
    query[:, 7], query[:, 8] = public, 1
    target = torch.where(role == 2, public,
                         torch.where(role == source2, value2,
                                     torch.where(role == source1, value1,
                                                 torch.zeros_like(public))))
    metadata = {"source1": source1, "source2": source2, "value1": value1,
                "value2": value2, "role": role, "public": public}
    return first, second, query, target, metadata


class Teacher(nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = rng(seed + 112)
        self.A = nn.Parameter(torch.randn(H, WIDTH, generator=g) * .12)
        self.B = nn.Parameter(torch.randn(IN, WIDTH, generator=g) * .12)
        self.bias = nn.Parameter(torch.zeros(WIDTH))
        self.C = nn.Parameter(torch.randn(WIDTH, H, generator=g) * .12)

    def features(self, state, event):
        return torch.tanh(state @ self.A + event @ self.B + self.bias)

    def step(self, state, event):
        return state + self.features(state, event) @ self.C

    def rollout(self, first, second, query):
        zero = torch.zeros(first.shape[0], H)
        one = self.step(zero, first)
        two = self.step(one, second)
        three = self.step(two, query)
        return one, two, three


def train_teacher(seed, steps=1200, batch=256):
    model = Teacher(seed)
    opt = torch.optim.Adam(model.parameters(), lr=.012)
    for update in range(steps):
        first, second, query, label, _ = sample(seed * 100000 + update, batch)
        h1, h2, h3 = model.rollout(first, second, query)
        loss = 3 * (h3[:, 0] - label).square().mean()
        loss += .4 * (h1[:, 0].square().mean() + h2[:, 0].square().mean())
        loss += .0001 * (model.A.square().mean() + model.C.square().mean())
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite teacher objective")
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


class Factors(nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = rng(seed + 82100)
        self.V = nn.Parameter(torch.randn(WIDTH, COMPONENTS, generator=g) * .11)
        self.U = nn.Parameter(torch.randn(COMPONENTS, H, generator=g) * .11)

    def weight(self):
        return self.V @ self.U


def train_factors(seed, teacher, steps=400, batch=256):
    factors = Factors(seed)
    opt = torch.optim.Adam(factors.parameters(), lr=.03)
    for update in range(steps):
        first, second, query, _, _ = sample(seed * 170000 + update + 300000, batch)
        with torch.no_grad():
            h1, h2, h3 = teacher.rollout(first, second, query)
            h0 = torch.zeros_like(h1)
            phi = torch.cat((teacher.features(h0, first),
                             teacher.features(h1, second),
                             teacher.features(h2, query)), dim=0)
            target = torch.cat((h1, h2 - h1, h3 - h2), dim=0)
        loss = 1.5 * (factors.weight() - teacher.C.detach()).square().mean()
        loss += (phi @ factors.weight() - target).square().mean()
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite factor objective")
        opt.zero_grad()
        loss.backward()
        opt.step()
    factors.eval()
    return factors


def paired_source_swap(seed, n):
    first, second, query, target, meta = sample(seed, n)
    donor = second.clone()
    donor[:, :2] = 1.0 - donor[:, :2]  # same values, new source identity
    role = meta["role"]
    source_new = 1 - meta["source2"]
    donor_target = torch.where(role == 2, meta["public"],
                               torch.where(role == source_new, meta["value2"],
                                           torch.where(role == meta["source1"],
                                                       meta["value1"],
                                                       torch.zeros_like(meta["public"]))))
    return (first, second, query, target, meta), donor, donor_target


def component_writes(teacher, factors, h1, second, donor):
    with torch.no_grad():
        diff = teacher.features(h1, donor) - teacher.features(h1, second)
        return (diff @ factors.V)[:, :, None] * factors.U[None, :, :]


def intervene(teacher, state2, query, writes, selected):
    delta = writes[:, selected, :].sum(dim=1).clone()
    delta[:, 0] = 0
    if not torch.equal((state2 + delta)[:, 0], state2[:, 0]):
        raise AssertionError("current readout mismatch after projected donor write")
    return teacher.step(state2 + delta, query)[:, 0], delta


def eligible(teacher, recipient, donor_second, donor_target):
    first, second, query, target, metadata = recipient
    with torch.no_grad():
        h1, h2, h3 = teacher.rollout(first, second, query)
        d2 = teacher.step(h1, donor_second)
        d3 = teacher.step(d2, query)
        good = (metadata["role"] != 2)
        good &= (h3[:, 0] - target).abs() <= .08
        good &= (d3[:, 0] - donor_target).abs() <= .08
        good &= (h2[:, 0] - d2[:, 0]).abs() <= .04
        good &= (h3[:, 0] - d3[:, 0]).abs() > .20
    return good, (h1, h2, h3, d2, d3)


def select_components(seed, teacher, factors, pairs=240, count=14):
    recipient, donor_second, donor_target = paired_source_swap(seed + 400000, pairs)
    valid, (h1, h2, h3, d2, d3) = eligible(teacher, recipient, donor_second, donor_target)
    if int(valid.sum()) < 20:
        raise ValueError("calibration has too few eligible source-label counterfactuals")
    writes = component_writes(teacher, factors, h1, recipient[1], donor_second)
    chosen, losses = [], []
    with torch.no_grad():
        for _ in range(count):
            best = None
            for candidate in range(COMPONENTS):
                if candidate in chosen:
                    continue
                q, _ = intervene(teacher, h2, recipient[2], writes, chosen + [candidate])
                score = float((q[valid] - d3[:, 0][valid]).square().mean())
                if best is None or score < best[0]:
                    best = score, candidate
            chosen.append(best[1])
            losses.append(best[0])
    return chosen, losses


def tangent_matched(delta, gradient, g):
    dark_gradient = gradient.clone()
    dark_gradient[:, 0] = 0
    u = dark_gradient / dark_gradient.norm(dim=1, keepdim=True).clamp_min(1e-8)
    alpha = (delta * u).sum(dim=1, keepdim=True)
    remaining = (delta.norm(dim=1, keepdim=True).square() - alpha.square()).clamp_min(0).sqrt()
    noise = torch.randn(delta.shape, generator=g)
    noise[:, 0] = 0
    noise = noise - (noise * u).sum(dim=1, keepdim=True) * u
    noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sham = alpha * u + remaining * noise
    sham[:, 0] = 0
    norm_error = float((sham.norm(dim=1) - delta.norm(dim=1)).abs().max())
    tangent_error = float(((dark_gradient * sham).sum(dim=1) -
                           (dark_gradient * delta).sum(dim=1)).abs().max())
    return sham, norm_error, tangent_error


def assay(seed, teacher, factors, n=512, selected_count=14, random_arms=64, signed_arms=64):
    recipient, donor_second, donor_target = paired_source_swap(seed + 710300, n)
    first, second, query, target, metadata = recipient
    valid, (h1, h2, h3, d2, d3) = eligible(teacher, recipient, donor_second, donor_target)
    selected, selection_losses = select_components(seed, teacher, factors, count=selected_count)
    writes = component_writes(teacher, factors, h1, second, donor_second)
    count = int(valid.sum())
    if count < 20:
        raise ValueError("heldout has too few eligible authorized recipients")

    with torch.no_grad():
        partial, delta = intervene(teacher, h2, query, writes, selected)
        full, _ = intervene(teacher, h2, query, writes, list(range(COMPONENTS)))
        exact_delta = (d2 - h2).clone()
        exact_delta[:, 0] = 0
        exact = teacher.step(h2 + exact_delta, query)[:, 0]
        public = metadata["role"] == 2
        if int(public.sum()) == 0:
            raise ValueError("public query arm absent")
        def recovery(out):
            return float(((out - d3[:, 0]).abs() <= .10)[valid].float().mean())
        score = recovery(partial)
        rng_null = rng(seed + 800900)
        dose = delta.norm(dim=1, keepdim=True)
        random_recovery = []
        for _ in range(random_arms):
            direction = torch.randn(delta.shape, generator=rng_null)
            direction[:, 0] = 0
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8) * dose
            random_recovery.append(recovery(teacher.step(h2 + direction, query)[:, 0]))

    state = h2.detach().requires_grad_(True)
    gradient = torch.autograd.grad(teacher.step(state, query)[:, 0].sum(), state)[0].detach()
    with torch.no_grad():
        signed_recovery, max_norm, max_tangent = [], 0., 0.
        for _ in range(signed_arms):
            sham, norm_e, tangent_e = tangent_matched(delta, gradient, rng_null)
            max_norm = max(max_norm, norm_e)
            max_tangent = max(max_tangent, tangent_e)
            signed_recovery.append(recovery(teacher.step(h2 + sham, query)[:, 0]))
        shear = torch.eye(COMPONENTS)
        shear[0, 1] = .7
        physical = factors.weight()
        preserved = (factors.V @ shear) @ torch.linalg.solve(shear, factors.U)
        blind = torch.where(metadata["role"] == 2, metadata["public"],
                            .5 * metadata["value2"] + .25 * metadata["value1"])
        return {
            "seed": seed,
            "generated": n, "authorized_eligible": count,
            "public_cases": int(public.sum()),
            "selected_indices": selected, "selection_losses": selection_losses,
            "teacher_mae": float((h3[:, 0] - target).abs().mean()),
            "source_blind_oracle_mae": float((blind - target).abs().mean()),
            "early_readout_abs": float(h2[:, 0].abs().mean()),
            "physical_factor_mse": float((physical - teacher.C).square().mean()),
            "partial_recovery": score, "full_factor_recovery": recovery(full),
            "full_dark_recovery": recovery(exact),
            "random_dark_sham_mean": sum(random_recovery) / random_arms,
            "random_dark_sham_best": max(random_recovery),
            "signed_tangent_sham_mean": sum(signed_recovery) / signed_arms,
            "partial_minus_signed_sham": score - sum(signed_recovery) / signed_arms,
            "public_partial_abs_change": float((partial[public] - h3[public, 0]).abs().mean()),
            "public_clean_swap_abs_change": float((d3[public, 0] - h3[public, 0]).abs().mean()),
            "public_full_factor_abs_change": float((full[public] - h3[public, 0]).abs().mean()),
            "gauge_full_max_error": float((preserved - physical).abs().max()),
            "immediate_readout_error": float(((h2 + delta)[:, 0] - h2[:, 0]).abs().max()),
            "max_sham_norm_error": max_norm,
            "max_sham_tangent_error": max_tangent,
        }


def run_seed(seed, teacher_steps=1200, factor_steps=400, pairs=512):
    torch.set_num_threads(1)
    teacher = train_teacher(seed, teacher_steps)
    factors = train_factors(seed, teacher, factor_steps)
    return assay(seed, teacher, factors, n=pairs)


def run(seeds):
    return {
        "scope": "BPD-SF04 prospective exploratory interleaved-source synthetic circuit",
        "torch_version": torch.__version__,
        "fresh_roots": list(seeds),
        "rows": [run_seed(seed) for seed in seeds],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("excluded", "fresh"), required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    seeds = [901, 902] if a.phase == "excluded" else [1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367]
    report = run(seeds)
    print(json.dumps(report, separators=(",", ":")))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
