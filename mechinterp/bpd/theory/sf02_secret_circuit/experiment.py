"""BPD-SF02: trained, weight-tied nonlinear secret routing and BPD factor audits.

This synthetic source is frozen before the fresh eight-root run. Previously
used engineering seeds 901/902 are excluded from scientific tallies.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

D, WIDTH, C = 12, 24, 12
TORCH_THREADS = 1


def init_rng(seed):
    return torch.Generator().manual_seed(int(seed))


class Teacher(nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = init_rng(seed + 99100)
        self.W1 = nn.Parameter(torch.randn(D, WIDTH, generator=g) * .18)
        self.b1 = nn.Parameter(torch.zeros(WIDTH))
        self.W2 = nn.Parameter(torch.randn(WIDTH, D, generator=g) * .14)

    def features(self, h):
        return torch.tanh(h @ self.W1 + self.b1)

    def forward(self, h):
        return self.features(h) @ self.W2

    def rollout(self, h):
        h1 = self(h)
        return h1, self(h1)


def sample(seed, n):
    g = init_rng(seed)
    z = torch.rand(n, 3, generator=g) * 1.6 - .8
    role = torch.randint(0, 3, (n,), generator=g)
    h = torch.zeros(n, D)
    h[:, 1:4] = z
    h[:, 4:7] = F.one_hot(role, 3).float()
    target = torch.where(role == 0, z[:, 0], torch.where(role == 1, z[:, 1], z[:, 2]))
    return h, z, role, target


def train_teacher(seed, steps=850, batch=256):
    teacher = Teacher(seed)
    opt = torch.optim.Adam(teacher.parameters(), lr=.015)
    rng = init_rng(seed + 120100)
    for step in range(steps):
        z = torch.rand(batch, 3, generator=rng) * 1.6 - .8
        role = torch.randint(0, 3, (batch,), generator=rng)
        h = torch.zeros(batch, D)
        h[:, 1:4] = z
        h[:, 4:7] = F.one_hot(role, 3).float()
        target = torch.where(role == 0, z[:, 0], torch.where(role == 1, z[:, 1], z[:, 2]))
        one, two = teacher.rollout(h)
        loss = 2 * (one[:, 0] - z[:, 2]).square().mean() + 3 * (two[:, 0] - target).square().mean()
        loss = loss + .0001 * (teacher.W1.square().mean() + teacher.W2.square().mean())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite teacher loss")
        opt.zero_grad()
        loss.backward()
        opt.step()
    teacher.eval()
    return teacher


class Factors(nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = init_rng(seed + 981000)
        self.V = nn.Parameter(torch.randn(WIDTH, C, generator=g) * .11)
        self.U = nn.Parameter(torch.randn(C, D, generator=g) * .11)

    def weight(self):
        return self.V @ self.U

    def forward(self, features):
        return features @ self.weight()


def train_factors(seed, teacher, steps=400, batch=256):
    f = Factors(seed)
    opt = torch.optim.Adam(f.parameters(), lr=.025)
    rng = init_rng(seed + 540200)
    target = teacher.W2.detach().clone()
    for step in range(steps):
        z = torch.rand(batch, 3, generator=rng) * 1.6 - .8
        role = torch.randint(0, 3, (batch,), generator=rng)
        h = torch.zeros(batch, D)
        h[:, 1:4] = z
        h[:, 4:7] = F.one_hot(role, 3).float()
        with torch.no_grad():
            t1, t2 = teacher.rollout(h)
            feat1 = teacher.features(h)
            feat2 = teacher.features(t1)
        loss = 1.5 * (f.weight() - target).square().mean()
        loss = loss + (f(feat1) - t1).square().mean() + (f(feat2) - t2).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite factor loss")
        opt.zero_grad()
        loss.backward()
        opt.step()
    f.eval()
    return f


def donor_pairs(seed, n):
    h, z, role, target = sample(seed, n)
    d = h.clone()
    g = init_rng(seed + 947)
    idx = torch.where(role == 1, 2, 1)
    flip = torch.where(torch.rand(n, generator=g) < .5, -1., 1.)
    displacement = .3 + torch.rand(n, generator=g) * .6
    d[torch.arange(n), idx] = (d[torch.arange(n), idx] + flip * displacement).clamp(-.8, .8)
    return h, d, z, role, target


def factor_writes(teacher, f, recipient, donor):
    with torch.no_grad():
        feat_diff = teacher.features(donor) - teacher.features(recipient)
        return (feat_diff @ f.V)[:, :, None] * f.U[None, :, :]


def patch(teacher, h1, writes, select):
    delta = writes[:, select, :].sum(1)
    delta = delta.clone()
    delta[:, 0] = 0
    edited = h1 + delta
    if torch.max(torch.abs(edited[:, 0] - h1[:, 0])).item() > 1e-7:
        raise AssertionError("readout preservation failed")
    return teacher(edited)[:, 0], delta


def calibrate(seed, teacher, f, n=240, k=6):
    h, d, z, role, _ = donor_pairs(seed + 350100, n)
    with torch.no_grad():
        h1, h2 = teacher.rollout(h)
        d1, d2 = teacher.rollout(d)
        writes = factor_writes(teacher, f, h, d)
        chosen, losses = [], []
        mask = role != 2
        for _ in range(k):
            best = None
            for c in range(C):
                if c in chosen:
                    continue
                q, _ = patch(teacher, h1, writes, chosen + [c])
                loss = float((q[mask] - d2[mask, 0]).square().mean())
                if best is None or loss < best[0]:
                    best = (loss, c)
            assert best is not None
            losses.append(best[0])
            chosen.append(best[1])
    return chosen, losses


def assay(seed, teacher, f, n=384, k=6, null_arms=255):
    h, d, z, role, target = donor_pairs(seed + 710300, n)
    rng = init_rng(seed + 770000)
    with torch.no_grad():
        h1, h2 = teacher.rollout(h)
        d1, d2 = teacher.rollout(d)
        selected, cal_losses = calibrate(seed, teacher, f, k=k)
        writes = factor_writes(teacher, f, h, d)
        q, delta = patch(teacher, h1, writes, selected)
        fulldark = d1 - h1
        fulldark[:, 0] = 0
        full = teacher(h1 + fulldark)[:, 0]
        all_block, _ = patch(teacher, h1, writes, list(range(C)))
        donor = d2[:, 0]
        base = h2[:, 0]
        eligible = (base - target).abs() <= .08
        eligible = eligible & ((h1[:, 0] - d1[:, 0]).abs() <= .04)
        eligible = eligible & ((donor - base).abs() >= .20)
        good = eligible & (role != 2)
        eligible_public = (role == 2) & ((base - target).abs() <= .08) & ((h1[:, 0] - d1[:, 0]).abs() <= .04)
        dose = delta.norm(dim=-1)
        shams = []
        for _ in range(null_arms):
            u = torch.randn(n, D, generator=rng)
            u[:, 0] = 0
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8) * dose[:, None]
            shams.append(teacher(h1 + u)[:, 0])
        count, pub_n = int(good.sum()), int(eligible_public.sum())

        def recovery(v):
            return float(((v - donor).abs() <= .10)[good].float().mean()) if count else 0.

        sham_rec = [recovery(v) for v in shams]
        shear = torch.eye(C)
        shear[0, 1] = .7
        vw = f.V @ shear
        uw = torch.linalg.solve(shear, f.U)
        gauge = float((vw @ uw - f.weight()).abs().max())
        rotated_writes = ((teacher.features(d) - teacher.features(h)) @ vw)[:, :, None] * uw[None, :, :]
        gauge_singleton = float((rotated_writes[:, 0, :] - writes[:, 0, :]).norm(dim=-1).mean())
        return {
            "seed": seed, "pairs": n, "selected_factors": selected,
            "calibration_losses": cal_losses,
            "teacher_readout1_mae": float((h1[:, 0] - z[:, 2]).abs().mean()),
            "teacher_readout2_mae": float((base - target).abs().mean()),
            "factor_weight_mse": float((f.weight() - teacher.W2).square().mean()),
            "gauge_full_max_error": gauge, "gauge_singleton_effect_l2": gauge_singleton,
            "auth_eligible": count, "public_eligible": pub_n,
            "auth_recovery": recovery(q),
            "sham_recovery_mean": sum(sham_rec) / null_arms,
            "sham_recovery_best": max(sham_rec),
            "full_dark_recovery": recovery(full),
            "auth_all_factor_recovery": recovery(all_block),
            "auth_delta_mean": float((donor - base).abs()[good].mean()) if count else 0.,
            "public_absolute_effect": float((q - base).abs()[eligible_public].mean()) if pub_n else 0.,
            "public_clean_donor_effect": float((donor - base).abs()[eligible_public].mean()) if pub_n else 0.,
            "public_full_dark_effect": float((full - base).abs()[eligible_public].mean()) if pub_n else 0.,
            "public_all_factor_effect": float((all_block - base).abs()[eligible_public].mean()) if pub_n else 0.,
            "public_null_effect_mean": sum(float((v - base).abs()[eligible_public].mean()) for v in shams) / null_arms if pub_n else 0.,
            "max_immediate_readout_change": float(((h1 + delta)[:, 0] - h1[:, 0]).abs().max()),
        }


def run_seed(seed, teacher_steps=850, factor_steps=400, pairs=384):
    torch.set_num_threads(TORCH_THREADS)
    teacher = train_teacher(seed, steps=teacher_steps)
    factors = train_factors(seed, teacher, steps=factor_steps)
    return assay(seed, teacher, factors, n=pairs)


def run(seeds, teacher_steps=850, factor_steps=400, pairs=384):
    return {
        "scope": "synthetic trained recurrent exploratory experiment",
        "config": {"seeds": list(seeds), "teacher_steps": teacher_steps,
                   "factor_steps": factor_steps, "pairs": pairs},
        "torch_version": torch.__version__,
        "rows": [run_seed(seed, teacher_steps, factor_steps, pairs) for seed in seeds],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[901])
    parser.add_argument("--teacher-steps", type=int, default=850)
    parser.add_argument("--factor-steps", type=int, default=400)
    parser.add_argument("--pairs", type=int, default=384)
    parser.add_argument("--out")
    a = parser.parse_args()
    r = run(a.seeds, a.teacher_steps, a.factor_steps, a.pairs)
    print(json.dumps(r, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2) + "\n")
