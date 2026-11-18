"""CQ-08 natural-state BPD mediation with nuisance-only source controls.

Synthetic block-structured rank-eight nonlinear recurrence. Both recipient and
counterfactual first states are produced by legitimate cues and the frozen
physical transition. Masking occurs on the NEXT shared BPD core invocation.
Per-case matched nuisance sources come from *input-space* draws, never an
additive projection of a hidden state. Historical sealed sets are never read.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

CORE, NUISANCE, D, INPUT, COMPONENTS = 12, 24, 36, 6, 8
CORE_INPUTS = tuple(range(CORE)) + (D, D + 1, D + 4, D + 5)
NUISANCE_INPUTS = tuple(range(CORE, D)) + (D + 2, D + 3)
DEVELOPMENT = (1301, 1303, 1307, 1319, 1321, 1327, 1361, 1373)
SHAM = tuple(9001 + 2 * i for i in range(59))
CONFIRMATION = tuple(11001 + 2 * i for i in range(32))
TEACHER_STEPS, FACTOR_STEPS = 1050, 650
DONOR_CASES, RESERVOIR = 128, 512
HEAD_TOL, DOSE_TOL = 0.5, 0.25
ROOT = Path(__file__).resolve().parent


def configure():
    torch.set_num_threads(1)


def source(seed: int, n: int, offset: int, paired_z=False):
    if paired_z and n % 2:
        raise ValueError("paired_z requires an even n")
    gen = torch.Generator().manual_seed(seed + offset)
    k = n // 2 if paired_z else n
    signs = torch.where(torch.randint(0, 2, (k, 3), generator=gen) == 0, -1.0, 1.0)
    mag = 0.65 + 0.55 * torch.rand(k, 3, generator=gen)
    values = signs * mag
    nuisance = (torch.rand(k, 2, generator=gen) * 2 - 1) * 2
    first = torch.cat((values[:, :2], nuisance, torch.zeros(k, 1), -torch.ones(k, 1)), 1)
    second = torch.cat((torch.zeros(k, 4), values[:, 2:3], torch.ones(k, 1)), 1)
    labels = torch.stack((signs[:, 0] * signs[:, 1], signs[:, 0] * signs[:, 2]), 1)
    if paired_z:
        first = first.repeat_interleave(2, 0)
        nuisance = nuisance.repeat_interleave(2, 0)
        second_flip = second.clone(); second_flip[:, 4] *= -1
        second = torch.stack((second, second_flip), 1).reshape(n, INPUT)
        labels_flip = labels.clone(); labels_flip[:, 1] *= -1
        labels = torch.stack((labels, labels_flip), 1).reshape(n, 2)
    return first, second, labels, nuisance


class PhysicalTeacher(nn.Module):
    """Learned rank-6 core and rank-2 nuisance carrier, tied for both visits."""
    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 1000)
        self.A = nn.Parameter(torch.randn(CORE + 4, 6, generator=g) * .21)
        self.B = nn.Parameter(torch.randn(6, CORE, generator=g) * .21)
        self.A_n = nn.Parameter(torch.randn(NUISANCE + 2, 2, generator=g) * .25)
        self.B_n = nn.Parameter(torch.randn(2, NUISANCE, generator=g) * .4)
        self.bias = nn.Parameter(torch.zeros(D))
        self.head = nn.Linear(CORE, 1, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(1, CORE, generator=g) * .2)
        g_n = torch.Generator().manual_seed(seed + 8881)
        self.register_buffer("nuisance_target_map", torch.randn(2, NUISANCE, generator=g_n) * .8)

    def weight(self):
        w = torch.zeros(D + INPUT, D, dtype=self.A.dtype, device=self.A.device)
        w = w.index_put((torch.tensor(CORE_INPUTS)[:, None], torch.arange(CORE)[None, :]), self.A @ self.B)
        w = w.index_put((torch.tensor(NUISANCE_INPUTS)[:, None], torch.arange(CORE, D)[None, :]), self.A_n @ self.B_n)
        return w

    def step(self, state, cue):
        x = torch.cat((state, cue), 1)
        core = (x[:, CORE_INPUTS] @ self.A) @ self.B + self.bias[:CORE]
        nuisance = (x[:, NUISANCE_INPUTS] @ self.A_n) @ self.B_n + self.bias[CORE:]
        return torch.tanh(torch.cat((core, nuisance), 1))

    def logit(self, states):
        return self.head(states[:, :CORE]).flatten()

    def forward(self, first_cue, second_cue):
        s1 = self.step(torch.zeros(len(first_cue), D), first_cue)
        s2 = self.step(s1, second_cue)
        return torch.stack((self.logit(s1), self.logit(s2)), 1), s1, s2


def train_teacher(seed: int, steps: int = TEACHER_STEPS):
    model = PhysicalTeacher(seed)
    opt = torch.optim.Adam(model.parameters(), lr=.018)
    for i in range(steps):
        u0, u1, labels, nuis = source(seed, 128, 20000 + i)
        logits, first, _ = model(u0, u1)
        flipped = u0.clone(); flipped[:, :2] *= -1
        logits_flip, _, _ = model(flipped, u1)
        labels_flip = labels.clone(); labels_flip[:, 1] *= -1
        loss = F.softplus(-logits * labels).mean()
        loss += .5 * F.mse_loss(first[:, CORE:], torch.tanh(nuis @ model.nuisance_target_map))
        loss += .5 * F.mse_loss(logits[:, 0], logits_flip[:, 0])
        loss += .25 * F.softplus(-logits_flip * labels_flip).mean()
        loss += .0004 * sum(p.square().mean() for p in (model.A, model.B, model.A_n, model.B_n))
        if not torch.isfinite(loss): raise RuntimeError("nonfinite teacher loss")
        opt.zero_grad(); loss.backward(); opt.step()
    return model


class Factors(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 87000)
        self.V = nn.Parameter(torch.randn(D + INPUT, COMPONENTS, generator=g) * .23)
        self.U = nn.Parameter(torch.randn(COMPONENTS, D, generator=g) * .23)

    def weight(self):
        return self.V @ self.U


def factor_step(t, f, first_state, cue, mask=None):
    if mask is None:
        mask = torch.ones(len(first_state), COMPONENTS)
    if mask.shape != (len(first_state), COMPONENTS):
        raise ValueError("mask must be [B,8]")
    acts = torch.cat((first_state, cue), 1) @ f.V
    return torch.tanh((acts * mask) @ f.U + t.bias)


def train_factors(seed: int, t: PhysicalTeacher, steps: int = FACTOR_STEPS):
    f = Factors(seed)
    opt = torch.optim.Adam((f.V, f.U), lr=.05)
    target = t.weight().detach()
    for i in range(steps):
        u0, u1, _, _ = source(seed, 96, 240000 + i)
        with torch.no_grad():
            s1 = t.step(torch.zeros(len(u0), D), u0)
            s2 = t.step(s1, u1)
        p1 = factor_step(t, f, torch.zeros(len(u0), D), u0)
        p2 = factor_step(t, f, p1, u1)
        loss = (F.mse_loss(p1[:, :CORE], s1[:, :CORE])
                + F.mse_loss(p1[:, CORE:], s1[:, CORE:])
                + F.mse_loss(p2[:, :CORE], s2[:, :CORE])
                + F.mse_loss(p2[:, CORE:], s2[:, CORE:])
                + F.mse_loss(f.weight(), target))
        if not torch.isfinite(loss): raise RuntimeError("nonfinite factor loss")
        opt.zero_grad(); loss.backward(); opt.step()
    return f


def canonicalize(f: Factors):
    with torch.no_grad():
        original = f.weight().detach().double()
        q, s, vh = torch.linalg.svd(original, full_matrices=False)
        root = s[:COMPONENTS].clamp_min(0).sqrt()
        v = q[:, :COMPONENTS] * root[None, :]
        u = root[:, None] * vh[:COMPONENTS, :]
        error = float((original - v @ u).abs().max())
        f.V.copy_(v.float()); f.U.copy_(u.float())
    return {"svd_full_reconstruction_error": error,
            "min_singular_gap": min(float(s[i] - s[i+1]) for i in range(COMPONENTS - 1)),
            "singular_values": [float(v) for v in s[:COMPONENTS]]}


def calibration_block(seed, t, f, n=256, k=2):
    u0, u1, labels, _ = source(seed, n, 610000)
    with torch.no_grad():
        s1 = t.step(torch.zeros(n, D), u0)
        losses = []
        for c in range(COMPONENTS):
            m = torch.ones(n, COMPONENTS); m[:, c] = 0
            output = t.logit(factor_step(t, f, s1, u1, m))
            losses.append(float(F.softplus(-labels[:, 1] * output).mean()))
    selected = sorted(range(COMPONENTS), key=lambda j: (-losses[j], j))[:k]
    return selected, losses


def donor_sources(seed, n=DONOR_CASES):
    gen = torch.Generator().manual_seed(seed + 811000)
    signs = torch.where(torch.randint(0, 2, (n, 3), generator=gen) == 0, -1., 1.)
    mag = .65 + .55 * torch.rand(n, 3, generator=gen)
    x, y, z = (signs * mag).unbind(1)
    nuis = (torch.rand(n, 2, generator=gen) * 2 - 1) * 2
    rec = torch.stack((x, y, nuis[:, 0], nuis[:, 1], torch.zeros(n), -torch.ones(n)), 1)
    don = torch.stack((-x, -y, nuis[:, 0], nuis[:, 1], torch.zeros(n), -torch.ones(n)), 1)
    u1 = torch.stack((torch.zeros(n), torch.zeros(n), torch.zeros(n), torch.zeros(n), z, torch.ones(n)), 1)
    return rec, don, u1, x, y, z, nuis


def matched_nuisance(seed, t, n=DONOR_CASES):
    ur, ud, u1, x, y, z, nuis = donor_sources(seed, n)
    with torch.no_grad():
        sr = t.step(torch.zeros(n, D), ur)
        sd = t.step(torch.zeros(n, D), ud)
        dose = (sd - sr).norm(dim=1)
        head_r = t.logit(sr)
        head_d = t.logit(sd)
        g = torch.Generator().manual_seed(seed + 812000)
        alternatives = (torch.rand(n, RESERVOIR, 2, generator=g) * 2 - 1) * 2
        xalt = x[:, None].expand(n, RESERVOIR)
        yalt = y[:, None].expand(n, RESERVOIR)
        altcue = torch.stack((xalt, yalt, alternatives[:, :, 0], alternatives[:, :, 1],
                              torch.zeros(n, RESERVOIR), -torch.ones(n, RESERVOIR)), -1).reshape(-1, INPUT)
        candidates = t.step(torch.zeros(len(altcue), D), altcue).reshape(n, RESERVOIR, D)
        candidate_dist = (candidates - sr[:, None, :]).norm(dim=2)
        candidate_head = t.head(candidates[:, :, :CORE]).squeeze(-1)
        hdelta = (candidate_head - head_r[:, None]).abs()
        ddelta = (candidate_dist - dose[:, None]).abs()
        feasible = (ddelta <= DOSE_TOL) & (hdelta <= HEAD_TOL) & (candidate_dist >= .1)
        score = torch.where(feasible, ddelta / DOSE_TOL + hdelta / HEAD_TOL, 1e6)
        index = score.argmin(1)
        idx = torch.arange(n)
        head_ok = (head_d - head_r).abs() <= HEAD_TOL
        first_ok = (head_r * torch.sign(x * y) > 0) & (head_d * torch.sign(x * y) > 0)
        match = feasible.any(1) & head_ok & first_ok
        selected_cue = altcue.reshape(n, RESERVOIR, INPUT)[idx, index]
        selected_s = candidates[idx, index]
        replay = t.step(torch.zeros(n, D), selected_cue)
        return {
            "match": match, "source_candidate_found": feasible.any(1),
            "first_head_ok": head_ok, "first_correct": first_ok,
            "rec_state": sr, "donor_state": sd, "nuisance_state": selected_s,
            "rec_cue": ur, "donor_cue": ud, "nuisance_cue": selected_cue, "second_cue": u1,
            "x": x, "y": y, "z": z, "nuisance": nuis,
            "head_rec": head_r, "head_donor": head_d, "head_nuisance": candidate_head[idx, index],
            "dose_donor": dose, "dose_nuisance": candidate_dist[idx, index],
            "max_source_replay_error": float((replay - selected_s).abs().max()),
            "max_state_box_violation": max(0., float(torch.cat((sr, sd, selected_s)).abs().max()) - 1.),
        }


def random_masks(seed, n, offset, k=2):
    g = torch.Generator().manual_seed(seed + offset)
    out = torch.ones(n, COMPONENTS)
    for i in range(n):
        out[i, torch.randperm(COMPONENTS, generator=g)[:k]] = 0.
    return out


class NativeModel(nn.Module):
    """True BPDModel/TiedComponentLinear mask schedule at pass two."""
    def __init__(self, teacher, factors):
        super().__init__()
        from mechinterp.bpd.components import BPDModel
        from mechinterp.bpd.config import AblationConfig
        class Base(nn.Module):
            def __init__(self, w, bias):
                super().__init__()
                self.core = nn.Linear(D + INPUT, D, bias=False)
                with torch.no_grad(): self.core.weight.copy_(w.T)
                self.register_buffer("fixed_bias", bias.detach().clone())
            def forward(self, joined):
                return torch.tanh(self.core(joined) + self.fixed_bias)
        self.bpd = BPDModel(Base(teacher.weight().detach(), teacher.bias),
                            {"core": COMPONENTS}, max_passes=2,
                            ablation=AblationConfig(mode="zero", sphere_rescale=False), init_svd=False)
        with torch.no_grad():
            wrapper = self.bpd.wrappers["core"]
            wrapper.V.copy_(factors.V); wrapper.U.copy_(factors.U)

    def step(self, first_state, second_cue, mask):
        from mechinterp.bpd.components import PassMaskSchedule
        n = len(first_state)
        schedule = PassMaskSchedule(mask.reshape(n, 1, 1, COMPONENTS), torch.zeros(n, 1, 1))
        with self.bpd.masked_components({"core": schedule}, pass_offset=1):
            return self.bpd.target_model(torch.cat((first_state, second_cue), 1).unsqueeze(1))[:, 0]


def continue_step(t, f, state, cue, mask, backend="independent", native=None):
    if backend == "native": return native.step(state, cue, mask)
    return factor_step(t, f, state, cue, mask)


def score_donors(seed, t, f, selected, backend="independent", native=None, include_pairs=True):
    group = matched_nuisance(seed, t)
    n = len(group["x"])
    masks = {"full": torch.ones(n, COMPONENTS), "selected_off": torch.ones(n, COMPONENTS),
             "random_off_1": random_masks(seed, n, 850000),
             "random_off_2": random_masks(seed, n, 850001)}
    masks["selected_off"][:, selected] = 0.
    with torch.no_grad():
        logits = {key: t.logit(continue_step(t, f, group["donor_state"], group["second_cue"], m, backend, native))
                  for key, m in masks.items()}
        label_donor = -torch.sign(group["x"] * group["z"])
        label_rec = -label_donor
        donor_correct = {key: logits[key] * label_donor > 0 for key in masks}
        full = donor_correct["full"]
        admissible = group["match"] & full
        damage = {key: (admissible & ~good) if key != "full" else torch.zeros(n, dtype=torch.bool)
                  for key, good in donor_correct.items()}
        nuisance_logit = t.logit(continue_step(t, f, group["nuisance_state"], group["second_cue"], masks["full"], backend, native))
        rec_logit = t.logit(continue_step(t, f, group["rec_state"], group["second_cue"], masks["full"], backend, native))
        preactivations = torch.cat((group["donor_state"], group["second_cue"]), 1) @ f.V
        ablate_norm = {key: (preactivations * (1 - m)) @ f.U for key, m in masks.items() if key != "full"}
        norm_summaries = {key: {"mean": float(v.norm(dim=1).mean()),
                                "median": float(v.norm(dim=1).median()),
                                "max": float(v.norm(dim=1).max())}
                          for key, v in ablate_norm.items()}
        out = {
            "seed": seed, "pairs": n, "eligible_match": int(group["match"].sum()),
            "source_match_available": int(group["source_candidate_found"].sum()),
            "first_head_ok": int(group["first_head_ok"].sum()),
            "first_clean_correct": int(group["first_correct"].sum()),
            "teacher_source_full_correct": int(full.sum()),
            "full_donor_correct": int(full.sum()),
            "selected_off_correct": int(donor_correct["selected_off"].sum()),
            "random_off_1_correct": int(donor_correct["random_off_1"].sum()),
            "random_off_2_correct": int(donor_correct["random_off_2"].sum()),
            "damage_counts_itt": {key: int(value.sum()) for key, value in damage.items()},
            "matched_nuisance_recipient_label_correct": int((group["match"] & (nuisance_logit * label_rec > 0)).sum()),
            "matched_nuisance_donor_label_correct": int((group["match"] & (nuisance_logit * label_donor > 0)).sum()),
            "recipient_sham_donor_label_correct": int((group["match"] & (rec_logit * label_donor > 0)).sum()),
            "mean_donor_state_dose": float(group["dose_donor"].mean()),
            "mean_matched_nuisance_state_dose": float(group["dose_nuisance"][group["match"]].mean()) if group["match"].any() else None,
            "max_matched_dose_error": float((group["dose_nuisance"]-group["dose_donor"]).abs()[group["match"]].max()) if group["match"].any() else None,
            "max_donor_head_difference": float((group["head_donor"]-group["head_rec"]).abs().max()),
            "max_matched_head_difference": float((group["head_nuisance"]-group["head_rec"]).abs()[group["match"]].max()) if group["match"].any() else None,
            "max_source_replay_error": group["max_source_replay_error"],
            "max_state_box_violation": group["max_state_box_violation"],
            "ablation_write_norms": norm_summaries,
            "selected_block": list(selected),
        }
        if include_pairs:
            out["pair_data"] = {
                "match": group["match"].int().tolist(), "donor_full_correct": full.int().tolist(),
                "damage_selected": damage["selected_off"].int().tolist(),
                "damage_random1": damage["random_off_1"].int().tolist(),
                "damage_random2": damage["random_off_2"].int().tolist(),
                "nuisance_own_correct": (nuisance_logit * label_rec > 0).int().tolist(),
                "donor_state_dose": group["dose_donor"].tolist(),
                "matched_state_dose": group["dose_nuisance"].tolist(),
            }
    return out


def paired_z_task(seed, t, f, backend="independent", native=None, n=256):
    u0, u1, labels, _ = source(seed, n, 810000, paired_z=True)
    with torch.no_grad():
        teacher_logits, s1, _ = t(u0, u1)
        full = torch.ones(n, COMPONENTS)
        second_logit = t.logit(continue_step(t, f, s1, u1, full, backend, native))
        first_correct = t.logit(s1) * labels[:, 0] > 0
        pair_flip = bool(torch.equal(u0[::2], u0[1::2]) and (labels[::2, 1] == -labels[1::2, 1]).all())
        # First-pass-only prediction repeats over each z-flipped pair.
        first_cue_pred = t.logit(s1)
        return {"n": n, "first_cue_identical_z_pairs": pair_flip,
                "teacher_both_correct": int(((teacher_logits * labels) > 0).all(1).sum()),
                "bpd_full_both_correct": int((first_correct & (second_logit * labels[:, 1] > 0)).sum()),
                "myopic_second_correct": int((first_cue_pred * labels[:, 1] > 0).sum()),
                "symbolic_exact_ceiling": n}


def gauge_audit(seed, t, f, selected, n=64):
    u0, u1, _, _ = source(seed, n, 820000)
    with torch.no_grad():
        s1 = t.step(torch.zeros(n, D), u0)
        mask = torch.ones(n, COMPONENTS); mask[:, selected] = 0.
        baseline = t.logit(factor_step(t, f, s1, u1, mask))
        v, u = f.V.detach().double(), f.U.detach().double()
        w = v @ u
        r1 = torch.eye(COMPONENTS, dtype=torch.float64); r1[0, 1] = .6
        r2 = torch.eye(COMPONENTS, dtype=torch.float64); r2[0, 2] = .45; r2[2, 0] = -.25
        max_full = 0.; max_canon = 0.; max_raw = 0.; min_gap = float("inf")
        for R in (r1, r2):
            vr = v @ R; ur = torch.linalg.inv(R) @ u
            max_full = max(max_full, float((vr @ ur - w).abs().max()))
            joined = torch.cat((s1.double(), u1.double()), 1)
            raw = torch.tanh(((joined @ vr) * mask.double()) @ ur + t.bias.double())
            max_raw = max(max_raw, float((t.logit(raw.float()) - baseline).abs().max()))
            q, sv, vh = torch.linalg.svd(vr @ ur, full_matrices=False)
            sq = sv[:COMPONENTS].sqrt()
            vc = q[:, :COMPONENTS] * sq[None, :]; uc = sq[:, None] * vh[:COMPONENTS]
            canonical = torch.tanh(((joined @ vc) * mask.double()) @ uc + t.bias.double())
            max_canon = max(max_canon, float((t.logit(canonical.float()) - baseline).abs().max()))
            min_gap = min(min_gap, min(float(sv[i]-sv[i+1]) for i in range(COMPONENTS-1)))
        return {"max_full_operator_shear_error": max_full,
                "max_raw_masked_logit_drift": max_raw,
                "max_recanonical_masked_logit_drift": max_canon,
                "min_singular_gap": min_gap}


def native_parity(seed, t, f, native):
    u0, u1, _, _ = source(seed, 48, 830000)
    with torch.no_grad():
        s1 = t.step(torch.zeros(len(u0), D), u0)
        masks = [torch.ones(len(u0), COMPONENTS), random_masks(seed, len(u0), 850000), random_masks(seed, len(u0), 850001)]
        err = 0.
        for m in masks:
            a = factor_step(t, f, s1, u1, m)
            b = native.step(s1, u1, m)
            err = max(err, float((a-b).abs().max()))
    return err


def run_root(seed, backend="independent", include_pairs=True):
    configure(); torch.manual_seed(seed)
    t = train_teacher(seed)
    f = train_factors(seed, t)
    fit_weight_mse = float(F.mse_loss(f.weight().detach(), t.weight().detach()))
    geometry = canonicalize(f)
    selected, losses = calibration_block(seed, t, f)
    native = NativeModel(t, f) if backend == "native" else None
    parity = native_parity(seed, t, f, native) if native is not None else None
    with torch.no_grad():
        donors = score_donors(seed, t, f, selected, backend, native, include_pairs)
        task = paired_z_task(seed, t, f, backend, native)
        gauge = gauge_audit(seed, t, f, selected)
    return {"seed": seed, "teacher_steps": TEACHER_STEPS, "factor_steps": FACTOR_STEPS,
            "teacher_rank": COMPONENTS, "model_dim": D,
            "fit_weight_mse": fit_weight_mse, "geometry": geometry,
            "calibration_selected": selected, "calibration_singleton_losses": losses,
            "native_parity_max": parity, "task": task, "donor": donors, "gauge": gauge}


def cohort(phase="development", backend="independent", seeds=None, include_pairs=True):
    if seeds is None:
        seeds = {"development": DEVELOPMENT, "sham": SHAM, "confirmation": CONFIRMATION}[phase]
    if phase == "confirmation" and not (ROOT / "CONFIRM_UNLOCK.json").exists():
        raise RuntimeError("CQ08 confirmation is sealed; source-locked sham receipt and preflight unlock required")
    roots = [run_root(int(s), backend, include_pairs) for s in seeds]
    return {"schema":"bpd.cq08.nuisance_mediation.v1", "phase":phase, "backend":backend,
            "seeds":[int(s) for s in seeds], "rows":roots, "n_donors_per_root":DONOR_CASES,
            "teacher_steps":TEACHER_STEPS, "factor_steps":FACTOR_STEPS,
            "nuisance_candidates_per_pair":RESERVOIR,
            "torch":torch.__version__, "python":platform.python_version(),
            "evidence_class":"synthetic structured nuisance-carrier recurrent family; source-reachable inputs"}


def summarize(result):
    rows = result["rows"]
    count = lambda name: sum(row["donor"][name] for row in rows)
    total = DONOR_CASES * len(rows)
    effects1=[(row["donor"]["damage_counts_itt"]["selected_off"]-row["donor"]["damage_counts_itt"]["random_off_1"])/DONOR_CASES for row in rows]
    effects2=[(row["donor"]["damage_counts_itt"]["selected_off"]-row["donor"]["damage_counts_itt"]["random_off_2"])/DONOR_CASES for row in rows]
    shams=[abs((row["donor"]["damage_counts_itt"]["random_off_1"]-row["donor"]["damage_counts_itt"]["random_off_2"])/DONOR_CASES) for row in rows]
    return {"roots":len(rows), "n_donor_pairs":total,
            "jointly_matched":count("eligible_match"),
            "full_donor_correct":count("full_donor_correct"),
            "damage_itt":{k:sum(row["donor"]["damage_counts_itt"][k] for row in rows) for k in rows[0]["donor"]["damage_counts_itt"]},
            "mean_primary_effect":sum(effects1)/len(effects1),
            "primary_effects_by_root":effects1,
            "second_effects_by_root":effects2,
            "random_random_abs_shams_by_root":shams,
            "largest_random_random_sham":max(shams),
            "roots_feasibility_at_least_80pct":sum(row["donor"]["eligible_match"]>=.8*DONOR_CASES for row in rows),
            "max_reachability_replay_error":max(row["donor"]["max_source_replay_error"] for row in rows),
            "max_box_violation":max(row["donor"]["max_state_box_violation"] for row in rows),
            "max_native_parity":max((row["native_parity_max"] or 0.) for row in rows),
            "teacher_task_hits":sum(row["task"]["teacher_both_correct"] for row in rows),
            "full_bpd_task_hits":sum(row["task"]["bpd_full_both_correct"] for row in rows),
            "paired_z_myopic_hits":sum(row["task"]["myopic_second_correct"] for row in rows),
            "status":"engineering development/sham calibration only; no confirmation claim"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("development","sham","confirmation"),default="development")
    p.add_argument("--backend",choices=("independent","native"),default="independent")
    p.add_argument("--seeds",nargs="*",type=int)
    p.add_argument("--out",type=Path)
    p.add_argument("--no-raw-pairs",action="store_true")
    args=p.parse_args()
    run=cohort(args.phase,args.backend,args.seeds or None,not args.no_raw_pairs)
    summary=summarize(run)
    if args.out:
        if args.out.exists():raise FileExistsError("CQ08 result file is write-once: "+str(args.out))
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps({"run":run,"summary":summary},indent=2,sort_keys=True)+"\n")
    print(json.dumps({"phase":args.phase,"backend":args.backend,"summary":summary},indent=2))

if __name__=="__main__": main()
