"""BPD-NC-01: source-registered recurrent neuron-community CPU engineering pilot.

All scientific gates are documented in PROTOCOL.md. This implementation
compares exact full-rank read/write coordinate systems of one physical shared
linear layer. The SVD arm is BPD's *initialization*, not a learned dictionary.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig

PASSES = 4
WIDTH = 12
INPUT_DIM = 10
TOP_K = 8
MIN_CO = 12
MIN_PHI = 0.15
NEIGHBORS = 4
MAX_COMMUNITY = 12
DOSE_TOLERANCE = 0.15
SHAMS = 8


class TinyLooper(nn.Module):
    """Trained, nonlinear, weight-tied recurrent student; one shared MLP write."""

    def __init__(self, d: int = INPUT_DIM, width: int = WIDTH) -> None:
        super().__init__()
        self.embed = nn.Linear(d, width)
        self.pre = nn.Linear(width, width)
        self.drive = nn.Linear(d, width)
        self.mix = nn.Linear(width, width, bias=False)
        self.head = nn.Linear(width, 1)
        self.acts: list[torch.Tensor] = []
        self.writes: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, *, capture: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.embed(x))
        logits, states = [], []
        self.acts = []
        self.writes = []
        for _ in range(PASSES):
            a = F.gelu(self.pre(h) + self.drive(x))
            w = self.mix(a.unsqueeze(1)).squeeze(1)
            if capture:
                self.acts.append(a)
                self.writes.append(w)
            h = torch.tanh(h + 0.45 * w)
            logits.append(self.head(h).squeeze(-1))
            states.append(h)
        return torch.stack(logits, dim=1), torch.stack(states, dim=1)


def sample_x(seed: int, split: int, n: int) -> torch.Tensor:
    """Disjoint deterministic pseudorandom streams; no held-out access in training."""
    if n < 1:
        raise ValueError("n must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed * 100_003 + split * 1_009)
    return torch.randn(n, INPUT_DIM, generator=generator)


def train_student(seed: int, steps: int = 120) -> tuple[TinyLooper, TinyLooper, dict]:
    torch.manual_seed(seed)
    teacher = TinyLooper()
    # Estimate the target-class split on an *independent training-only* pool.
    with torch.no_grad():
        median = teacher(sample_x(seed, 1, 2048))[0][:, -1].median()
        teacher.head.bias.sub_(median)
    student = copy.deepcopy(teacher)
    g = torch.Generator().manual_seed(seed + 43_003)
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.14)
    x_train = sample_x(seed, 2, 2048)
    with torch.no_grad():
        t_logits, t_states = teacher(x_train)
    opt = torch.optim.Adam(student.parameters(), lr=0.008)
    for step in range(steps):
        idx = torch.randint(0, len(x_train), (128,), generator=g)
        out, states = student(x_train[idx])
        loss = F.mse_loss(out, t_logits[idx]) + 0.3 * F.mse_loss(states, t_states[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    student.eval()
    teacher.eval()
    return teacher, student, {"seed": seed, "training_steps": steps, "last_minibatch_mse": float(loss.detach())}


def labels_from_teacher(teacher: TinyLooper, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return teacher(x)[0][:, -1] >= 0.0


def make_basis(w: torch.Tensor, kind: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return V [in,C], U [C,out], with (V@U).T = W."""
    width = w.shape[1]
    if w.shape != (width, width):
        raise ValueError("NC-01 requires a square full-rank shared layer")
    if kind == "native":
        v = torch.eye(width)
        u = w.T.contiguous()
    elif kind == "bpd_svd":
        left, singular, vh = torch.linalg.svd(w.double(), full_matrices=False)
        root = singular.sqrt()
        v = (vh.T * root.unsqueeze(0)).float()
        u = (root.unsqueeze(1) * left.T).float()
    elif kind == "rotated":
        g = torch.Generator().manual_seed(seed + 77_501)
        mat = torch.randn(width, width, generator=g, dtype=torch.float64)
        q, r = torch.linalg.qr(mat)
        q = q * torch.sign(torch.diag(r)).unsqueeze(0)
        v = q.float()
        u = v.T @ w.T
    else:
        raise ValueError(f"unknown basis: {kind}")
    if (v @ u).T.sub(w).norm() / w.norm().clamp_min(1e-12) > 1e-5:
        raise RuntimeError("non-closing factor reconstruction")
    return v, u


def set_basis(bpd: BPDModel, kind: str, seed: int) -> None:
    layer = bpd.wrappers["mix"]
    v, u = make_basis(layer.W_target, kind, seed)
    with torch.no_grad():
        layer.V.copy_(v)
        layer.U.copy_(u)


def factor_schedule(batch: int, mask: torch.Tensor) -> PassMaskSchedule:
    """mask [T,C]; zero delta excludes an unmeasured residual safety channel."""
    if mask.shape != (PASSES, WIDTH):
        raise ValueError("expected pass-by-coordinate mask")
    return PassMaskSchedule(
        comp_mask=mask.view(1, PASSES, 1, WIDTH).expand(batch, -1, -1, -1).contiguous(),
        delta_mask=torch.zeros(batch, PASSES, 1),
    )


def binary_kl(clean: torch.Tensor, altered: torch.Tensor) -> torch.Tensor:
    """Per-example, per-pass KL for two-class logits [-s,+s], in nats."""
    # Reference logits may differ by <1e-7. Float32 softmax-KL cancellation
    # can report an unphysical positive KL near 1e-7; audit in float64.
    clean64, altered64 = clean.double(), altered.double()
    p = F.softmax(torch.stack((-clean64, clean64), dim=-1), dim=-1)
    q_log = F.log_softmax(torch.stack((-altered64, altered64), dim=-1), dim=-1)
    p_log = F.log_softmax(torch.stack((-clean64, clean64), dim=-1), dim=-1)
    return (p * (p_log - q_log)).sum(dim=-1)


def closure_check(bpd: BPDModel, model: TinyLooper, x: torch.Tensor) -> dict:
    model.eval()
    layer = bpd.wrappers["mix"]
    ratio = float(layer.weight_delta().norm() / layer.W_target.norm().clamp_min(1e-12))
    with torch.no_grad():
        with bpd.target_capture():
            target, _ = model(x)
        with bpd.masked_components({"mix": factor_schedule(len(x), torch.ones(PASSES, WIDTH))}):
            opened, _ = model(x)
    max_abs = float((target - opened).abs().max())
    max_kl = float(binary_kl(target, opened).clamp_min(0).max())
    passed = ratio <= 1e-4 and max_abs <= 5e-5 and max_kl <= 1e-7
    return {"passed": passed, "residual_ratio": ratio, "max_abs_logits": max_abs, "max_kl_nats": max_kl}


def attribution_selection(bpd: BPDModel, model: TinyLooper, x: torch.Tensor,
                          y: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    """Pass-indexed factor attribution z * d(score)/dz; score is label-aligned."""
    layer = bpd.wrappers["mix"]
    rows: list[torch.Tensor] = []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].clone().requires_grad_(True)
        yb = y[start:start + batch_size]
        with bpd.target_capture():
            output, _ = model(xb, capture=True)
            sign = torch.where(yb, 1.0, -1.0)
            score = (sign * output[:, -1]).sum()
            grads = torch.autograd.grad(score, model.writes)
        a = torch.stack(model.acts, dim=1).detach()  # [B,T,C]
        gw = torch.stack(grads, dim=1).detach()
        z = a @ layer.V.detach()
        gz = gw @ layer.U.detach().T
        importance = (z * gz).reshape(len(xb), -1)
        top_value, top_index = importance.topk(min(TOP_K, importance.shape[1]), dim=1)
        selected = torch.zeros_like(importance)
        selected.scatter_(1, top_index, (top_value > 0).float())
        rows.append(selected)
    return torch.cat(rows).to(torch.float64)


def communities_from_selection(selected: torch.Tensor,
                               min_phi: float = MIN_PHI,
                               min_co: int = MIN_CO) -> list[dict]:
    """Frequency-corrected phi graph; mutual top-k edges; deterministic components."""
    if selected.ndim != 2:
        raise ValueError("selected must be [sample, node]")
    n, width = selected.shape
    if n < 1:
        raise ValueError("empty selection")
    hits = selected.double()
    freq = hits.mean(dim=0)
    joint = hits.T @ hits
    cov = joint / n - torch.outer(freq, freq)
    denom = (freq * (1 - freq)).clamp_min(0).sqrt()
    denom = torch.outer(denom, denom).clamp_min(1e-12)
    phi = cov / denom
    neighbors: list[list[int]] = []
    for i in range(width):
        ranked = [
            j for j in range(width)
            if j != i and joint[i, j] >= min_co and float(phi[i, j]) >= min_phi
        ]
        ranked.sort(key=lambda j: (-float(phi[i, j]), j))
        neighbors.append(ranked[:NEIGHBORS])
    adjacency: dict[int, dict[int, float]] = {i: {} for i in range(width)}
    for i in range(width):
        for j in neighbors[i]:
            if i in neighbors[j]:
                adjacency[i][j] = float(phi[i, j])
                adjacency[j][i] = float(phi[i, j])
    seen: set[int] = set()
    out: list[dict] = []
    for i in range(width):
        if i in seen or not adjacency[i]:
            continue
        stack, nodes = [i], []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            nodes.append(v)
            stack.extend(sorted(adjacency[v], reverse=True))
        nodes.sort()
        if not 2 <= len(nodes) <= MAX_COMMUNITY:
            continue
        score = sum(adjacency[a].get(b, 0.0) for a in nodes for b in nodes if a < b)
        out.append({
            "nodes": nodes,
            "size": len(nodes),
            "passes": sorted({i // WIDTH for i in nodes}),
            "internal_phi_mass": round(score, 9),
        })
    out.sort(key=lambda c: (-c["internal_phi_mass"], c["nodes"]))
    return out


def node_mask(nodes: list[int]) -> torch.Tensor:
    mask = torch.zeros(PASSES * WIDTH)
    if len(set(nodes)) != len(nodes) or any(i < 0 or i >= PASSES * WIDTH for i in nodes):
        raise ValueError("invalid selected nodes")
    mask[nodes] = 1
    return mask.view(PASSES, WIDTH)


def direct_write_rms(acts: torch.Tensor, v: torch.Tensor, u: torch.Tensor,
                     nodes: list[int]) -> float:
    """Discovery-only physical write RMS in clean activation context."""
    m = node_mask(nodes)
    z = acts @ v
    selected_write = ((z * m) @ u)
    return float(selected_write.square().mean().sqrt())


def choose_matched_shams(nodes: list[int], acts: torch.Tensor, v: torch.Tensor,
                         u: torch.Tensor, seed: int, count: int = SHAMS) -> dict:
    """Same per-pass counts; choose RMS-matched unique shams on discovery only."""
    counts = [sum(i // WIDTH == t for i in nodes) for t in range(PASSES)]
    # Exact clean-context write RMS via per-pass second-moment Gram matrices.
    # Avoid recomputing a full [B,T,C] matrix product for each random null.
    z = acts @ v
    gram = (
        torch.einsum("bti,btj->tij", z, z) / len(z)
        * (u @ u.T).unsqueeze(0) / (PASSES * u.shape[1])
    )
    def from_gram(group: list[int]) -> float:
        m = node_mask(group)
        return float(torch.einsum("ti,tij,tj->", m, gram, m).clamp_min(0).sqrt())
    target_dose = from_gram(nodes)
    g = np.random.default_rng(seed)
    target = tuple(nodes)
    candidates: list[tuple[float, list[int], float]] = []
    seen = {target}
    for _ in range(1024):
        sample = sorted(
            int(t * WIDTH + j)
            for t, count_t in enumerate(counts)
            for j in g.choice(WIDTH, size=count_t, replace=False)
        )
        key = tuple(sample)
        if key in seen:
            continue
        seen.add(key)
        dose = from_gram(sample)
        rel = abs(dose - target_dose) / max(target_dose, 1e-10)
        if rel <= DOSE_TOLERANCE:
            candidates.append((rel, sample, dose))
    candidates.sort(key=lambda e: (e[0], e[1]))
    shams = [{"nodes": c[1], "discovery_dose_rms": c[2], "relative_dose_error": c[0]}
             for c in candidates[:count]]
    return {
        "eligible": len(shams) == count and target_dose > 1e-10,
        "community_discovery_dose_rms": target_dose,
        "eligible_candidate_count": len(candidates),
        "shams": shams,
    }


@torch.no_grad()
def evaluate_masks(bpd: BPDModel, model: TinyLooper, x: torch.Tensor,
                   y: torch.Tensor, nodes: list[int]) -> dict:
    selection = node_mask(nodes)
    mask_ablate = 1 - selection
    mask_retain = selection
    with bpd.target_capture():
        clean, _ = model(x, capture=True)
        writes = torch.stack(model.writes, dim=1)
    result = {"teacher_label_accuracy_clean": float(((clean[:, -1] >= 0) == y).float().mean())}
    sign = torch.where(y, 1.0, -1.0)
    nll_clean = F.softplus(-2 * sign * clean[:, -1])
    for name, mask in (("ablate", mask_ablate), ("retain", mask_retain)):
        with bpd.masked_components({"mix": factor_schedule(len(x), mask)}):
            altered, _ = model(x, capture=True)
            changed_write = torch.stack(model.writes, dim=1)
        nll = F.softplus(-2 * sign * altered[:, -1])
        kl = binary_kl(clean, altered).clamp_min(0)
        result[name] = {
            "nll_change_nats": float((nll - nll_clean).mean()),
            "kl_by_pass_nats": [float(k) for k in kl.mean(dim=0)],
            "final_kl_nats": float(kl[:, -1].mean()),
            "teacher_label_accuracy": float(((altered[:, -1] >= 0) == y).float().mean()),
            "closed_loop_write_delta_rms": float((changed_write - writes).square().mean().sqrt()),
        }
    return result


def run_pilot(seed: int = 100, *, steps: int = 120,
              discovery_size: int = 1024, evaluation_size: int = 1024) -> dict:
    torch.set_num_threads(1)
    teacher, model, training = train_student(seed, steps)
    x_discovery = sample_x(seed, 3, discovery_size)
    x_evaluation = sample_x(seed, 4, evaluation_size)
    y_discovery = labels_from_teacher(teacher, x_discovery)
    y_evaluation = labels_from_teacher(teacher, x_evaluation)
    bpd = BPDModel(model, {"mix": WIDTH}, max_passes=PASSES,
                   ablation=AblationConfig(mode="zero", sphere_rescale=False), init_svd=True)
    output = {
        "protocol": "BPD-NC-01",
        "evidence_class": "engineering_CPU_only",
        "seed": seed,
        "data": {"discovery": discovery_size, "evaluation": evaluation_size},
        "training": training, "bases": {},
    }
    for basis in ("native", "bpd_svd", "rotated"):
        set_basis(bpd, basis, seed)
        closure = closure_check(bpd, model, x_evaluation[:min(64, len(x_evaluation))])
        details: dict = {"closure": closure, "communities": [], "eligible": False}
        output["bases"][basis] = details
        if not closure["passed"]:
            details["status"] = "BLOCKED_PARITY"
            continue
        selected = attribution_selection(bpd, model, x_discovery, y_discovery)
        groups = communities_from_selection(selected)
        details["detected_communities"] = len(groups)
        details["selection_node_frequencies"] = selected.mean(dim=0).tolist()
        with torch.no_grad():
            with bpd.target_capture():
                model(x_discovery, capture=True)
                acts = torch.stack(model.acts, dim=1)
        layer = bpd.wrappers["mix"]
        for index, group in enumerate(groups[:3]):
            matched = choose_matched_shams(
                group["nodes"], acts, layer.V.detach(), layer.U.detach(),
                seed * 10_000 + index + {"native": 0, "bpd_svd": 100, "rotated": 200}[basis],
            )
            record = {**group, "matching": matched}
            if matched["eligible"]:
                record["observed"] = evaluate_masks(bpd, model, x_evaluation, y_evaluation, group["nodes"])
                record["sham_outcomes"] = [
                    evaluate_masks(bpd, model, x_evaluation, y_evaluation, sham["nodes"])
                    for sham in matched["shams"]
                ]
                med_necessity = float(np.median([
                    o["ablate"]["nll_change_nats"] for o in record["sham_outcomes"]
                ]))
                med_suff = float(np.median([
                    o["retain"]["final_kl_nats"] for o in record["sham_outcomes"]
                ]))
                record["effects_vs_median_sham"] = {
                    "necessity_delta_nll_nats": record["observed"]["ablate"]["nll_change_nats"] - med_necessity,
                    "sufficiency_kl_advantage_nats": med_suff - record["observed"]["retain"]["final_kl_nats"],
                }
                details["eligible"] = True
            else:
                record["status"] = "BLOCKED_INSUFFICIENT_DOSE_MATCHED_SHAMS"
            details["communities"].append(record)
        details["status"] = "ENGINEERING_ONLY" if details["eligible"] else "NO_ELIGIBLE_COMMUNITIES"
    output["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--discovery-size", type=int, default=1024)
    p.add_argument("--evaluation-size", type=int, default=1024)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.seed not in (100, 101, 102, 103):
        raise SystemExit("Initial NC-01 runner permits only registered engineering roots 100-103")
    output = run_pilot(args.seed, steps=args.steps, discovery_size=args.discovery_size,
                       evaluation_size=args.evaluation_size)
    report = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as f:
            f.write(report)


if __name__ == "__main__":
    main()
