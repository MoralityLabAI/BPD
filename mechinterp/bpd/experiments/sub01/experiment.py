"""BPD-SUB-01: causal quotient-grain engineering on trained four-pass recurrences.

Prospective protocol is PROTOCOL.md (committed before this file).
Only an excluded smoke root and eight independent development roots are allowed.
Do not treat any developmental score as a scientific confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
STUDY = "BPD-SUB-01"
ROOTS = tuple(range(1200, 1208))
SMOKE = 1199
WIDTH = 12
INPUTS = 4
PASSES = 4
RANKS = (2, 4, 6, 8, 10)
SHAMS = 8
STEPS = 256
TRAIN_SIZE = 2048
DISCOVERY_SIZE = 512
SELECTION_SIZE = 512
HELDOUT_SIZE = 1024
KL_GATE = 0.025
FIBER_GATE = 0.050
ALPHA_FRACTION = 0.25
MECHANISTIC_DELTA = 0.03
BOOTSTRAP_DRAWS = 2000


def source_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def protocol_sha() -> str:
    return hashlib.sha256((ROOT / "PROTOCOL.md").read_bytes()).hexdigest()


def sample_x(seed: int, split: int, n: int) -> torch.Tensor:
    if split not in (1, 2, 3, 4) or n < 1:
        raise ValueError("Unregistered data split")
    g = torch.Generator().manual_seed(seed * 10000 + split * 17011)
    return torch.randn(n, INPUTS, generator=g)


def external_labels(x: torch.Tensor) -> torch.Tensor:
    return ((0.85 * x[:, 0] + 0.6 * x[:, 1]
             + 0.35 * x[:, 2] * x[:, 3]) >= 0).float()


class Looper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(WIDTH, WIDTH, bias=False)
        self.inp = nn.Linear(INPUTS, WIDTH)
        self.head = nn.Linear(WIDTH, 1)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.inp.weight)

    def step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.hidden(h) + self.inp(x))

    def rollout(self, x: torch.Tensor, impulse: torch.Tensor | None = None):
        h = x.new_zeros(len(x), WIDTH)
        states = []
        for t in range(PASSES):
            h = self.step(h, x)
            if t == 0 and impulse is not None:
                h = h + impulse
            states.append(h)
        return self.head(h).squeeze(-1), states


def train_source(seed: int, steps: int = STEPS) -> tuple[Looper, dict]:
    if seed not in ROOTS and seed != SMOKE:
        raise ValueError("Unregistered SUB-01 model root")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = Looper()
    x = sample_x(seed, 1, TRAIN_SIZE)
    y = external_labels(x)
    g = torch.Generator().manual_seed(811031 + seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        idx = torch.randint(TRAIN_SIZE, (128,), generator=g)
        logits, _ = model.rollout(x[idx])
        loss = F.binary_cross_entropy_with_logits(logits, y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model.rollout(x)
        accuracy = float(((logits >= 0) == y.bool()).float().mean())
    return model, {
        "train_accuracy": accuracy, "training_steps": steps,
        "train_examples": TRAIN_SIZE, "optimizer": "Adam",
        "batch_size": 128, "learning_rate": 0.01,
    }


@torch.no_grad()
def pca_projectors(model: Looper, x: torch.Tensor):
    _, states = model.rollout(x)
    observations = torch.cat(states, dim=0).double()
    gram = observations.T @ observations / len(observations)
    vals, vectors = torch.linalg.eigh(gram)
    p = {r: (vectors[:, -r:] @ vectors[:, -r:].T).float() for r in RANKS}
    p[WIDTH] = torch.eye(WIDTH)
    nested = max(
        float((p[r] @ p[s] - p[r]).abs().max())
        for r, s in zip(RANKS, RANKS[1:])
    )
    alpha = float(states[0].double().square().sum(dim=1).mean().sqrt()
                  * ALPHA_FRACTION)
    if alpha <= 0:
        raise ValueError("Zero discovery state norm")
    return p, alpha, {
        "spectrum": vals.tolist(), "nested_max_abs": nested,
        "discovery_examples": len(x),
        "discover_h1_rms_norm": alpha / ALPHA_FRACTION,
    }


def random_orthogonal(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(WIDTH, WIDTH, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)
    return q.float()


def sham_null_projector(rank: int, seed: int, sham: int) -> torch.Tensor:
    if rank not in RANKS or not (0 <= sham < SHAMS):
        raise ValueError("Unregistered rank or sham")
    gen = torch.Generator().manual_seed(370001 + seed * 1009 + rank * 31 + sham)
    a = torch.randn(WIDTH, WIDTH - rank, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(a, mode="reduced")
    return (q @ q.T).float()


def impulse_directions(
    seed: int, split: int, x: torch.Tensor,
    null_projector: torch.Tensor, alpha: float,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(490019 + seed * 1009 + split * 31)
    raw = torch.randn(len(x), WIDTH, generator=gen)
    v = raw @ null_projector
    lengths = v.norm(dim=1, keepdim=True)
    if float(lengths.min()) < 1e-8:
        raise ValueError("Impulse projection is degenerate")
    return alpha * v / lengths


@torch.no_grad()
def projected_rollout(model: Looper, x: torch.Tensor, p: torch.Tensor):
    h = x.new_zeros(len(x), WIDTH)
    states = []
    for _ in range(PASSES):
        h = model.step(h, x) @ p
        states.append(h)
    return model.head(h).squeeze(-1), states


@torch.no_grad()
def gauge_rollout(
    model: Looper, x: torch.Tensor, q: torch.Tensor,
    impulse_z: torch.Tensor | None = None,
):
    z = x.new_zeros(len(x), WIDTH)
    states = []
    for t in range(PASSES):
        z = model.step(z @ q, x) @ q.T
        if t == 0 and impulse_z is not None:
            z = z + impulse_z
        states.append(z)
    return model.head(z @ q).squeeze(-1), states


def bernoulli_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor):
    p = torch.sigmoid(reference_logits)
    kl = (F.softplus(candidate_logits) - F.softplus(reference_logits)
          - p * (candidate_logits - reference_logits))
    return kl.clamp_min(0)


def rms(values: torch.Tensor) -> float:
    return float(values.double().square().mean().sqrt())


@torch.no_grad()
def fiber_score(
    model: Looper, x: torch.Tensor, p: torch.Tensor,
    null_p: torch.Tensor, alpha: float, seed: int, split: int,
) -> dict:
    clean_logits, clean_states = model.rollout(x)
    delta = impulse_directions(seed, split, x, null_p, alpha)
    shifted_logits, shifted_states = model.rollout(x, delta)
    next_diff = ((model.step(clean_states[0] + delta, x)
                  - model.step(clean_states[0], x)) @ p)
    future_shift = [
        rms(model.head(shifted_states[t]).squeeze(-1)
            - model.head(clean_states[t]).squeeze(-1))
        for t in (1, 2, 3)
    ]
    return {
        "terminal_logit_rms": rms(shifted_logits - clean_logits),
        "future_logit_rms_pass2_3_4": future_shift,
        "projected_next_hidden_norm_rms":
            float(next_diff.double().square().sum(dim=-1).mean().sqrt()),
        "physical_dose": alpha,
        "max_dose_deviation": float((delta.norm(dim=-1) - alpha).abs().max()),
        "max_null_projection": float((delta @ p).abs().max()),
    }


@torch.no_grad()
def rank_score(
    model: Looper, x: torch.Tensor, p: torch.Tensor,
    alpha: float, seed: int, split: int,
    *, include_shams: bool,
) -> dict:
    teacher, _ = model.rollout(x)
    projected, _ = projected_rollout(model, x, p)
    null_p = torch.eye(WIDTH) - p
    fiber = fiber_score(model, x, p, null_p, alpha, seed, split)
    result = {
        "clean_to_projected_kl_nats": float(
            bernoulli_kl(teacher, projected).mean()),
        "fiber": fiber,
        "projected_task_nll": float(F.binary_cross_entropy_with_logits(
            projected, external_labels(x))),
        "projected_accuracy": float(
            ((projected >= 0) == external_labels(x).bool()).float().mean()),
    }
    if include_shams:
        shams = [
            fiber_score(
                model, x, p, sham_null_projector(p_rank, seed, j),
                alpha, seed, split,
            )["terminal_logit_rms"]
            for j in range(SHAMS)
            for p_rank in (int(round(float(torch.trace(p)))),)
        ]
        result["eight_same_dose_sham_logit_rms"] = shams
        result["median_sham_minus_pca_logit_rms"] = float(
            np.median(shams) - fiber["terminal_logit_rms"])
    return result


@torch.no_grad()
def gauge_score(model: Looper, x: torch.Tensor, seed: int, alpha: float):
    q = random_orthogonal(700007 + seed)
    full = torch.eye(WIDTH)
    delta = impulse_directions(seed, 4, x, full, alpha)
    clean, clean_states = model.rollout(x)
    original, original_states = model.rollout(x, delta)
    conjugate_clean, gauge_states = gauge_rollout(model, x, q)
    conjugate_impulse, intervened_states = gauge_rollout(model, x, q, delta @ q.T)
    wrong_impulse, _ = gauge_rollout(model, x, q, delta)
    return {
        "orthogonality_max_abs": float(
            (q.T @ q - torch.eye(WIDTH)).abs().max()),
        "max_clean_logit_error": float((conjugate_clean-clean).abs().max()),
        "max_transported_impulse_logit_error":
            float((conjugate_impulse-original).abs().max()),
        "max_clean_state_error": max(float((gauge_states[t] @ q
                                      - clean_states[t]).abs().max())
                                     for t in range(PASSES)),
        "max_intervened_state_error": max(
            float((intervened_states[t] @ q
                   - original_states[t]).abs().max())
            for t in range(PASSES)),
        "untransported_impulse_logit_rms": rms(wrong_impulse - original),
    }


def evaluate_root(seed: int, *, steps: int = STEPS) -> dict:
    if seed not in ROOTS and seed != SMOKE:
        raise ValueError("Unregistered root")
    model, training = train_source(seed, steps=steps)
    discovery = sample_x(seed, 2, DISCOVERY_SIZE)
    selection = sample_x(seed, 3, SELECTION_SIZE)
    heldout = sample_x(seed, 4, HELDOUT_SIZE)
    projectors, alpha, geometry = pca_projectors(model, discovery)
    source_heldout_logits, _ = model.rollout(heldout)
    y = external_labels(heldout)
    contexts = {
        "x3_nonnegative": float(((source_heldout_logits[heldout[:, 3] >= 0] >= 0)
            == y[heldout[:, 3] >= 0].bool()).float().mean()),
        "x3_negative": float(((source_heldout_logits[heldout[:, 3] < 0] >= 0)
            == y[heldout[:, 3] < 0].bool()).float().mean()),
    }
    selected_metrics = {
        str(rank): rank_score(
            model, selection, projectors[rank], alpha, seed, 3,
            include_shams=False,
        )
        for rank in RANKS
    }
    eligible_ranks = [
        r for r in RANKS
        if selected_metrics[str(r)]["clean_to_projected_kl_nats"] <= KL_GATE
        and selected_metrics[str(r)]["fiber"]["terminal_logit_rms"] <= FIBER_GATE
    ]
    selected = min(eligible_ranks) if eligible_ranks else None
    heldout_metrics = {
        str(rank): rank_score(
            model, heldout, projectors[rank], alpha, seed, 4,
            include_shams=True,
        )
        for rank in RANKS
    }
    gauge = gauge_score(model, heldout[:128], seed, alpha)
    gauge_ok = (
        gauge["max_clean_logit_error"] <= 5e-5
        and gauge["max_transported_impulse_logit_error"] <= 5e-5
        and gauge["max_clean_state_error"] <= 5e-5
        and gauge["max_intervened_state_error"] <= 5e-5
        and gauge["untransported_impulse_logit_rms"] > 1e-5
    )
    dose_ok = all(
        metric["fiber"]["max_dose_deviation"] <= 5e-6
        and metric["fiber"]["max_null_projection"] <= 5e-6
        for metric in heldout_metrics.values()
    )
    competent = min(contexts.values()) >= .80
    ready = bool(selected is not None and competent and gauge_ok and dose_ok
                 and geometry["nested_max_abs"] <= 1e-5)
    return {
        "study": STUDY, "phase": "EXCLUDED_SMOKE" if seed == SMOKE else "ENGINEERING",
        "root": seed, "protocol_sha256": protocol_sha(),
        "source_sha256": source_sha(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__,
        },
        "training": training,
        "data_sizes": {
            "train": TRAIN_SIZE, "discovery": DISCOVERY_SIZE,
            "selection": SELECTION_SIZE, "heldout": HELDOUT_SIZE,
        },
        "geometry": geometry, "impulse_norm": alpha,
        "clean_heldout_task_nll": float(F.binary_cross_entropy_with_logits(
            source_heldout_logits, y)),
        "contexts": contexts, "competent": competent,
        "selection": selected_metrics, "selected_rank": selected,
        "all_ranks": heldout_metrics, "gauge": gauge,
        "gauge_ok": gauge_ok, "dose_ok": dose_ok,
        "root_eligible": ready,
        "selected_heldout": heldout_metrics.get(str(selected))
            if selected is not None else None,
    }


def root_bootstrap(values: list[float], seed: int = 120101):
    rng = np.random.default_rng(seed)
    a = np.asarray(values, dtype=float)
    sampled = rng.choice(a, size=(BOOTSTRAP_DRAWS, len(a)), replace=True).mean(1)
    return [float(np.quantile(sampled, .025)),
            float(np.quantile(sampled, .975))]


def aggregate(reports: list[dict]):
    if len(reports) != len(ROOTS) or sorted(z["root"] for z in reports) != list(ROOTS):
        raise ValueError("Exactly eight distinct preregistered model roots required")
    chosen = [
        z["selected_heldout"]["median_sham_minus_pca_logit_rms"]
        for z in reports if z["root_eligible"]
    ]
    ci = root_bootstrap(chosen) if chosen else None
    mean = float(np.mean(chosen)) if chosen else None
    passed = (
        len(chosen) >= 5 and all(z["gauge_ok"] and z["dose_ok"] for z in reports)
        and mean >= MECHANISTIC_DELTA and ci[0] > 0
    )
    return {
        "study": STUDY, "phase": "ENGINEERING",
        "protocol_sha256": reports[0]["protocol_sha256"],
        "source_sha256": reports[0]["source_sha256"],
        "units": len(reports), "root_ids": list(ROOTS),
        "eligible_units": len(chosen), "root_effects": chosen,
        "mean_sham_minus_pca_logit_rms": mean,
        "whole_root_bootstrap95": ci,
        "registered_scientific_minimum": MECHANISTIC_DELTA,
        "scientific_sham_tail_calibrated": False,
        "evalmath_strict_preflight": "PENDING_INDEPENDENT_CALIBRATION",
        "verdict": "ENGINEERING_SIGNAL_ONLY" if passed
            else "ENGINEERING_NULL_OR_INELIGIBLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "development"), required=True)
    parser.add_argument("--ack-development", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "development" and not args.ack_development:
        parser.error("Development requires explicit --ack-development")
    if args.phase == "smoke":
        receipt = evaluate_root(SMOKE, steps=16)
    else:
        roots = [evaluate_root(seed) for seed in ROOTS]
        receipt = {"summary": aggregate(roots), "roots": roots}
    serialized = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if "summary" in receipt:
        compact = receipt["summary"]
    else:
        compact = {
            "study": STUDY, "root": SMOKE,
            "gauge": receipt["gauge"],
            "selected_rank": receipt["selected_rank"],
        }
    print(json.dumps({"summary": compact}, sort_keys=True))


if __name__ == "__main__":
    main()
