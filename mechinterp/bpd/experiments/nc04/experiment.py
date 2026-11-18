"""NC-04: self-scored causal subspaces in a freely trained dense recurrence.

Source-registered separately at PROTOCOL_COMMIT before this implementation.
Only engineering roots 400..407 are available; no confirmation population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import (
    PASSES, WIDTH, sample_x, external_labels, make_basis,
    binary_kl, parity, communities, shuffle_null,
)
from mechinterp.bpd.experiments.nc03.experiment import (
    sham_projectors, masked_rollout, basis_transport, discovery_moments,
    clean_write_dose,
)

ROOTS = tuple(range(400, 408))
PROTOCOL_COMMIT = "0c84c56f8f747737833f0c51201d369dcb24a790"
RANK = 2
GAP_RATIO = 1.10
SHAMS = 8
PROPOSALS = 2048


class DenseLooper(nn.Module):
    """Unconstrained trainable 20D nonlinear transition shared across four passes."""

    def __init__(self):
        super().__init__()
        self.encode = nn.Linear(5, WIDTH)
        self.pre = nn.Linear(WIDTH, WIDTH)
        self.mix = nn.Linear(WIDTH, WIDTH, bias=False)
        self.head = nn.Linear(WIDTH, 1)
        self.acts: list[Tensor] = []
        self.writes: list[Tensor] = []

    def forward(self, x: Tensor, *, capture: bool = False) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or x.shape[-1] != 5:
            raise ValueError("expected [batch,5] inputs")
        h = torch.tanh(self.encode(x))
        self.acts = []
        self.writes = []
        readouts, states = [], []
        for _ in range(PASSES):
            a = torch.tanh(self.pre(h))
            w = self.mix(a.unsqueeze(1)).squeeze(1)
            if capture:
                self.acts.append(a)
                self.writes.append(w)
            h = torch.tanh(0.60 * h + 0.65 * w)
            readouts.append(self.head(h).squeeze(-1))
            states.append(h)
        return torch.stack(readouts, dim=1), torch.stack(states, dim=1)


def train_model(seed: int, steps: int = 384) -> tuple[DenseLooper, dict]:
    if seed not in ROOTS:
        raise ValueError("only engineering roots 400..407 are registered")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = DenseLooper()
    x_train = sample_x(seed, 1, 2048)
    y_train = torch.where(external_labels(x_train), 1., -1.)
    gen = torch.Generator().manual_seed(seed + 1_737_701)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(steps):
        ids = torch.randint(len(x_train), (128,), generator=gen)
        logit = model(x_train[ids])[0][:, -1]
        loss = F.softplus(-2 * y_train[ids] * logit).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        z = model(x_train)[0][:, -1]
        train_acc = float(((z >= 0) == external_labels(x_train)).float().mean())
        mix_w = model.mix.weight.detach()
        density = float((mix_w.abs() >= 1e-4).float().mean())
    return model, {
        "seed": seed, "training_steps": steps, "batch_size": 128,
        "learning_rate": 0.005, "train_size": 2048,
        "last_minibatch_nll_nats": float(loss.detach()),
        "train_accuracy": train_acc,
        "mix_weight_density_abs_ge_1e-4": density,
        "mix_weight_frobenius": float(mix_w.norm()),
        "n_trainable_parameters": sum(p.numel() for p in model.parameters()),
    }


def factor_basis(bpd: BPDModel, kind: str, seed: int) -> None:
    layer = bpd.wrappers["mix"]
    v, u = make_basis(layer.W_target, kind, seed)
    with torch.no_grad():
        layer.V.copy_(v)
        layer.U.copy_(u)


def self_score_discovery(bpd: BPDModel, model: DenseLooper, x: Tensor,
                         batch_size: int = 128) -> tuple[Tensor, Tensor, Tensor, dict]:
    """Oracle-blind: does not accept or retrieve external task labels.

    Returns z (activations), dz (score gradients) and Goodfire-style hits.
    A detached model-prediction sign aligns confidence for positive attributions.
    """
    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("discovery input must be [N,5]")
    layer = bpd.wrappers["mix"]
    activations, gradients, hits = [], [], []
    nonzero_score = 0
    for start in range(0, len(x), batch_size):
        xx = x[start:start+batch_size].clone().requires_grad_(True)
        with bpd.target_capture():
            logits, _ = model(xx, capture=True)
            score = logits[:, -1]
            sign = torch.where(score.detach() >= 0, 1., -1.)
            grads = torch.autograd.grad((sign * score).sum(), model.writes)
        a = torch.stack(model.acts, 1).detach()
        g_w = torch.stack(grads, 1).detach()
        z = (a @ layer.V.detach()).double()
        g_z = (g_w @ layer.U.detach().T).double()
        importance = (z*g_z).reshape(len(xx), PASSES * WIDTH)
        top_v, top_i = importance.topk(8, dim=1)
        selected = torch.zeros_like(importance)
        selected.scatter_(1, top_i, (top_v > 0).double())
        hits.append(selected)
        activations.append(z)
        gradients.append(g_z)
        nonzero_score += int((score.detach() != 0).sum())
    return (torch.cat(activations), torch.cat(gradients),
            torch.cat(hits), {"model_score_nonzero_count": nonzero_score,
                              "independent_discovery_examples": len(x),
                              "score": "self_sign_times_final_model_logit"})


def self_gradient_projectors(grad: Tensor) -> tuple[Tensor | None, dict]:
    """Top two directions of uncentered score-gradient second moment."""
    if grad.ndim != 3 or grad.shape[1:] != (PASSES, WIDTH) or len(grad) < 1:
        raise ValueError("gradients must be [N,4,20] with N >=1")
    c = torch.einsum("nti,ntj->tij", grad.double(), grad.double()) / len(grad)
    values, vectors = torch.linalg.eigh(c)
    spectra = []
    for t in range(PASSES):
        second = float(values[t, -2])
        third = float(values[t, -3])
        ratio = second / max(third, 1e-12)
        spectra.append({
            "pass": t, "largest_eigenvalue": float(values[t, -1]),
            "second_eigenvalue": second, "third_eigenvalue": third,
            "eigengap_ratio": ratio,
            "top2_fraction_second_moment": float(
                values[t, -2:].sum() / values[t].clamp_min(0).sum().clamp_min(1e-30)),
        })
    if any(p["second_eigenvalue"] <= 1e-12 or p["eigengap_ratio"] < GAP_RATIO
           for p in spectra):
        return None, {"status": "UNIDENTIFIABLE", "spectrum": spectra}
    u = vectors[:, :, -RANK:]
    p = u @ u.transpose(-1, -2)
    return p, {
        "status": "IDENTIFIED_RANK_TWO",
        "spectrum": spectra,
        "symmetry_max_abs": float((p-p.transpose(-1,-2)).abs().max()),
        "idempotence_max_abs": float((p@p-p).abs().max()),
    }


@torch.no_grad()
def evaluate_labels(bpd: BPDModel, model: DenseLooper,
                    x: Tensor, y: Tensor, p: Tensor) -> dict:
    """External task labels enter *only here*, after discovery and null selection."""
    with bpd.target_capture():
        clean, _ = model(x, capture=True)
        write = torch.stack(model.writes, 1)
    sign = torch.where(y, 1., -1.)
    nll = F.softplus(-2 * sign * clean[:, -1])
    acc = [(float(((clean[:,t] >=0)==y).float().mean())) for t in range(PASSES)]
    report = {
        "clean_final_nll_nats": float(nll.mean()),
        "clean_pass_accuracy": acc,
        "clean_final_accuracy": acc[-1],
        "arms": {},
    }
    for mode in ("retain", "ablate"):
        logits, new_w = masked_rollout(bpd, model, x, p, mode=mode)
        intervention_nll = F.softplus(-2 * sign * logits[:, -1])
        kl = binary_kl(clean, logits).clamp_min(0)
        report["arms"][mode] = {
            "final_nll_nats": float(intervention_nll.mean()),
            "nll_change_from_clean_nats": float((intervention_nll - nll).mean()),
            "final_label_accuracy": float(((logits[:, -1]>=0)==y).float().mean()),
            "kl_by_pass_nats": [float(v) for v in kl.mean(0)],
            "final_kl_nats": float(kl[:, -1].mean()),
            "closed_loop_write_delta_rms": float((new_w-write).square().mean().sqrt()),
        }
    return report


def context_accuracies(model: DenseLooper, x: Tensor, y: Tensor) -> dict:
    with torch.no_grad():
        z = model(x)[0][:, -1]
        return {
            "context_positive": float(((z[x[:,4] > 0]>=0)==y[x[:,4]>0]).float().mean()),
            "context_nonpositive": float(((z[x[:,4]<=0]>=0)==y[x[:,4]<=0]).float().mean()),
        }


def evaluate_root(seed: int, *, steps: int = 384,
                  discovery_size: int = 1024, evaluation_size: int = 1024,
                  proposals: int = PROPOSALS, shams: int = SHAMS) -> dict:
    if seed not in ROOTS:
        raise ValueError("NC-04 uses engineering roots 400..407 only")
    torch.set_num_threads(1)
    model, train = train_model(seed, steps)
    xd = sample_x(seed, 2, discovery_size)
    xe = sample_x(seed, 3, evaluation_size)
    ye = external_labels(xe)  # never passed to discovery
    context_acc = context_accuracies(model, xe, ye)
    bpd = BPDModel(model, {"mix": WIDTH}, max_passes=PASSES,
                   ablation=AblationConfig(mode="zero", sphere_rescale=False))
    record = {
        "study": "BPD-NC-04", "evidence_class": "ENGINEERING_ONLY",
        "protocol_commit": PROTOCOL_COMMIT,
        "phase": "ENGINEERING", "seed": seed,
        "data": {"train": 2048, "discovery": discovery_size,
                 "evaluation": evaluation_size},
        "training": train, "context_accuracy": context_acc,
        "model_task_passed": min(context_acc.values()) >= .85
                             and train["mix_weight_density_abs_ge_1e-4"] >= .90,
        "bases": {}, "negative_controls": {},
    }
    native_projector = None
    native_map = None
    native_retained = None
    same_shams: list[Tensor] = []
    for kind in ("native", "rotated"):
        factor_basis(bpd, kind, seed)
        layer = bpd.wrappers["mix"]
        closure = parity(bpd, model, xe[:min(len(xe),64)])
        arm = {"closure": closure, "status": "PENDING"}
        record["bases"][kind] = arm
        if not closure["passed"]:
            arm["status"] = "BLOCKED_PARITY"
            continue
        acts, grad, hits, self_score = self_score_discovery(bpd, model, xd)
        p, spectral = self_gradient_projectors(grad)
        arm["self_score"] = self_score
        arm["projector_spectrum"] = spectral
        observed_groups = communities(hits)
        arm["native_style_community_diagnostic"] = {
            "n_communities": len(observed_groups),
            "cross_pass_community_count": sum(len(g["passes"])>1 for g in observed_groups),
            "top_four_groups": observed_groups[:4],
            "shuffled_null": shuffle_null(hits, seed+{"native": 0,"rotated":17}[kind], n_shuffles=16),
        }
        if p is None:
            arm["status"] = "UNIDENTIFIABLE"
            continue
        arm["physical_map_norms"] = [
            float(v) for v in (layer.V.detach().double() @ p @
                               layer.U.detach().double()).norm(dim=(1,2))
        ]
        physical_map = layer.V.detach().double() @ p @ layer.U.detach().double()
        arm["evaluation"] = evaluate_labels(bpd, model, xe, ye, p)
        kept, _ = masked_rollout(bpd, model, xe, p, mode="retain")
        if kind == "native":
            native_projector = p
            native_map = physical_map
            native_retained = kept
            same_shams, matching = sham_projectors(
                acts, layer.U.detach(), p, seed,
                proposals=proposals, want=shams,
            )
            arm["sham_matching"] = matching
            moment, uu = discovery_moments(acts, layer.V.detach(), layer.U.detach())
            arm["projector_clean_discovery_write_rms"] = clean_write_dose(moment, uu, p)
        elif native_projector is not None and native_map is not None:
            arm["native_rotation_equivariance"] = {
                "relative_operator_difference": float(
                    (physical_map-native_map).norm(dim=(1,2)).max() /
                    layer.W_target.detach().double().norm().clamp_min(1e-12)),
                "max_kept_logit_difference": float((kept-native_retained).abs().max()),
                "max_projector_transport_difference": float(
                    (p-basis_transport(native_projector, layer.V.detach())).abs().max()),
            }
        if len(same_shams) == shams:
            null_scores = []
            for control in same_shams:
                control_here = (control if kind == "native" else
                                basis_transport(control, layer.V.detach()))
                null_scores.append(evaluate_labels(bpd, model, xe, ye, control_here))
            arm["matched_sham_scores"] = [{
                "final_label_nll_nats": e["arms"]["retain"]["final_nll_nats"],
                "keep_final_kl_nats": e["arms"]["retain"]["final_kl_nats"],
                "retained_accuracy": e["arms"]["retain"]["final_label_accuracy"],
                "physical_write_delta_rms": e["arms"]["retain"]["closed_loop_write_delta_rms"],
            } for e in null_scores]
            arm["primary_external_nll_advantage_nats"] = float(
                np.median([e["final_label_nll_nats"] for e in arm["matched_sham_scores"]])
                - arm["evaluation"]["arms"]["retain"]["final_nll_nats"]
            )
            arm["secondary_keep_final_kl_advantage_nats"] = float(
                np.median([e["keep_final_kl_nats"] for e in arm["matched_sham_scores"]])
                - arm["evaluation"]["arms"]["retain"]["final_kl_nats"]
            )
            arm["status"] = "ENGINEERING"
        else:
            arm["status"] = "BLOCKED_SHAMS"
    # True zero-readout negative control, on discovery examples only.
    factor_basis(bpd, "native", seed)
    with torch.no_grad():
        head_w, head_b = model.head.weight.detach().clone(), model.head.bias.detach().clone()
        model.head.weight.zero_()
        model.head.bias.zero_()
    try:
        _, g_zero, h_zero, _ = self_score_discovery(bpd, model, xd[:min(256,len(xd))])
        null_p, null_diag = self_gradient_projectors(g_zero)
        record["negative_controls"]["zero_readout"] = {
            "max_abs_gradient": float(g_zero.abs().max()),
            "positive_attribution_hits": int(h_zero.sum()),
            "subspace_status": null_diag["status"],
            "projector_returned": null_p is not None,
        }
    finally:
        with torch.no_grad():
            model.head.weight.copy_(head_w)
            model.head.bias.copy_(head_b)
    record["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record["protocol_sha256"] = hashlib.sha256(
        Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest()
    return record


def main() -> None:
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--seed", type=int, default=400)
    arg.add_argument("--steps", type=int, default=384)
    arg.add_argument("--discovery-size", type=int, default=1024)
    arg.add_argument("--evaluation-size", type=int, default=1024)
    arg.add_argument("--proposals", type=int, default=PROPOSALS)
    arg.add_argument("--out", type=Path)
    opts = arg.parse_args()
    result = evaluate_root(opts.seed, steps=opts.steps,
                           discovery_size=opts.discovery_size,
                           evaluation_size=opts.evaluation_size,
                           proposals=opts.proposals)
    data = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if opts.out:
        opts.out.parent.mkdir(parents=True, exist_ok=True)
        with opts.out.open("x", encoding="utf-8") as f:
            f.write(data)
    else:
        print(data)


if __name__ == "__main__":
    main()
