"""NC-05: source-registered rank-dose frontier on fresh dense recurrent roots.

The frozen source-only PROTOCOL.md precedes this implementation. All
admissions are discovery-only; external labels enter held-out scoring only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import (
    PASSES, WIDTH, sample_x, external_labels, binary_kl, parity,
)
from mechinterp.bpd.experiments.nc04.experiment import (
    DenseLooper, factor_basis, self_score_discovery, context_accuracies,
)
from mechinterp.bpd.experiments.nc03.experiment import (
    basis_transport,
)

PROTOCOL_COMMIT = "f7f1d2346f9d3f871049a3d03416ef7980fe90f8"
ROOTS = tuple(range(500, 508))
RANKS = (2, 4, 6, 8)
GAINS = (0.75, 1.0)
ANGLES_RAD = (0.45, 0.65, 0.85, 1.05)
PROPOSALS = 1024
SHAMS = 8
MATCH_TOL = 0.15


def train_model(seed: int, steps: int = 384) -> tuple[DenseLooper, dict]:
    if seed not in ROOTS:
        raise ValueError("NC-05 is limited to engineering seeds 500..507")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = DenseLooper()
    x = sample_x(seed, 1, 2048)
    signs = torch.where(external_labels(x), 1., -1.)
    rng = torch.Generator().manual_seed(seed + 1_737_701)
    optim = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(steps):
        idx = torch.randint(len(x), (128,), generator=rng)
        pred, _ = model(x[idx])
        loss = F.softplus(-2 * signs[idx] * pred[:, -1]).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        pred, _ = model(x)
        density = float((model.mix.weight.detach().abs() >= 1e-4).float().mean())
        accuracy = float(((pred[:, -1] >= 0) == external_labels(x)).float().mean())
    return model, {
        "seed": seed, "training_steps": steps, "train_size": 2048,
        "batch_size": 128, "learning_rate": 0.005,
        "training_accuracy": accuracy,
        "last_minibatch_nll_nats": float(loss.detach()),
        "mix_weight_density": density,
        "n_trainable_parameters": sum(q.numel() for q in model.parameters()),
    }


def projectors_at_ranks(grads: Tensor) -> tuple[dict[int, Tensor | None], dict[int, dict], dict[int, Tensor]]:
    """Uncentered label-blind per-pass gradient second moments, fixed rank grid."""
    if grads.ndim != 3 or grads.shape[1:] != (PASSES, WIDTH) or len(grads) == 0:
        raise ValueError("expected [N,4,20] nonempty gradient array")
    x = grads.double()
    c = torch.einsum("nti,ntj->tij", x, x) / len(x)
    eigs, vectors = torch.linalg.eigh(c)
    result, diagnostics, bases = {}, {}, {}
    for r in RANKS:
        spectral = []
        for t in range(PASSES):
            lam_r = float(eigs[t, -r])
            next_lam = float(eigs[t, -r-1])
            spectral.append({
                "pass": t, "lambda_rank": lam_r, "lambda_next": next_lam,
                "gap_ratio": lam_r / max(next_lam, 1e-12),
                "top_rank_fraction": float(eigs[t, -r:].sum() / eigs[t].clamp_min(0).sum().clamp_min(1e-30)),
            })
        identifiable = all(x["lambda_rank"] > 1e-12 and x["gap_ratio"] >= 1.001 for x in spectral)
        e = vectors[:, :, -r:]
        p = e @ e.transpose(-1, -2)
        bases[r] = e
        result[r] = p if identifiable else None
        diagnostics[r] = {
            "status": "IDENTIFIABLE" if identifiable else "UNIDENTIFIABLE",
            "spectrum": spectral,
            "symmetry_max_abs": float((p-p.transpose(-1,-2)).abs().max()),
            "idempotence_max_abs": float((p @ p - p).abs().max()),
        }
    return result, diagnostics, bases


def clean_dose_rms(activations: Tensor, u: Tensor, p: Tensor) -> float:
    """Physical retained write RMS, using only clean discovery inputs."""
    if activations.ndim != 3 or activations.shape[1:] != (PASSES, WIDTH):
        raise ValueError("expected [N,4,20] activations")
    if p.shape != (PASSES, WIDTH, WIDTH):
        raise ValueError("expected one [20,20] projector per pass")
    a = activations.double()
    m = torch.einsum("nti,ntj->tij", a, a) / len(a)
    b = u.double() @ u.double().T
    dosesq = torch.einsum("tij,tjk,kl,tli->", m, p, b, p) / (PASSES * WIDTH)
    return float(dosesq.clamp_min(0).sqrt())


def local_dose_matched_shams(
    acts: Tensor, u: Tensor, eigbasis: Tensor, rank: int, seed: int,
    *, proposals: int = PROPOSALS, want: int = SHAMS,
) -> tuple[list[Tensor], dict]:
    """First eight qualifying **local** randomized physical rank-r subspaces.

    Distinct Gaussian complement directions are generated from discovery
    eigvectors. Physical dose is matched with a frozen 15% threshold. No
    evaluation labels, outcomes or post hoc nearest-neighbor selection.
    """
    if rank not in RANKS or eigbasis.shape != (PASSES, WIDTH, rank):
        raise ValueError("unregistered rank or eigenvector dimensions")
    e = eigbasis.detach().cpu().double().numpy()
    a = acts.detach().cpu().double().numpy()
    u_np = u.detach().cpu().double().numpy()
    gram_w = u_np @ u_np.T
    m = np.einsum("nti,ntj->tij", a, a) / len(a)
    def dose_from_basis(basis: np.ndarray) -> float:
        # trace(E^T M E E^T U U^T E) per pass, avoiding 20x20 projectors
        dosesq = 0.0
        for t in range(PASSES):
            e_t = basis[t]
            am = e_t.T @ m[t] @ e_t
            bw = e_t.T @ gram_w @ e_t
            dosesq += float(np.trace(am @ bw))
        return float(np.sqrt(max(0.0, dosesq) / (PASSES * WIDTH)))
    base_dose = dose_from_basis(e)
    if base_dose <= 1e-12:
        return [], {
            "eligible": False, "reason": "zero clean physical write dose",
            "total_proposals": proposals, "qualifying_candidates": 0,
        }
    g = np.random.default_rng(seed + 5_051_701 + rank * 139)
    chosen, candidates = [], 0
    seen: set[int] = set()
    for index in range(proposals):
        angle = float(ANGLES_RAD[int(g.integers(0, len(ANGLES_RAD)))])
        q = np.zeros_like(e)
        for t in range(PASSES):
            rand = g.standard_normal((WIDTH, rank))
            residual = rand - e[t] @ (e[t].T @ rand)
            complement, _ = np.linalg.qr(residual, mode="reduced")
            q[t] = np.cos(angle) * e[t] + np.sin(angle) * complement
        rms = dose_from_basis(q)
        rel = abs(rms - base_dose) / base_dose
        if rel <= MATCH_TOL:
            candidates += 1
            if len(chosen) < want and index not in seen:
                seen.add(index)
                proj = torch.from_numpy(q @ q.transpose(0, 2, 1)).double()
                chosen.append({
                    "proposal_index": index,
                    "angle_radians": angle, "relative_dose_error": rel,
                    "discovery_write_rms": rms, "projector": proj,
                })
    return [v["projector"] for v in chosen], {
        "eligible": len(chosen) == want,
        "total_proposals": proposals, "qualifying_candidates": candidates,
        "base_discovery_write_rms": base_dose,
        "sham_proposal_indices": [v["proposal_index"] for v in chosen],
        "sham_angles_radians": [v["angle_radians"] for v in chosen],
        "sham_dose_errors": [v["relative_dose_error"] for v in chosen],
        "sham_discovery_rms": [v["discovery_write_rms"] for v in chosen],
    }


@torch.no_grad()
def projected_rollout(bpd: BPDModel, model: DenseLooper, x: Tensor, p: Tensor,
                      gain: float) -> tuple[Tensor, Tensor]:
    """Native per-pass physical BPD counterfactual for alpha times rank-r P."""
    if gain not in GAINS:
        raise ValueError("gain was not registered")
    layer = bpd.wrappers["mix"]
    reads = (layer.V.detach().double() @ (gain*p)).float()
    with bpd.intervene("mix", tuple(range(WIDTH)), mode="override_read",
                       read_override=reads, scale=0.0):
        logits, _ = model(x, capture=True)
        writes = torch.stack(model.writes, dim=1)
    return logits, writes


@torch.no_grad()
def score_outputs(clean: Tensor, clean_writes: Tensor,
                  altered: Tensor, altered_writes: Tensor, y: Tensor) -> dict:
    sign = torch.where(y, 1.0, -1.0)
    true_nll = F.softplus(-2*sign*altered[:, -1])
    clean_nll = F.softplus(-2*sign*clean[:, -1])
    kl = binary_kl(clean, altered).clamp_min(0)
    right = ((altered[:, -1] >=0) == y)
    return {
        "nll_nats": float(true_nll.mean()),
        "nll_delta_from_clean_nats": float((true_nll-clean_nll).mean()),
        "correct_count": int(right.sum()),
        "accuracy": float(right.float().mean()),
        "kl_by_pass_nats": [float(a) for a in kl.mean(0)],
        "final_kl_nats": float(kl[:, -1].mean()),
        "closed_loop_write_displacement_rms": float((altered_writes-clean_writes).square().mean().sqrt()),
    }


@torch.no_grad()
def score_keep(bpd: BPDModel, model: DenseLooper, x: Tensor, y: Tensor,
               clean: Tensor, clean_write: Tensor, p: Tensor, gain: float) -> dict:
    logits, writes = projected_rollout(bpd, model, x, p, gain)
    return score_outputs(clean, clean_write, logits, writes, y)


def evaluate_root(seed: int, *, steps: int = 384, discovery_size: int = 1024,
                  evaluation_size: int = 1024,
                  proposals: int = PROPOSALS, shams: int = SHAMS) -> dict:
    if seed not in ROOTS:
        raise ValueError("NC-05 uses only new development roots 500..507")
    torch.set_num_threads(1)
    model, training = train_model(seed, steps=steps)
    x_discovery = sample_x(seed, 2, discovery_size)
    x_evaluation = sample_x(seed, 3, evaluation_size)
    y_evaluation = external_labels(x_evaluation)
    contexts = context_accuracies(model, x_evaluation, y_evaluation)
    bpd = BPDModel(model, {"mix": WIDTH}, max_passes=PASSES,
                   ablation=AblationConfig(mode="zero", sphere_rescale=False))
    result = {
        "study": "BPD-NC-05", "phase": "ENGINEERING", "seed": seed,
        "protocol_commit": PROTOCOL_COMMIT,
        "data": {"training": 2048, "discovery": discovery_size,
                 "evaluation": evaluation_size},
        "training": training, "contexts": contexts,
        "task_passed": all(val >= .85 for val in contexts.values())
                    and training["mix_weight_density"] >= .90,
        "bases": {}, "cells": {}, "negative_controls": {},
    }
    clean, clean_writes = None, None
    native_proj: dict[int, Tensor | None] = {}
    native_eigenvectors: dict[int, Tensor] = {}
    native_shams: dict[int, list[Tensor]] = {}
    native_acts: Tensor | None = None
    for basis in ("native", "rotated"):
        factor_basis(bpd, basis, seed)
        layer = bpd.wrappers["mix"]
        closure = parity(bpd, model, x_evaluation[:min(64,len(x_evaluation))])
        arm = {"closure": closure, "rank_diagnostics": {}}
        result["bases"][basis] = arm
        if not closure["passed"]:
            arm["status"] = "BLOCKED_PARITY"
            continue
        acts, gradients, _, info = self_score_discovery(bpd, model, x_discovery)
        projectors, diagnostics, eigenvectors = projectors_at_ranks(gradients)
        arm["score"] = info
        arm["rank_diagnostics"] = {str(r): diagnostics[r] for r in RANKS}
        if basis == "native":
            native_proj = projectors
            native_eigenvectors = eigenvectors
            native_acts = acts
            with torch.no_grad():
                with bpd.target_capture():
                    clean, _ = model(x_evaluation, capture=True)
                    clean_writes = torch.stack(model.writes, dim=1)
            result["clean"] = {
                "external_label_nll_nats": float(
                    F.softplus(-2*torch.where(y_evaluation,1.,-1.)*clean[:, -1]).mean()
                ),
                "accuracy_by_pass": [
                    float(((clean[:,t]>=0)==y_evaluation).float().mean()) for t in range(PASSES)
                ],
                "correct_count": int(((clean[:,-1]>=0)==y_evaluation).sum()),
            }
        for rank in RANKS:
            key=str(rank)
            if key not in result["cells"]:
                result["cells"][key] = {"rank": rank, "gains": {}, "sham_selection": None}
            cell = result["cells"][key]
            p = projectors[rank]
            if p is None:
                for alpha in GAINS:
                    cell["gains"].setdefault(str(alpha), {})[basis] = {
                        "status": "UNIDENTIFIABLE"
                    }
                continue
            if basis == "native":
                if native_acts is None:
                    raise RuntimeError("native discovery missing")
                control, matching = local_dose_matched_shams(
                    native_acts, layer.U.detach(), native_eigenvectors[rank],
                    rank, seed, proposals=proposals, want=shams
                )
                native_shams[rank] = control
                cell["sham_selection"] = matching
            else:
                if native_proj.get(rank) is not None:
                    np_native = native_proj[rank]
                    physical = layer.V.detach().double() @ p @ layer.U.detach().double()
                    w_native = bpd.wrappers["mix"].W_target.double().T
                    physical_ref = np_native @ w_native
                    arm["rank_diagnostics"][str(rank)]["rotation_comparison"] = {
                        "operator_relative_error": float(
                            (physical-physical_ref).norm(dim=(1,2)).max()
                            / layer.W_target.double().norm().clamp_min(1e-12)),
                        "projector_transport_max_abs": float(
                            (p-basis_transport(np_native, layer.V.detach())).abs().max()),
                    }
            controls = native_shams.get(rank, [])
            if len(controls) != shams:
                for alpha in GAINS:
                    cell["gains"].setdefault(str(alpha), {})[basis] = {
                        "status": "BLOCKED_SHAMS"
                    }
                continue
            if clean is None or clean_writes is None:
                raise RuntimeError("native clean model reference absent")
            for alpha in GAINS:
                ag=str(alpha)
                selection = score_keep(bpd, model, x_evaluation, y_evaluation,
                                       clean, clean_writes, p, alpha)
                matched = [
                    score_keep(
                        bpd, model, x_evaluation, y_evaluation,
                        clean, clean_writes,
                        sh if basis=="native" else basis_transport(sh,layer.V.detach()),
                        alpha
                    ) for sh in controls
                ]
                sham_nll=[c["nll_nats"] for c in matched]
                item = {
                    "status": "ENGINEERING",
                    "retained": selection,
                    "shams": matched,
                    "sham_external_nll_median": float(np.median(sham_nll)),
                    "sham_adjusted_external_nll_nats": float(np.median(sham_nll) - selection["nll_nats"]),
                    "gain": alpha,
                    "selected_discovery_write_rms": alpha * cell["sham_selection"]["base_discovery_write_rms"],
                }
                if basis=="rotated" and "native" in cell["gains"].get(ag,{}):
                    native_entry=cell["gains"][ag]["native"]
                    # Paired target outputs on identical untouched held-out inputs.
                    factor_basis(bpd, "native", seed)
                    native_kept, _ = projected_rollout(
                        bpd, model, x_evaluation, native_proj[rank], alpha
                    )
                    factor_basis(bpd, "rotated", seed)
                    rotated_kept, _ = projected_rollout(
                        bpd, model, x_evaluation, p, alpha
                    )
                    item["rotation_max_kept_logit_delta"] = float(
                        (native_kept-rotated_kept).abs().max()
                    )
                    item["paired_native_nll_delta"] = float(
                        selection["nll_nats"] - native_entry["retained"]["nll_nats"]
                    )
                cell["gains"].setdefault(ag,{})[basis] = item
        arm["status"] = "ENGINEERING"
    # Zero-head readout, discovery-only. No true-label oracle in this path.
    factor_basis(bpd, "native", seed)
    with torch.no_grad():
        head_weight=model.head.weight.detach().clone()
        head_bias=model.head.bias.detach().clone()
        model.head.weight.zero_()
        model.head.bias.zero_()
    try:
        _, zero_grad, hits, _ = self_score_discovery(
            bpd, model, x_discovery[:min(256,len(x_discovery))]
        )
        zero, zero_diag, _ = projectors_at_ranks(zero_grad)
        result["negative_controls"]["zero_head"] = {
            "max_abs_grad": float(zero_grad.abs().max()),
            "positive_hits": int(hits.sum()),
            "all_ranks_unidentifiable": all(p is None for p in zero.values()),
            "statuses": {str(r): zero_diag[r]["status"] for r in RANKS},
        }
    finally:
        with torch.no_grad():
            model.head.weight.copy_(head_weight)
            model.head.bias.copy_(head_bias)
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["protocol_sha256"] = hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest()
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=500)
    parser.add_argument("--steps",type=int,default=384)
    parser.add_argument("--discovery-size",type=int,default=1024)
    parser.add_argument("--evaluation-size",type=int,default=1024)
    parser.add_argument("--proposals",type=int,default=PROPOSALS)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    result=evaluate_root(args.seed,steps=args.steps,
                         discovery_size=args.discovery_size,
                         evaluation_size=args.evaluation_size,
                         proposals=args.proposals)
    output=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if args.out is None:
        print(output)
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as f:
            f.write(output)


if __name__=="__main__":
    main()
