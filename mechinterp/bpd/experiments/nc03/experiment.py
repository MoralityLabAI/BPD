"""NC-03: gauge-covariant gradient subspaces of a planted trained recurrence.

Engineering-only; see PROTOCOL.md at the prior source-only commit. This
estimator uses gradient second moments and rank-two *projectors*, distinct
from Goodfire neuron-community co-attribution graphs.
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
    PASSES, LANES, WIDTH, PlantedLooper, train_model, sample_x, external_labels,
    make_basis, binary_kl, parity,
)

ROOTS = tuple(range(300, 308))
PROTOCOL_COMMIT = "a48871f2612f9032ddbf612747f972f701585536"
RANK = 2
ACTIVE_RANK = 4
SHAMS = 8
PROPOSALS = 2048
DOSE_TOL = 0.15


def configure_basis(bpd: BPDModel, model: PlantedLooper, kind: str, seed: int) -> None:
    v, u = make_basis(bpd.wrappers["mix"].W_target, kind, seed)
    with torch.no_grad():
        bpd.wrappers["mix"].V.copy_(v)
        bpd.wrappers["mix"].U.copy_(u)


def activation_and_gradients(bpd: BPDModel, model: PlantedLooper,
                             x: Tensor, y: Tensor, bs: int = 128) -> tuple[Tensor, Tensor]:
    """Return factor activations and d(signed final-logit)/dz, each [N,T,C]."""
    a_rows, g_rows = [], []
    layer = bpd.wrappers["mix"]
    for offset in range(0, len(x), bs):
        # BPDModel freezes the target weights; the input must keep a graph
        # to obtain gradients at intermediate shared physical write sites.
        xx = x[offset:offset+bs].clone().requires_grad_(True)
        yy = y[offset:offset+bs]
        with bpd.target_capture():
            scores, _ = model(xx, capture=True)
            sign = torch.where(yy, 1., -1.)
            grad_w = torch.autograd.grad((sign * scores[:, -1]).sum(), model.writes)
        act = torch.stack(model.acts, 1).detach()
        grad = torch.stack(grad_w, 1).detach()
        a_rows.append(act @ layer.V.detach())
        g_rows.append(grad @ layer.U.detach().T)
    return torch.cat(a_rows).double(), torch.cat(g_rows).double()


def covariant_projectors(grads: Tensor) -> tuple[Tensor | None, dict]:
    """Top-2 eigenspace of uncentered per-pass factor-gradient moments."""
    if grads.ndim != 3 or grads.shape[1:] != (PASSES, WIDTH):
        raise ValueError("gradients must be [sample, pass, width]")
    if grads.shape[0] < 1:
        raise ValueError("empty gradients")
    moment = torch.einsum("nti,ntj->tij", grads, grads) / len(grads)
    eigvals, eigvecs = torch.linalg.eigh(moment)
    spectral = []
    for t in range(PASSES):
        two = float(eigvals[t, -RANK])
        third = max(float(eigvals[t, -(RANK+1)]), 0.0)
        ratio = two / max(third, 1e-12)
        spectral.append({"pass": t, "second_eigenvalue": two,
                         "third_eigenvalue": third, "gap_ratio": ratio})
    if any(r["second_eigenvalue"] <= 1e-12 or r["gap_ratio"] < 100 for r in spectral):
        return None, {"status": "NO_IDENTIFIABLE_SUBSPACE", "spectrum": spectral}
    e = eigvecs[:, :, -RANK:]  # [T,C,2]
    p = e @ e.transpose(-1, -2)
    eye = torch.eye(WIDTH, dtype=p.dtype)
    error_sym = float((p - p.transpose(-1, -2)).abs().max())
    error_idem = float((p @ p - p).abs().max())
    result = {"status": "RANK_TWO_IDENTIFIABLE", "spectrum": spectral,
              "symmetry_max_error": error_sym, "idempotence_max_error": error_idem}
    return p, result


def physical_truth_projectors() -> Tensor:
    """Retrospective oracle only; never used in discovery or sham selection."""
    result = torch.zeros(PASSES, WIDTH, WIDTH, dtype=torch.float64)
    for t in range(PASSES):
        for lane in (0, 1):
            i = t * LANES + lane
            result[t, i, i] = 1.
    return result


def oracle_diagnostics(projectors: Tensor, v: Tensor) -> dict:
    true = physical_truth_projectors()
    v = v.double()
    basis = v.transpose(-1, -2)
    transformed = torch.stack([basis @ true[t] @ v for t in range(PASSES)])
    overlap = torch.einsum("tij,tji->t", projectors, transformed) / RANK
    angle = []
    for t in range(PASSES):
        gt = torch.linalg.eigh(transformed[t])[1][:, -RANK:]
        pred = torch.linalg.eigh(projectors[t])[1][:, -RANK:]
        cosine = torch.linalg.svdvals(gt.T @ pred).clamp(0.0, 1.0)
        angle.append(float(torch.rad2deg(torch.acos(cosine).max())))
    return {"overlap_by_pass": [float(v) for v in overlap],
            "min_overlap": float(overlap.min()),
            "max_principal_angle_degrees": max(angle)}


@torch.no_grad()
def masked_rollout(bpd: BPDModel, model: PlantedLooper, x: Tensor,
                   projectors: Tensor, *, mode: str) -> tuple[Tensor, Tensor]:
    """Exact native BPD override-read path; P or I-P on all tied components."""
    if mode not in ("retain", "ablate"):
        raise ValueError("mode must be retain or ablate")
    p = projectors
    if mode == "ablate":
        p = torch.eye(WIDTH, dtype=p.dtype).unsqueeze(0) - p
    layer = bpd.wrappers["mix"]
    reads = torch.matmul(layer.V.detach().double(), p).float()
    with bpd.intervene("mix", tuple(range(WIDTH)), mode="override_read",
                       read_override=reads, passes=None, scale=0.0):
        logits, _ = model(x, capture=True)
        writes = torch.stack(model.writes, dim=1)
    return logits, writes


@torch.no_grad()
def assess(bpd: BPDModel, model: PlantedLooper, x: Tensor,
           y: Tensor, projectors: Tensor) -> dict:
    with bpd.target_capture():
        clean, _ = model(x, capture=True)
        original_writes = torch.stack(model.writes, dim=1)
    sign = torch.where(y, 1., -1.)
    clean_nll = F.softplus(-2 * sign * clean[:, -1])
    out = {"clean_accuracy": float(((clean[:, -1] >= 0) == y).float().mean()),
           "arms": {}}
    for mode in ("retain", "ablate"):
        logits, writes = masked_rollout(bpd, model, x, projectors, mode=mode)
        kl = binary_kl(clean, logits).clamp_min(0.)
        nll = F.softplus(-2 * sign * logits[:, -1])
        out["arms"][mode] = {
            "nll_delta_nats": float((nll - clean_nll).mean()),
            "kl_by_pass_nats": [float(q) for q in kl.mean(0)],
            "final_kl_nats": float(kl[:, -1].mean()),
            "accuracy": float(((logits[:, -1] >= 0) == y).float().mean()),
            "closed_loop_write_delta_rms": float((writes - original_writes).square().mean().sqrt()),
        }
    return out


def clean_write_dose(moment: Tensor, b: Tensor, p: Tensor) -> float:
    """Retained clean-context physical write RMS for any orthogonal projector."""
    if p.shape != (PASSES, WIDTH, WIDTH):
        raise ValueError("projector shape must be [T,C,C]")
    tmp = moment @ p @ b @ p
    return float((torch.diagonal(tmp, dim1=-2, dim2=-1).sum() / (PASSES * WIDTH))
                 .clamp_min(0.).sqrt())


def discovery_moments(acts: Tensor, v: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
    """Compute input-moment and physical-write-Gram matrices in factor coordinates."""
    if acts.ndim != 3 or acts.shape[1:] != (PASSES, WIDTH):
        raise ValueError("activation dimension mismatch")
    moments = torch.einsum("nti,ntj->tij", acts, acts) / len(acts)
    b = u.double() @ u.double().T
    return moments, b


def sham_projectors(acts_native: Tensor, u_native: Tensor, reference_p: Tensor, seed: int,
                    proposals: int = PROPOSALS, want: int = SHAMS) -> tuple[list[Tensor], dict]:
    """Oracle-free discovery-dose-matched Haar projectors inside active rank-4 states.

    Candidate projectors are physical/native, and transported unchanged by Q
    for every rotated-basis evaluation. Selection uses only discovery inputs.
    """
    moments, b = discovery_moments(acts_native, torch.eye(WIDTH), u_native)
    reference_dose = clean_write_dose(moments, b, reference_p)
    if reference_dose <= 1e-12:
        return [], {"eligible": False, "reason": "zero discovery write dose", "qualified": 0}
    active = torch.linalg.eigh(moments)[1][:, :, -ACTIVE_RANK:]
    reduced_m = active.transpose(-1, -2) @ moments @ active
    reduced_b = active.transpose(-1, -2) @ b @ active
    g = np.random.default_rng(seed + 1_000_033)
    candidates = []
    qualifies = 0
    for index in range(proposals):
        small = []
        p_full = []
        for t in range(PASSES):
            rand = torch.tensor(g.standard_normal((ACTIVE_RANK, RANK)), dtype=torch.float64)
            q, r = torch.linalg.qr(rand)
            q = q * torch.where(torch.diag(r) >= 0, 1., -1.).unsqueeze(0)
            small_p = q @ q.T
            small.append(small_p)
            p_full.append(active[t] @ small_p @ active[t].T)
        dose_sq = sum(float(torch.trace(reduced_m[t] @ small[t] @ reduced_b[t] @ small[t]))
                      for t in range(PASSES)) / (PASSES * WIDTH)
        dose = float(max(dose_sq, 0.0) ** .5)
        rel = abs(dose - reference_dose) / reference_dose
        if rel <= DOSE_TOL:
            qualifies += 1
            candidates.append((rel, index, torch.stack(p_full), dose))
    candidates.sort(key=lambda row: (row[0], row[1]))
    chosen = candidates[:want]
    return [row[2] for row in chosen], {
        "eligible": len(chosen) == want,
        "n_proposals": proposals,
        "eligible_candidates": qualifies,
        "selected_candidate_indices": [int(row[1]) for row in chosen],
        "reference_retained_discovery_write_rms": reference_dose,
        "matched_doses": [float(row[3]) for row in chosen],
        "matched_relative_errors": [float(row[0]) for row in chosen],
    }


def basis_transport(p_native: Tensor, v: Tensor) -> Tensor:
    """Native physical-input projector -> factor coordinates for orthogonal V."""
    return v.double().T @ p_native @ v.double()


def evaluate_root(seed: int, *, steps: int = 128, discovery_size: int = 1024,
                  evaluation_size: int = 1024, shams: int = SHAMS,
                  proposals: int = PROPOSALS) -> dict:
    if seed not in ROOTS:
        raise ValueError("NC-03 permits engineering roots 300..307 only")
    torch.set_num_threads(1)
    model, training = train_model(seed, steps=steps)
    xd, xe = sample_x(seed, 2, discovery_size), sample_x(seed, 3, evaluation_size)
    yd, ye = external_labels(xd), external_labels(xe)
    bpd = BPDModel(model, {"mix": WIDTH}, max_passes=PASSES,
                   ablation=AblationConfig(mode="zero", sphere_rescale=False))
    rec = {"study": "BPD-NC-03", "phase": "ENGINEERING", "seed": seed,
           "protocol_commit": PROTOCOL_COMMIT,
           "data": {"training": 2048, "discovery": discovery_size, "evaluation": evaluation_size},
           "training": training, "bases": {}, "nulls": {}}
    model.eval()
    baseline = None
    shams_native = None
    for kind in ("native", "rotated"):
        configure_basis(bpd, model, kind, seed)
        layer = bpd.wrappers["mix"]
        with torch.no_grad():
            close = parity(bpd, model, xe[:min(64, len(xe))])
        status = {"closure": close}
        rec["bases"][kind] = status
        if not close["passed"]:
            status["status"] = "BLOCKED_PARITY"
            continue
        activations, grads = activation_and_gradients(bpd, model, xd, yd)
        p, eigdiag = covariant_projectors(grads)
        status["gradient_spectrum"] = eigdiag
        if p is None:
            status["status"] = "NO_IDENTIFIABLE_SUBSPACE"
            continue
        status["oracle"] = oracle_diagnostics(p, layer.V.detach())
        native_map = torch.matmul(layer.V.detach().double(), p) @ layer.U.detach().double()
        status["physical_map_frobenius_norm_by_pass"] = [
            float(v) for v in native_map.norm(dim=(1,2))
        ]
        status["evaluation"] = assess(bpd, model, xe, ye, p)
        with torch.no_grad():
            full_logits, _ = model(xe)
            retained, _ = masked_rollout(bpd, model, xe, p, mode="retain")
        status["full_to_retain_logit_delta_max"] = float((retained - full_logits).abs().max())
        if kind == "native":
            baseline = {"p": p, "map": native_map, "retained": retained}
            shams_native, matching = sham_projectors(
                activations, layer.U.detach(), p, seed,
                proposals=proposals, want=shams
            )
            status["sham_matching"] = matching
        else:
            if baseline:
                status["comparison_to_native"] = {
                    "projector_map_relative_error": float(
                        (native_map - baseline["map"]).norm(dim=(1,2)).max()
                        / layer.W_target.detach().double().norm()
                    ),
                    "retain_logit_max_delta": float((retained - baseline["retained"]).abs().max()),
                    "factor_projector_transport_error": float(
                        (p - basis_transport(baseline["p"], layer.V.detach())).abs().max()
                    ),
                }
        if shams_native is not None and len(shams_native) == shams:
            # Always evaluate the same matched *physical* projector masks in both bases.
            sham_scores = [assess(bpd, model, xe, ye,
                           candidate if kind == "native"
                           else basis_transport(candidate, layer.V.detach()))
                           for candidate in shams_native]
            status["sham_keep_final_kl"] = [
                float(s["arms"]["retain"]["final_kl_nats"]) for s in sham_scores
            ]
            status["primary_contrast_nats"] = float(np.median(status["sham_keep_final_kl"])
                       - status["evaluation"]["arms"]["retain"]["final_kl_nats"])
            status["sham_nec_nll_deltas"] = [s["arms"]["ablate"]["nll_delta_nats"]
                                              for s in sham_scores]
        status["status"] = ("ENGINEERING" if "primary_contrast_nats" in status
                            else "BLOCKED_INSUFFICIENT_SHAMS")
    # Zero-readout negative control, examined in the discovery population only.
    configure_basis(bpd, model, "native", seed)
    with torch.no_grad():
        backup = model.readout.detach().clone()
        model.readout.zero_()
    try:
        _, null_grads = activation_and_gradients(bpd, model, xd[:min(256, len(xd))],
                                                  yd[:min(256, len(yd))])
        null_p, null_info = covariant_projectors(null_grads)
        rec["nulls"]["zero_readout"] = {
            "status": null_info["status"],
            "gradient_max_abs": float(null_grads.abs().max()),
            "projector_returned": null_p is not None
        }
    finally:
        with torch.no_grad():
            model.readout.copy_(backup)
    with torch.no_grad():
        pred, _ = model(xe)
        acc={}
        for lane, valid in ((0, xe[:,LANES] > 0), (1, xe[:,LANES] <= 0)):
            acc[str(lane)] = float(((pred[valid,-1] >= 0) == ye[valid]).float().mean())
        rec["heldout_context_accuracy"] = acc
    rec["task_passed"] = training["structure"]["passed"] and all(v > .95 for v in acc.values())
    rec["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rec["protocol_sha256"] = hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest()
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--discovery-size", type=int, default=1024)
    parser.add_argument("--evaluation-size", type=int, default=1024)
    parser.add_argument("--proposals", type=int, default=2048)
    parser.add_argument("--out", type=Path)
    a = parser.parse_args()
    result = evaluate_root(a.seed, steps=a.steps, discovery_size=a.discovery_size,
                           evaluation_size=a.evaluation_size, proposals=a.proposals)
    data = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("x", encoding="utf-8") as dest:
            dest.write(data)
    else:
        print(data)

if __name__ == "__main__":
    main()
