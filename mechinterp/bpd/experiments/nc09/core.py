"""NC-09: trained multitask recurrent model and oracle-blind projector mechanics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import sample_x

ROOTS = tuple(range(900, 908))
EXCLUDED_ENGINEERING_ROOT = 899
PROTOCOL_COMMIT = "22e00fc22cdd5e3403d936294232bcfc051aa2e2"
PASSES = 4
WIDTH = 32
RANK = 4
SHAMS = 8
PROPOSALS = 2048
ANGLES = (0.30, 0.50, 0.70, 0.90)
TASK_MARKERS = {"A": -1.0, "B": +1.0}


def sample_base(seed: int, split: int, size: int) -> Tensor:
    """Five independent continuous features on a root- and split-specific stream."""
    return sample_x(seed, split, size)


def mark_task(base: Tensor, task: str) -> Tensor:
    if task not in TASK_MARKERS or base.ndim != 2 or base.shape[-1] != 5:
        raise ValueError("expected task A or B on [N,5] base features")
    marker = base.new_full((len(base), 1), TASK_MARKERS[task])
    return torch.cat((base, marker), dim=-1)


def task_labels(base: Tensor, task: str) -> Tensor:
    """External input-defined oracle. Never call from projector or sham discovery."""
    if task not in TASK_MARKERS:
        raise ValueError("task must be A or B")
    target_else = 1 if task == "A" else 2
    return torch.where(base[:, 4] > 0, base[:, 0], base[:, target_else]) > 0


def context_masks(base: Tensor) -> dict[str, Tensor]:
    return {"shared": base[:, 4] > 0, "private": base[:, 4] <= 0}


class JointLooper(nn.Module):
    """32-dimensional freely trained tied nonlinear recurrence, 2 task markers."""

    def __init__(self) -> None:
        super().__init__()
        self.encode = nn.Linear(6, WIDTH)
        self.pre = nn.Linear(WIDTH, WIDTH)
        self.mix = nn.Linear(WIDTH, WIDTH, bias=False)
        self.head = nn.Linear(WIDTH, 1)
        self.acts: list[Tensor] = []
        self.writes: list[Tensor] = []

    def forward(self, x: Tensor, *, capture: bool = False) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or x.shape[-1] != 6:
            raise ValueError("input must be [batch,6]")
        hidden = torch.tanh(self.encode(x))
        logits, states = [], []
        self.acts = []
        self.writes = []
        for _ in range(PASSES):
            a = torch.tanh(self.pre(hidden))
            write = self.mix(a.unsqueeze(1)).squeeze(1)
            if capture:
                self.acts.append(a)
                self.writes.append(write)
            hidden = torch.tanh(0.60 * hidden + 0.65 * write)
            logits.append(self.head(hidden).squeeze(-1))
            states.append(hidden)
        return torch.stack(logits, dim=1), torch.stack(states, dim=1)


def train_model(seed: int, *, steps: int = 640) -> tuple[JointLooper, dict]:
    if seed not in ROOTS and seed != EXCLUDED_ENGINEERING_ROOT:
        raise ValueError("unregistered training seed")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = JointLooper()
    base = sample_base(seed, 1, 2048)
    xa, xb = mark_task(base, "A"), mark_task(base, "B")
    ya = torch.where(task_labels(base, "A"), 1.0, -1.0)
    yb = torch.where(task_labels(base, "B"), 1.0, -1.0)
    generator = torch.Generator().manual_seed(seed + 1_991_017)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(steps):
        ia = torch.randint(len(base), (64,), generator=generator)
        ib = torch.randint(len(base), (64,), generator=generator)
        x = torch.cat((xa[ia], xb[ib]))
        y = torch.cat((ya[ia], yb[ib]))
        prediction = model(x)[0][:, -1]
        loss = F.softplus(-2 * y * prediction).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred_a = model(xa)[0][:, -1]
        pred_b = model(xb)[0][:, -1]
        weight = model.mix.weight.detach()
    report = {
        "seed": seed, "steps": steps, "base_training_cases": 2048,
        "task_marked_training_cases": 4096,
        "learning_rate": 0.005, "batch_per_task": 64,
        "last_minibatch_nll_nats": float(loss.detach()),
        "training_accuracy_A": float(((pred_a >= 0) == task_labels(base, "A")).float().mean()),
        "training_accuracy_B": float(((pred_b >= 0) == task_labels(base, "B")).float().mean()),
        "mix_weight_density": float((weight.abs() >= 1e-4).float().mean()),
        "n_trainable_parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, report


def wrap_bpd(model: JointLooper) -> BPDModel:
    return BPDModel(
        model, {"mix": WIDTH}, max_passes=PASSES,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=False,
    )


def set_basis(bpd: BPDModel, seed: int, kind: str) -> None:
    layer = bpd.wrappers["mix"]
    weight = layer.W_target.detach()
    if kind == "native":
        v = torch.eye(WIDTH, dtype=torch.float32)
    elif kind == "rotated":
        gen = torch.Generator().manual_seed(seed + 33_093)
        q, r = torch.linalg.qr(
            torch.randn(WIDTH, WIDTH, generator=gen, dtype=torch.float64)
        )
        v = (q * torch.where(torch.diag(r) >= 0, 1., -1.).unsqueeze(0)).float()
    else:
        raise ValueError("unregistered factor basis")
    u = v.T @ weight.T
    if float(((v @ u).T - weight).norm() / weight.norm().clamp_min(1e-12)) > 1e-5:
        raise RuntimeError("basis does not reconstruct target physical weight")
    with torch.no_grad():
        layer.V.copy_(v)
        layer.U.copy_(u)


def bernoulli_kl(clean: Tensor, altered: Tensor) -> Tensor:
    p2 = torch.stack((-clean.double(), clean.double()), dim=-1)
    q2 = torch.stack((-altered.double(), altered.double()), dim=-1)
    lp = F.log_softmax(p2, dim=-1)
    lq = F.log_softmax(q2, dim=-1)
    return (lp.exp() * (lp - lq)).sum(-1).clamp_min(0.)


@torch.no_grad()
def full_mask_parity(bpd: BPDModel, model: JointLooper, x: Tensor) -> dict:
    layer = bpd.wrappers["mix"]
    comp = PassMaskSchedule(
        comp_mask=torch.ones(len(x), PASSES, 1, WIDTH),
        delta_mask=torch.zeros(len(x), PASSES, 1),
    )
    with bpd.target_capture():
        clean, _ = model(x)
    with bpd.masked_components({"mix": comp}):
        restored, _ = model(x)
    residual = float(layer.weight_delta().norm() / layer.W_target.norm().clamp_min(1e-12))
    err = float((clean-restored).abs().max())
    kl = float(bernoulli_kl(clean, restored).max())
    return {
        "passed": residual <= 1e-4 and err <= 5e-5 and kl <= 1e-7,
        "relative_weight_residual": residual,
        "maximum_logit_error": err,
        "maximum_kl_nats": kl,
    }


def discover(bpd: BPDModel, model: JointLooper, x: Tensor, *, batch_size: int = 128
             ) -> tuple[Tensor, Tensor]:
    """Capture passwise factor activations and self-logit gradients without labels."""
    if x.ndim != 2 or x.shape[-1] != 6:
        raise ValueError("discovery accepts only task-marked input")
    layer = bpd.wrappers["mix"]
    all_acts, all_grad = [], []
    for i in range(0, len(x), batch_size):
        xx = x[i:i+batch_size].clone().requires_grad_(True)
        with bpd.target_capture():
            logits, _ = model(xx, capture=True)
            sign = torch.where(logits[:, -1].detach() >= 0., 1., -1.)
            grad_writes = torch.autograd.grad(
                (sign*logits[:, -1]).sum(), model.writes
            )
        activations = torch.stack(model.acts, dim=1).detach()
        grads = torch.stack(grad_writes, dim=1).detach()
        all_acts.append((activations @ layer.V.detach()).double())
        all_grad.append((grads @ layer.U.detach().T).double())
    return torch.cat(all_acts), torch.cat(all_grad)


def projector_from_gradients(grads: Tensor) -> tuple[Tensor | None, Tensor, dict]:
    if grads.ndim != 3 or grads.shape[1:] != (PASSES, WIDTH) or len(grads) < 1:
        raise ValueError("gradient tensor must be [N,4,32]")
    g = grads.double()
    cov = torch.einsum("nti,ntj->tij", g, g) / len(g)
    values, vectors = torch.linalg.eigh(cov)
    diagnostics = []
    for t in range(PASSES):
        r4 = float(values[t, -RANK])
        r5 = float(values[t, -(RANK+1)])
        diagnostics.append({
            "pass": t, "lambda4": r4, "lambda5": r5,
            "gap_ratio": r4/max(1e-12,r5),
        })
    e = vectors[:, :, -RANK:]
    projector = e @ e.transpose(1,2)
    meets = all(item["lambda4"] > 1e-12 and item["gap_ratio"] >= 1.001
                for item in diagnostics)
    info = {
        "status": "IDENTIFIABLE" if meets else "UNIDENTIFIABLE",
        "per_pass": diagnostics,
        "symmetry_error": float((projector-projector.transpose(1,2)).abs().max()),
        "idempotence_error": float((projector @ projector-projector).abs().max()),
    }
    return (projector if meets else None), e, info


def physical_write_rms(acts: Tensor, u: Tensor, p: Tensor) -> float:
    if acts.ndim != 3 or acts.shape[1:] != (PASSES, WIDTH):
        raise ValueError("invalid physical-dose activations")
    m = torch.einsum("nti,ntj->tij", acts.double(), acts.double()) / len(acts)
    b = u.double() @ u.double().T
    val = torch.einsum("tij,tjk,kl,tli->", m, p.double(), b, p.double())
    return float((val/(PASSES*WIDTH)).clamp_min(0).sqrt())


def source_doses(acts_b: Tensor, continuous_b: Tensor, u: Tensor,
                 p_a: Tensor, p_b: Tensor) -> dict:
    masks = context_masks(continuous_b)
    out = {}
    for k, mask in (("all", torch.ones(len(acts_b), dtype=torch.bool)),
                    ("shared", masks["shared"]), ("private", masks["private"])):
        if mask.sum() == 0:
            raise ValueError("empty discovery context subset")
        dose_a = physical_write_rms(acts_b[mask], u, p_a)
        dose_b = physical_write_rms(acts_b[mask], u, p_b)
        out[k] = {"A": dose_a, "B": dose_b}
    a, b = out["all"]["A"], out["all"]["B"]
    gain = b/max(a,1e-12)
    out["transfer_gain"] = gain
    out["gain_admissible"] = .75 <= gain <= 1.25
    out["gain_adjusted_relative_mismatch"] = {
        k: abs(gain*out[k]["A"]-out[k]["B"])/max(out[k]["B"],1e-12)
        for k in ("all","shared","private")
    }
    out["branch_dose_admissible"] = (
        out["gain_adjusted_relative_mismatch"]["shared"] <= .25
        and out["gain_adjusted_relative_mismatch"]["private"] <= .40
    )
    return out


def local_matched_shams(
    activations_b: Tensor, continuous_b: Tensor, u_native: Tensor,
    eigenvectors_a: Tensor, seed: int, *, n_proposals: int = PROPOSALS
) -> tuple[list[Tensor],dict]:
    """Discovery-only local angular controls, matched on target-B write RMS."""
    if eigenvectors_a.shape != (PASSES, WIDTH, RANK):
        raise ValueError("wrong A eigenvectors")
    masks = context_masks(continuous_b)
    a = activations_b.double().cpu().numpy()
    e = eigenvectors_a.double().cpu().numpy()
    u = u_native.double().cpu().numpy()
    wgram = u @ u.T
    moment = {
        "all": np.einsum("nti,ntj->tij", a, a) / len(a),
        "shared": np.einsum("nti,ntj->tij", a[masks["shared"].numpy()], a[masks["shared"].numpy()])
                  / int(masks["shared"].sum()),
        "private": np.einsum("nti,ntj->tij", a[masks["private"].numpy()], a[masks["private"].numpy()])
                   / int(masks["private"].sum()),
    }
    def dose(basis: np.ndarray, name: str) -> float:
        value = 0.
        for t in range(PASSES):
            p = basis[t]
            m = p.T @ moment[name][t] @ p
            b = p.T @ wgram @ p
            value += float(np.trace(m @ b))
        return np.sqrt(max(value,0.)/(PASSES*WIDTH))
    reference = {key: dose(e,key) for key in ("all","shared","private")}
    if any(v <= 1e-12 for v in reference.values()):
        return [],{"eligible":False,"reason":"zero source discovery write","qualifying":0}
    gen = np.random.default_rng(seed + 9_009_013)
    selected, eligible_count = [], 0
    for index in range(n_proposals):
        angle = float(ANGLES[int(gen.integers(0,len(ANGLES)))])
        basis = np.empty_like(e)
        for t in range(PASSES):
            noise = gen.standard_normal((WIDTH,RANK))
            complement = noise - e[t] @ (e[t].T @ noise)
            f, _ = np.linalg.qr(complement, mode="reduced")
            basis[t] = np.cos(angle)*e[t] + np.sin(angle)*f
        doses = {key:dose(basis,key) for key in reference}
        relative = {key:abs(doses[key]-reference[key])/reference[key]
                    for key in reference}
        if (relative["all"] <= .15 and relative["shared"] <=.20
                and relative["private"] <=.20):
            eligible_count += 1
            if len(selected) < SHAMS:
                pp = torch.from_numpy(basis @ np.swapaxes(basis,1,2)).double()
                selected.append((pp,{
                    "proposal_index":index,
                    "angle_rad":angle, "rms_doses":doses, "relative_errors":relative,
                }))
    return [x[0] for x in selected],{
        "eligible":len(selected)==SHAMS,
        "candidate_budget":n_proposals,
        "qualifying":eligible_count,
        "reference_dose":reference,
        "selected":[x[1] for x in selected],
    }


@torch.no_grad()
def run_projector(bpd: BPDModel, model: JointLooper, x: Tensor, projector: Tensor,
                  gain: float = 1.) -> tuple[Tensor, Tensor]:
    layer = bpd.wrappers["mix"]
    if projector.shape != (PASSES,WIDTH,WIDTH) or gain <=0 or not np.isfinite(gain):
        raise ValueError("invalid closed-loop projector/gain")
    reads = (layer.V.detach().double() @ (gain*projector.double())).float()
    with bpd.intervene("mix",tuple(range(WIDTH)),mode="override_read",
                       read_override=reads,scale=0.):
        logits,_ = model(x,capture=True)
        writes = torch.stack(model.writes,dim=1)
    return logits,writes


def transport_to_rotated(p: Tensor, v: Tensor) -> Tensor:
    return v.double().T @ p.double() @ v.double()
