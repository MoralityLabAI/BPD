"""SUB-04: frozen eight-root multipass finite-dose causal quotient engineering.

Registered PROTOCOL.md precedes this source. This runner never generates a
confirmation population; the 1500..1507 roots are engineering development.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mechinterp.bpd.experiments.sub01.experiment import (
    WIDTH, PASSES, Looper, sample_x, external_labels, pca_projectors,
    projected_rollout, bernoulli_kl,
)
from mechinterp.bpd.experiments.sub02.experiment import (
    future_logit_gradients, obs_projectors,
)
from mechinterp.bpd.experiments.sub03.experiment import orthogonal

STUDY = "BPD-SUB-04"
ROOT = Path(__file__).resolve().parent
DEV_ROOTS = tuple(range(1500, 1508))
SMOKE = 1499
RANK = 8
STEPS = 256
TRAIN_N = 2048
DISC_N = 512
CHECK_N = 512
EVAL_N = 1024
OPT_STEPS = 80
OPT_LR = .02
OUTPUT_WEIGHT = .5
MIN_STATE_GAIN = .020
ANGLES = (.20, .40, .60, .80)
SHAM_COUNT = 8
BOOTSTRAPS = 2000
ALPHA_RATIO = .25
EVALMATH_COMMIT = "6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a"


def filehash(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_source(seed: int, *, steps: int = STEPS):
    if seed not in DEV_ROOTS and seed != SMOKE:
        raise ValueError("SUB04 source root outside frozen development roster")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = Looper()
    x = sample_x(seed, 1, TRAIN_N)
    y = external_labels(x)
    generator = torch.Generator().manual_seed(811031 + seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(steps):
        idx = torch.randint(TRAIN_N, (128,), generator=generator)
        prediction, _ = model.rollout(x[idx])
        loss = F.binary_cross_entropy_with_logits(prediction, y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction, _ = model.rollout(x)
        train_acc = float(((prediction >= 0) == y.bool()).float().mean())
    return model, {
        "root": seed, "steps": steps, "train_count": TRAIN_N,
        "batch_size": 128, "learning_rate": .01,
        "train_accuracy": train_acc,
        "shared_weight_density": float(
            (model.hidden.weight.detach().abs() >= 1e-4).float().mean()),
    }


@torch.no_grad()
def linearized_source(model: Looper, x: Tensor):
    """Physical D_h F over clean post-pass-1..3 states (N,3,12,12)."""
    _, states = model.rollout(x)
    w = model.hidden.weight.detach().double()
    output = [
        (1. - states[t+1].detach().double().square()).unsqueeze(-1)
        * w[None, :, :]
        for t in range(3)
    ]
    return torch.stack(output, dim=1)


def linearized_loss(p: Tensor, transition_jac: Tensor, cov_out: Tensor):
    eye = torch.eye(WIDTH, dtype=p.dtype, device=p.device)
    ip = eye - p
    cross = (p @ transition_jac) @ ip
    a = cross.square().sum(dim=(-2,-1)).mean()
    b = torch.trace(ip @ cov_out)
    return a, b


def source_joint_projector(model: Looper, discovery: Tensor):
    """Fixed source-only QR/Adam Grassmann objective, no target evaluation."""
    gradients, _ = future_logit_gradients(model, discovery)
    g = gradients.double()
    c_out = (g.T @ g / len(g)).detach()
    transition = linearized_source(model, discovery).detach()
    eigenvalues, eigenvectors = torch.linalg.eigh(c_out)
    start = eigenvectors[:, -RANK:].flip(-1).contiguous()
    initial_p = start @ start.T
    with torch.no_grad():
        a0_t, b0_t = linearized_loss(initial_p, transition, c_out)
        a0 = float(a0_t.clamp_min(1e-12))
        b0 = float(b0_t.clamp_min(1e-12))
    z = nn.Parameter(start.clone())
    optimizer = torch.optim.Adam([z], lr=OPT_LR)
    trace = []
    for step in range(OPT_STEPS):
        q, _ = torch.linalg.qr(z, mode="reduced")
        p = q @ q.T
        a, b = linearized_loss(p, transition, c_out)
        objective = a / a0 + OUTPUT_WEIGHT * b / b0
        if not torch.isfinite(objective):
            raise ValueError("Nonfinite joint-closure source objective")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if z.grad is None or not torch.isfinite(z.grad).all():
            raise ValueError("Nonfinite joint-projector source gradient")
        optimizer.step()
        trace.append({
            "step": step+1,
            "objective_before_update": float(objective.detach()),
            "closure_before_update": float(a.detach()),
            "future_output_before_update": float(b.detach()),
        })
    with torch.no_grad():
        q, _ = torch.linalg.qr(z, mode="reduced")
        p = q @ q.T
        a, b = linearized_loss(p, transition, c_out)
        final_loss = float(a/a0 + OUTPUT_WEIGHT*b/b0)
        reference = start @ start.T
        diagnostics = {
            "initial_closure_loss": float(a0_t),
            "initial_future_output_loss": float(b0_t),
            "final_closure_loss": float(a),
            "final_future_output_loss": float(b),
            "initial_objective": float(a0_t/a0 + OUTPUT_WEIGHT*b0_t/b0),
            "final_objective": final_loss,
            "optimizer_steps": OPT_STEPS, "optimizer_lr": OPT_LR,
            "terminal_weight": OUTPUT_WEIGHT,
            "eigenspectrum_c_out": eigenvalues.tolist(),
            "projector_idempotence_max": float((p @ p-p).abs().max()),
            "projector_symmetry_max": float((p-p.T).abs().max()),
            "projector_trace_error": abs(float(torch.trace(p))-RANK),
            "distance_from_observable_fro": float((p-reference).norm()),
            "source_loss_trace": trace,
        }
    return p.float(), q.detach(), c_out, diagnostics


def frame_for_local_controls(joint_basis: Tensor, c_out: Tensor):
    """Sort retained joint directions by their own source output importance."""
    local_cov = joint_basis.T @ c_out @ joint_basis
    vals, v = torch.linalg.eigh(local_cov)
    retained = joint_basis @ v.flip(-1)
    _, _, vt = torch.linalg.svd(joint_basis.T, full_matrices=True)
    discarded = vt[-(WIDTH-RANK):].T.contiguous()
    frame = torch.cat([discarded, retained.flip(-1)], dim=1)
    assert float((frame.T@frame-torch.eye(WIDTH,dtype=torch.float64))
                 .abs().max()) < 1e-9
    return retained, discarded, frame


@torch.no_grad()
def joint_rotated_shams(root: int, p_joint: Tensor,
                        joint_basis: Tensor, c_out: Tensor):
    """Four local rotations and four global, all exact-spectrum in float64."""
    retained, discarded, frame = frame_for_local_controls(joint_basis, c_out)
    top = retained[:, :4]
    weak = retained[:, 4:8]
    eigvals = torch.linalg.eigvalsh(c_out)
    artificial = (frame * eigvals[None, :]) @ frame.T
    full = joint_basis @ joint_basis.T
    out = []
    for index in range(SHAM_COUNT):
        if index < 4:
            theta = ANGLES[index]
            missing = discarded @ orthogonal(410017 + root*101 + index,4)
            rot = (torch.eye(WIDTH,dtype=torch.float64)
                  +(math.cos(theta)-1)*(weak@weak.T+missing@missing.T)
                  +math.sin(theta)*(missing@weak.T-weak@missing.T))
            kind = "local_joint_rotation"
        else:
            theta = None
            rot = orthogonal(420017 + root*101 + index,WIDTH)
            kind = "global_joint_rotation"
        proj = rot @ full @ rot.T
        eigen_after = torch.linalg.eigvalsh(rot @ artificial @ rot.T)
        check = {
            "orthogonality_max": float((rot.T@rot-torch.eye(
                WIDTH,dtype=torch.float64)).abs().max()),
            "projector_idempotence_max": float((proj@proj-proj).abs().max()),
            "projector_symmetry_max": float((proj-proj.T).abs().max()),
            "projector_trace_error": abs(float(torch.trace(proj))-RANK),
            "artificial_spectrum_difference_max":
                float((eigen_after-eigvals).abs().max()),
            "projection_distance_to_joint": float((proj-full).norm()),
        }
        out.append({
            "index": index, "type": kind,
            "theta": theta, "projector": proj.float(),
            "geometry": check,
        })
    return out


def raw_directions(seed: int, split: int, t: int, n: int):
    gen = torch.Generator().manual_seed(707100+seed*4001+split*17011+t*101)
    return torch.randn(n, WIDTH, generator=gen)


def physical_delta(seed: int, split: int, t: int, n: int,
                   projector: Tensor, alpha: float):
    raw = raw_directions(seed,split,t,n)
    d = raw @ (torch.eye(WIDTH)-projector)
    norms = d.norm(dim=-1,keepdim=True)
    if float(norms.min()) < 1e-8:
        raise ValueError("Degenerate null-fiber impulse")
    return alpha*d/norms


@torch.no_grad()
def score_transition(model: Looper, x: Tensor, projector: Tensor,
                     alpha: float, seed: int, split: int):
    """Genuinely reruns nonlinear source transition after each injected impulse."""
    clean_logits, clean_states = model.rollout(x)
    y = external_labels(x)
    clean_ok = (clean_logits >= 0) == y.bool()
    closure = []
    terminal = []
    future_displacements = []
    correct_corrupted = []
    wrong_repaired = []
    next_radius_clean = []
    next_radius_altered = []
    dose_error = 0.
    null_error = 0.
    for t in range(3):
        delta = physical_delta(seed,split,t,len(x),projector,alpha)
        dose_error = max(dose_error,float(
            (delta.norm(dim=-1)-alpha).abs().max()))
        null_error = max(null_error,float(
            (delta@projector).norm(dim=-1).max()))
        next_altered = model.step(clean_states[t]+delta,x)
        step_diff = (next_altered-clean_states[t+1])@projector
        closure.append(float(step_diff.double().square().sum(-1).mean().sqrt()))
        next_radius_clean.append(float(
            clean_states[t+1].double().norm(dim=-1).mean()))
        next_radius_altered.append(float(
            next_altered.double().norm(dim=-1).mean()))
        shifted = clean_states[t]+delta
        for _ in range(t+1,PASSES):
            shifted = model.step(shifted,x)
        changed_logits = model.head(shifted).squeeze(-1)
        terminal.append(float(
            (changed_logits-clean_logits).double().square().mean().sqrt()))
        future_displacements.append(float(
            (shifted-clean_states[-1]).double().square().sum(-1).mean().sqrt()))
        new_ok = (changed_logits>=0)==y.bool()
        correct_corrupted.append(int((clean_ok & ~new_ok).sum()))
        wrong_repaired.append(int((~clean_ok & new_ok).sum()))
    return {
        "per_pass_projected_next_state_euclidean_rms": closure,
        "pooled_projected_next_state_euclidean_rms":
            float(np.sqrt(np.mean(np.square(closure)))),
        "per_pass_terminal_logit_rms": terminal,
        "pooled_terminal_logit_rms":
            float(np.sqrt(np.mean(np.square(terminal)))),
        "per_pass_terminal_state_displacement_euclidean_rms":
            future_displacements,
        "per_pass_correct_corrupted_count": correct_corrupted,
        "per_pass_wrong_repaired_count": wrong_repaired,
        "per_pass_clean_next_state_mean_norm": next_radius_clean,
        "per_pass_altered_next_state_mean_norm": next_radius_altered,
        "max_physical_dose_error": dose_error,
        "max_null_projection_norm": null_error,
        "alpha": alpha,
    }


@torch.no_grad()
def task_fidelity(model: Looper, x: Tensor, projector: Tensor):
    clean, _ = model.rollout(x)
    candidate, _ = projected_rollout(model,x,projector)
    y = external_labels(x)
    return {
        "mean_clean_to_projected_kl": float(
            bernoulli_kl(clean,candidate).mean()),
        "true_label_nll": float(F.binary_cross_entropy_with_logits(candidate,y)),
        "correct_count": int(((candidate>=0)==y.bool()).sum()),
        "accuracy": float(((candidate>=0)==y.bool()).float().mean()),
    }


@torch.no_grad()
def transported_gauge_passes(model: Looper, x: Tensor, p: Tensor,
                            alpha: float, seed: int):
    """Exact conjugacy of each postpass intervention, plus wrong-vector sham."""
    gauge=orthogonal(750011+seed,WIDTH).float()
    clean_logits, clean_states=model.rollout(x)
    clean_out, g_clean_states = None, []
    h=x.new_zeros(len(x),WIDTH)
    for _ in range(PASSES):
        h=model.step(h@gauge,x)@gauge.T
        g_clean_states.append(h)
    clean_out=model.head(h@gauge).squeeze(-1)
    clean_max=float((clean_out-clean_logits).abs().max())
    clean_state_max=max(float((s@gauge-clean_states[t]).abs().max())
        for t,s in enumerate(g_clean_states))
    altered_max=0.;altered_state_max=0.; wrong_errors=[]
    for t in range(3):
        delta=physical_delta(seed,4,t,len(x),p,alpha)
        original=clean_states[t]+delta
        for _ in range(t+1,PASSES):
            original=model.step(original,x)
        original_logits=model.head(original).squeeze(-1)
        z=g_clean_states[t]+delta@gauge.T
        for _ in range(t+1,PASSES):
            z=model.step(z@gauge,x)@gauge.T
        altered_max=max(altered_max,float(
            (model.head(z@gauge).squeeze(-1)-original_logits).abs().max()))
        altered_state_max=max(altered_state_max,float(
            (z@gauge-original).abs().max()))
        z_wrong=g_clean_states[t]+delta
        for _ in range(t+1,PASSES):
            z_wrong=model.step(z_wrong@gauge,x)@gauge.T
        wrong_errors.append(float(
            (model.head(z_wrong@gauge).squeeze(-1)-original_logits)
            .double().square().mean().sqrt()))
    return {
        "orthogonality_max":float(
            (gauge.T@gauge-torch.eye(WIDTH)).abs().max()),
        "clean_logit_max_error":clean_max,
        "clean_state_max_error":clean_state_max,
        "intervention_logit_max_error":altered_max,
        "intervention_state_max_error":altered_state_max,
        "wrong_untransported_logit_rms_by_pass":wrong_errors,
    }


def evaluate_root(seed:int, *, steps:int=STEPS):
    if seed not in DEV_ROOTS and seed != SMOKE:
        raise ValueError("SUB04 root outside frozen roster")
    model, training=train_source(seed,steps=steps)
    xd=sample_x(seed,2,DISC_N)
    xs=sample_x(seed,3,CHECK_N)
    xe=sample_x(seed,4,EVAL_N)
    pca,alpha,pca_geometry=pca_projectors(model,xd)
    obs,obs_geometry=obs_projectors(model,xd)
    joint,joint_basis,c_out,joint_geometry=source_joint_projector(model,xd)
    shams=joint_rotated_shams(seed,joint,joint_basis,c_out)
    candidates={"pca":pca[RANK],"obs":obs[RANK],"joint":joint}
    with torch.no_grad():
        check_doses={
            key:max(float((physical_delta(seed,3,t,len(xs),p,alpha).norm(dim=-1)
                         -alpha).abs().max()) for t in range(3))
            for key,p in candidates.items()
        }
        results={
            name:{
                "transition":score_transition(model,xe,p,alpha,seed,4),
                "fidelity":task_fidelity(model,xe,p),
            } for name,p in candidates.items()
        }
        sham_results=[]
        for entry in shams:
            row={
                "index":entry["index"],"type":entry["type"],
                "theta":entry["theta"],"geometry":entry["geometry"],
                "transition":score_transition(
                    model,xe,entry["projector"],alpha,seed,4),
            }
            sham_results.append(row)
        gauge=transported_gauge_passes(model,xe[:64],joint,alpha,seed)
        source_logits,_=model.rollout(xe)
        y=external_labels(xe)
        context={}
        for name,mask in (
            ("x3_nonnegative",xe[:,3]>=0),
            ("x3_negative",xe[:,3]<0),
        ):
            context[name]=float(
                ((source_logits[mask]>=0)==y[mask].bool()).float().mean())
        source_acc=float(((source_logits>=0)==y.bool()).float().mean())
        source_nll=float(F.binary_cross_entropy_with_logits(source_logits,y))
        lipschitz=float(torch.linalg.matrix_norm(
            model.hidden.weight.detach().double(),ord=2))
    projector_ok=all(
        values["fidelity"]["correct_count"]>=0
        for values in results.values()
    ) and (
        joint_geometry["projector_idempotence_max"] <=1e-5
        and joint_geometry["projector_symmetry_max"] <=1e-5
        and joint_geometry["projector_trace_error"] <=1e-4
        and pca_geometry["nested_max_abs"]<=1e-5
        and obs_geometry["finite_difference_max_abs"]<.02
    )
    sham_geometry_ok=all(all(
        a["geometry"][key] <= 1e-6 for key in (
            "orthogonality_max","projector_idempotence_max",
            "projector_symmetry_max","projector_trace_error",
            "artificial_spectrum_difference_max",
        )) for a in sham_results)
    dose_ok=all(
        row["transition"]["max_physical_dose_error"]<=5e-6
        and row["transition"]["max_null_projection_norm"]<=5e-6
        for row in (*results.values(),)
    ) if False else all(
        row["transition"]["max_physical_dose_error"]<=5e-6
        and row["transition"]["max_null_projection_norm"]<=5e-6
        for row in list(results.values())+sham_results
    ) and all(z<=5e-6 for z in check_doses.values())
    gauge_ok=(
        gauge["clean_logit_max_error"]<=5e-5
        and gauge["clean_state_max_error"]<=5e-5
        and gauge["intervention_logit_max_error"]<=5e-5
        and gauge["intervention_state_max_error"]<=5e-5
        and max(gauge["wrong_untransported_logit_rms_by_pass"])>1e-5
    )
    root_eligible=bool(
        min(context.values())>=.8 and projector_ok
        and sham_geometry_ok and dose_ok and gauge_ok
        and all(np.isfinite(row["transition"]["pooled_projected_next_state_euclidean_rms"])
                for row in list(results.values())+sham_results)
    )
    primary=(results["obs"]["transition"]["pooled_projected_next_state_euclidean_rms"]
             -results["joint"]["transition"]["pooled_projected_next_state_euclidean_rms"])
    near=(sham_results[0]["transition"]["pooled_projected_next_state_euclidean_rms"]
          -results["joint"]["transition"]["pooled_projected_next_state_euclidean_rms"])
    return {
        "study":STUDY,"phase":"EXCLUDED_SMOKE" if seed==SMOKE else "ENGINEERING",
        "root":seed,
        "protocol_sha256":filehash(ROOT/"PROTOCOL.md"),
        "source_sha256":filehash(Path(__file__)),
        "dependency_sub01_sha256":filehash(ROOT.parent/"sub01"/"experiment.py"),
        "dependency_sub02_sha256":filehash(ROOT.parent/"sub02"/"experiment.py"),
        "environment":{"python":platform.python_version(),
                       "torch":torch.__version__,"numpy":np.__version__},
        "training":training,
        "data_sizes":{"train":TRAIN_N,"discovery":DISC_N,
                      "selection":CHECK_N,"heldout":EVAL_N},
        "alpha":alpha,
        "source":{"task_accuracy":source_acc,"true_label_nll":source_nll,
                  "contexts":context,"global_step_lipschitz_upper":lipschitz},
        "geometry":{"pca":pca_geometry,"obs":obs_geometry,
                    "joint":joint_geometry},
        "check_doses":check_doses,
        "methods":results,
        "shams":sham_results,
        "gauge":gauge,
        "projector_ok":projector_ok,
        "sham_geometry_ok":sham_geometry_ok,
        "dose_ok":dose_ok,
        "gauge_ok":gauge_ok,
        "root_eligible":root_eligible,
        "primary_obs_minus_joint_state_rms":primary,
        "direct_near_joint_sham_minus_joint_state_rms":near,
    }


def bootstrap(values:list[float],seed:int=150404):
    if not values:
        return None
    data=np.asarray(values,dtype=float)
    gen=np.random.default_rng(seed)
    samples=gen.choice(data,size=(BOOTSTRAPS,len(data)),replace=True).mean(1)
    return [float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]


def bounded_signed(v:float):
    return v/(1.+abs(v))


def preflight_spec(records:list[dict]):
    return {
        "schema_version":1, "experiment_id":STUDY,
        "independent_unit":"independently trained model root",
        "alpha":"0.05",
        "scientific_min_effect":"0.0196078431372549",
        "target_effect":"0.04761904761904762",
        "measurement_resolution":"0.002",
        "sham_quantile":"0.95","sham_confidence":"0.95",
        "planned_comparisons":1,"confirm_units_planned":64,
        "pilot_from_development_only":True,
        "heldout_sealed":True,"units_independent":True,
        "pilot":[{
            "unit_id":f"root-{r['root']}",
            "paired_effect":repr(bounded_signed(
                r["primary_obs_minus_joint_state_rms"])),
            "sham_effects":[repr(bounded_signed(max(
                0.,r["methods"]["obs"]["transition"]
                ["pooled_projected_next_state_euclidean_rms"]
                - s["transition"]["pooled_projected_next_state_euclidean_rms"]
            ))) for s in r["shams"]],
        } for r in records],
    }


def aggregate(rows:list[dict]):
    if [r["root"] for r in rows]!=list(DEV_ROOTS):
        raise ValueError("Must retain all eight registered engineering roots")
    qualified=[r for r in rows if r.get("root_eligible",False)]
    effects=[r["primary_obs_minus_joint_state_rms"] for r in qualified]
    mean=float(np.mean(effects)) if effects else None
    ci=bootstrap(effects) if len(effects)==len(DEV_ROOTS) else None
    positive=sum(v>0 for v in effects)
    def avgmetric(name,metric,key):
        return (float(np.mean([r["methods"][name][metric][key]
                     for r in qualified])) if qualified else None)
    mean_kl_joint=avgmetric("joint","fidelity","mean_clean_to_projected_kl")
    mean_kl_delta=(float(np.mean([
        r["methods"]["joint"]["fidelity"]["mean_clean_to_projected_kl"]
        -r["methods"]["obs"]["fidelity"]["mean_clean_to_projected_kl"]
        for r in qualified])) if qualified else None)
    mean_nll_delta=(float(np.mean([
        r["methods"]["joint"]["fidelity"]["true_label_nll"]
        -r["methods"]["obs"]["fidelity"]["true_label_nll"]
        for r in qualified])) if qualified else None)
    mean_logit_delta=(float(np.mean([
        r["methods"]["joint"]["transition"]["pooled_terminal_logit_rms"]
        -r["methods"]["obs"]["transition"]["pooled_terminal_logit_rms"]
        for r in qualified])) if qualified else None)
    gate=bool(
        len(qualified)==len(DEV_ROOTS)
        and mean>=MIN_STATE_GAIN and positive>=7 and ci[0]>0
        and mean_kl_joint<=.040 and mean_kl_delta<=.015
        and mean_nll_delta<=.030 and mean_logit_delta<=.040
    )
    return {
        "study":STUDY,"phase":"ENGINEERING",
        "attempted_roots":len(rows),"root_ids":list(DEV_ROOTS),
        "eligible_roots":len(qualified),
        "ineligible_roots":[r["root"] for r in rows if not r.get("root_eligible",False)],
        "registered_primary":"pooled_next_state_euclidean_rms_obs_minus_joint",
        "root_primary_effects":effects,
        "mean_primary_effect":mean,
        "bootstrap95_whole_root":ci,
        "positive_roots":positive,
        "minimum_mechanistic_effect":MIN_STATE_GAIN,
        "mean_joint_clean_kl":mean_kl_joint,
        "mean_joint_minus_obs_clean_kl":mean_kl_delta,
        "mean_joint_minus_obs_true_nll":mean_nll_delta,
        "mean_joint_minus_obs_terminal_logit_rms":mean_logit_delta,
        "mean_obs_closure":avgmetric("obs","transition",
                                   "pooled_projected_next_state_euclidean_rms"),
        "mean_joint_closure":avgmetric("joint","transition",
                                   "pooled_projected_next_state_euclidean_rms"),
        "mean_pca_closure":avgmetric("pca","transition",
                                   "pooled_projected_next_state_euclidean_rms"),
        "mean_direct_near_sham_minus_joint":(
            float(np.mean([r["direct_near_joint_sham_minus_joint_state_rms"]
                            for r in qualified])) if qualified else None),
        "independent_sham_tail_calibrated":False,
        "strict_evalmath_preflight":"NOT_RUN_YET",
        "future_confirmation":"UNGENERATED_SEALED",
        "verdict":"ENGINEERING_SIGNAL_ONLY" if gate
            else "ENGINEERING_NULL_OR_INELIGIBLE",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",required=True,choices=("smoke","development"))
    parser.add_argument("--ack-development",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--pilot-out",type=Path)
    args=parser.parse_args()
    if args.phase=="development" and not args.ack_development:
        parser.error("Development requires explicit --ack-development")
    if args.phase=="smoke":
        record=evaluate_root(SMOKE,steps=16)
        receipt=record
        short={"root":SMOKE,"phase":"EXCLUDED_SMOKE",
               "eligible":record["root_eligible"],
               "primary":record["primary_obs_minus_joint_state_rms"]}
    else:
        records=[]
        for seed in DEV_ROOTS:
            try:
                records.append(evaluate_root(seed))
            except Exception as err:
                records.append({"study":STUDY,"phase":"ENGINEERING",
                                "root":seed,"root_eligible":False,
                                "error_type":type(err).__name__,
                                "error_message":str(err)})
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps({
                    "study":STUDY,"phase":"PARTIAL",
                    "roots":records,"completed_roots":len(records)
                },sort_keys=True,indent=2,allow_nan=False)+"\n")
        stats=aggregate(records)
        receipt={"study":STUDY,"summary":stats,"roots":records}
        short=stats
        if args.pilot_out:
            admitted=[r for r in records if r.get("root_eligible",False)]
            if admitted:
                args.pilot_out.parent.mkdir(parents=True,exist_ok=True)
                args.pilot_out.write_text(json.dumps(preflight_spec(admitted),
                                    sort_keys=True,indent=2,allow_nan=False)+"\n")
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(receipt,sort_keys=True,indent=2,
                                           allow_nan=False)+"\n")
    print(json.dumps(short,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
