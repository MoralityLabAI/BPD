"""SUB-05: clock-indexed causal quotients, frozen source-only optimizer and finite-dose test.

PROTOCOL.md was committed before this module or fresh 1600..1607 model data.
No confirmation or independent sham-calibration population can be generated here.
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
    WIDTH, PASSES, Looper, sample_x, external_labels, bernoulli_kl,
)
from mechinterp.bpd.experiments.sub03.experiment import orthogonal
from mechinterp.bpd.experiments.sub04.experiment import (
    source_joint_projector, linearized_source, physical_delta,
)

STUDY = "BPD-SUB-05"
ROOT = Path(__file__).resolve().parent
ROOTS = tuple(range(1600, 1608))
SMOKE = 1599
RANK = 8
STEPS = 256
TRAIN_N = 2048
DISCOVERY_N = 512
CHECK_N = 512
HELDOUT_N = 1024
OPT_STEPS = 120
OPT_LR = .02
OUTPUT_WEIGHT = .35
SMOOTH_WEIGHT = .50
MIN_EFFECT = .015
BOOTSTRAPS = 2000
ANGLES = (.20, .40, .60, .80)
TIME_PERMUTATIONS = (
    (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2),
    (1, 0, 3, 2), (2, 1, 0, 3), (0, 2, 1, 3), (3, 2, 1, 0),
)
EVALMATH_COMMIT = "6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a"


def filehash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_source(seed: int, *, steps: int = STEPS):
    if seed not in ROOTS and seed != SMOKE:
        raise ValueError("SUB05 root outside frozen training roster")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = Looper()
    x = sample_x(seed, 1, TRAIN_N)
    y = external_labels(x)
    generator = torch.Generator().manual_seed(811031 + seed)
    optim = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(steps):
        idx = torch.randint(TRAIN_N, (128,), generator=generator)
        logits, _ = model.rollout(x[idx])
        loss = F.binary_cross_entropy_with_logits(logits, y[idx])
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model.rollout(x)
        acc = float(((logits>=0)==y.bool()).float().mean())
        density=float((model.hidden.weight.detach().abs()>=1e-4).float().mean())
    return model, {
        "training_steps": steps, "training_examples": TRAIN_N,
        "batch_size": 128, "learning_rate": .01,
        "training_accuracy":acc, "physical_matrix_density":density,
    }


def pass_output_covariances(model: Looper, discovery: Tensor):
    """Source gradients d terminal logit / d each physical post-pass hidden state."""
    with torch.enable_grad():
        _, states = model.rollout(discovery)
        output = model.head(states[-1]).squeeze(-1)
        grads = [
            torch.autograd.grad(output.sum(), states[t],
                                retain_graph=(t<3))[0].detach().double()
            for t in range(PASSES)
        ]
    c = torch.stack([g.T@g/len(g) for g in grads])
    return c.detach(), [float(g.norm(dim=-1).mean()) for g in grads]


def bundle_loss(p: Tensor, transitions: Tensor, covariances: Tensor):
    """Exact source-only Frobenius bridge, output observability and smoothness."""
    identity=torch.eye(WIDTH,device=p.device,dtype=p.dtype)
    left=p[1:][None, :, :, :]
    right=(identity-p[:-1])[None, :, :, :]
    bridge=left@transitions@right
    a=bridge.square().sum(dim=(-1,-2)).mean()
    b=torch.einsum("tij,tji->",identity[None]-p,covariances)/PASSES
    smooth=(p[1:]-p[:-1]).square().sum(dim=(-1,-2)).mean()/(2*RANK)
    return a,b,smooth


def source_bundle(model: Looper, discovery: Tensor, static_basis: Tensor):
    """120-step QR/Adam source-only optimization, starting from stationary subspace."""
    if static_basis.shape!=(WIDTH,RANK):
        raise ValueError("Expected frozen static full-column orthogonal basis")
    transition=linearized_source(model,discovery).detach().double()
    cov, gradnorm=pass_output_covariances(model,discovery)
    basis=static_basis.detach().double()
    original=basis@basis.T
    initial=torch.stack([original for _ in range(PASSES)])
    with torch.no_grad():
        aa,bb,ss=bundle_loss(initial,transition,cov)
        a0=float(aa.clamp_min(1e-12))
        b0=float(bb.clamp_min(1e-12))
    parameters=nn.Parameter(basis.repeat(PASSES,1,1).clone())
    opt=torch.optim.Adam([parameters],lr=OPT_LR)
    trace=[]
    for step in range(OPT_STEPS):
        q,_=torch.linalg.qr(parameters,mode="reduced")
        p=q@q.transpose(-2,-1)
        a,b,s=bundle_loss(p,transition,cov)
        objective=a/a0+OUTPUT_WEIGHT*b/b0+SMOOTH_WEIGHT*s
        if not torch.isfinite(objective):
            raise ValueError("Nonfinite clock-bundle discovery objective")
        opt.zero_grad(set_to_none=True)
        objective.backward()
        if parameters.grad is None or not torch.isfinite(parameters.grad).all():
            raise ValueError("Nonfinite clock-bundle QR gradient")
        opt.step()
        trace.append({
            "step":step+1,"objective_before_update":float(objective.detach()),
            "closure_before_update":float(a.detach()),
            "output_before_update":float(b.detach()),
            "smoothness_before_update":float(s.detach()),
        })
    with torch.no_grad():
        q,_=torch.linalg.qr(parameters,mode="reduced")
        p=q@q.transpose(-2,-1)
        a,b,s=bundle_loss(p,transition,cov)
        final=float(a/a0+OUTPUT_WEIGHT*b/b0+SMOOTH_WEIGHT*s)
        diagnostics={
            "optimizer_steps":OPT_STEPS,"optimizer_lr":OPT_LR,
            "output_weight":OUTPUT_WEIGHT,"smooth_weight":SMOOTH_WEIGHT,
            "initial_closure":float(aa),"initial_output":float(bb),
            "initial_smoothness":float(ss),
            "initial_objective":float(aa/a0+OUTPUT_WEIGHT*bb/b0),
            "final_closure":float(a),"final_output":float(b),
            "final_smoothness":float(s),"final_objective":final,
            "grad_norm_per_pass":gradnorm,
            "trace":trace,
            "projector_symmetry_max":float(
                (p-p.transpose(-2,-1)).abs().max()),
            "projector_idempotence_max":float((p@p-p).abs().max()),
            "projector_trace_max_error":max(
                abs(float(torch.trace(pt))-RANK) for pt in p),
            "distance_to_stationary_fro_per_pass":[
                float((pt-original).norm()) for pt in p],
            "distance_adjacent_fro":[
                float((p[t+1]-p[t]).norm()) for t in range(3)],
        }
    return p.float(),q.detach(),cov,diagnostics


def local_rotation(q: Tensor, cov: Tensor, theta: float, seed: int):
    """Rotate weak four retained modes into the four missing modes."""
    if theta not in ANGLES:
        raise ValueError("Unregistered local rotation angle")
    retained_cov=q.T@cov@q
    _,v=torch.linalg.eigh(retained_cov)
    ordered=q@v.flip(-1)
    strong=ordered[:,:4]
    weak=ordered[:,4:]
    _,_,vh=torch.linalg.svd(q.T,full_matrices=True)
    complement=vh[-(WIDTH-RANK):].T
    missing=complement@orthogonal(seed,WIDTH-RANK)
    co,si=math.cos(theta),math.sin(theta)
    rot=(torch.eye(WIDTH,dtype=torch.float64)
        +(co-1.)*(weak@weak.T+missing@missing.T)
        +si*(missing@weak.T-weak@missing.T))
    return rot, strong


@torch.no_grad()
def bundle_shams(root:int, q:Tensor, cov:Tensor):
    if q.shape!=(PASSES,WIDTH,RANK):
        raise ValueError("Incorrect learned bundle bases")
    entries=[]
    full=q@q.transpose(-2,-1)
    for i in range(8):
        if i<4:
            angle=ANGLES[i]
            rotations=[
                local_rotation(q[t],cov[t],angle,
                    960011+root*313+101*i+17*t)[0]
                for t in range(PASSES)
            ]
            kind="local_angular"
        else:
            angle=None
            universal=orthogonal(970011+root*313+i,WIDTH)
            rotations=[universal for _ in range(PASSES)]
            kind="shared_global_haar"
        ps=torch.stack([rotations[t]@full[t]@rotations[t].T
                        for t in range(PASSES)])
        checks={
            "orthogonality_max":max(float(
                (rot.T@rot-torch.eye(WIDTH,dtype=torch.float64)).abs().max())
                for rot in rotations),
            "rank_trace_max_error":max(abs(float(torch.trace(pt))-RANK)
                                        for pt in ps),
            "idempotence_max":float((ps@ps-ps).abs().max()),
            "symmetry_max":float((ps-ps.transpose(-2,-1)).abs().max()),
            "joint_distance_fro_by_pass":[
                float((ps[t]-full[t]).norm()) for t in range(PASSES)],
        }
        entries.append({"index":i,"kind":kind,"theta":angle,
                        "projectors":ps.float(),"geometry":checks})
    return entries


def stationary_bundle(p:Tensor):
    if p.shape!=(WIDTH,WIDTH):
        raise ValueError("Stationary physical projector must be 12x12")
    return torch.stack([p for _ in range(PASSES)])


def permuted_bundle(bundle:Tensor, permutation:tuple[int,...]):
    if permutation not in TIME_PERMUTATIONS:
        raise ValueError("Unregistered clock permutation")
    return bundle[list(permutation)]


@torch.no_grad()
def causal_response(model:Looper,x:Tensor,bundle:Tensor,alpha:float,
                    seed:int,split:int):
    if bundle.shape!=(PASSES,WIDTH,WIDTH):
        raise ValueError("Expected four physical rank-eight projectors")
    teacher,states=model.rollout(x)
    y=external_labels(x)
    base_correct=(teacher>=0)==y.bool()
    step_errors=[]
    logit_errors=[]
    terminal_displacements=[]
    clean_radii=[]
    changed_radii=[]
    corrupted=[]
    repaired=[]
    physical_error=0.
    null_error=0.
    for t in range(PASSES-1):
        src=bundle[t]
        dst=bundle[t+1]
        delta=physical_delta(seed,split,t,len(x),src,alpha)
        physical_error=max(physical_error,float(
            (delta.norm(dim=-1)-alpha).abs().max()))
        null_error=max(null_error,float(
            (delta@src).norm(dim=-1).max()))
        next0=model.step(states[t],x)
        next1=model.step(states[t]+delta,x)
        step_errors.append(float(
            ((next1-next0)@dst).double().square().sum(-1).mean().sqrt()))
        clean_radii.append(float(next0.double().norm(dim=-1).mean()))
        changed_radii.append(float(next1.double().norm(dim=-1).mean()))
        h=states[t]+delta
        for _ in range(t+1,PASSES):
            h=model.step(h,x)
        logits=model.head(h).squeeze(-1)
        logit_errors.append(float(
            (logits-teacher).double().square().mean().sqrt()))
        terminal_displacements.append(float(
            (h-states[-1]).double().square().sum(-1).mean().sqrt()))
        changed_correct=(logits>=0)==y.bool()
        corrupted.append(int((base_correct & ~changed_correct).sum()))
        repaired.append(int((~base_correct & changed_correct).sum()))
    return {
        "per_pass_next_state_rms":step_errors,
        "pooled_next_state_rms":float(np.sqrt(
            np.mean(np.square(step_errors)))),
        "per_pass_terminal_logit_rms":logit_errors,
        "pooled_terminal_logit_rms":float(np.sqrt(
            np.mean(np.square(logit_errors)))),
        "per_pass_terminal_state_displacement_rms":terminal_displacements,
        "per_pass_clean_next_radius":clean_radii,
        "per_pass_changed_next_radius":changed_radii,
        "per_pass_correct_corrupted":corrupted,
        "per_pass_wrong_repaired":repaired,
        "physical_alpha":alpha,
        "max_physical_dose_error":physical_error,
        "max_null_impulse_projection":null_error,
    }


@torch.no_grad()
def clocked_student(model:Looper,x:Tensor,bundle:Tensor):
    h=x.new_zeros(len(x),WIDTH)
    for t in range(PASSES):
        h=model.step(h,x)@bundle[t]
    out=model.head(h).squeeze(-1)
    truth=external_labels(x)
    clean,_=model.rollout(x)
    return {
        "mean_clean_bernoulli_kl":float(bernoulli_kl(clean,out).mean()),
        "true_nll":float(F.binary_cross_entropy_with_logits(out,truth)),
        "correct_count":int(((out>=0)==truth.bool()).sum()),
        "accuracy":float(((out>=0)==truth.bool()).float().mean()),
        "max_full_rank_parity":None,
    }


@torch.no_grad()
def gauge_checks(model:Looper,x:Tensor,bundle:Tensor,alpha:float,seed:int):
    """Transport each clock projector and source impulse under z=Qh."""
    q=orthogonal(980011+seed,WIDTH).float()
    rotated=torch.stack([q@p@q.T for p in bundle])
    clean,states=model.rollout(x)
    state_g=[h@q.T for h in states]
    error_step=0.
    error_logit=0.
    wrong=[]
    for t in range(3):
        delta=physical_delta(seed,4,t,len(x),bundle[t],alpha)
        source0=model.step(states[t],x)
        source1=model.step(states[t]+delta,x)
        source_diff=(source1-source0)@bundle[t+1]
        gauged0=model.step(state_g[t]@q,x)@q.T
        gauged1=model.step((state_g[t]+delta@q.T)@q,x)@q.T
        gauged_diff=(gauged1-gauged0)@rotated[t+1]
        error_step=max(error_step,float(
            (gauged_diff@q-source_diff).abs().max()))
        h0=states[t]+delta
        for _ in range(t+1,PASSES):
            h0=model.step(h0,x)
        first=model.head(h0).squeeze(-1)
        z=state_g[t]+delta@q.T
        for _ in range(t+1,PASSES):
            z=model.step(z@q,x)@q.T
        second=model.head(z@q).squeeze(-1)
        error_logit=max(error_logit,float((first-second).abs().max()))
        zwrong=state_g[t]+delta
        for _ in range(t+1,PASSES):
            zwrong=model.step(zwrong@q,x)@q.T
        wrong.append(float(
            (model.head(zwrong@q).squeeze(-1)-first)
            .double().square().mean().sqrt()))
    z=x.new_zeros(len(x),WIDTH)
    for t in range(PASSES):
        z=model.step(z@q,x)@q.T
    clean_g=model.head(z@q).squeeze(-1)
    return {
        "q_orthogonality_error":float(
            (q.T@q-torch.eye(WIDTH)).abs().max()),
        "clean_logit_error":float((clean_g-clean).abs().max()),
        "transported_next_state_error":error_step,
        "transported_logit_error":error_logit,
        "wrong_untransported_logit_rms_by_pass":wrong,
    }


def evaluate_root(seed:int,*,steps:int=STEPS):
    model,training=train_source(seed,steps=steps)
    discovery=sample_x(seed,2,DISCOVERY_N)
    check=sample_x(seed,3,CHECK_N)
    heldout=sample_x(seed,4,HELDOUT_N)
    static,static_basis,_,static_info=source_joint_projector(model,discovery)
    bundle,source_basis,source_cov,source_info=source_bundle(
        model,discovery,static_basis)
    shams=bundle_shams(seed,source_basis,source_cov)
    arms={"aligned":bundle,
          "stationary":stationary_bundle(static)}
    arms.update({
        "time_perm_"+str(i):permuted_bundle(bundle,p)
        for i,p in enumerate(TIME_PERMUTATIONS)
    })
    arms.update({"angular_"+str(i):v["projectors"]
                 for i,v in enumerate(shams)})
    with torch.no_grad():
        calibration={
            k: [
                float((physical_delta(seed,3,t,len(check),p[t],
                                      .25*float(model.rollout(discovery)[1][0]
                                      .double().square().sum(-1).mean().sqrt()))
                       .norm(dim=-1)-
                       .25*float(model.rollout(discovery)[1][0]
                       .double().square().sum(-1).mean().sqrt())
                       ).abs().max())
                for t in range(3)]
            for k,p in (("aligned",bundle),("stationary",arms["stationary"]))
        }
        _,disc_states=model.rollout(discovery)
        alpha=.25*float(disc_states[0].double().square().sum(-1).mean().sqrt())
        scores={name:causal_response(model,heldout,ps,alpha,seed,4)
                for name,ps in arms.items()}
        students={name:clocked_student(model,heldout,ps)
                  for name,ps in (("aligned",bundle),
                                  ("stationary",arms["stationary"]))}
        full=stationary_bundle(torch.eye(WIDTH))
        full_student=clocked_student(model,heldout,full)
        clean,clean_states=model.rollout(heldout)
        full_logits,_=model.rollout(heldout)
        source_y=external_labels(heldout)
        masks={"x3_nonnegative":heldout[:,3]>=0,
               "x3_negative":heldout[:,3]<0}
        contexts={
            k:float(((clean[mask]>=0)==source_y[mask].bool()).float().mean())
            for k,mask in masks.items()
        }
        gauge=gauge_checks(model,heldout[:64],bundle,alpha,seed)
        source_nll=float(F.binary_cross_entropy_with_logits(clean,source_y))
    input_checks=all(
        all(v<=5e-6 for v in item)
        for item in calibration.values())
    angular_geom=all(
        all(z["geometry"][k]<=1e-6 for k in
            ("orthogonality_max","rank_trace_max_error",
             "idempotence_max","symmetry_max"))
        for z in shams)
    source_geom=(source_info["projector_idempotence_max"]<1e-5
        and source_info["projector_symmetry_max"]<1e-5
        and source_info["projector_trace_max_error"]<1e-4
        and static_info["projector_idempotence_max"]<1e-5)
    all_dose=all(
        z["max_physical_dose_error"]<=5e-6
        and z["max_null_impulse_projection"]<=5e-6
        for z in scores.values())
    gauge_ok=(gauge["clean_logit_error"]<=5e-5
        and gauge["transported_next_state_error"]<=5e-5
        and gauge["transported_logit_error"]<=5e-5
        and max(gauge["wrong_untransported_logit_rms_by_pass"])>1e-5)
    fullparity=full_student["mean_clean_bernoulli_kl"]<=1e-7
    eligible=bool(min(contexts.values())>=.80 and
        angular_geom and source_geom and all_dose and input_checks
        and gauge_ok and fullparity and len(scores)==17)
    best_control=min(
        [scores["stationary"]["pooled_next_state_rms"]]+[
        scores["time_perm_"+str(i)]["pooled_next_state_rms"]
        for i in range(len(TIME_PERMUTATIONS))])
    primary=best_control-scores["aligned"]["pooled_next_state_rms"]
    return {
        "study":STUDY,
        "phase":"EXCLUDED_SMOKE" if seed==SMOKE else "ENGINEERING",
        "root":seed,
        "protocol_sha256":filehash(ROOT/"PROTOCOL.md"),
        "source_sha256":filehash(Path(__file__)),
        "dependency_sub04_sha256":filehash(ROOT.parent/"sub04"/"experiment.py"),
        "environment":{"python":platform.python_version(),
            "torch":torch.__version__,"numpy":np.__version__},
        "training":training,
        "data_sizes":{"train":TRAIN_N,"discovery":DISCOVERY_N,
            "selection":CHECK_N,"heldout":HELDOUT_N},
        "alpha":alpha,"source_nll":source_nll,
        "source_context_accuracy":contexts,
        "static_source":static_info,"clock_source":source_info,
        "temporal_permutations":[list(p) for p in TIME_PERMUTATIONS],
        "angular_shams":[{"index":v["index"],"kind":v["kind"],
            "theta":v["theta"],"geometry":v["geometry"]}
            for v in shams],
        "calibration_max_dose_errors":calibration,
        "scores":scores,"students":students,
        "full_rank_student_kl":full_student["mean_clean_bernoulli_kl"],
        "gauge":gauge,
        "geometry_ok":source_geom and angular_geom,
        "dose_ok":all_dose and input_checks,
        "gauge_ok":gauge_ok,
        "root_eligible":eligible,
        "strongest_control_closure":best_control,
        "aligned_closure":scores["aligned"]["pooled_next_state_rms"],
        "stationary_minus_aligned":
            scores["stationary"]["pooled_next_state_rms"]-
            scores["aligned"]["pooled_next_state_rms"],
        "strongest_permutation_minus_aligned":min(
            scores["time_perm_"+str(i)]["pooled_next_state_rms"]
            for i in range(len(TIME_PERMUTATIONS)))-
            scores["aligned"]["pooled_next_state_rms"],
        "primary_strongest_null_minus_aligned":primary,
    }


def bootstrap(values:list[float]):
    if not values:
        return None
    a=np.asarray(values,dtype=float)
    gen=np.random.default_rng(160505)
    out=gen.choice(a,size=(BOOTSTRAPS,len(a)),replace=True).mean(1)
    return [float(np.quantile(out,.025)),float(np.quantile(out,.975))]


def bounded(v:float):
    return v/(1.+abs(v))


def preflight_spec(roots:list[dict]):
    return {
        "schema_version":1,"experiment_id":STUDY,
        "independent_unit":"independently trained model root",
        "alpha":"0.05","scientific_min_effect":"0.014778325123152709",
        "target_effect":"0.038461538461538464",
        "measurement_resolution":"0.002","sham_quantile":"0.95",
        "sham_confidence":"0.95","planned_comparisons":1,
        "confirm_units_planned":64,"pilot_from_development_only":True,
        "heldout_sealed":True,"units_independent":True,
        "pilot":[{
            "unit_id":f"root-{r['root']}",
            "paired_effect":repr(bounded(
                r["primary_strongest_null_minus_aligned"])),
            "sham_effects":[repr(bounded(max(0.,
                r["scores"]["stationary"]["pooled_next_state_rms"]-
                r["scores"]["time_perm_"+str(i)]["pooled_next_state_rms"]
            ))) for i in range(len(TIME_PERMUTATIONS))],
        } for r in roots],
    }


def aggregate(rows:list[dict]):
    if [r["root"] for r in rows]!=list(ROOTS):
        raise ValueError("Must retain original eight SUB05 independent roots")
    qualified=[r for r in rows if r.get("root_eligible",False)]
    effects=[r["primary_strongest_null_minus_aligned"] for r in qualified]
    mean=float(np.mean(effects)) if effects else None
    ci=bootstrap(effects) if len(qualified)==8 else None
    positive=sum(v>0 for v in effects)
    def mean_by(path):
        vals=[]
        for r in qualified:
            v=r
            for key in path:v=v[key]
            vals.append(v)
        return float(np.mean(vals)) if vals else None
    kl=mean_by(["students","aligned","mean_clean_bernoulli_kl"])
    kl_extra=(float(np.mean([
        r["students"]["aligned"]["mean_clean_bernoulli_kl"]-
        r["students"]["stationary"]["mean_clean_bernoulli_kl"]
        for r in qualified])) if qualified else None)
    nll_extra=(float(np.mean([
        r["students"]["aligned"]["true_nll"]-
        r["students"]["stationary"]["true_nll"]
        for r in qualified])) if qualified else None)
    terminal_extra=(float(np.mean([
        r["scores"]["aligned"]["pooled_terminal_logit_rms"]-
        r["scores"]["stationary"]["pooled_terminal_logit_rms"]
        for r in qualified])) if qualified else None)
    allowed=bool(len(qualified)==8 and mean>=MIN_EFFECT and positive>=7
        and ci[0]>0 and kl<=.040 and kl_extra<=.015
        and nll_extra<=.030 and terminal_extra<=.040)
    return {
        "study":STUDY,"phase":"ENGINEERING",
        "attempted_roots":len(rows),
        "root_ids":list(ROOTS),"eligible_roots":len(qualified),
        "ineligible_roots":[r["root"] for r in rows if not r.get("root_eligible",False)],
        "registered_primary":"min_stationary_all_permutations_minus_aligned_state_rms",
        "root_primary_effects":effects,
        "mean_primary_effect":mean,"bootstrap95_whole_roots":ci,
        "positive_root_count":positive,
        "registered_minimum":MIN_EFFECT,
        "mean_aligned_closure":mean_by(["aligned_closure"]),
        "mean_stationary_closure":mean_by(["scores","stationary","pooled_next_state_rms"]),
        "mean_best_permutation_closure":float(np.mean([
           min(r["scores"]["time_perm_"+str(i)]["pooled_next_state_rms"]
               for i in range(len(TIME_PERMUTATIONS)))
           for r in qualified])) if qualified else None,
        "mean_stationary_minus_aligned":
            mean_by(["stationary_minus_aligned"]),
        "mean_strongest_permutation_minus_aligned":
            mean_by(["strongest_permutation_minus_aligned"]),
        "mean_aligned_clean_kl":kl,"mean_aligned_minus_static_clean_kl":kl_extra,
        "mean_aligned_minus_static_true_nll":nll_extra,
        "mean_aligned_minus_static_terminal_logit_rms":terminal_extra,
        "scientific_sham_tail_calibrated":False,
        "strict_evalmath":"PENDING_PILOT_ONLY_PREFLIGHT",
        "future_confirmation":"UNGENERATED_SEALED",
        "verdict":"ENGINEERING_SIGNAL_ONLY" if allowed
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
        parser.error("Development requires --ack-development")
    if args.phase=="smoke":
        receipt=evaluate_root(SMOKE,steps=16)
        brief={"phase":"EXCLUDED_SMOKE","root":SMOKE,
            "eligible":receipt["root_eligible"],
            "primary":receipt["primary_strongest_null_minus_aligned"]}
    else:
        rows=[]
        for seed in ROOTS:
            try:
                rows.append(evaluate_root(seed))
            except Exception as err:
                rows.append({"study":STUDY,"phase":"ENGINEERING","root":seed,
                    "root_eligible":False,"error_type":type(err).__name__,
                    "error_message":str(err)})
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps({
                    "study":STUDY,"phase":"PARTIAL",
                    "completed_roots":len(rows),"roots":rows,
                },sort_keys=True,indent=2,allow_nan=False)+"\n")
        summary=aggregate(rows)
        receipt={"study":STUDY,"summary":summary,"roots":rows}
        brief=summary
        if args.pilot_out:
            eligible=[r for r in rows if r.get("root_eligible",False)]
            if eligible:
                args.pilot_out.parent.mkdir(parents=True,exist_ok=True)
                args.pilot_out.write_text(json.dumps(
                    preflight_spec(eligible),sort_keys=True,
                    indent=2,allow_nan=False)+"\n")
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(
            receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps(brief,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
