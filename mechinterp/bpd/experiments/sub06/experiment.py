"""SUB-06: matched source-fitted wrong-clock causal quotient controls.

Read PROTOCOL.md. Source registration predates code and all fresh model data.
Exactly 8 independently trained engineering roots are allowed; no confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from mechinterp.bpd.experiments.sub01.experiment import (
    WIDTH, PASSES, Looper, sample_x, external_labels, pca_projectors,
)
from mechinterp.bpd.experiments.sub02.experiment import obs_projectors
from mechinterp.bpd.experiments.sub04.experiment import (
    source_joint_projector, linearized_source, physical_delta,
)
from mechinterp.bpd.experiments.sub05.experiment import (
    pass_output_covariances, bundle_loss, causal_response, clocked_student,
    gauge_checks, stationary_bundle, permuted_bundle, bundle_shams,
)

STUDY = "BPD-SUB-06"
ROOT = Path(__file__).resolve().parent
ROOTS = tuple(range(1700, 1708))
SMOKE = 1699
RANK = 8
TRAIN_STEPS = 256
TRAIN_N = 2048
DISC_N = 512
CHECK_N = 512
HELDOUT_N = 1024
FIT_STEPS = 120
FIT_LR = .02
OUTPUT_WEIGHT = .35
SMOOTH_WEIGHT = .50
MINIMUM_EFFECT = .010
BOOTSTRAP_DRAWS = 2000

# Exact seven maps committed in the original protocol, zero-based:
CLOCK_MAPS = (
    (1,2,3,0), (2,3,0,1), (3,0,1,2), (1,0,3,2),
    (2,1,0,3), (0,2,1,3), (3,2,1,0),
)
STEP_MAPS = (
    (1,2,0), (2,0,1), (0,1,2), (1,0,2),
    (2,1,0), (0,2,1), (2,1,0),
)
EVALMATH_COMMIT = "6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_model(seed: int, steps: int = TRAIN_STEPS):
    if seed != SMOKE and seed not in ROOTS:
        raise ValueError("SUB06 seed outside frozen engineering roster")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = Looper()
    x = sample_x(seed, 1, TRAIN_N)
    y = external_labels(x)
    rng = torch.Generator().manual_seed(811031 + seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(steps):
        idx = torch.randint(TRAIN_N, (128,), generator=rng)
        logits, _ = model.rollout(x[idx])
        loss = F.binary_cross_entropy_with_logits(logits, y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model.rollout(x)
        accuracy = float(((logits >= 0) == y.bool()).float().mean())
        density = float((model.hidden.weight.detach().abs() >= 1e-4)
                        .float().mean())
    return model, {
        "seed": seed, "steps": steps, "train_examples": TRAIN_N,
        "minibatch": 128, "optimizer": "Adam", "learning_rate": .01,
        "train_accuracy": accuracy, "shared_weight_density": density,
    }


def source_arrays(model: Looper, x: Tensor):
    """Same physical discovery tensors as SUB-05, once per source root."""
    transition = linearized_source(model, x).detach().double()
    covariances, gradient_norms = pass_output_covariances(model, x)
    return transition, covariances.detach().double(), gradient_norms


def schedule_arrays(d: Tensor, c: Tensor, arm: int):
    """Identity arm -1; otherwise fixed source-clock mislabelling."""
    if arm == -1:
        return d, c, (0,1,2), (0,1,2,3)
    if arm not in range(7):
        raise ValueError("Unregistered wrong-clock source fit")
    tau = STEP_MAPS[arm]
    pi = CLOCK_MAPS[arm]
    if sorted(tau) != [0,1,2] or sorted(pi) != [0,1,2,3]:
        raise AssertionError("Clock schedule is not a true permutation")
    return d[:, list(tau)], c[list(pi)], tau, pi


def fit_source_schedule(
    transition: Tensor,
    cov: Tensor,
    static_basis: Tensor,
    a0: float, b0: float,
    *,
    label: str,
):
    """Exact SUB-05 QR/Adam steps, fit to explicitly supplied source schedule.

    Output is fitted in native positions. Matching a0,b0, optimizer, initialization
    and data ensures comparisons differ only in the frozen source-clock schedules.
    """
    if transition.ndim != 4 or transition.shape[1:] != (3, WIDTH, WIDTH):
        raise ValueError("Invalid physical transition source tensor")
    if cov.shape != (PASSES, WIDTH, WIDTH):
        raise ValueError("Invalid output-covariance source tensor")
    if static_basis.shape != (WIDTH, RANK):
        raise ValueError("Invalid common stationary basis")
    basis = static_basis.detach().double()
    p_static = basis @ basis.T
    initial = torch.stack([p_static for _ in range(PASSES)])
    with torch.no_grad():
        ai, bi, si = bundle_loss(initial, transition, cov)
    parameters = nn.Parameter(basis.repeat(PASSES,1,1).clone())
    optimizer = torch.optim.Adam([parameters], lr=FIT_LR)
    history = []
    for step in range(FIT_STEPS):
        q, _ = torch.linalg.qr(parameters, mode="reduced")
        p = q @ q.transpose(-1,-2)
        a,b,s = bundle_loss(p, transition, cov)
        loss = a / a0 + OUTPUT_WEIGHT * b / b0 + SMOOTH_WEIGHT * s
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite wrong-clock source objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if parameters.grad is None or not torch.isfinite(parameters.grad).all():
            raise ValueError("Nonfinite wrong-clock QR gradient")
        optimizer.step()
        history.append({
            "step":step+1,
            "objective_before_update":float(loss.detach()),
            "closure_before_update":float(a.detach()),
            "output_before_update":float(b.detach()),
            "smoothness_before_update":float(s.detach()),
        })
    with torch.no_grad():
        q,_ = torch.linalg.qr(parameters, mode="reduced")
        p = q @ q.transpose(-1,-2)
        a,b,s = bundle_loss(p, transition, cov)
        final_loss = float(a/a0 + OUTPUT_WEIGHT*b/b0 + SMOOTH_WEIGHT*s)
        diag = {
            "arm":label,
            "optimizer_steps":FIT_STEPS,
            "optimizer_lr":FIT_LR,
            "output_weight":OUTPUT_WEIGHT,
            "smooth_weight":SMOOTH_WEIGHT,
            "initial_closure":float(ai),
            "initial_output":float(bi),
            "initial_smoothness":float(si),
            "initial_objective":float(ai/a0+OUTPUT_WEIGHT*bi/b0),
            "final_objective":final_loss,
            "final_closure":float(a),
            "final_output":float(b),
            "final_smoothness":float(s),
            "idempotence_max":float((p@p-p).abs().max()),
            "symmetry_max":float((p-p.transpose(-1,-2)).abs().max()),
            "rank_trace_max_error":max(abs(float(torch.trace(pt))-RANK)
                                        for pt in p),
            "distance_to_static_fro": [
                float((pt-p_static).norm()) for pt in p
            ],
            "source_loss_trace":history,
        }
    return p.float(), q.detach(), diag


def source_fit_family(model: Looper, discovery: Tensor, static_basis: Tensor):
    d,c,gradient_norms = source_arrays(model, discovery)
    initial = torch.stack([
        static_basis.double() @ static_basis.double().T
        for _ in range(PASSES)
    ])
    with torch.no_grad():
        aa,bb,_ = bundle_loss(initial,d,c)
        a0 = float(aa.clamp_min(1e-12))
        b0 = float(bb.clamp_min(1e-12))
    projectors, bases, diagnostics = {},{},{}
    for arm in (-1,*range(7)):
        ds,cs,tau,pi = schedule_arrays(d,c,arm)
        label = "aligned" if arm==-1 else f"wrong_{arm}"
        ps,q,diag=fit_source_schedule(ds,cs,static_basis,a0,b0,label=label)
        with torch.no_grad():
            original_a, original_b, original_s = bundle_loss(
                ps.double(),d,c)
            diag["true_clock_discovery_closure"] = float(original_a)
            diag["true_clock_discovery_output_loss"] = float(original_b)
            diag["true_clock_discovery_smoothness"] = float(original_s)
            diag["source_step_schedule"] = list(tau)
            diag["source_output_schedule"] = list(pi)
            diag["input_transition_count"] = int(d.shape[0])
            diag["input_future_output_count"] = int(c.shape[0])
            diag["normalizer_a0"] = a0
            diag["normalizer_b0"] = b0
        projectors[label]=ps
        bases[label]=q
        diagnostics[label]=diag
    return projectors,bases,diagnostics,c,{"a0":a0,"b0":b0,
        "gradient_norm_per_pass":gradient_norms,
        "source_discovery_size":len(discovery)}


def all_scores(model:Looper, heldout:Tensor,projectors:dict,alpha:float,seed:int,
               *, aligned_bases:Tensor, source_cov:Tensor,
               pca:Tensor, obs:Tensor):
    """Score predeclared fixed arms without any fit/selection based on outcomes."""
    aligned=projectors["aligned"]
    fixed={"stationary":projectors["stationary"],**projectors}
    for i,pi in enumerate(CLOCK_MAPS):
        fixed[f"postfit_perm_{i}"]=permuted_bundle(aligned, pi)
    angular=bundle_shams(seed,aligned_bases,source_cov)
    for i,z in enumerate(angular):
        fixed[f"angular_{i}"]=z["projectors"]
    fixed["pca"]=stationary_bundle(pca)
    fixed["obs"]=stationary_bundle(obs)
    scored={}
    students={}
    with torch.no_grad():
        for name,bundle in fixed.items():
            scored[name]=causal_response(model,heldout,bundle,alpha,seed,4)
            if name in ("aligned","stationary","pca","obs") or name.startswith("wrong_"):
                students[name]=clocked_student(model,heldout,bundle)
    return scored,students,[{
        "index":int(z["index"]),
        "kind":z["kind"],
        "theta":z["theta"],
        "geometry":z["geometry"],
    } for z in angular]


def evaluate_root(seed:int, *, steps:int=TRAIN_STEPS):
    model,training=train_model(seed,steps)
    x_discover=sample_x(seed,2,DISC_N)
    x_check=sample_x(seed,3,CHECK_N)
    x_heldout=sample_x(seed,4,HELDOUT_N)
    static,static_basis,_,static_diag=source_joint_projector(
        model,x_discover)
    ps,bases,source_fits,c_source,source_info=source_fit_family(
        model,x_discover,static_basis)
    ps["stationary"]=stationary_bundle(static)
    with torch.no_grad():
        _,disc_states=model.rollout(x_discover)
        alpha=.25*float(disc_states[0].double().square().sum(-1).mean().sqrt())
        pca,_,_=pca_projectors(model,x_discover)
        obs,_=obs_projectors(model,x_discover)
        scored,students,angular=all_scores(
            model,x_heldout,ps,alpha,seed,aligned_bases=bases["aligned"],
            source_cov=c_source,pca=pca[RANK],obs=obs[RANK])
        dose_checks={
            label:[
                float((physical_delta(seed,3,t,len(x_check),bundle[t],alpha)
                      .norm(dim=-1)-alpha).abs().max())
                for t in range(3)
            ] for label,bundle in (
                ("aligned",ps["aligned"]),("stationary",ps["stationary"]),
                *((f"wrong_{i}",ps[f"wrong_{i}"]) for i in range(7)),
            )
        }
        clean, _=model.rollout(x_heldout)
        y=external_labels(x_heldout)
        contexts={
            "x3_nonnegative":float(
                ((clean[x_heldout[:,3]>=0]>=0)==
                 y[x_heldout[:,3]>=0].bool()).float().mean()),
            "x3_negative":float(
                ((clean[x_heldout[:,3]<0]>=0)==
                 y[x_heldout[:,3]<0].bool()).float().mean()),
        }
        source_nll=float(F.binary_cross_entropy_with_logits(clean,y))
        full_student=clocked_student(
            model,x_heldout,stationary_bundle(torch.eye(WIDTH)))
        gauge={name:gauge_checks(model,x_heldout[:64],bundle,alpha,seed)
               for name,bundle in ps.items()}
    source_geometry_ok=all(
        d["optimizer_steps"]==FIT_STEPS
        and d["idempotence_max"]<=1e-5
        and d["symmetry_max"]<=1e-5
        and d["rank_trace_max_error"]<=1e-4
        and d["initial_objective"]==d["initial_objective"]
        for d in source_fits.values()
    ) and static_diag["projector_idempotence_max"]<=1e-5
    angular_geometry_ok=all(all(
        z["geometry"][key]<=1e-6
        for key in ("orthogonality_max","rank_trace_max_error",
                    "idempotence_max","symmetry_max"))
        for z in angular)
    dose_ok=all(
        row["max_physical_dose_error"]<=5e-6
        and row["max_null_impulse_projection"]<=5e-6
        for row in scored.values()) and all(
            v<=5e-6 for checks in dose_checks.values() for v in checks)
    gauge_ok=all(
        g["clean_logit_error"]<=5e-5
        and g["transported_next_state_error"]<=5e-5
        and g["transported_logit_error"]<=5e-5
        and max(g["wrong_untransported_logit_rms_by_pass"])>1e-5
        for g in gauge.values())
    parity_ok=(full_student["max_full_rank_parity"]<=5e-6
               and full_student["mean_clean_bernoulli_kl"]<=1e-7)
    competence=min(contexts.values())>=.80
    complete=(len(scored)==26 and len(students)==11
              and len(source_fits)==8 and len(angular)==8)
    eligible=bool(
        competence and parity_ok and source_geometry_ok
        and angular_geometry_ok and dose_ok and gauge_ok and complete)
    def closure(name):
        return scored[name]["pooled_next_state_rms"]
    primary_null=min([closure("stationary")]+[
        closure(f"wrong_{i}") for i in range(7)])
    primary=primary_null-closure("aligned")
    return {
        "study":STUDY,
        "phase":"EXCLUDED_SMOKE" if seed==SMOKE else "ENGINEERING",
        "root":seed,
        "protocol_sha256":file_sha(ROOT/"PROTOCOL.md"),
        "source_sha256":file_sha(Path(__file__)),
        "dependency_sub04_sha256":file_sha(
            ROOT.parent/"sub04"/"experiment.py"),
        "dependency_sub05_sha256":file_sha(
            ROOT.parent/"sub05"/"experiment.py"),
        "environment":{"python":platform.python_version(),
            "torch":torch.__version__,"numpy":np.__version__},
        "training":training,
        "data_sizes":{"train":TRAIN_N,"discovery":DISC_N,
            "selection":CHECK_N,"heldout":HELDOUT_N},
        "source":{"nll":source_nll,"contexts":contexts},
        "alpha":alpha,
        "stationary_source":static_diag,
        "fitted_source_diagnostics":source_fits,
        "shared_source_info":source_info,
        "wrong_clock_source_schedules":[{
            "output":list(pi),"transition":list(tau)}
            for pi,tau in zip(CLOCK_MAPS,STEP_MAPS)],
        "scores":scored,
        "students":students,
        "angular_controls":angular,
        "dose_checks":dose_checks,
        "gauge":gauge,
        "full_rank_parity":full_student["max_full_rank_parity"],
        "full_rank_kl":full_student["mean_clean_bernoulli_kl"],
        "competent":competence,
        "source_geometry_ok":source_geometry_ok,
        "angular_geometry_ok":angular_geometry_ok,
        "dose_ok":dose_ok,"gauge_ok":gauge_ok,
        "control_roster_ok":complete,
        "root_eligible":eligible,
        "primary_aligned":closure("aligned"),
        "primary_strongest_fitted_null":primary_null,
        "primary_strongest_null_minus_aligned":primary,
        "stationary_minus_aligned":closure("stationary")-closure("aligned"),
        "best_wrong_fitted_minus_aligned":min(
            closure(f"wrong_{i}") for i in range(7))-closure("aligned"),
        "postfit_strongest_minus_aligned":min(
            closure(f"postfit_perm_{i}") for i in range(7))-closure("aligned"),
    }


def bootstrap(values:list[float]):
    if not values:
        return None
    data=np.asarray(values,dtype=float)
    rng=np.random.default_rng(170606)
    samples=rng.choice(data,size=(BOOTSTRAP_DRAWS,len(data)),replace=True).mean(1)
    return [float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]


def bounded(value:float):
    return value/(1.+abs(value))


def preflight_spec(rows:list[dict]):
    return {
        "schema_version":1,
        "experiment_id":STUDY,
        "independent_unit":"independently trained model root",
        "alpha":"0.05",
        "scientific_min_effect":"0.009900990099009901",
        "target_effect":"0.024390243902439025",
        "measurement_resolution":"0.002",
        "sham_quantile":"0.95",
        "sham_confidence":"0.95",
        "planned_comparisons":1,
        "confirm_units_planned":64,
        "pilot_from_development_only":True,
        "heldout_sealed":True,
        "units_independent":True,
        "pilot":[{
            "unit_id":f"root-{r['root']}",
            "paired_effect":repr(bounded(
                r["primary_strongest_null_minus_aligned"])),
            "sham_effects":[repr(bounded(max(0.,
                r["scores"]["stationary"]["pooled_next_state_rms"]-
                r["scores"][f"wrong_{i}"]["pooled_next_state_rms"]
            ))) for i in range(7)],
        } for r in rows],
    }


def aggregate(rows:list[dict]):
    if [r["root"] for r in rows]!=list(ROOTS):
        raise ValueError("Must retain exactly 8 registered independent source roots")
    admitted=[r for r in rows if r.get("root_eligible",False)]
    effects=[r["primary_strongest_null_minus_aligned"] for r in admitted]
    mean=float(np.mean(effects)) if effects else None
    ci=bootstrap(effects) if len(admitted)==8 else None
    positive=sum(z>0 for z in effects)
    def mean_of(fn):
        return float(np.mean([fn(r) for r in admitted])) if admitted else None
    student_kl=mean_of(
        lambda r:r["students"]["aligned"]["mean_clean_bernoulli_kl"])
    excess_kl=mean_of(lambda r:
        r["students"]["aligned"]["mean_clean_bernoulli_kl"]-
        r["students"]["stationary"]["mean_clean_bernoulli_kl"])
    excess_nll=mean_of(lambda r:
        r["students"]["aligned"]["true_nll"]-
        r["students"]["stationary"]["true_nll"])
    excess_logit=mean_of(lambda r:
        r["scores"]["aligned"]["pooled_terminal_logit_rms"]-
        r["scores"]["stationary"]["pooled_terminal_logit_rms"])
    engineering=bool(
        len(admitted)==8 and mean>=MINIMUM_EFFECT
        and positive>=7 and ci[0]>0
        and student_kl<=.040 and excess_kl<=.015
        and excess_nll<=.030 and excess_logit<=.040)
    return {
        "study":STUDY,
        "phase":"ENGINEERING",
        "attempted_roots":len(rows),
        "root_ids":list(ROOTS),
        "eligible_roots":len(admitted),
        "ineligible_roots":[r["root"] for r in rows
                            if not r.get("root_eligible",False)],
        "registered_primary":"min_static_seven_fitted_wrongclock_minus_aligned",
        "registered_mean_effect_minimum":MINIMUM_EFFECT,
        "root_primary_effects":effects,
        "mean_primary_effect":mean,
        "whole_root_bootstrap95":ci,
        "positive_roots":positive,
        "mean_aligned_closure":mean_of(lambda r:r["primary_aligned"]),
        "mean_stationary_closure":mean_of(lambda r:
            r["scores"]["stationary"]["pooled_next_state_rms"]),
        "mean_best_fitted_wrong_closure":mean_of(lambda r:min(
            r["scores"][f"wrong_{i}"]["pooled_next_state_rms"]
            for i in range(7))),
        "mean_best_postfit_wrong_closure":mean_of(lambda r:min(
            r["scores"][f"postfit_perm_{i}"]["pooled_next_state_rms"]
            for i in range(7))),
        "mean_strongest_fitted_null":mean_of(lambda r:
            r["primary_strongest_fitted_null"]),
        "mean_stationary_minus_aligned":mean_of(
            lambda r:r["stationary_minus_aligned"]),
        "mean_fitted_wrong_minus_aligned":mean_of(
            lambda r:r["best_wrong_fitted_minus_aligned"]),
        "mean_aligned_clean_kl":student_kl,
        "mean_aligned_minus_static_clean_kl":excess_kl,
        "mean_aligned_minus_static_true_nll":excess_nll,
        "mean_aligned_minus_static_logit_response":excess_logit,
        "independent_95_95_null_tail_calibrated":False,
        "evalmath_strict":"PENDING_PILOT_ONLY_PREFLIGHT",
        "future_calibration":"UNGENERATED_SEALED",
        "future_confirmation":"UNGENERATED_SEALED",
        "verdict":"ENGINEERING_SIGNAL_ONLY" if engineering
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
        data=evaluate_root(SMOKE,steps=16)
        short={"phase":"EXCLUDED_SMOKE","root":SMOKE,
               "root_eligible":data["root_eligible"],
               "primary":data["primary_strongest_null_minus_aligned"]}
    else:
        roots=[]
        for seed in ROOTS:
            try:
                roots.append(evaluate_root(seed))
            except Exception as exc:
                roots.append({
                    "study":STUDY,"phase":"ENGINEERING","root":seed,
                    "root_eligible":False,"error_type":type(exc).__name__,
                    "error_message":str(exc),
                })
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(
                    {"study":STUDY,"phase":"PARTIAL",
                     "completed_roots":len(roots),"roots":roots},
                    indent=2,sort_keys=True,allow_nan=False)+"\n")
        stats=aggregate(roots)
        data={"study":STUDY,"summary":stats,"roots":roots}
        short=stats
        if args.pilot_out:
            selected=[r for r in roots if r.get("root_eligible",False)]
            if selected:
                args.pilot_out.parent.mkdir(parents=True,exist_ok=True)
                args.pilot_out.write_text(json.dumps(
                    preflight_spec(selected),indent=2,
                    sort_keys=True,allow_nan=False)+"\n")
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(data,indent=2,
                                          sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(short,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
