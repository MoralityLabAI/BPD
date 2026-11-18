"""SUB-07: exact-oracle driven GRU, matched fitted wrong-clock causal experiment.

The protocol and the exact finite-sequence source were committed before data.
Only 8 prospective model roots 1800..1807; 1799 is excluded source smoke.
Scientific confirmation and 59-root null calibration are unavailable here.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .oracle import exact_task_certificate
from .source import (
    WIDTH, RANK, PASSES, ROOTS, SMOKE, TRAIN_STEPS, FIT_STEPS,
    DISCOVERY_JACOBIAN_ROWS, CLOCK_MAPS, STEP_MAPS,
    train_source, tensor_split, discovered_source, stationary_source,
    source_fit_family, digest, 
)

STUDY = "BPD-SUB-07"
ROOT = Path(__file__).resolve().parent
MINIMUM_EFFECT = .010
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 180707
EVALMATH_COMMIT="6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a"


def physical_delta(seed:int,split:int,pass_index:int,n:int,
                   p:Tensor,alpha:float):
    rng=torch.Generator().manual_seed(
        807000+seed*4001+split*17011+pass_index*101)
    raw=torch.randn(n,WIDTH,generator=rng)
    null=torch.eye(WIDTH)-p
    v=raw@null
    lengths=v.norm(dim=-1,keepdim=True)
    if float(lengths.min())<1e-8:
        raise ValueError("Degenerate projected physical impulse")
    return alpha*v/lengths


def fixed_stationary(p:Tensor):
    if p.shape!=(WIDTH,WIDTH):
        raise ValueError("Wrong stationary physical projector")
    return torch.stack([p for _ in range(PASSES)])


@torch.no_grad()
def source_causal_response(model,x:Tensor,y:Tensor,p:Tensor,alpha:float,
                           seed:int,split:int):
    if p.shape!=(PASSES,WIDTH,WIDTH):
        raise ValueError("Expected six registered clocked projectors")
    teacher,states=model.rollout(x)
    correct=(teacher>=0)==y.bool()
    step_rms,terminal_rms,terminal_state=[],[],[]
    changed_radius,clean_radius,corrupt,repair=[],[],[],[]
    max_dose,max_null=0.,0.
    for t in range(PASSES-1):
        d=physical_delta(seed,split,t,len(x),p[t],alpha)
        max_dose=max(max_dose,float((d.norm(dim=-1)-alpha).abs().max()))
        max_null=max(max_null,float((d@p[t]).norm(dim=-1).max()))
        next_clean=model.step(states[t],x[:,t+1])
        next_shift=model.step(states[t]+d,x[:,t+1])
        projected=(next_shift-next_clean)@p[t+1]
        step_rms.append(float(
            projected.double().square().sum(-1).mean().sqrt()))
        clean_radius.append(float(
            next_clean.double().norm(dim=-1).mean()))
        changed_radius.append(float(
            next_shift.double().norm(dim=-1).mean()))
        h=states[t]+d
        for j in range(t+1,PASSES):
            h=model.step(h,x[:,j])
        shifted_logits=model.head(h).squeeze(-1)
        terminal_rms.append(float(
            (shifted_logits-teacher).double().square().mean().sqrt()))
        terminal_state.append(float(
            (h-states[-1]).double().square().sum(-1).mean().sqrt()))
        shifted_correct=(shifted_logits>=0)==y.bool()
        corrupt.append(int((correct & ~shifted_correct).sum()))
        repair.append(int((~correct & shifted_correct).sum()))
    return {
        "per_pass_projected_next_state_rms":step_rms,
        "pooled_next_state_rms":float(
            np.sqrt(np.mean(np.square(step_rms)))),
        "per_pass_terminal_logit_rms":terminal_rms,
        "pooled_terminal_logit_rms":float(
            np.sqrt(np.mean(np.square(terminal_rms)))),
        "per_pass_terminal_state_displacement_rms":terminal_state,
        "per_pass_clean_next_state_mean_norm":clean_radius,
        "per_pass_altered_next_state_mean_norm":changed_radius,
        "per_pass_correct_corrupted":corrupt,
        "per_pass_wrong_repaired":repair,
        "physical_alpha":alpha,
        "max_physical_dose_error":max_dose,
        "max_null_projection_norm":max_null,
    }


@torch.no_grad()
def clocked_student(model,x:Tensor,y:Tensor,projectors:Tensor):
    h=x.new_zeros(len(x),WIDTH)
    for t in range(PASSES):
        h=model.step(h,x[:,t])@projectors[t]
    logits=model.head(h).squeeze(-1)
    teacher,_=model.rollout(x)
    p=torch.sigmoid(teacher)
    kl=(F.softplus(logits)-F.softplus(teacher)
        -p*(logits-teacher)).clamp_min(0.)
    return {
        "clean_to_student_kl":float(kl.mean()),
        "true_label_nll":float(F.binary_cross_entropy_with_logits(logits,y)),
        "accuracy":float(((logits>=0)==y.bool()).float().mean()),
        "correct_count":int(((logits>=0)==y.bool()).sum()),
        "full_head_max_abs_error":float((logits-teacher).abs().max()),
    }


@torch.no_grad()
def gauge_transport(model,x:Tensor,p:Tensor,alpha:float,seed:int):
    """Physical orthogonal gauge with correct next-token transport every pass."""
    g=torch.linalg.qr(
        torch.randn(WIDTH,WIDTH,
            generator=torch.Generator().manual_seed(921007+seed),
            dtype=torch.float64)
    )[0].float()
    gt=g.T
    source,states=model.rollout(x)
    z=x.new_zeros(len(x),WIDTH)
    for j in range(PASSES):
        z=model.step(z@g,x[:,j])@gt
    clean_error=float((model.head(z@g).squeeze(-1)-source).abs().max())
    state_error,logit_error=0.,0.
    wrong=[]
    for t in range(PASSES-1):
        impulse=physical_delta(seed,4,t,len(x),p[t],alpha)
        h0=model.step(states[t],x[:,t+1])
        h1=model.step(states[t]+impulse,x[:,t+1])
        physical_step=(h1-h0)@p[t+1]
        gp=g@p[t+1]@gt
        z0=model.step((states[t]@gt)@g,x[:,t+1])@gt
        z1=model.step((states[t]@gt+impulse@gt)@g,
                      x[:,t+1])@gt
        gauge_step=(z1-z0)@gp
        state_error=max(state_error,float(
            (gauge_step@g-physical_step).abs().max()))
        physical=states[t]+impulse
        for j in range(t+1,PASSES):
            physical=model.step(physical,x[:,j])
        original=model.head(physical).squeeze(-1)
        zr=states[t]@gt+impulse@gt
        for j in range(t+1,PASSES):
            zr=model.step(zr@g,x[:,j])@gt
        projected=model.head(zr@g).squeeze(-1)
        logit_error=max(logit_error,float(
            (original-projected).abs().max()))
        z_wrong=states[t]@gt+impulse
        for j in range(t+1,PASSES):
            z_wrong=model.step(z_wrong@g,x[:,j])@gt
        wrong.append(float(
            (model.head(z_wrong@g).squeeze(-1)-original)
            .double().square().mean().sqrt()))
    return {
        "orthogonality_error":float(
            (gt@g-torch.eye(WIDTH)).abs().max()),
        "clean_logit_error":clean_error,
        "next_state_transported_max_error":state_error,
        "terminal_logit_transported_max_error":logit_error,
        "wrong_untransported_logit_rms":wrong,
    }


@torch.no_grad()
def task_competence(model,x:Tensor,y:Tensor):
    logits,states=model.rollout(x)
    correct=((logits>=0)==y.bool())
    tokens=x.argmax(dim=-1)
    last= tokens[:,-1]
    context={f"last_token_{t}":float(correct[last==t].float().mean())
             for t in range(4)}
    context["label_0"]=float(correct[y==0].float().mean())
    context["label_1"]=float(correct[y==1].float().mean())
    prefix2=model.head(states[1]).squeeze(-1)
    prefix4=model.head(states[3]).squeeze(-1)
    return {
        "accuracy":float(correct.float().mean()),
        "true_label_nll":float(F.binary_cross_entropy_with_logits(logits,y)),
        "contexts":context,
        "prefix2_source_head_accuracy":float(
            ((prefix2>=0)==y.bool()).float().mean()),
        "prefix4_source_head_accuracy":float(
            ((prefix4>=0)==y.bool()).float().mean()),
        "positive_rate":float(y.mean()),
    }


def score_root(seed:int,steps:int=TRAIN_STEPS):
    if seed not in ROOTS and seed!=SMOKE:
        raise ValueError("Unregistered SUB-07 root")
    oracle_certificate=exact_task_certificate()
    model,training=train_source(seed,steps)
    xd,yd,discovery_ids=tensor_split("discovery")
    xc,yc,dose_ids=tensor_split("dose")
    xh,yh,heldout_ids=tensor_split("heldout")
    # Freeze source geometry and every candidate on discovery-only inputs.
    d,c,grad_norms=discovered_source(model,xd)
    pstatic,static_basis,static_info=stationary_source(d,c)
    fitted,base,source_fits,common=source_fit_family(d,c,static_basis)
    stationary=fixed_stationary(pstatic)
    controls={"aligned":fitted["aligned"],"stationary":stationary}
    controls.update({f"wrong_{i}":fitted[f"wrong_{i}"] for i in range(7)})
    controls.update({
        f"postfit_perm_{i}":fitted["aligned"][list(pi)]
        for i,pi in enumerate(CLOCK_MAPS)
    })
    with torch.no_grad():
        _,discovery_states=model.rollout(xd)
        alpha=.25*float(
            discovery_states[0].double().square().sum(-1).mean().sqrt())
        dose_checks={
            key:[
                float((physical_delta(seed,3,t,len(xc),p[t],alpha)
                       .norm(dim=-1)-alpha).abs().max())
                for t in range(PASSES-1)
            ] for key,p in controls.items()
            if key=="stationary" or key=="aligned" or key.startswith("wrong_")
        }
        scored={
            name:source_causal_response(model,xh,yh,p,alpha,seed,4)
            for name,p in controls.items()
        }
        students={
            name:clocked_student(model,xh,yh,p)
            for name,p in controls.items()
            if name=="stationary" or name=="aligned" or name.startswith("wrong_")
        }
        full_head=clocked_student(
            model,xh,yh,fixed_stationary(torch.eye(WIDTH)))
        competence=task_competence(model,xh,yh)
        gauges={name:gauge_transport(model,xh[:64],p,alpha,seed)
                for name,p in controls.items()
                if name=="stationary" or name=="aligned" or name.startswith("wrong_")}
    source_ok=all(
        diag["optimizer_steps"]==FIT_STEPS
        and len(diag["source_loss_trace"])==FIT_STEPS
        and diag["rank_trace_max_error"]<=1e-4
        and diag["idempotence_max"]<=1e-5
        and diag["symmetry_max"]<=1e-5
        for diag in source_fits.values()
    ) and static_info["optimizer_steps"]==80 and (
        static_info["idempotence_max"]<=1e-5)
    dose_ok=all(
        item["max_physical_dose_error"]<=5e-6
        and item["max_null_projection_norm"]<=5e-6
        for item in scored.values()
    ) and all(
        d<=5e-6 for checks in dose_checks.values() for d in checks)
    gauge_ok=all(
        z["clean_logit_error"]<=5e-5
        and z["next_state_transported_max_error"]<=5e-5
        and z["terminal_logit_transported_max_error"]<=5e-5
        and max(z["wrong_untransported_logit_rms"])>1e-5
        for z in gauges.values()
    )
    parity_ok=(
        full_head["full_head_max_abs_error"]<=5e-6
        and full_head["clean_to_student_kl"]<=1e-7)
    competent=(
        competence["accuracy"]>=.90
        and min(competence["contexts"].values())>=.85
    )
    complete=(len(controls)==16 and len(scored)==16
              and len(students)==9 and len(gauges)==9
              and len(source_fits)==8)
    eligible=bool(
        competent and complete and source_ok and dose_ok
        and gauge_ok and parity_ok and alpha>0)
    C=lambda key:scored[key]["pooled_next_state_rms"]
    strongest=min([C("stationary")]+[C(f"wrong_{i}") for i in range(7)])
    return {
        "study":STUDY,
        "phase":"EXCLUDED_SMOKE" if seed==SMOKE else "ENGINEERING",
        "root":seed,
        "protocol_sha256":digest(ROOT/"PROTOCOL.md"),
        "source_sha256":digest(Path(__file__)),
        "oracle_sha256":digest(ROOT/"oracle.py"),
        "fitting_source_sha256":digest(ROOT/"source.py"),
        "environment":{
            "python":platform.python_version(),
            "torch":torch.__version__,"numpy":np.__version__,
        },
        "training":training,
        "data_sizes":{
            "train":2048,"discovery":512,"dose":512,"heldout":1024,
            "unique_sequences_full_universe":4096,
        },
        "oracle_certificate":oracle_certificate,
        "source_competence":competence,
        "discovery_gradient_norms":grad_norms,
        "source_jacobian_rows":DISCOVERY_JACOBIAN_ROWS,
        "stationary_source":static_info,
        "source_fits":source_fits,
        "shared_source_normalizers":common,
        "clock_maps":[list(x) for x in CLOCK_MAPS],
        "step_maps":[list(x) for x in STEP_MAPS],
        "physical_alpha":alpha,
        "dose_checks":dose_checks,
        "scores":scored,
        "students":students,
        "gauges":gauges,
        "full_rank_max_logit_error":full_head["full_head_max_abs_error"],
        "full_rank_clean_kl":full_head["clean_to_student_kl"],
        "source_geometry_ok":source_ok,
        "dose_ok":dose_ok,
        "gauge_ok":gauge_ok,
        "competent":competent,
        "complete_roster":complete,
        "root_eligible":eligible,
        "strongest_control":strongest,
        "aligned_closure":C("aligned"),
        "stationary_closure":C("stationary"),
        "best_fitted_wrong_closure":min(C(f"wrong_{i}") for i in range(7)),
        "best_postfit_wrong_closure":min(C(f"postfit_perm_{i}") for i in range(7)),
        "primary_strongest_minus_aligned":strongest-C("aligned"),
        "stationary_minus_aligned":C("stationary")-C("aligned"),
        "fitted_wrong_minus_aligned":min(
            C(f"wrong_{i}") for i in range(7))-C("aligned"),
    }


def bootstrap(values:list[float]):
    if not values: return None
    a=np.asarray(values)
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    draws=rng.choice(a,size=(BOOTSTRAPS,len(a)),replace=True).mean(axis=1)
    return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]


def bounded(v:float)->float:
    return v/(1+abs(v))


def preflight_spec(records:list[dict]):
    return {
        "schema_version":1,"experiment_id":STUDY,
        "independent_unit":"independently trained model root",
        "alpha":"0.05","scientific_min_effect":"0.009900990099009901",
        "target_effect":"0.024390243902439025",
        "measurement_resolution":"0.002",
        "sham_quantile":"0.95","sham_confidence":"0.95",
        "planned_comparisons":1,"confirm_units_planned":64,
        "pilot_from_development_only":True,"heldout_sealed":True,
        "units_independent":True,
        "pilot":[{
            "unit_id":f"root-{r['root']}",
            "paired_effect":repr(bounded(r["primary_strongest_minus_aligned"])),
            "sham_effects":[repr(bounded(max(0.,
                r["scores"]["stationary"]["pooled_next_state_rms"]-
                r["scores"][f"wrong_{i}"]["pooled_next_state_rms"])))
                for i in range(7)],
        } for r in records],
    }


def aggregate(rows:list[dict]):
    if [r["root"] for r in rows]!=list(ROOTS):
        raise ValueError("Exactly eight registered source roots required")
    eligible=[r for r in rows if r.get("root_eligible",False)]
    effects=[r["primary_strongest_minus_aligned"] for r in eligible]
    mean=float(np.mean(effects)) if effects else None
    ci=bootstrap(effects) if len(eligible)==8 else None
    pos=sum(v>0 for v in effects)
    def m(fn):
        return float(np.mean([fn(r) for r in eligible])) if eligible else None
    kl=m(lambda r:r["students"]["aligned"]["clean_to_student_kl"])
    excess_kl=m(lambda r:
        r["students"]["aligned"]["clean_to_student_kl"]-
        r["students"]["stationary"]["clean_to_student_kl"])
    excess_nll=m(lambda r:
        r["students"]["aligned"]["true_label_nll"]-
        r["students"]["stationary"]["true_label_nll"])
    excess_logit=m(lambda r:
        r["scores"]["aligned"]["pooled_terminal_logit_rms"]-
        r["scores"]["stationary"]["pooled_terminal_logit_rms"])
    allowed=bool(len(eligible)==8 and mean>=MINIMUM_EFFECT
        and pos>=7 and ci[0]>0 and kl<=.04
        and excess_kl<=.015 and excess_nll<=.030 and excess_logit<=.040)
    return {
        "study":STUDY,"phase":"ENGINEERING",
        "root_ids":list(ROOTS),"attempted_roots":len(rows),
        "eligible_roots":len(eligible),
        "ineligible_roots":[r["root"] for r in rows
             if not r.get("root_eligible",False)],
        "registered_primary":"min_stationary_seven_source_fitted_wrongclock_minus_aligned",
        "registered_minimum_effect":MINIMUM_EFFECT,
        "root_primary_effects":effects,
        "mean_primary_effect":mean,
        "bootstrap95_whole_roots":ci,
        "positive_roots":pos,
        "mean_aligned_closure":m(lambda r:r["aligned_closure"]),
        "mean_static_closure":m(lambda r:r["stationary_closure"]),
        "mean_best_fitted_wrong_closure":m(lambda r:r["best_fitted_wrong_closure"]),
        "mean_best_postfit_wrong_closure":m(lambda r:r["best_postfit_wrong_closure"]),
        "mean_aligned_student_kl":kl,
        "mean_aligned_minus_static_student_kl":excess_kl,
        "mean_aligned_minus_static_true_nll":excess_nll,
        "mean_aligned_minus_static_terminal_response":excess_logit,
        "scientific_sham_tail_calibrated":False,
        "evalmath":"PENDING_STRICT_PILOT_ONLY_PREFLIGHT",
        "future_calibration":"UNGENERATED_SEALED",
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
    cert=exact_task_certificate()
    if args.phase=="development" and not args.ack_development:
        parser.error("Development requires explicit --ack-development")
    if args.phase=="smoke":
        out=score_root(SMOKE,steps=16)
        brief={
            "phase":"EXCLUDED_SMOKE","root":SMOKE,
            "task_certificate":cert["positives"],
            "competence":out["source_competence"]["accuracy"],
            "primary":out["primary_strongest_minus_aligned"],
        }
    else:
        roots=[]
        for seed in ROOTS:
            try:
                roots.append(score_root(seed))
            except Exception as exc:
                roots.append({
                    "study":STUDY,"phase":"ENGINEERING","root":seed,
                    "root_eligible":False,
                    "error_type":type(exc).__name__,
                    "error_message":str(exc),
                })
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps({
                    "study":STUDY,"phase":"PARTIAL",
                    "completed_roots":len(roots),"roots":roots,
                },sort_keys=True,indent=2,allow_nan=False)+"\n")
        summary=aggregate(roots)
        out={"study":STUDY,"summary":summary,"roots":roots}
        brief=summary
        if args.pilot_out:
            admitted=[r for r in roots if r.get("root_eligible",False)]
            if admitted:
                args.pilot_out.parent.mkdir(parents=True,exist_ok=True)
                args.pilot_out.write_text(json.dumps(
                    preflight_spec(admitted),sort_keys=True,
                    indent=2,allow_nan=False)+"\n")
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(
            out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps(brief,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
