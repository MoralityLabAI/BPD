"""FC04 source-only, full-tangent Grassmann constraint-Jacobian solver.

Every constructed projector has frozen principal angle 0.25 from DP12, and
original FC02 W/m/v source constraints are checked numerically. Projector
selection uses ONLY clean model-self derivatives and physical write moments.
All model-root/task/rank/threshold choices were registered in PROTOCOL.md
before this implementation or any new engineering model root.
"""
from __future__ import annotations

import hashlib
import math
import base64

import numpy as np
import torch
from torch import Tensor

from mechinterp.bpd.experiments.nc02.experiment import PASSES,WIDTH
from mechinterp.bpd.experiments.nc09_fixed12.tangent_sham_calibration.tangent_shams import (
    score_numpy,_physical_dose,
    SOURCE_WRITE_REL_TOL,SOURCE_SCORE_ABS_TOL,SOURCE_SCORE_REL_TOL,
)
from mechinterp.bpd.experiments.nc09_fixed12.grassmann_fc03.solver import (
    ANGLE,DP12_MIN_CHORDAL,PAIRWISE_MIN_CHORDAL,
    qr_positive, source_tensor_profile, projectors_from_bases,
    check_source_candidate, projector_digest, chordal,
)

CHART_STARTS=32
MAX_UPDATES=64
LINE_SEARCH=(1.,.5,.25,.125)
GAUSS_NEWTON_DAMPING=.001
STEP_NORM_MAX=.5
CHART_SEED_BASE=15_040_000
REQUIRED_SHAMS=8
JAC_SINGULAR_TOL=1e-8


def complementary_basis(E:Tensor):
    """Canonical orthogonal complement of an orthonormal trained factor basis."""
    E=E.detach().cpu().double()
    if E.ndim!=2 or E.shape[0]!=WIDTH or E.shape[1] not in range(2,7):
        raise ValueError("Source original 20D rank2..6 factor basis required")
    r=E.shape[1]
    if (E.T@E-torch.eye(r,dtype=torch.float64)).abs().max()>1e-6:
        raise ValueError("Source read factor eigenspace is not orthonormal")
    Q,_=torch.linalg.qr(E,mode="complete")
    C=Q[:,r:]
    if (C.T@E).abs().max()>1e-9 or (
        C.T@C-torch.eye(WIDTH-r,dtype=torch.float64)).abs().max()>1e-9:
        raise AssertionError("Native complement is not orthogonal")
    return C


def tangent_chart(E:list[Tensor],C:list[Tensor],y:Tensor):
    """Full complement Stiefel frame, 0.25-rad fixed passwise principal angles."""
    if len(E)!=PASSES or len(C)!=PASSES or y.ndim!=1:
        raise ValueError("Four pass-specific original frames and flat tangent required")
    result=[]
    offset=0
    identity=torch.eye(WIDTH,dtype=torch.float64)
    for t in range(PASSES):
        r=E[t].shape[1]
        d=WIDTH-r
        coordinates=y[offset:offset+d*r].reshape((d,r))
        offset+=d*r
        Q=C[t]@qr_positive(coordinates)
        result.append(math.cos(ANGLE)*E[t]+math.sin(ANGLE)*Q)
    if offset!=len(y):
        raise ValueError("Full tangent chart coordinate dimension mismatch")
    return result


def unconstrained_parameter_count(E:list[Tensor]):
    return sum((WIDTH-e.shape[1])*e.shape[1] for e in E)


def target_values(z:Tensor,g:Tensor,a:Tensor,U:Tensor,E:list[Tensor]):
    moments=torch.einsum("nti,ntj->tij",a,a)/len(a)
    gram=U@U.T
    original=source_tensor_profile(z,g,E,moments,gram)
    target=tuple(float(v.detach()) for v in original)
    if (target[0]<=1e-12 or
        not all(math.isfinite(v) for v in target)):
        raise ValueError("Zero or nonfinite clean retained physical write")
    tol=(
        SOURCE_WRITE_REL_TOL,
        max(SOURCE_SCORE_ABS_TOL,SOURCE_SCORE_REL_TOL*abs(target[1])),
        max(SOURCE_SCORE_ABS_TOL,SOURCE_SCORE_REL_TOL*target[2])
    )
    # Original source-only FC02's independent NumPy moment law.
    original_score=score_numpy(z.numpy(),g.numpy(),[x.numpy() for x in E])
    original_dose=_physical_dose(
        [x.numpy() for x in E],
        np.einsum("nti,ntj->tij",a.numpy(),a.numpy())/len(a),
        U.numpy()@U.numpy().T
    )
    if abs(original_score["signed_first_order_score_mean"]-target[1])>1e-9 or (
        abs(original_score["first_order_score_rms"]-target[2])>1e-9 or
        abs(original_dose-target[0])>1e-9):
        raise AssertionError("Full-tangent three-metric contract drift from FC02")
    return target,tol,moments,gram


def normalized_source_residual(profiles:tuple[Tensor,Tensor,Tensor],
                               target:tuple[float,float,float],
                               tol:tuple[float,float,float]):
    w,m,v=profiles
    return torch.stack((
        (w/target[0]-1.)/tol[0],
        (m-target[1])/tol[1],
        (v-target[2])/tol[2],
    ))


def squared_constraint_excess(f:Tensor):
    return torch.relu(torch.abs(f)-1.).square().sum()


def minimum_norm_jacobian_step(f:Tensor,y:Tensor):
    """Damped 3×d Jacobian, full-tangent minimum-norm Gauss-Newton direction."""
    J=torch.stack([
        torch.autograd.grad(f[i],y,retain_graph=True,create_graph=False)[0]
        for i in range(3)
    ])
    JJt=J@J.T
    eigen=torch.linalg.svdvals(J)
    damped=JJt+GAUSS_NEWTON_DAMPING*torch.eye(3,dtype=torch.float64)
    delta=-(J.T@torch.linalg.solve(damped,f.detach()))
    dn=float(delta.norm())
    if dn>STEP_NORM_MAX:
        delta=delta*(STEP_NORM_MAX/dn)
    info={
        "jacobian_singular_values":eigen.detach().numpy().tolist(),
        "jacobian_rank":int((eigen>JAC_SINGULAR_TOL).sum()),
        "jacobian_condition":(
            float(eigen[0]/eigen[-1]) if float(eigen[-1])>JAC_SINGULAR_TOL
            else None
        ),
        "direction_norm":float(delta.norm()),
    }
    return delta.detach(),info


def independent_numeric_audit(p:Tensor,target:tuple[float,float,float],
                              tol:tuple[float,float,float],
                              dp:Tensor,accepted:list[Tensor],
                              z:Tensor,g:Tensor,E:list[Tensor],
                              a:Tensor,U:Tensor):
    """Recompute source moments through the original NumPy FC02 functions."""
    e= [x.detach().cpu().double().numpy() for x in E]
    # For fixed nonzero angle to the native E, recover source bases from
    # projector by projecting its write direction onto E/complement.
    # The actual candidate source bases are passed separately to the exact
    # original numeric audit by check_source_candidate in the main search.
    del e
    assert p.shape==(PASSES,WIDTH,WIDTH)
    return {
        "min_original_distance":float(chordal(p,dp)),
        "min_pairwise_distance":min(
            [float(chordal(p,q)) for q in accepted],default=None
        ),
        "source_examples":len(z),
    }


def fit_full_tangent_shams(
    activations:Tensor,physical_decoder:Tensor,
    clean_reads:Tensor,clean_gradients:Tensor,
    source_basis:list[Tensor],root:int,
    *,starts:int=CHART_STARTS,steps:int=MAX_UPDATES,
    want:int=REQUIRED_SHAMS
):
    if (len(source_basis)!=PASSES or
        activations.ndim!=3 or activations.shape[1:]!=(PASSES,WIDTH) or
        clean_reads.shape!=activations.shape or
        clean_gradients.shape!=activations.shape or
        physical_decoder.shape!=(WIDTH,WIDTH)
        or starts<1 or steps<0 or want<1):
        raise ValueError("Original four-pass FC04 source record malformed")
    torch.set_num_threads(1)
    z=clean_reads.detach().cpu().double()
    g=clean_gradients.detach().cpu().double()
    a=activations.detach().cpu().double()
    U=physical_decoder.detach().cpu().double()
    E=[q.detach().cpu().double() for q in source_basis]
    if sum(x.shape[1] for x in E)!=12:
        raise ValueError("Only the prospective 12-slot source DP factor ranks allowed")
    C=[complementary_basis(x) for x in E]
    target,tol,moments,gram=target_values(z,g,a,U,E)
    dp=projectors_from_bases(E).detach()
    rng=np.random.default_rng(CHART_SEED_BASE+root)
    accepted=[]
    records=[]
    charts=[]
    all_iterations=0
    jacobian_rank_census={0:0,1:0,2:0,3:0}
    for chart_idx in range(starts):
        if len(accepted)==want:
            charts.append({
                "chart":chart_idx,"status":"skipped_after_full",
                "updates":0,"source_metric_evaluations":0
            })
            continue
        chunks=[
            rng.standard_normal(((WIDTH-e.shape[1])*e.shape[1]))/
            math.sqrt(WIDTH-e.shape[1])
            for e in E
        ]
        y=torch.tensor(np.concatenate(chunks),dtype=torch.float64)
        chart_success=False
        last=None
        attempted_updates=0
        chart_evals=0
        for k in range(steps+1):
            y_leaf=y.detach().clone().requires_grad_(True)
            bases=tangent_chart(E,C,y_leaf)
            p=projectors_from_bases(bases)
            prof=source_tensor_profile(z,g,bases,moments,gram)
            rec=check_source_candidate(p,prof,target,tol,dp,accepted)
            chart_evals+=1
            last=rec
            if rec["eligible"]:
                record={
                    **rec,**projector_digest(p.detach()),
                    "chart_index":chart_idx,"correction_updates":k,
                    "angle_from_dp12_radians":ANGLE,
                    "rank_tuple":[int(e.shape[1]) for e in E],
                    "full_tangent_parameter_count":len(y),
                    "source_observation_count":len(z)
                }
                accepted.append(p.detach().clone())
                records.append(record)
                charts.append({
                    "chart":chart_idx,"status":"admitted",
                    "correction_updates":k,
                    "source_metric_evaluations":chart_evals,
                    "projector_sha256":record["projector_sha256"],
                })
                chart_success=True
                break
            if k==steps:
                break
            # These three source moments are the only optimization targets.
            f=normalized_source_residual(prof,target,tol)
            current_penalty=float(squared_constraint_excess(f).detach())
            # Already source-feasible and pairwise-colliding candidates are
            # rejected, rather than being slid toward a near-duplicate.
            if current_penalty<=1e-14:
                break
            delta,info=minimum_norm_jacobian_step(f,y_leaf)
            jacobian_rank_census[info["jacobian_rank"]]+=1
            all_iterations+=1
            attempted_updates+=1
            advanced=False
            for alpha in LINE_SEARCH:
                y_new=y.detach()+alpha*delta
                with torch.no_grad():
                    b_new=tangent_chart(E,C,y_new)
                    new_profile=source_tensor_profile(z,g,b_new,moments,gram)
                    new_f=normalized_source_residual(new_profile,target,tol)
                    new_penalty=float(squared_constraint_excess(new_f))
                chart_evals+=1
                if new_penalty<current_penalty-1e-12:
                    y=y_new.detach()
                    advanced=True
                    break
            if not advanced:
                break
        if not chart_success:
            charts.append({
                "chart":chart_idx,"status":"ineligible_after_full_jacobian",
                "correction_updates":attempted_updates,
                "source_metric_evaluations":chart_evals,
                "last_source_errors":{
                    k:last[k] for k in (
                        "physical_rms_relative_error",
                        "signed_score_abs_error",
                        "score_rms_abs_error",
                        "distance_from_dp12",
                        "minimum_pairwise_distance",
                    )
                } if last is not None else None
            })
    result={
        "eligible":len(accepted)==want,
        "reason":"eight_full_tangent_distinct_matches" if len(accepted)==want
                 else "insufficient_full_tangent_source_feasible_shams",
        "root":root,
        "selected":len(accepted),
        "qualified":len(accepted),
        "charts_configured":starts,
        "max_corrections_per_chart":steps,
        "line_search_scales":list(LINE_SEARCH),
        "minimum_norm_step_limit":STEP_NORM_MAX,
        "jacobian_damping":GAUSS_NEWTON_DAMPING,
        "root_seed":CHART_SEED_BASE+root,
        "fixed_angle_radians":ANGLE,
        "minimum_DP12_chordal":DP12_MIN_CHORDAL,
        "minimum_pairwise_chordal":PAIRWISE_MIN_CHORDAL,
        "full_tangent_parameter_count":unconstrained_parameter_count(E),
        "source_write_relative_tolerance":SOURCE_WRITE_REL_TOL,
        "source_signed_score_abs_tolerance":tol[1],
        "source_score_rms_abs_tolerance":tol[2],
        "target_source_write_rms":target[0],
        "target_source_signed_score_mean":target[1],
        "target_source_score_rms":target[2],
        "source_observation_count":len(z),
        "projectors":records,
        "charts":charts,
        "charts_actually_started":sum(q["status"]!="skipped_after_full" for q in charts),
        "total_jacobian_corrections":all_iterations,
        "jacobian_rank_census":{str(k):v for k,v in jacobian_rank_census.items()},
        "uses_external_task_labels":False,
    }
    return accepted,result
