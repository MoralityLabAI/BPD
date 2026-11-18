"""FC03: optimize locally separated Grassmann projectors to satisfy FC02 sham source moments.

All projector search uses clean discovery activations and model-self score
gradients. It has no true task labels, heldout task NLL or model validation
selection. A frozen rank-tuple Grassmann chart guarantees nonzero angle to DP12.
"""
from __future__ import annotations

import base64
import hashlib
import math

import numpy as np
import torch
from torch import Tensor

from mechinterp.bpd.experiments.nc02.experiment import PASSES,WIDTH
from mechinterp.bpd.experiments.nc09_fixed12.tangent_sham_calibration.tangent_shams import (
    _physical_dose, score_numpy, SOURCE_SCORE_ABS_TOL,
    SOURCE_SCORE_REL_TOL, SOURCE_WRITE_REL_TOL,
)

CHART_STARTS=32
MAX_ADAM_STEPS=64
LEARNING_RATE=.08
ANGLE=.25
DP12_MIN_CHORDAL=.20
PAIRWISE_MIN_CHORDAL=.18
REGULARIZATION=.001
CHART_SEED_BASE=14_030_000
TARGET_SHAMS=8
PARAMETER_BOUND=1.


def qr_positive(a:Tensor) -> Tensor:
    """Reduced QR with a deterministic nonnegative R diagonal convention."""
    q,r=torch.linalg.qr(a,mode="reduced")
    d=torch.sign(torch.diagonal(r))
    d=torch.where(d==0,torch.ones_like(d),d)
    return q*d[None,:]


def numpy_qr_positive(a:np.ndarray) -> np.ndarray:
    q,r=np.linalg.qr(a,mode="reduced")
    d=np.sign(np.diag(r))
    d[d==0]=1.
    return q*d[None,:]


def chordal(source:Tensor,target:Tensor) -> Tensor:
    """Normalized rank-tuple chordal distance (exact same total rank 12)."""
    if source.shape!=target.shape or source.shape!=(PASSES,WIDTH,WIDTH):
        raise ValueError("Expected four full rank-20 orthogonal projector matrices")
    return torch.sqrt(
        ((source.double()-target.double())**2).sum().clamp_min(0)/24.
    )


def source_tensor_profile(reads:Tensor,grads:Tensor,bases:list[Tensor],
                          moments:Tensor,gram:Tensor):
    """Differentiable FC02 exact physical dose and signed/RMS score statistics."""
    n=len(reads)
    total=(reads*grads).sum(dim=(1,2))
    squared=reads.new_tensor(0.)
    for t,b in enumerate(bases):
        val=(reads[:,t]@b)*(grads[:,t]@b)
        total=total-val.sum(dim=1)
        a=b.T@moments[t]@b
        u=b.T@gram@b
        squared=squared+torch.trace(a@u)
    w=torch.sqrt(squared.clamp_min(1e-30)/(PASSES*WIDTH))
    m=total.mean()
    v=torch.sqrt(total.square().mean().clamp_min(1e-30))
    return w,m,v


def complement_frames(E:list[Tensor],rng:np.random.Generator):
    """For each pass, three independently seeded rank-r orthogonal complements."""
    frames=[]
    for base in E:
        p=base@base.T
        complement=torch.eye(WIDTH,dtype=torch.float64)-p
        f=[]
        for _ in range(3):
            a=torch.from_numpy(rng.standard_normal((WIDTH,base.shape[1]))).double()
            f.append(qr_positive(complement@a).detach())
        frames.append(f)
    return frames


def chart_bases(E:list[Tensor],frames:list[list[Tensor]],y:Tensor):
    """Fixed 0.25-rad Grassmann angle, with a QR-complement chart."""
    output=[]
    cos=math.cos(ANGLE)
    sin=math.sin(ANGLE)
    identity=torch.eye(WIDTH,dtype=torch.float64)
    for t,b in enumerate(E):
        f=frames[t]
        raw=f[0]+y[t,0]*f[1]+y[t,1]*f[2]
        Q=qr_positive((identity-b@b.T)@raw)
        output.append(cos*b+sin*Q)
    return output


def projectors_from_bases(B:list[Tensor]):
    return torch.stack([v@v.T for v in B])


def projector_digest(P:Tensor):
    data=P.detach().cpu().contiguous().double().numpy().astype("<f8").tobytes()
    return {
        "projector_sha256":hashlib.sha256(data).hexdigest(),
        "projector_b64_le_float64":base64.b64encode(data).decode("ascii")
    }


def check_source_candidate(P:Tensor,profile:tuple[Tensor,Tensor,Tensor],
                           target:tuple[float,float,float],
                           tolerance:tuple[float,float,float],
                           dp12_P:Tensor,admitted:list[Tensor],
                           *,diagnostics:bool=True):
    w,m,v=(float(t.detach()) for t in profile)
    dW=abs(w/target[0]-1.)
    dm=abs(m-target[1])
    dv=abs(v-target[2])
    sep=float(chordal(P,dp12_P).detach())
    pairwise=min((float(chordal(P,q).detach()) for q in admitted),default=None)
    wrank=[float(torch.trace(P[t]).detach()) for t in range(PASSES)]
    idempotence=float((P@P-P).abs().max().detach())
    symmetry=float((P-P.transpose(-1,-2)).abs().max().detach())
    eligible=bool(
        math.isfinite(w) and math.isfinite(m) and math.isfinite(v)
        and dW<=SOURCE_WRITE_REL_TOL+1e-12
        and dm<=tolerance[1]+1e-12
        and dv<=tolerance[2]+1e-12
        and sep>=DP12_MIN_CHORDAL-1e-12
        and (pairwise is None or pairwise>=PAIRWISE_MIN_CHORDAL-1e-12)
        and idempotence<=1e-6 and symmetry<=1e-6
    )
    return {
        "eligible":eligible,
        "physical_write_rms":w,
        "signed_first_order_score_mean":m,
        "first_order_score_rms":v,
        "physical_rms_relative_error":dW,
        "signed_score_abs_error":dm,
        "score_rms_abs_error":dv,
        "distance_from_dp12":sep,
        "minimum_pairwise_distance":pairwise,
        "passwise_projector_trace":wrank,
        "projector_idempotence_max":idempotence,
        "projector_symmetry_max":symmetry,
    }


def fit_constrained_grassmann(
    activations:Tensor,decoder:Tensor,clean_reads:Tensor,
    clean_gradients:Tensor,dp12_bases:list[Tensor],seed:int,
    *,starts:int=CHART_STARTS,steps:int=MAX_ADAM_STEPS,
    want:int=TARGET_SHAMS
):
    """Frozen 32 starts × 64 Adam steps, first 8 independently source-admitted."""
    if (activations.ndim!=3 or activations.shape[1:]!=(PASSES,WIDTH)
        or clean_reads.shape!=activations.shape
        or clean_gradients.shape!=activations.shape
        or decoder.shape!=(WIDTH,WIDTH)
        or len(dp12_bases)!=PASSES
        or starts<1 or steps<0 or want<1):
        raise ValueError("FC03 requires complete clean root source and four DP bases")
    torch.set_num_threads(1)
    A=activations.detach().double().cpu()
    z=clean_reads.detach().double().cpu()
    g=clean_gradients.detach().double().cpu()
    U=decoder.detach().double().cpu()
    E=[x.detach().double().cpu() for x in dp12_bases]
    if sum(b.shape[1] for b in E)!=12:
        raise ValueError("Original fixed 12-slot source projectors required")
    for b in E:
        if b.shape[0]!=WIDTH or (b.T@b-torch.eye(b.shape[1],dtype=torch.float64)).abs().max()>1e-6:
            raise ValueError("Original source basis is not orthonormal")
    moments=torch.einsum("nti,ntj->tij",A,A)/len(A)
    gram=U@U.T
    orig=source_tensor_profile(z,g,E,moments,gram)
    target=tuple(float(x.detach()) for x in orig)
    if target[0]<=1e-12 or not all(math.isfinite(x) for x in target):
        return [],{"admitted":False,"source_zero_or_invalid":True,"qualified":0,
                   "selected":0,"charts_configured":starts,"steps_per_chart":steps}
    tolerance=(
        SOURCE_WRITE_REL_TOL,
        max(SOURCE_SCORE_ABS_TOL,SOURCE_SCORE_REL_TOL*abs(target[1])),
        max(SOURCE_SCORE_ABS_TOL,SOURCE_SCORE_REL_TOL*target[2])
    )
    # Cross-check against the untouched FC02 numpy reference implementation.
    en=[x.numpy() for x in E]
    np_target=score_numpy(z.numpy(),g.numpy(),en)
    npdose=_physical_dose(
        en,
        np.einsum("nti,ntj->tij",A.numpy(),A.numpy())/len(A),
        U.numpy()@U.numpy().T
    )
    if (abs(npdose-target[0])>1e-9 or
        abs(np_target["signed_first_order_score_mean"]-target[1])>1e-9 or
        abs(np_target["first_order_score_rms"]-target[2])>1e-9):
        raise AssertionError("FC03 source moments disagree with frozen FC02 metric")
    base_P=projectors_from_bases(E)
    rng=np.random.default_rng(CHART_SEED_BASE+seed)
    admitted=[]
    source_records=[]
    chart_records=[]
    rejection_count=0
    for start in range(starts):
        if len(admitted)==want:
            chart_records.append({"chart":start,"status":"skipped_after_full",
                                  "steps_used":0})
            continue
        frame=complement_frames(E,rng)
        params=torch.nn.Parameter(torch.zeros((PASSES,2),dtype=torch.float64))
        optimizer=torch.optim.Adam([params],lr=LEARNING_RATE)
        success=False
        last_record=None
        for step in range(steps+1):
            bases=chart_bases(E,frame,params)
            P=projectors_from_bases(bases)
            stats=source_tensor_profile(z,g,bases,moments,gram)
            rec=check_source_candidate(
                P,stats,target,tolerance,base_P,admitted
            )
            last_record=rec
            if rec["eligible"]:
                frozen_P=P.detach().clone()
                frozen_rec={
                    **rec,
                    **projector_digest(frozen_P),
                    "chart_index":start,
                    "optimization_step":step,
                    "angle_from_dp12_radians":ANGLE,
                    "rank_tuple":[int(b.shape[1]) for b in E],
                    "optimized_parameter_y":params.detach().numpy().tolist(),
                    "source_observation_count":len(A),
                }
                admitted.append(frozen_P)
                source_records.append(frozen_rec)
                chart_records.append({
                    "chart":start,"status":"admitted",
                    "steps_used":step,
                    "projector_sha256":frozen_rec["projector_sha256"]
                })
                success=True
                break
            if step==steps:
                break
            w,m,v=stats
            ratio_w=torch.abs(w/target[0]-1.)/tolerance[0]
            ratio_m=torch.abs(m-target[1])/tolerance[1]
            ratio_v=torch.abs(v-target[2])/tolerance[2]
            over=torch.stack([
                torch.relu(ratio_w-1),torch.relu(ratio_m-1),
                torch.relu(ratio_v-1)
            ])
            penalty=over.square().sum()
            if admitted:
                distances=torch.stack([chordal(P,q) for q in admitted])
                penalty=penalty+torch.relu(1-distances/PAIRWISE_MIN_CHORDAL).square().sum()
            loss=penalty+REGULARIZATION*params.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([params],10.)
            optimizer.step()
            with torch.no_grad():
                params.clamp_(-PARAMETER_BOUND,PARAMETER_BOUND)
        if not success:
            rejection_count+=1
            chart_records.append({
                "chart":start,"status":"ineligible_after_optimizer",
                "steps_used":steps,
                "last_source_errors":{
                    k:last_record[k] for k in (
                        "physical_rms_relative_error","signed_score_abs_error",
                        "score_rms_abs_error","distance_from_dp12",
                        "minimum_pairwise_distance"
                    )
                }
            })
    result={
        "admitted":len(admitted)==want,
        "reason":"eight_constrained_distinct_shams" if len(admitted)==want
                 else "insufficient_constrained_grassmann_solutions",
        "root":seed,
        "selected":len(admitted),
        "qualified":len(admitted),
        "charts_configured":starts,
        "charts_actually_attempted":len([x for x in chart_records if x["status"]!="skipped_after_full"]),
        "charts_skipped_after_full":len([x for x in chart_records if x["status"]=="skipped_after_full"]),
        "charts_ineligible_after_optimization":rejection_count,
        "max_adam_steps_per_chart":steps,
        "adam_lr":LEARNING_RATE,
        "chart_seed":CHART_SEED_BASE+seed,
        "chart_angle_radians":ANGLE,
        "minimum_dp12_chordal":DP12_MIN_CHORDAL,
        "minimum_pairwise_chordal":PAIRWISE_MIN_CHORDAL,
        "source_write_relative_tolerance":SOURCE_WRITE_REL_TOL,
        "source_signed_score_abs_tolerance":tolerance[1],
        "source_score_rms_abs_tolerance":tolerance[2],
        "target_source_write_rms":target[0],
        "target_source_signed_score_mean":target[1],
        "target_source_score_rms":target[2],
        "accepted":source_records,
        "charts":chart_records,
        "source_observation_count":len(A),
        "uses_external_labels":False,
    }
    return admitted,result
