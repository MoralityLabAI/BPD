"""Prospectively frozen source-only physical/tangent sham construction for FC02.

No task labels, final heldout samples or task NLL enter this module.
One independent random projector proposal uses the same angle at four passes
but separate random orthogonal complements, retaining exact original ranks.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from mechinterp.bpd.experiments.nc02.experiment import PASSES, WIDTH
from mechinterp.bpd.experiments.nc08.experiment import OPTIONS

SHAM_PROPOSALS=4096
SHAMS=8
ANGLE_GRID=(.20,.30,.45,.65)
SOURCE_WRITE_REL_TOL=.15
SOURCE_SCORE_ABS_TOL=.005
SOURCE_SCORE_REL_TOL=.25
SHAM_SEED_BASE=13_130_000


def source_score_profile(reads:Tensor,gradients:Tensor,
                         bases:list[Tensor]) -> dict:
    """RMS and signed mean of sum_t g_t · [(I-P_t)z_t] on clean source."""
    if len(bases)!=PASSES or reads.shape!=gradients.shape or (
        reads.ndim!=3 or reads.shape[1:]!=(PASSES,WIDTH)
    ):
        raise ValueError("Full clean source [N,4,20] and four bases required")
    z=reads.detach().double().cpu().numpy()
    g=gradients.detach().double().cpu().numpy()
    return score_numpy(z,g,[b.detach().double().cpu().numpy() for b in bases])


def score_numpy(z:np.ndarray,g:np.ndarray,bases:list[np.ndarray]) -> dict:
    """Signed first-order omitted model-self score, never external task loss."""
    if z.shape!=g.shape or z.ndim!=3 or z.shape[1:]!=(PASSES,WIDTH):
        raise ValueError("Clean source and gradients must be [N,4,20]")
    if len(bases)!=PASSES:
        raise ValueError("One basis per pass required")
    effect=np.einsum("nti,nti->n",z,g,optimize=True)
    for t,b in enumerate(bases):
        if b.ndim!=2 or b.shape[0]!=WIDTH or b.shape[1] not in OPTIONS:
            raise ValueError("Unregistered sham basis size")
        effect-=np.einsum("nr,nr->n",z[:,t]@b,g[:,t]@b,optimize=True)
    mean=float(effect.mean())
    rms=float(np.sqrt(np.mean(effect**2)))
    if not (np.isfinite(mean) and np.isfinite(rms)):
        raise ValueError("Nonfinite source score effect")
    return {"signed_first_order_score_mean":mean,
            "first_order_score_rms":rms}


def _physical_dose(bases:list[np.ndarray], moment:np.ndarray,
                   gram:np.ndarray) -> float:
    total=0.
    for t in range(PASSES):
        b=bases[t]
        total+=float(np.trace((b.T@moment[t]@b)@(b.T@gram@b)))
    return float(np.sqrt(max(total,0.)/(PASSES*WIDTH)))


def build_derivative_matched_shams(
    activations:Tensor,physical_decoder:Tensor,
    clean_reads:Tensor,clean_gradients:Tensor,
    basis:list[Tensor],root:int,
    *,proposals:int=SHAM_PROPOSALS,want:int=SHAMS
):
    """First N same-rank candidates meeting dose and score source-only gates."""
    if (len(basis)!=PASSES or clean_reads.shape!=clean_gradients.shape or
        clean_reads.ndim!=3 or clean_reads.shape[1:]!=(PASSES,WIDTH) or
        activations.shape!=clean_reads.shape or
        physical_decoder.shape!=(WIDTH,WIDTH) or want<1 or proposals<1):
        raise ValueError("Incompatible source read, physical writes, or budget")
    e=[b.detach().double().cpu().numpy() for b in basis]
    for b in e:
        if b.ndim!=2 or b.shape[0]!=WIDTH or b.shape[1] not in OPTIONS:
            raise ValueError("Original source basis has unsupported rank")
        if np.max(np.abs(b.T@b-np.eye(b.shape[1])))>1e-6:
            raise ValueError("Original source basis is not orthonormal")
    reads=clean_reads.detach().double().cpu().numpy()
    grads=clean_gradients.detach().double().cpu().numpy()
    a=activations.detach().double().cpu().numpy()
    U=physical_decoder.detach().double().cpu().numpy()
    moment=np.einsum("nti,ntj->tij",a,a,optimize=True)/len(a)
    gram=U@U.T
    dose_target=_physical_dose(e,moment,gram)
    if dose_target<=1e-12:
        return [],{"eligible":False,"reason":"zero_source_physical_dose",
                   "proposals":proposals,"qualified":0,"selected":0,
                   "target_source_write_rms":dose_target}
    target=score_numpy(reads,grads,e)
    signed_limit=max(SOURCE_SCORE_ABS_TOL,
                     SOURCE_SCORE_REL_TOL*abs(target["signed_first_order_score_mean"]))
    rms_limit=max(SOURCE_SCORE_ABS_TOL,
                  SOURCE_SCORE_REL_TOL*target["first_order_score_rms"])
    rng=np.random.default_rng(SHAM_SEED_BASE+root)
    qualified=0
    accepted=[]
    min_angle=float("inf")
    stats_minmax={
        "write_relative_error_min":float("inf"),
        "score_mean_abs_error_min":float("inf"),
        "score_rms_abs_error_min":float("inf"),
    }
    for proposal_index in range(proposals):
        theta=ANGLE_GRID[int(rng.integers(len(ANGLE_GRID)))]
        rotated=[]
        for base in e:
            d=base.shape[1]
            z=rng.standard_normal((WIDTH,d))
            z=z-base@(base.T@z)
            q,_=np.linalg.qr(z,mode="reduced")
            rotated.append(np.cos(theta)*base+np.sin(theta)*q)
        candidate_dose=_physical_dose(rotated,moment,gram)
        dose_relative=abs(candidate_dose/dose_target-1.)
        stats_minmax["write_relative_error_min"]=min(
            stats_minmax["write_relative_error_min"],dose_relative)
        # Only source-dose matched orientations need score calculations.
        if dose_relative>SOURCE_WRITE_REL_TOL:
            continue
        profile=score_numpy(reads,grads,rotated)
        signed_abs=abs(profile["signed_first_order_score_mean"]-
                       target["signed_first_order_score_mean"])
        rms_abs=abs(profile["first_order_score_rms"]-
                    target["first_order_score_rms"])
        stats_minmax["score_mean_abs_error_min"]=min(
            stats_minmax["score_mean_abs_error_min"],signed_abs)
        stats_minmax["score_rms_abs_error_min"]=min(
            stats_minmax["score_rms_abs_error_min"],rms_abs)
        if signed_abs>signed_limit or rms_abs>rms_limit:
            continue
        qualified+=1
        min_angle=min(min_angle,theta)
        if len(accepted)<want:
            p=torch.stack([
                torch.from_numpy(v@v.T).double() for v in rotated
            ])
            accepted.append({
                "projector":p,"proposal_index":proposal_index,
                "principal_angle":theta,
                "source_write_rms":candidate_dose,
                "write_relative_error":dose_relative,
                "score_signed_abs_error":signed_abs,
                "score_rms_abs_error":rms_abs,
                "signed_first_order_score_mean":profile["signed_first_order_score_mean"],
                "first_order_score_rms":profile["first_order_score_rms"],
            })
    report={
        "eligible":len(accepted)==want,
        "reason":"matched_eight" if len(accepted)==want
                 else "insufficient_three_way_source_matches",
        "root":root,
        "proposals":proposals,
        "qualified":qualified,
        "selected":len(accepted),
        "principal_angle_grid":list(ANGLE_GRID),
        "target_source_write_rms":dose_target,
        "target_signed_score_mean":target["signed_first_order_score_mean"],
        "target_first_order_score_rms":target["first_order_score_rms"],
        "write_relative_tolerance":SOURCE_WRITE_REL_TOL,
        "signed_score_abs_tolerance":signed_limit,
        "rms_score_abs_tolerance":rms_limit,
        "selected_proposal_indices":[v["proposal_index"] for v in accepted],
        "selected_angles":[v["principal_angle"] for v in accepted],
        "selected_dose_relative_errors":[v["write_relative_error"] for v in accepted],
        "selected_score_mean_abs_errors":[v["score_signed_abs_error"] for v in accepted],
        "selected_score_rms_abs_errors":[v["score_rms_abs_error"] for v in accepted],
        "minimum_matched_angle":min_angle if qualified else None,
        "minimum_candidate_errors":{k:(v if np.isfinite(v) else None)
                                    for k,v in stats_minmax.items()},
        "source_observation_count":len(reads),
        "uses_external_labels":False
    }
    return accepted,report
