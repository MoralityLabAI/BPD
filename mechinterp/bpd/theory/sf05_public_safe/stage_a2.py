"""SF05-A2: finite public level-set repair on *previously excluded* roots only.

Designed after SF05 Stage A exposed a nonlinear public effect. This script
is development engineering; it has no new-root scientific acceptance gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechinterp.bpd.theory.sf04_source_binding.experiment import (
    COMPONENTS, eligible, paired_source_swap, rng, select_components,
    tangent_matched, train_factors, train_teacher,
)
from mechinterp.bpd.theory.sf05_public_safe.finite_correct import finite_public_repair
from mechinterp.bpd.theory.sf05_public_safe.geometry import (
    physical_output_subspace_write, public_safe_projection,
)
from mechinterp.bpd.theory.sf05_public_safe.stage_a import (
    EXCLUDED_ROOTS, authorized_and_public_gradients,
)


def _readout_recovery(output, donor_clean, eligible_auth):
    return float(((output-donor_clean).abs() <= .10)[eligible_auth].float().mean())


def _diagnostic(teacher, h2, query, pubquery, donor, eligible_auth, initial):
    raw = initial.detach().clone()
    raw[:, 0] = 0
    public_fn = lambda state: teacher.step(state, pubquery)[:, 0]
    repaired = finite_public_repair(h2, raw, public_fn)
    with torch.no_grad():
        auth_raw = teacher.step(h2+raw, query)[:, 0]
        auth_corrected = teacher.step(h2+repaired.delta, query)[:, 0]
        all_conv = float(repaired.converged.float().mean())
        good_conv = float(repaired.converged[eligible_auth].float().mean())
        raw_norm = raw.norm(dim=-1)
        update_ratio = repaired.correction_norm/raw_norm.clamp_min(1e-12)
        max_current = float(((h2+repaired.delta)[:, 0]-h2[:, 0]).abs().max())
        return {
            "authorized_recovery_raw": _readout_recovery(auth_raw,donor,eligible_auth),
            "authorized_recovery_finite_corrected": _readout_recovery(auth_corrected,donor,eligible_auth),
            "all_case_convergence": all_conv,
            "eligible_auth_convergence": good_conv,
            "public_abs_change_raw": float(repaired.public_error_before.abs().mean()),
            "public_abs_change_finite_corrected": float(repaired.public_error_after.abs().mean()),
            "public_abs_change_finite_corrected_eligible": float(repaired.public_error_after[eligible_auth].abs().mean()),
            "mean_correction_norm": float(repaired.correction_norm.mean()),
            "mean_correction_to_raw_dose": float(update_ratio.mean()),
            "max_current_readout_error": max_current,
            "zero_gradient_failures": int(repaired.failed_zero_gradient.sum()),
            "nonconverged_cases": int((~repaired.converged).sum()),
        }


def evaluate_root(seed: int, *, n: int=512, shams: int=16):
    if seed not in EXCLUDED_ROOTS:
        raise ValueError("SF05-A2 development may only execute excluded roots 901/902")
    torch.set_num_threads(1)
    teacher = train_teacher(seed,steps=1200)
    factors = train_factors(seed,teacher,steps=400)
    recipient, donor_second, donor_target = paired_source_swap(seed+710300,n)
    first, second, query, target, metadata = recipient
    valid,(h1,h2,h3,d2,d3) = eligible(teacher,recipient,donor_second,donor_target)
    chosen,_ = select_components(seed,teacher,factors,pairs=240,count=14)
    with torch.no_grad():
        diff=teacher.features(h1,donor_second)-teacher.features(h1,second)
        component=(diff@factors.V)[:,:,None]*factors.U[None,:,:]
        writes={
            "singleton_14":component[:,chosen,:].sum(1),
            "bpd_full_physical":diff@factors.weight(),
            "teacher_full_physical":diff@teacher.C.detach(),
            "exact_hidden_donor":d2-h2,
        }
        rank16,gap=physical_output_subspace_write(diff,factors.weight(),16)
        writes["bpd_svd_rank_16"]=rank16
    public,gpub,gauth=authorized_and_public_gradients(teacher,h2,query)
    outcomes={}
    for name,raw in writes.items():
        outcomes[name]=_diagnostic(teacher,h2,query,public,d3[:,0],valid,raw)
    full_raw=writes["bpd_full_physical"].clone()
    full_raw[:,0]=0
    tangent=public_safe_projection(full_raw,gpub).value
    outcomes["bpd_full_tangent_then_finite"]=_diagnostic(
        teacher,h2,query,public,d3[:,0],valid,tangent)

    rng_null=rng(seed+917410)
    nulls={}
    for name in ("singleton_14","bpd_full_physical","bpd_svd_rank_16"):
        raw=writes[name].clone()
        raw[:,0]=0
        dose=raw.norm(dim=1,keepdim=True)
        arms={"random_dark":[],"signed_authorized_dark":[]}
        for _ in range(shams):
            random=torch.randn(raw.shape,generator=rng_null)
            random[:,0]=0
            random=random / random.norm(dim=1,keepdim=True).clamp_min(1e-12)*dose
            signed,_,_=tangent_matched(raw,gauth,rng_null)
            for label,delta in (("random_dark",random),("signed_authorized_dark",signed)):
                arms[label].append(
                    _diagnostic(teacher,h2,query,public,d3[:,0],valid,delta))
        nulls[name]={
            label:{
                "n_arms":shams,
                "authorized_corrected_recovery_mean":sum(
                    x["authorized_recovery_finite_corrected"] for x in vals)/shams,
                "finite_corrected_public_abs_mean":sum(
                    x["public_abs_change_finite_corrected"] for x in vals)/shams,
                "convergence_all_mean":sum(x["all_case_convergence"] for x in vals)/shams,
                "correction_to_raw_dose_mean":sum(x["mean_correction_to_raw_dose"] for x in vals)/shams,
            } for label,vals in arms.items()
        }
    return {
        "seed":seed,"evidence_class":"post-stage-A excluded-root development",
        "n":n,"authorized_eligible":int(valid.sum()),"sham_arms":shams,
        "teacher_mae":float((h3[:,0]-target).abs().mean()),
        "factor_mse":float((factors.weight()-teacher.C).square().mean()),
        "rank16_singular_gap":gap,
        "arms":outcomes,"null_controls":nulls,
    }


def run(seeds=EXCLUDED_ROOTS,shams=16):
    if any(s not in EXCLUDED_ROOTS for s in seeds):
        raise ValueError("SF05-A2 prohibits new scientific roots")
    return {"scope":"BPD-SF05-A2 privileged public level-set solver; excluded roots",
            "torch_version":torch.__version__,"roots":list(seeds),
            "rows":[evaluate_root(s,shams=shams) for s in seeds]}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,nargs="+",default=list(EXCLUDED_ROOTS))
    p.add_argument("--shams",type=int,default=16)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    result=run(tuple(a.seeds),a.shams)
    print(json.dumps(result,separators=(",",":")))
    if a.out:
        a.out.write_text(json.dumps(result,indent=2)+"\n")
