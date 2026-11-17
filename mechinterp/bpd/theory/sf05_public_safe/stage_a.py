"""BPD-SF05 Stage A: *excluded-root-only* public-safe source-binding pilot.

All teacher/factor training uses SF04's canonical implementation. Seeds 901
and 902 were used for design in SF04 and are forbidden as new-root evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechinterp.bpd.theory.sf04_source_binding.experiment import (
    COMPONENTS, eligible, paired_source_swap, rng, select_components,
    train_factors, train_teacher,
)
from mechinterp.bpd.theory.sf05_public_safe.geometry import (
    linearized_authorized_capacity, physical_output_subspace_write,
    public_safe_projection, random_public_safe_sham,
    signed_authorized_matched_sham,
)

EXCLUDED_ROOTS = (901, 902)
RANKS = (8, 12, 16, 20)


def authorized_and_public_gradients(teacher, state, query):
    """Actual gradients at the *same pre-query state* for two recipient roles."""
    public_query = query.clone()
    public_query[:, 4:7] = 0.0
    public_query[:, 6] = 1.0
    state_grad = state.detach().requires_grad_(True)
    visible_public = teacher.step(state_grad, public_query)[:, 0]
    visible_authorized = teacher.step(state_grad, query)[:, 0]
    gp = torch.autograd.grad(visible_public.sum(), state_grad, retain_graph=True)[0]
    ga = torch.autograd.grad(visible_authorized.sum(), state_grad)[0]
    return public_query, gp.detach(), ga.detach()


def test_arm(teacher, state, query, public_query, raw_delta, grad_public,
             grad_auth, donor_visible, eligible_auth):
    """Report the raw/current-dark and public-tangent-projected arm separately."""
    with torch.no_grad():
        raw = raw_delta.clone()
        raw[:, 0] = 0.0
        proj = public_safe_projection(raw, grad_public)
        safe = proj.value
        qpublic0 = teacher.step(state, public_query)[:, 0]
        qauth_raw = teacher.step(state + raw, query)[:, 0]
        qpublic_raw = teacher.step(state + raw, public_query)[:, 0]
        qauth_safe = teacher.step(state + safe, query)[:, 0]
        qpublic_safe = teacher.step(state + safe, public_query)[:, 0]

        def recovery(out):
            if not bool(eligible_auth.any()):
                return 0.0
            return float(((out-donor_visible).abs() <= .10)[eligible_auth].float().mean())

        current_error = float(((state+safe)[:, 0]-state[:, 0]).abs().max())
        public_dot_error = float(proj.public_tangent.abs().max())
        dose = raw.norm(dim=-1)
        safe_dose = safe.norm(dim=-1)
        capacity = linearized_authorized_capacity(grad_auth, grad_public, safe)
        return {
            "authorized_raw_dark_recovery": recovery(qauth_raw),
            "authorized_public_tangent_safe_recovery": recovery(qauth_safe),
            "finite_public_raw_mean_abs_change": float((qpublic_raw-qpublic0).abs().mean()),
            "finite_public_safe_mean_abs_change": float((qpublic_safe-qpublic0).abs().mean()),
            "public_change_on_auth_eligible": float((qpublic_safe-qpublic0).abs()[eligible_auth].mean()) if bool(eligible_auth.any()) else 0.0,
            "mean_raw_dark_norm": float(dose.mean()),
            "mean_safe_norm": float(safe_dose.mean()),
            "mean_norm_removed_ratio": float(((dose-safe_dose)/dose.clamp_min(1e-12)).mean()),
            "max_current_readout_error": current_error,
            "max_public_tangent_error": public_dot_error,
            "degenerate_public_gradient_cases": int(proj.zero_public_gradient.sum()),
            "mean_authorized_capacity": float(capacity[eligible_auth].mean()) if bool(eligible_auth.any()) else 0.0,
        }, safe


def evaluate_root(seed, sham_arms=16, n=512):
    if seed not in EXCLUDED_ROOTS:
        raise ValueError("Stage A may only execute previously excluded 901/902")
    torch.set_num_threads(1)
    teacher = train_teacher(seed, steps=1200)
    factors = train_factors(seed, teacher, steps=400)
    recipient, donor_second, donor_target = paired_source_swap(seed+710300, n)
    first, second, query, target, _metadata = recipient
    source_valid, (h1,h2,h3,d2,d3) = eligible(teacher,recipient,donor_second,donor_target)
    # Parent SF04 already verifies clean donor/recipient eligibility and source tags.
    selected, _cal = select_components(seed, teacher, factors, pairs=240, count=14)
    with torch.no_grad():
        feature_delta = teacher.features(h1, donor_second) - teacher.features(h1, second)
        component_writes = (feature_delta @ factors.V)[:, :, None] * factors.U[None, :, :]
        physical = factors.weight().detach()
        partial = component_writes[:,selected,:].sum(dim=1)
        full = feature_delta @ physical
        teacher_full = feature_delta @ teacher.C.detach()
        exact_state = d2 - h2
        svd = {f"bpd_svd_rank_{k}":physical_output_subspace_write(feature_delta,physical,k)
               for k in RANKS}
        teacher_rank16, teacher_rank16_gap = physical_output_subspace_write(
            feature_delta,teacher.C.detach(),16)
        shear = torch.eye(COMPONENTS)
        shear[0,1] = .7
        transformed = (factors.V@shear) @ torch.linalg.solve(shear,factors.U)
        gauge_full_error = float((transformed-physical).abs().max())
        gauge_svd_max = max(
            float((svd[f"bpd_svd_rank_{k}"][0] -
                   physical_output_subspace_write(feature_delta,transformed,k)[0]).abs().max())
            for k in RANKS
        )
    qpublic,gp,ga=authorized_and_public_gradients(teacher,h2,query)
    donor_visible=d3[:,0]
    arms={"singleton_14":(partial,None),"physical_full":(full,None),
          "teacher_full":(teacher_full,None),"exact_donor_state":(exact_state,None),
          "teacher_svd_rank_16":(teacher_rank16,teacher_rank16_gap)}
    arms.update(svd)
    results={}
    safes={}
    for name,(delta,gap) in arms.items():
        result,safe=test_arm(teacher,h2,query,qpublic,delta,gp,ga,donor_visible,source_valid)
        result["physical_singular_gap"]=gap
        results[name]=result
        safes[name]=safe
    control_seed=rng(seed+887700)
    controls={}
    with torch.no_grad():
        for name in ("singleton_14","physical_full","bpd_svd_rank_16"):
            safe=safes[name]
            random_rec, signed_rec, random_pub, signed_pub=[],[],[],[]
            baseline=teacher.step(h2,qpublic)[:,0]
            for _ in range(sham_arms):
                random=random_public_safe_sham(safe,gp,control_seed)
                signed=signed_authorized_matched_sham(safe,gp,ga,control_seed)
                for intervention,rec,pub in [(random,random_rec,random_pub),
                                             (signed,signed_rec,signed_pub)]:
                    out=teacher.step(h2+intervention,query)[:,0]
                    po=teacher.step(h2+intervention,qpublic)[:,0]
                    rec.append(float(((out-donor_visible).abs()<=.10)[source_valid].float().mean()))
                    pub.append(float((po-baseline).abs().mean()))
            controls[name]={
                "matched_arms":sham_arms,
                "random_intersection_recovery_mean":sum(random_rec)/sham_arms,
                "signed_authorized_intersection_recovery_mean":sum(signed_rec)/sham_arms,
                "random_intersection_public_abs_mean":sum(random_pub)/sham_arms,
                "signed_authorized_intersection_public_abs_mean":sum(signed_pub)/sham_arms,
                "signed_authorized_recovery_max":max(signed_rec),
            }
    return {
        "seed":seed,"evidence_class":"excluded-root development diagnostic",
        "generated":n,"authorized_eligible":int(source_valid.sum()),
        "source_binding_teacher_mae":float((h3[:,0]-recipient[3]).abs().mean()),
        "factor_weight_mse":float((physical-teacher.C).square().mean()),
        "full_operator_gauge_error":gauge_full_error,
        "svd_gauge_max_write_error":gauge_svd_max,
        "arm_results":results,"null_controls":controls,
    }


def run(seeds=EXCLUDED_ROOTS,sham_arms=16,n=512):
    if any(s not in EXCLUDED_ROOTS for s in seeds):
        raise ValueError("SF05 Stage A prohibits fresh-science roots")
    return {"scope":"SF05 Stage A excluded-root-only",
            "torch_version":torch.__version__,"roots":list(seeds),
            "rows":[evaluate_root(seed,sham_arms,n) for seed in seeds]}


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,nargs="+",default=list(EXCLUDED_ROOTS))
    p.add_argument("--shams",type=int,default=16)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    outcome=run(tuple(a.seeds),a.shams)
    print(json.dumps(outcome,separators=(",",":")))
    if a.out:
        a.out.write_text(json.dumps(outcome,indent=2)+"\n")
