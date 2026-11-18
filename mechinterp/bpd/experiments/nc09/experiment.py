"""NC-09: one-source-to-another task transfer, exact BPD closed-loop scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .core import (
    ROOTS, EXCLUDED_ENGINEERING_ROOT, PROTOCOL_COMMIT, PASSES, WIDTH, SHAMS,
    PROPOSALS, sample_base, mark_task, task_labels, context_masks,
    train_model, wrap_bpd, set_basis, full_mask_parity, bernoulli_kl,
    discover, projector_from_gradients, source_doses, local_matched_shams,
    run_projector, transport_to_rotated,
)


@torch.no_grad()
def clean_forward(bpd, model, x: Tensor) -> tuple[Tensor, Tensor]:
    with bpd.target_capture():
        logits,_ = model(x,capture=True)
        writes = torch.stack(model.writes,dim=1)
    return logits,writes


@torch.no_grad()
def evaluate(logits: Tensor, writes: Tensor, reference: Tensor,
             reference_writes: Tensor, base: Tensor, labels: Tensor) -> dict:
    if logits.shape != reference.shape or len(base) != len(logits):
        raise ValueError("invalid paired intervention measurements")
    contexts = context_masks(base)
    sign = torch.where(labels,1.,-1.)
    loss = F.softplus(-2*sign*logits[:, -1])
    correct = (logits[:, -1]>=0) == labels
    kl = bernoulli_kl(reference,logits)
    out={
        "nll_mean_nats":float(loss.mean()),
        "accuracy":float(correct.float().mean()),
        "by_context":{},
        "kl_by_pass_nats":[float(v) for v in kl.mean(dim=0)],
        "physical_write_delta_rms":float((writes-reference_writes).square().mean().sqrt()),
        # Retain raw independent-evaluation case outcomes for independent audit;
        # examples within each model root remain correlated for inference.
        "per_case_nll":[float(v) for v in loss],
        "per_case_correct":[int(v) for v in correct],
    }
    for name,mask in contexts.items():
        if mask.sum()==0:
            raise ValueError("empty held-out task context")
        out["by_context"][name]={
            "n_cases":int(mask.sum()),
            "nll_nats":float(loss[mask].mean()),
            "accuracy":float(correct[mask].float().mean()),
            "correct_count":int(correct[mask].sum()),
            "kl_final_nats":float(kl[mask,-1].mean()),
        }
    return out


def assess_root(seed: int, *, steps: int = 640, discovery_size: int = 1024,
                evaluation_size: int = 1024, proposals: int = PROPOSALS) -> dict:
    if seed not in ROOTS and seed!=EXCLUDED_ENGINEERING_ROOT:
        raise ValueError("NC-09 source-excluded test seed or development roots only")
    torch.set_num_threads(1)
    model, training = train_model(seed,steps=steps)
    xdisc=sample_base(seed,2,discovery_size)
    xe=sample_base(seed,3,evaluation_size)
    xda,xdb=mark_task(xdisc,"A"),mark_task(xdisc,"B")
    xea,xeb=mark_task(xe,"A"),mark_task(xe,"B")
    bpd=wrap_bpd(model)
    out={
        "study":"BPD-NC-09", "phase":"EXCLUDED_SOURCE_TEST" if seed==EXCLUDED_ENGINEERING_ROOT else "ENGINEERING",
        "evidence_class":"SYNTHETIC_TRAINED_WITHIN_MODEL_TASK_TRANSFER",
        "seed":seed, "protocol_commit":PROTOCOL_COMMIT,
        "data":{"train_base":2048,"train_task_marked":4096,
                "discovery_base":discovery_size,"evaluation_base":evaluation_size},
        "training":training,
        "bases":{}, "discovery":{}, "dose_matching":{}, "sham_matching":{},
        "negative_controls":{}, "heldout":{}, "primary":{},
    }
    set_basis(bpd,seed,"native")
    native_parity=full_mask_parity(bpd,model,torch.cat((xea[:64],xeb[:64])))
    out["bases"]["native"]={"closure":native_parity}
    acts_a, grads_a=discover(bpd,model,xda)
    acts_b, grads_b=discover(bpd,model,xdb)
    pa, ea, diag_a=projector_from_gradients(grads_a)
    pb, eb, diag_b=projector_from_gradients(grads_b)
    out["discovery"]["A"]=diag_a
    out["discovery"]["B"]=diag_b
    shams=[]
    gain=None
    if pa is not None and pb is not None:
        # Independent model-score A and B projectors are fitted before
        # querying any held-out task labels.
        out["discovery"]["subspace_overlap_by_pass"] = [
            float(torch.trace(pa[t]@pb[t])/4.) for t in range(PASSES)
        ]
        native_u=bpd.wrappers["mix"].U.detach().clone()
        out["dose_matching"] = source_doses(acts_b,xdisc,native_u,pa,pb)
        gain=out["dose_matching"]["transfer_gain"]
        shams,info=local_matched_shams(
            acts_b,xdisc,native_u,ea,seed,n_proposals=proposals
        )
        out["sham_matching"]=info
    else:
        out["dose_matching"]={"gain_admissible":False,"branch_dose_admissible":False,
                              "status":"UNIDENTIFIABLE"}
        out["sham_matching"]={"eligible":False,"status":"UNIDENTIFIABLE"}
    # Source-only zero-head negative control, independent of evaluation labels.
    original_w=model.head.weight.detach().clone()
    original_b=model.head.bias.detach().clone()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    try:
        _, zero_g=discover(bpd,model,xda[:min(128,len(xda))])
        zero_p,_,zero_diag=projector_from_gradients(zero_g)
        out["negative_controls"]["zero_head"]={
            "max_gradient":float(zero_g.abs().max()),
            "projector_returned":zero_p is not None,
            "status":zero_diag["status"],
        }
    finally:
        with torch.no_grad():
            model.head.weight.copy_(original_w)
            model.head.bias.copy_(original_b)

    # Only now expose the true external test labels.
    ya,yb=task_labels(xe,"A"),task_labels(xe,"B")
    with torch.no_grad():
        ref_a,ref_wa=clean_forward(bpd,model,xea)
        ref_b,ref_wb=clean_forward(bpd,model,xeb)
        clean_a=evaluate(ref_a,ref_wa,ref_a,ref_wa,xe,ya)
        clean_b=evaluate(ref_b,ref_wb,ref_b,ref_wb,xe,yb)
    out["heldout"]["clean_A"]=clean_a
    out["heldout"]["clean_B"]=clean_b
    out["heldout"]["label_disagreement_private"] = float(
        (ya[context_masks(xe)["private"]] != yb[context_masks(xe)["private"]]).float().mean()
    )
    out["task_passed"]=(
        training["mix_weight_density"] >= .90
        and all(clean_a["by_context"][k]["accuracy"]>=.85 and
                clean_b["by_context"][k]["accuracy"]>=.85 for k in ("shared","private"))
        and out["heldout"]["label_disagreement_private"] >= .20
    )
    native_logits={}
    if pa is not None and pb is not None and gain is not None:
        with torch.no_grad():
            for key,x,p,g,ref,write,lab in (
                ("A_to_B",xeb,pa,gain,ref_b,ref_wb,yb),
                ("B_to_B",xeb,pb,1.,ref_b,ref_wb,yb),
                ("A_to_A",xea,pa,1.,ref_a,ref_wa,ya),
                ("B_to_A",xea,pb,1.,ref_a,ref_wa,ya),
            ):
                logits,writes=run_projector(bpd,model,x,p,gain=g)
                native_logits[key]=logits
                out["heldout"][key]=evaluate(logits,writes,ref,write,xe,lab)
            out["heldout"]["A_shams_to_B"]=[]
            for p in shams:
                logits,writes=run_projector(bpd,model,xeb,p,gain=gain)
                native_logits[f"sham_{len(out['heldout']['A_shams_to_B'])}"]=logits
                out["heldout"]["A_shams_to_B"].append(
                    evaluate(logits,writes,ref_b,ref_wb,xe,yb)
                )
        a=out["heldout"]["A_to_B"]["by_context"]
        b=out["heldout"]["B_to_B"]["by_context"]
        excess_shared=a["shared"]["nll_nats"]-b["shared"]["nll_nats"]
        excess_private=a["private"]["nll_nats"]-b["private"]["nll_nats"]
        out["primary"]={
            "shared_excess_nll_nats":float(excess_shared),
            "private_excess_nll_nats":float(excess_private),
            "specificity_difference_in_differences_nats":float(excess_private-excess_shared),
            "shared_sham_advantage_nats":float(
                np.median([c["by_context"]["shared"]["nll_nats"]
                           for c in out["heldout"]["A_shams_to_B"]])
                - a["shared"]["nll_nats"]
            ) if len(shams)==SHAMS else None,
        }
    else:
        out["heldout"]["A_shams_to_B"]=[]
    # Independently recover the two covariance projectors in an orthogonal
    # factor chart and verify equal physical write and complete heldout effects.
    set_basis(bpd,seed,"rotated")
    rotated_parity=full_mask_parity(bpd,model,torch.cat((xea[:64],xeb[:64])))
    out["bases"]["rotated"]={"closure":rotated_parity}
    if pa is not None and pb is not None:
        _,grad_ra=discover(bpd,model,xda)
        _,grad_rb=discover(bpd,model,xdb)
        pra,_,dra=projector_from_gradients(grad_ra)
        prb,_,drb=projector_from_gradients(grad_rb)
        out["bases"]["rotated"]["projector_A"]=dra
        out["bases"]["rotated"]["projector_B"]=drb
        if pra is not None and prb is not None:
            layer=bpd.wrappers["mix"]
            v,u=layer.V.detach().double(),layer.U.detach().double()
            w=layer.W_target.detach().double().T
            relative={}
            transport={}
            for k, original, rot in (("A",pa,pra),("B",pb,prb)):
                original_map=original@w
                rotated_map=v@rot@u
                relative[k]=float(
                    (original_map-rotated_map).norm(dim=(1,2)).max()
                    /w.norm().clamp_min(1e-12)
                )
                transport[k]=float(
                    (rot-transport_to_rotated(original,layer.V.detach())).abs().max()
                )
            errors={}
            with torch.no_grad():
                for key,p,g,x in (
                    ("A_to_B",pra,gain,xeb),("B_to_B",prb,1.,xeb),
                    ("A_to_A",pra,1.,xea),("B_to_A",prb,1.,xea),
                ):
                    logits,_=run_projector(bpd,model,x,p,gain=g)
                    errors[key]=float((logits-native_logits[key]).abs().max())
                for i,p in enumerate(shams):
                    transported=transport_to_rotated(p,layer.V.detach())
                    z,_=run_projector(bpd,model,xeb,transported,gain=gain)
                    errors[f"sham_{i}"]=float((z-native_logits[f"sham_{i}"]).abs().max())
            out["bases"]["rotated"]["comparison_to_native"]={
                "physical_map_error_by_projector":relative,
                "factor_transport_error_by_projector":transport,
                "heldout_logit_error_by_intervention":errors,
                "max_heldout_logit_error":max(errors.values()) if errors else None,
                "max_operator_relative_error":max(relative.values()),
            }

    checks={
        "task_model":out["task_passed"],
        "native_full_mask":native_parity["passed"],
        "rotated_full_mask":rotated_parity["passed"],
        "eigengap_both":pa is not None and pb is not None,
        "aligned_gain":bool(out["dose_matching"].get("gain_admissible",False)),
        "aligned_branch_dose":bool(out["dose_matching"].get("branch_dose_admissible",False)),
        "eight_shams":len(shams)==SHAMS and bool(out["sham_matching"].get("eligible",False)),
        "zero_head":out["negative_controls"]["zero_head"]["max_gradient"]==0
                    and not out["negative_controls"]["zero_head"]["projector_returned"],
    }
    comparison=out["bases"]["rotated"].get("comparison_to_native")
    checks["orthogonal_map_and_output"] = (
        comparison is not None
        and comparison["max_operator_relative_error"]<=1e-4
        and comparison["max_heldout_logit_error"]<=5e-5
    )
    out["admission"]=checks
    out["eligible"]=all(checks.values())
    out["status"]="ENGINEERING_ELIGIBLE" if out["eligible"] else "ENGINEERING_BLOCKED"
    out["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out["core_sha256"]=hashlib.sha256(Path(__file__).with_name("core.py").read_bytes()).hexdigest()
    out["protocol_sha256"]=hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest()
    return out


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=899)
    parser.add_argument("--steps",type=int,default=640)
    parser.add_argument("--discovery-size",type=int,default=1024)
    parser.add_argument("--evaluation-size",type=int,default=1024)
    parser.add_argument("--proposals",type=int,default=PROPOSALS)
    parser.add_argument("--out",type=Path)
    options=parser.parse_args()
    result=assess_root(options.seed,steps=options.steps,
                       discovery_size=options.discovery_size,
                       evaluation_size=options.evaluation_size,
                       proposals=options.proposals)
    output=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if options.out is None:
        print(output)
    else:
        options.out.parent.mkdir(parents=True,exist_ok=True)
        with options.out.open("x",encoding="utf-8") as f:
            f.write(output)


if __name__=="__main__":
    main()
