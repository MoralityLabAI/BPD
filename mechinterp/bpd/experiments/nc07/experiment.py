"""BPD-NC-07: clean-model KL-selected rank in a four-pass trained recurrence.

Prospective protocol committed FIRST at c798a5b50ee78a0b430a69dfe34a474b89919b37.
Only roots 700..707 are scored for engineering development; root 699 is excluded
smoke. Any future sham-calibration or confirmation roots remain sealed.

The rank selector, gradient covariance and physical sham matcher never access
external task labels. Labels enter teacher training and final evaluation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import (
    PASSES, WIDTH, sample_x, external_labels, binary_kl, parity,
)
from mechinterp.bpd.experiments.nc04.experiment import (
    DenseLooper, factor_basis, self_score_discovery, context_accuracies,
)
from mechinterp.bpd.experiments.nc05.experiment import (
    clean_dose_rms, projected_rollout, score_keep,
)
from mechinterp.bpd.experiments.nc03.experiment import basis_transport

ROOT=Path(__file__).resolve().parent
PROTO_HASH=hashlib.sha256((ROOT/"PROTOCOL.md").read_bytes()).hexdigest()
DEVELOPMENT=tuple(range(700,708))
SMOKE=699
RANKS=(2,3,4,6)
SELECTION={
    "mean_final_kl_max":0.027,
    "mean_per_pass_kl_max":0.050,
    "p95_final_kl_max":0.250,
    "model_sign_flip_fraction_max":0.040,
    "min_discovery_retained_write_fraction":0.350,
    "eigengap_min":1.001,
    "eig_floor":1e-12,
}
SHAM_ANGLES=(.45,.65,.85,1.05)
DOSE_TOL=.15
SHAM_PROPOSALS=1024
SHAMS=8


def train_model(seed:int,*,steps:int=384):
    """Exact NC05 model training contract, on completely fresh NC07 seeds."""
    if seed not in DEVELOPMENT and seed!=SMOKE:
        raise ValueError("NC07 admits excluded 699 or independent roots 700..707")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model=DenseLooper()
    source=sample_x(seed,1,2048)
    true_sign=torch.where(external_labels(source),1.,-1.)
    generator=torch.Generator().manual_seed(seed+1_737_701)
    opt=torch.optim.Adam(model.parameters(),lr=.005)
    for _ in range(steps):
        idx=torch.randint(len(source),(128,),generator=generator)
        logit=model(source[idx])[0][:,-1]
        loss=F.softplus(-2*true_sign[idx]*logit).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        preds=model(source)[0][:,-1]
        train_acc=float(((preds>=0)==external_labels(source)).float().mean())
        density=float((model.mix.weight.detach().abs()>=1e-4).float().mean())
    return model,{"seed":seed,"steps":steps,"train_size":len(source),
                  "batch_size":128,"lr":.005,
                  "train_accuracy":train_acc,"mix_density":density}


def rank_projectors(model_gradients:Tensor):
    """Model-score-only passwise covariance, nested top eigenspace ranks."""
    if model_gradients.ndim!=3 or model_gradients.shape[1:]!=(PASSES,WIDTH):
        raise ValueError("Expected [N,4,20] self-score gradients")
    g=model_gradients.double()
    cov=torch.einsum("nti,ntj->tij",g,g)/len(g)
    eig,vectors=torch.linalg.eigh(cov)
    result={}
    for rank in RANKS:
        basis=vectors[:,:,-rank:]
        projector=basis@basis.transpose(-1,-2)
        spectrum=[]
        for t in range(PASSES):
            retained=float(eig[t,-rank])
            discarded=float(eig[t,-rank-1])
            spectrum.append({
                "pass":t,"lambda_rank":retained,"lambda_next":discarded,
                "eigengap_ratio":retained/max(discarded,1e-12),
                "retained_second_moment_fraction":float(
                    eig[t,-rank:].sum()/eig[t].clamp_min(0).sum().clamp_min(1e-30)
                ),
            })
        eligible=all(
            x["lambda_rank"]>SELECTION["eig_floor"] and
            x["eigengap_ratio"]>=SELECTION["eigengap_min"]
            for x in spectrum
        )
        result[rank]={
            "rank":rank,"p":projector if eligible else None,
            "eigbasis":basis,"spectrum":spectrum,
            "eigengap_eligible":eligible,
            "symmetry_max":float((projector-projector.transpose(-1,-2)).abs().max()),
            "idempotence_max":float((projector@projector-projector).abs().max()),
        }
    return result


@torch.no_grad()
def clean_model_kl_selection(bpd:BPDModel,model:DenseLooper,
                             selector_x:Tensor,activations:Tensor,
                             candidates:dict):
    """Rank selection uses model output and clean discovery write only."""
    if selector_x.ndim!=2 or selector_x.shape[1]!=5:
        raise ValueError("Selector source must have 5 observed inputs")
    if len(selector_x)<1:
        raise ValueError("Selector source is empty")
    with bpd.target_capture():
        clean,_=model(selector_x)
    layer=bpd.wrappers["mix"]
    full_p=torch.eye(WIDTH,dtype=torch.float64).repeat(PASSES,1,1)
    full_dose=clean_dose_rms(activations,layer.U.detach(),full_p)
    if full_dose<=1e-12:
        raise ValueError("Clean physical full write RMS is zero")
    records={}
    for rank in RANKS:
        entry=candidates[rank]
        out={"rank":rank,"eigengap_eligible":entry["eigengap_eligible"],
             "spectrum":entry["spectrum"],"eligible":False}
        records[rank]=out
        if entry["p"] is None:
            out["reason"]="eigengap_failed"
            continue
        retained,_=projected_rollout(bpd,model,selector_x,entry["p"],1.)
        per_case=binary_kl(clean,retained).double().clamp_min(0)
        mean_by_pass=[float(p) for p in per_case.mean(0)]
        kl_final=per_case[:,-1]
        signflip=float(((clean[:,-1]>=0)!=(retained[:,-1]>=0)).float().mean())
        dose=clean_dose_rms(activations,layer.U.detach(),entry["p"])
        fraction=dose/full_dose
        allowed=bool(
            mean_by_pass[-1]<=SELECTION["mean_final_kl_max"] and
            max(mean_by_pass)<=SELECTION["mean_per_pass_kl_max"] and
            float(torch.quantile(kl_final,0.95))<=SELECTION["p95_final_kl_max"] and
            signflip<=SELECTION["model_sign_flip_fraction_max"] and
            fraction>=SELECTION["min_discovery_retained_write_fraction"]
        )
        out.update({
            "mean_kl_by_pass":mean_by_pass,
            "mean_final_kl_nats":mean_by_pass[-1],
            "p95_final_kl_nats":float(torch.quantile(kl_final,0.95)),
            "model_final_sign_flip_fraction":signflip,
            "discovery_retained_write_rms":dose,
            "discovery_full_write_rms":full_dose,
            "retained_write_fraction":fraction,
            "eligible":allowed,
            "reason":"meets_source_only_rule" if allowed else "source_only_threshold_failed",
        })
    chosen=next((r for r in RANKS if records[r]["eligible"]),None)
    return chosen,records


def local_physical_shams(acts:Tensor,u:Tensor,eigbasis:Tensor,seed:int,
                         *,proposals:int=SHAM_PROPOSALS,want:int=SHAMS):
    """First 8 qualifying NC05-style local rank-r physical-dose controls."""
    rank=eigbasis.shape[-1]
    if rank not in RANKS or eigbasis.shape[:2]!=(PASSES,WIDTH):
        raise ValueError("Bad rank or physical factor dimension")
    e=eigbasis.detach().cpu().double().numpy()
    x=acts.detach().cpu().double().numpy()
    U=u.detach().cpu().double().numpy()
    moment=np.einsum("nti,ntj->tij",x,x)/len(x)
    gram=U@U.T
    def dose(basis):
        total=0.
        for t in range(PASSES):
            b=basis[t]
            total+=float(np.trace((b.T@moment[t]@b)@(b.T@gram@b)))
        return float(np.sqrt(max(total,0.)/(PASSES*WIDTH)))
    target=dose(e)
    if target<=1e-12:
        return [],{"admitted":False,"reason":"zero_physical_write_dose",
                   "selected":0,"qualified":0}
    rng=np.random.default_rng(seed+7_071_701+139*rank)
    accepted,qualified=[],0
    for ix in range(proposals):
        theta=SHAM_ANGLES[int(rng.integers(len(SHAM_ANGLES)))]
        candidate=np.empty_like(e)
        for t in range(PASSES):
            z=rng.standard_normal((WIDTH,rank))
            z=z-e[t]@(e[t].T@z)
            complement,_=np.linalg.qr(z,mode="reduced")
            candidate[t]=np.cos(theta)*e[t]+np.sin(theta)*complement
        error=abs(dose(candidate)-target)/target
        if error<=DOSE_TOL:
            qualified+=1
            if len(accepted)<want:
                p=torch.from_numpy(candidate@candidate.transpose(0,2,1)).double()
                accepted.append({
                    "p":p,"proposal_index":ix,
                    "relative_discovery_dose_error":error,
                    "angle_radians":theta,
                })
    return accepted,{
        "admitted":len(accepted)==want,"qualified":qualified,
        "selected":len(accepted),"proposal_budget":proposals,
        "target_discovery_write_rms":target,
        "selected_indices":[c["proposal_index"] for c in accepted],
        "selected_relative_dose_errors":[c["relative_discovery_dose_error"] for c in accepted],
    }


def physical_gauge_audit(bpd:BPDModel,model:DenseLooper,selector_x:Tensor,
                         seed:int,p:Tensor):
    """Exact same physical projected write under orthonormal factor rotations."""
    factor_basis(bpd,"native",seed)
    native=bpd.wrappers["mix"]
    a_native=native.V.detach().double()@p@native.U.detach().double()
    with torch.no_grad():
        native_logits,_=projected_rollout(bpd,model,selector_x[:64],p,1.)
    factor_basis(bpd,"rotated",seed)
    layer=bpd.wrappers["mix"]
    prot=basis_transport(p,layer.V.detach())
    a_rot=layer.V.detach().double()@prot@layer.U.detach().double()
    with torch.no_grad():
        rot_logits,_=projected_rollout(bpd,model,selector_x[:64],prot,1.)
    factor_basis(bpd,"native",seed)
    return {
        "physical_operator_rel_error":float(
            (a_rot-a_native).norm()/native.W_target.detach().double().norm().clamp_min(1e-12)
        ),
        "max_rotated_logit_error":float((native_logits-rot_logits).abs().max()),
    }


def evaluate_root(seed:int,*,steps:int=384,derivative_n:int=1024,
                  selection_n:int=512,heldout_n:int=1024,
                  sham_proposals:int=SHAM_PROPOSALS,shams:int=SHAMS):
    torch.set_num_threads(1)
    model,training=train_model(seed,steps=steps)
    xd=sample_x(seed,2,derivative_n)
    xs=sample_x(seed,4,selection_n)
    bpd=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                 ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bpd,"native",seed)
    full_parity=parity(bpd,model,xs[:min(64,len(xs))])
    acts,grad,hits,discovery=self_score_discovery(bpd,model,xd)
    candidates=rank_projectors(grad)
    chosen,cards=clean_model_kl_selection(bpd,model,xs,acts,candidates)
    record={
        "study":"BPD-NC-07","phase":"DEVELOPMENT",
        "root":seed,"protocol_sha256":PROTO_HASH,
        "training":training,
        "split_sizes":{"train":2048,"derivative_discovery":len(xd),
                       "rank_selection":len(xs),"final_heldout":heldout_n},
        "score_nonzero_count":discovery["model_score_nonzero_count"],
        "full_mask_closure":full_parity,
        "selected_rank":chosen,
        "selector_by_rank":{str(k):v for k,v in cards.items()},
        "selected_sham_admission":None,
        "selected_gauge":None,
        "heldout":None,
    }
    selected_shams=[]
    if chosen is not None:
        selected_shams,admission=local_physical_shams(
            acts,bpd.wrappers["mix"].U.detach(),
            candidates[chosen]["eigbasis"],seed,
            proposals=sham_proposals,want=shams,
        )
        record["selected_sham_admission"]=admission
        record["selected_gauge"]=physical_gauge_audit(
            bpd,model,xs,seed,candidates[chosen]["p"]
        )
    # Final evaluation and its true task labels are opened ONLY after rank
    # and all sham choices are fully locked using independent source splits.
    xe=sample_x(seed,3,heldout_n)
    ye=external_labels(xe)
    task=context_accuracies(model,xe,ye)
    record["task_context_accuracy"]=task
    source_admissible=bool(
        min(task.values())>=.85 and training["mix_density"]>=.90
        and full_parity["passed"]
    )
    with torch.no_grad(),bpd.target_capture():
        clean,_=model(xe,capture=True)
        clean_writes=torch.stack(model.writes,1)
    sign=torch.where(ye,1.,-1.)
    record["clean_final_nll_nats"]=float(F.softplus(-2*sign*clean[:,-1]).mean())
    record["clean_final_accuracy"]=float(((clean[:,-1]>=0)==ye).float().mean())
    scores={}
    if chosen is not None:
        for label,r in (("selected",chosen),("fixed_rank3",3),("fixed_rank4",4)):
            if candidates[r]["p"] is None:
                scores[label]=None
            else:
                scores[label]=score_keep(
                    bpd,model,xe,ye,clean,clean_writes,candidates[r]["p"],1.
                )
        if selected_shams and scores.get("selected") is not None:
            nullscores=[
                score_keep(bpd,model,xe,ye,clean,clean_writes,item["p"],1.)
                for item in selected_shams
            ]
            record["sham_heldout_nll_nats"]=[
                float(v["nll_nats"]) for v in nullscores
            ]
            record["median_sham_minus_selected_nll_nats"]=float(
                np.median(record["sham_heldout_nll_nats"])-
                scores["selected"]["nll_nats"]
            )
    record["heldout"]=scores
    gauge=record["selected_gauge"] or {}
    record["eligible"]=bool(
        source_admissible and chosen is not None
        and candidates[chosen]["p"] is not None
        and len(selected_shams)==shams
        and scores.get("selected") is not None
        and scores.get("fixed_rank3") is not None
        and scores.get("fixed_rank4") is not None
        and gauge.get("physical_operator_rel_error",float("inf"))<=1e-4
        and gauge.get("max_rotated_logit_error",float("inf"))<=5e-5
    )
    record["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return record


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=SMOKE)
    ap.add_argument("--steps",type=int,default=32)
    ap.add_argument("--discovery",type=int,default=128)
    ap.add_argument("--selection",type=int,default=128)
    ap.add_argument("--heldout",type=int,default=128)
    ap.add_argument("--proposals",type=int,default=128)
    ap.add_argument("--shams",type=int,default=4)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    r=evaluate_root(args.seed,steps=args.steps,derivative_n=args.discovery,
                    selection_n=args.selection,heldout_n=args.heldout,
                    sham_proposals=args.proposals,shams=args.shams)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"root":r["root"],"rank":r["selected_rank"],
                      "eligible":r["eligible"]},sort_keys=True))


if __name__=="__main__":
    main()
