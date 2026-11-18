"""BPD-NC-08: passwise rank allocation, with all choices source-only.

The prospective PROTOCOL.md was committed before any NC-08 source code and
before training roots 800..807. Only 799 is an excluded engineering smoke.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import (
    PASSES, WIDTH, sample_x, external_labels, binary_kl, parity
)
from mechinterp.bpd.experiments.nc04.experiment import (
    DenseLooper, factor_basis, self_score_discovery, context_accuracies
)
from mechinterp.bpd.experiments.nc05.experiment import (
    clean_dose_rms, projected_rollout, score_keep
)
from mechinterp.bpd.experiments.nc03.experiment import basis_transport

ROOT=Path(__file__).resolve().parent
PROTO_HASH=hashlib.sha256((ROOT/"PROTOCOL.md").read_bytes()).hexdigest()
ROOTS=tuple(range(800,808))
SMOKE=799
OPTIONS=(2,3,4,5,6)
BUDGETS=(8,10,12,14)
SHAM_ANGLES=(.45,.65,.85,1.05)
SOURCE_THRESHOLDS={
    "mean_final_kl_max":.027,
    "per_pass_mean_kl_max":.05,
    "p95_final_kl_max":.25,
    "sign_flip_max":.04,
    "retained_full_write_fraction_min":.35,
    "eigenvalue_min":1e-12,
    "eigengap_ratio_min":1.001,
}
SHAMS=8
PROPOSALS=2048
DOSE_TOL=.15


def train_model(seed:int, *,steps:int=384):
    """Reuse the NC05/06/07 20D dense looper training distribution."""
    if seed not in ROOTS and seed!=SMOKE:
        raise ValueError("NC08 admits only engineering seed 799 and roots 800..807")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model=DenseLooper()
    train_x=sample_x(seed,1,2048)
    y=torch.where(external_labels(train_x),1.,-1.)
    generator=torch.Generator().manual_seed(seed+1_737_701)
    optimizer=torch.optim.Adam(model.parameters(),lr=.005)
    for _ in range(steps):
        ix=torch.randint(len(train_x),(128,),generator=generator)
        pred=model(train_x[ix])[0][:,-1]
        loss=F.softplus(-2*y[ix]*pred).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        out=model(train_x)[0][:,-1]
        accuracy=float(((out>=0)==external_labels(train_x)).float().mean())
        density=float((model.mix.weight.detach().abs()>=1e-4).float().mean())
    return model,{"seed":seed,"steps":steps,"train_size":len(train_x),
                  "train_accuracy":accuracy,"mix_density":density,
                  "training_loss_last_batch":float(loss.detach())}


def passwise_source_candidates(reads:Tensor,gradients:Tensor):
    """Source-only first-order omitted-score loss and eigengap admission."""
    if (reads.ndim!=3 or reads.shape[1:]!=(PASSES,WIDTH) or
        gradients.shape!=reads.shape):
        raise ValueError("Expected clean read and gradient [N,4,20]")
    z=reads.double()
    g=gradients.double()
    c=torch.einsum("nti,ntj->tij",g,g)/len(g)
    eig,vec=torch.linalg.eigh(c)
    identity=torch.eye(WIDTH,dtype=torch.float64)
    out={}
    for t in range(PASSES):
        per={}
        for rank in OPTIONS:
            basis=vec[t,:,-rank:]
            p=basis@basis.T
            eig_r=float(eig[t,-rank])
            eig_next=float(eig[t,-rank-1])
            ratio=eig_r/max(eig_next,1e-12)
            eligible=(eig_r>SOURCE_THRESHOLDS["eigenvalue_min"] and
                      ratio>=SOURCE_THRESHOLDS["eigengap_ratio_min"])
            residue=z[:,t,:]@(identity-p)
            scalar=(g[:,t,:]*residue).sum(dim=1)
            omitted=float(scalar.square().mean())
            per[rank]={
                "rank":rank,"eligible":bool(eligible),
                "eigvalue_at_rank":eig_r,"eigvalue_next":eig_next,
                "eigengap_ratio":ratio,
                "squared_local_omitted_model_score":omitted,
                "projector":p,"basis":basis,
                "idempotence_max":float((p@p-p).abs().max())
            }
        out[t]=per
    return out


def candidate_budget_knapsack(candidates:dict):
    """Exhaustive exact finite multiple-choice cost knapsack, fixed tie rule."""
    if set(candidates)!=set(range(PASSES)):
        raise ValueError("Need exactly 4 passwise source candidates")
    choices={}
    for budget in BUDGETS:
        best=None
        for tup in itertools.product(OPTIONS,repeat=PASSES):
            if sum(tup)!=budget or any(
                not candidates[t][rank]["eligible"] for t,rank in enumerate(tup)
            ):
                continue
            score=sum(candidates[t][rank]["squared_local_omitted_model_score"]
                      for t,rank in enumerate(tup))
            if (best is None or score<best["proxy_loss"]-1e-12 or
                (abs(score-best["proxy_loss"])<=1e-12 and
                 tup<best["ranks"])):
                best={"budget":budget,"ranks":tup,"proxy_loss":score}
        choices[budget]=best
    return choices


def tuple_projector(candidates:dict,ranks:tuple[int,...]):
    if len(ranks)!=PASSES or any(rank not in OPTIONS for rank in ranks):
        raise ValueError("Unregistered rank tuple")
    return torch.stack([candidates[t][rank]["projector"]
                        for t,rank in enumerate(ranks)])


def tuple_basis(candidates:dict,ranks:tuple[int,...]):
    if len(ranks)!=PASSES:
        raise ValueError("Wrong pass count")
    return [candidates[t][rank]["basis"] for t,rank in enumerate(ranks)]


@torch.no_grad()
def source_selector(bpd:BPDModel,model:DenseLooper,selector_x:Tensor,
                    discovery_reads:Tensor,discovery_acts:Tensor,
                    pass_candidates:dict):
    """Select smallest passing total rank without external labels."""
    budgets=candidate_budget_knapsack(pass_candidates)
    with bpd.target_capture():
        clean,_=model(selector_x)
    layer=bpd.wrappers["mix"]
    full_p=torch.eye(WIDTH,dtype=torch.float64).repeat(PASSES,1,1)
    full_dose=clean_dose_rms(discovery_acts,layer.U.detach(),full_p)
    if full_dose<=1e-12:
        raise ValueError("Full clean physical write dose zero")
    rec={}
    for budget in BUDGETS:
        proposal=budgets[budget]
        item={
            "budget":budget,"candidate_exists":proposal is not None,
            "eligible":False,
            "reason":"no_source_eligible_tuple" if proposal is None else "pending"
        }
        rec[budget]=item
        if proposal is None:
            continue
        ranks=proposal["ranks"]
        p=tuple_projector(pass_candidates,ranks)
        retained,_=projected_rollout(bpd,model,selector_x,p,1.)
        kl=binary_kl(clean,retained).double().clamp_min(0)
        perpass=[float(q) for q in kl.mean(0)]
        tail=float(torch.quantile(kl[:,-1],.95))
        signflip=float(((clean[:,-1]>=0)!=(retained[:,-1]>=0)).float().mean())
        dose=clean_dose_rms(discovery_acts,layer.U.detach(),p)
        fraction=dose/full_dose
        passed=bool(
            perpass[-1]<=SOURCE_THRESHOLDS["mean_final_kl_max"] and
            max(perpass)<=SOURCE_THRESHOLDS["per_pass_mean_kl_max"] and
            tail<=SOURCE_THRESHOLDS["p95_final_kl_max"] and
            signflip<=SOURCE_THRESHOLDS["sign_flip_max"] and
            fraction>=SOURCE_THRESHOLDS["retained_full_write_fraction_min"]
        )
        item.update({
            "ranks":list(ranks),"source_proxy_loss":proposal["proxy_loss"],
            "mean_kl_by_pass":perpass,"mean_final_kl_nats":perpass[-1],
            "p95_final_kl_nats":tail,
            "model_decision_flip_fraction":signflip,
            "retained_write_fraction":fraction,
            "clean_retained_write_rms":dose,
            "clean_full_write_rms":full_dose,
            "eligible":passed,
            "reason":"meets_source_only_kl" if passed else "source_only_threshold_failed",
        })
    selected=next((b for b in BUDGETS if rec[b]["eligible"]),None)
    return selected,rec,budgets


def local_dose_matched_shams(activations:Tensor,u:Tensor,
                             bases:list[Tensor],seed:int,
                             *,proposals:int=PROPOSALS,want:int=SHAMS):
    """One fixed-angle per proposal; pass-specific rank and QR complements."""
    if len(bases)!=PASSES:
        raise ValueError("Need four pass-indexed bases")
    e=[b.detach().double().cpu().numpy() for b in bases]
    if any(q.shape[0]!=WIDTH or q.shape[1] not in OPTIONS for q in e):
        raise ValueError("Unsupported physical pass rank")
    x=activations.detach().double().cpu().numpy()
    uu=u.detach().double().cpu().numpy()
    moment=np.einsum("nti,ntj->tij",x,x)/len(x)
    gram=uu@uu.T
    def dose(vectors):
        s=0.
        for t in range(PASSES):
            q=vectors[t]
            s+=float(np.trace((q.T@moment[t]@q)@(q.T@gram@q)))
        return float(np.sqrt(max(s,0.)/(PASSES*WIDTH)))
    reference=dose(e)
    if reference<=1e-12:
        return [],{"eligible":False,"reason":"zero_source_dose","qualified":0,
                   "selected":0,"proposals":proposals}
    rng=np.random.default_rng(8_080_000+seed)
    admitted,qualified=[],0
    for i in range(proposals):
        theta=SHAM_ANGLES[int(rng.integers(0,len(SHAM_ANGLES)))]
        rotated=[]
        for t in range(PASSES):
            v=e[t]
            rank=v.shape[1]
            z=rng.standard_normal((WIDTH,rank))
            z=z-v@(v.T@z)
            comp,_=np.linalg.qr(z,mode="reduced")
            rotated.append(np.cos(theta)*v+np.sin(theta)*comp)
        matched=dose(rotated)
        relative=abs(matched-reference)/reference
        if relative<=DOSE_TOL:
            qualified+=1
            if len(admitted)<want:
                p=torch.stack([torch.from_numpy(v@v.T).double()
                               for v in rotated])
                admitted.append({
                    "p":p,"index":i,
                    "angle":theta,"relative_dose_error":relative
                })
    return admitted,{
        "eligible":len(admitted)==want,"proposals":proposals,
        "qualified":qualified,"selected":len(admitted),
        "reference_source_write_rms":reference,
        "first_selected_indices":[x["index"] for x in admitted],
        "relative_dose_errors":[x["relative_dose_error"] for x in admitted],
    }


@torch.no_grad()
def physical_gauge_audit(bpd:BPDModel,model:DenseLooper,source_x:Tensor,
                         seed:int,p:Tensor):
    factor_basis(bpd,"native",seed)
    native=bpd.wrappers["mix"]
    reference=native.V.detach().double()@p@native.U.detach().double()
    logits_native,_=projected_rollout(bpd,model,source_x[:64],p,1.)
    factor_basis(bpd,"rotated",seed)
    layer=bpd.wrappers["mix"]
    transformed=basis_transport(p,layer.V.detach())
    operator=layer.V.detach().double()@transformed@layer.U.detach().double()
    logits_rotated,_=projected_rollout(bpd,model,source_x[:64],transformed,1.)
    factor_basis(bpd,"native",seed)
    return {
        "relative_physical_operator_error":float(
            (operator-reference).norm()/native.W_target.detach().double().norm().clamp_min(1e-12)
        ),
        "max_full_rollout_logit_error":float((logits_native-logits_rotated).abs().max())
    }


def evaluate_root(seed:int,*,steps:int=384,derivative_n:int=1024,
                  selection_n:int=512,heldout_n:int=1024,
                  proposals:int=PROPOSALS,shams:int=SHAMS):
    torch.set_num_threads(1)
    model,training=train_model(seed,steps=steps)
    xd=sample_x(seed,2,derivative_n)
    xs=sample_x(seed,4,selection_n)
    bpd=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                 ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bpd,"native",seed)
    closure=parity(bpd,model,xs[:min(len(xs),64)])
    z,g,hits,source_score=self_score_discovery(bpd,model,xd)
    with bpd.target_capture():
        model(xd,capture=True)
        physical_acts=torch.stack(model.acts,1).detach()
    candidates=passwise_source_candidates(z,g)
    chosen,cards,budgets=source_selector(bpd,model,xs,z,physical_acts,candidates)
    selected_ranks=tuple(cards[chosen]["ranks"]) if chosen is not None else None
    record={
        "study":"BPD-NC-08","phase":"DEVELOPMENT",
        "root":seed,"protocol_sha256":PROTO_HASH,
        "training":training,
        "splits":{"train":2048,"discovery":len(xd),
                  "selection":len(xs),"heldout":heldout_n},
        "clean_closure":closure,
        "score_nonzero_count":source_score["model_score_nonzero_count"],
        "passwise_spectral_and_proxy":{
            str(t):{str(r):{k:v for k,v in candidates[t][r].items()
                            if k not in ("projector","basis")}
                    for r in OPTIONS} for t in range(PASSES)
        },
        "candidates_by_budget":{str(b):cards[b] for b in BUDGETS},
        "selected_total_rank":chosen,"selected_rank_tuple":list(selected_ranks)
            if selected_ranks is not None else None,
        "sham_admission":None,"gauge":None,"heldout":{},
    }
    shams_selected=[]
    projector=tuple_projector(candidates,selected_ranks) if selected_ranks else None
    if projector is not None:
        shams_selected,admission=local_dose_matched_shams(
            physical_acts,bpd.wrappers["mix"].U.detach(),
            tuple_basis(candidates,selected_ranks),seed,
            proposals=proposals,want=shams
        )
        record["sham_admission"]=admission
        record["gauge"]=physical_gauge_audit(bpd,model,xs,seed,projector)
    # Labels are retrieved only after the whole model-only selector and
    # every sham proposal choice have been completed.
    xe=sample_x(seed,3,heldout_n)
    ye=external_labels(xe)
    contexts=context_accuracies(model,xe,ye)
    with bpd.target_capture():
        clean,_=model(xe,capture=True)
        clean_writes=torch.stack(model.writes,1)
    truth_sign=torch.where(ye,1.,-1.)
    record["clean_final_nll_nats"]=float(F.softplus(-2*truth_sign*clean[:,-1]).mean())
    record["clean_final_accuracy"]=float(((clean[:,-1]>=0)==ye).float().mean())
    record["task_context_accuracy"]=contexts
    frozen_controls={
        "selected":projector,
        "flat_rank3":tuple_projector(candidates,(3,3,3,3)),
        "flat_rank4":tuple_projector(candidates,(4,4,4,4)),
        "dp_budget12":tuple_projector(candidates,budgets[12]["ranks"]) if budgets[12]
                      is not None else None,
    }
    for name,p in frozen_controls.items():
        record["heldout"][name]=score_keep(
            bpd,model,xe,ye,clean,clean_writes,p,1.) if p is not None else None
    if projector is not None and shams_selected:
        local=[score_keep(bpd,model,xe,ye,clean,clean_writes,s["p"],1.)
               for s in shams_selected]
        record["sham_heldout_nlls"]=[x["nll_nats"] for x in local]
        record["median_sham_minus_selected_nll_nats"]=float(
            np.median(record["sham_heldout_nlls"])-
            record["heldout"]["selected"]["nll_nats"]
        )
    gauge=record["gauge"] or {}
    record["eligible"]=bool(
        min(contexts.values())>=.85 and training["mix_density"]>=.9 and
        closure["passed"] and projector is not None and
        record["sham_admission"]["selected"]==shams and
        record["heldout"]["selected"] is not None and
        record["heldout"]["flat_rank3"] is not None and
        record["heldout"]["flat_rank4"] is not None and
        gauge.get("relative_physical_operator_error",float("inf"))<=1e-4 and
        gauge.get("max_full_rollout_logit_error",float("inf"))<=5e-5
    )
    record["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return record


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=SMOKE)
    ap.add_argument("--steps",type=int,default=16)
    ap.add_argument("--discovery",type=int,default=128)
    ap.add_argument("--selection",type=int,default=128)
    ap.add_argument("--heldout",type=int,default=128)
    ap.add_argument("--proposals",type=int,default=128)
    ap.add_argument("--shams",type=int,default=2)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    rec=evaluate_root(args.seed,steps=args.steps,derivative_n=args.discovery,
                      selection_n=args.selection,heldout_n=args.heldout,
                      proposals=args.proposals,shams=args.shams)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"root":rec["root"],"selected_rank_tuple":rec["selected_rank_tuple"],
                      "selected_total_rank":rec["selected_total_rank"],
                      "eligible":rec["eligible"]},sort_keys=True))


if __name__=="__main__":
    main()
