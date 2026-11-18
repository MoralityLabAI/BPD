"""BPD-NC-09 fixed-12-slot passwise causal allocation, source-registered.

The original PROTOCOL preceded this code and every scientific/development model.
Developer smoke 899; first eight NC09 cross-task-transfer branch is untouched.
Here model roots 900..915 are independent 16-root engineering development ONLY.
No external task labels enter passwise allocation, gain, shams, or gauge checks.
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
    PASSES, WIDTH, sample_x, external_labels, binary_kl, parity,
)
from mechinterp.bpd.experiments.nc04.experiment import (
    DenseLooper, factor_basis, self_score_discovery, context_accuracies,
)
from mechinterp.bpd.experiments.nc05.experiment import clean_dose_rms, score_outputs
from mechinterp.bpd.experiments.nc03.experiment import basis_transport
from mechinterp.bpd.experiments.nc08.experiment import (
    OPTIONS, passwise_source_candidates, candidate_budget_knapsack,
    tuple_projector, tuple_basis,
)

ROOT=Path(__file__).resolve().parent
PROTO_SHA=hashlib.sha256((ROOT/"PROTOCOL.md").read_bytes()).hexdigest()
ROOTS=tuple(range(900,916))
SMOKE=899
TOTAL_RANK=12
GAINS=(.5,1.5)
ANGLES=(.45,.65,.85,1.05)
SHAMS=8
PROPOSALS=4096
DOSE_TOL=.15
ARM_KEYS=("dp12","flat3","flat3_dose","flat4","reverse")


def train_model(seed:int, *,steps:int=384):
    if seed not in ROOTS and seed!=SMOKE:
        raise ValueError("NC09 only admits excluded 899 and new roots 900..915")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model=DenseLooper()
    xt=sample_x(seed,1,2048)
    signs=torch.where(external_labels(xt),1.,-1.)
    gen=torch.Generator().manual_seed(seed+1_737_701)
    optimizer=torch.optim.Adam(model.parameters(),lr=.005)
    for _ in range(steps):
        ix=torch.randint(len(xt),(128,),generator=gen)
        logits=model(xt[ix])[0][:,-1]
        loss=F.softplus(-2*signs[ix]*logits).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred=model(xt)[0][:,-1]
        acc=float(((pred>=0)==external_labels(xt)).float().mean())
        density=float((model.mix.weight.detach().abs()>=1e-4).float().mean())
    return model, {
        "root":seed,"steps":steps,"train_size":len(xt),
        "last_train_loss":float(loss.detach()),
        "train_accuracy":acc,"mix_density":density,
    }


def optimal_fixed12(pass_candidates:dict):
    """Exact NC08 source-only DP at a single frozen total rank 12."""
    # This method does not accept true labels, heldout outcomes or model scores.
    candidate=candidate_budget_knapsack(pass_candidates)[TOTAL_RANK]
    if candidate is None:
        return None
    ranks=tuple(candidate["ranks"])
    assert sum(ranks)==TOTAL_RANK and len(ranks)==PASSES
    return {
        "ranks":ranks,
        "total_rank":TOTAL_RANK,
        "source_proxy_loss":float(candidate["proxy_loss"]),
        "method":"NC08_first_order_omitted_self_score_exact_12_slot_DP",
    }


def fixed_arm_projectors(pass_candidates:dict, optimum:dict|None):
    def admissible(ranks):
        return all(pass_candidates[t][ranks[t]]["eligible"]
                   for t in range(PASSES))
    result={}
    specs={
        "dp12":optimum["ranks"] if optimum is not None else None,
        "flat3":(3,3,3,3),
        "flat3_dose":(3,3,3,3),
        "flat4":(4,4,4,4),
        "reverse":tuple(reversed(optimum["ranks"])) if optimum else None,
    }
    for arm,ranks in specs.items():
        eligible=bool(ranks is not None and admissible(ranks))
        result[arm]={
            "ranks":ranks,
            "total_rank":sum(ranks) if ranks is not None else None,
            "eligible":eligible,
            "projector":tuple_projector(pass_candidates,ranks)
                if eligible else None,
        }
    return result


@torch.no_grad()
def closed_loop_rollout(bpd:BPDModel,model:DenseLooper,x:Tensor,
                        projector:Tensor, gain:float):
    """True four-pass native physical override-read, including arbitrary gain."""
    if projector.shape!=(PASSES,WIDTH,WIDTH) or gain<=0:
        raise ValueError("Frozen physical projector or gain invalid")
    layer=bpd.wrappers["mix"]
    reads=(layer.V.detach().double() @ (float(gain)*projector.double())).float()
    with bpd.intervene("mix", tuple(range(WIDTH)), mode="override_read",
                       read_override=reads, scale=0.):
        logits,_=model(x,capture=True)
        writes=torch.stack(model.writes,1)
    return logits,writes


@torch.no_grad()
def model_only_diagnostic(bpd:BPDModel,model:DenseLooper,
                          selector_x:Tensor,arm:dict):
    """No true labels. All arms fixed before these independent diagnostics."""
    with bpd.target_capture():
        clean,_=model(selector_x,capture=True)
    kept,_=closed_loop_rollout(bpd,model,selector_x,
                              arm["projector"],arm["gain"])
    kl=binary_kl(clean,kept).double().clamp_min(0)
    return {
        "mean_kl_by_pass":[float(z) for z in kl.mean(0)],
        "p95_final_kl":float(torch.quantile(kl[:,-1],.95)),
        "sign_flip_fraction":float(
            ((clean[:,-1]>=0)!=(kept[:,-1]>=0)).float().mean()
        ),
        "source_samples":len(selector_x),
    }


@torch.no_grad()
def gauge_audit(bpd:BPDModel,model:DenseLooper,x:Tensor,
                seed:int,p:Tensor,gain:float):
    """Orthogonal factor coordinate transport, preserving real physical write."""
    factor_basis(bpd,"native",seed)
    layer=bpd.wrappers["mix"]
    native_map=float(gain)*(layer.V.detach().double()@p@layer.U.detach().double())
    nat,_=closed_loop_rollout(bpd,model,x[:64],p,gain)
    factor_basis(bpd,"rotated",seed)
    rot_layer=bpd.wrappers["mix"]
    pr=basis_transport(p,rot_layer.V.detach())
    rotated_map=float(gain)*(rot_layer.V.detach().double()@pr@rot_layer.U.detach().double())
    rotated,_=closed_loop_rollout(bpd,model,x[:64],pr,gain)
    factor_basis(bpd,"native",seed)
    return {
        "gain":float(gain),
        "physical_map_relative_error":float(
            (native_map-rotated_map).norm()/
            layer.W_target.detach().double().norm().clamp_min(1e-12)
        ),
        "max_complete_rollout_logit_error":float((nat-rotated).abs().max()),
    }


def source_matched_local_shams(activations:Tensor,u:Tensor,
                              bases:list[Tensor],seed:int,
                              *,proposals:int=PROPOSALS,want:int=SHAMS):
    """Same passwise-rank tuple, first eight 15%-write-dose source shams."""
    if len(bases)!=PASSES:
        raise ValueError("Four source-only eigenspace bases required")
    vecs=[v.detach().double().cpu().numpy() for v in bases]
    if any(v.shape[0]!=WIDTH or v.shape[1] not in OPTIONS for v in vecs):
        raise ValueError("Unregistered pass rank")
    a=activations.detach().double().cpu().numpy()
    U=u.detach().double().cpu().numpy()
    moment=np.einsum("nti,ntj->tij",a,a)/len(a)
    gram=U@U.T
    def dose(basis):
        amount=0.
        for t in range(PASSES):
            q=basis[t]
            amount+=float(np.trace((q.T@moment[t]@q)@(q.T@gram@q)))
        return float(np.sqrt(max(amount,0.)/(PASSES*WIDTH)))
    target=dose(vecs)
    if target<=1e-12:
        return [],{"eligible":False,"reason":"zero_source_dose",
                   "qualified":0,"selected":0,"proposals":proposals}
    rng=np.random.default_rng(9_090_000+seed)
    qualified,accepted=0,[]
    for i in range(proposals):
        theta=ANGLES[int(rng.integers(0,len(ANGLES)))]
        perturbed=[]
        for v in vecs:
            rank=v.shape[1]
            z=rng.standard_normal((WIDTH,rank))
            z=z-v@(v.T@z)
            q,_=np.linalg.qr(z,mode="reduced")
            perturbed.append(np.cos(theta)*v+np.sin(theta)*q)
        rel=abs(dose(perturbed)-target)/target
        if rel<=DOSE_TOL:
            qualified+=1
            if len(accepted)<want:
                pp=torch.stack([torch.from_numpy(v@v.T).double()
                                for v in perturbed])
                accepted.append({
                    "projector":pp,"proposal_index":i,
                    "angle_radians":theta,"source_dose_relative_error":rel,
                })
    return accepted,{
        "eligible":len(accepted)==want,
        "proposals":proposals,"qualified":qualified,"selected":len(accepted),
        "source_dp12_write_rms":target,
        "sham_indices":[x["proposal_index"] for x in accepted],
        "sham_dose_errors":[x["source_dose_relative_error"] for x in accepted],
    }


@torch.no_grad()
def evaluate_task(bpd:BPDModel,model:DenseLooper,heldout_x:Tensor,
                  y:Tensor,clean:Tensor,clean_writes:Tensor,arm:dict):
    kept,writes=closed_loop_rollout(
        bpd,model,heldout_x,arm["projector"],arm["gain"]
    )
    return score_outputs(clean,clean_writes,kept,writes,y)


def evaluate_root(seed:int,*,steps:int=384,derivative_n:int=1024,
                  selector_n:int=512,heldout_n:int=1024,
                  sham_proposals:int=PROPOSALS,shams:int=SHAMS):
    torch.set_num_threads(1)
    model,training=train_model(seed,steps=steps)
    xd=sample_x(seed,2,derivative_n)
    xs=sample_x(seed,4,selector_n)
    bpd=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                 ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bpd,"native",seed)
    clean_closure=parity(bpd,model,xs[:min(64,len(xs))])
    z,g,hits,discovery=self_score_discovery(bpd,model,xd)
    with bpd.target_capture():
        model(xd,capture=True)
        acts=torch.stack(model.acts,1).detach()
    pass_candidates=passwise_source_candidates(z,g)
    optimum=optimal_fixed12(pass_candidates)
    arms=fixed_arm_projectors(pass_candidates,optimum)
    U=bpd.wrappers["mix"].U.detach()
    dpdose=clean_dose_rms(acts,U,arms["dp12"]["projector"]) if arms["dp12"]["eligible"] else None
    flatdose=clean_dose_rms(acts,U,arms["flat3"]["projector"]) if arms["flat3"]["eligible"] else None
    alpha=(dpdose/flatdose if dpdose is not None and flatdose is not None
           and flatdose>1e-12 else None)
    dose_admissible=bool(alpha is not None and GAINS[0]<=alpha<=GAINS[1])
    for key,item in arms.items():
        item["gain"]=float(alpha) if key=="flat3_dose" and alpha is not None else 1.
        item["source_dose_rms"]=(
            clean_dose_rms(acts,U,item["projector"])*item["gain"]
            if item["eligible"] else None
        )
        item["selector_model_only"]=(
            model_only_diagnostic(bpd,model,xs,item)
            if item["eligible"] else None
        )
    if not dose_admissible:
        arms["flat3_dose"]["eligible"]=False
    shams_selected=[]
    sham_admission=None
    if optimum is not None:
        shams_selected,sham_admission=source_matched_local_shams(
            acts,U,tuple_basis(pass_candidates,optimum["ranks"]),seed,
            proposals=sham_proposals,want=shams
        )
    gauges={}
    for key in ("dp12","flat3","flat3_dose","flat4"):
        if arms[key]["eligible"]:
            gauges[key]=gauge_audit(bpd,model,xs,seed,arms[key]["projector"],
                                   arms[key]["gain"])
    factor_basis(bpd,"native",seed)
    record={
        "study":"BPD-NC-09-fixed12","phase":"DEVELOPMENT",
        "root":seed,"protocol_sha256":PROTO_SHA,
        "training":training,
        "splits":{"train":2048,"discovery":len(xd),
                  "selector_diagnostics":len(xs),"heldout":heldout_n},
        "clean_closure":clean_closure,
        "score_nonzero_count":discovery["model_score_nonzero_count"],
        "passwise_source_candidates":{
            str(t):{str(rank):{k:v for k,v in pass_candidates[t][rank].items()
                    if k not in ("basis","projector")}
                    for rank in OPTIONS} for t in range(PASSES)
        },
        "fixed_budget12_allocation":optimum,
        "flat3_source_write_rms":flatdose,
        "dp12_source_write_rms":dpdose,
        "dose_matched_flat3_gain":alpha,
        "dose_matched_flat3_gain_eligible":dose_admissible,
        "arms":{key:{k:v for k,v in arm.items() if k!="projector"}
                for key,arm in arms.items()},
        "sham_admission":sham_admission,
        "gauge":gauges,
        "heldout":{},
    }
    # All fitted projectors, gains, model-only diagnostics, shams and
    # rotated physical-gauge admissions are locked before final labels.
    xe=sample_x(seed,3,heldout_n)
    ye=external_labels(xe)
    context=context_accuracies(model,xe,ye)
    with bpd.target_capture():
        clean,_=model(xe,capture=True)
        clean_writes=torch.stack(model.writes,1)
    sign=torch.where(ye,1.,-1.)
    record["clean_final_nll_nats"]=float(F.softplus(-2*sign*clean[:,-1]).mean())
    record["clean_final_accuracy"]=float(((clean[:,-1]>=0)==ye).float().mean())
    record["task_context_accuracy"]=context
    for key,item in arms.items():
        if item["eligible"]:
            record["heldout"][key]=evaluate_task(
                bpd,model,xe,ye,clean,clean_writes,item
            )
        else:
            record["heldout"][key]=None
    if record["heldout"]["dp12"] is not None and shams_selected:
        sham_scores=[
            evaluate_task(bpd,model,xe,ye,clean,clean_writes,
                          {"projector":z["projector"],"gain":1.})
            for z in shams_selected
        ]
        record["sham_task_nlls"]=[r["nll_nats"] for r in sham_scores]
        record["median_sham_minus_dp12_nll"]=float(
            np.median(record["sham_task_nlls"])-
            record["heldout"]["dp12"]["nll_nats"]
        )
    record["eligible"]=bool(
        min(context.values())>=.85 and
        training["mix_density"]>=.90 and clean_closure["passed"] and
        optimum is not None and dose_admissible and
        all(arms[k]["eligible"] for k in ("dp12","flat3","flat3_dose","flat4")) and
        sham_admission is not None and sham_admission["selected"]==shams and
        all(record["heldout"][k] is not None
            for k in ("dp12","flat3","flat3_dose","flat4")) and
        all(gauges[k]["physical_map_relative_error"]<=1e-4 and
            gauges[k]["max_complete_rollout_logit_error"]<=5e-5
            for k in ("dp12","flat3","flat3_dose","flat4"))
    )
    record["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=int,default=SMOKE)
    parser.add_argument("--steps",type=int,default=16)
    parser.add_argument("--discovery",type=int,default=128)
    parser.add_argument("--selector",type=int,default=128)
    parser.add_argument("--heldout",type=int,default=128)
    parser.add_argument("--sham-proposals",type=int,default=128)
    parser.add_argument("--shams",type=int,default=2)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    rec=evaluate_root(args.root,steps=args.steps,derivative_n=args.discovery,
                      selector_n=args.selector,heldout_n=args.heldout,
                      sham_proposals=args.sham_proposals,shams=args.shams)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "root":rec["root"],"eligible":rec["eligible"],
        "ranks":rec["fixed_budget12_allocation"]["ranks"]
          if rec["fixed_budget12_allocation"] else None,
        "qualified_shams":rec["sham_admission"]["qualified"]
          if rec["sham_admission"] else None,
    },sort_keys=True))


if __name__=="__main__":
    main()
