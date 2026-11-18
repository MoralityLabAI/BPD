"""NC-06: norm-conditioned passwise rank-3 physical projectors in trained recurrences.

All source and scientific development decisions are frozen in PROTOCOL.md.
The only scored roots are new development seeds 600..607; 599 is smoke-only.
The estimator/shams receive no external task label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import (
    PASSES, WIDTH, sample_x, external_labels, parity
)
from mechinterp.bpd.experiments.nc04.experiment import (
    DenseLooper, factor_basis, self_score_discovery, context_accuracies
)
from mechinterp.bpd.experiments.nc05.experiment import (
    score_keep, projected_rollout, clean_dose_rms
)
from mechinterp.bpd.experiments.nc03.experiment import basis_transport

ROOT=Path(__file__).resolve().parent
PROTOCOL_HASH=hashlib.sha256((ROOT/"PROTOCOL.md").read_bytes()).hexdigest()
DEV_ROOTS=tuple(range(600,608))
SMOKE=599
RANKS={"norm_r3":3,"uniform_r3":3,"shuffle_r3":3,"uniform_r4_reference":4}
SHAMS=8
PROPOSALS=1024
ANGLES=(.45,.65,.85,1.05)
DOSE_TOL=.15
GAP_RATIO=1.001


def train_model(seed:int,*,steps:int=384):
    if seed not in DEV_ROOTS and seed!=SMOKE:
        raise ValueError("NC06 only permits 599 smoke and 600..607 development")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model=DenseLooper()
    x=sample_x(seed,1,2048)
    sign=torch.where(external_labels(x),1.,-1.)
    rng=torch.Generator().manual_seed(seed+1_737_701)
    opt=torch.optim.Adam(model.parameters(),lr=.005)
    for _ in range(steps):
        idx=torch.randint(len(x),(128,),generator=rng)
        logit=model(x[idx])[0][:,-1]
        loss=F.softplus(-2*sign[idx]*logit).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        density=float((model.mix.weight.detach().abs()>=1e-4).float().mean())
        preds=model(x)[0][:,-1]
        acc=float(((preds>=0)==external_labels(x)).float().mean())
    return model,{"seed":seed,"steps":steps,"train_size":len(x),
                  "train_accuracy":acc,"mix_density":density}


def norm_weights(activations:Tensor):
    """Unlabelled pass-dependent energy; normalized by source-only E||a||²."""
    if activations.ndim!=3 or activations.shape[1:]!=(PASSES,WIDTH):
        raise ValueError("Expected clean [N,4,20] prephysical activations")
    e=activations.double().square().sum(-1)
    mu=e.mean(0).clamp_min(1e-12)
    return torch.exp(-e/mu[None,:]),{"source_pass_mean_energy":mu.tolist()}


def shuffled_weights(weights:Tensor,seed:int):
    if weights.ndim!=2 or weights.shape[1]!=PASSES:
        raise ValueError("Expect per-pass weight matrix")
    out=torch.empty_like(weights)
    hashes=[]
    for t in range(PASSES):
        gen=torch.Generator().manual_seed(9_160_000+100*seed+t)
        perm=torch.randperm(len(weights),generator=gen)
        out[:,t]=weights[perm,t]
        hashes.append(hashlib.sha256(perm.numpy().tobytes()).hexdigest())
        assert torch.equal(torch.sort(out[:,t]).values,torch.sort(weights[:,t]).values)
    return out,hashes


def eigen_projectors(grads:Tensor,weights:Tensor,rank:int):
    if grads.ndim!=3 or grads.shape[1:]!=(PASSES,WIDTH):
        raise ValueError("Expected [N,4,20] score gradients")
    if weights.shape!=grads.shape[:2] or rank not in (3,4):
        raise ValueError("Rank or weights mismatch")
    g=grads.double()
    w=weights.double()
    cov=torch.einsum("nt,nti,ntj->tij",w,g,g)/w.sum(0)[:,None,None]
    eig,vec=torch.linalg.eigh(cov)
    out=[];spectra=[]
    for t in range(PASSES):
        high=float(eig[t,-rank])
        low=float(eig[t,-rank-1])
        ratio=high/max(low,1e-12)
        basis=vec[t,:,-rank:]
        proj=basis@basis.T
        spectra.append({"pass":t,"retained_lambda_min":high,
                        "next_lambda":low,"eigengap":ratio,
                        "retained_fraction":float(eig[t,-rank:].sum()/
                             eig[t].clamp_min(0).sum().clamp_min(1e-30))})
        out.append(basis)
    eligible=all(row["retained_lambda_min"]>1e-12 and
                 row["eigengap"]>=GAP_RATIO for row in spectra)
    basis=torch.stack(out)
    projector=basis@basis.transpose(-1,-2)
    return (projector if eligible else None),basis,{
        "eigengap_eligible":eligible,"spectrum":spectra,
        "projector_symmetry_max":float((projector-projector.transpose(-1,-2)).abs().max()),
        "projector_idempotence_max":float((projector@projector-projector).abs().max()),
    }


def local_matched_shams(activations:Tensor,u:Tensor,eigbasis:Tensor,
                        seed:int,arm_index:int,*,proposals:int=PROPOSALS,
                        want:int=SHAMS):
    """NC05 local-angle/physical write-dose rule, generalized to rank 3."""
    rank=eigbasis.shape[-1]
    if rank not in (3,4) or eigbasis.shape[:2]!=(PASSES,WIDTH):
        raise ValueError("Expected rank 3/4 source-only basis")
    e=eigbasis.detach().double().cpu().numpy()
    acts=activations.detach().double().cpu().numpy()
    U=u.detach().double().cpu().numpy()
    moment=np.einsum("nti,ntj->tij",acts,acts)/len(acts)
    gram=U@U.T
    def dose(b):
        q=0.
        for t in range(PASSES):
            v=b[t]
            q+=float(np.trace((v.T@moment[t]@v)@(v.T@gram@v)))
        return float(np.sqrt(max(q,0.)/(PASSES*WIDTH)))
    target=dose(e)
    if target<=1e-12:
        return [],{"admitted":False,"reason":"zero discovery dose","qualified":0}
    rng=np.random.default_rng(seed+5_051_701+139*rank+10007*arm_index)
    chosen=[];qual=0
    for ix in range(proposals):
        theta=ANGLES[int(rng.integers(len(ANGLES)))]
        perturb=np.empty_like(e)
        for t in range(PASSES):
            z=rng.standard_normal((WIDTH,rank))
            z=z-e[t]@(e[t].T@z)
            complement,_=np.linalg.qr(z,mode="reduced")
            perturb[t]=np.cos(theta)*e[t]+np.sin(theta)*complement
        d=dose(perturb)
        relative=abs(d-target)/target
        if relative<=DOSE_TOL:
            qual+=1
            if len(chosen)<want:
                chosen.append(torch.from_numpy(
                    perturb@perturb.transpose(0,2,1)
                ).double())
    return chosen,{"admitted":len(chosen)==want,"proposals":proposals,
                    "qualified":qual,"selected":len(chosen),
                    "reference_write_rms":target}


def source_projectors(activations:Tensor,gradients:Tensor,seed:int):
    w,energy=norm_weights(activations)
    shuffled,hashes=shuffled_weights(w,seed)
    ones=torch.ones_like(w)
    weights={"norm_r3":w,"shuffle_r3":shuffled,
             "uniform_r3":ones,"uniform_r4_reference":ones}
    result={}
    for name,rank in RANKS.items():
        p,b,s=eigen_projectors(gradients,weights[name],rank)
        wmode=weights[name]
        ess=(wmode.sum(0).square()/wmode.square().sum(0)).tolist()
        if name=="shuffle_r3":
            s["permutation_shas"]=hashes
        s["weight_effective_sample_fraction_by_pass"]=[
            float(v/len(wmode)) for v in ess]
        s["source_pass_mean_energy"]=energy["source_pass_mean_energy"]
        result[name]={"p":p,"basis":b,"diagnostic":s}
    return result


@torch.no_grad()
def test_score(model:DenseLooper,bpd:BPDModel,x:Tensor,y:Tensor,
               p:Tensor,clean:Tensor,clean_writes:Tensor):
    return score_keep(bpd,model,x,y,clean,clean_writes,p,1.)


def evaluate_root(seed:int,*,steps:int=384,discovery_size:int=1024,
                  eval_size:int=1024,proposals:int=PROPOSALS,shams:int=SHAMS):
    torch.set_num_threads(1)
    model,training=train_model(seed,steps=steps)
    xd=sample_x(seed,2,discovery_size)
    xe=sample_x(seed,3,eval_size)
    ye=external_labels(xe)
    task_context=context_accuracies(model,xe,ye)
    bpd=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                 ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bpd,"native",seed)
    layer=bpd.wrappers["mix"]
    clean_parity=parity(bpd,model,xe[:min(64,len(xe))])
    acts,grads,hits,self_score=self_score_discovery(bpd,model,xd)
    candidates=source_projectors(acts,grads,seed)
    with bpd.target_capture():
        clean,_=model(xe,capture=True)
        clean_writes=torch.stack(model.writes,dim=1)
    sign=torch.where(ye,1.,-1.)
    clean_nll=float(F.softplus(-2*sign*clean[:,-1]).mean())
    report={"study":"BPD-NC-06","phase":"ENGINEERING","root":seed,
            "training":training,"task_context":task_context,
            "clean_parity":clean_parity,"clean_nll":clean_nll,
            "discovery_examples":len(xd),"evaluation_examples":len(xe),
            "score_nonzero_count":self_score["model_score_nonzero_count"],
            "methods":{}}
    arms={}
    for ix,name in enumerate(RANKS):
        entry=candidates[name];p=entry["p"]
        info={"rank":RANKS[name],"diagnostic":entry["diagnostic"],
              "status":"PENDING"}
        report["methods"][name]=info
        if p is None:
            info["status"]="UNIDENTIFIABLE"
            continue
        info["scored"]=test_score(model,bpd,xe,ye,p,clean,clean_writes)
        if name.endswith("_r3"):
            sham_set,sham_match=local_matched_shams(
                acts,layer.U.detach(),entry["basis"],seed,ix,
                proposals=proposals,want=shams
            )
            info["sham_admission"]=sham_match
            if len(sham_set)!=shams:
                info["status"]="BLOCKED_SHAMS"
                continue
            sham_scores=[
                test_score(model,bpd,xe,ye,s,clean,clean_writes)
                for s in sham_set
            ]
            info["sham_nlls"]=[z["nll_nats"] for z in sham_scores]
            info["median_sham_minus_kept_nll"]=float(
                np.median(info["sham_nlls"])-info["scored"]["nll_nats"]
            )
        info["status"]="ELIGIBLE"
        arms[name]=p
    # Physical orthogonal gauge test: transport the same source-only
    # native projectors, test actual closed-loop logit equivalence.
    if arms and seed!=SMOKE:
        factor_basis(bpd,"rotated",seed)
        layer=bpd.wrappers["mix"]
        for name,p in arms.items():
            prot=basis_transport(p,layer.V.detach())
            physical=(layer.V.detach().double()@prot@layer.U.detach().double())
            # Native basis V=I: U is original W^T
            factor_basis(bpd,"native",seed)
            layer_native=bpd.wrappers["mix"]
            native_map=(layer_native.V.detach().double()@p@layer_native.U.detach().double())
            native_logits,_=projected_rollout(bpd,model,xe[:min(len(xe),64)],p,1.)
            factor_basis(bpd,"rotated",seed)
            rotated_logits,_=projected_rollout(bpd,model,xe[:min(len(xe),64)],prot,1.)
            report["methods"][name]["gauge"]={
                "relative_physical_operator_error":float(
                    (physical-native_map).norm()/layer.W_target.detach().double().norm().clamp_min(1e-12)
                ),
                "max_logit_error":float((native_logits-rotated_logits).abs().max())
            }
        factor_basis(bpd,"native",seed)
    report["eligible"]=bool(
        min(task_context.values())>=.85 and training["mix_density"]>=.90
        and clean_parity["passed"]
        and all(report["methods"][a]["status"]=="ELIGIBLE"
                for a in ("norm_r3","uniform_r3","shuffle_r3"))
        and all(
            report["methods"][a].get("gauge",{}).get("max_logit_error",0.)<=5e-5
            for a in arms
        )
    )
    report["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["protocol_sha256"]=PROTOCOL_HASH
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=SMOKE)
    ap.add_argument("--steps",type=int,default=64)
    ap.add_argument("--discovery",type=int,default=256)
    ap.add_argument("--eval",type=int,default=256)
    ap.add_argument("--proposals",type=int,default=256)
    ap.add_argument("--shams",type=int,default=8)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    rec=evaluate_root(args.seed,steps=args.steps,discovery_size=args.discovery,
                      eval_size=args.eval,proposals=args.proposals,shams=args.shams)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"root":rec["root"],"eligible":rec["eligible"],
                      "methods":{k:v["status"] for k,v in rec["methods"].items()}},indent=2))


if __name__=="__main__":
    main()
