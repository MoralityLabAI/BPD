"""BPD-NC-11: Jacobian-response alignment across separately trained recurrences.

Engineering-only, first source-only protocol commit 60b65642c03cb3064912aa3929582479aa21dd20.
All causal alignment choices use unlabeled source/calibration data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import PASSES, WIDTH, sample_x, parity
from mechinterp.bpd.experiments.nc03.experiment import basis_transport
from mechinterp.bpd.experiments.nc04.experiment import factor_basis
from mechinterp.bpd.experiments.nc05.experiment import (
    projectors_at_ranks, clean_dose_rms,
)
from mechinterp.bpd.experiments.nc06.experiment import local_matched_shams
from mechinterp.bpd.experiments.nc10.experiment import (
    task_labels, train_task as train_previous_task, procrustes, transport,
    discover, prewrite_states, clean_reference, score_logits,
)

PROTOCOL_COMMIT = "60b65642c03cb3064912aa3929582479aa21dd20"
DEVELOPMENT = tuple(range(1100, 1108))
SMOKE = 1099
STEPS = 384
CALIBRATION_POOL = 512
TARGET_CALIBRATION = 128
DISCOVERY = 1024
DOSE_POOL = 1024
EVALUATION = 1024
RANK = 4
SHAMS = 8
PROPOSALS = 2048
GAIN_LOWER = 0.5
GAIN_UPPER = 1.5


def train_pair_task(root: int, task: str, *, steps: int=STEPS):
    """Same optimizer/data architecture as NC10, fresh root population."""
    if root not in DEVELOPMENT and root != SMOKE:
        raise ValueError("NC-11 permits only engineering 1099 and 1100..1107")
    # NC10's restricted-root trainer is reused with the same code contract
    # through a faithful local implementation, not through NC10 root IDs.
    torch.set_num_threads(1)
    from mechinterp.bpd.experiments.nc04.experiment import DenseLooper
    model_seed = root if task == "A" else root + 100_000
    if task not in ("A","B"):
        raise ValueError("task must be A or B")
    split = 1 if task == "A" else 11
    torch.manual_seed(model_seed)
    model = DenseLooper()
    x = sample_x(root,split,2048)
    yy = torch.where(task_labels(x,task),1.,-1.)
    rng = torch.Generator().manual_seed(model_seed+1_737_701)
    opt = torch.optim.Adam(model.parameters(),lr=.005)
    for _ in range(steps):
        ids = torch.randint(len(x),(128,),generator=rng)
        logits,_ = model(x[ids])
        loss = F.softplus(-2*yy[ids]*logits[:,-1]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        final,_ = model(x)
        density = float((model.mix.weight.detach().abs() >= 1e-4).float().mean())
        accuracy = float(((final[:,-1]>=0)==task_labels(x,task)).float().mean())
    return model,{
        "task":task,"root":root,"model_seed":model_seed,"train_split":split,
        "training_examples":2048,"training_steps":steps,
        "learning_rate":.005,"batch_size":128,
        "train_accuracy":accuracy,"mix_density":density,
        "last_minibatch_nll_nats":float(loss.detach()),
    }


def read_and_score_jacobian(bpd: BPDModel, model, x: Tensor,
                            batch_size: int=64) -> tuple[Tensor,Tensor]:
    """Physical a_t and unsiged d(final model logit)/da_t (no true labels)."""
    if x.ndim != 2 or x.shape[1] != 5 or len(x)==0:
        raise ValueError("requires nonempty [N,5] input")
    acts, grads = [],[]
    wrapper=bpd.wrappers["mix"]
    for start in range(0,len(x),batch_size):
        xx=x[start:start+batch_size].clone().requires_grad_(True)
        with bpd.target_capture():
            scores,_ = model(xx,capture=True)
            dw = torch.autograd.grad(scores[:,-1].sum(),model.writes)
        a = torch.stack(model.acts,dim=1).detach()
        gw = torch.stack(dw,dim=1).detach()
        z=a @ wrapper.V.detach()
        gz=gw @ wrapper.U.detach().T
        acts.append(z.double())
        grads.append(gz.double())
    return torch.cat(acts),torch.cat(grads)


def unit_jacobians(g: Tensor) -> tuple[Tensor,dict]:
    if g.ndim != 3 or g.shape[1:] != (PASSES,WIDTH):
        raise ValueError("requires per-example per-pass Jacobians")
    norms=g.double().norm(dim=-1,keepdim=True)
    return g.double()/norms.clamp_min(1e-12),{
        "min_row_norm":float(norms.min()),
        "zero_row_count":int((norms.squeeze(-1)<=1e-12).sum()),
        "mean_row_norm_by_pass":[float(q) for q in norms.mean(0).flatten()],
    }


def transfer_map(source_rank_basis: Tensor, alignment: Tensor):
    return transport(source_rank_basis,alignment)


@torch.no_grad()
def projected_with_gain(bpd:BPDModel,model,x:Tensor,p:Tensor,gain:float):
    if p.shape!=(PASSES,WIDTH,WIDTH):
        raise ValueError("invalid projector dimensions")
    if not (0.0<=gain<=2.0):
        raise ValueError("physical retained write gain outside global safety range")
    wrapper=bpd.wrappers["mix"]
    reads=(wrapper.V.detach().double() @ (gain*p.double())).float()
    with bpd.intervene("mix",tuple(range(WIDTH)),mode="override_read",
                       scale=0.0,read_override=reads):
        logits,_=model(x,capture=True)
        writes=torch.stack(model.writes,dim=1)
    return logits,writes


@torch.no_grad()
def score_projector(bpd,model,x,y,clean,clean_w,p,gain=1.0):
    logits,writes=projected_with_gain(bpd,model,x,p,gain)
    return score_logits(clean,logits,clean_w,writes,x,y)


def fixed_gain(reference_dose:float, acts:Tensor, u:Tensor, p:Tensor)->dict:
    observed=clean_dose_rms(acts,u,p)
    raw_gain=reference_dose/max(observed,1e-12)
    eligible=GAIN_LOWER<=raw_gain<=GAIN_UPPER and reference_dose>1e-12
    return {"raw_gain":float(raw_gain),"applied_gain":float(raw_gain) if eligible else 1.0,
            "eligible":bool(eligible),"source_dose_rms":float(observed),
            "matched_dose_rms":float(raw_gain*observed) if eligible else float(observed),
            "reference_dose_rms":float(reference_dose)}


def _projection_diagnostics(p:Tensor)->dict:
    return {"symmetry_max_error":float((p-p.transpose(-1,-2)).abs().max()),
            "idempotence_max_error":float((p@p-p).abs().max()),
            "rank_trace_by_pass":[float(t) for t in torch.diagonal(p,dim1=-2,dim2=-1).sum(-1)]}


def _context_accuracies(clean:dict):
    return [clean["by_context"][kind]["accuracy"] for kind in ("shared","specific")]


def run(root:int,*,steps:int=STEPS,discovery_size:int=DISCOVERY,
        calibration_pool_size:int=CALIBRATION_POOL,
        dose_size:int=DOSE_POOL,evaluation_size:int=EVALUATION,
        proposals:int=PROPOSALS) -> dict:
    if root!=SMOKE and root not in DEVELOPMENT:
        raise ValueError("NC-11 can run only excluded smoke 1099 and new roots 1100..1107")
    torch.set_num_threads(1)
    model_a,trained_a=train_pair_task(root,"A",steps=steps)
    model_b,trained_b=train_pair_task(root,"B",steps=steps)
    bpd_a=BPDModel(model_a,{"mix":WIDTH},max_passes=PASSES,
                    ablation=AblationConfig(mode="zero",sphere_rescale=False))
    bpd_b=BPDModel(model_b,{"mix":WIDTH},max_passes=PASSES,
                    ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bpd_a,"native",root)
    factor_basis(bpd_b,"native",root+100_000)
    x_source=sample_x(root,2,discovery_size)
    x_cal=sample_x(root,4,calibration_pool_size)
    positive=x_cal[x_cal[:,4]>0]
    accepted=positive[:TARGET_CALIBRATION]
    x_dose=sample_x(root,6,dose_size)
    x_eval=sample_x(root,3,evaluation_size)
    close_a=parity(bpd_a,model_a,x_dose[:min(64,len(x_dose))])
    close_b=parity(bpd_b,model_b,x_dose[:min(64,len(x_dose))])
    rec={
        "study":"BPD-NC-11","phase":"ENGINEERING","root":root,
        "protocol_commit":PROTOCOL_COMMIT,
        "source_hash_sha256":None,"protocol_sha256":None,
        "budgets":{"steps":steps,"train_A":2048,"train_B":2048,
                   "discovery":discovery_size,"calibration_pool":calibration_pool_size,
                   "target_gradient_calibration":TARGET_CALIBRATION,
                   "dose":dose_size,"evaluation":evaluation_size,
                   "sham_proposals":proposals},
        "training":{"A":trained_a,"B":trained_b},
        "closure":{"A":close_a,"B":close_b},
        "source":{},"calibration":{},"shams":{},"gauge":{},
        "negative_control":{},"evaluation":{},"comparisons":{},
        "excluded_reasons":[],
    }
    if not close_a["passed"] or not close_b["passed"]:
        rec["excluded_reasons"].append("full_mask_parity")
    if len(accepted)<TARGET_CALIBRATION:
        rec["excluded_reasons"].append("insufficient_positive_calibration_examples")
    # All grad directions are discovered on disjoint unlabeled sources.
    pa,ea,da,_=discover(bpd_a,model_a,x_source,root,"A")
    rec["source"]["A_rank4"]=da["rank4"]
    rec["source"]["A_selfscore_observational_graph_communities"]=da["graph_n_communities"]
    if pa is None:
        rec["excluded_reasons"].append("source_rank4_unidentifiable")
    if len(accepted)==TARGET_CALIBRATION:
        ac,ga=read_and_score_jacobian(bpd_a,model_a,accepted)
        bc,gb=read_and_score_jacobian(bpd_b,model_b,accepted)
        ga_unit,ga_info=unit_jacobians(ga)
        gb_unit,gb_info=unit_jacobians(gb)
        rec["calibration"]["jacobian_norm_A"]=ga_info
        rec["calibration"]["jacobian_norm_B"]=gb_info
        bcal,bcaldiag,bcalbase=projectors_at_ranks(gb_unit)
        pb_cal=bcal[RANK]
        rec["calibration"]["target_calibration_only_rank4"]=bcaldiag[RANK]
        if pb_cal is None:
            rec["excluded_reasons"].append("target_128_gradient_projector_unidentifiable")
        if ga_info["zero_row_count"] or gb_info["zero_row_count"]:
            rec["excluded_reasons"].append("zero_jacobian_calibration_row")
        rj, dj=procrustes(ga_unit,gb_unit)
        ract,dact=procrustes(ac,bc)
        rng=np.random.default_rng(root+11_110_011)
        permutation=rng.permutation(TARGET_CALIBRATION)
        rperm,dperm=procrustes(ga_unit,gb_unit[
            torch.as_tensor(permutation,dtype=torch.long)])
        rec["calibration"].update({
            "positive_context_examples_in_pool":len(positive),
            "calibration_used":TARGET_CALIBRATION,
            "jacobian_procrustes":dj,"activation_procrustes":dact,
            "permuted_jacobian_procrustes":dperm,
            "permutation_sha256":hashlib.sha256(permutation.tobytes()).hexdigest(),
        })
        if pa is not None and pb_cal is not None:
            pj,ej=transfer_map(ea,rj)
            pact,_=transfer_map(ea,ract)
            pperm,_=transfer_map(ea,rperm)
            with torch.no_grad():
                dose_acts=prewrite_states(bpd_b,model_b,x_dose)
            u=bpd_b.wrappers["mix"].U.detach()
            reference=clean_dose_rms(dose_acts,u,pj)
            projectors={"jacobian":pj,"activation":pact,"permuted_jacobian":pperm,
                        "target_calibration":pb_cal,"identity":pa}
            gains={name:fixed_gain(reference,dose_acts,u,p)
                   for name,p in projectors.items()}
            gains["jacobian"]["applied_gain"]=1.0
            gains["jacobian"]["eligible"]=reference>1e-12
            gains["jacobian"]["raw_gain"]=1.0
            rec["calibration"].update({
                "reference_physical_write_dose":reference,
                "rank_four_projector_diagnostics":{name:_projection_diagnostics(p)
                   for name,p in projectors.items()},
                "relative_projector_overlap_vs_target128":{
                    name:[float(torch.trace(p[t]@pb_cal[t])/RANK)
                          for t in range(PASSES)]
                    for name,p in projectors.items()},
                "dose_matched_gains":gains,
            })
            for name in ("activation","permuted_jacobian","target_calibration"):
                if not gains[name]["eligible"]:
                    rec["excluded_reasons"].append("gain_out_of_bounds_"+name)
            shams,sham_stats=local_matched_shams(
                dose_acts,u,ej,seed=root,arm_index=11,proposals=proposals,want=SHAMS)
            rec["shams"]=sham_stats
            if not sham_stats.get("admitted",False):
                rec["excluded_reasons"].append("insufficient_eight_dose_matched_shams")
            # The B-full projector requires 1024 B gradients, and may be scored
            # only as an explicit higher-B-query descriptive upper reference.
            _,g_b_full=read_and_score_jacobian(bpd_b,model_b,x_source)
            pfull,dfull,_=projectors_at_ranks(g_b_full)
            rec["source"]["B_full_rank4"]=dfull[RANK]
            if pfull[RANK] is not None:
                projectors["target_full_gradient_reference"]=pfull[RANK]
                gains["target_full_gradient_reference"]=fixed_gain(
                    reference,dose_acts,u,pfull[RANK])
                rec["calibration"]["dose_matched_gains"]=gains
            # Within-B gauge is the exact numerical calibration, not A/B
            # model equivalence.
            check_inputs=x_dose[:min(64,len(x_dose))]
            native_out,_=projected_with_gain(
                bpd_b,model_b,check_inputs,pj,1.0)
            native_map=pj@bpd_b.wrappers["mix"].U.detach().double()
            factor_basis(bpd_b,"rotated",root+100_000)
            wrapped=bpd_b.wrappers["mix"]
            rotated_p=basis_transport(pj,wrapped.V.detach())
            rotated_map=wrapped.V.detach().double()@rotated_p@wrapped.U.detach().double()
            rotated_out,_=projected_with_gain(
                bpd_b,model_b,check_inputs,rotated_p,1.0)
            gauge_parity=parity(bpd_b,model_b,check_inputs)
            rec["gauge"]={
                "full_weight_parity":gauge_parity,
                "physical_operator_error_relative":float(
                    (native_map-rotated_map).norm(dim=(1,2)).max()
                    / wrapped.W_target.double().norm().clamp_min(1e-12)),
                "heldout_free_source_kept_max_logit_error":float(
                    (native_out-rotated_out).abs().max()),
            }
            factor_basis(bpd_b,"native",root+100_000)
            if (not gauge_parity["passed"]
                or rec["gauge"]["physical_operator_error_relative"]>1e-4
                or rec["gauge"]["heldout_free_source_kept_max_logit_error"]>5e-5):
                rec["excluded_reasons"].append("within_target_factor_gauge")
            # Negative head on a distinct source-only input subset.
            with torch.no_grad():
                hw=model_b.head.weight.detach().clone()
                hb=model_b.head.bias.detach().clone()
                model_b.head.weight.zero_(); model_b.head.bias.zero_()
            try:
                _,g0=read_and_score_jacobian(bpd_b,model_b,x_dose[:min(32,len(x_dose))])
                rec["negative_control"]={
                    "zero_head_jacobian_max_abs":float(g0.abs().max()),
                    "zero_head_subspace_rejected":float(g0.abs().max())==0.,
                }
            finally:
                with torch.no_grad():
                    model_b.head.weight.copy_(hw)
                    model_b.head.bias.copy_(hb)
            if not rec["negative_control"]["zero_head_subspace_rejected"]:
                rec["excluded_reasons"].append("zero_head_negative_control")
            # Final labels and causal outcomes are accessed after every fixed
            # projector, mapping, sham and gain is selected and recorded.
            ya=task_labels(x_eval,"A")
            yb=task_labels(x_eval,"B")
            ca,wa,clean_a=clean_reference(bpd_a,model_a,x_eval,ya)
            cb,wb,clean_b=clean_reference(bpd_b,model_b,x_eval,yb)
            scores={"A_clean":clean_a,"B_clean":clean_b}
            scores["A_owned"]=score_projector(
                bpd_a,model_a,x_eval,ya,ca,wa,pa,1.)
            for name,p in projectors.items():
                scores[name]=score_projector(
                    bpd_b,model_b,x_eval,yb,cb,wb,p,gains[name]["applied_gain"])
            scores["shams"]=[score_projector(
                bpd_b,model_b,x_eval,yb,cb,wb,p,1.) for p in shams]
            rec["evaluation"]=scores
            competence=[scores["A_clean"]["by_context"][c]["accuracy"]
                        for c in ("shared","specific")]
            competence.extend(scores["B_clean"]["by_context"][c]["accuracy"]
                              for c in ("shared","specific"))
            rec["clean_context_accuracies"]=competence
            if min(competence)<.85 or min(trained_a["mix_density"],
                 trained_b["mix_density"])<.9:
                rec["excluded_reasons"].append("model_task_or_weight_density")
            def nll(arm:str,ctx:str):
                return scores[arm]["by_context"][ctx]["final_nll_nats"]
            contrasts={
                "jacobian_vs_activation_shared":nll("activation","shared")-nll("jacobian","shared"),
                "jacobian_vs_permuted_shared":nll("permuted_jacobian","shared")-nll("jacobian","shared"),
                "jacobian_minus_target_cal_shared":nll("jacobian","shared")-nll("target_calibration","shared"),
                "specific_minus_shared_transfer_deficit":(
                    nll("jacobian","specific")-nll("target_calibration","specific")
                    -nll("jacobian","shared")+nll("target_calibration","shared")),
                "identity_minus_jacobian_shared":nll("identity","shared")-nll("jacobian","shared"),
            }
            if len(scores["shams"])==SHAMS:
                contrasts["matched_sham_advantage_shared"]=float(
                    np.median([s["by_context"]["shared"]["final_nll_nats"]
                               for s in scores["shams"]])
                    -nll("jacobian","shared"))
            rec["comparisons"]=contrasts
    if not rec["negative_control"]:
        rec["excluded_reasons"].append("no_zero_head_control")
    rec["eligible"]=len(rec["excluded_reasons"])==0
    rec["source_hash_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rec["protocol_sha256"]=hashlib.sha256(
        Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest()
    return rec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=int,default=SMOKE)
    p.add_argument("--steps",type=int,default=STEPS)
    p.add_argument("--discovery-size",type=int,default=DISCOVERY)
    p.add_argument("--calibration-pool-size",type=int,default=CALIBRATION_POOL)
    p.add_argument("--dose-size",type=int,default=DOSE_POOL)
    p.add_argument("--evaluation-size",type=int,default=EVALUATION)
    p.add_argument("--proposals",type=int,default=PROPOSALS)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=run(args.root,steps=args.steps,discovery_size=args.discovery_size,
               calibration_pool_size=args.calibration_pool_size,
               dose_size=args.dose_size,evaluation_size=args.evaluation_size,
               proposals=args.proposals)
    payload=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if args.out is None:
        print(payload)
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as f:f.write(payload)

if __name__=="__main__":
    main()
