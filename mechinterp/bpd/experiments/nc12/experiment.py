"""NC-12: nested low-target-gradient cross-model causal transfer frontier.

Prospective registration: PROTOCOL.md first commit 6a4b5ebc16fae730f17b0e5dac5564eaf419c071.
Only fresh development roots 1200..1207 and excluded source smoke 1199.
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
from mechinterp.bpd.experiments.nc04.experiment import DenseLooper, factor_basis
from mechinterp.bpd.experiments.nc05.experiment import projectors_at_ranks, clean_dose_rms
from mechinterp.bpd.experiments.nc06.experiment import local_matched_shams
from mechinterp.bpd.experiments.nc10.experiment import task_labels, procrustes, prewrite_states, transport, clean_reference
from mechinterp.bpd.experiments.nc11.experiment import (
    read_and_score_jacobian, unit_jacobians, fixed_gain, score_projector, projected_with_gain,
)

PROTOCOL_COMMIT = "6a4b5ebc16fae730f17b0e5dac5564eaf419c071"
ROOTS = tuple(range(1200, 1208))
SMOKE = 1199
BUDGETS = (4, 8, 16, 32)
TRAIN_STEPS = 384
DISCOVERY = 1024
CAL_POOL = 512
ACTIVATION_CAL = 128
MAX_B_GRAD = 32
DOSE_SIZE = 1024
EVAL_SIZE = 1024
RANK = 4
SHAMS = 8
PROPOSALS = 2048
RIDGE = 0.05


def train_task(root: int, task: str, steps: int = TRAIN_STEPS):
    """Unconstrained 20D trained recurrence, disjoint A/B initializations."""
    if root not in ROOTS and root != SMOKE:
        raise ValueError("NC12 permits source smoke 1199 and roots 1200..1207 only")
    if task not in ("A", "B"):
        raise ValueError("task must be A or B")
    torch.set_num_threads(1)
    model_seed = root if task == "A" else root + 100_000
    split = 1 if task == "A" else 11
    torch.manual_seed(model_seed)
    model = DenseLooper()
    training = sample_x(root, split, 2048)
    sign = torch.where(task_labels(training, task), 1., -1.)
    rng = torch.Generator().manual_seed(model_seed + 1_737_701)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(steps):
        ids = torch.randint(len(training), (128,), generator=rng)
        logits, _ = model(training[ids])
        loss = F.softplus(-2 * sign[ids] * logits[:, -1]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        scores, _ = model(training)
        density = float((model.mix.weight.detach().abs() >= 1e-4).float().mean())
        accuracy = float(((scores[:, -1] >= 0) == task_labels(training, task)).float().mean())
    return model, {
        "root": root, "task": task, "model_seed": model_seed,
        "training_split": split, "training_size": 2048,
        "steps": steps, "batch_size": 128, "learning_rate": .005,
        "train_accuracy": accuracy, "mix_density": density,
        "last_minibatch_nll_nats": float(loss.detach()),
    }


def regularized_jacobian_procrustes(
    a_unit: Tensor, b_unit: Tensor, activation_prior: Tensor,
) -> tuple[Tensor, dict]:
    """Prior-anchored orthogonal polar alignment for q < hidden dimension.

    The added 0.05 R_act resolves unobserved Jacobian directions rather than
    relying on arbitrary SVD choices in the nullspace of q paired rows.
    """
    if a_unit.shape != b_unit.shape or a_unit.ndim != 3 or len(a_unit) < 2:
        raise ValueError("paired gradient matrices require matching [q,T,W]")
    if a_unit.shape[1:] != (PASSES, WIDTH) or activation_prior.shape != (PASSES,WIDTH,WIDTH):
        raise ValueError("wrong gradient or prior shapes")
    q = len(a_unit)
    cross = torch.einsum("nti,ntj->tij",a_unit.double(),b_unit.double()) / q
    regularized = cross + RIDGE * activation_prior.double()
    u, singular, vh = torch.linalg.svd(regularized, full_matrices=False)
    aligned = u @ vh
    predicted = torch.einsum("nti,tij->ntj", a_unit.double(), aligned)
    residual = ((predicted-b_unit.double()).norm(dim=(0,2)) /
                b_unit.double().norm(dim=(0,2)).clamp_min(1e-12))
    orth = aligned.transpose(-1,-2)@aligned
    return aligned,{
        "q":q,"fixed_ridge":RIDGE,
        "minimum_regularized_singular_by_pass":[float(v) for v in singular.min(-1).values],
        "max_orthogonality_abs":float((orth-torch.eye(WIDTH,dtype=torch.float64)).abs().max()),
        "normalized_gradient_fit_residual_by_pass":[float(v) for v in residual],
        "eligible":bool(float(singular.min())>1e-12),
    }


def split_calibration(x: Tensor) -> tuple[Tensor, Tensor, dict]:
    """Take deterministic positive-context 128, nested first 32 gradient rows."""
    if x.shape != (CAL_POOL,5):
        raise ValueError("expected frozen 512-case calibration pool")
    positive = x[x[:,4]>0]
    first128 = positive[:ACTIVATION_CAL]
    first32 = first128[:MAX_B_GRAD]
    return first128, first32,{
        "positive_context_count":len(positive),
        "activation_examples":len(first128),
        "nested_gradient_union":len(first32),
        "nested_budgets":list(BUDGETS),
        "prefixes_are_nested":True,
    }


def _projector_diagnostics(p:Tensor) -> dict:
    return {"symmetry_max_abs":float((p-p.transpose(-1,-2)).abs().max()),
            "idempotence_max_abs":float((p@p-p).abs().max()),
            "rank_trace_by_pass":[float(v) for v in
                torch.diagonal(p,dim1=-2,dim2=-1).sum(-1)]}


def _context_nll(scores:dict,context:str="shared") -> float:
    return float(scores["by_context"][context]["final_nll_nats"])


def run(
    root: int, *, steps: int = TRAIN_STEPS,
    discovery_size: int = DISCOVERY, dose_size: int = DOSE_SIZE,
    evaluation_size: int = EVAL_SIZE, proposals: int = PROPOSALS,
) -> dict:
    if root not in ROOTS and root != SMOKE:
        raise ValueError("NC12 permits only source smoke 1199 or fresh roots 1200..1207")
    torch.set_num_threads(1)
    a, train_a = train_task(root,"A",steps)
    b, train_b = train_task(root,"B",steps)
    ba=BPDModel(a,{"mix":WIDTH},max_passes=PASSES,
                ablation=AblationConfig(mode="zero",sphere_rescale=False))
    bb=BPDModel(b,{"mix":WIDTH},max_passes=PASSES,
                ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(ba,"native",root)
    factor_basis(bb,"native",root+100_000)
    x_source=sample_x(root,2,discovery_size)
    x_cal=sample_x(root,4,CAL_POOL)
    x_dose=sample_x(root,6,dose_size)
    x_eval=sample_x(root,3,evaluation_size)
    full128, first32, split_diag=split_calibration(x_cal)
    close_a=parity(ba,a,x_dose[:min(64,len(x_dose))])
    close_b=parity(bb,b,x_dose[:min(64,len(x_dose))])
    rec={
        "study":"BPD-NC-12","phase":"ENGINEERING","root":root,
        "protocol_commit":PROTOCOL_COMMIT,
        "budgets":{"steps":steps,"train_A":2048,"train_B":2048,
                   "discovery":discovery_size,"calibration_pool":CAL_POOL,
                   "activation_calibration":ACTIVATION_CAL,
                   "target_gradient_union":MAX_B_GRAD,
                   "dose":dose_size,"evaluation":evaluation_size,
                   "sham_proposals":proposals},
        "training":{"A":train_a,"B":train_b},
        "closure":{"A":close_a,"B":close_b},
        "calibration":split_diag,
        "source":{},"q_arms":{},"negative_control":{},"gauge":{},
        "heldout":{},"excluded_global":[],
    }
    if len(full128)!=ACTIVATION_CAL:
        rec["excluded_global"].append("insufficient_positive_context_activation_pool")
    if len(first32)!=MAX_B_GRAD:
        rec["excluded_global"].append("insufficient_nested_target_gradient_cases")
    if not close_a["passed"] or not close_b["passed"]:
        rec["excluded_global"].append("physical_BPD_fullmask_parity")
    # Source rank-four projector: A only and model-only final-logit derivatives.
    _,grad_a=read_and_score_jacobian(ba,a,x_source)
    proj_a, diag_a, basis_a=projectors_at_ranks(grad_a)
    pa=proj_a[RANK]
    rec["source"]["source_A_rank4"]=diag_a[RANK]
    if pa is None:
        rec["excluded_global"].append("source_rank_four_eigenspace")
    if len(first32)==MAX_B_GRAD:
        act_a=prewrite_states(ba,a,full128)
        act_b=prewrite_states(bb,b,full128)
        prior,diag_prior=procrustes(act_a,act_b)
        rec["calibration"]["activation_prior"]=diag_prior
        ga_raw=read_and_score_jacobian(ba,a,first32)[1]
        gb_raw=read_and_score_jacobian(bb,b,first32)[1]
        ga_unit,ga_norm=unit_jacobians(ga_raw)
        gb_unit,gb_norm=unit_jacobians(gb_raw)
        rec["calibration"]["A_gradient_norms"]=ga_norm
        rec["calibration"]["B_gradient_norms"]=gb_norm
        if ga_norm["zero_row_count"] or gb_norm["zero_row_count"]:
            rec["excluded_global"].append("zero_calibration_jacobian_row")
        # No B gradient more than these 32 is consulted by primary q choices.
        with torch.no_grad():
            dose_acts=prewrite_states(bb,b,x_dose)
        u_b=bb.wrappers["mix"].U.detach()
        sources: dict[int,dict]={}
        for q in BUDGETS:
            a_q,b_q=ga_unit[:q],gb_unit[:q]
            r,dreg=regularized_jacobian_procrustes(a_q,b_q,prior)
            rperm,dperm=regularized_jacobian_procrustes(a_q,b_q.roll(1,0),prior)
            target_projectors,target_diag,_=projectors_at_ranks(b_q)
            pb=target_projectors[RANK]
            record={
                "budget":q,"target_gradient_queries":q,"activation_rows_used":ACTIVATION_CAL,
                "ridge_alignment":dreg,"circular_permutation_alignment":dperm,
                "target_only_rank4":target_diag[RANK],"sham_matching":{},
                "dose_gains":{},"projectors":{},"observed":{},"excluded_reasons":[],
            }
            rec["q_arms"][str(q)]=record
            if pa is None or pb is None:
                record["excluded_reasons"].append("source_or_target_rank4_unidentifiable")
                continue
            if not dreg["eligible"] or not dperm["eligible"]:
                record["excluded_reasons"].append("regularized_polar_numerical_failure")
            pj,ej=transport(basis_a[RANK],r)
            pact,_=transport(basis_a[RANK],prior)
            pperm,_=transport(basis_a[RANK],rperm)
            projects={
                "transfer":pj, "target_only":pb,
                "activation":pact, "permuted":pperm,
            }
            record["projectors"]={name:_projector_diagnostics(p)
                                  for name,p in projects.items()}
            reference=clean_dose_rms(dose_acts,u_b,pj)
            gains={name:fixed_gain(reference,dose_acts,u_b,p)
                   for name,p in projects.items()}
            gains["transfer"]["raw_gain"]=1.0
            gains["transfer"]["applied_gain"]=1.0
            gains["transfer"]["eligible"]=reference>1e-12
            record["dose_gains"]=gains
            record["reference_discovery_write_RMS"]=reference
            for name in ("target_only","activation","permuted"):
                if not gains[name]["eligible"]:
                    record["excluded_reasons"].append("gain_outside_registered_bounds_"+name)
            shams,sd=local_matched_shams(
                dose_acts,u_b,ej,seed=root,arm_index=q,
                proposals=proposals,want=SHAMS
            )
            record["sham_matching"]=sd
            if not sd.get("admitted",False):
                record["excluded_reasons"].append("fewer_than_eight_local_physical_shams")
            sources[q]={"projections":projects, "shams":shams,
                        "gains":gains, "source_basis":ej,
                        "regularized":r}
        # Gauge is a single prespecified q=32 physical intervention audit.
        if 32 in sources:
            pj=sources[32]["projections"]["transfer"]
            check=x_dose[:min(64,len(x_dose))]
            with torch.no_grad():
                native_logits,_=projected_with_gain(bb,b,check,pj,1.0)
                native_map=pj@bb.wrappers["mix"].U.detach().double()
                factor_basis(bb,"rotated",root+100_000)
                layer=bb.wrappers["mix"]
                rotated_proj=basis_transport(pj,layer.V.detach())
                rotated_map=layer.V.detach().double()@rotated_proj@layer.U.detach().double()
                rotated_logits,_=projected_with_gain(bb,b,check,rotated_proj,1.0)
                rotated_closure=parity(bb,b,check)
                rec["gauge"]={
                    "rotated_fullmask":rotated_closure,
                    "max_physical_operator_relative_error":float(
                        (native_map-rotated_map).norm(dim=(1,2)).max()
                        /layer.W_target.double().norm().clamp_min(1e-12)),
                    "max_closed_loop_logit_delta":float((native_logits-rotated_logits).abs().max()),
                }
                factor_basis(bb,"native",root+100_000)
            if (not rotated_closure["passed"]
                or rec["gauge"]["max_physical_operator_relative_error"]>1e-4
                or rec["gauge"]["max_closed_loop_logit_delta"]>5e-5):
                rec["excluded_global"].append("within_target_B_factor_gauge")
        else:
            rec["excluded_global"].append("missing_q32_gauge_projector")
        # Known negative B score: zero head on dose-only examples.
        with torch.no_grad():
            hw=b.head.weight.detach().clone()
            hb=b.head.bias.detach().clone()
            b.head.weight.zero_()
            b.head.bias.zero_()
        try:
            _,gzero=read_and_score_jacobian(bb,b,x_dose[:min(32,len(x_dose))])
            rec["negative_control"]={
                "gradient_max_abs":float(gzero.abs().max()),
                "zero_head_rejected":float(gzero.abs().max())==0.,
            }
        finally:
            with torch.no_grad():
                b.head.weight.copy_(hw)
                b.head.bias.copy_(hb)
        if not rec["negative_control"]["zero_head_rejected"]:
            rec["excluded_global"].append("zero_head_source_gradient_control")
        # Higher-target-gradient ceiling after the entire q frontier and
        # source-only matching choices have been frozen.
        _,g_full=read_and_score_jacobian(bb,b,x_source)
        g_full_unit,_=unit_jacobians(g_full)
        pfull,dfull,_=projectors_at_ranks(g_full_unit)
        rec["source"]["B_full1024_descriptive_rank4"]=dfull[RANK]
        # With q projectors and sham selections frozen, open final task labels.
        ya=task_labels(x_eval,"A")
        yb=task_labels(x_eval,"B")
        ca,wa,clean_a=clean_reference(ba,a,x_eval,ya)
        cb,wb,clean_b=clean_reference(bb,b,x_eval,yb)
        competence=[clean_a["by_context"][k]["accuracy"] for k in ("shared","specific")]
        competence += [clean_b["by_context"][k]["accuracy"] for k in ("shared","specific")]
        rec["heldout"]["clean_context_accuracies"]=competence
        rec["heldout"]["A_clean"]=clean_a
        rec["heldout"]["B_clean"]=clean_b
        if (min(competence)<.85 or min(train_a["mix_density"],
                                      train_b["mix_density"])<.90):
            rec["excluded_global"].append("model_task_competence_or_density")
        if pa is not None:
            rec["heldout"]["A_owned_rank4"]=score_projector(
                ba,a,x_eval,ya,ca,wa,pa,1.)
        if pfull[RANK] is not None:
            rec["heldout"]["B_full1024_descriptive_rank4"]=score_projector(
                bb,b,x_eval,yb,cb,wb,pfull[RANK],1.)
        for q,src in sources.items():
            arm=rec["q_arms"][str(q)]
            results={}
            for name,p in src["projections"].items():
                results[name]=score_projector(
                    bb,b,x_eval,yb,cb,wb,p,src["gains"][name]["applied_gain"])
            results["shams"]=[
                score_projector(bb,b,x_eval,yb,cb,wb,s,1.)
                for s in src["shams"]
            ]
            arm["observed"]=results
            if len(src["shams"])==SHAMS:
                def nll(k):
                    return _context_nll(results[k],"shared")
                arm["comparisons"]={
                    "target_only_minus_transfer":nll("target_only")-nll("transfer"),
                    "activation_minus_transfer":nll("activation")-nll("transfer"),
                    "permuted_minus_transfer":nll("permuted")-nll("transfer"),
                    "median_sham_minus_transfer":float(np.median([
                        _context_nll(row,"shared") for row in results["shams"]
                    ])-nll("transfer")),
                    "transfer_minus_clean_B":nll("transfer")-_context_nll(clean_b,"shared"),
                }
            arm["technically_admitted"]=not (
                rec["excluded_global"] or arm["excluded_reasons"])
    else:
        rec["excluded_global"].append("calibration_pool_missing_gradients")
    for q in BUDGETS:
        arm=rec["q_arms"].setdefault(str(q),{"budget":q,"excluded_reasons":["unavailable_source"]})
        arm["technically_admitted"]=bool(
            not rec["excluded_global"] and not arm.get("excluded_reasons")
            and len(arm.get("observed",{}).get("shams",[]))==SHAMS
            and len(arm.get("comparisons",{}))>=4
        )
    rec["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rec["protocol_sha256"]=hashlib.sha256(
        Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest()
    return rec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=int,default=SMOKE)
    parser.add_argument("--steps",type=int,default=TRAIN_STEPS)
    parser.add_argument("--discovery-size",type=int,default=DISCOVERY)
    parser.add_argument("--dose-size",type=int,default=DOSE_SIZE)
    parser.add_argument("--evaluation-size",type=int,default=EVAL_SIZE)
    parser.add_argument("--proposals",type=int,default=PROPOSALS)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    r=run(args.root,steps=args.steps,discovery_size=args.discovery_size,
          dose_size=args.dose_size,evaluation_size=args.evaluation_size,
          proposals=args.proposals)
    result=json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as f:
            f.write(result)
    else:
        print(result)


if __name__=="__main__":
    main()
