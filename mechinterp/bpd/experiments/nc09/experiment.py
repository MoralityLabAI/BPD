"""NC-09: cross-task causal projector transport between independently trained recurrences.

The preregistration was committed separately before this module existed.
Only engineering pairs 900..907 and excluded smoke 899 are enabled.
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
from mechinterp.bpd.experiments.nc02.experiment import (
    PASSES, WIDTH, sample_x, parity, binary_kl, communities, shuffle_null
)
from mechinterp.bpd.experiments.nc04.experiment import (
    DenseLooper, factor_basis, self_score_discovery
)
from mechinterp.bpd.experiments.nc05.experiment import (
    projectors_at_ranks, local_dose_matched_shams, projected_rollout
)
from mechinterp.bpd.experiments.nc03.experiment import basis_transport

PROTOCOL_COMMIT = "e85724a42b97d9fa41f82cf76d537e71832250a7"
DEVELOPMENT = tuple(range(900, 908))
SMOKE = 899
RANK = 4
STEPS = 384
DISCOVERY = 1024
ALIGNMENT = 1024
EVALUATION = 1024
SHAMS = 8
PROPOSALS = 2048


def task_labels(x: Tensor, task: str) -> Tensor:
    """External labels; call only for training and final independent scoring."""
    if x.ndim != 2 or x.shape[1] != 5 or task not in ("A", "B"):
        raise ValueError("invalid task or inputs")
    second = 1 if task == "A" else 2
    return torch.where(x[:, 4] > 0, x[:, 0], x[:, second]) > 0


def train_task(root: int, task: str, *, steps: int = STEPS) -> tuple[DenseLooper, dict]:
    if root not in DEVELOPMENT and root != SMOKE:
        raise ValueError("NC09 allows only root 899 and 900..907")
    if task not in ("A", "B"):
        raise ValueError("unknown task")
    torch.set_num_threads(1)
    model_seed = root if task == "A" else root + 100_000
    split = 1 if task == "A" else 11
    torch.manual_seed(model_seed)
    model = DenseLooper()
    training = sample_x(root, split, 2048)
    signs = torch.where(task_labels(training, task), 1., -1.)
    rng = torch.Generator().manual_seed(model_seed + 1_737_701)
    optim = torch.optim.Adam(model.parameters(), lr=.005)
    for _ in range(steps):
        ids = torch.randint(len(training), (128,), generator=rng)
        preds, _ = model(training[ids])
        loss = F.softplus(-2 * signs[ids] * preds[:, -1]).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        p, _ = model(training)
        acc = float(((p[:,-1] >= 0) == task_labels(training,task)).float().mean())
        density = float((model.mix.weight.detach().abs() >= 1e-4).float().mean())
    return model, {
        "task": task, "root": root, "model_seed": model_seed,
        "training_split": split, "training_size": len(training),
        "training_steps": steps, "batch_size": 128, "learning_rate": .005,
        "training_accuracy": acc, "mix_density": density,
        "final_minibatch_loss_nats": float(loss.detach()),
    }


@torch.no_grad()
def prewrite_states(bpd: BPDModel, model: DenseLooper, x: Tensor) -> Tensor:
    """Record physical pre-mix reads, independent of task labels and logits."""
    with bpd.target_capture():
        model(x, capture=True)
        acts = torch.stack(model.acts, dim=1)
    if acts.shape != (len(x), PASSES, WIDTH):
        raise RuntimeError("incomplete shared-physical write capture")
    return acts.double()


def procrustes(source: Tensor, target: Tensor) -> tuple[Tensor, dict]:
    """Per-pass orthogonal source-to-target input-coordinate alignment."""
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError("paired activations must be [N,passes,width]")
    if source.shape[1:] != (PASSES,WIDTH) or len(source)<2:
        raise ValueError("insufficient or malformed Procrustes input")
    cross = torch.einsum("nti,ntj->tij", source.double(), target.double())
    u, singular, vh = torch.linalg.svd(cross, full_matrices=False)
    r = u @ vh
    aligned = source.double() @ r
    errs = (aligned-target.double()).norm(dim=(0,2)) / target.double().norm(dim=(0,2)).clamp_min(1e-12)
    orth = torch.eye(WIDTH, dtype=torch.float64)
    return r, {
        "alignment_source_cases": len(source),
        "relative_alignment_residual_by_pass": [float(e) for e in errs],
        "max_orthogonality_error": float((r.transpose(-1,-2)@r - orth).abs().max()),
        "min_singular_value_by_pass": [float(v) for v in singular.min(-1).values],
    }


def transport(source_basis: Tensor, r: Tensor) -> tuple[Tensor, Tensor]:
    """If Z_B ≈ Z_A R, its corresponding subspace basis is R^T E_A."""
    if source_basis.ndim != 3 or source_basis.shape != (PASSES,WIDTH,RANK):
        raise ValueError("wrong source rank-four basis")
    if r.shape != (PASSES,WIDTH,WIDTH):
        raise ValueError("wrong alignment tensor")
    e = r.transpose(-1,-2) @ source_basis.double()
    return e@e.transpose(-1,-2),e


def projector_overlap(a: Tensor, b: Tensor) -> list[float]:
    if a.shape != b.shape or a.shape != (PASSES,WIDTH,WIDTH):
        raise ValueError("expected pair of projector matrices")
    return [float(torch.trace(a[t] @ b[t]) / RANK) for t in range(PASSES)]


@torch.no_grad()
def clean_reference(bpd: BPDModel, model: DenseLooper, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, dict]:
    with bpd.target_capture():
        logits,_ = model(x, capture=True)
        writes = torch.stack(model.writes, dim=1)
    return logits,writes,score_logits(logits,logits,writes,writes,x,y)


@torch.no_grad()
def score_logits(clean: Tensor, changed: Tensor, clean_w: Tensor, changed_w: Tensor,
                 x: Tensor, y: Tensor) -> dict:
    """External labels enter only at final held-out scoring."""
    sign = torch.where(y,1.,-1.)
    nll = F.softplus(-2*sign*changed[:,-1])
    nll_clean = F.softplus(-2*sign*clean[:,-1])
    kl = binary_kl(clean,changed).clamp_min(0)
    out = {
        "final_nll_nats": float(nll.mean()),
        "nll_change_from_clean_nats": float((nll-nll_clean).mean()),
        "final_accuracy": float(((changed[:,-1]>=0)==y).float().mean()),
        "kl_by_pass_nats": [float(v) for v in kl.mean(0)],
        "closed_loop_write_displacement_rms": float((changed_w-clean_w).square().mean().sqrt()),
        "by_context": {},
    }
    for name,mask in (("shared",x[:,4]>0),("specific",x[:,4]<=0)):
        if not bool(mask.any()):
            raise RuntimeError("evaluation lacks registered task context")
        out["by_context"][name] = {
            "n": int(mask.sum()),
            "final_nll_nats": float(nll[mask].mean()),
            "nll_change_from_clean_nats": float((nll[mask]-nll_clean[mask]).mean()),
            "accuracy": float(((changed[mask,-1]>=0)==y[mask]).float().mean()),
        }
    return out


@torch.no_grad()
def score_projector(bpd: BPDModel, model: DenseLooper, x: Tensor, y: Tensor,
                    clean: Tensor, clean_w: Tensor, projector: Tensor) -> dict:
    new_logits, new_writes = projected_rollout(bpd, model, x, projector, 1.0)
    return score_logits(clean,new_logits,clean_w,new_writes,x,y)


def discover(bpd: BPDModel, model: DenseLooper, x: Tensor, root: int,
             task: str) -> tuple[Tensor|None,Tensor,dict,Tensor]:
    """Self-score discovery: no external labels or training features accepted."""
    acts,grads,hits,info = self_score_discovery(bpd,model,x)
    projectors,diagnostics,bases = projectors_at_ranks(grads)
    graph = communities(hits)
    desc = {
        "source_info":info,
        "rank4":diagnostics[RANK],
        "graph_cross_pass_count":sum(len(g["passes"])>=2 for g in graph),
        "graph_n_communities":len(graph),
        "graph_shuffle_null":shuffle_null(hits, root + (0 if task=="A" else 10000), n_shuffles=16),
    }
    return projectors[RANK], bases[RANK], desc, acts


def run(root:int,*,steps:int=STEPS,discovery_size:int=DISCOVERY,
        alignment_size:int=ALIGNMENT,evaluation_size:int=EVALUATION,
        proposals:int=PROPOSALS) -> dict:
    if root != SMOKE and root not in DEVELOPMENT:
        raise ValueError("NC-09 permits only root 899 smoke and 900..907 development")
    torch.set_num_threads(1)
    model_a,trained_a = train_task(root,"A",steps=steps)
    model_b,trained_b = train_task(root,"B",steps=steps)
    xa = sample_x(root,2,discovery_size)
    xalign = sample_x(root,4,alignment_size)
    xe = sample_x(root,3,evaluation_size)
    source_positive = xalign[:,4]>0
    positive_n = int(source_positive.sum())
    bpd_a = BPDModel(model_a,{"mix":WIDTH},max_passes=PASSES,
                     ablation=AblationConfig(mode="zero",sphere_rescale=False))
    bpd_b = BPDModel(model_b,{"mix":WIDTH},max_passes=PASSES,
                     ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bpd_a,"native",root)
    factor_basis(bpd_b,"native",root+100_000)
    closure_a = parity(bpd_a,model_a,xe[:min(64,len(xe))])
    closure_b = parity(bpd_b,model_b,xe[:min(64,len(xe))])
    pa,ea,da,acts_a=discover(bpd_a,model_a,xa,root,"A")
    pb,eb,db,acts_b=discover(bpd_b,model_b,xa,root,"B")
    receipt={
        "study":"BPD-NC-09","phase":"ENGINEERING",
        "root":root,"protocol_commit":PROTOCOL_COMMIT,
        "splits":{"A_train":1,"B_train":11,"discovery":2,"alignment":4,"evaluation":3},
        "budgets":{"train_A":2048,"train_B":2048,"steps":steps,
                   "discovery":discovery_size,"alignment":alignment_size,
                   "evaluation":evaluation_size,"sham_proposals":proposals},
        "models":{"A":trained_a,"B":trained_b},
        "closure":{"A":closure_a,"B":closure_b},
        "discovery":{"A":da,"B":db},
        "alignment":{},
        "sham_matching":{},
        "gauge":{},
        "negative_control":{},
        "evaluations":{},
        "contrasts":{},
        "eligibility_reasons":[],
    }
    if positive_n<300:
        receipt["eligibility_reasons"].append("insufficient_positive_context_alignment")
        # Smoke uses 256 alignment inputs and is an engineering-only source test.
    if pa is None or pb is None:
        receipt["eligibility_reasons"].append("source_or_target_rank4_unidentifiable")
    if not closure_a["passed"] or not closure_b["passed"]:
        receipt["eligibility_reasons"].append("full_mask_parity_failed")
    if pa is not None and pb is not None and positive_n>=2 and all(
        x["passed"] for x in (closure_a,closure_b)
    ):
        za=prewrite_states(bpd_a,model_a,xalign[source_positive])
        zb=prewrite_states(bpd_b,model_b,xalign[source_positive])
        r,al = procrustes(za,zb)
        rng=np.random.default_rng(root+9_090_009)
        permutation=rng.permutation(len(zb))
        r_null,al_null=procrustes(za,zb[torch.as_tensor(permutation,dtype=torch.long)])
        transported,e_trans=transport(ea,r)
        transported_perm,e_perm=transport(ea,r_null)
        receipt["alignment"]={
            "paired_procrustes":al,"permuted_procrustes":al_null,
            "permutation_sha256":hashlib.sha256(permutation.tobytes()).hexdigest(),
            "task_owned_projector_overlap_by_pass":projector_overlap(transported,pb),
            "positive_context_count":positive_n,
            "transport_projector_idempotence_error":float((transported@transported-transported).abs().max()),
        }
        shams,matching=local_dose_matched_shams(
            acts_b,bpd_b.wrappers["mix"].U.detach(),e_trans,RANK,root,
            proposals=proposals,want=SHAMS
        )
        receipt["sham_matching"]=matching
        if not matching["eligible"]:
            receipt["eligibility_reasons"].append("fewer_than_eight_discovery_dose_matched_shams")
        # Target B factor-gauge control after discovery and shams; target weights
        # are identical, and Q is a true factor coordinate change.
        native_logits,_=projected_rollout(bpd_b,model_b,xe[:min(64,len(xe))],transported,1.)
        native_map=transported @ bpd_b.wrappers["mix"].U.detach().double()
        factor_basis(bpd_b,"rotated",root+100_000)
        pr=basis_transport(transported,bpd_b.wrappers["mix"].V.detach())
        rotated_map=(bpd_b.wrappers["mix"].V.detach().double()@pr@
                     bpd_b.wrappers["mix"].U.detach().double())
        rotated_logits,_=projected_rollout(bpd_b,model_b,xe[:min(64,len(xe))],pr,1.)
        rotated_closure=parity(bpd_b,model_b,xe[:min(64,len(xe))])
        receipt["gauge"]={
            "rotated_closure":rotated_closure,
            "max_transported_operator_relative_error":float(
                (native_map-rotated_map).norm(dim=(1,2)).max() /
                bpd_b.wrappers["mix"].W_target.double().norm().clamp_min(1e-12)),
            "max_transported_kept_logit_error":float((native_logits-rotated_logits).abs().max()),
        }
        factor_basis(bpd_b,"native",root+100_000)
        if (not rotated_closure["passed"] or
            receipt["gauge"]["max_transported_operator_relative_error"]>1e-4 or
            receipt["gauge"]["max_transported_kept_logit_error"]>5e-5):
            receipt["eligibility_reasons"].append("target_factor_gauge_parity_failed")
        # The first evaluation labels are generated only after source-only
        # projector/rank/Procrustes/sham selection and gauge diagnostics.
        ya=task_labels(xe,"A")
        yb=task_labels(xe,"B")
        ca,wa,clean_a=clean_reference(bpd_a,model_a,xe,ya)
        cb,wb,clean_b=clean_reference(bpd_b,model_b,xe,yb)
        scored={
            "A_clean":clean_a,"B_clean":clean_b,
            "A_owned":score_projector(bpd_a,model_a,xe,ya,ca,wa,pa),
            "B_owned":score_projector(bpd_b,model_b,xe,yb,cb,wb,pb),
            "A_transferred_to_B":score_projector(bpd_b,model_b,xe,yb,cb,wb,transported),
            "A_identity_to_B":score_projector(bpd_b,model_b,xe,yb,cb,wb,pa),
            "A_permuted_alignment_to_B":score_projector(bpd_b,model_b,xe,yb,cb,wb,transported_perm),
            "matched_shams": [
                score_projector(bpd_b,model_b,xe,yb,cb,wb,q) for q in shams
            ],
        }
        receipt["evaluations"]=scored
        clean_gate=[]
        for name in ("A_clean","B_clean"):
            obj=scored[name]
            accs=[obj["by_context"][k]["accuracy"] for k in ("shared","specific")]
            clean_gate.extend(accs)
        receipt["model_competence_context_accuracies"]=clean_gate
        if (min(clean_gate) <.85 or
            trained_a["mix_density"]<.90 or trained_b["mix_density"]<.90):
            receipt["eligibility_reasons"].append("model_task_or_dense_weight_gate")
        def nll(name,ctx):
            return scored[name]["by_context"][ctx]["final_nll_nats"]
        receipt["contrasts"]={
            "shared_transfer_deficit_vs_B_owned":nll("A_transferred_to_B","shared")-nll("B_owned","shared"),
            "specific_transfer_deficit_vs_B_owned":nll("A_transferred_to_B","specific")-nll("B_owned","specific"),
            "branch_specificity": (
                nll("A_transferred_to_B","specific")-nll("B_owned","specific")
                -nll("A_transferred_to_B","shared")+nll("B_owned","shared")
            ),
            "paired_alignment_gain_shared":nll("A_permuted_alignment_to_B","shared")-nll("A_transferred_to_B","shared"),
            "identity_alignment_gain_shared":nll("A_identity_to_B","shared")-nll("A_transferred_to_B","shared"),
        }
        if len(shams)==SHAMS:
            receipt["contrasts"]["matched_sham_advantage_shared"]=float(np.median(
                [v["by_context"]["shared"]["final_nll_nats"] for v in scored["matched_shams"]]
            ) -nll("A_transferred_to_B","shared"))
    # Known zero-head negative control, discovery-only.
    factor_basis(bpd_b,"native",root+100_000)
    with torch.no_grad():
        old_w=model_b.head.weight.detach().clone()
        old_b=model_b.head.bias.detach().clone()
        model_b.head.weight.zero_()
        model_b.head.bias.zero_()
    try:
        _,g0,h0,_=self_score_discovery(bpd_b,model_b,xa[:min(256,len(xa))])
        zproj,zdiag,_=projectors_at_ranks(g0)
        receipt["negative_control"]={
            "max_abs_gradient":float(g0.abs().max()),
            "positive_attribution_hits":int(h0.sum()),
            "rank4_unidentifiable":zproj[RANK] is None,
            "rank4_status":zdiag[RANK]["status"],
        }
    finally:
        with torch.no_grad():
            model_b.head.weight.copy_(old_w)
            model_b.head.bias.copy_(old_b)
    if (receipt["negative_control"]["max_abs_gradient"] !=0 or
        receipt["negative_control"]["positive_attribution_hits"] !=0 or
        not receipt["negative_control"]["rank4_unidentifiable"]):
        receipt["eligibility_reasons"].append("zero_head_negative_control")
    # Smoke is non-inferential; this flag describes technical admission only.
    receipt["eligible"]=len(receipt["eligibility_reasons"])==0
    receipt["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt["protocol_sha256"]=hashlib.sha256(
        Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest()
    return receipt


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=int,default=SMOKE)
    parser.add_argument("--steps",type=int,default=STEPS)
    parser.add_argument("--discovery-size",type=int,default=DISCOVERY)
    parser.add_argument("--alignment-size",type=int,default=ALIGNMENT)
    parser.add_argument("--evaluation-size",type=int,default=EVALUATION)
    parser.add_argument("--proposals",type=int,default=PROPOSALS)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    data=run(args.root,steps=args.steps,discovery_size=args.discovery_size,
             alignment_size=args.alignment_size,evaluation_size=args.evaluation_size,
             proposals=args.proposals)
    payload=json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as f:f.write(payload)
    else:
        print(payload)


if __name__=="__main__":
    main()
