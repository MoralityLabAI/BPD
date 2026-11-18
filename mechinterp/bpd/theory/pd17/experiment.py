"""BPD-PD-17: paired original/radial-camouflage mechanism discovery.

Frozen CONFIG and PROTOCOL preceded this source. All trained arms consume
only dense observed x. Q/support are confined to the source generator, the
privileged oracle, and independent post-fit physical/diagnostic evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import resource
import time
from pathlib import Path

import torch

from mechinterp.bpd.components import PassMaskSchedule
from mechinterp.bpd.theory.pd02.experiment import exact_mcnemar_p, wilson_ci
from mechinterp.bpd.theory.pd12.experiment import (
    initial_random_basis, physical_model, basis_alignment
)
from mechinterp.bpd.theory.pd15.experiment import fit_basis as pd15_fit
from mechinterp.bpd.theory.pd17.angular import angular_basis
from mechinterp.bpd.theory.pd17.source import (
    CONFIG, D, FAMILIES, training_source, heldout_source
)

ROOT=Path(__file__).resolve().parent
SEEDS=tuple(CONFIG["seeds"])
LAWS=tuple(CONFIG["source_laws"])
ARMS=tuple(a["key"] for a in CONFIG["arms"])
SC=CONFIG["scoring"]


@torch.no_grad()
def evaluate_family_exact(bpd, q, seed:int, law:str, family:str,
                          alignment:dict, boards:int):
    """Native physical tied-component calls, same PD12 mask semantics."""
    src=heldout_source(seed,law,family,boards=boards)
    x, latent_support=src["x"],src["mask"]
    v=bpd.wrappers["core"].V
    normalized=v/v.square().sum(0).sqrt().clamp_min(1e-6)
    predicted_mask=(x@normalized).abs().ge(SC["fixed_gate"]).float()
    expected=latent_support[...,alignment["component_to_latent"]]
    if predicted_mask.shape!=expected.shape:
        raise AssertionError("Wrong physical component-mask dimensions")
    schedule=PassMaskSchedule(
        comp_mask=predicted_mask,
        delta_mask=torch.zeros_like(predicted_mask[...,0])
    )
    with bpd.masked_components({"core":schedule}):
        y=torch.stack([bpd.target_model(x[:,t]) for t in range(4)],dim=1)
    correct=(predicted_mask==expected).all(dim=(-1,-2,-3))
    return {
        "mse":float((y-x).square().mean()),
        "support_fraction":float(correct.float().mean()),
        "correct_boards":int(correct.sum()),
        "boards":int(boards),
        "false_off":int(((predicted_mask==0)&(expected==1)).sum()),
        "false_on":int(((predicted_mask==1)&(expected==0)).sum()),
        "binary_mask":bool(((predicted_mask==0)|(predicted_mask==1)).all()),
        "support_sha256":src["support_sha256"],
    }


def qualifies(row:dict) -> bool:
    if row["training"].get("abstained",False):
        return False
    h=row["heldout"]
    if h["alignment"]["mean_cosine"]<SC["mean_read_cos_min"]:
        return False
    if h["overall_mse"]>SC["overall_MSE_max"]:
        return False
    for fam in FAMILIES:
        m=h["per_family"][fam]
        floor=(SC["unseen_full_fourpass_support_min"]
               if "unseen" in fam else SC["seen_full_fourpass_support_min"])
        if m["mse"]>SC["per_family_MSE_max"] or m["support_fraction"]<floor:
            return False
        if not m["binary_mask"]:
            return False
    if row["arm"]=="planted_Q_oracle":
        return True
    t=row["training"]
    return (t["nonzero_moves"]>0 and t["orthogonality_max_error"] is not None
            and t["orthogonality_max_error"]<SC["orthogonality_error_max"])


def run_arm(seed:int, law:str, arm:str, *, pool:dict,
            boards:int|None=None, sweeps:int|None=None):
    if law not in LAWS or arm not in ARMS:
        raise ValueError((law,arm))
    torch.set_num_threads(1)
    started=time.monotonic()
    q,x=pool["q"],pool["x"]
    initial=initial_random_basis(seed,D)
    if arm=="planted_Q_oracle":
        read=q.double()
        tr={
            "nonzero_moves":0,"candidate_angle_scores":0,
            "candidate_scalar_projection_evaluations":0,
            "orthogonality_max_error":float(
                (read.T@read-torch.eye(D,dtype=torch.float64)).abs().max()
            ),
            "training_seconds":0.0
        }
    elif arm=="angular_atoms":
        read,tr=angular_basis(x)
        if read is None:
            read=initial.double()
            tr["orthogonality_max_error"]=float(
                (read.T@read-torch.eye(D,dtype=torch.float64)).abs().max())
    else:
        lam=1. if arm=="norm_givens" else 0.
        sigma=CONFIG["givens"]["sigma"]
        read,tr=pd15_fit(x,initial,sigma,lam,sweeps=sweeps)
        tr["angular_selection_unmatched"]=False
    bpd=physical_model(D,read)
    alignment=basis_alignment(read,q)
    n=CONFIG["source_rng"]["eval_boards_per_family"] if boards is None else boards
    with torch.no_grad():
        per={
            fam:evaluate_family_exact(
                bpd,q,seed,law,fam,alignment,n
            ) for fam in FAMILIES
        }
    record={
        "study_id":"BPD-PD-17",
        "seed":seed,"law":law,"arm":arm,
        "q_sha256":pool["q_sha256"],
        "training_support_sha256":pool["support_sha256"],
        "training_pool_sha256":pool["pool_sha256"],
        "initial_random_basis_sha256":hashlib.sha256(
            initial.numpy().tobytes()).hexdigest(),
        "uses_Q_in_optimizer":arm=="planted_Q_oracle",
        "training":tr,
        "heldout":{
            "alignment":alignment,
            "overall_mse":sum(m["mse"] for m in per.values())/len(per),
            "physical_weight_delta_mse":float(
                bpd.wrappers["core"].weight_delta().square().mean()
            ),
            "per_family":per,
        },
        "walltime_seconds":round(time.monotonic()-started,6),
    }
    record["success"]=qualifies(record)
    return record


def pearson(a:torch.Tensor,b:torch.Tensor) -> float:
    a,b=a.double().flatten(),b.double().flatten()
    a,b=a-a.mean(),b-b.mean()
    denom=a.norm()*b.norm()
    if denom<=0:
        raise ValueError("Pearson undefined on a zero-variance source")
    return float((a@b)/denom)


def root_source_diagnostic(seed:int,orig:dict,cam:dict):
    """Use known K only post-fit, never in PD15 fit or angular_ray selection."""
    if (orig["q_sha256"]!=cam["q_sha256"] or
        orig["support_sha256"]!=cam["support_sha256"]):
        raise AssertionError("Paired source Q/support mismatch")
    k=orig["mask"].reshape(-1,D).sum(-1)
    orig2=orig["x"].reshape(-1,D).double().square().sum(-1)
    cam2=cam["x"].reshape(-1,D).double().square().sum(-1)
    c0,c1=pearson(orig2,k),pearson(cam2,k)
    pass_gate=(
        c0>=SC["original_corr_K_norm2_min"] and
        abs(c1)<=SC["camouflaged_abs_corr_K_norm2_max"]
    )
    return {
        "seed":seed,
        "original_corr_K_norm2":c0,
        "camouflaged_corr_K_norm2":c1,
        "source_independence_gate_passed":bool(pass_gate),
        "paired_support_sha256":orig["support_sha256"],
        "paired_q_sha256":orig["q_sha256"],
        "camouflage_radial_min":float(cam2.sqrt().min()),
        "camouflage_radial_max":float(cam2.sqrt().max()),
        "true_K_used_only_postfit":True,
    }


def paired_root_bootstrap(values:list[int]):
    """Approximate percentile CI, resampling entire independent roots."""
    if len(values)!=32:
        raise ValueError("PD17 statistics require 32 independent roots")
    n=SC["bootstrap_repeats"]
    rng=random.Random(SC["bootstrap_seed"])
    out=[]
    for _ in range(n):
        out.append(sum(values[rng.randrange(len(values))]
                       for j in range(len(values)))/len(values))
    out.sort()
    return {
        "resamples":n,
        "seed":SC["bootstrap_seed"],
        "lower95":out[int(.025*(n-1))],
        "upper95":out[int(.975*(n-1))],
    }


def summarize(rows:list[dict],diagnostics:list[dict],*,registered:bool):
    expected={(s,law,arm) for s in SEEDS for law in LAWS for arm in ARMS}
    by={(r["seed"],r["law"],r["arm"]):r for r in rows}
    dby={d["seed"]:d for d in diagnostics}
    if (len(rows)!=len(by) or set(by)!=expected or
        len(diagnostics)!=len(dby) or set(dby)!=set(SEEDS)):
        raise ValueError("Incomplete or duplicated PD17 paired source cohort")
    counts={law:{
        arm:sum(by[(s,law,arm)]["success"] for s in SEEDS)
        for arm in ARMS
    } for law in LAWS}
    effects=[]
    for s in SEEDS:
        def outcome(law,arm):return int(by[(s,law,arm)]["success"])
        effects.append(
            (outcome("original","norm_givens")-outcome("original","uniform_givens"))-
            (outcome("radial_camouflage","norm_givens")-
             outcome("radial_camouflage","uniform_givens"))
        )
    boot=paired_root_bootstrap(effects)
    mean=sum(effects)/len(effects)
    ngate=sum(dby[s]["source_independence_gate_passed"] for s in SEEDS)
    admitted=all(counts[law]["planted_Q_oracle"]==32 for law in LAWS)
    all_rows=bool(registered and len(rows)==32*2*4 and len(diagnostics)==32)
    effect_pass=bool(mean>=SC["primary_min_mean_interaction"] and
                     boot["lower95"]>SC["primary_bootstrap_lower95_min"])
    success=bool(
        all_rows and admitted and effect_pass
        and counts["original"]["norm_givens"]>=SC["original_norm_replication_min_successes"]
        and counts["radial_camouflage"]["angular_atoms"]>=
            SC["angular_camouflage_feasibility_min_successes"]
        and ngate>=SC["source_independence_gate_min_roots"]
    )
    paired_mcnemar={}
    for law in LAWS:
        pos=sum(by[(s,law,"norm_givens")]["success"] and
                not by[(s,law,"uniform_givens")]["success"] for s in SEEDS)
        neg=sum(by[(s,law,"uniform_givens")]["success"] and
                not by[(s,law,"norm_givens")]["success"] for s in SEEDS)
        paired_mcnemar[law]={
            "norm_only":pos,"uniform_only":neg,
            "exact_two_sided_p_descriptive":exact_mcnemar_p(pos,neg),
        }
    return {
        "study_id":"BPD-PD-17",
        "registered":all_rows,
        "units":32,"arms":len(rows),
        "success_by_law_and_arm":counts,
        "Wilson95_by_law_and_arm":{
            law:{arm:wilson_ci(counts[law][arm],32) for arm in ARMS}
            for law in LAWS
        },
        "root_interactions":dict(zip(SEEDS,effects)),
        "mean_root_interaction":mean,
        "root_bootstrap":boot,
        "original_replication_gate_passed":
            counts["original"]["norm_givens"]>=SC["original_norm_replication_min_successes"],
        "camouflage_observability_gate_passed":
            counts["radial_camouflage"]["angular_atoms"]>=
            SC["angular_camouflage_feasibility_min_successes"],
        "independent_radial_source_gate_passes":ngate,
        "radial_source_gate_passed":ngate>=SC["source_independence_gate_min_roots"],
        "physical_oracles_admitted":admitted,
        "interaction_gate_passed":effect_pass,
        "registered_scientific_passed":success,
        "per_law_paired_mcnemar":paired_mcnemar,
        "per_root":[{
            "seed":s,
            "interaction":effects[i],
            "source_independence_gate":dby[s]["source_independence_gate_passed"],
            "source_original_corr":dby[s]["original_corr_K_norm2"],
            "source_camouflaged_corr":dby[s]["camouflaged_corr_K_norm2"],
            "success":{
                law:{arm:by[(s,law,arm)]["success"] for arm in ARMS}
                for law in LAWS
            },
        } for i,s in enumerate(SEEDS)]
    }


def run_registered():
    torch.set_num_threads(1)
    t0=time.monotonic()
    rows,diagnostics=[],[]
    for seed in SEEDS:
        original=training_source(seed,"original")
        cam=training_source(seed,"radial_camouflage")
        for law,pool in (("original",original),("radial_camouflage",cam)):
            for arm in ARMS:
                rows.append(run_arm(seed,law,arm,pool=pool))
        diagnostics.append(root_source_diagnostic(seed,original,cam))
        if len([r for r in rows if r["seed"]==seed])!=8:
            raise AssertionError("Missing paired-source model arms")
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        if peak>CONFIG["execution"]["max_RSS_GiB"]:
            raise MemoryError(f"PD17 peak RSS {peak:.3f} GiB")
    return {
        "study_id":CONFIG["study_id"],
        "status":"first_registered_two_source_identifiability_and_radial_camouflage",
        "source_hashes":{
            name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("CONFIG.json","PROTOCOL.md","source.py","angular.py","experiment.py")
        },
        "python_version":platform.python_version(),
        "torch_version":torch.__version__,
        "scipy_version":__import__("scipy").__version__,
        "elapsed_seconds":round(time.monotonic()-t0,6),
        "peak_RSS_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        "summary":summarize(rows,diagnostics,registered=True),
        "arms":rows,
        "diagnostics":diagnostics,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    record=run_registered()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print(json.dumps(record["summary"],indent=2,sort_keys=True))


if __name__=="__main__":
    main()
