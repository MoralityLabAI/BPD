"""BPD-PD-09 source-locked unsupervised smooth hyperplane-mass basis audit.

The orthogonal arms train a differentiable QR parameter A, with physical
read/write V=Q(A), U=Q(A).T installed into real BPDModel/TiedComponentLinear
for every heldout intervention. Untied_sparse trains both actual BPD factors.

Only the explicitly named count_supervised_positive arm accesses cardinality
labels during training. All unsupervised objectives accept x alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import time
from pathlib import Path

import torch

from mechinterp.bpd.components import PassMaskSchedule
from mechinterp.bpd.theory.pd06.experiment import (
    build as build_pd06, objective as pd06_objective,
    rank4_alignment, sample_batch, exact_mcnemar_p, wilson_ci,
)

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/"CONFIG.json").read_text(encoding="utf-8"))
SEEDS=tuple(CONFIG["seeds"])
ARMS=tuple(x["key"] for x in CONFIG["arms"])
FAMILIES=tuple(CONFIG["source"]["family_order"])
ARM_CONFIG={x["key"]:x for x in CONFIG["arms"]}
WIDTH=4


def qr_orthogonal(matrix):
    """Differentiable QR with deterministic R diagonal sign convention."""
    q,r=torch.linalg.qr(matrix)
    sign=torch.where(torch.diagonal(r)>=0,1.,-1.)
    return q*sign.unsqueeze(0)


def smoothed_l0(x,read,sigma):
    """Unsupervised expected-L0 surrogate: no access to support/count labels."""
    norm=read.pow(2).sum(dim=0).sqrt().clamp_min(1e-6)
    z=(x@read)/norm
    return (1.-torch.exp(-.5*(z/sigma).square())).mean()


def sigma_for_step(arm,step):
    schedule=ARM_CONFIG[arm].get("sigma_schedule")
    if not schedule:
        raise ValueError("this arm has no smoothing schedule")
    index=min(step//120,2)
    return float(schedule[index])


def untied_penalty(bpd,x,sigma):
    """Train both physical factors while penalizing read mixing/scale."""
    wrapper=bpd.wrappers["core"]
    v=wrapper.V
    norm=v.pow(2).sum(dim=0).sqrt().clamp_min(1e-6)
    unit=v/norm
    gram=unit.T@unit
    off=gram-torch.diag(torch.diagonal(gram))
    sparsity=smoothed_l0(x,v,sigma)
    faith=wrapper.weight_delta().square().mean()
    column_norm=(norm-1.).square().mean()
    o=CONFIG["optimization"]
    return (sparsity+.30*faith+.50*off.square().mean()
            +.15*column_norm)


def source_only_objective(arm,x,*,raw_rotation=None,bpd=None,step=0):
    """Pure unlabelled objectives ignore source labels by construction."""
    sigma=sigma_for_step(arm,step)
    if arm in ("orthogonal_annealed","orthogonal_fixed"):
        if raw_rotation is None:
            raise ValueError("orthogonal arm needs trainable raw matrix")
        return smoothed_l0(x,qr_orthogonal(raw_rotation),sigma)
    if arm=="untied_sparse":
        if bpd is None:
            raise ValueError("untied arm needs actual BPD physical factors")
        return untied_penalty(bpd,x,sigma)
    raise ValueError("source-only objective only for unsupervised trainable arms")


def set_physical_orthogonal(bpd,rotation):
    """Install square orthogonal read/write physical BPD factors."""
    with torch.no_grad():
        wrapper=bpd.wrappers["core"]
        wrapper.V.copy_(rotation)
        wrapper.U.copy_(rotation.T)


def parameter_fingerprint(bpd):
    wrapper=bpd.wrappers["core"]
    return hashlib.sha256(json.dumps([
        wrapper.V.detach().tolist(),wrapper.U.detach().tolist()
    ],separators=(",",":")).encode()).hexdigest()


def train_one(seed,arm,*,steps=None,batch_size=None):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.set_num_threads(1)
    conf=CONFIG["optimization"]
    bpd,stage1_gate,q,initial_sha,q_hash=build_pd06(
        seed,frozen_planted=False
    )
    if steps is None:
        steps=conf["steps"]
    if batch_size is None:
        batch_size=conf["batch_size"]
    originals=[p.detach().clone() for p in bpd.component_parameters()]
    first_grad=0.
    if arm in ("orthogonal_annealed","orthogonal_fixed","random_orthogonal"):
        raw=torch.nn.Parameter(bpd.wrappers["core"].V.detach().clone())
        if arm=="random_orthogonal":
            set_physical_orthogonal(bpd,qr_orthogonal(raw).detach())
            update=0.
            count=0
        else:
            optimizer=torch.optim.Adam([raw],lr=conf["Adam_lr"],weight_decay=0.)
            gen=torch.Generator().manual_seed(
                CONFIG["source"]["train_generator_offset"]+seed
            )
            for step in range(steps):
                x,_support,_=sample_batch(gen,batch_size,q,family="train")
                optimizer.zero_grad(set_to_none=True)
                loss=source_only_objective(
                    arm,x,raw_rotation=raw,step=step
                )
                loss.backward()
                if step==0:
                    first_grad=float(raw.grad.square().sum().sqrt())
                optimizer.step()
            set_physical_orthogonal(bpd,qr_orthogonal(raw).detach())
            update=float((raw.detach()-originals[0]).square().sum().sqrt())
            count=steps
    elif arm=="untied_sparse":
        params=bpd.component_parameters()
        optimizer=torch.optim.Adam(params,lr=conf["Adam_lr"],weight_decay=0.)
        gen=torch.Generator().manual_seed(
            CONFIG["source"]["train_generator_offset"]+seed
        )
        for step in range(steps):
            x,_support,_=sample_batch(gen,batch_size,q,family="train")
            optimizer.zero_grad(set_to_none=True)
            loss=source_only_objective(
                arm,x,bpd=bpd,step=step
            )
            loss.backward()
            if step==0:
                first_grad=math.sqrt(sum(
                    float(p.grad.square().sum()) for p in params
                    if p.grad is not None
                ))
            optimizer.step()
        update=math.sqrt(sum(
            float((p.detach()-old).square().sum())
            for p,old in zip(params,originals)
        ))
        count=steps
    elif arm=="count_supervised_positive":
        params=bpd.component_parameters()
        optimizer=torch.optim.Adam(
            list(stage1_gate.parameters())+params,
            lr=conf["Adam_lr"],weight_decay=0.,
        )
        gen=torch.Generator().manual_seed(
            CONFIG["source"]["train_generator_offset"]+seed
        )
        for step in range(steps):
            x,support,_=sample_batch(gen,batch_size,q,family="train")
            optimizer.zero_grad(set_to_none=True)
            loss,_=pd06_objective(
                bpd,stage1_gate,x,support,"count_supervised"
            )
            loss.backward()
            if step==0:
                first_grad=math.sqrt(sum(
                    float(p.grad.square().sum()) for p in params
                    if p.grad is not None
                ))
            optimizer.step()
        update=math.sqrt(sum(
            float((p.detach()-old).square().sum())
            for p,old in zip(params,originals)
        ))
        count=steps
    else:
        raise ValueError(arm)
    return bpd,q,{
        "seed":seed,
        "arm":arm,
        "rotation_sha256":q_hash,
        "initial_parameters_sha256":initial_sha,
        "physical_factor_sha256":parameter_fingerprint(bpd),
        "training_updates":count,
        "first_trainable_gradient_norm":first_grad,
        "trainable_update_norm":update,
        "raw_matrix_QR_gauge":arm in ("orthogonal_annealed","orthogonal_fixed","random_orthogonal"),
        "label_privileged":arm=="count_supervised_positive",
    }


@torch.no_grad()
def evaluate_family(bpd,q,seed,arm,family,mapping,boards):
    gen=torch.Generator().manual_seed(
        CONFIG["source"]["heldout_generator_offset"]+seed+
        CONFIG["source"]["heldout_family_stride"]*FAMILIES.index(family)
    )
    x,latent_support,_=sample_batch(gen,boards,q,family=family)
    v=bpd.wrappers["core"].V
    normalized=v/v.pow(2).sum(dim=0).sqrt().clamp_min(1e-6)
    z=x@normalized
    mask=(z.abs()>=.5).float()
    assert bool(((mask==0)|(mask==1)).all())
    schedule=PassMaskSchedule(
        comp_mask=mask,
        delta_mask=torch.zeros_like(mask[...,0]),
    )
    with bpd.masked_components({"core":schedule}):
        output=torch.stack([
            bpd.target_model(x[:,t]) for t in range(4)
        ],dim=1)
    truth=latent_support[...,list(mapping)]
    complete=(mask==truth).all(dim=(-1,-2,-3))
    return {
        "mse":float((output-x).square().mean()),
        "support_fraction":float(complete.float().mean()),
        "correct_boards":int(complete.sum()),
        "boards":int(boards),
        "false_off":int(((mask==0)&(truth==1)).sum()),
        "false_on":int(((mask==1)&(truth==0)).sum()),
        "executed_gate_fraction":float(mask.mean()),
        "mask_exact_binary":bool(((mask==0)|(mask==1)).all()),
    }


def seed_success(metrics,trained):
    s=CONFIG["scoring"]
    if metrics["alignment"]["mean_cosine"]<s["mean_read_cos_min"]:
        return False
    if metrics["overall_mse"]>s["overall_MSE_max"]:
        return False
    for family in FAMILIES:
        m=metrics["per_family"][family]
        unseen="unseen" in family
        if (m["mse"]>s["unseen_family_MSE_max" if unseen else "seen_family_MSE_max"]
           or m["support_fraction"]<s["unseen_fourpass_support_min" if unseen
                                     else "seen_fourpass_support_min"]
           or not m["mask_exact_binary"]):
            return False
    if trained["arm"]=="random_orthogonal":
        return True
    return bool(trained["first_trainable_gradient_norm"]>0
                and trained["trainable_update_norm"]>0)


def run_arm(seed,arm,*,steps=None,batch_size=None,heldout_boards=None):
    start=time.monotonic()
    bpd,q,trained=train_one(
        seed,arm,steps=steps,batch_size=batch_size
    )
    align=rank4_alignment(bpd,q)
    if heldout_boards is None:
        heldout_boards=CONFIG["source"]["heldout_boards_per_family"]
    families={
        f:evaluate_family(
            bpd,q,seed,arm,f,align["component_to_latent_permutation"],
            heldout_boards
        ) for f in FAMILIES
    }
    measured={
        "alignment":align,
        "overall_mse":sum(m["mse"] for m in families.values())/len(families),
        "physical_weight_delta_mse":float(
            bpd.wrappers["core"].weight_delta().square().mean()
        ),
        "per_family":families
    }
    return {
        **trained,"heldout":measured,
        "success":seed_success(measured,trained),
        "elapsed_seconds":round(time.monotonic()-start,4),
    }


def summarize(arms,*,registered):
    by={(r["seed"],r["arm"]):r for r in arms}
    seeds=sorted({r["seed"] for r in arms})
    if len(arms)!=len(by) or set(by)!={(s,a) for s in seeds for a in ARMS}:
        raise ValueError("missing or duplicate PD09 seed/arm")
    counts={a:sum(by[(s,a)]["success"] for s in seeds) for a in ARMS}
    a=CONFIG["scoring"]["primary_arm"]
    b=CONFIG["scoring"]["primary_control"]
    a_only=sum(by[(s,a)]["success"] and not by[(s,b)]["success"] for s in seeds)
    b_only=sum(by[(s,b)]["success"] and not by[(s,a)]["success"] for s in seeds)
    p=exact_mcnemar_p(a_only,b_only)
    genuine=bool(registered and tuple(seeds)==SEEDS)
    s=CONFIG["scoring"]
    return {
        "study_id":"BPD-PD-09","registered":genuine,
        "seed_count":len(seeds),"arm_count":len(arms),
        "success_by_arm":counts,
        "Wilson95_by_arm":{name:wilson_ci(x,len(seeds)) for name,x in counts.items()},
        "annealed_only":a_only,"fixed_only":b_only,
        "net_annealed_seed_wins":a_only-b_only,
        "McNemar_exact_two_sided_p":p,
        "primary_passed":bool(
            genuine and counts[a]>=s["primary_successes_min"]
            and a_only-b_only>=s["primary_net_wins_min"]
            and p<=s["primary_exact_two_sided_McNemar_max"]
        ),
        "paired_seeds":[{
            "seed":seed,
            "annealed_success":by[(seed,a)]["success"],
            "fixed_success":by[(seed,b)]["success"],
            "annealed_read_cosine":by[(seed,a)]["heldout"]["alignment"]["mean_cosine"],
            "fixed_read_cosine":by[(seed,b)]["heldout"]["alignment"]["mean_cosine"],
            "annealed_unseen_pair":by[(seed,a)]["heldout"]["per_family"]["k2_unseen"]["support_fraction"],
            "fixed_unseen_pair":by[(seed,b)]["heldout"]["per_family"]["k2_unseen"]["support_fraction"],
            "annealed_unseen_triple":by[(seed,a)]["heldout"]["per_family"]["k3_unseen"]["support_fraction"],
            "fixed_unseen_triple":by[(seed,b)]["heldout"]["per_family"]["k3_unseen"]["support_fraction"],
        } for seed in seeds],
    }


def run_registered():
    start=time.monotonic()
    rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(run_arm(seed,arm))
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2)
            if peak>=CONFIG["execution"]["RSS_limit_GiB"]:
                raise MemoryError(f"PD09 peak RSS {peak:.3f} GiB")
        source_shas={r["rotation_sha256"] for r in rows if r["seed"]==seed}
        inits={r["initial_parameters_sha256"] for r in rows if r["seed"]==seed}
        if len(source_shas)!=1 or len(inits)!=1:
            raise AssertionError("PD09 changed seeded source or initial factors across arms")
    return {
        "study_id":CONFIG["study_id"],
        "status":"first_prospective_unlabelled_hyperplane_cohort",
        "source_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                         for f in ("CONFIG.json","PROTOCOL.md","experiment.py")},
        "python_version":platform.python_version(),
        "torch_version":torch.__version__,
        "elapsed_seconds":round(time.monotonic()-start,4),
        "peak_RSS_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
        "summary":summarize(rows,registered=True),
        "arms":rows
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    result=run_registered()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result["summary"],sort_keys=True,indent=2))


if __name__=="__main__":
    main()
