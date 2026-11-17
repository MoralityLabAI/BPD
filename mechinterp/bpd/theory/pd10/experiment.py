"""BPD-PD-10 registered cold-start smooth-hyperplane-mass experiment.

The only unsupervised train inputs are observed dense x. The true source Q
is used to generate data, to evaluate read alignment, and ONLY in the
explicitly privileged pair-mix initialization stress arm. All output
interventions use the original physical BPDModel shared component wrapper.
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

from mechinterp.bpd.theory.pd06.experiment import (
    build as build_pd06, sample_batch, rank4_alignment,
    exact_mcnemar_p, wilson_ci,
)
from mechinterp.bpd.theory.pd09.experiment import (
    qr_orthogonal, smoothed_l0, set_physical_orthogonal,
    parameter_fingerprint, evaluate_family as pd09_evaluate_family,
    train_one as pd09_train_one,
)

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/"CONFIG.json").read_text(encoding="utf-8"))
SEEDS=tuple(CONFIG["seeds"])
ARMS=tuple(x["key"] for x in CONFIG["arms"])
ARM_CONFIG={x["key"]:x for x in CONFIG["arms"]}
FAMILIES=tuple(CONFIG["generator"]["eval_families"])


def sigma_for_step(arm: str, step: int) -> float:
    config=ARM_CONFIG[arm]
    if "sigma_segments" not in config:
        raise ValueError(f"{arm} has no sigma schedule")
    return float(config["sigma_segments"][min(step//120,2)])


def initial_orthogonal_matrix(bpd, q: torch.Tensor, arm: str) -> torch.Tensor:
    """The privileged control is the *only* initializer allowed to read Q."""
    if arm=="privileged_pairmix015":
        theta=math.radians(ARM_CONFIG[arm]["initial_mix_angle_deg"])
        c,s=math.cos(theta),math.sin(theta)
        pairmix=torch.eye(4)
        i,j=ARM_CONFIG[arm]["mixed_indices"]
        pairmix[i,i]=c
        pairmix[i,j]=-s
        pairmix[j,i]=s
        pairmix[j,j]=c
        return q @ pairmix
    return bpd.wrappers["core"].V.detach().clone()


def make_orthogonal_physical(seed: int, arm: str, *, steps=None, batch_size=None):
    conf=CONFIG["training"]
    steps=conf["steps"] if steps is None else steps
    batch_size=conf["batch_size"] if batch_size is None else batch_size
    bpd,stage_gate,q,initial_sha,q_hash=build_pd06(
        seed, frozen_planted=False
    )
    initial=initial_orthogonal_matrix(bpd,q,arm)
    raw=torch.nn.Parameter(initial.clone())
    optimizer=torch.optim.Adam(
        [raw], lr=conf["lr"],weight_decay=conf["weight_decay"],
    )
    gen=torch.Generator().manual_seed(
        CONFIG["generator"]["train_generator_offset"]+seed
    )
    first_grad=0.0
    for step in range(steps):
        x,_ignored_support,_ignored_latent=sample_batch(
            gen,batch_size,q,family="train"
        )
        optimizer.zero_grad(set_to_none=True)
        # Unlabelled objective receives observed x and trainable read factor
        # only. Source support/count labels never enter loss computation.
        read=qr_orthogonal(raw)
        loss=smoothed_l0(x,read,sigma_for_step(arm,step))
        loss.backward()
        if step==0:
            first_grad=float(raw.grad.square().sum().sqrt())
        optimizer.step()
    updated_raw=float((raw.detach()-initial).square().sum().sqrt())
    final_orthogonal=qr_orthogonal(raw).detach()
    set_physical_orthogonal(bpd,final_orthogonal)
    return bpd,q,{
        "seed":seed,
        "arm":arm,
        "source_rotation_sha256":q_hash,
        "initial_parameter_sha256":initial_sha,
        "starting_raw_basis_sha256":hashlib.sha256(
            json.dumps(initial.tolist(),sort_keys=True).encode()
        ).hexdigest(),
        "starting_from_true_Q":ARM_CONFIG[arm]["uses_true_Q_for_initialization"],
        "used_cardinality_labels":False,
        "orthogonal_tied_factors":True,
        "raw_matrix_parameter_update_norm":updated_raw,
        "first_trainable_gradient_norm":first_grad,
        "physical_factor_sha256":parameter_fingerprint(bpd),
        "optimizer_updates":steps,
        "sigma_values":[sigma_for_step(arm, x) for x in (0,120,240)],
    }


def train_arm(seed: int, arm: str, *, steps=None, batch_size=None):
    if arm not in ARMS:
        raise ValueError(arm)
    if arm=="count_supervised_positive":
        bpd,q,trained=pd09_train_one(
            seed,"count_supervised_positive",
            steps=steps,batch_size=batch_size,
        )
        return bpd,q,{
            **trained,
            "arm":arm,
            "source_rotation_sha256":trained["rotation_sha256"],
            "used_cardinality_labels":True,
            "starting_from_true_Q":False,
            "orthogonal_tied_factors":False,
            "raw_matrix_parameter_update_norm":trained["trainable_update_norm"],
            "optimizer_updates":trained["training_updates"],
            "sigma_values":[],
        }
    return make_orthogonal_physical(
        seed,arm,steps=steps,batch_size=batch_size
    )


def success(measured: dict, trained: dict) -> bool:
    cfg=CONFIG["scoring"]
    if measured["alignment"]["mean_cosine"]<cfg["read_mean_cos_min"]:
        return False
    if measured["overall_mse"]>cfg["total_MSE_max"]:
        return False
    for family in FAMILIES:
        row=measured["per_family"][family]
        unseen="unseen" in family
        if row["mse"]>cfg["unseen_family_MSE_max" if unseen else "seen_family_MSE_max"]:
            return False
        if row["support_fraction"]<cfg["unseen_fourpass_support_min" if unseen else "seen_fourpass_support_min"]:
            return False
        if not row["mask_exact_binary"]:
            return False
    return bool(
        trained["first_trainable_gradient_norm"]>0
        and trained["raw_matrix_parameter_update_norm"]>0
    )


def run_arm(seed: int, arm: str, *, steps=None,batch_size=None,heldout_boards=None):
    torch.set_num_threads(1)
    start=time.monotonic()
    bpd,q,trained=train_arm(seed,arm,steps=steps,batch_size=batch_size)
    alignment=rank4_alignment(bpd,q)
    boards=CONFIG["generator"]["heldout_boards_per_family"] if heldout_boards is None else heldout_boards
    with torch.no_grad():
        families={
            fam:pd09_evaluate_family(
                bpd,q,seed,arm,fam,
                alignment["component_to_latent_permutation"],
                boards
            ) for fam in FAMILIES
        }
    measured={
        "alignment":alignment,
        "overall_mse":sum(row["mse"] for row in families.values())/len(families),
        "physical_weight_delta_mse":float(
            bpd.wrappers["core"].weight_delta().square().mean()
        ),
        "per_family":families,
    }
    return {
        **trained,
        "seed":seed,"arm":arm,
        "heldout":measured,
        "success":success(measured,trained),
        "elapsed_seconds":round(time.monotonic()-start,4)
    }


def summarize(rows,*,registered):
    by={(r["seed"],r["arm"]):r for r in rows}
    seeds=sorted({r["seed"] for r in rows})
    if len(by)!=len(rows) or set(by)!={(seed,arm) for seed in seeds for arm in ARMS}:
        raise ValueError("missing or duplicate PD10 arm results")
    counts={arm:sum(by[(seed,arm)]["success"] for seed in seeds) for arm in ARMS}
    primary=CONFIG["scoring"]["primary_arm"]
    control=CONFIG["scoring"]["primary_control"]
    po=sum(by[(s,primary)]["success"] and not by[(s,control)]["success"] for s in seeds)
    co=sum(by[(s,control)]["success"] and not by[(s,primary)]["success"] for s in seeds)
    p=exact_mcnemar_p(po,co)
    valid=bool(registered and tuple(seeds)==SEEDS)
    c=CONFIG["scoring"]
    return {
        "study_id":"BPD-PD-10",
        "registered":valid,
        "seed_count":len(seeds),
        "arm_count":len(rows),
        "success_by_arm":counts,
        "Wilson95_by_arm":{arm:wilson_ci(n,len(seeds)) for arm,n in counts.items()},
        "cold_only":po,
        "annealed_only":co,
        "net_cold_seed_wins":po-co,
        "McNemar_exact_two_sided_p":p,
        "registered_primary_passed":bool(
            valid and counts[primary]>=c["primary_min_successes"]
            and po-co>=c["primary_net_paired_wins_min"]
            and p<=c["primary_exact_two_sided_McNemar_p_max"]
        ),
        "paired_seed_rows":[{
            "seed":s,
            "cold_success":by[(s,primary)]["success"],
            "annealed_success":by[(s,control)]["success"],
            "cold_read_cosine":by[(s,primary)]["heldout"]["alignment"]["mean_cosine"],
            "annealed_read_cosine":by[(s,control)]["heldout"]["alignment"]["mean_cosine"],
            "cold_min_cosine":by[(s,primary)]["heldout"]["alignment"]["min_cosine"],
            "annealed_min_cosine":by[(s,control)]["heldout"]["alignment"]["min_cosine"],
            "cold_ood_pair":by[(s,primary)]["heldout"]["per_family"]["k2_unseen"]["support_fraction"],
            "annealed_ood_pair":by[(s,control)]["heldout"]["per_family"]["k2_unseen"]["support_fraction"],
        } for s in seeds],
    }


def run_registered():
    start=time.monotonic()
    rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(run_arm(seed,arm))
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2)
            if peak>=CONFIG["runtime"]["process_RSS_limit_GiB"]:
                raise MemoryError(f"PD10 process RSS {peak:.3f} GiB exceeds limit")
        rot={r["source_rotation_sha256"] for r in rows if r["seed"]==seed}
        init={r["initial_parameter_sha256"] for r in rows if r["seed"]==seed}
        raw={r["starting_raw_basis_sha256"] for r in rows if r["seed"]==seed and r["arm"] in
             ("cold015","cold030","warm030_to015","anneal060_030_015")}
        if len(rot)!=1 or len(init)!=1 or len(raw)!=1:
            raise AssertionError(f"PD10 source or matched random initialization changed in seed {seed}")
    return {
        "study_id":CONFIG["study_id"],
        "status":"first_registered_cold_sigma_cohort",
        "source_sha256":{
            name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("CONFIG.json","PROTOCOL.md","experiment.py")
        },
        "python_version":platform.python_version(),
        "torch_version":torch.__version__,
        "elapsed_seconds":round(time.monotonic()-start,4),
        "peak_RSS_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
        "summary":summarize(rows,registered=True),
        "arms":rows,
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
