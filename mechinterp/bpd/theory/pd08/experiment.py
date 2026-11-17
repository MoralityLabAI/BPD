"""BPD-PD-08: staged count-supervised factor learning, frozen-gate transfer.

Stage 1 trains shared physical BPD V,U with the PD06 count-supervised top-k
objective, once per seed. Stage 2 freezes those same factor tensors bitwise
and refits label-free read-conditioned or raw-input gates on fresh training
data, retaining the forbidden support family split.
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
import torch.nn.functional as F

from mechinterp.bpd.components import PassMaskSchedule
from mechinterp.bpd.theory.pd03.experiment import ThresholdSTE
from mechinterp.bpd.theory.pd06.experiment import (
    GateMLP, build as build_pd06, sample_batch,
    rank4_alignment, masked_rollout as pd06_rollout,
    objective as pd06_objective, exact_mcnemar_p, wilson_ci,
)
from mechinterp.bpd.theory.pd07.experiment import (
    GateChoice, gate_read_coefficients,
)

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/"CONFIG.json").read_text(encoding="utf-8"))
SEEDS=tuple(CONFIG["seeds"])
ARMS=tuple(a["key"] for a in CONFIG["arms"])
FAMILIES=tuple(CONFIG["source"]["eval_families"])
WIDTH=CONFIG["model"]["width"]


def factor_snapshot(bpd):
    """Bitwise factor receipt and an independently reproducible SHA256."""
    factors=bpd.component_parameters()
    raw=json.dumps([p.detach().tolist() for p in factors],
                   separators=(",", ":")).encode()
    return [p.detach().clone() for p in factors], hashlib.sha256(raw).hexdigest()


def build_and_pretrain(seed, *, steps=None, batch_size=None):
    torch.set_num_threads(1)
    bpd, stage1_gate, q, init_sha, q_sha=build_pd06(
        seed,frozen_planted=False
    )
    factors=bpd.component_parameters()
    before=[p.detach().clone() for p in factors]
    cfg=CONFIG["optimization"]
    steps=cfg["stage1_steps"] if steps is None else steps
    batch_size=CONFIG["source"]["train_batch"] if batch_size is None else batch_size
    optimizer=torch.optim.Adam(
        list(stage1_gate.parameters())+factors,
        lr=cfg["stage1_learning_rate"],weight_decay=cfg["weight_decay"]
    )
    gen=torch.Generator().manual_seed(CONFIG["source"]["stage1_generator_offset"]+seed)
    first_grad=0.0
    for step in range(steps):
        x,support,_=sample_batch(gen,batch_size,q,family="train")
        optimizer.zero_grad(set_to_none=True)
        loss,_=pd06_objective(bpd,stage1_gate,x,support,"count_supervised")
        loss.backward()
        if step==0:
            first_grad=math.sqrt(sum(
                float(p.grad.square().sum()) for p in factors if p.grad is not None
            ))
        optimizer.step()
    update=math.sqrt(sum(float((p.detach()-p0).square().sum())
                         for p,p0 in zip(factors,before)))
    factors_before,sha=factor_snapshot(bpd)
    alignment=rank4_alignment(bpd,q)
    for p in factors:
        p.requires_grad_(False)
        p.grad=None
    return {
        "bpd":bpd, "count_gate":stage1_gate, "q":q,
        "factor_snapshot":factors_before,
        "factor_sha256":sha,
        "rotation_sha256":q_sha,
        "initialization_sha256":init_sha,
        "alignment":alignment,
        "initial_factor_grad":first_grad,
        "factor_update":update,
        "stage1_steps":steps,
    }


def init_stage2_gate(seed):
    torch.manual_seed(CONFIG["model"]["stage2_gate_init_seed_offset"]+seed)
    gate=GateChoice(GateMLP())
    raw=json.dumps({k:v.detach().tolist()
        for k,v in gate.state_dict().items()},
        sort_keys=True,separators=(",",":")).encode()
    return gate,hashlib.sha256(raw).hexdigest()


def stage2_mask(bpd, gate, x, support, arm, *, mapping=None, count_gate=None):
    if arm=="oracle_support":
        if mapping is None:
            raise ValueError("oracle requires explicit planted-component alignment")
        return support[...,list(mapping)]
    if arm=="stage1_count_topk":
        if count_gate is None:
            raise ValueError("missing frozen stage1 gate")
        _,mask,_,_=pd06_rollout(
            bpd,count_gate,x,support,"count_supervised"
        )
        return mask
    if arm in ("read_refit","read_nofit","read_shuffled_refit"):
        z=gate_read_coefficients(bpd,x,detach_features=True).abs()
        if arm=="read_shuffled_refit":
            z=z.roll(shifts=-1,dims=-1)
        p=torch.sigmoid(10.0*(z-gate.read_threshold()))
        return ThresholdSTE.apply(p,.5)
    if arm=="raw_refit":
        logits,_=gate.raw(x)
        return ThresholdSTE.apply(logits.sigmoid(),.5)
    raise ValueError(arm)


def masked_forward(bpd,gate,x,support,arm,*,mapping=None,count_gate=None):
    mask=stage2_mask(bpd,gate,x,support,arm,mapping=mapping,count_gate=count_gate)
    if not bool(((mask.detach()==0)|(mask.detach()==1)).all()):
        raise AssertionError("PD08 stage-two mask is not binary")
    sch=PassMaskSchedule(comp_mask=mask,
                        delta_mask=torch.zeros_like(mask[...,0]))
    with bpd.masked_components({"core":sch}):
        y=torch.stack([bpd.target_model(x[:,t]) for t in range(4)],dim=1)
    return y,mask


def stage2_objective(bpd,gate,x,support,arm):
    if arm not in ("read_refit","raw_refit","read_shuffled_refit"):
        raise ValueError("stage2 objective is gate-only and label-free")
    y,mask=masked_forward(bpd,gate,x,support,arm)
    c=CONFIG["optimization"]
    return c["stage2_loss_mse"]*(y-x).square().mean()+c["stage2_loss_binary_gate_fraction"]*mask.mean()


@torch.no_grad()
def evaluate_one(bpd,gate,count_gate,q,seed,arm,alignment,*,boards):
    per={}
    mapping=alignment["component_to_latent_permutation"]
    for i,family in enumerate(FAMILIES):
        rng=torch.Generator().manual_seed(
            CONFIG["source"]["heldout_generator_offset"]+seed+
            CONFIG["source"]["heldout_family_stride"]*i
        )
        x,support,_=sample_batch(rng,boards,q,family=family)
        y,mask=masked_forward(bpd,gate,x,support,arm,
                              mapping=mapping,count_gate=count_gate)
        expected=support[...,list(mapping)]
        correct=(mask==expected).all(dim=(-1,-2,-3))
        per[family]={
            "mse":float((y-x).square().mean()),
            "exact_fourpass_support":float(correct.float().mean()),
            "correct_boards":int(correct.sum()),
            "boards":int(x.shape[0]),
            "false_off":int(((mask==0)&(expected==1)).sum()),
            "false_on":int(((mask==1)&(expected==0)).sum()),
            "mask_binary":bool(((mask==0)|(mask==1)).all()),
            "gate_fraction":float(mask.float().mean()),
        }
    return {
        "alignment":alignment,
        "overall_mse":sum(row["mse"] for row in per.values())/len(per),
        "per_family":per
    }


def success(m,stage1):
    c=CONFIG["success"]
    if m["alignment"]["mean_cosine"]<c["axis_cos_min"] or m["overall_mse"]>c["overall_mse_max"]:
        return False
    for fam in FAMILIES:
        row=m["per_family"][fam]
        un="unseen" in fam
        if row["mse"]>c["unseen_family_mse_max" if un else "seen_family_mse_max"]:
            return False
        if row["exact_fourpass_support"]<c["unseen_exact_fourpass_support_min" if un
                                            else "seen_exact_fourpass_support_min"]:
            return False
        if not row["mask_binary"]:
            return False
    return bool(stage1["initial_factor_grad"]>0 and stage1["factor_update"]>0)


def run_seed(seed,*,stage1_steps=None,stage2_steps=None,
             batch_size=None,heldout_boards=None):
    if seed is None:
        raise ValueError("explicit seeded experiment required")
    started=time.monotonic()
    x=build_and_pretrain(seed,steps=stage1_steps,batch_size=batch_size)
    bpd=x["bpd"]
    cfg=CONFIG["optimization"]
    stage2_steps=cfg["stage2_steps"] if stage2_steps is None else stage2_steps
    batch_size=CONFIG["source"]["train_batch"] if batch_size is None else batch_size
    boards=CONFIG["source"]["heldout_boards_per_family"] if heldout_boards is None else heldout_boards
    rows=[]
    gate_initial_hashes=[]
    for arm in ARMS:
        gate, gate_sha=init_stage2_gate(seed)
        gate_initial_hashes.append(gate_sha)
        if arm in ("read_refit","raw_refit","read_shuffled_refit"):
            opt=torch.optim.Adam(gate.parameters(),lr=cfg["stage2_learning_rate"],
                                 weight_decay=cfg["weight_decay"])
            rng=torch.Generator().manual_seed(
                CONFIG["source"]["stage2_generator_offset"]+seed
            )
            for step in range(stage2_steps):
                train_x,train_support,_=sample_batch(rng,batch_size,x["q"],family="train")
                opt.zero_grad(set_to_none=True)
                loss=stage2_objective(bpd,gate,train_x,train_support,arm)
                loss.backward()
                opt.step()
        after,sha=factor_snapshot(bpd)
        unchanged=sha==x["factor_sha256"] and all(
            torch.equal(a,b) for a,b in zip(after,x["factor_snapshot"])
        )
        if not unchanged:
            raise AssertionError("stage-two frozen physical V,U changed")
        score=evaluate_one(bpd,gate,x["count_gate"],x["q"],seed,arm,x["alignment"],
                           boards=boards)
        rows.append({
            "seed":seed,"arm":arm,
            "rotation_sha256":x["rotation_sha256"],
            "stage1_initialization_sha256":x["initialization_sha256"],
            "frozen_factor_sha256":sha,
            "stage2_gate_initialization_sha256":gate_sha,
            "factor_bitwise_frozen":unchanged,
            "stage1_first_factor_gradient":x["initial_factor_grad"],
            "stage1_factor_update":x["factor_update"],
            "stage1_steps":x["stage1_steps"],
            "stage2_steps":stage2_steps if arm in
                ("read_refit","raw_refit","read_shuffled_refit") else 0,
            "read_thresholds":gate.read_threshold().detach().tolist(),
            "heldout":score,
            "success":success(score,x),
            "elapsed_seed_seconds":round(time.monotonic()-started,4),
        })
    if len(set(gate_initial_hashes))!=1:
        raise AssertionError("gate initialization differs across stage-two arms")
    return rows


def summary(rows,*,registered):
    by={(r["seed"],r["arm"]):r for r in rows}
    seeds=sorted({r["seed"] for r in rows})
    if len(rows)!=len(by) or set(by)!={(s,a) for s in seeds for a in ARMS}:
        raise ValueError("PD08 missing or duplicate seed-arm rows")
    counts={arm:sum(by[(s,arm)]["success"] for s in seeds) for arm in ARMS}
    primary=CONFIG["success"]["primary_arm"]
    control=CONFIG["success"]["control_arm"]
    one=sum(by[(s,primary)]["success"] and not by[(s,control)]["success"] for s in seeds)
    two=sum(by[(s,control)]["success"] and not by[(s,primary)]["success"] for s in seeds)
    p=exact_mcnemar_p(one,two)
    valid=bool(registered and tuple(seeds)==SEEDS)
    cfg=CONFIG["success"]
    return {
        "study_id":"BPD-PD-08","registered":valid,"seed_count":len(seeds),
        "arm_count":len(rows),"success_by_arm":counts,
        "Wilson95_by_arm":{arm:wilson_ci(n,len(seeds)) for arm,n in counts.items()},
        "primary_only":one,"control_only":two,"net_primary_seed_wins":one-two,
        "McNemar_exact_two_sided_p":p,
        "preregistered_primary_passed":bool(
            valid and counts[primary]>=cfg["primary_min_successes"]
            and one-two>=cfg["primary_net_seed_wins_min"]
            and p<=cfg["primary_two_sided_McNemar_p_max"]
        ),
        "seed_pairs":[{
            "seed":s,"read_refit_success":by[(s,primary)]["success"],
            "raw_refit_success":by[(s,control)]["success"],
            "stage1_read_axis_cosine":by[(s,primary)]["heldout"]["alignment"]["mean_cosine"],
            "read_unseen_pair":by[(s,primary)]["heldout"]["per_family"]["k2_unseen"]["exact_fourpass_support"],
            "raw_unseen_pair":by[(s,control)]["heldout"]["per_family"]["k2_unseen"]["exact_fourpass_support"],
            "read_unseen_triple":by[(s,primary)]["heldout"]["per_family"]["k3_unseen"]["exact_fourpass_support"],
            "raw_unseen_triple":by[(s,control)]["heldout"]["per_family"]["k3_unseen"]["exact_fourpass_support"],
        } for s in seeds]
    }


def run_registered():
    started=time.monotonic()
    rows=[]
    for seed in SEEDS:
        rows.extend(run_seed(seed))
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2)
        if rss>=CONFIG["runtime"]["RSS_GiB_limit"]:
            raise MemoryError("BPD PD08 physical RSS limit exceeded")
    return {
        "study_id":CONFIG["study_id"],"status":"first_source_locked_24_seed_cohort",
        "source_hashes":{
            f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
            for f in ("CONFIG.json","PROTOCOL.md","experiment.py")
        },
        "torch_version":torch.__version__,"python_version":platform.python_version(),
        "elapsed_seconds":round(time.monotonic()-started,4),
        "peak_RSS_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
        "summary":summary(rows,registered=True),"arms":rows,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    r=run_registered()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(r["summary"],sort_keys=True,indent=2))

if __name__=="__main__":
    main()
