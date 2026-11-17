"""BPD-PD-08: two-stage frozen-factor gate transfer after count-supervised basis fit.

Stage 1 trains a real BPDModel/TiedComponentLinear V,U pair with the
source-locked PD06 count-supervised top-k objective. Stage 2 freezes the
exact same learned V,U bitwise in every matched arm and trains read
thresholds or an independent raw-input gate on identical new train batches.

No stage-2 learned gate is given latent support or count labels. The count
labels in STAGE 1 are a named privileged initialization and remain a
substantive limit on mechanism discovery claims.
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
from torch import nn
import torch.nn.functional as F

from mechinterp.bpd.components import PassMaskSchedule
from mechinterp.bpd.theory.pd03.experiment import ThresholdSTE
from mechinterp.bpd.theory.pd06.experiment import (
    GateMLP, build as build_pd06, objective as pd06_objective,
    rank4_alignment, rotation_for_seed, sample_batch,
    exact_mcnemar_p, wilson_ci,
)

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
SEEDS = tuple(CONFIG["seeds"])
ARMS = tuple(a["key"] for a in CONFIG["arms"])
FAMILIES = tuple(CONFIG["source"]["family_order"])
ARM_INFO = {a["key"]: a for a in CONFIG["arms"]}
WIDTH = CONFIG["source"]["components"] if "components" in CONFIG["source"] else 4


class SecondStageGate(nn.Module):
    """Common seeded initialization of raw MLP plus read-threshold scalars."""

    def __init__(self):
        super().__init__()
        self.raw = GateMLP()
        self.threshold_raw = nn.Parameter(torch.zeros(WIDTH))

    def thresholds(self):
        return .25 + .5 * self.threshold_raw.sigmoid()


def factors_digest(bpd):
    wrapper=bpd.wrappers["core"]
    data = [wrapper.V.detach().tolist(), wrapper.U.detach().tolist()]
    return hashlib.sha256(json.dumps(
        data, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def discover_dictionary(seed, *, steps=None, batch_size=None):
    conf=CONFIG["training"]
    bpd, first_gate, q, _, rotation_sha = build_pd06(
        seed, frozen_planted=False
    )
    factor_params = bpd.component_parameters()
    start_factors = [p.detach().clone() for p in factor_params]
    optimizer = torch.optim.Adam(
        list(first_gate.parameters()) + factor_params,
        lr=conf["stage1_lr"], weight_decay=0.0,
    )
    steps = conf["stage1_updates"] if steps is None else steps
    batch_size = CONFIG["source"]["stage1_batch"] if batch_size is None else batch_size
    rng=torch.Generator().manual_seed(CONFIG["source"]["stage1_rng_offset"]+seed)
    first_grad=0.0
    for i in range(steps):
        x, support, _ = sample_batch(rng, batch_size, q, family="train")
        optimizer.zero_grad(set_to_none=True)
        loss, _ = pd06_objective(
            bpd, first_gate, x, support, "count_supervised"
        )
        loss.backward()
        if i == 0:
            first_grad=math.sqrt(sum(
                float(p.grad.square().sum()) for p in factor_params
                if p.grad is not None
            ))
        optimizer.step()
    update=math.sqrt(sum(
        float((p.detach()-old).square().sum())
        for p,old in zip(factor_params,start_factors)
    ))
    align=rank4_alignment(bpd,q)
    return {
        "source_seed":seed,
        "rotation_sha256":rotation_sha,
        "factor_sha256":factors_digest(bpd),
        "V":bpd.wrappers["core"].V.detach().clone(),
        "U":bpd.wrappers["core"].U.detach().clone(),
        "first_factor_grad_norm":first_grad,
        "factor_update_norm":update,
        "read_alignment":align,
        "all_axes_meet_sufficient_margin":all(
            c > CONFIG["scoring"]["diagnostic_min_cosine_margin"]
            for c in align["per_component_cosines"]
        ),
        "updates":steps,
    }


def instantiate_stage2(seed, learned, *, planted):
    bpd, _, q, _, q_sha=build_pd06(seed, frozen_planted=planted)
    if not planted:
        with torch.no_grad():
            bpd.wrappers["core"].V.copy_(learned["V"])
            bpd.wrappers["core"].U.copy_(learned["U"])
    for p in bpd.component_parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed + 700000)
    gate=SecondStageGate()
    return bpd, gate, q, q_sha


def component_mask(bpd, gate, x, arm, *, support=None, alignment=None):
    if arm == "oracle_support_learned_dictionary":
        if support is None or alignment is None:
            raise ValueError("labelled oracle requires explicit ground-truth alignment")
        return support[...,alignment["component_to_latent_permutation"]]
    if arm in ("read_trained", "read_fixed_threshold", "read_planted_frozen"):
        v=bpd.wrappers["core"].V
        z=(x@v)/v.pow(2).sum(dim=0).sqrt().clamp_min(1e-6)
        probability=torch.sigmoid(10.0*(z.abs()-gate.thresholds()))
        return ThresholdSTE.apply(probability,.5)
    if arm == "raw_independent":
        assignment,_=gate.raw(x)
        return ThresholdSTE.apply(assignment.sigmoid(),.5)
    raise ValueError(arm)


def masked_forward(bpd, gate, x, arm, *, support=None, alignment=None):
    mask=component_mask(
        bpd,gate,x,arm,support=support,alignment=alignment
    )
    if not bool(((mask.detach()==0)|(mask.detach()==1)).all()):
        raise AssertionError("All PD08 physical component writes require hard binary gates")
    sched=PassMaskSchedule(
        comp_mask=mask,
        delta_mask=torch.zeros_like(mask[...,0]),
    )
    with bpd.masked_components({"core":sched}):
        output=torch.stack([
            bpd.target_model(x[:,t]) for t in range(4)
        ],dim=1)
    return output,mask


def stage2_loss(bpd,gate,x,arm):
    if ARM_INFO[arm]["uses_support_labels"]:
        raise ValueError("stage2 labeled oracle has no training objective")
    pred,mask=masked_forward(bpd,gate,x,arm)
    o=CONFIG["training"]
    return ((pred-x).square().mean()*o["stage2_recon_weight"]
            +o["stage2_gate_frac_weight"]*mask.mean())


def gate_init_digest(gate):
    return hashlib.sha256(json.dumps(
        {k:v.detach().tolist() for k,v in gate.state_dict().items()},
        sort_keys=True,separators=(",",":")
    ).encode()).hexdigest()


@torch.no_grad()
def evaluate_family(bpd,gate,x,support,arm,alignment):
    pred,mask=masked_forward(
        bpd,gate,x,arm,support=support,alignment=alignment,
    )
    target=support[...,alignment["component_to_latent_permutation"]]
    exact=(mask==target).all(dim=(-1,-2,-3))
    return {
        "mse":float((pred-x).square().mean()),
        "support_fraction":float(exact.float().mean()),
        "correct_boards":int(exact.sum().item()),
        "boards":int(x.shape[0]),
        "false_off":int(((mask==0)&(target==1)).sum()),
        "false_on":int(((mask==1)&(target==0)).sum()),
        "mask_density":float(mask.mean()),
        "mean_count":float(mask.sum(dim=-1).float().mean()),
        "exact_binary":bool(((mask==0)|(mask==1)).all()),
    }


def qualifies(measures,learned,*,frozen_after):
    s=CONFIG["scoring"]
    if learned["read_alignment"]["mean_cosine"]<s["axis_mean_abs_cos_min"]:
        return False
    if measures["overall_mse"]>s["overall_MSE_max"]:
        return False
    for family in FAMILIES:
        row=measures["per_family"][family]
        is_unseen="unseen" in family
        mse_limit=s["unseen_family_MSE_max" if is_unseen else "seen_family_MSE_max"]
        support_floor=s[
            "unseen_exact_fourpass_support_min" if is_unseen
            else "seen_exact_fourpass_support_min"
        ]
        if (row["mse"]>mse_limit or row["support_fraction"]<support_floor
                or not row["exact_binary"]):
            return False
    return bool(
        frozen_after
        and learned["first_factor_grad_norm"]>0
        and learned["factor_update_norm"]>0
    )


def run_stage2_arm(seed, arm, learned, *, updates=None, batch_size=None,
                   heldout_boards=None):
    if arm not in ARMS:
        raise ValueError(arm)
    start=time.monotonic()
    conf=CONFIG
    planted=ARM_INFO[arm]["planted"]
    bpd,gate,q,q_sha=instantiate_stage2(
        seed,learned,planted=planted
    )
    original_factor_digest=factors_digest(bpd)
    matched_gate_init=gate_init_digest(gate)
    original_physical=[p.detach().clone() for p in bpd.component_parameters()]
    threshold_first_grad=0.0
    raw_first_grad=0.0
    updates=conf["training"]["stage2_updates"] if updates is None else updates
    batch_size=conf["source"]["stage2_batch"] if batch_size is None else batch_size
    heldout_boards=(conf["source"]["heldout_boards_per_family"]
                    if heldout_boards is None else heldout_boards)
    if ARM_INFO[arm]["stage2_updates"]:
        if arm=="read_trained":
            parameters=[gate.threshold_raw]
        elif arm=="raw_independent":
            parameters=list(gate.raw.hidden.parameters())+list(gate.raw.assignment.parameters())
        else:
            raise ValueError(arm)
        optimizer=torch.optim.Adam(
            parameters,lr=conf["training"]["stage2_lr"],weight_decay=0,
        )
        rng=torch.Generator().manual_seed(
            conf["source"]["stage2_rng_offset"]+seed
        )
        for i in range(updates):
            x, _support, _ = sample_batch(rng,batch_size,q,family="train")
            optimizer.zero_grad(set_to_none=True)
            loss=stage2_loss(bpd,gate,x,arm)
            loss.backward()
            if i==0:
                threshold_first_grad=(float(gate.threshold_raw.grad.square().sum().sqrt())
                        if gate.threshold_raw.grad is not None else 0.0)
                raw_first_grad=math.sqrt(sum(
                    float(p.grad.square().sum()) for p in parameters
                    if p.grad is not None
                ))
            optimizer.step()
    frozen_after=all(
        torch.equal(p.detach(),old) for p,old in zip(
            bpd.component_parameters(),original_physical
        )
    )
    if not frozen_after:
        raise AssertionError("PD08 stage2 mutated the frozen physical dictionary")
    alignment=rank4_alignment(bpd,q)
    per={}
    with torch.no_grad():
        for i,family in enumerate(FAMILIES):
            rng=torch.Generator().manual_seed(
                conf["source"]["heldout_rng_offset"]+seed+
                conf["source"]["heldout_family_stride"]*i
            )
            x,support,_=sample_batch(
                rng,heldout_boards,q,family=family
            )
            per[family]=evaluate_family(
                bpd,gate,x,support,arm,alignment
            )
    measurements={
        "alignment":alignment,
        "overall_mse":sum(v["mse"] for v in per.values())/len(FAMILIES),
        "per_family":per,
    }
    # Use each arm's actual factor alignment; the planted positive control
    # explicitly differs from learned factors but has no stage-one claim.
    proxy_learned=learned if not planted else dict(
        learned,read_alignment=alignment
    )
    return {
        "seed":seed,"arm":arm,
        "rotation_sha256":q_sha,
        "stage1_factor_sha256":learned["factor_sha256"],
        "actual_factor_sha256":original_factor_digest,
        "gate_initial_sha256":matched_gate_init,
        "factors_bitwise_frozen":frozen_after,
        "read_threshold_final":gate.thresholds().detach().tolist(),
        "stage2_updates":updates if ARM_INFO[arm]["stage2_updates"] else 0,
        "threshold_first_gradient_norm":threshold_first_grad,
        "raw_first_gradient_norm":raw_first_grad,
        "heldout":measurements,
        "success":qualifies(
            measurements,proxy_learned,frozen_after=frozen_after
        ),
        "elapsed_seconds":round(time.monotonic()-start,4)
    }


def summarize(rows,*,registered):
    by={(r["seed"],r["arm"]):r for r in rows}
    seeds=sorted({r["seed"] for r in rows})
    if len(rows)!=len(by) or set(by)!={(s,a) for s in seeds for a in ARMS}:
        raise ValueError("duplicate or missing PD08 arms")
    counts={a:sum(by[(s,a)]["success"] for s in seeds) for a in ARMS}
    prim=CONFIG["scoring"]["primary_arm"]
    ctrl=CONFIG["scoring"]["control_arm"]
    po=sum(by[(s,prim)]["success"] and not by[(s,ctrl)]["success"] for s in seeds)
    co=sum(by[(s,ctrl)]["success"] and not by[(s,prim)]["success"] for s in seeds)
    p=exact_mcnemar_p(po,co)
    good=bool(registered and tuple(seeds)==SEEDS)
    gates=CONFIG["scoring"]
    return {
        "study_id":CONFIG["study_id"],"registered":good,
        "seeds":len(seeds),"arms":len(rows),
        "success_by_arm":counts,
        "Wilson95_by_arm":{a:wilson_ci(n,len(seeds)) for a,n in counts.items()},
        "read_only":po,"raw_only":co,"net_read_wins":po-co,
        "McNemar_exact_two_sided_p":p,
        "primary_passed":bool(
            good and counts[prim]>=gates["primary_read_successes_min"]
            and po-co>=gates["primary_net_seed_wins_min"]
            and p<=gates["primary_paired_two_sided_mcnemar_p_max"]
        ),
        "paired_seed": [{
            "seed":s,
            "read_success":by[(s,prim)]["success"],
            "raw_success":by[(s,ctrl)]["success"],
            "read_ood_pair_support":by[(s,prim)]["heldout"]["per_family"]["k2_unseen"]["support_fraction"],
            "raw_ood_pair_support":by[(s,ctrl)]["heldout"]["per_family"]["k2_unseen"]["support_fraction"],
            "read_ood_triple_support":by[(s,prim)]["heldout"]["per_family"]["k3_unseen"]["support_fraction"],
            "raw_ood_triple_support":by[(s,ctrl)]["heldout"]["per_family"]["k3_unseen"]["support_fraction"],
            "read_alignment":by[(s,prim)]["heldout"]["alignment"]["mean_cosine"],
            "raw_alignment":by[(s,ctrl)]["heldout"]["alignment"]["mean_cosine"],
        } for s in seeds],
    }


def run_registered():
    start=time.monotonic()
    rows=[]
    learned_rows=[]
    for seed in SEEDS:
        learned=discover_dictionary(seed)
        learned_rows.append({
            key:value for key,value in learned.items()
            if key not in ("V","U")
        })
        for arm in ARMS:
            rows.append(run_stage2_arm(seed,arm,learned))
            used=[r for r in rows if r["seed"]==seed and
                  r["arm"]!="read_planted_frozen"]
            if len({r["actual_factor_sha256"] for r in used})!=1:
                raise AssertionError("Stage2 matched physical factors diverged")
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2)
            if peak>=CONFIG["execution"]["memory_limit_GiB"]:
                raise MemoryError(f"PD08 peak RSS {peak:.3f} GiB exceeded limit")
    return {
        "study_id":CONFIG["study_id"],
        "status":"original_source_locked_two_stage_cohort",
        "source_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                         for f in ("CONFIG.json","PROTOCOL.md","experiment.py")},
        "torch_version":torch.__version__,
        "python_version":platform.python_version(),
        "elapsed_seconds":round(time.monotonic()-start,4),
        "peak_RSS_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
        "summary":summarize(rows,registered=True),
        "first_stage":learned_rows,
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
