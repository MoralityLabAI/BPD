"""BPD-PD-07: factor-read conditional gates and combinatorial support transfer.

The source law, per-seed QR rotations, heldout support combinations and physical
BPDModel wrapper are the frozen BPD-PD-06 source. Only model seeds and the
registered gate algorithms are changed. Label-bearing support is observed in
the named oracle and count-supervised arms only.
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
    build as build_pd06,
    rotation_for_seed, sample_batch, rank4_alignment,
    ste_mask as pd06_ste_mask, exact_mcnemar_p, wilson_ci,
)

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
SEEDS = tuple(CONFIG["seeds"])
ARMS = tuple(a["key"] for a in CONFIG["arms"])
ARM_INFO = {a["key"]: a for a in CONFIG["arms"]}
FAMILIES = tuple(CONFIG["source"]["evaluation_families"])
WIDTH = CONFIG["source"]["physical_dimension"]


class GateChoice(nn.Module):
    """Matched raw gate parameters plus independent scalar read thresholds."""

    def __init__(self, previous_raw_gate):
        super().__init__()
        self.raw = previous_raw_gate
        self.threshold_raw = nn.Parameter(torch.zeros(WIDTH))

    def read_threshold(self):
        return .25 + .5 * self.threshold_raw.sigmoid()

    def forward_raw(self, x):
        return self.raw(x)


def build(seed, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    frozen = not ARM_INFO[arm]["factor_train"]
    bpd, raw_gate, q, old_hash, q_hash = build_pd06(
        seed, frozen_planted=frozen
    )
    gate = GateChoice(raw_gate)
    init = {
        "V": bpd.wrappers["core"].V.detach().tolist(),
        "U": bpd.wrappers["core"].U.detach().tolist(),
        "raw_gate": {k: v.detach().tolist()
                     for k,v in raw_gate.state_dict().items()},
        "threshold_raw": gate.threshold_raw.detach().tolist(),
    }
    digest = hashlib.sha256(json.dumps(
        init, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return bpd, gate, q, digest, q_hash


def gate_read_coefficients(bpd, x, *, detach_features):
    v = bpd.wrappers["core"].V
    if detach_features:
        v = v.detach()
    magnitude = v.pow(2).sum(dim=0).sqrt().clamp_min(1e-6)
    return (x @ v) / magnitude


def mask_and_diagnostics(bpd, gate, x, support, arm):
    spec = ARM_INFO[arm]
    mode = spec["kind"]
    if mode == "oracle":
        # Privileged support labels may enter only the explicitly named oracle arm.
        mask = support
        return mask, {"kind":"oracle"}
    if mode == "raw_count":
        assignment, count_logits = gate.forward_raw(x)
        # PD06 source-locked algorithm uses true count ONLY for training CE,
        # not the unprivileged count-supervised forward.
        mask = pd06_ste_mask(
            assignment, count_logits, support, "count_supervised"
        )
        return mask, {"kind":"raw_count", "count_logits":count_logits}
    if mode == "raw_independent":
        logits, _ = gate.forward_raw(x)
        prob = logits.sigmoid()
        return ThresholdSTE.apply(prob, .5), {
            "kind":mode, "probabilities":prob, "logits":logits,
        }
    if mode == "read":
        z = gate_read_coefficients(
            bpd, x, detach_features=not spec["read_feature_gradient_to_V"]
            if "read_feature_gradient_to_V" in spec else False
        )
        threshold = gate.read_threshold()
        p = torch.sigmoid(10.0 * (z.abs() - threshold))
        mask = ThresholdSTE.apply(p, .5)
        return mask, {
            "kind":mode, "probabilities":p, "read_abs":z.abs(),
            "threshold":threshold,
        }
    raise ValueError(mode)


def masked_forward(bpd, gate, x, support, arm):
    mask, diag = mask_and_diagnostics(bpd, gate, x, support, arm)
    if not bool(((mask.detach()==0)|(mask.detach()==1)).all()):
        raise AssertionError("PD07 physical component masks must be binary")
    schedule = PassMaskSchedule(
        comp_mask=mask,
        delta_mask=torch.zeros_like(mask[...,0]),
    )
    with bpd.masked_components({"core":schedule}):
        y = torch.stack([
            bpd.target_model(x[:,t]) for t in range(4)
        ], dim=1)
    return y, mask, diag


def objective(bpd, gate, x, support, arm):
    y, mask, diag = masked_forward(bpd, gate, x, support, arm)
    conf = CONFIG["train"]
    loss = (
        conf["coef_output_mse"] * (y-x).square().mean()
        + conf["coef_matrix_faith"] * bpd.wrappers["core"].weight_delta().square().mean()
        + conf["coef_gate_frac"] * mask.mean()
    )
    if ARM_INFO[arm]["uses_count_label"]:
        _, count = gate.forward_raw(x)
        target = support.sum(dim=-1).long() - 1
        loss = loss + conf["coef_raw_count_ce"] * F.cross_entropy(
            count.reshape(-1,3), target.reshape(-1)
        )
    return loss


@torch.no_grad()
def evaluate_family(bpd, gate, q, x, support, arm, mapping):
    pred, mask, diag = masked_forward(bpd, gate, x, support, arm)
    expected = support[...,list(mapping)]
    correct = (mask==expected).all(dim=(-1,-2,-3))
    mask_density = float(mask.float().mean())
    data = {
        "mse":float((pred-x).square().mean()),
        "exact_fourpass_support":float(correct.float().mean()),
        "correct_boards":int(correct.sum()),
        "boards":int(x.shape[0]),
        "false_on":int(((mask==1)&(expected==0)).sum()),
        "false_off":int(((mask==0)&(expected==1)).sum()),
        "mask_density":mask_density,
        "mask_binary":bool(((mask==0)|(mask==1)).all()),
        "mean_executed_cardinality":float(mask.sum(dim=-1).float().mean()),
        "mean_true_cardinality":float(expected.sum(dim=-1).float().mean()),
    }
    if "read_abs" in diag:
        zabs = diag["read_abs"]
        data["active_read_magnitude_mean"] = float(
            zabs[expected.bool()].mean()
        )
        data["inactive_read_magnitude_mean"] = float(
            zabs[~expected.bool()].mean()
        ) if (~expected.bool()).any() else None
        data["learned_threshold_mean"] = float(diag["threshold"].mean())
    return data


def is_success(metrics, *, frozen, first_factor_grad, factor_update):
    s = CONFIG["success"]
    if metrics["alignment"]["mean_cosine"] < s["mean_abs_read_cosine_min"]:
        return False
    if metrics["overall_mse"] > s["overall_mse_max"]:
        return False
    for family in FAMILIES:
        row=metrics["per_family"][family]
        unseen = "unseen" in family
        if row["mse"] > s["unseen_family_mse_max" if unseen else "seen_family_mse_max"]:
            return False
        if row["exact_fourpass_support"] < s[
            "unseen_fourpass_exact_support_fraction_min" if unseen else
            "seen_fourpass_exact_support_fraction_min"
        ]:
            return False
        if not row["mask_binary"]:
            return False
    return bool(frozen or (first_factor_grad > 0 and factor_update > 0))


def run_arm(seed, arm, *, steps=None, batch_size=None, heldout_boards=None):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.set_num_threads(1)
    conf=CONFIG["train"]
    source=CONFIG["source"]
    steps=conf["steps"] if steps is None else steps
    batch_size=source["train_batch_size"] if batch_size is None else batch_size
    heldout_boards=source["heldout_boards_per_family"] if heldout_boards is None else heldout_boards
    bpd,gate,q,initial_sha,rotation_sha=build(seed,arm)
    frozen=not ARM_INFO[arm]["factor_train"]
    factors=bpd.component_parameters()
    originals=[p.detach().clone() for p in factors]
    rng=torch.Generator().manual_seed(100000+seed)
    opt=torch.optim.Adam(
        list(gate.parameters())+([] if frozen else factors),
        lr=conf["lr"], weight_decay=conf["weight_decay"]
    )
    first_factor_grad=0.0
    first_threshold_grad=0.0
    start=time.monotonic()
    for k in range(steps):
        x,support,_=sample_batch(rng,batch_size,q,family="train")
        opt.zero_grad(set_to_none=True)
        loss=objective(bpd,gate,x,support,arm)
        loss.backward()
        if k==0:
            first_factor_grad=math.sqrt(sum(
                float(p.grad.square().sum()) for p in factors if p.grad is not None
            ))
            first_threshold_grad=float(
                gate.threshold_raw.grad.square().sum().sqrt()
            ) if gate.threshold_raw.grad is not None else 0.
        opt.step()
    update=math.sqrt(sum(
        float((p.detach()-orig).square().sum())
        for p,orig in zip(factors,originals)
    ))
    alignment=rank4_alignment(bpd,q)
    family_scores={}
    with torch.no_grad():
        for i,family in enumerate(FAMILIES):
            gen=torch.Generator().manual_seed(200000+seed+10000*i)
            x,support,_=sample_batch(gen,heldout_boards,q,family=family)
            family_scores[family]=evaluate_family(
                bpd,gate,q,x,support,arm,
                alignment["component_to_latent_permutation"],
            )
    metrics={
        "alignment":alignment,
        "overall_mse":sum(a["mse"] for a in family_scores.values())/len(FAMILIES),
        "per_family":family_scores,
    }
    return {
        "seed":seed,"arm":arm,
        "rotation_sha256":rotation_sha,"initialization_sha256":initial_sha,
        "frozen_planted":frozen,
        "factor_first_gradient_norm":first_factor_grad,
        "threshold_first_gradient_norm":first_threshold_grad,
        "factor_update_norm":update,
        "training_steps":steps,
        "learned_read_thresholds":gate.read_threshold().detach().tolist(),
        "heldout":metrics,
        "success":is_success(metrics,frozen=frozen,
                             first_factor_grad=first_factor_grad,
                             factor_update=update),
        "elapsed_seconds":round(time.monotonic()-start,4),
    }


def summarize(rows, *, registered):
    by={(r["seed"],r["arm"]):r for r in rows}
    seeds=sorted({r["seed"] for r in rows})
    if len(by)!=len(rows) or set(by)!={(s,a) for s in seeds for a in ARMS}:
        raise ValueError("PD07 rows incomplete or duplicate")
    counts={a:sum(by[(s,a)]["success"] for s in seeds) for a in ARMS}
    primary=CONFIG["success"]["primary_arm"]
    control=CONFIG["success"]["control_arm"]
    po=sum(by[(s,primary)]["success"] and not by[(s,control)]["success"] for s in seeds)
    co=sum(by[(s,control)]["success"] and not by[(s,primary)]["success"] for s in seeds)
    p=exact_mcnemar_p(po,co)
    full=bool(registered and tuple(seeds)==SEEDS)
    sc=CONFIG["success"]
    return {
        "study_id":"BPD-PD-07","registered":full,
        "seed_count":len(seeds),"arm_count":len(rows),
        "success_by_arm":counts,
        "Wilson95_by_arm":{a:wilson_ci(x,len(seeds)) for a,x in counts.items()},
        "primary_only":po,"control_only":co,"net_primary_seed_wins":po-co,
        "McNemar_exact_two_sided_p":p,
        "preregistered_primary_passed":bool(
            full and counts[primary]>=sc["primary_min_successes"]
            and po-co>=sc["primary_net_seed_wins_min"]
            and p<=sc["primary_exact_mcnemar_p_max"]
        ),
        "seed_pairs":[{
            "seed":s,
            "primary_success":by[(s,primary)]["success"],
            "control_success":by[(s,control)]["success"],
            "read_coupled_mse":by[(s,primary)]["heldout"]["overall_mse"],
            "raw_independent_mse":by[(s,control)]["heldout"]["overall_mse"],
            "read_coupled_cos":by[(s,primary)]["heldout"]["alignment"]["mean_cosine"],
            "raw_independent_cos":by[(s,control)]["heldout"]["alignment"]["mean_cosine"],
            "read_coupled_unseen_pair":by[(s,primary)]["heldout"]["per_family"]["k2_unseen"]["exact_fourpass_support"],
            "raw_independent_unseen_pair":by[(s,control)]["heldout"]["per_family"]["k2_unseen"]["exact_fourpass_support"],
            "read_coupled_unseen_triple":by[(s,primary)]["heldout"]["per_family"]["k3_unseen"]["exact_fourpass_support"],
            "raw_independent_unseen_triple":by[(s,control)]["heldout"]["per_family"]["k3_unseen"]["exact_fourpass_support"],
        } for s in seeds],
    }


def run_registered():
    start=time.monotonic()
    rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(run_arm(seed,arm))
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2)
            if peak >= CONFIG["runtime"]["process_RSS_GiB_max"]:
                raise MemoryError(f"PD07 process peak RSS {peak:.3f} GiB")
        fingerprints={r["initialization_sha256"] for r in rows
                      if r["seed"]==seed and not r["frozen_planted"]}
        q={r["rotation_sha256"] for r in rows if r["seed"]==seed}
        if len(fingerprints)!=1 or len(q)!=1:
            raise AssertionError(f"Seed {seed} source or factor initialization changed across arms")
    return {
        "study_id":CONFIG["study_id"],"status":"first_registered_source_locked_cohort",
        "source_sha256":{
            f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
            for f in ("CONFIG.json","PROTOCOL.md","experiment.py")
        },
        "torch_version":torch.__version__,
        "python_version":platform.python_version(),
        "peak_RSS_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
        "elapsed_seconds":round(time.monotonic()-start,4),
        "summary":summarize(rows,registered=True),
        "arms":rows,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    r=run_registered()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(r["summary"],indent=2,sort_keys=True))


if __name__=="__main__":
    main()
