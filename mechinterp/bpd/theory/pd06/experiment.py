"""BPD-PD-06: rotated 4D sparse latent factors with compositional holdout.

Source-locked scientific experiment. No training on registered seeds before
CONFIG.json and PROTOCOL.md. Uses actual BPDModel and physical tied nn.Linear.

Inputs to learned gates are dense signed rotated x. Only labelled/oracle arms
receive latent cardinality or support labels. Distinct seed-rotation units
and withheld support combinations are the scientific generalization test.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import resource
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.theory.pd02.experiment import exact_mcnemar_p, wilson_ci

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
SEEDS = tuple(CONFIG["seeds"])
ARMS = tuple(a["key"] for a in CONFIG["arms"])
ARM_CONFIG = {a["key"]: a for a in CONFIG["arms"]}
SCORING = CONFIG["scoring"]
FAMILIES = tuple(CONFIG["source"]["eval_families"])
WIDTH = 4

class IdentityTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Linear(WIDTH, WIDTH, bias=False)
        with torch.no_grad():
            self.core.weight.copy_(torch.eye(WIDTH))

    def forward(self, x):
        return self.core(x)

class GateMLP(nn.Module):
    """The gate sees only signed observed x; no source-family or Q features."""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(WIDTH, 32)
        self.assignment = nn.Linear(32, WIDTH)
        self.count = nn.Linear(32, 3)

    def forward(self, x):
        h = F.gelu(self.hidden(x))
        return self.assignment(h), self.count(h)

def rotation_for_seed(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(900000 + seed)
    sample = torch.randn(WIDTH, WIDTH, generator=g)
    q, r = torch.linalg.qr(sample)
    signs = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0)
    return q * signs.unsqueeze(0)

def build(seed: int, *, frozen_planted: bool):
    torch.manual_seed(seed)
    teacher = IdentityTeacher()
    bpd = BPDModel(
        teacher, {"core": WIDTH}, max_passes=4,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=False,
    )
    gate = GateMLP()
    q = rotation_for_seed(seed)
    factors = bpd.component_parameters()
    if frozen_planted:
        with torch.no_grad():
            bpd.wrappers["core"].V.copy_(q)
            bpd.wrappers["core"].U.copy_(q.T)
        for param in factors:
            param.requires_grad_(False)
    init_data = {
        "V": bpd.wrappers["core"].V.detach().tolist(),
        "U": bpd.wrappers["core"].U.detach().tolist(),
        "gate": {k: v.detach().tolist() for k, v in gate.state_dict().items()},
    }
    init_sha = hashlib.sha256(json.dumps(
        init_data, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    q_sha = hashlib.sha256(q.numpy().tobytes()).hexdigest()
    return bpd, gate, q, init_sha, q_sha

def _supports_for_source(tag: str):
    c = CONFIG["source"]
    mapping = {
        "k1_seen": c["train_support_k1"],
        "k2_seen": c["train_support_k2"],
        "k2_unseen": c["holdout_support_k2"],
        "k3_seen": c["train_support_k3"],
        "k3_unseen": c["holdout_support_k3"],
    }
    if tag not in mapping:
        raise ValueError(tag)
    return mapping[tag]

def sample_batch(
    gen: torch.Generator,
    boards: int,
    q: torch.Tensor,
    *,
    family: str,
):
    """Return dense observed x, latent support and latent a, all B,4,1,4.

    The x-only neural gate sees neither labels nor Q. A caller may use labels
    only in explicitly supervised or oracle arm branches.
    """
    source = CONFIG["source"]
    if family == "train":
        # Uniform k=1,2,3 per pass; subset chosen uniformly within k.
        k = torch.randint(0, 3, (boards, 4), generator=gen)
        chosen = torch.empty((boards, 4, WIDTH), dtype=torch.bool)
        groups = [
            source["train_support_k1"],
            source["train_support_k2"],
            source["train_support_k3"],
        ]
        for b in range(boards):
            for t in range(4):
                opts = groups[int(k[b, t])]
                selection = int(torch.randint(len(opts), (1,), generator=gen))
                chosen[b, t].fill_(False)
                chosen[b, t, opts[selection]] = True
    elif family == "transition_unseen":
        pattern = [
            source["train_support_k1"],
            source["holdout_support_k2"],
            source["holdout_support_k3"],
            source["holdout_support_k2"],
        ]
        chosen = torch.empty((boards, 4, WIDTH), dtype=torch.bool)
        for b in range(boards):
            for t in range(4):
                opts = pattern[t]
                selection = int(torch.randint(len(opts), (1,), generator=gen))
                chosen[b, t].fill_(False)
                chosen[b, t, opts[selection]] = True
    else:
        opts = _supports_for_source(family)
        chosen = torch.empty((boards, 4, WIDTH), dtype=torch.bool)
        for b in range(boards):
            for t in range(4):
                selection = int(torch.randint(len(opts), (1,), generator=gen))
                chosen[b, t].fill_(False)
                chosen[b, t, opts[selection]] = True

    active_min, active_max = source["active_amplitude_uniform"]
    magnitude = active_min + (active_max-active_min) * torch.rand(
        boards, 4, WIDTH, generator=gen,
    )
    signs = torch.where(
        torch.randint(0, 2, (boards,4,WIDTH), generator=gen).bool(),
        1.0, -1.0,
    )
    a = magnitude * signs * chosen.float()
    x = torch.matmul(a, q.T).unsqueeze(2)
    return x, chosen.unsqueeze(2).float(), a.unsqueeze(2)

def rank4_alignment(bpd: BPDModel, q: torch.Tensor) -> dict:
    v = F.normalize(bpd.wrappers["core"].V.detach(), dim=0)
    sim = (v.T @ q).abs()
    choices = [
        (sum(float(sim[i, p[i]]) for i in range(WIDTH)), p)
        for p in itertools.permutations(range(WIDTH))
    ]
    score, mapping = max(choices, key=lambda t: t[0])
    cosines = [float(sim[i, mapping[i]]) for i in range(WIDTH)]
    return {
        "component_to_latent_permutation": list(mapping),
        "mean_cosine": score / WIDTH,
        "min_cosine": min(cosines),
        "per_component_cosines": cosines,
    }

def exact_topk_mask(assignment: torch.Tensor, k: torch.Tensor):
    """Global assignment logits with deterministic lower-index tie break."""
    order = torch.argsort(assignment, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order).scatter_(
        -1, order,
        torch.arange(WIDTH, device=order.device).view(1, 1, 1, WIDTH).expand_as(order),
    )
    return (ranks < k.unsqueeze(-1)).float()

def soft_inclusion(assignment: torch.Tensor, rank: int, temperature: float):
    sorted_scores = assignment.sort(dim=-1, descending=True).values
    mid = .5 * (sorted_scores[..., rank-1:rank] + sorted_scores[..., rank:rank+1])
    return ((assignment-mid) / temperature).sigmoid()

def ste_mask(
    assignment: torch.Tensor, count_logits: torch.Tensor,
    support: torch.Tensor, arm: str,
):
    spec = ARM_CONFIG[arm]
    if spec["oracle_support"]:
        return support
    if arm == "dense_all":
        return torch.ones_like(assignment)

    true_k = support.sum(dim=-1).long()
    selected_k = true_k if spec["oracle_count"] else count_logits.argmax(dim=-1) + 1
    hard = exact_topk_mask(assignment, selected_k)
    temperature = CONFIG["training"]["temperature"]
    smoothed = torch.stack(
        [soft_inclusion(assignment, rank, temperature)
         for rank in (1,2,3)],
        dim=-2,
    )
    if spec["oracle_count"]:
        count_weights = F.one_hot((true_k-1).long(), 3).float()
    else:
        count_weights = count_logits.softmax(dim=-1)
    surrogate = (smoothed * count_weights.unsqueeze(-1)).sum(dim=-2)
    return hard + surrogate - surrogate.detach()

def masked_rollout(bpd, gate, x, support, arm):
    assignment, count_logits = gate(x)
    mask = ste_mask(assignment, count_logits, support, arm)
    if not bool(((mask.detach() == 0) | (mask.detach() == 1)).all()):
        raise AssertionError("PD06 exact-binary mask invariant violated")
    schedule = PassMaskSchedule(
        comp_mask=mask,
        delta_mask=torch.zeros_like(mask[...,0]),
    )
    with bpd.masked_components({"core": schedule}):
        output = torch.stack(
            [bpd.target_model(x[:, t]) for t in range(4)], dim=1
        )
    return output, mask, assignment, count_logits

def objective(bpd, gate, x, support, arm):
    predicted, mask, assignment, count = masked_rollout(
        bpd, gate, x, support, arm
    )
    cfg = CONFIG["training"]
    mse = (predicted - x).square().mean()
    faith = bpd.wrappers["core"].weight_delta().square().mean()
    active = mask.mean()
    total = cfg["coeff_recon"]*mse + cfg["coeff_faithfulness"]*faith
    total = total + cfg["coeff_gate_activity"]*active
    label_count = support.sum(dim=-1).long() - 1
    label_terms = {}
    if ARM_CONFIG[arm]["count_supervision"]:
        ce = F.cross_entropy(count.reshape(-1, 3), label_count.reshape(-1))
        total = total + cfg["coeff_count_supervision"] * ce
        label_terms["count_CE"] = float(ce.detach())
    if ARM_CONFIG[arm]["support_supervision"]:
        bce = F.binary_cross_entropy_with_logits(assignment, support)
        total = total + cfg["coeff_support_supervision"] * bce
        label_terms["support_BCE"] = float(bce.detach())
    return total, label_terms

@torch.no_grad()
def evaluate_family(bpd, gate, q, seed, family, boards):
    gen = torch.Generator().manual_seed(
        CONFIG["source"]["heldout_generator_seed_offset"]+seed
        + 10000 * FAMILIES.index(family)
    )
    x, support, _ = sample_batch(gen, boards, q, family=family)
    return x, support

@torch.no_grad()
def score_family(bpd, gate, x, latent_support, arm, mapping):
    predicted, mask, assignment, count = masked_rollout(
        bpd, gate, x, latent_support, arm
    )
    expected = latent_support[..., list(mapping)]
    per_board = (mask == expected).all(dim=(-1,-2,-3))
    true_k = latent_support.sum(dim=-1).long()
    pred_k = mask.sum(dim=-1).long()
    count_pred = count.argmax(dim=-1)+1
    per_k_accuracy = float((count_pred==true_k).float().mean())
    support_correct = float(per_board.float().mean())
    return {
        "mse":float((predicted-x).square().mean()),
        "exact_support_fraction":support_correct,
        "exact_support_boards":int(per_board.sum()),
        "board_count":int(x.shape[0]),
        "false_off":int(((mask == 0) & (expected == 1)).sum()),
        "false_on":int(((mask == 1) & (expected == 0)).sum()),
        "observed_gate_density":float(mask.float().mean()),
        "mean_executed_cardinality":float(mask.sum(dim=-1).float().mean()),
        "mean_true_cardinality":float(true_k.float().mean()),
        "count_logit_accuracy":per_k_accuracy,
        "executed_count_accuracy":float((pred_k==true_k).float().mean()),
        "execution_binary":bool(((mask==0)|(mask==1)).all()),
    }

def success(measures, *, initial_grad, factor_update, frozen):
    cfg = SCORING
    alignment = measures["alignment"]
    if alignment["mean_cosine"] < cfg["axis_cosine_min"]:
        return False
    if measures["overall_mse"] > cfg["full_recon_MSE_max"]:
        return False
    for family in FAMILIES:
        row = measures["per_family"][family]
        unseen = ("unseen" in family)
        min_frac = cfg[
            "exact_support_unseen_min" if unseen else "exact_support_seen_min"
        ]
        max_mse = cfg[
            "each_unseen_family_MSE_max" if unseen else "each_seen_family_MSE_max"
        ]
        if (row["exact_support_fraction"] < min_frac or row["mse"] > max_mse
                or not row["execution_binary"]):
            return False
    return bool(frozen or (initial_grad > 0 and factor_update > 0))

def run_arm(seed, arm, *, steps=None, batch_size=None, heldout_boards=None):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.set_num_threads(1)
    spec=ARM_CONFIG[arm]
    frozen=spec["frozen"]
    bpd,gate,q,initial_sha,q_sha=build(seed, frozen_planted=frozen)
    factors=bpd.component_parameters()
    initial=[p.detach().clone() for p in factors]
    cfg=CONFIG["training"]
    steps=cfg["steps"] if steps is None else steps
    batch_size=CONFIG["source"]["train_batch"] if batch_size is None else batch_size
    heldout_boards=CONFIG["source"]["heldout_boards_per_family"] if heldout_boards is None else heldout_boards
    rng=torch.Generator().manual_seed(CONFIG["source"]["training_generator_seed_offset"]+seed)
    optimizer=torch.optim.Adam(
        list(gate.parameters())+([] if frozen else factors),
        lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"],
    )
    first_factor_grad=0.0
    first_count_grad=0.0
    first_assignment_grad=0.0
    start=time.monotonic()
    for update in range(steps):
        x,support,_=sample_batch(rng,batch_size,q,family="train")
        optimizer.zero_grad(set_to_none=True)
        total,_=objective(bpd,gate,x,support,arm)
        total.backward()
        if update==0:
            first_factor_grad=math.sqrt(sum(
                float(p.grad.square().sum()) for p in factors if p.grad is not None))
            first_count_grad=math.sqrt(sum(
                float(p.grad.square().sum()) for p in gate.count.parameters() if p.grad is not None))
            first_assignment_grad=math.sqrt(sum(
                float(p.grad.square().sum()) for p in gate.assignment.parameters() if p.grad is not None))
        optimizer.step()
    total_update=math.sqrt(sum(
        float((p.detach()-a).square().sum()) for p,a in zip(factors,initial)))
    alignment=rank4_alignment(bpd,q)
    per_family={}
    with torch.no_grad():
        for family in FAMILIES:
            x,support=evaluate_family(
                bpd,gate,q,seed,family,heldout_boards
            )
            per_family[family]=score_family(
                bpd,gate,x,support,arm,
                alignment["component_to_latent_permutation"]
            )
    measurements={
        "alignment":alignment,
        "overall_mse":sum(v["mse"] for v in per_family.values())/len(FAMILIES),
        "per_family":per_family,
    }
    return {
        "seed":seed,"arm":arm,
        "rotation_sha256":q_sha,
        "initial_parameters_sha256":initial_sha,
        "frozen_planted":frozen,
        "training_steps":steps,
        "factor_first_gradient_norm":first_factor_grad,
        "count_head_first_gradient_norm":first_count_grad,
        "assignment_first_gradient_norm":first_assignment_grad,
        "factor_update_norm":total_update,
        "heldout":measurements,
        "success":success(measurements,
            initial_grad=first_factor_grad,factor_update=total_update,
            frozen=frozen),
        "elapsed_seconds":round(time.monotonic()-start,4)
    }

def cohort_summary(arms, *,registered):
    by={(r["seed"],r["arm"]):r for r in arms}
    seeds=sorted({r["seed"] for r in arms})
    if len(arms)!=len(by) or set(by)!={(s,a) for s in seeds for a in ARMS}:
        raise ValueError("Missing or duplicate registered arm")
    sums={a:sum(by[(s,a)]["success"] for s in seeds) for a in ARMS}
    main=SCORING["primary_arm"]
    other=SCORING["primary_control"]
    vals=[by[(s,main)]["success"] for s in seeds]
    ctr=[by[(s,other)]["success"] for s in seeds]
    aonly=sum(a and not b for a,b in zip(vals,ctr))
    bonly=sum(b and not a for a,b in zip(vals,ctr))
    p=exact_mcnemar_p(aonly,bonly)
    valid=registered and tuple(seeds)==SEEDS
    return {
        "study_id":CONFIG["study_id"],"registered":bool(valid),
        "seeds":len(seeds),"arms":len(arms),
        "success_by_arm":sums,
        "Wilson95_by_arm":{a:wilson_ci(n,len(seeds)) for a,n in sums.items()},
        "supervised_only":aonly,"unsupervised_only":bonly,
        "net_primary_wins":aonly-bonly,
        "McNemar_exact_two_sided_p":p,
        "primary_success":bool(
            valid and sums[main]>=SCORING["primary_min_successes"]
            and aonly-bonly>=SCORING["primary_net_paired_wins_min"]
            and p<=SCORING["primary_mcnemar_two_sided_p_max"]),
        "seed_pairs":[{
            "seed":s,"supervised_success":vals[i],"unsupervised_success":ctr[i],
            "supervised_axis_cosine":by[(s,main)]["heldout"]["alignment"]["mean_cosine"],
            "unsupervised_axis_cosine":by[(s,other)]["heldout"]["alignment"]["mean_cosine"],
            "supervised_ood_pair_support":by[(s,main)]["heldout"]["per_family"]["k2_unseen"]["exact_support_fraction"],
            "unsupervised_ood_pair_support":by[(s,other)]["heldout"]["per_family"]["k2_unseen"]["exact_support_fraction"],
            "supervised_ood_triple_support":by[(s,main)]["heldout"]["per_family"]["k3_unseen"]["exact_support_fraction"],
            "unsupervised_ood_triple_support":by[(s,other)]["heldout"]["per_family"]["k3_unseen"]["exact_support_fraction"],
        } for i,s in enumerate(seeds)]
    }

def run_registered():
    started=time.monotonic()
    rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(run_arm(seed,arm))
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2)
            if rss>=CONFIG["execution"]["RSS_limit_GiB"]:
                raise MemoryError(f"PD06 process peak RSS {rss:.3f} GiB")
        hashes={r["initial_parameters_sha256"] for r in rows
                if r["seed"]==seed and not r["frozen_planted"]}
        rot={r["rotation_sha256"] for r in rows if r["seed"]==seed}
        if len(hashes)!=1 or len(rot)!=1:
            raise AssertionError(f"Seed {seed} initialization/rotation mismatch")
    return {
        "study_id":CONFIG["study_id"],"status":"first_frozen_prospective_32_seed_run",
        "source_hashes":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                         for f in ("CONFIG.json","PROTOCOL.md","experiment.py")},
        "torch_version":torch.__version__,
        "python_version":platform.python_version(),
        "elapsed_seconds":round(time.monotonic()-started,4),
        "peak_process_RSS_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
        "summary":cohort_summary(rows,registered=True),"arms":rows,
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    report=run_registered()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report["summary"],sort_keys=True,indent=2))

if __name__=="__main__":
    main()
