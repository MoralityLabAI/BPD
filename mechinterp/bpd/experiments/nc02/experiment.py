"""BPD-NC-02: planted paths through a trained tied nonlinear recurrence.

A positive-control instrument, not a language-model mechanism result. See
PROTOCOL.md for preregistration, seals, independent units and exact gates.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig

PASSES = 4
LANES = 4
STAGES = PASSES + 1
WIDTH = LANES * STAGES
INPUT_DIM = LANES + 1
TOP_K = 8
MIN_CO = 12
MIN_PHI = 0.15
NEIGHBORS = 4
MAX_COMMUNITY = 12
DOSE_TOLERANCE = 0.15
SHAMS = 8
PROPOSALS = 2048
PROTOCOL_COMMIT = "0a8b0c1d676b24dbc657c9709792efb483f768f2"


def sample_x(seed: int, split: int, n: int) -> torch.Tensor:
    if n < 1:
        raise ValueError("n must be positive")
    g = torch.Generator().manual_seed(seed * 100_003 + split * 1_009)
    return torch.randn(n, INPUT_DIM, generator=g)


def external_labels(x: torch.Tensor) -> torch.Tensor:
    """Pure input-defined oracle, computed without looking inside the model."""
    return torch.where(x[:, LANES] > 0, x[:, 0], x[:, 1]) > 0


class PlantedLooper(nn.Module):
    """A trained physical shared matrix with four equal-energy signal lanes."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.mix = nn.Linear(WIDTH, WIDTH, bias=False)
        self.readout = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
        self.acts: list[torch.Tensor] = []
        self.writes: list[torch.Tensor] = []
        self.register_buffer("structural_mask", torch.zeros(WIDTH, WIDTH))
        g = torch.Generator().manual_seed(seed + 123_707)
        self.register_buffer("initial_gains", 1.05 + 0.04 * torch.rand(PASSES, generator=g))
        with torch.no_grad():
            self.mix.weight.zero_()
            for t in range(PASSES):
                for lane in range(LANES):
                    src, dst = t * LANES + lane, (t + 1) * LANES + lane
                    self.structural_mask[dst, src] = 1.0
                    self.mix.weight[dst, src] = self.initial_gains[t]

    @torch.no_grad()
    def project_shared_stage_gains(self) -> None:
        """Exact post-step equality of shift gains across true and decoy lanes."""
        w = self.mix.weight
        gains = []
        for t in range(PASSES):
            vals = torch.stack([w[(t + 1) * LANES + lane, t * LANES + lane]
                                for lane in range(LANES)])
            gains.append(vals.mean().clamp(0.6, 1.8))
        w.zero_()
        for t, gain in enumerate(gains):
            for lane in range(LANES):
                w[(t + 1) * LANES + lane, t * LANES + lane] = gain
        self.readout.clamp_(0.05, 8.0)

    def forward(self, x: torch.Tensor, *, capture: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != INPUT_DIM:
            raise ValueError("x must have four independent signals and one context")
        h = x.new_zeros((len(x), WIDTH))
        h = torch.cat((x[:, :LANES], h[:, LANES:]), dim=1)
        ctx = x[:, LANES] > 0
        logits: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        self.acts = []
        self.writes = []
        for t in range(PASSES):
            before = h
            w = self.mix(h.unsqueeze(1)).squeeze(1)
            if capture:
                self.acts.append(before)
                self.writes.append(w)
            h = torch.tanh(w)
            logits.append(torch.where(
                ctx, self.readout[0] * h[:, (t + 1) * LANES],
                self.readout[1] * h[:, (t + 1) * LANES + 1],
            ))
            states.append(h)
        return torch.stack(logits, dim=1), torch.stack(states, dim=1)

    def structural_receipt(self) -> dict:
        w = self.mix.weight.detach()
        outside = float((w * (1 - self.structural_mask)).abs().max())
        gains = [[float(w[(t + 1) * LANES + l, t * LANES + l]) for l in range(LANES)]
                 for t in range(PASSES)]
        equal_max = max(max(g) - min(g) for g in gains)
        return {"max_absolute_offpath_weight": outside,
                "max_within_stage_gain_difference": equal_max,
                "gains_by_stage_lane": gains,
                "readout_scales": self.readout.detach().tolist(),
                "passed": outside == 0.0 and equal_max <= 1e-7 and min(min(g) for g in gains) > 0.0}


def train_model(seed: int, steps: int = 128) -> tuple[PlantedLooper, dict]:
    torch.manual_seed(seed)
    model = PlantedLooper(seed)
    x = sample_x(seed, 1, 2048)
    sign = torch.where(external_labels(x), 1.0, -1.0)
    g = torch.Generator().manual_seed(seed + 918_771)
    opt = torch.optim.Adam(model.parameters(), lr=0.035)
    for _ in range(steps):
        indices = torch.randint(len(x), (128,), generator=g)
        scores = model(x[indices])[0][:, -1]
        loss = F.softplus(-2 * sign[indices] * scores).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.project_shared_stage_gains()
    model.eval()
    with torch.no_grad():
        final = model(x)[0][:, -1]
        accuracy = float(((final >= 0) == external_labels(x)).float().mean())
    return model, {"seed": seed, "training_steps": steps,
                   "last_minibatch_loss_nats": float(loss.detach()),
                   "training_accuracy": accuracy, "structure": model.structural_receipt()}


def make_basis(w: torch.Tensor, kind: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if w.shape != (WIDTH, WIDTH):
        raise ValueError("expected shared width 20")
    if kind == "native":
        v, u = torch.eye(WIDTH), w.T.contiguous()
    elif kind == "bpd_svd":
        left, singular, vh = torch.linalg.svd(w.double(), full_matrices=False)
        root = singular.sqrt()
        v, u = (vh.T * root.unsqueeze(0)).float(), (root.unsqueeze(1) * left.T).float()
    elif kind == "rotated":
        g = torch.Generator().manual_seed(seed + 121_013)
        q, r = torch.linalg.qr(torch.randn(WIDTH, WIDTH, generator=g, dtype=torch.float64))
        q = q * torch.sign(torch.diag(r)).unsqueeze(0)
        v = q.float()
        u = v.T @ w.T
    else:
        raise ValueError(f"unknown basis: {kind}")
    ratio = (v @ u).T.sub(w).norm() / w.norm().clamp_min(1e-12)
    if float(ratio) > 1e-5:
        raise RuntimeError(f"factor residual {ratio} too large")
    return v, u


def set_basis(bpd: BPDModel, kind: str, seed: int) -> None:
    layer = bpd.wrappers["mix"]
    v, u = make_basis(layer.W_target, kind, seed)
    with torch.no_grad():
        layer.V.copy_(v)
        layer.U.copy_(u)


def mask_schedule(n: int, mask: torch.Tensor) -> PassMaskSchedule:
    if mask.shape != (PASSES, WIDTH):
        raise ValueError("wrong pass/factor mask shape")
    return PassMaskSchedule(
        comp_mask=mask.reshape(1, PASSES, 1, WIDTH).expand(n, -1, -1, -1).contiguous(),
        delta_mask=torch.zeros(n, PASSES, 1),
    )


def binary_kl(clean: torch.Tensor, altered: torch.Tensor) -> torch.Tensor:
    s, a = clean.double(), altered.double()
    p = torch.softmax(torch.stack([-s, s], dim=-1), dim=-1)
    lp = torch.log_softmax(torch.stack([-s, s], dim=-1), dim=-1)
    lq = torch.log_softmax(torch.stack([-a, a], dim=-1), dim=-1)
    return (p * (lp - lq)).sum(dim=-1)


@torch.no_grad()
def parity(bpd: BPDModel, model: PlantedLooper, x: torch.Tensor) -> dict:
    layer = bpd.wrappers["mix"]
    ratio = float(layer.weight_delta().norm() / layer.W_target.norm().clamp_min(1e-12))
    with bpd.target_capture():
        clean, _ = model(x)
    with bpd.masked_components({"mix": mask_schedule(len(x), torch.ones(PASSES, WIDTH))}):
        full, _ = model(x)
    err = float((clean - full).abs().max())
    kl = float(binary_kl(clean, full).clamp_min(0).max())
    return {"passed": ratio <= 1e-4 and err <= 5e-5 and kl <= 1e-7,
            "residual_ratio": ratio, "max_abs_logit": err, "max_kl_nats": kl}


def selections(bpd: BPDModel, model: PlantedLooper, x: torch.Tensor,
               y: torch.Tensor, bs: int = 64) -> torch.Tensor:
    layer = bpd.wrappers["mix"]
    output = []
    for i in range(0, len(x), bs):
        xx = x[i:i+bs].clone().requires_grad_(True)
        yy = y[i:i+bs]
        with bpd.target_capture():
            pred, _ = model(xx, capture=True)
            sign = torch.where(yy, 1.0, -1.0)
            gs = torch.autograd.grad((sign * pred[:, -1]).sum(), model.writes)
        acts = torch.stack(model.acts, dim=1).detach()
        gradients = torch.stack(gs, dim=1).detach()
        z = acts @ layer.V.detach()
        gz = gradients @ layer.U.detach().T
        attr = (z * gz).reshape(len(xx), PASSES * WIDTH)
        top, index = attr.topk(TOP_K, dim=1)
        included = torch.zeros_like(attr)
        included.scatter_(1, index, (top > 0).to(attr.dtype))
        output.append(included)
    return torch.cat(output).double()


def communities(selected: torch.Tensor) -> list[dict]:
    if selected.ndim != 2 or selected.shape[1] != PASSES * WIDTH or len(selected) == 0:
        raise ValueError("invalid attribution selection matrix")
    n = len(selected)
    hits = selected.double()
    f = hits.mean(0)
    both = hits.T @ hits
    cov = both / n - torch.outer(f, f)
    d = torch.sqrt((f * (1-f)).clamp_min(0))
    phi = cov / torch.outer(d,d).clamp_min(1e-12)
    neighbors = []
    for i in range(PASSES * WIDTH):
        js = [j for j in range(PASSES * WIDTH)
              if j != i and both[i,j] >= MIN_CO and phi[i,j] >= MIN_PHI]
        js.sort(key=lambda j:(-float(phi[i,j]),j))
        neighbors.append(js[:NEIGHBORS])
    edges = {i: {} for i in range(PASSES * WIDTH)}
    for i, js in enumerate(neighbors):
        for j in js:
            if i in neighbors[j]:
                edges[i][j] = float(phi[i,j])
                edges[j][i] = float(phi[i,j])
    seen: set[int] = set()
    out = []
    for root in range(PASSES * WIDTH):
        if root in seen or not edges[root]:
            continue
        stack = [root]
        nodes = []
        while stack:
            i = stack.pop()
            if i in seen: continue
            seen.add(i)
            nodes.append(i)
            stack.extend(sorted(edges[i],reverse=True))
        nodes.sort()
        if not 2 <= len(nodes) <= MAX_COMMUNITY:
            continue
        mass = sum(edges[i].get(j, 0.0) for i in nodes for j in nodes if i < j)
        out.append({"nodes": nodes, "size": len(nodes),
                    "passes": sorted({i//WIDTH for i in nodes}),
                    "phi_mass": round(mass,9)})
    out.sort(key=lambda z: (-z["phi_mass"],z["nodes"]))
    return out


def shuffle_null(selected: torch.Tensor, seed: int, n_shuffles: int = 32) -> dict:
    g = torch.Generator().manual_seed(seed + 89_991)
    counts = []
    for _ in range(n_shuffles):
        blocks = []
        for t in range(PASSES):
            perm = torch.randperm(len(selected), generator=g)
            blocks.append(selected[perm, t*WIDTH:(t+1)*WIDTH])
        null = torch.cat(blocks, dim=1)
        counts.append(sum(len(c["passes"]) >= 2 for c in communities(null)))
    return {"cross_pass_community_counts": counts, "n_shuffles": n_shuffles,
            "mean_cross_pass_communities": float(np.mean(counts))}


def true_path(lane: int) -> list[int]:
    if lane not in (0,1):
        raise ValueError("oracle reference paths only lanes 0 and 1")
    return [t*WIDTH + t*LANES + lane for t in range(PASSES)]


def best_native_recovery(groups: list[dict]) -> dict:
    truth = [set(true_path(0)), set(true_path(1))]
    candidate = [set(c["nodes"]) for c in groups[:4]]
    all_f1 = []
    # Explicit one-to-one matching; a selected component cannot explain both paths.
    for gi0 in range(-1,len(candidate)):
        for gi1 in range(-1,len(candidate)):
            if gi0 >= 0 and gi0 == gi1: continue
            scores = []
            for t, index in enumerate((gi0,gi1)):
                c = candidate[index] if index >= 0 else set()
                inter = len(c & truth[t])
                precision = inter / len(c) if c else 0.0
                recall = inter / len(truth[t])
                f1 = 2*precision*recall/(precision+recall) if inter else 0.0
                scores.append({"path_lane":t,"matched_discovery_rank":index,
                               "precision":precision,"recall":recall,"f1":f1,
                               "jaccard":inter/len(c|truth[t]),
                               "exact_match":c==truth[t]})
            all_f1.append(scores)
    best = max(all_f1,key=lambda p:(sum(x["f1"] for x in p),
                                   sum(x["exact_match"] for x in p)))
    return {"paths":best,"mean_f1":float(np.mean([x["f1"] for x in best])),
            "exact_recovered_paths":sum(x["exact_match"] for x in best)}


def node_mask(nodes: list[int]) -> torch.Tensor:
    if len(set(nodes))!=len(nodes) or any(not 0<=i<PASSES*WIDTH for i in nodes):
        raise ValueError("invalid node set")
    m = torch.zeros(PASSES*WIDTH)
    m[nodes] = 1
    return m.view(PASSES,WIDTH)


def _dose_gram(acts: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> np.ndarray:
    z = acts @ v
    gram = torch.einsum("bti,btj->tij",z,z)/len(z)
    gram = gram * (u@u.T).unsqueeze(0)/(PASSES*u.shape[1])
    return gram.detach().double().numpy()


def _dose_from_masks(gram: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0.0,np.einsum("mti,tij,mtj->m",masks,gram,masks,optimize=True)))


def matched_shams(nodes: list[int], gram: np.ndarray, kind: str, seed: int,
                  count: int = SHAMS, proposals: int = PROPOSALS,
                  decoy_only: bool = False) -> dict:
    profile = [sum(i//WIDTH==t for i in nodes) for t in range(PASSES)]
    target = node_mask(nodes).numpy()[None,:,:]
    target_dose = float(_dose_from_masks(gram,target)[0])
    if decoy_only:
        if kind!="native" or any(c>1 for c in profile):
            return {"eligible":False,"reason":"decoy site controls require native one-per-pass paths",
                    "target_dose_rms":target_dose,"shams":[]}
        all_nodes = []
        for lanes in itertools.product((2,3),repeat=sum(profile)):
            it = iter(lanes)
            all_nodes.append([t*WIDTH + t*LANES + next(it)
                              for t in range(PASSES) for _ in range(profile[t])])
    else:
        g = np.random.default_rng(seed)
        seen = {tuple(nodes)}
        all_nodes=[]
        for _ in range(proposals):
            candidate = sorted(
                t*WIDTH + int(j)
                for t,c in enumerate(profile)
                for j in g.choice([t*LANES + l for l in range(LANES)] if kind=="native" else WIDTH,
                                  size=c,replace=False))
            key = tuple(candidate)
            if key in seen: continue
            seen.add(key)
            all_nodes.append(candidate)
    if not all_nodes or target_dose<=1e-10:
        return {"eligible":False,"target_dose_rms":target_dose,"qualified_candidates":0,"shams":[]}
    masks = np.stack([node_mask(n).numpy() for n in all_nodes])
    doses = _dose_from_masks(gram,masks)
    errs = np.abs(doses-target_dose)/target_dose
    qualified = sorted((float(errs[k]),nodeset,float(doses[k]))
                       for k,nodeset in enumerate(all_nodes) if errs[k]<=DOSE_TOLERANCE)
    items=[{"nodes":n,"discovery_dose_rms":dose,"relative_dose_error":e}
           for e,n,dose in qualified[:count]]
    return {"eligible":len(items)==count,"target_dose_rms":target_dose,
            "qualified_candidates":len(qualified),"shams":items}


@torch.no_grad()
def evaluate(bpd: BPDModel, model: PlantedLooper, x: torch.Tensor,
             y: torch.Tensor, nodes: list[int]) -> dict:
    m = node_mask(nodes)
    with bpd.target_capture():
        clean,_=model(x,capture=True)
        clean_w=torch.stack(model.writes,dim=1)
    sign=torch.where(y,1.0,-1.0)
    nll0=F.softplus(-2*sign*clean[:,-1])
    out={"clean_accuracy":float(((clean[:,-1]>=0)==y).float().mean()),"arms":{}}
    for name,mask in (("ablate",1-m),("retain",m)):
        with bpd.masked_components({"mix":mask_schedule(len(x),mask)}):
            trial,_=model(x,capture=True)
            trial_w=torch.stack(model.writes,dim=1)
        k=binary_kl(clean,trial).clamp_min(0)
        nll=F.softplus(-2*sign*trial[:,-1])
        out["arms"][name]={
            "nll_delta_nats":float((nll-nll0).mean()),
            "kl_by_pass_nats":[float(q) for q in k.mean(0)],
            "final_kl_nats":float(k[:,-1].mean()),
            "accuracy":float(((trial[:,-1]>=0)==y).float().mean()),
            "closed_loop_write_change_rms":float((trial_w-clean_w).square().mean().sqrt()),
        }
    return out


def _context_accuracies(model: PlantedLooper, x: torch.Tensor, y: torch.Tensor) -> dict:
    with torch.no_grad():
        scores=model(x)[0][:,-1]
        result={}
        for lane,on in ((0,x[:,LANES]>0),(1,x[:,LANES]<=0)):
            result[str(lane)]=float(((scores[on]>=0)==y[on]).float().mean()) if on.any() else float("nan")
    return result


def run_pilot(seed: int = 200, *, steps: int = 128, discovery_size: int = 1024,
              evaluation_size: int = 1024, basis_names: tuple[str,...] = ("native","bpd_svd","rotated")) -> dict:
    if seed not in range(200,208):
        raise ValueError("only preregistered engineering roots 200..207 permitted")
    torch.set_num_threads(1)
    model,training=train_model(seed,steps)
    xd=sample_x(seed,2,discovery_size)
    xe=sample_x(seed,3,evaluation_size)
    yd,ye=external_labels(xd),external_labels(xe)
    bpd=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                 ablation=AblationConfig(mode="zero",sphere_rescale=False),init_svd=True)
    acc=_context_accuracies(model,xe,ye)
    result={"study":"BPD-NC-02","phase":"ENGINEERING","seed":seed,
            "protocol_commit":PROTOCOL_COMMIT,
            "data":{"training":2048,"discovery":discovery_size,"evaluation":evaluation_size},
            "training":training,"heldout_context_accuracy":acc,"bases":{}}
    for kind in basis_names:
        set_basis(bpd,kind,seed)
        close=parity(bpd,model,xe[:min(64,len(xe))])
        b={"closure":close,"groups":[],"status":"PENDING"}
        result["bases"][kind]=b
        if not close["passed"]:
            b["status"]="BLOCKED_PARITY"
            continue
        selected=selections(bpd,model,xd,yd)
        groups=communities(selected)
        b["n_communities"]=len(groups)
        b["selection_support"] = int((selected.sum(1)>0).sum())
        b["shuffled_graph_null"]=shuffle_null(selected,seed+{"native":0,"bpd_svd":100,"rotated":200}[kind])
        if kind=="native":
            b["oracle_recovery"]=best_native_recovery(groups)
            scales=model.readout.detach().clone()
            with torch.no_grad(): model.readout.zero_()
            try:
                null_sel=selections(bpd,model,xd[:min(256,len(xd))],yd[:min(256,len(xd))])
                b["readout_zero_null_communities"]=len(communities(null_sel))
            finally:
                with torch.no_grad(): model.readout.copy_(scales)
        with torch.no_grad():
            with bpd.target_capture():
                model(xd,capture=True)
                acts=torch.stack(model.acts,dim=1)
        layer=bpd.wrappers["mix"]
        gram=_dose_gram(acts,layer.V.detach(),layer.U.detach())
        for rank,group in enumerate(groups[:4]):
            matched=matched_shams(group["nodes"],gram,kind,seed*10_000+rank+{"native":0,"bpd_svd":100,"rotated":200}[kind])
            item={**group,"primary_sham_matching":matched}
            if matched["eligible"]:
                item["evaluation"]=evaluate(bpd,model,xe,ye,group["nodes"])
                scores=[evaluate(bpd,model,xe,ye,z["nodes"]) for z in matched["shams"]]
                item["sham_evaluations"]=scores
                median_kl=float(np.median([q["arms"]["retain"]["final_kl_nats"] for q in scores]))
                median_delta=float(np.median([q["arms"]["ablate"]["nll_delta_nats"] for q in scores]))
                item["vs_primary_sham"]={
                    "retain_final_kl_advantage_nats":median_kl-item["evaluation"]["arms"]["retain"]["final_kl_nats"],
                    "ablate_nll_advantage_nats":item["evaluation"]["arms"]["ablate"]["nll_delta_nats"]-median_delta,
                }
            else:
                item["status"]="BLOCKED_DOSE_SHAMS"
            if kind=="native":
                decoy=matched_shams(group["nodes"],gram,kind,seed*10_000+rank,decoy_only=True)
                item["oracle_decoy_matching"]=decoy
                if decoy["eligible"]:
                    effects=[evaluate(bpd,model,xe,ye,z["nodes"])["arms"]["ablate"]["nll_delta_nats"]
                             for z in decoy["shams"]]
                    item["oracle_decoy_median_ablation_nll_delta"]=float(np.median(effects))
            b["groups"].append(item)
        b["status"]="ENGINEERING" if b["groups"] and all(
            g["primary_sham_matching"]["eligible"] for g in b["groups"]) else "UNRESOLVED_OR_BLOCKED"
    result["task_passed"]=training["structure"]["passed"] and all(v>0.95 for v in acc.values())
    result["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["protocol_sha256"]=hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest()
    return result


def root_bootstrap(effects: list[float], seed: int = 1701, rounds: int = 4096) -> dict:
    if not effects:
        return {"n_independent_roots":0,"status":"NO_ELIGIBLE_ROOTS"}
    g=np.random.default_rng(seed)
    e=np.asarray(effects,dtype=float)
    indices=g.integers(0,len(e),size=(rounds,len(e)))
    boot=e[indices].mean(axis=1)
    return {"n_independent_roots":len(e),"mean_effect_nats":float(e.mean()),
            "root_bootstrap_95pct_CI":[float(q) for q in np.quantile(boot,[0.025,0.975])],
            "scientific_status":"ENGINEERING_ONLY" if len(e)<64 else "REQUIRES_SEPARATE_CONFIRMATION_PROTOCOL"}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,default=200)
    p.add_argument("--steps",type=int,default=128)
    p.add_argument("--discovery-size",type=int,default=1024)
    p.add_argument("--evaluation-size",type=int,default=1024)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    if args.seed not in range(200,208):
        raise SystemExit("NC-02 confirms no roots: engineering seeds 200–207 only")
    receipt=run_pilot(args.seed,steps=args.steps,discovery_size=args.discovery_size,
                      evaluation_size=args.evaluation_size)
    data=json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if args.out is None:
        print(data)
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as f: f.write(data)


if __name__=="__main__":
    main()
