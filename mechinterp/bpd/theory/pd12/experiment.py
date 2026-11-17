"""BPD-PD-12: 8D/16D Givens scaling under correlated supports and weaker margins.

The three optimizer arms use only the observed rotated x and their current
orthogonal read basis. Support labels and Q are used for generating x and
for held-out evaluation, or in the named planted_Q_oracle positive control.
See frozen CONFIG.json and PROTOCOL.md.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
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
from scipy.optimize import linear_sum_assignment

from mechinterp.bpd.components import BPDModel, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.theory.pd02.experiment import exact_mcnemar_p, wilson_ci

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/"CONFIG.json").read_text(encoding="utf-8"))
CELLS=tuple(CONFIG["factorial"]["cells"])
UNITS=tuple((cell,seed) for cell in CELLS for seed in cell["seed_list"])
ARMS=tuple(a["key"] for a in CONFIG["arms"])
ARM_SPEC={a["key"]:a for a in CONFIG["arms"]}
FAMILIES=tuple(CONFIG["source"]["heldout_families"])


class IdentityTeacher(nn.Module):
    def __init__(self,dim:int):
        super().__init__()
        self.core=nn.Linear(dim,dim,bias=False)
        with torch.no_grad():
            self.core.weight.copy_(torch.eye(dim))

    def forward(self,x):
        return self.core(x)


def rotation_for_seed(seed:int,dim:int)->torch.Tensor:
    gen=torch.Generator().manual_seed(CONFIG["source"]["gaussian_qr_seed_offset"]+seed)
    raw=torch.randn(dim,dim,generator=gen,dtype=torch.float64)
    q,r=torch.linalg.qr(raw)
    signs=torch.where(torch.diagonal(r)>=0,1.0,-1.0)
    return (q*signs.unsqueeze(0)).float()


def initial_random_basis(seed:int,dim:int)->torch.Tensor:
    gen=torch.Generator().manual_seed(610000+seed)
    raw=torch.randn(dim,dim,generator=gen,dtype=torch.float64)
    q,r=torch.linalg.qr(raw)
    signs=torch.where(torch.diagonal(r)>=0,1.0,-1.0)
    return q*signs.unsqueeze(0)


@lru_cache(None)
def support_banks(dim:int):
    if dim not in (8,16):
        raise ValueError("PD12 dimension must be 8 or 16")
    withheld={
        1:tuple(),
        2:tuple(sorted({
            tuple(sorted((i,(i+dim//2)%dim)))
            for i in range(dim//2)
        })),
        3:tuple(sorted({
            tuple(sorted(((i+2*j)%dim for j in range(3))))
            for i in range(dim)
        })),
        4:tuple(sorted({
            tuple(sorted(((i+2*j)%dim for j in range(4))))
            for i in range(dim)
        })),
    }
    available={}
    clustered={}
    for k in (1,2,3,4):
        hidden=set(withheld[k])
        possible=tuple(t for t in itertools.combinations(range(dim),k)
                       if t not in hidden)
        if not possible or len(hidden.intersection(possible)):
            raise AssertionError("Source support banks incorrectly overlap")
        available[k]=possible
        if k==1:
            clustered[k]=possible
        else:
            anchors=tuple(tuple(sorted(((4*b+j)%dim for j in range(k))))
                          for b in range(dim//4))
            clustered[k]=tuple(t for t in sorted(set(anchors)) if t in possible)
            if not clustered[k]:
                raise AssertionError("Empty clustered source bank")
        if k>1 and not withheld[k]:
            raise AssertionError("Source must reserve nonempty held-out bank")
    return {"train":available,"heldout":withheld,"cluster":clustered}


def sample_batch(gen:torch.Generator,boards:int,q:torch.Tensor,
                 *,cell:dict,family:str="train"):
    dim=cell["dimension"]
    if q.shape!=(dim,dim):
        raise ValueError("Source rotation does not match study cell")
    src=CONFIG["source"]
    banks=support_banks(dim)
    active=torch.zeros(boards,4,dim,dtype=torch.bool)
    for b in range(boards):
        for pass_idx in range(4):
            if family=="train":
                k=int(torch.randint(1,5,(1,),generator=gen))
                use_held=False
            elif family=="transition_unseen":
                k=pass_idx+1
                use_held=k>1
            elif family.endswith("_seen"):
                k=int(family[1])
                use_held=False
            elif family.endswith("_unseen"):
                k=int(family[1])
                use_held=True
            else:
                raise ValueError(family)
            choices=banks["heldout" if use_held else "train"][k]
            if (not use_held and cell["support_distribution"]=="clustered"
                    and k>1 and float(torch.rand((),generator=gen)) <
                    src["cluster_probability"]):
                choices=banks["cluster"][k]
            index=int(torch.randint(len(choices),(1,),generator=gen))
            active[b,pass_idx,list(choices[index])]=True
    lo,hi=src["amplitude_ranges"][cell["margin"]]
    amplitude=lo+(hi-lo)*torch.rand(boards,4,dim,generator=gen)
    signs=torch.where(torch.randint(0,2,(boards,4,dim),generator=gen).bool(),
                      1.0,-1.0)
    a=amplitude*signs*active.float()
    x=torch.matmul(a,q.T).unsqueeze(2)
    return x,active.unsqueeze(2).float(),a.unsqueeze(2)


def angle_grid(key:str)->torch.Tensor:
    spec=ARM_SPEC[key]
    if spec["kind"]!="search":
        raise ValueError("Oracle has no candidate grid")
    lo,hi=spec["range_deg"]
    return torch.linspace(math.radians(lo),math.radians(hi),
                          spec["angle_grid"],dtype=torch.float64)


def gaussian_mass(samples:torch.Tensor,read:torch.Tensor,sigma:float)->float:
    z=samples.double()@read.double()
    return float(torch.exp(-.5*(z/sigma).square()).mean())


def best_pair_angle(u:torch.Tensor,v:torch.Tensor,
                    angles:torch.Tensor,sigma:float):
    cs,ss=angles.cos()[:,None],angles.sin()[:,None]
    a=cs*u[None,:]+ss*v[None,:]
    b=-ss*u[None,:]+cs*v[None,:]
    score=(torch.exp(-.5*(a/sigma).square()).mean(-1)
           +torch.exp(-.5*(b/sigma).square()).mean(-1))
    high=float(score.max())
    tied=[i for i,v in enumerate(score.tolist()) if high-v<=1e-12]
    best=min(tied,key=lambda idx:(abs(float(angles[idx])),float(angles[idx])))
    return float(angles[best]),high


def search_basis(x:torch.Tensor,initial:torch.Tensor,cell:dict,key:str,
                 *,sweeps:int|None=None):
    if ARM_SPEC[key]["kind"]!="search":
        raise ValueError(key)
    d=cell["dimension"]
    sigma=CONFIG["objective"]["sigma_by_margin"][cell["margin"]]
    steps=CONFIG["objective"]["sweeps"] if sweeps is None else sweeps
    angles=angle_grid(key)
    z=x.reshape(-1,d).double()@initial.double()
    v=initial.clone().double()
    starting=float(torch.exp(-.5*(z/sigma).square()).mean())
    score_path=[starting]
    moves=0
    count=0
    for _ in range(steps):
        for i in range(d):
            for j in range(i+1,d):
                theta,_=best_pair_angle(z[:,i],z[:,j],angles,sigma)
                count+=len(angles)
                if abs(theta)<=1e-12:
                    continue
                c,s=math.cos(theta),math.sin(theta)
                vi,vj=v[:,i].clone(),v[:,j].clone()
                zi,zj=z[:,i].clone(),z[:,j].clone()
                v[:,i]=c*vi+s*vj
                v[:,j]=-s*vi+c*vj
                z[:,i]=c*zi+s*zj
                z[:,j]=-s*zi+c*zj
                moves+=1
        score_path.append(float(torch.exp(-.5*(z/sigma).square()).mean()))
    ortho=float((v.T@v-torch.eye(d,dtype=v.dtype)).abs().max())
    return v,{
        "initial_mass":starting,"final_mass":score_path[-1],
        "gaussian_mass_by_sweep":score_path,
        "candidate_angle_scores":count,
        "candidate_scalar_projection_evaluations":count*z.shape[0],
        "number_of_pair_steps":steps*d*(d-1)//2,
        "nonzero_moves":moves,
        "orthogonality_max_error":ortho,
        "training_observed_vectors":int(z.shape[0]),
    }


def basis_alignment(read:torch.Tensor,true_q:torch.Tensor):
    sim=(read.double().T@true_q.double()).abs().cpu().numpy()
    rows,cols=linear_sum_assignment(-sim)
    if list(rows)!=list(range(read.shape[1])):
        raise AssertionError("Non-canonical assignment row order")
    matched=[float(sim[i,cols[i]]) for i in range(read.shape[1])]
    return {
        "component_to_latent":list(map(int,cols.tolist())),
        "mean_cosine":sum(matched)/len(matched),
        "min_cosine":min(matched),
        "per_component_cosine":matched,
    }


def physical_model(dim:int,read:torch.Tensor):
    torch.manual_seed(0)
    bpd=BPDModel(
        IdentityTeacher(dim),{"core":dim},max_passes=4,
        ablation=AblationConfig(mode="zero",sphere_rescale=False),
        init_svd=False
    )
    with torch.no_grad():
        bpd.wrappers["core"].V.copy_(read.float())
        bpd.wrappers["core"].U.copy_(read.T.float())
    return bpd


@torch.no_grad()
def evaluate_family(bpd,read,q,seed,cell,family,alignment,boards):
    src=CONFIG["source"]
    generator=torch.Generator().manual_seed(
        src["heldout_seed_offset"]+seed+
        src["heldout_family_stride"]*FAMILIES.index(family)
    )
    x,latent,_=sample_batch(generator,boards,q,cell=cell,family=family)
    v=bpd.wrappers["core"].V
    normalized=v/v.square().sum(0).sqrt().clamp_min(1e-6)
    threshold=CONFIG["physical"]["fixed_read_gate_threshold_by_margin"][cell["margin"]]
    mask=(x@normalized).abs().ge(threshold).float()
    expected=latent[...,alignment["component_to_latent"]]
    assert mask.shape==expected.shape
    schedule=PassMaskSchedule(comp_mask=mask,
        delta_mask=torch.zeros_like(mask[...,0]))
    with bpd.masked_components({"core":schedule}):
        y=torch.stack([bpd.target_model(x[:,t]) for t in range(4)],dim=1)
    correct=(mask==expected).all(dim=(-1,-2,-3))
    return {
        "mse":float((y-x).square().mean()),
        "support_fraction":float(correct.float().mean()),
        "correct_boards":int(correct.sum()),
        "boards":int(boards),
        "false_off":int(((mask==0)&(expected==1)).sum()),
        "false_on":int(((mask==1)&(expected==0)).sum()),
        "binary_mask":bool(((mask==0)|(mask==1)).all()),
        "mean_gate_fraction":float(mask.mean()),
        "mean_expected_cardinality":float(expected.sum(-1).float().mean()),
    }


def seed_success(row:dict,arm:str)->bool:
    conf=CONFIG["scoring"]
    measured=row["heldout"]
    if measured["alignment"]["mean_cosine"]<conf["mean_abs_read_cos_min"]:
        return False
    if measured["overall_mse"]>conf["heldout_overall_mse_max"]:
        return False
    for fam in FAMILIES:
        m=measured["per_family"][fam]
        limit=(conf["unseen_family_fourpass_support_min"]
               if "unseen" in fam else conf["seen_family_fourpass_support_min"])
        if m["mse"]>conf["per_family_mse_max"] or m["support_fraction"]<limit or not m["binary_mask"]:
            return False
    if ARM_SPEC[arm]["kind"]=="oracle":
        return True
    return row["training"]["nonzero_moves"]>0 and row["training"]["orthogonality_max_error"]<1e-4


def run_arm(seed:int,cell:dict,arm:str,*,train_boards=None,heldout_boards=None,
            sweeps=None):
    torch.set_num_threads(1)
    start=time.monotonic()
    d=cell["dimension"]
    q=rotation_for_seed(seed,d)
    initial=initial_random_basis(seed,d)
    init_sha=hashlib.sha256(initial.numpy().tobytes()).hexdigest()
    q_sha=hashlib.sha256(q.numpy().tobytes()).hexdigest()
    if ARM_SPEC[arm]["kind"]=="oracle":
        final=q.double()
        stats={
            "initial_mass":None,"final_mass":None,
            "gaussian_mass_by_sweep":[],
            "candidate_angle_scores":0,
            "candidate_scalar_projection_evaluations":0,
            "number_of_pair_steps":0,
            "nonzero_moves":0,
            "orthogonality_max_error":float((final.T@final-torch.eye(d,dtype=final.dtype)).abs().max()),
            "training_observed_vectors":0
        }
    else:
        generator=torch.Generator().manual_seed(CONFIG["source"]["train_seed_offset"]+seed)
        x,_support,_latent=sample_batch(generator,
            CONFIG["source"]["training_boards"] if train_boards is None else train_boards,
            q,cell=cell,family="train")
        final,stats=search_basis(x,initial,cell,arm,sweeps=sweeps)
    bpd=physical_model(d,final)
    alignment=basis_alignment(final,q)
    test_boards=(CONFIG["source"]["eval_boards_per_family"]
                 if heldout_boards is None else heldout_boards)
    per={family:evaluate_family(bpd,final,q,seed,cell,family,alignment,test_boards)
         for family in FAMILIES}
    record={
        "study_id":CONFIG["study_id"],
        "cell":cell["id"],"dimension":d,
        "support_distribution":cell["support_distribution"],
        "margin":cell["margin"],
        "seed":seed,"arm":arm,
        "q_sha256":q_sha,
        "initial_random_basis_sha256":init_sha,
        "uses_true_q_for_optimization":ARM_SPEC[arm]["uses_true_Q_for_training"],
        "training":stats,
        "heldout":{
            "alignment":alignment,
            "overall_mse":sum(m["mse"] for m in per.values())/len(per),
            "per_family":per,
            "physical_weight_delta_mse":float(bpd.wrappers["core"].weight_delta().square().mean()),
        },
        "walltime_seconds":round(time.monotonic()-start,4),
    }
    record["success"]=seed_success(record,arm)
    return record


def summarize(rows:list[dict],*,registered:bool):
    by={(r["seed"],r["arm"]):r for r in rows}
    cells={c["id"]:c for c in CELLS}
    all_seeds=[s for c in CELLS for s in c["seed_list"]]
    if len(rows)!=len(by) or set(by)!={(s,a) for s in all_seeds for a in ARMS}:
        raise ValueError("Missing or duplicate factorial arm")
    global_name,local_name=CONFIG["scoring"]["primary_arm"],CONFIG["scoring"]["paired_control"]
    aonly=sum(by[(s,global_name)]["success"] and not by[(s,local_name)]["success"] for s in all_seeds)
    bonly=sum(by[(s,local_name)]["success"] and not by[(s,global_name)]["success"] for s in all_seeds)
    counts={arm:sum(by[(s,arm)]["success"] for s in all_seeds) for arm in ARMS}
    p=exact_mcnemar_p(aonly,bonly)
    dcounts={}
    cell_summary={}
    for dim in (8,16):
        matching=[s for c in CELLS if c["dimension"]==dim for s in c["seed_list"]]
        dcounts[str(dim)]={a:sum(by[(s,a)]["success"] for s in matching) for a in ARMS}
    for cell in CELLS:
        cell_summary[cell["id"]]={
            "seed_count":len(cell["seed_list"]),
            "success_by_arm":{a:sum(by[(s,a)]["success"] for s in cell["seed_list"]) for a in ARMS},
            "mean_candidate_scores_by_arm":{a:sum(by[(s,a)]["training"]["candidate_angle_scores"] for s in cell["seed_list"])/len(cell["seed_list"]) for a in ARMS},
            "mean_walltime_seconds_by_arm":{a:sum(by[(s,a)]["walltime_seconds"] for s in cell["seed_list"])/len(cell["seed_list"]) for a in ARMS},
        }
    conf=CONFIG["scoring"]
    valid=bool(registered and len(rows)==len(all_seeds)*len(ARMS))
    primary_pass=bool(
        valid
        and counts[global_name]>=conf["primary_global_total_successes_min"]
        and dcounts["8"][global_name]>=conf["primary_global_d8_successes_min"]
        and dcounts["16"][global_name]>=conf["primary_global_d16_successes_min"]
        and aonly-bonly>=conf["primary_net_paired_seed_wins_min"]
        and p<=conf["primary_two_sided_exact_McNemar_p_max"]
    )
    return {
        "study_id":CONFIG["study_id"],"registered":valid,
        "units":len(all_seeds),"arm_count":len(rows),
        "success_by_arm":counts,"success_by_dimension":dcounts,
        "success_by_cell":cell_summary,
        "global_only":aonly,"local_only":bonly,"net_global_wins":aonly-bonly,
        "exact_McNemar_two_sided_p":p,
        "wilson95_pooled":{a:wilson_ci(x,len(all_seeds)) for a,x in counts.items()},
        "wilson95_by_dimension":{d:{a:wilson_ci(n,32) for a,n in c.items()} for d,c in dcounts.items()},
        "primary_passed":primary_pass,
        "per_seed":[{
            "seed":seed,"cell":next(c["id"] for c in CELLS if seed in c["seed_list"]),
            "success_by_arm":{a:by[(seed,a)]["success"] for a in ARMS},
            "global_read_cosine":by[(seed,global_name)]["heldout"]["alignment"]["mean_cosine"],
            "global_min_cosine":by[(seed,global_name)]["heldout"]["alignment"]["min_cosine"],
            "global_ood_k4_support":by[(seed,global_name)]["heldout"]["per_family"]["k4_unseen"]["support_fraction"],
            "local_ood_k4_support":by[(seed,local_name)]["heldout"]["per_family"]["k4_unseen"]["support_fraction"],
        } for seed in all_seeds],
    }


def run_registered():
    start=time.monotonic()
    results=[]
    for cell in CELLS:
        for seed in cell["seed_list"]:
            for arm in ARMS:
                results.append(run_arm(seed,cell,arm))
                peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2)
                if peak>=CONFIG["execution"]["peak_RSS_GiB_limit"]:
                    raise MemoryError(f"PD12 peak RSS {peak:.3f} GiB exceeded registered limit")
            # All four arms get identical Q and initial random read bases.
            for field in ("q_sha256","initial_random_basis_sha256"):
                if len({r[field] for r in results if r["seed"]==seed})!=1:
                    raise AssertionError(f"Factorial seed {seed} lost matched {field}")
    return {
        "study_id":CONFIG["study_id"],
        "status":"original_registered_8D_16D_factorial",
        "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                         for p in ("CONFIG.json","PROTOCOL.md","experiment.py")},
        "python_version":platform.python_version(),
        "torch_version":torch.__version__,
        "scipy_version":__import__("scipy").__version__,
        "elapsed_seconds":round(time.monotonic()-start,4),
        "peak_RSS_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
        "summary":summarize(results,registered=True),
        "arms":results,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    report=run_registered()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report["summary"],indent=2,sort_keys=True))


if __name__=="__main__":
    main()
