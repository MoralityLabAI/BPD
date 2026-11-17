"""BPD-PD-13: unlabeled Givens pair-priority and sample-volume study in clustered 16D.

The optimizer accepts *only* observed training x, current read matrix and the
predeclared margin regime. Ground-truth Q/support enter source generation and
the named planted oracle, or the held-out physical BPD evaluator after fitting.

The physical source and evaluator import the frozen PD12 implementations.
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

from mechinterp.bpd.theory.pd12.experiment import (
    rotation_for_seed, initial_random_basis, sample_batch,
    angle_grid, best_pair_angle, physical_model, basis_alignment,
    evaluate_family,
)
from mechinterp.bpd.theory.pd02.experiment import exact_mcnemar_p, wilson_ci

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/"CONFIG.json").read_text(encoding="utf-8"))
CELLS=tuple(CONFIG["seed_cells"])
UNITS=tuple((c,s) for c in CELLS for s in c["seeds"])
ARMS=tuple(a["key"] for a in CONFIG["arms"])
ARM_INFO={a["key"]:a for a in CONFIG["arms"]}
FAMILIES=tuple(CONFIG["source"]["heldout_families"])
DIM=16


def pool_for_seed(seed:int,cell:dict,*,boards:int|None=None):
    """Draw *once* from the unchanged PD12 source; no label enters the search."""
    if cell["dimension"]!=DIM or cell["support_distribution"]!="clustered":
        raise ValueError("PD13 only tests clustered 16D")
    torch.set_num_threads(1)
    gen=torch.Generator().manual_seed(CONFIG["source"]["training_rng_offset"]+seed)
    q=rotation_for_seed(seed,DIM)
    boards=CONFIG["source"]["train_full_boards"] if boards is None else boards
    x,hidden_support,hidden_latent=sample_batch(
        gen,boards,q,cell=cell,family="train"
    )
    pool_hash=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
    return q,x,pool_hash,hidden_support,hidden_latent


def priority_order(projected:torch.Tensor,sigma:float):
    """Visit pairs of the least sparse-looking read columns first.

    The marginal read scores are computed at the *start of each sweep*.
    This routine has no access to the true source rotation or support.
    """
    if projected.ndim!=2 or projected.shape[1]!=DIM:
        raise ValueError("PD13 expects observed projected reads N×16")
    marginal=torch.exp(-.5*(projected/sigma).square()).mean(dim=0)
    masses=[float(x) for x in marginal]
    pairs=[(i,j) for i in range(DIM) for j in range(i+1,DIM)]
    pairs.sort(key=lambda ij:(masses[ij[0]]+masses[ij[1]],ij[0],ij[1]))
    return pairs,masses


def search_basis(observed:torch.Tensor,initial:torch.Tensor,margin:str,arm:str,
                 *,sweeps:int|None=None):
    """Equal candidate budgets, differing only in pair order and sample volume."""
    if ARM_INFO[arm]["kind"]!="search":
        raise ValueError("Only unlabelled search arms can call search_basis")
    if observed.shape[-1]!=DIM or initial.shape!=(DIM,DIM):
        raise ValueError("Wrong PD13 dimension")
    sigma=CONFIG["objective"]["sigma_by_margin"][margin]
    sweeps=CONFIG["objective"]["sweeps"] if sweeps is None else sweeps
    n=observed.reshape(-1,DIM).shape[0]
    z=observed.reshape(-1,DIM).double()@initial.double()
    read=initial.clone().double()
    angles=angle_grid("global37")
    if len(angles)!=CONFIG["objective"]["candidate_grid"]:
        raise AssertionError("Registered 37-angle grid is required")
    initial_score=float(torch.exp(-.5*(z/sigma).square()).mean())
    path=[initial_score]
    total_angles=0
    nonzero=0
    proxy_ops=0
    pair_orders=[]
    pairs_lex=[(i,j) for i in range(DIM) for j in range(i+1,DIM)]
    train_started=time.monotonic()
    for _ in range(sweeps):
        if ARM_INFO[arm]["pair_order"]=="low_mass_first":
            pairs,_marginals=priority_order(z,sigma)
            proxy_ops+=n*DIM
        elif ARM_INFO[arm]["pair_order"]=="lexicographic":
            pairs=pairs_lex
        else:
            raise ValueError("Invalid PD13 pair order")
        pair_orders.append([list(p) for p in pairs])
        for i,j in pairs:
            theta,score=best_pair_angle(z[:,i],z[:,j],angles,sigma)
            total_angles+=len(angles)
            if abs(theta)<=1e-12:
                continue
            c,s=math.cos(theta),math.sin(theta)
            vi,vj=read[:,i].clone(),read[:,j].clone()
            zi,zj=z[:,i].clone(),z[:,j].clone()
            read[:,i]=c*vi+s*vj
            read[:,j]=-s*vi+c*vj
            z[:,i]=c*zi+s*zj
            z[:,j]=-s*zi+c*zj
            nonzero+=1
        path.append(float(torch.exp(-.5*(z/sigma).square()).mean()))
    train_seconds=time.monotonic()-train_started
    ortho=float((read.T@read-torch.eye(DIM,dtype=torch.float64)).abs().max())
    return read,{
        "sigma":sigma,
        "training_observed_vectors":n,
        "training_board_count":int(observed.shape[0]),
        "initial_mass":path[0],
        "final_mass":path[-1],
        "mass_by_sweep":path,
        "nonzero_moves":nonzero,
        "candidate_angle_scores":total_angles,
        "candidate_scalar_projection_evaluations":total_angles*n,
        "priority_proxy_marginal_scalar_evaluations":proxy_ops,
        "pair_orders_by_sweep":pair_orders,
        "pair_step_count":sweeps*DIM*(DIM-1)//2,
        "orthogonality_max_error":ortho,
        "training_seconds":round(train_seconds,6),
        "candidate_angle_scores_per_training_second":
            total_angles/max(train_seconds,1e-12),
    }


def qualifies(row:dict):
    sc=CONFIG["scoring"]
    metrics=row["heldout"]
    if metrics["alignment"]["mean_cosine"]<sc["mean_read_cos_min"]:
        return False
    if metrics["overall_mse"]>sc["overall_MSE_max"]:
        return False
    for fam in FAMILIES:
        m=metrics["per_family"][fam]
        threshold=(sc["unseen_full_fourpass_support_min"]
                   if "unseen" in fam else sc["seen_full_fourpass_support_min"])
        if m["mse"]>sc["per_family_MSE_max"] or m["support_fraction"]<threshold:
            return False
        if not m["binary_mask"]:
            return False
    return row["training"]["nonzero_moves"]>0 if row["arm"]!="planted_Q_oracle" else True


def run_arm(seed:int,cell:dict,arm:str,*,pool:torch.Tensor|None=None,
            q:torch.Tensor|None=None,pool_hash:str|None=None,
            heldout_boards:int|None=None,sweeps:int|None=None):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.set_num_threads(1)
    started=time.monotonic()
    if pool is None:
        q,pool,pool_hash,_,_=pool_for_seed(seed,cell)
    if q is None or pool_hash is None:
        raise ValueError("Require paired source rotation and full-pool hash")
    basis=initial_random_basis(seed,DIM)
    q_hash=hashlib.sha256(q.numpy().tobytes()).hexdigest()
    init_hash=hashlib.sha256(basis.numpy().tobytes()).hexdigest()
    spec=ARM_INFO[arm]
    if spec["kind"]=="oracle":
        read=q.double()
        stats={
            "sigma":None,
            "training_observed_vectors":0,
            "training_board_count":0,
            "initial_mass":None,
            "final_mass":None,
            "mass_by_sweep":[],
            "nonzero_moves":0,
            "candidate_angle_scores":0,
            "candidate_scalar_projection_evaluations":0,
            "priority_proxy_marginal_scalar_evaluations":0,
            "pair_orders_by_sweep":[],
            "pair_step_count":0,
            "orthogonality_max_error":
                float((read.T@read-torch.eye(DIM,dtype=torch.float64)).abs().max()),
            "training_seconds":0,
            "candidate_angle_scores_per_training_second":None,
        }
    else:
        observed=pool[:spec["train_boards"]]
        read,stats=search_basis(observed,basis,cell["margin"],arm,sweeps=sweeps)
    bpd=physical_model(DIM,read)
    aligned=basis_alignment(read,q)
    boards=(CONFIG["source"]["heldout_boards_per_family"]
            if heldout_boards is None else heldout_boards)
    with torch.no_grad():
        per={family:evaluate_family(
            bpd,read,q,seed,cell,family,aligned,boards
        ) for family in FAMILIES}
    record={
        "study_id":"BPD-PD-13","seed":seed,"cell":cell["id"],
        "dimension":DIM,"margin":cell["margin"],"arm":arm,
        "hidden_rotation_sha256":q_hash,
        "initial_random_basis_sha256":init_hash,
        "training_full_pool_sha256":pool_hash,
        "uses_true_Q_for_optimizer":spec["kind"]=="oracle",
        "training":stats,
        "heldout":{
            "alignment":aligned,
            "overall_mse":sum(r["mse"] for r in per.values())/len(per),
            "physical_weight_delta_mse":float(
                bpd.wrappers["core"].weight_delta().square().mean()
            ),
            "per_family":per,
        },
        "walltime_seconds":round(time.monotonic()-started,6),
    }
    record["success"]=qualifies(record)
    return record


def summarize(rows:list[dict],*,registered:bool):
    seeds=[s for cell in CELLS for s in cell["seeds"]]
    by={(r["seed"],r["arm"]):r for r in rows}
    if len(rows)!=len(by) or set(by)!={(s,a) for s in seeds for a in ARMS}:
        raise ValueError("Missing or duplicate PD13 seeds/arms")
    counts={a:sum(by[(s,a)]["success"] for s in seeds) for a in ARMS}
    primary,control=CONFIG["scoring"]["primary_arm"],CONFIG["scoring"]["primary_control"]
    primary_only=sum(by[(s,primary)]["success"] and not by[(s,control)]["success"] for s in seeds)
    control_only=sum(by[(s,control)]["success"] and not by[(s,primary)]["success"] for s in seeds)
    p=exact_mcnemar_p(primary_only,control_only)
    cell_counts={c["id"]:{
        a:sum(by[(s,a)]["success"] for s in c["seeds"]) for a in ARMS
    } for c in CELLS}
    sc=CONFIG["scoring"]
    is_full=registered and len(seeds)==32 and len(rows)==160
    return {
        "study_id":"BPD-PD-13","registered":bool(is_full),
        "units":len(seeds),"arms":len(rows),
        "success_by_arm":counts,"success_by_cell":cell_counts,
        "Wilson95_by_arm":{a:wilson_ci(n,len(seeds)) for a,n in counts.items()},
        "primary_only":primary_only,"control_only":control_only,
        "net_primary_wins":primary_only-control_only,
        "McNemar_exact_two_sided_p":p,
        "registered_primary_passed":bool(
            is_full and counts[primary]>=sc["primary_min_successes"]
            and primary_only-control_only>=sc["primary_net_paired_wins_min"]
            and p<=sc["primary_exact_mcnemar_p_max"]
        ),
        "paired_volume":{
            "priority_2048_only":sum(
                by[(s,"priority2048")]["success"] and not by[(s,"priority256")]["success"]
                for s in seeds
            ),
            "priority_256_only":sum(
                by[(s,"priority256")]["success"] and not by[(s,"priority2048")]["success"]
                for s in seeds
            ),
            "lex_2048_only":sum(
                by[(s,"lex2048")]["success"] and not by[(s,"lex256")]["success"]
                for s in seeds
            ),
            "lex_256_only":sum(
                by[(s,"lex256")]["success"] and not by[(s,"lex2048")]["success"]
                for s in seeds
            ),
        },
        "mean_training_seconds_by_arm":{
            a:sum(by[(s,a)]["training"]["training_seconds"] for s in seeds)/len(seeds)
            for a in ARMS
        },
        "per_seed":[{
            "seed":s,
            "cell":next(c["id"] for c in CELLS if s in c["seeds"]),
            "success_by_arm":{a:by[(s,a)]["success"] for a in ARMS},
            "priority_2048_mean_cos":by[(s,primary)]["heldout"]["alignment"]["mean_cosine"],
            "lex_2048_mean_cos":by[(s,control)]["heldout"]["alignment"]["mean_cosine"],
            "priority_2048_unseen_k4":by[(s,primary)]["heldout"]["per_family"]["k4_unseen"]["support_fraction"],
            "lex_2048_unseen_k4":by[(s,control)]["heldout"]["per_family"]["k4_unseen"]["support_fraction"],
        } for s in seeds],
    }


def run_registered():
    started=time.monotonic()
    records=[]
    for cell in CELLS:
        for seed in cell["seeds"]:
            # The sole per-seed source pool is shared by all five arms.
            q,pool,sha,_,_=pool_for_seed(seed,cell)
            for arm in ARMS:
                record=run_arm(seed,cell,arm,pool=pool,q=q,pool_hash=sha)
                records.append(record)
                peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
                if peak>=CONFIG["execution"]["process_RSS_GiB_max"]:
                    raise MemoryError(f"PD13 RSS {peak:.3f} GiB exceeds 3 GiB")
            rs=[r for r in records if r["seed"]==seed]
            for k in ("training_full_pool_sha256","hidden_rotation_sha256",
                      "initial_random_basis_sha256"):
                if len({r[k] for r in rs})!=1:
                    raise AssertionError(f"Seed {seed} arm mismatch in {k}")
    return {
        "study_id":CONFIG["study_id"],
        "status":"first_registered_clustered_16d_priority_and_volume",
        "source_hashes":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                         for f in ("CONFIG.json","PROTOCOL.md","experiment.py")},
        "torch_version":torch.__version__,
        "python_version":platform.python_version(),
        "scipy_version":__import__("scipy").__version__,
        "elapsed_seconds":round(time.monotonic()-started,6),
        "peak_RSS_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        "summary":summarize(records,registered=True),
        "arms":records,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=run_registered()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result["summary"],indent=2,sort_keys=True))


if __name__=="__main__":
    main()
