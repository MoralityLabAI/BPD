"""BPD-PD-16: source-locked norm-weight shuffle placebo against PD15.

Only observed dense x, QR basis and a source-independent seeded permutation
enter the optimizers. Hidden support/Q is used in the source generator,
privileged oracle and strictly post-fit physical/diagnostic evaluation.
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

from mechinterp.bpd.theory.pd02.experiment import exact_mcnemar_p, wilson_ci
from mechinterp.bpd.theory.pd12.experiment import (
    angle_grid, best_pair_angle, basis_alignment, evaluate_family,
    initial_random_basis, physical_model
)
from mechinterp.bpd.theory.pd15.experiment import (
    source_for_seed, fit_basis as pd15_fit_basis,
    weights_from_observed, best_pair_angle_weighted, weighted_mass,
    effective_sample_size
)

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
CELL = CONFIG["seed_cell"]
SEEDS = tuple(CELL["seeds"])
ARMS = tuple(x["key"] for x in CONFIG["arms"])
SPEC = {x["key"]: x for x in CONFIG["arms"]}
FAMILIES = tuple(CONFIG["source"]["heldout_families"])
DIM = 16
PAIRS = tuple((i, j) for i in range(DIM) for j in range(i+1, DIM))
ANGLES = angle_grid("global37")
assert len(PAIRS) == 120 and len(ANGLES) == 37


def shuffled_weights(observed: torch.Tensor, seed: int):
    """One fixed random weight-to-example reassignment; no hidden labels."""
    w = weights_from_observed(observed, 1.)
    gen = torch.Generator().manual_seed(910000 + seed)
    perm = torch.randperm(w.shape[0], generator=gen)
    shuffled = w[perm]
    perm_sha = hashlib.sha256(perm.numpy().tobytes()).hexdigest()
    if not torch.equal(torch.sort(w).values, torch.sort(shuffled).values):
        raise AssertionError("Shuffle changed the weight multiset")
    return shuffled, perm_sha


def fit_shuffled(observed: torch.Tensor, initial: torch.Tensor,
                 shuffled: torch.Tensor, *, sweeps: int | None = None):
    """PD15 identical optimizer equations, swapping only weight assignment."""
    if observed.ndim != 4 or observed.shape[-3:] != (4, 1, DIM):
        raise ValueError("Expected dense [boards,4,1,16] source")
    if initial.shape != (DIM,DIM):
        raise ValueError("Unregistered QR dimensions")
    sweeps = CONFIG["objective"]["sweeps"] if sweeps is None else sweeps
    if sweeps < 1:
        raise ValueError("Invalid source-locked sweep count")
    sigma = CONFIG["objective"]["sigma"]
    flat = observed.reshape(-1, DIM).double()
    w = shuffled.double()
    if w.shape != (flat.shape[0],):
        raise ValueError("Shuffled weights require one value per observed vector")
    z = flat @ initial.double()
    read = initial.clone().double()
    initial_mass = weighted_mass(z, sigma, w)
    mass_path = [initial_mass]
    candidates, moves = 0, 0
    started = time.monotonic()
    for _ in range(sweeps):
        for i,j in PAIRS:
            theta, _ = best_pair_angle_weighted(
                z[:, i], z[:, j], ANGLES, sigma, w
            )
            candidates += len(ANGLES)
            if abs(theta) <= 1e-12:
                continue
            c, s = math.cos(theta), math.sin(theta)
            vi, vj = read[:,i].clone(), read[:,j].clone()
            zi, zj = z[:,i].clone(), z[:,j].clone()
            read[:,i] = c * vi + s * vj
            read[:,j] = -s * vi + c * vj
            z[:,i] = c * zi + s * zj
            z[:,j] = -s * zi + c * zj
            moves += 1
        mass_path.append(weighted_mass(z,sigma,w))
        if mass_path[-1] < mass_path[-2] - 1e-9:
            raise AssertionError("Shuffled Gaussian search lost objective mass")
    error = float((read.T @ read - torch.eye(DIM,dtype=torch.float64)).abs().max())
    stats = {
        "sigma": sigma, "lambda": 1., "weight_mode": "shuffled",
        "training_board_count": int(observed.shape[0]),
        "training_observed_vectors": int(flat.shape[0]),
        "candidate_angle_scores": candidates,
        "candidate_scalar_projection_evaluations": candidates * int(flat.shape[0]),
        "training_weight_effective_sample_size": effective_sample_size(w),
        "training_weight_effective_sample_fraction": effective_sample_size(w)/len(w),
        "training_weight_mean": float(w.mean()),
        "training_weight_min": float(w.min()),
        "training_weight_max": float(w.max()),
        "mass_by_sweep": mass_path, "initial_mass": initial_mass,
        "final_mass": mass_path[-1],
        "nonzero_moves": moves,
        "orthogonality_max_error": error,
        "training_seconds": round(time.monotonic()-started,6),
        "pair_sweeps": sweeps,
        "extra_weighted_observation_reductions": candidates * int(flat.shape[0]),
    }
    assert candidates == sweeps*120*37 and error<1e-4
    return read, stats


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    ac,bc = a-a.mean(),b-b.mean()
    denom=(ac.square().sum()*bc.square().sum()).sqrt()
    if denom == 0:
        raise ValueError("Correlation denominator zero")
    return float((ac*bc).sum()/denom)


def diagnose_weights(seed: int, observed: torch.Tensor,
                     source_support: torch.Tensor):
    """Post-fit diagnostic only; the support is forbidden inside optimizers."""
    k=source_support.reshape(-1,DIM).sum(-1).double()
    aligned=weights_from_observed(observed,1.)
    shuffled,perm_sha=shuffled_weights(observed,seed)
    ess_a,ess_s = effective_sample_size(aligned),effective_sample_size(shuffled)
    corr_a,corr_s=correlation(aligned,k),correlation(shuffled,k)
    sc=CONFIG["scoring"]
    passed=bool(
        corr_a<=sc["aligned_weight_k_corr_max"] and
        abs(corr_s)<=sc["shuffled_weight_k_abs_corr_max"] and
        abs(ess_a-ess_s)<=sc["ESS_abs_diff_max"]
    )
    return {
        "seed": seed, "source_cardinality_count": len(k),
        "aligned_weight_k_pearson":corr_a,
        "shuffled_weight_k_pearson":corr_s,
        "aligned_ESS":ess_a,"shuffled_ESS":ess_s,
        "ESS_abs_diff":abs(ess_a-ess_s),
        "weight_multiset_exact":torch.equal(
            torch.sort(aligned).values,torch.sort(shuffled).values
        ),
        "permutation_sha256":perm_sha,
        "privileged_support_used_only_for_postfit_diagnostic":True,
        "passed":passed,
    }


def qualifies(row: dict) -> bool:
    sc=CONFIG["scoring"]
    held=row["heldout"]
    if held["alignment"]["mean_cosine"]<sc["mean_read_cos_min"]:
        return False
    if held["overall_mse"]>sc["overall_MSE_max"]:
        return False
    for fam in FAMILIES:
        f=held["per_family"][fam]
        floor=(sc["unseen_full_fourpass_support_min"] if "unseen" in fam
               else sc["seen_full_fourpass_support_min"])
        if f["mse"]>sc["per_family_MSE_max"] or f["support_fraction"]<floor:
            return False
        if not f["binary_mask"]:
            return False
    if row["arm"]=="planted_Q_oracle":
        return True
    return (row["training"]["nonzero_moves"]>0 and
            row["training"]["orthogonality_max_error"]<1e-4)


def run_arm(seed:int, arm:str, *, q:torch.Tensor, pool:torch.Tensor,
            pool_sha:str, heldout_boards:int|None=None,
            sweeps:int|None=None) -> dict:
    torch.set_num_threads(1)
    started=time.monotonic()
    initial=initial_random_basis(seed,DIM)
    kind=SPEC[arm]["kind"]
    if kind=="oracle":
        read=q.double()
        training={
            "candidate_angle_scores":0,
            "candidate_scalar_projection_evaluations":0,
            "nonzero_moves":0,
            "training_board_count":0,
            "orthogonality_max_error":float(
                (read.T@read-torch.eye(DIM,dtype=read.dtype)).abs().max()
            )
        }
    elif arm=="norm_shuffled":
        weights,perm_sha=shuffled_weights(pool,seed)
        read,training=fit_shuffled(pool,initial,weights,sweeps=sweeps)
        training["permutation_sha256"]=perm_sha
    else:
        lam=1. if arm=="norm_aligned" else 0.
        read,training=pd15_fit_basis(
            pool,initial,CONFIG["objective"]["sigma"],lam,sweeps=sweeps
        )
        training["weight_mode"]=SPEC[arm]["weight_mode"]
    bpd=physical_model(DIM,read)
    alignment=basis_alignment(read,q)
    n=(CONFIG["source"]["heldout_boards_per_family"]
       if heldout_boards is None else heldout_boards)
    with torch.no_grad():
        per={fam:evaluate_family(
            bpd,read,q,seed,CELL,fam,alignment,n
        ) for fam in FAMILIES}
    result={
        "study_id":"BPD-PD-16","seed":seed,
        "cell":CELL["id"],"arm":arm,
        "hidden_rotation_sha256":hashlib.sha256(q.numpy().tobytes()).hexdigest(),
        "initial_random_basis_sha256":hashlib.sha256(
            initial.numpy().tobytes()).hexdigest(),
        "training_full_pool_sha256":pool_sha,
        "uses_true_Q_for_optimizer":kind=="oracle",
        "training":training,
        "heldout":{
            "alignment":alignment,
            "overall_mse":sum(x["mse"] for x in per.values())/len(per),
            "physical_weight_delta_mse":float(
                bpd.wrappers["core"].weight_delta().square().mean()
            ),
            "per_family":per,
        },
        "walltime_seconds":round(time.monotonic()-started,6),
    }
    result["success"]=qualifies(result)
    return result


def summarize(rows:list[dict],diagnostics:list[dict],*,registered:bool) -> dict:
    by={(r["seed"],r["arm"]):r for r in rows}
    dby={d["seed"]:d for d in diagnostics}
    if (len(rows)!=len(by) or
        set(by)!={(seed,arm) for seed in SEEDS for arm in ARMS} or
        len(diagnostics)!=len(dby) or set(dby)!=set(SEEDS)):
        raise ValueError("Missing/duplicate PD16 seeded arms or diagnostics")
    sc=CONFIG["scoring"]
    a,b=sc["primary_arm"],sc["primary_control"]
    aonly=sum(by[(s,a)]["success"] and not by[(s,b)]["success"] for s in SEEDS)
    bonly=sum(by[(s,b)]["success"] and not by[(s,a)]["success"] for s in SEEDS)
    p=exact_mcnemar_p(aonly,bonly)
    counts={arm:sum(by[(s,arm)]["success"] for s in SEEDS) for arm in ARMS}
    ngate=sum(dby[s]["passed"] for s in SEEDS)
    full=bool(registered and len(rows)==128 and len(diagnostics)==32)
    mechanism=bool(full and counts[a]>=sc["primary_min_successes"]
                   and aonly-bonly>=sc["primary_net_paired_wins_min"]
                   and p<=sc["primary_exact_mcnemar_p_max"])
    diagnostic=bool(full and ngate>=sc["diagnostic_min_passing_seeds"])
    return {
        "study_id":"BPD-PD-16","registered":full,
        "units":len(SEEDS),"arms":len(rows),
        "success_by_arm":counts,
        "Wilson95_by_arm":{
            arm:wilson_ci(counts[arm],len(SEEDS)) for arm in ARMS
        },
        "primary_only":aonly,"control_only":bonly,
        "net_primary_wins":aonly-bonly,
        "McNemar_exact_two_sided_p":p,
        "primary_mechanism_gate_passed":mechanism,
        "norm_cardinality_diagnostic_passes":ngate,
        "norm_cardinality_diagnostic_gate_passed":diagnostic,
        "registered_scientific_passed":mechanism and diagnostic,
        "uniform_secondary_pairs":{
            arm:{
                "this_only":sum(by[(s,arm)]["success"] and
                                not by[(s,"uniform_control")]["success"] for s in SEEDS),
                "uniform_only":sum(by[(s,"uniform_control")]["success"] and
                                  not by[(s,arm)]["success"] for s in SEEDS),
            } for arm in ("norm_aligned","norm_shuffled")
        },
        "per_seed":[{
            "seed":s,
            "success_by_arm":{arm:by[(s,arm)]["success"] for arm in ARMS},
            "cosine_by_arm":{
                arm:by[(s,arm)]["heldout"]["alignment"]["mean_cosine"]
                for arm in ARMS
            },
            "k4_unseen_by_arm":{
                arm:by[(s,arm)]["heldout"]["per_family"]["k4_unseen"]["support_fraction"]
                for arm in ARMS
            },
            "diagnostic_passed":dby[s]["passed"],
        } for s in SEEDS],
    }


def run_registered() -> dict:
    t0=time.monotonic()
    records,diagnostics=[],[]
    for seed in SEEDS:
        q,pool,sha,support,_=source_for_seed(seed)
        for arm in ARMS:
            records.append(run_arm(seed,arm,q=q,pool=pool,pool_sha=sha))
        diagnostics.append(diagnose_weights(seed,pool,support))
        rs=[r for r in records if r["seed"]==seed]
        for k in ("hidden_rotation_sha256","initial_random_basis_sha256",
                  "training_full_pool_sha256"):
            if len({r[k] for r in rs})!=1:
                raise AssertionError("Paired source hash disagreement")
        for r in rs:
            if r["arm"]=="planted_Q_oracle":
                continue
            tr=r["training"]
            assert tr["candidate_angle_scores"]==39960
            assert tr["training_observed_vectors"]==8192
            assert tr["candidate_scalar_projection_evaluations"]==327352320
            assert tr["pair_sweeps"]==9
        pair=[r for r in rs if r["arm"] in ("norm_aligned","norm_shuffled")]
        assert len(pair)==2
        assert abs(pair[0]["training"]["training_weight_effective_sample_size"]-
                   pair[1]["training"]["training_weight_effective_sample_size"])<1e-10
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        if peak>CONFIG["execution"]["max_RSS_GiB"]:
            raise MemoryError(f"PD16 RSS {peak} GiB exceeded contract")
    return {
        "study_id":"BPD-PD-16",
        "status":"original_pd16_first_registered_perm_placebo_cohort",
        "source_hashes":{
            name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("CONFIG.json","PROTOCOL.md","experiment.py")
        },
        "torch_version":torch.__version__,
        "scipy_version":__import__("scipy").__version__,
        "python_version":platform.python_version(),
        "elapsed_seconds":round(time.monotonic()-t0,6),
        "peak_RSS_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        "summary":summarize(records,diagnostics,registered=True),
        "arms":records,"diagnostics":diagnostics,
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    result=run_registered()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result["summary"],indent=2,sort_keys=True))


if __name__=="__main__":
    main()
