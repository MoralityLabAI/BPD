"""Source-observable projective-ray identification comparator.

This algorithm uses only dense observed training x. It is a separately
costed feasibility comparator, not a 39,960-angle matched Givens arm.
No support labels or true Q enter the search.
"""
from __future__ import annotations

import hashlib
import time

import torch

from mechinterp.bpd.theory.pd17.source import CONFIG, D

SPEC=CONFIG["angular_control"]


def angular_basis(observed:torch.Tensor):
    """Return polar-orthogonalized 16-ray basis or an explicit abstention."""
    if observed.ndim!=4 or observed.shape[-3:]!=(4,1,D):
        raise ValueError("Expected [boards,4,1,16] observed x")
    t0=time.monotonic()
    x=observed.reshape(-1,D).double()
    n=len(x)
    prefix=SPEC["candidate_prefix"]
    if n<prefix:
        raise ValueError("Observed pool must contain angular candidate prefix")
    unit=x/x.norm(dim=1,keepdim=True).clamp_min(1e-30)
    anchors=unit[:prefix]
    counts=torch.zeros(prefix,dtype=torch.int64)
    # Chunking bounds peak memory; all 8192 samples remain eligible.
    for begin in range(0,prefix,128):
        sims=(anchors[begin:begin+128]@unit.T).abs()
        counts[begin:begin+128]=(sims>=SPEC["min_abs_cosine"]).sum(1)
    order=sorted(range(prefix),key=lambda i:(-int(counts[i]),i))
    chosen=[]
    for idx in order:
        if int(counts[idx])<SPEC["min_neighbors"]:
            break
        v=anchors[idx]
        if all(abs(float(v@anchors[j]))<SPEC["max_selected_abs_cosine"]
               for j in chosen):
            chosen.append(idx)
            if len(chosen)==D:
                break
    result={
        "candidate_count":prefix,
        "observed_vector_count":n,
        "angular_scalar_multiplications":prefix*n*D,
        "min_neighbors":SPEC["min_neighbors"],
        "cosine_threshold":SPEC["min_abs_cosine"],
        "selection_separation":SPEC["max_selected_abs_cosine"],
        "selected_training_indices":chosen,
        "selected_neighbor_counts":[int(counts[i]) for i in chosen],
        "abstained":len(chosen)!=D,
        "candidate_angle_scores":0,
        "nonzero_moves":len(chosen),
        "training_seconds":round(time.monotonic()-t0,6),
    }
    if len(chosen)!=D:
        result["orthogonality_max_error"]=None
        return None,result
    candidate=anchors[chosen].T
    u,sv,vh=torch.linalg.svd(candidate,full_matrices=False)
    basis=u@vh
    result["minimum_prepolar_singular_value"]=float(sv[-1])
    result["orthogonality_max_error"]=float(
        (basis.T@basis-torch.eye(D,dtype=torch.float64)).abs().max())
    result["basis_sha256"]=hashlib.sha256(basis.numpy().tobytes()).hexdigest()
    return basis,result
