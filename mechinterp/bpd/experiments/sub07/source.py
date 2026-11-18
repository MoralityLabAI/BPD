"""BPD-SUB-07 trained six-pass GRU and frozen 32D/20D source-only QR fits."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .oracle import partition, SEQ_LEN, VOCAB

WIDTH=32
RANK=20
PASSES=6
TRAIN_STEPS=1200
FIT_STEPS=120
STATIC_STEPS=80
TRAIN_LR=.003
FIT_LR=.02
OUTPUT_WEIGHT=.35
SMOOTH_WEIGHT=.50
DISCOVERY_JACOBIAN_ROWS=64
ROOTS=tuple(range(1800,1808))
SMOKE=1799

CLOCK_MAPS=(
    (1,2,3,4,5,0), (2,3,4,5,0,1), (3,4,5,0,1,2),
    (1,0,3,2,5,4), (2,1,0,3,4,5), (0,2,1,3,4,5),
    (5,4,3,2,1,0),
)
STEP_MAPS=(
    (1,2,3,4,0), (2,3,4,0,1), (3,4,0,1,2),
    (1,0,3,2,4), (2,1,0,3,4), (0,2,1,3,4),
    (4,3,2,1,0),
)


def digest(path:Path)->str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_split(name:str):
    seq,labels,ids=partition(name)
    return (
        F.one_hot(torch.as_tensor(seq,dtype=torch.int64),VOCAB).float(),
        torch.as_tensor(labels,dtype=torch.float32),
        ids,
    )


class TiedGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell=nn.GRUCell(VOCAB,WIDTH)
        self.head=nn.Linear(WIDTH,1)

    def step(self,h:Tensor,token:Tensor):
        return self.cell(token,h)

    def rollout(self,x:Tensor):
        if x.ndim!=3 or x.shape[1:]!=(PASSES,VOCAB):
            raise ValueError("Expected [batch, six, four] one-hot token inputs")
        h=x.new_zeros(len(x),WIDTH)
        states=[]
        for t in range(PASSES):
            h=self.step(h,x[:,t])
            states.append(h)
        return self.head(h).squeeze(-1),states


def train_source(seed:int,steps:int=TRAIN_STEPS):
    if seed not in ROOTS and seed!=SMOKE:
        raise ValueError("SUB07 seed outside frozen development roster")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model=TiedGRU()
    x,y,_=tensor_split("train")
    batch_generator=torch.Generator().manual_seed(seed+44)
    optimizer=torch.optim.Adam(model.parameters(),lr=TRAIN_LR)
    for _ in range(steps):
        idx=torch.randint(len(x),(128,),generator=batch_generator)
        prediction,_=model.rollout(x[idx])
        loss=F.binary_cross_entropy_with_logits(prediction,y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits,_=model.rollout(x)
        acc=float(((logits>=0)==y.bool()).float().mean())
        train_nll=float(F.binary_cross_entropy_with_logits(logits,y))
    return model, {
        "seed":seed,"training_steps":steps,"train_rows":len(x),
        "batch_size":128,"training_lr":TRAIN_LR,
        "train_accuracy":acc,"train_nll":train_nll,
        "source_class":"TiedGRUCell-32D-6passes",
    }


def transition_jacobians(model:TiedGRU,x:Tensor):
    """One source Jacobian per input and each post-pass t=1..5."""
    _,states=model.rollout(x)
    out=[]
    for t in range(PASSES-1):
        h=states[t][:DISCOVERY_JACOBIAN_ROWS].detach()
        next_token=x[:DISCOVERY_JACOBIAN_ROWS,t+1].detach()
        with torch.enable_grad():
            j=torch.func.vmap(
                torch.func.jacrev(lambda a,b:model.step(a,b),argnums=0)
            )(h,next_token)
        out.append(j.detach().double())
    return torch.stack(out,dim=1)


def future_covariances(model:TiedGRU,x:Tensor):
    """C_t=E[grad_h_t final_logit outer grad], t=1..6."""
    with torch.enable_grad():
        logits,states=model.rollout(x)
        grads=[
            torch.autograd.grad(logits.sum(),states[t],retain_graph=True)[0]
            .detach().double()
            for t in range(PASSES)
        ]
    cov=torch.stack([g.T@g/len(g) for g in grads])
    return cov, [float(g.norm(dim=-1).mean()) for g in grads]


def discovered_source(model:TiedGRU,x:Tensor):
    d=transition_jacobians(model,x)
    c,norms=future_covariances(model,x)
    if d.shape!=(DISCOVERY_JACOBIAN_ROWS,PASSES-1,WIDTH,WIDTH):
        raise ValueError("Jacobian source shape mismatch")
    if c.shape!=(PASSES,WIDTH,WIDTH):
        raise ValueError("Output covariance shape mismatch")
    return d.detach(),c.detach(),norms


def bundle_loss(p:Tensor,d:Tensor,c:Tensor):
    eye=torch.eye(WIDTH,dtype=torch.float64,device=p.device)
    bridge=((p[1:][None]@d)@(eye-p[:-1])[None])
    a=bridge.square().sum(dim=(-1,-2)).mean()
    b=torch.einsum("tij,tji->",eye[None]-p,c)/PASSES
    smooth=(p[1:]-p[:-1]).square().sum(dim=(-1,-2)).mean()/(2*RANK)
    return a,b,smooth


def stationary_loss(p:Tensor,d:Tensor,c:Tensor):
    eye=torch.eye(WIDTH,dtype=torch.float64,device=p.device)
    cross=(p@d)@(eye-p)
    a=cross.square().sum(dim=(-1,-2)).mean()
    b=torch.einsum("ij,tji->",eye-p,c)/PASSES
    return a,b


def stationary_source(d:Tensor,c:Tensor):
    vals,v=torch.linalg.eigh(c.mean(dim=0))
    start=v[:,-RANK:].flip(-1).contiguous()
    p0=start@start.T
    with torch.no_grad():
        ai,bi=stationary_loss(p0,d,c)
        a0=float(ai.clamp_min(1e-12))
        b0=float(bi.clamp_min(1e-12))
    z=nn.Parameter(start.clone())
    opt=torch.optim.Adam([z],lr=FIT_LR)
    history=[]
    for t in range(STATIC_STEPS):
        q,_=torch.linalg.qr(z,mode="reduced")
        p=q@q.T
        a,b=stationary_loss(p,d,c)
        loss=a/a0+.5*b/b0
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite stationary source loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if z.grad is None or not torch.isfinite(z.grad).all():
            raise ValueError("Nonfinite stationary QR gradient")
        opt.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        q,_=torch.linalg.qr(z,mode="reduced")
        p=q@q.T
        a,b=stationary_loss(p,d,c)
        diag={
            "optimizer_steps":STATIC_STEPS,
            "initial_objective":float(ai/a0+.5*bi/b0),
            "final_objective":float(a/a0+.5*b/b0),
            "initial_closure":float(ai),"initial_output":float(bi),
            "final_closure":float(a),"final_output":float(b),
            "rank_error":abs(float(torch.trace(p))-RANK),
            "idempotence_max":float((p@p-p).abs().max()),
            "symmetry_max":float((p-p.T).abs().max()),
            "source_trace":history,
            "mean_c_eigenspectrum":vals.tolist(),
        }
    return p.float(),q.detach(),diag


def schedule_arrays(d:Tensor,c:Tensor,arm:int):
    if arm==-1:
        return d,c,tuple(range(PASSES-1)),tuple(range(PASSES))
    if arm not in range(7):
        raise ValueError("Wrong-clock arm outside registered control family")
    pi=CLOCK_MAPS[arm]
    tau=STEP_MAPS[arm]
    if sorted(pi)!=list(range(PASSES)) or sorted(tau)!=list(range(PASSES-1)):
        raise ValueError("Invalid frozen time permutation")
    return d[:,list(tau)],c[list(pi)],tau,pi


def fit_one_bundle(d:Tensor,c:Tensor,initial_basis:Tensor,
                   a0:float,b0:float,name:str):
    basis=initial_basis.detach().double()
    params=nn.Parameter(basis.repeat(PASSES,1,1).clone())
    opt=torch.optim.Adam([params],lr=FIT_LR)
    trace=[]
    for step in range(FIT_STEPS):
        q,_=torch.linalg.qr(params,mode="reduced")
        p=q@q.transpose(-1,-2)
        a,b,s=bundle_loss(p,d,c)
        objective=a/a0+OUTPUT_WEIGHT*b/b0+SMOOTH_WEIGHT*s
        if not torch.isfinite(objective):
            raise ValueError("Nonfinite matched-fit source objective")
        opt.zero_grad(set_to_none=True)
        objective.backward()
        if params.grad is None or not torch.isfinite(params.grad).all():
            raise ValueError("Nonfinite matched-fit source gradient")
        opt.step()
        trace.append({
            "step":step+1,"objective_before_update":float(objective.detach()),
            "closure_before_update":float(a.detach()),
            "output_before_update":float(b.detach()),
            "smoothness_before_update":float(s.detach()),
        })
    with torch.no_grad():
        q,_=torch.linalg.qr(params,mode="reduced")
        p=q@q.transpose(-1,-2)
        a,b,s=bundle_loss(p,d,c)
        diag={
            "arm":name,"optimizer_steps":FIT_STEPS,
            "optimizer_lr":FIT_LR,
            "output_weight":OUTPUT_WEIGHT,
            "smooth_weight":SMOOTH_WEIGHT,
            "normalizer_a0":a0,"normalizer_b0":b0,
            "source_fit_final_closure":float(a),
            "source_fit_final_output":float(b),
            "source_fit_final_smoothness":float(s),
            "source_fit_final_objective":float(
                a/a0+OUTPUT_WEIGHT*b/b0+SMOOTH_WEIGHT*s),
            "rank_trace_max_error":max(abs(float(torch.trace(pt))-RANK)
                                       for pt in p),
            "idempotence_max":float((p@p-p).abs().max()),
            "symmetry_max":float((p-p.transpose(-1,-2)).abs().max()),
            "source_loss_trace":trace,
        }
    return p.float(),q.detach(),diag


def source_fit_family(d:Tensor,c:Tensor,static_basis:Tensor):
    pstatic=static_basis@static_basis.T
    p0=torch.stack([pstatic for _ in range(PASSES)])
    with torch.no_grad():
        aa,bb,_=bundle_loss(p0,d,c)
        a0=float(aa.clamp_min(1e-12))
        b0=float(bb.clamp_min(1e-12))
    projectors,bases,diags={},{},{}
    for arm in (-1,*range(7)):
        d_arm,c_arm,tau,pi=schedule_arrays(d,c,arm)
        label="aligned" if arm==-1 else f"wrong_{arm}"
        ps,q,diag=fit_one_bundle(d_arm,c_arm,static_basis,a0,b0,label)
        with torch.no_grad():
            true_a,true_b,true_s=bundle_loss(ps.double(),d,c)
        diag.update({
            "true_clock_discovery_closure":float(true_a),
            "true_clock_discovery_output":float(true_b),
            "true_clock_discovery_smoothness":float(true_s),
            "source_step_schedule":list(tau),
            "source_output_schedule":list(pi),
            "source_jacobian_rows":int(d.shape[0]),
            "source_covariance_rows":DISCOVERY_JACOBIAN_ROWS*8,
        })
        projectors[label]=ps
        bases[label]=q
        diags[label]=diag
    return projectors,bases,diags,{"a0":a0,"b0":b0}
