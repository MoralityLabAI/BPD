"""Depth-transport causal steering with full-observer checks and strong nulls.

The evaluator is generic over a learned recurrent transition `step` and
host-owned readouts. Interventions are selected on calibration examples only.
A signed future-task Jacobian matched sham is a *privileged* control.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import torch


@dataclass(frozen=True)
class DepthContract:
    fit: tuple[int,...]=(2,4,6)
    test: tuple[int,...]=(8,12)
    rank: int=2
    readout_tolerance: float=1e-6
    donor_tolerance: float=.1

    def __post_init__(self):
        if (not self.fit or not self.test or
            any(d<1 for d in (*self.fit,*self.test)) or
            len(set((*self.fit,*self.test)))!=len((*self.fit,*self.test)) or
            min(self.test)<=max(self.fit) or self.rank<1):
            raise ValueError("fit and extrapolation depths must be disjoint and ordered")


def nullspace_basis(observer_jac:torch.Tensor,*,tol:float=1e-8)->torch.Tensor:
    """Orthogonal kernel in the declared physical state metric."""
    if observer_jac.ndim!=2:raise ValueError("expected [P,D] observer Jacobian")
    u,s,vh=torch.linalg.svd(observer_jac.double(),full_matrices=True)
    rank=int((s>tol*max(1.,float(s.max()))).sum()) if s.numel() else 0
    return vh[rank:].T.float()


def continuation(step:Callable,initial:torch.Tensor,depth:int,context=None):
    h=initial
    for _ in range(depth):h=step(h,context)
    return h


def select_task_block(step:Callable,task:Callable,develop_states:torch.Tensor,
                      protected_basis:torch.Tensor,contract:DepthContract,
                      context=None)->tuple[torch.Tensor,dict]:
    """Freeze physical top-r direction of development future-task Gramian."""
    if contract.rank>protected_basis.shape[1]:raise ValueError("rank exceeds protected dimension")
    gram=None
    for depth in contract.fit:
        state=develop_states.detach().clone().requires_grad_(True)
        future=task(continuation(step,state,depth,context))
        if future.ndim==1:future=future.unsqueeze(-1)
        grads=[]
        for j in range(future.shape[-1]):
            grad=torch.autograd.grad(future[:,j].sum(),state,retain_graph=j<future.shape[-1]-1)[0]
            grads.append(grad.detach()@protected_basis)
        jac=torch.stack(grads,dim=1).double()
        current=torch.einsum('bmi,bmj->ij',jac,jac)/len(state)
        gram=current if gram is None else gram+current
    gram=gram/len(contract.fit)
    spectrum,basis=torch.linalg.eigh(gram)
    rank=contract.rank
    physical=protected_basis@basis[:,-rank:].float()
    gap=float(spectrum[-rank]-spectrum[-rank-1]) if rank<len(spectrum) else None
    return physical.detach(),{"fit_depths":list(contract.fit),"rank":rank,
                              "gap":gap,"eigenvalues":spectrum.flip(0).tolist()}


def signed_task_matched_sham(jac:torch.Tensor,selected:torch.Tensor,
                             protected_basis:torch.Tensor,generator:torch.Generator):
    """Physical-norm and full-vector signed task tangent match within K."""
    j=(jac@protected_basis).double()
    vec=(selected@protected_basis).double()
    pin=torch.linalg.pinv(j,rtol=1e-11)
    required=(pin@(j@vec.unsqueeze(-1))).squeeze(-1)
    random=torch.randn(vec.shape,generator=generator,dtype=torch.float64)
    random=random-(pin@(j@random.unsqueeze(-1))).squeeze(-1)
    random=random/random.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    radius=(vec.square().sum(-1,keepdim=True)-required.square().sum(-1,keepdim=True)).clamp_min(0).sqrt()
    control=required+radius*random
    out=(control@protected_basis.double().T).float()
    signed_error=float((j@control.unsqueeze(-1)-j@vec.unsqueeze(-1)).abs().max())
    norm_error=float((control.norm(dim=-1)-vec.norm(dim=-1)).abs().max())
    return out,{"signed_full_task_jacobian_max_error":signed_error,
                "l2_max_error":norm_error,"privileged":True}


def evaluate_patch(step:Callable,task:Callable,observer:Callable,
                   recipient:torch.Tensor,donor:torch.Tensor,edit:torch.Tensor,
                   depth:int,*,donor_threshold:float=.1,observer_tol:float=1e-6,
                   context=None)->dict:
    if len(recipient)!=len(donor) or recipient.shape!=edit.shape:
        raise ValueError("recipient/donor/edit shape mismatch")
    with torch.no_grad():
        measured=(observer(recipient+edit)-observer(recipient)).abs()
        max_error=float(measured.max())
        if max_error>observer_tol:raise AssertionError("full current observer changed")
        future=task(continuation(step,recipient+edit,depth,context))
        ground=task(continuation(step,donor,depth,context))
        error=(future-ground).abs()
        if error.ndim==1:error=error.unsqueeze(-1)
        return {"depth":depth,"pairs":len(recipient),"protected_max_error":max_error,
                "donor_recovery":float((error.amax(-1)<=donor_threshold).float().mean()),
                "donor_mse":float(error.square().mean()),
                "dose_mean":float(edit.norm(dim=-1).mean())}
