"""BPD-SF05B Stage 1 bounded correction policy.

Only the *training-label generator* may call the frozen public readout
oracle. Policy.forward and predict_from_envelope accept four explicit
permitted tensors and never obtain a model, future output, derivative,
donor gold label or previously scored heldout result.
"""
from __future__ import annotations

import torch
from torch import nn

from mechinterp.bpd.theory.sf05_public_safe.finite_correct import finite_public_repair
from mechinterp.bpd.theory.sf05b_source_families.source_families import make_batch, change_last_source
from mechinterp.bpd.theory.sf05b_stage1.teacher import (
    HIDDEN, EVENT_WIDTH, Factors, Teacher, generator, public_query,
    recipient_donor_edit,
)

POLICY_UPDATES = 240
POLICY_BATCH = 128
CORRECTION_FRACTION = .25
CORRECTION_MAX_NORM = .5


class NonprivilegedPolicy(nn.Module):
    """Small model with a fixed, restricted inference signature."""

    def __init__(self, seed: int):
        super().__init__()
        # Policy initialization is independent of teacher, factor and dataset seeds.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed)+400177)
            self.net = nn.Sequential(
                nn.Linear(2*HIDDEN+EVENT_WIDTH+1, 96), nn.Tanh(),
                nn.Linear(96, 96), nn.Tanh(),
                nn.Linear(96, HIDDEN),
            )

    def forward(self, state: torch.Tensor, candidate_delta: torch.Tensor,
                public_query: torch.Tensor, current_readout: torch.Tensor
                ) -> torch.Tensor:
        """Return bounded correction, not the final edit."""
        n = len(state)
        if (state.shape != (n,HIDDEN) or candidate_delta.shape != (n,HIDDEN)
                or public_query.shape != (n,EVENT_WIDTH)
                or current_readout.shape != (n,1)):
            raise ValueError("restricted inference tensor shapes changed")
        inputs = torch.cat((state,candidate_delta,public_query,current_readout),dim=-1)
        corr = self.net(inputs)
        corr = torch.cat((torch.zeros_like(corr[:, :1]),corr[:, 1:]),dim=-1)
        budget = torch.minimum(CORRECTION_FRACTION*candidate_delta.norm(
            dim=-1,keepdim=True), torch.full((n,1),CORRECTION_MAX_NORM,device=corr.device))
        corr_norm = corr.norm(dim=-1,keepdim=True)
        corr = corr * torch.minimum(torch.ones_like(budget),budget/corr_norm.clamp_min(1e-12))
        return corr

    def corrected_edit(self, state:torch.Tensor, candidate_delta:torch.Tensor,
                       public_query:torch.Tensor, current_readout:torch.Tensor
                       ) -> torch.Tensor:
        dark = candidate_delta.clone()
        dark[:,0] = 0
        corrected = dark + self.forward(state,dark,public_query,current_readout)
        corrected[:,0] = 0
        return corrected


def predict_from_envelope(policy:NonprivilegedPolicy,envelope:dict)->torch.Tensor:
    """Whitelist only permitted fields; ignore poisoned/untrusted metadata."""
    return policy.corrected_edit(
        envelope["state"],envelope["candidate_delta"],
        envelope["public_query"],envelope["current_readout"])


def privileged_targets(teacher:Teacher, state:torch.Tensor,
                       delta:torch.Tensor, public:torch.Tensor
                       )->tuple[torch.Tensor,torch.Tensor]:
    """Separate training-only oracle; never passed to policy inference."""
    repaired=finite_public_repair(
        state,delta,lambda st:teacher.step(st,public)[:,0],
        max_iterations=8,tolerance=1e-4,per_step_relative_cap=.25,
        per_step_absolute_cap=.5,gradient_tol=1e-9)
    return (repaired.delta-delta).detach(),repaired.converged.detach()


def train_policy(seed:int,teacher:Teacher,factors:Factors,
                 updates:int=POLICY_UPDATES,batch_size:int=POLICY_BATCH
                 )->tuple[NonprivilegedPolicy,dict]:
    """Supervise on disjoint F_train samples, with half random-dark writes."""
    policy=NonprivilegedPolicy(seed)
    opt=torch.optim.Adam(policy.parameters(),lr=.003)
    g=generator(seed+817400)
    converged,n_total=0,0
    for step in range(updates):
        batch=make_batch(seed*190000+500000+step,batch_size,"train")
        donor_events,_,_=change_last_source(batch,seed*280000+710000+step)
        with torch.no_grad():
            states=teacher.rollout(batch)
            _,edit,_=recipient_donor_edit(teacher,factors,batch,donor_events,states)
            public=public_query(batch)
            randomized=torch.randn(edit.shape,generator=g)
            randomized[:,0]=0
            randomized=randomized/randomized.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            randomized*=edit.norm(dim=-1,keepdim=True)
            replacement=torch.rand((batch_size,),generator=g)<.5
            chosen=torch.where(replacement[:,None],randomized,edit)
            state=states.prequery.detach()
            visible=state[:,0:1].detach()
        target,success=privileged_targets(teacher,state,chosen,public)
        converged+=int(success.sum())
        n_total+=len(success)
        predicted=policy(state,chosen,public,visible)
        loss=(predicted-target).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite policy fit at {seed}, update {step}")
        opt.zero_grad()
        loss.backward()
        opt.step()
    policy.eval().requires_grad_(False)
    return policy,{
        "train_oracle_labels":n_total,
        "train_oracle_convergence_rate":converged/max(n_total,1),
        "training_family":"train",
        "policy_updates":updates,
        "privileged_training_label_source":"finite public level-set Newton oracle",
        "inference_uses_oracle":False,
    }
