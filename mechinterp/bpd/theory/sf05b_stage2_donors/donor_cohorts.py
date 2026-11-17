"""SF05B Stage 2: deterministic source-aware authorized and public donor cohorts.

Generator-only. No model, BPD factor, hidden state, gradient, future target
model or scientific root is consulted. Bayes baselines condition on the
source-dependent query-role selection and optional high-contrast admission.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations

import torch
from torch.nn import functional as F

from mechinterp.bpd.theory.sf05b_source_families.source_families import (
    HistoryBatch, HELDOUT_TEMPLATES, MAX_WRITES, ROLES, SOURCES,
    make_batch, target_from_records,
)

HIGH_CONTRAST_GAP = .30
HIGH_CONTRAST_OVERSAMPLE = 4
STRATA = ("authorized_unconditional","authorized_high_contrast","public_U")


@dataclass
class DonorCohort:
    recipient: HistoryBatch
    donor_events: torch.Tensor
    donor_source_ids: torch.Tensor
    donor_target: torch.Tensor
    blind_recipient: torch.Tensor
    blind_donor: torch.Tensor
    stratum: str
    attempted_candidates: int
    qualified_candidates: int
    accepted: int

    @property
    def label_change_fraction(self):
        return float((self.recipient.target!=self.donor_target).float().mean())


def _select_batch(batch: HistoryBatch, idx: torch.Tensor) -> HistoryBatch:
    return HistoryBatch(
        batch.events[idx],batch.event_mask[idx],batch.query[idx],
        batch.target[idx],batch.source_ids[idx],batch.values[idx],
        batch.query_role[idx],batch.public_value[idx],batch.lengths[idx],
        batch.source_blind_conditional_mean[idx],batch.family,
    )


def source_aware_blind_means(
    values: torch.Tensor, lengths: torch.Tensor, query_role: torch.Tensor,
    public: torch.Tensor, *, contrast: bool=False,
    threshold: float=HIGH_CONTRAST_GAP,
) -> tuple[torch.Tensor,torch.Tensor]:
    """Exact conditional-mean oracle for the source-aware heldout generator.

    Enumerate all 3! abstract-source name permutations, the two distinct
    destination-source choices, and both authorized role selections. The
    accepted high-contrast population is conditioned on the *source-oracle*
    target gap, without querying any neural model.
    """
    n=values.shape[0]
    if values.shape!=(n,MAX_WRITES) or lengths.shape!=(n,) or query_role.shape!=(n,):
        raise ValueError("source-aware blind oracle shape mismatch")
    rec_sum=torch.zeros(n,dtype=values.dtype)
    don_sum=torch.zeros(n,dtype=values.dtype)
    weights=torch.zeros(n,dtype=values.dtype)
    for length,template in HELDOUT_TEMPLATES.items():
        length_mask=lengths==length
        if not bool(length_mask.any()):
            continue
        for perm in permutations(range(SOURCES)):
            tagged=[perm[j] for j in template]
            old=tagged[-1]
            for offset in (1,2):
                new=(old+offset)%SOURCES
                changed=list(tagged)
                changed[-1]=new
                for role_choice in (old,new):
                    mask=length_mask & (query_role==role_choice)
                    if not bool(mask.any()):
                        continue
                    rec_idx=max(j for j,k in enumerate(tagged) if k==role_choice)
                    don_idx=max(j for j,k in enumerate(changed) if k==role_choice)
                    rv=values[:,rec_idx]
                    dv=values[:,don_idx]
                    if contrast:
                        mask=mask & ((rv-dv).abs()>=threshold)
                    f=mask.to(values.dtype)
                    rec_sum+=f*rv
                    don_sum+=f*dv
                    weights+=f
    authorized=query_role!=3
    if bool((weights[authorized]==0).any()):
        raise ValueError("no blind-oracle support for an authorized accepted record")
    rec=torch.where(authorized,rec_sum/weights.clamp_min(1),public)
    don=torch.where(authorized,don_sum/weights.clamp_min(1),public)
    return rec,don


def make_cohort(seed: int,n: int,stratum: str,
                *,family: str="heldout")->DonorCohort:
    if stratum not in STRATA or n<1 or family!="heldout":
        raise ValueError("Stage 2 source-aware cohorts require heldout family and valid stratum")
    attempted=n*HIGH_CONTRAST_OVERSAMPLE if stratum=="authorized_high_contrast" else n
    base=make_batch(seed,attempted,"heldout")
    row=torch.arange(attempted)
    last=base.lengths-1
    old=base.source_ids[row,last]
    g=torch.Generator().manual_seed(seed+120133)
    offset=1+torch.randint(0,SOURCES-1,(attempted,),generator=g)
    new=(old+offset)%SOURCES
    tag_donor=base.source_ids.clone()
    tag_donor[row,last]=new
    donor_events=base.events.clone()
    donor_events[row,last,:SOURCES]=F.one_hot(new,SOURCES).float()

    choice=torch.randint(0,2,(attempted,),generator=g)
    role=torch.where(choice==0,old,new)
    if stratum=="public_U":
        role=torch.full_like(role,3)
    query=base.query.clone()
    query[:,5:9]=F.one_hot(role,ROLES).float()
    target=target_from_records(base.source_ids,base.values,base.lengths,
                               role,base.public_value)
    donor_target=target_from_records(tag_donor,base.values,base.lengths,
                                     role,base.public_value)
    contrast_ok=(target-donor_target).abs()>=HIGH_CONTRAST_GAP
    qualified=int(contrast_ok.sum()) if stratum=="authorized_high_contrast" else attempted
    if stratum=="authorized_high_contrast":
        selected=contrast_ok.nonzero(as_tuple=True)[0][:n]
        if len(selected)!=n:
            raise RuntimeError(f"high-contrast source cohort admission {len(selected)}/{n} from {attempted}")
    else:
        selected=torch.arange(n)
    batch=_select_batch(replace(base,query=query,query_role=role,target=target),
                        selected)
    donor_events=donor_events[selected]
    tag_donor=tag_donor[selected]
    donor_target=donor_target[selected]
    blind_rec,blind_don=source_aware_blind_means(
        batch.values,batch.lengths,batch.query_role,batch.public_value,
        contrast=(stratum=="authorized_high_contrast"),
    )
    batch=replace(batch,source_blind_conditional_mean=blind_rec)
    return DonorCohort(
        batch,donor_events,tag_donor,donor_target,blind_rec,blind_don,
        stratum,attempted,qualified,n,
    )
