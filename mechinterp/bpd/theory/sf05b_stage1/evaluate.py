"""BPD-SF05B Stage 1 excluded-root assessment on disjoint history families.

Scientific admission is explicitly deferred: Stage 1 only diagnoses two
previously excluded roots. The reserved 59/32 fresh rosters stay untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechinterp.bpd.theory.sf04_source_binding.experiment import tangent_matched
from mechinterp.bpd.theory.sf05_public_safe.geometry import physical_output_subspace_write
from mechinterp.bpd.theory.sf05_public_safe.finite_correct import finite_public_repair
from mechinterp.bpd.theory.sf05b_source_families.source_families import (
    make_batch, change_last_source,
)
from mechinterp.bpd.theory.sf05b_stage1.teacher import (
    Teacher, Factors, generator, train_teacher, train_factors,
    recipient_donor_edit, public_query, competence,
)
from mechinterp.bpd.theory.sf05b_stage1.policy import (
    NonprivilegedPolicy, train_policy, predict_from_envelope,
)

EXCLUDED=(901,902)
RANK=24
EVAL_ROWS=512
SHAM_ARMS=16


@torch.no_grad()
def _score_arm(teacher:Teacher,state:torch.Tensor,query:torch.Tensor,
               public:torch.Tensor,edit:torch.Tensor,
               recipient_public:torch.Tensor,donor_clean:torch.Tensor,
               eligible:torch.Tensor)->dict:
    d=edit.clone()
    d[:,0]=0.
    if not torch.equal((state+d)[:,0],state[:,0]):
        raise AssertionError("prequery readout not preserved")
    final=teacher.step(state+d,query)[:,0]
    post_public=teacher.step(state+d,public)[:,0]
    pchange=(post_public-recipient_public).abs()
    dose=d.norm(dim=-1)
    n=int(eligible.sum())
    return {
        "authorized_eligible":n,
        "authorized_donor_recovery":float(((final-donor_clean).abs()<=.10)[eligible].float().mean()) if n else None,
        "public_abs_mean":float(pchange.mean()),
        "public_abs_eligible_mean":float(pchange[eligible].mean()) if n else None,
        "public_below_0p02_fraction":float((pchange<=.02).float().mean()),
        "dose_mean":float(dose.mean()),
        "prequery_readout_max_error":float(((state+d)[:,0]-state[:,0]).abs().max()),
    }


def _shams(teacher,state,query,public,recipient_public,donor,eligible,
           raw_edit,policy,seed,n=SHAM_ARMS):
    state_for_grad=state.detach().requires_grad_(True)
    grad=torch.autograd.grad(
        teacher.step(state_for_grad,query)[:,0].sum(),state_for_grad)[0].detach()
    gen=generator(seed)
    nulls={"random_dark":[],"signed_authorized_gradient":[]}
    with torch.no_grad():
        for j in range(n):
            random=torch.randn(raw_edit.shape,generator=gen)
            random[:,0]=0.
            random=random/random.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            random*=raw_edit.norm(dim=-1,keepdim=True)
            signed, _, _ = tangent_matched(raw_edit,grad,gen)
            for label,edit in (("random_dark",random),
                               ("signed_authorized_gradient",signed)):
                corrected=predict_from_envelope(policy,{
                    "state":state,"candidate_delta":edit,
                    "public_query":public,"current_readout":state[:,0:1],
                })
                nulls[label].append(_score_arm(
                    teacher,state,query,public,corrected,
                    recipient_public,donor,eligible))
    return {
        k:{
            "arms":n,
            "mean_authorized_recovery":sum(
                (v["authorized_donor_recovery"] or 0.) for v in vs)/n,
            "mean_public_abs":sum(v["public_abs_mean"] for v in vs)/n,
            "mean_dose":sum(v["dose_mean"] for v in vs)/n,
            "matching_applied_before_correction":True,
        } for k,vs in nulls.items()
    }


def evaluate_family(seed:int,teacher:Teacher,factors:Factors,
                    policy:NonprivilegedPolicy,family:str,
                    n:int=EVAL_ROWS,shams:int=SHAM_ARMS)->dict:
    offset=905300 if family=="train" else 915300
    batch=make_batch(seed*320000+offset,n,family)
    donor_events,donor_target,_=change_last_source(
        batch,seed*330000+offset+17)
    with torch.no_grad():
        rec=teacher.rollout(batch)
        donor=teacher.rollout(batch,donor_events)
        donor_state,raw,phi=recipient_donor_edit(
            teacher,factors,batch,donor_events,rec)
        public=public_query(batch)
        recipient_public=teacher.step(rec.prequery,public)[:,0]
        raw24,gap=physical_output_subspace_write(phi,factors.weight(),RANK)
        raw24[:,0]=0
        teacher_raw=(phi@teacher.C).detach()
        teacher_raw[:,0]=0
        exact=(donor_state-rec.prequery).clone()
        exact[:,0]=0
        eligible=(
            (batch.query_role!=3)
            & ((rec.final_output-batch.target).abs()<=.10)
            & ((donor.final_output-donor_target).abs()<=.10)
            & ((rec.prequery[:,0]-donor.prequery[:,0]).abs()<=.05)
            & ((rec.final_output-donor.final_output).abs()>=.20)
        )
        if not torch.equal(donor_state,donor.prequery):
            if not torch.allclose(donor_state,donor.prequery,atol=1e-6):
                raise AssertionError("donor last event needs to share pre-last history")
        state=rec.prequery.detach()
        query=batch.query
        donor_clean=donor.final_output.detach()
        physical=factors.weight().detach()
        R=torch.eye(len(factors.U))
        R[0,1]=.7
        sheared=(factors.V.detach()@R)@torch.linalg.solve(R,factors.U.detach())
        gauge=float((physical-sheared).abs().max())

    arms={
        "zero":torch.zeros_like(raw),
        "bpd_full_raw":raw,
        "bpd_rank24_raw":raw24,
        "teacher_physical_raw":teacher_raw,
        "exact_dark_donor":exact,
    }
    with torch.no_grad():
        for name,delta in (("bpd_full",raw),("bpd_rank24",raw24)):
            arms[name+"_learned"]=predict_from_envelope(policy,{
                "state":state,"candidate_delta":delta,
                "public_query":public,"current_readout":state[:,0:1],
                "future_public":"POISON","donor_label":"POISON",
            })
    oracle_meta={}
    for name,delta in (("bpd_full",raw),("bpd_rank24",raw24)):
        repaired=finite_public_repair(
            state,delta,lambda x:teacher.step(x,public)[:,0],
            max_iterations=8,tolerance=1e-4)
        arms[name+"_privileged_oracle"]=repaired.delta.detach()
        oracle_meta[name]={
            "convergence_all":float(repaired.converged.float().mean()),
            "nonconverged":int((~repaired.converged).sum()),
            "mean_additional_correction_norm":float(repaired.correction_norm.mean()),
        }
    scores={
        name:_score_arm(teacher,state,query,public,delta,
                        recipient_public,donor_clean,eligible)
        for name,delta in arms.items()
    }
    nulls={}
    for name,delta in (("bpd_full",raw),("bpd_rank24",raw24)):
        nulls[name]=_shams(teacher,state,query,public,
                           recipient_public,donor_clean,eligible,
                           delta,policy,seed+982700+(0 if name=="bpd_full" else 1000),shams)
    with torch.no_grad():
        last_change=(batch.target!=donor_target)
        output_change=(rec.final_output-donor.final_output).abs()
        return {
            "seed":seed, "family":family,"total_pairs":n,
            "role_counts":{str(k):int((batch.query_role==k).sum()) for k in range(4)},
            "changed_source_target_pairs":int(last_change.sum()),
            "authorized_eligible":int(eligible.sum()),
            "clean_source_tag_only_future_difference_mean":float(output_change.mean()),
            "teacher_recipient_mae":float((rec.final_output-batch.target).abs().mean()),
            "teacher_donor_mae":float((donor.final_output-donor_target).abs().mean()),
            "source_blind_mae":float((batch.source_blind_conditional_mean-batch.target).abs().mean()),
            "source_blind_mse":float((batch.source_blind_conditional_mean-batch.target).square().mean()),
            "physical_reconstruction_mse":float((physical-teacher.C).square().mean()),
            "bpd_gl32_gauge_max_error":gauge,
            "bpd_rank24_singular_gap":gap,
            "oracle_meta":oracle_meta,
            "arms":scores,
            "corrected_nulls":nulls,
        }


def run_root(seed:int,train_steps:int=2600,
             factor_steps:int=450,policy_steps:int=240,
             n:int=EVAL_ROWS,shams:int=SHAM_ARMS)->dict:
    if seed not in EXCLUDED:
        raise ValueError("Stage 1 engineering is restricted to excluded roots 901/902")
    torch.set_num_threads(1)
    teacher=train_teacher(seed,train_steps)
    factors=train_factors(seed,teacher,factor_steps)
    policy,policy_meta=train_policy(seed,teacher,factors,policy_steps)
    return {
        "seed":seed,"evidence_class":"excluded-root engineering only",
        "training":{"teacher_steps":train_steps,"factor_steps":factor_steps,
                    **policy_meta},
        "train_competence":competence(teacher,seed*400000+899000,"train",n),
        "heldout_competence":competence(teacher,seed*400000+999000,"heldout",n),
        "train_family":evaluate_family(seed,teacher,factors,policy,"train",n,shams),
        "heldout_family":evaluate_family(seed,teacher,factors,policy,"heldout",n,shams),
    }


def run(seeds=EXCLUDED,train_steps=2600,factor_steps=450,policy_steps=240,
        n=EVAL_ROWS,shams=SHAM_ARMS):
    if any(seed not in EXCLUDED for seed in seeds):
        raise ValueError("fresh 59/32 scientific roots remain reserved")
    return {
        "scope":"BPD-SF05B Stage 1: excluded engineering roots, structurally disjoint OOD",
        "roots":list(seeds),"torch":torch.__version__,
        "rows":[run_root(s,train_steps,factor_steps,policy_steps,n,shams)
                for s in seeds],
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--roots",type=int,nargs="+",default=list(EXCLUDED))
    p.add_argument("--teacher-updates",type=int,default=2600)
    p.add_argument("--factor-updates",type=int,default=450)
    p.add_argument("--policy-updates",type=int,default=240)
    p.add_argument("--samples",type=int,default=512)
    p.add_argument("--shams",type=int,default=16)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    r=run(tuple(a.roots),a.teacher_updates,a.factor_updates,
          a.policy_updates,a.samples,a.shams)
    print(json.dumps(r,separators=(",",":")))
    if a.out:
        a.out.write_text(json.dumps(r,indent=2)+"\n")
