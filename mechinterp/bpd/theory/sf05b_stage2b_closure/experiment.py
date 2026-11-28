"""BPD-SF05B Stage 2-B: four preregistered source-closure training arms.

Development-only roots 901 and 902. Model training is confined to Stage 0
F_train histories; F_heldout is evaluated once the source hashes are locked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from mechinterp.bpd.theory.sf05b_source_families.source_families import (
    make_batch, change_last_source,
)
from mechinterp.bpd.theory.sf05b_stage1.teacher import (
    Teacher, generator, train_teacher, train_factors, public_query, competence,
)
from mechinterp.bpd.theory.sf05b_stage1.policy import train_policy
from mechinterp.bpd.theory.sf05b_stage2_admission.evaluate import (
    _score_cohort, N_PER_COHORT, SHAM_ARMS, EXCLUDED_ROOTS,
)

ARM_WEIGHTS={
    "baseline":(0.0,0.0),
    "current_only":(1.0,0.0),
    "public_only":(0.0,1.0),
    "joint":(1.0,1.0),
}
UPDATES=2600
FACTOR_UPDATES=450
POLICY_UPDATES=240
BATCH=128


def train_teacher_arm(seed:int,arm:str,updates:int=UPDATES,
                      batch_size:int=BATCH)->Teacher:
    if seed not in EXCLUDED_ROOTS or arm not in ARM_WEIGHTS:
        raise ValueError("Stage2-B only runs excluded 901/902 and frozen four arms")
    if arm=="baseline":
        return train_teacher(seed,updates=updates,batch_size=batch_size)
    current_coefficient,public_coefficient=ARM_WEIGHTS[arm]
    teacher=Teacher(seed)
    optimizer=torch.optim.Adam(teacher.parameters(),lr=.012)
    for update in range(updates):
        batch=make_batch(seed*100000+update,batch_size,"train")
        recipient=teacher.rollout(batch)
        final_loss=(recipient.final_output-batch.target).square().mean()
        early=(recipient.intermediate_readout*batch.event_mask).square().sum()/batch.event_mask.sum().clamp_min(1)
        objective=4*final_loss+.15*early
        objective+=.0001*(teacher.A.square().mean()+teacher.C.square().mean())
        if current_coefficient or public_coefficient:
            donor_events,_,_=change_last_source(
                batch,seed*290000+880000+update)
            donor=teacher.rollout(batch,donor_events)
            if current_coefficient:
                objective+=current_coefficient*(
                    recipient.prequery[:,0]-donor.prequery[:,0]).square().mean()
            if public_coefficient:
                query_u=public_query(batch)
                pub_rec=teacher.step(recipient.prequery,query_u)[:,0]
                pub_donor=teacher.step(donor.prequery,query_u)[:,0]
                objective+=public_coefficient*(pub_rec-pub_donor).square().mean()
        if not torch.isfinite(objective):
            raise RuntimeError(f"nonfinite objective: root={seed} arm={arm} update={update}")
        optimizer.zero_grad()
        objective.backward()
        nn.utils.clip_grad_norm_(teacher.parameters(),5.)
        optimizer.step()
    return teacher.eval().requires_grad_(False)


def training_only_closure_diagnostics(seed:int,arm:str,teacher:Teacher,n:int=512)->dict:
    """Independent F_train source-only donor sample; no heldout observed."""
    batch=make_batch(seed*720000+403100,n,"train")
    donor_events,_,_=change_last_source(batch,seed*730000+410400)
    with torch.no_grad():
        r=teacher.rollout(batch)
        d=teacher.rollout(batch,donor_events)
        public=public_query(batch)
        pr=teacher.step(r.prequery,public)[:,0]
        pd=teacher.step(d.prequery,public)[:,0]
        return {
            "family":"train","n":n,"arm":arm,
            "current_prequery_abs_difference_mean":float(
                (r.prequery[:,0]-d.prequery[:,0]).abs().mean()),
            "current_prequery_mean_square_difference":float(
                (r.prequery[:,0]-d.prequery[:,0]).square().mean()),
            "public_clean_tag_only_abs_change_mean":float((pr-pd).abs().mean()),
            "public_clean_tag_only_mse":float((pr-pd).square().mean()),
        }


def evaluate_arm(seed:int,arm:str,n:int=N_PER_COHORT,
                 shams:int=SHAM_ARMS,updates:int=UPDATES,
                 factor_steps:int=FACTOR_UPDATES,
                 policy_steps:int=POLICY_UPDATES)->dict:
    if seed not in EXCLUDED_ROOTS or arm not in ARM_WEIGHTS:
        raise ValueError("only excluded teacher roots and preregistered arms")
    torch.set_num_threads(1)
    teacher=train_teacher_arm(seed,arm,updates,BATCH)
    factors=train_factors(seed,teacher,updates=factor_steps,batch_size=BATCH)
    policy,meta=train_policy(seed,teacher,factors,updates=policy_steps,
                            batch_size=BATCH)
    heldout=competence(teacher,seed*400000+999000,"heldout",n)
    heldout["descriptive_admission"]=bool(
        heldout["mae"]<=.12 and heldout["public_mae"]<=.05
        and heldout["mae"]<=.8*heldout["blind_mae"])
    return {
        "seed":seed,"arm":arm,
        "current_penalty":ARM_WEIGHTS[arm][0],
        "public_penalty":ARM_WEIGHTS[arm][1],
        "evidence_class":"excluded-root 4-arm teacher training objective ablation",
        "train_closure":training_only_closure_diagnostics(seed,arm,teacher),
        "heldout_competence":heldout,
        "policy_training":meta,
        "cohorts":{
            name:_score_cohort(seed,teacher,factors,policy,name,n,shams)
            for name in ("authorized_unconditional","authorized_high_contrast","public_U")
        },
    }


def run(roots=EXCLUDED_ROOTS,n=N_PER_COHORT,shams=SHAM_ARMS,
        updates=UPDATES,factor_steps=FACTOR_UPDATES,policy_steps=POLICY_UPDATES):
    if tuple(roots)!=EXCLUDED_ROOTS:
        raise ValueError("Stage2-B 901,902 only; scientific roots untouched")
    return {
        "experiment":"BPD-SF05B Stage2-B source-closure teacher objective ablation",
        "evidence_scope":"development-only, excluded roots, no 59/32 calibration",
        "roots":list(roots),"arms":list(ARM_WEIGHTS),
        "config":{"n_per_cohort":n,"sham_arms":shams,
                  "teacher_updates":updates,"factor_updates":factor_steps,
                  "policy_updates":policy_steps},
        "torch_version":torch.__version__,
        "rows":[evaluate_arm(seed,arm,n,shams,updates,factor_steps,policy_steps)
                for seed in roots for arm in ARM_WEIGHTS],
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--roots",nargs="+",type=int,default=list(EXCLUDED_ROOTS))
    p.add_argument("--samples",type=int,default=N_PER_COHORT)
    p.add_argument("--shams",type=int,default=SHAM_ARMS)
    p.add_argument("--teacher-updates",type=int,default=UPDATES)
    p.add_argument("--factor-updates",type=int,default=FACTOR_UPDATES)
    p.add_argument("--policy-updates",type=int,default=POLICY_UPDATES)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    data=run(tuple(args.roots),args.samples,args.shams,
             args.teacher_updates,args.factor_updates,args.policy_updates)
    print(json.dumps(data,separators=(",",":")))
    if args.out:
        args.out.write_text(json.dumps(data,indent=2)+"\n")
