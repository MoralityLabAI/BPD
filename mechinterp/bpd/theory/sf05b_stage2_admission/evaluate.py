"""SF05B Stage 2-A excluded-root admission, trained model and separate public cohorts.

Reuses frozen Stage 1 teacher/factors/policy and Stage 2 source-only generator.
Only previously excluded roots 901/902 are permitted. No new calibration
or confirmation roots, private user data or external API models are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechinterp.bpd.theory.sf05_public_safe.finite_correct import finite_public_repair
from mechinterp.bpd.theory.sf05_public_safe.geometry import physical_output_subspace_write
from mechinterp.bpd.theory.sf05b_stage1.teacher import (
    train_teacher, train_factors, recipient_donor_edit, public_query, competence,
)
from mechinterp.bpd.theory.sf05b_stage1.policy import (
    train_policy, predict_from_envelope, CORRECTION_FRACTION, CORRECTION_MAX_NORM,
)
from mechinterp.bpd.theory.sf05b_stage1.evaluate import _score_arm, _shams
from mechinterp.bpd.theory.sf05b_stage2_donors.donor_cohorts import make_cohort


EXCLUDED_ROOTS=(901,902)
N_PER_COHORT=512
SHAM_ARMS=16
COHORT_SEED_OFFSETS={
    "authorized_unconditional":530000,
    "authorized_high_contrast":540000,
    "public_U":550000,
}
RANK=24


def eligibility_ladder(batch, donor_target, recipient_states, donor_states):
    """All preregistered donor filters retained, with independent denominators."""
    n=len(batch.target)
    role_auth=batch.query_role!=3
    label_gap=(batch.target-donor_target).abs()
    rec_err=(recipient_states.final_output-batch.target).abs()
    donor_err=(donor_states.final_output-donor_target).abs()
    clean=(rec_err<=.10)&(donor_err<=.10)
    boundary=(recipient_states.prequery[:,0]-donor_states.prequery[:,0]).abs()<=.05
    clean_output_diff=(recipient_states.final_output-donor_states.final_output).abs()
    future=clean_output_diff>=.20
    joint=role_auth&clean&boundary&future
    report={
        "generated":n,
        "authorized_role_count":int(role_auth.sum()),
        "source_label_change_count":int((label_gap>0).sum()),
        "source_label_gap_ge_0p20":int((label_gap>=.20).sum()),
        "source_label_gap_ge_0p30":int((label_gap>=.30).sum()),
        "source_label_abs_gap_mean":float(label_gap.mean()),
        "both_clean_teacher_correct_count":int((role_auth&clean).sum()),
        "boundary_pass_count_independent":int((role_auth&boundary).sum()),
        "future_delta_pass_count_independent":int((role_auth&future).sum()),
        "after_clean_then_boundary_count":int((role_auth&clean&boundary).sum()),
        "all_model_filters_count":int(joint.sum()),
        "clean_recipient_mae":float(rec_err.mean()),
        "clean_donor_mae":float(donor_err.mean()),
        "prequery_difference_abs_mean":float(
            (recipient_states.prequery[:,0]-donor_states.prequery[:,0]).abs().mean()),
        "clean_donor_future_difference_abs_mean":float(clean_output_diff.mean()),
        "zero_eligible":not bool(joint.any()),
    }
    return joint,report


def _score_cohort(
    seed,teacher,factors,policy,name,n=N_PER_COHORT,sham_arms=SHAM_ARMS
):
    if name not in COHORT_SEED_OFFSETS:
        raise ValueError("unknown source-only cohort")
    cohort_seed=seed*430000+COHORT_SEED_OFFSETS[name]
    cohort=make_cohort(cohort_seed,n,name)
    batch=cohort.recipient

    with torch.no_grad():
        rec=teacher.rollout(batch)
        donor=teacher.rollout(batch,cohort.donor_events)
        donor_state,raw,phi=recipient_donor_edit(
            teacher,factors,batch,cohort.donor_events,rec)
        if not torch.allclose(donor_state,donor.prequery,atol=1e-6):
            raise AssertionError("last-tag donor must preserve all prior writes")
        eligible,ladder=eligibility_ladder(
            batch,cohort.donor_target,rec,donor)
        physical=factors.weight().detach()
        bpd_rank24,gap=physical_output_subspace_write(phi,physical,RANK)
        bpd_rank24=bpd_rank24.clone()
        bpd_rank24[:,0]=0
        teacher_phys=(phi@teacher.C.detach()).clone()
        teacher_phys[:,0]=0
        exact=(donor_state-rec.prequery).clone()
        exact[:,0]=0
        public=public_query(batch)
        pre=rec.prequery.detach()
        orig_public=teacher.step(pre,public)[:,0]
        don_public=teacher.step(donor.prequery,public)[:,0]
        g=torch.eye(len(factors.U))
        g[0,1]=.7
        sheared=(factors.V.detach()@g)@torch.linalg.solve(g,factors.U.detach())
        gauge_max_error=float((physical-sheared).abs().max())
        if gauge_max_error>2e-5:
            raise AssertionError("physical BPD gauge reconstruction failed")
        zero=torch.zeros_like(raw)
        arms={
            "zero":zero,
            "bpd_full_raw":raw,
            "bpd_rank24_raw":bpd_rank24,
            "teacher_full_raw":teacher_phys,
            "exact_dark_donor":exact,
        }

        policy_correction={}
        for label,proposal in (("bpd_full",raw),("bpd_rank24",bpd_rank24)):
            corrected=predict_from_envelope(policy,{
                "state":pre,"candidate_delta":proposal,
                "public_query":public,"current_readout":pre[:,0:1],
                "future_public_model":object(),
                "future_public_gradient":object(),
                "hidden_donor_gold":object(),
            })
            budget=torch.minimum(
                CORRECTION_FRACTION*proposal.norm(dim=-1),
                torch.full((n,),CORRECTION_MAX_NORM))
            correction=(corrected-proposal).norm(dim=-1)
            violation=float((correction-budget).max())
            if violation>1e-6:
                raise AssertionError(f"nonprivileged correction over budget {violation}")
            arms[label+"_learned"]=corrected
            policy_correction[label]={
                "mean_correction_to_original_dose":float(
                    (correction/proposal.norm(dim=-1).clamp_min(1e-12)).mean()),
                "mean_correction_norm":float(correction.mean()),
                "max_budget_excess":violation,
            }
    oracle_meta={}
    for label,proposal in (("bpd_full",raw),("bpd_rank24",bpd_rank24)):
        repaired=finite_public_repair(
            pre,proposal,lambda state:teacher.step(state,public)[:,0],
            max_iterations=8,tolerance=1e-4)
        arms[label+"_oracle"]=repaired.delta.detach()
        oracle_meta[label]={
            "converged_count":int(repaired.converged.sum()),
            "nonconverged_count":int((~repaired.converged).sum()),
            "mean_public_abs_after":float(repaired.public_error_after.abs().mean()),
            "mean_correction_norm":float(repaired.correction_norm.mean()),
        }
    scores={
        label:_score_arm(teacher,pre,batch.query,public,delta,
                         orig_public,donor.final_output,eligible)
        for label,delta in arms.items()
    }
    # The public cohort is deliberately scored on its entire unfiltered population.
    # Authorized donor retrieval uses the same clean eligible source/model gate.
    nulls={}
    if name!="public_U":
        for label,proposal in (("bpd_full",raw),("bpd_rank24",bpd_rank24)):
            nulls[label]=_shams(
                teacher,pre,batch.query,public,orig_public,
                donor.final_output,eligible,proposal,policy,
                seed*500000+COHORT_SEED_OFFSETS[name]+
                (0 if label=="bpd_full" else 113),n=sham_arms,
            )
    return {
        "stratum":name,
        "cohort_seed":cohort_seed,
        "attempted_candidates":cohort.attempted_candidates,
        "qualified_candidates":cohort.qualified_candidates,
        "retained":cohort.accepted,
        "exact_source_label_change_fraction":cohort.label_change_fraction,
        "recipient_blind_oracle_mse":float(
            (cohort.blind_recipient-batch.target).square().mean()),
        "donor_blind_oracle_mse":float(
            (cohort.blind_donor-cohort.donor_target).square().mean()),
        "ladder":ladder,
        "clean_public_source_tag_change_mean":float(
            (don_public-orig_public).abs().mean()),
        "clean_public_recipient_mae":float(
            (orig_public-batch.public_value).abs().mean()),
        "clean_public_donor_mae":float(
            (don_public-batch.public_value).abs().mean()),
        "factor_physical_mse":float(
            (physical-teacher.C.detach()).square().mean()),
        "physical_gl32_gauge_max_error":gauge_max_error,
        "bpd_rank24_singular_gap":gap,
        "scores":scores,
        "policy_correction":policy_correction,
        "privileged_oracle":oracle_meta,
        "corrected_nulls":nulls,
    }


def evaluate_root(seed, n=N_PER_COHORT,shams=SHAM_ARMS):
    if seed not in EXCLUDED_ROOTS:
        raise ValueError("Stage 2-A only runs excluded development roots 901/902")
    torch.set_num_threads(1)
    teacher=train_teacher(seed,2600)
    factors=train_factors(seed,teacher,450)
    policy,policy_training=train_policy(seed,teacher,factors,240)
    independent_baseline=competence(
        teacher,seed*400000+999000,"heldout",n)
    return {
        "seed":seed,
        "evidence_class":"excluded-root model-level admission preflight",
        "teacher_heldout_admission":{
            **independent_baseline,
            "descriptive_pass":bool(
                independent_baseline["mae"]<=.12
                and independent_baseline["public_mae"]<=.05
                and independent_baseline["mae"]<=.8*independent_baseline["blind_mae"])
        },
        "policy_training":policy_training,
        "cohorts":{
            name:_score_cohort(seed,teacher,factors,policy,name,n,shams)
            for name in COHORT_SEED_OFFSETS
        },
    }


def run(seeds=EXCLUDED_ROOTS,n=N_PER_COHORT,shams=SHAM_ARMS):
    if tuple(seeds)!=EXCLUDED_ROOTS:
        raise ValueError("two excluded roots 901,902 are required in registered order")
    return {
        "experiment":"BPD-SF05B Stage2-A excluded-root source-aware donor admission",
        "roots":list(seeds),"torch_version":torch.__version__,
        "config":{"n_per_cohort":n,"shams_per_kind":shams,
                  "teacher_steps":2600,"factor_steps":450,"policy_steps":240,
                  "source_family":"heldout"},
        "rows":[evaluate_root(seed,n,shams) for seed in seeds],
    }


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--roots",type=int,nargs="+",default=list(EXCLUDED_ROOTS))
    ap.add_argument("--samples",type=int,default=N_PER_COHORT)
    ap.add_argument("--shams",type=int,default=SHAM_ARMS)
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    result=run(tuple(args.roots),args.samples,args.shams)
    print(json.dumps(result,separators=(",",":")))
    if args.out:
        args.out.write_text(json.dumps(result,indent=2)+"\n")
