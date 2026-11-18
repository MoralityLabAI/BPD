"""Independent read-only auditor of SUB-07 original eight trained-source receipts.

Reconstructs finite oracle proof, group-disjoint population, all fitted controls,
physical and task admissions, strongest-null root effect and root-bootstrap.
Never loads a model checkpoint or changes the registered protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from mechinterp.bpd.experiments.sub07.oracle import (
    exact_task_certificate, split_indices,
)
from mechinterp.bpd.experiments.sub07.source import (
    CLOCK_MAPS, STEP_MAPS, RANK, PASSES, FIT_STEPS, ROOTS,
)

ROOT=Path(__file__).resolve().parent
EXPECTED_NAMES={
    "aligned","stationary",
    *(f"wrong_{i}" for i in range(7)),
    *(f"postfit_perm_{i}" for i in range(7)),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a,b,tol=1e-8):
    return abs(float(a)-float(b))<=tol


def pool(vals):
    assert len(vals)==PASSES-1
    return math.sqrt(sum(v*v for v in vals)/(PASSES-1))


def audit(receipt:dict):
    assert receipt["study"]=="BPD-SUB-07"
    rows=receipt["roots"]
    summary=receipt["summary"]
    assert [r["root"] for r in rows]==list(ROOTS)
    assert summary["attempted_roots"]==8
    assert summary["root_ids"]==list(ROOTS)
    assert summary["registered_primary"]==(
        "min_stationary_seven_source_fitted_wrongclock_minus_aligned")
    assert summary["registered_minimum_effect"]==.010
    assert summary["scientific_sham_tail_calibrated"] is False
    assert summary["future_calibration"]=="UNGENERATED_SEALED"
    assert summary["future_confirmation"]=="UNGENERATED_SEALED"
    cert=exact_task_certificate()
    assert cert["positives"]==2291
    sets=split_indices()
    assert len(set(int(v) for seqs in sets.values() for v in seqs))==4096
    protocol=digest(ROOT/"PROTOCOL.md")
    src=digest(ROOT/"experiment.py")
    oracle_digest=digest(ROOT/"oracle.py")
    fitting_digest=digest(ROOT/"source.py")
    admitted=[]
    ineligible=[]
    table=[]
    for row in rows:
        if "error_type" in row:
            assert row["root_eligible"] is False
            ineligible.append(row["root"])
            table.append({"root":row["root"],"eligible":False,
                          "error_type":row["error_type"]})
            continue
        assert row["study"]=="BPD-SUB-07" and row["phase"]=="ENGINEERING"
        assert row["protocol_sha256"]==protocol
        assert row["source_sha256"]==src
        assert row["oracle_sha256"]==oracle_digest
        assert row["fitting_source_sha256"]==fitting_digest
        assert row["oracle_certificate"]==cert
        assert row["training"]["training_steps"]==1200
        assert row["training"]["batch_size"]==128
        assert row["training"]["train_rows"]==2048
        assert row["data_sizes"]=={
            "train":2048,"discovery":512,"dose":512,
            "heldout":1024,"unique_sequences_full_universe":4096}
        assert row["source_jacobian_rows"]==64
        assert row["clock_maps"]==[list(p) for p in CLOCK_MAPS]
        assert row["step_maps"]==[list(p) for p in STEP_MAPS]
        assert row["stationary_source"]["optimizer_steps"]==80
        assert len(row["stationary_source"]["source_trace"])==80
        fits=row["source_fits"]
        assert set(fits)=={"aligned",*(f"wrong_{i}" for i in range(7))}
        fit_ok=all(
            v["optimizer_steps"]==FIT_STEPS and
            len(v["source_loss_trace"])==FIT_STEPS and
            v["rank_trace_max_error"]<=1e-4 and
            v["idempotence_max"]<=1e-5 and
            v["symmetry_max"]<=1e-5 for v in fits.values()
        ) and row["stationary_source"]["idempotence_max"]<=1e-5
        assert fit_ok==row["source_geometry_ok"]
        for i in range(7):
            assert fits[f"wrong_{i}"]["source_step_schedule"]==list(STEP_MAPS[i])
            assert fits[f"wrong_{i}"]["source_output_schedule"]==list(CLOCK_MAPS[i])
        assert fits["aligned"]["source_output_schedule"]==list(range(PASSES))
        assert fits["aligned"]["source_step_schedule"]==list(range(PASSES-1))
        assert set(row["scores"])==EXPECTED_NAMES
        assert len(row["scores"])==16
        assert set(row["students"])==(
            {"aligned","stationary"}|{f"wrong_{i}" for i in range(7)})
        assert set(row["gauges"])==set(row["students"])
        assert set(row["dose_checks"])==set(row["students"])
        for name,sc in row["scores"].items():
            assert len(sc["per_pass_projected_next_state_rms"])==5
            assert len(sc["per_pass_terminal_logit_rms"])==5
            assert len(sc["per_pass_correct_corrupted"])==5
            assert len(sc["per_pass_wrong_repaired"])==5
            assert len(sc["per_pass_clean_next_state_mean_norm"])==5
            assert len(sc["per_pass_altered_next_state_mean_norm"])==5
            assert close(pool(sc["per_pass_projected_next_state_rms"]),
                         sc["pooled_next_state_rms"])
            assert close(pool(sc["per_pass_terminal_logit_rms"]),
                         sc["pooled_terminal_logit_rms"])
            assert close(sc["physical_alpha"],row["physical_alpha"])
            assert math.isfinite(sc["pooled_next_state_rms"])
        dose_ok=all(
            s["max_physical_dose_error"]<=5e-6 and
            s["max_null_projection_norm"]<=5e-6
            for s in row["scores"].values()
        ) and all(v<=5e-6 for a in row["dose_checks"].values()
                  for v in a)
        assert dose_ok==row["dose_ok"]
        gauge_ok=all(
            g["clean_logit_error"]<=5e-5 and
            g["next_state_transported_max_error"]<=5e-5 and
            g["terminal_logit_transported_max_error"]<=5e-5 and
            max(g["wrong_untransported_logit_rms"])>1e-5
            for g in row["gauges"].values()
        )
        assert gauge_ok==row["gauge_ok"]
        assert row["full_rank_max_logit_error"]<=5e-6 or not row["root_eligible"]
        assert row["full_rank_clean_kl"]<=1e-7 or not row["root_eligible"]
        group=row["source_competence"]["contexts"]
        assert len(group)==6
        competent=(row["source_competence"]["accuracy"]>=.90
                   and min(group.values())>=.85)
        assert competent==row["competent"]
        expected_complete=(len(row["scores"])==16 and
            len(row["students"])==9 and len(row["gauges"])==9 and
            len(row["source_fits"])==8)
        assert expected_complete==row["complete_roster"]
        eligible=bool(competent and expected_complete and fit_ok
            and dose_ok and gauge_ok and
            row["full_rank_max_logit_error"]<=5e-6 and
            row["full_rank_clean_kl"]<=1e-7 and
            row["physical_alpha"]>0)
        assert eligible==row["root_eligible"]
        C=lambda name:row["scores"][name]["pooled_next_state_rms"]
        strongest=min([C("stationary")]+[C(f"wrong_{i}") for i in range(7)])
        effect=strongest-C("aligned")
        assert close(effect,row["primary_strongest_minus_aligned"])
        assert close(strongest,row["strongest_control"])
        assert close(C("aligned"),row["aligned_closure"])
        assert close(C("stationary"),row["stationary_closure"])
        assert close(C("stationary")-C("aligned"),
                     row["stationary_minus_aligned"])
        if eligible:admitted.append(row)
        else:ineligible.append(row["root"])
        table.append({
            "root":row["root"],"eligible":eligible,
            "source_accuracy":row["source_competence"]["accuracy"],
            "aligned":C("aligned"),"stationary":C("stationary"),
            "best_fitted_wrong":min(C(f"wrong_{i}") for i in range(7)),
            "effect":effect,
        })
    assert summary["eligible_roots"]==len(admitted)
    assert summary["ineligible_roots"]==ineligible
    effects=[r["primary_strongest_minus_aligned"] for r in admitted]
    assert len(effects)==len(summary["root_primary_effects"])
    assert all(close(a,b) for a,b in zip(effects,summary["root_primary_effects"]))
    mean=float(np.mean(effects)) if effects else None
    assert close(mean,summary["mean_primary_effect"]) if mean is not None else (
        summary["mean_primary_effect"] is None)
    positive=sum(x>0 for x in effects)
    assert positive==summary["positive_roots"]
    if len(admitted)==8:
        gen=np.random.default_rng(180707)
        sample=gen.choice(np.asarray(effects),size=(2000,8),replace=True).mean(1)
        ci=[float(np.quantile(sample,.025)),
            float(np.quantile(sample,.975))]
        assert all(close(a,b) for a,b in zip(ci,summary["bootstrap95_whole_roots"]))
    else:
        ci=None
        assert summary["bootstrap95_whole_roots"] is None
    avg=lambda f:float(np.mean([f(r) for r in admitted])) if admitted else None
    kl=avg(lambda r:r["students"]["aligned"]["clean_to_student_kl"])
    excess_kl=avg(lambda r:r["students"]["aligned"]["clean_to_student_kl"]-
                  r["students"]["stationary"]["clean_to_student_kl"])
    excess_nll=avg(lambda r:r["students"]["aligned"]["true_label_nll"]-
                   r["students"]["stationary"]["true_label_nll"])
    excess_logit=avg(lambda r:r["scores"]["aligned"]["pooled_terminal_logit_rms"]-
                     r["scores"]["stationary"]["pooled_terminal_logit_rms"])
    for key,value in (
        ("mean_aligned_student_kl",kl),
        ("mean_aligned_minus_static_student_kl",excess_kl),
        ("mean_aligned_minus_static_true_nll",excess_nll),
        ("mean_aligned_minus_static_terminal_response",excess_logit),
    ):
        if value is None:assert summary[key] is None
        else:assert close(value,summary[key])
    passed=bool(len(admitted)==8 and mean>=.010
        and positive>=7 and ci[0]>0 and kl<=.04
        and excess_kl<=.015 and excess_nll<=.030 and excess_logit<=.04)
    expected="ENGINEERING_SIGNAL_ONLY" if passed else "ENGINEERING_NULL_OR_INELIGIBLE"
    assert summary["verdict"]==expected
    return {
        "audit":"SUB07_ORIGINAL_EIGHT_INDEPENDENT_AUDIT_PASS",
        "study":"BPD-SUB-07",
        "original_verdict":expected,
        "eligible_roots":len(admitted),
        "mean_primary_effect":mean,
        "bootstrap95_whole_roots":ci,
        "protocol_sha256":protocol,
        "source_sha256":src,
        "oracle_sha256":oracle_digest,
        "future_science":"UNCHANGED_SEALED",
        "root_table":table,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    audit_result=audit(json.loads(args.receipt.read_text()))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(audit_result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in audit_result.items() if k!="root_table"},
                     sort_keys=True))


if __name__=="__main__":
    main()
