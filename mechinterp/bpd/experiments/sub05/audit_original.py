"""Independent read-only audit for original SUB-05 eight-model engineering data.

Audit reconstructs all root-level controls, strongest-negative-control selection,
physical dose and gauge admission, primary bootstrap and registered fidelity gates.
Never retrains models or selects alternative controls after observing outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
ROOTS=list(range(1600,1608))
ANGLES=[.2,.4,.6,.8]
PERMUTATIONS=[
    [1,2,3,0],[2,3,0,1],[3,0,1,2],[1,0,3,2],
    [2,1,0,3],[0,2,1,3],[3,2,1,0],
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(x,y,tol=1e-8):
    return abs(x-y)<tol


def score_rms(xs):
    assert len(xs)==3
    return math.sqrt(sum(x*x for x in xs)/3)


def audit(receipt:dict):
    rows=receipt["roots"]
    stats=receipt["summary"]
    assert receipt["study"]=="BPD-SUB-05"
    assert [r["root"] for r in rows]==ROOTS
    assert stats["attempted_roots"]==8
    assert stats["root_ids"]==ROOTS
    assert stats["study"]=="BPD-SUB-05"
    assert stats["phase"]=="ENGINEERING"
    assert stats["scientific_sham_tail_calibrated"] is False
    assert stats["future_confirmation"]=="UNGENERATED_SEALED"
    assert stats["registered_primary"]==(
        "min_stationary_all_permutations_minus_aligned_state_rms")
    assert stats["registered_minimum"]==.015
    source=digest(ROOT/"experiment.py")
    protocol=digest(ROOT/"PROTOCOL.md")
    eligible=[]
    excluded=[]
    root_table=[]
    for row in rows:
        if "error_type" in row:
            assert row["root_eligible"] is False
            excluded.append(row["root"])
            root_table.append({"root":row["root"],"eligible":False,
                               "error":row["error_type"]})
            continue
        assert row["study"]=="BPD-SUB-05"
        assert row["phase"]=="ENGINEERING"
        assert row["protocol_sha256"]==protocol
        assert row["source_sha256"]==source
        assert row["training"]["training_steps"]==256
        assert row["data_sizes"]=={
            "train":2048,"discovery":512,"selection":512,"heldout":1024}
        assert row["temporal_permutations"]==PERMUTATIONS
        assert len(row["angular_shams"])==8
        assert [z["index"] for z in row["angular_shams"]]==list(range(8))
        assert [z["theta"] for z in row["angular_shams"][:4]]==ANGLES
        assert all(z["theta"] is None for z in row["angular_shams"][4:])
        assert all(z["kind"]=="local_angular" for z in row["angular_shams"][:4])
        assert all(z["kind"]=="shared_global_haar" for z in row["angular_shams"][4:])
        assert row["clock_source"]["optimizer_steps"]==120
        assert len(row["clock_source"]["trace"])==120
        assert row["static_source"]["optimizer_steps"]==80
        assert len(row["static_source"]["source_loss_trace"])==80
        assert row["clock_source"]["projector_trace_max_error"]<=1e-4 or not row["geometry_ok"]
        assert row["clock_source"]["projector_symmetry_max"]<=1e-5 or not row["geometry_ok"]
        assert row["clock_source"]["projector_idempotence_max"]<=1e-5 or not row["geometry_ok"]
        family=set(row["scores"])
        assert family==(
            {"aligned","stationary"}
            | {"time_perm_"+str(i) for i in range(7)}
            | {"angular_"+str(i) for i in range(8)}
        )
        assert set(row["students"])=={"aligned","stationary"}
        assert set(row["calibration_max_dose_errors"])=={"aligned","stationary"}
        dose_ok=all(
            sc["max_physical_dose_error"]<=5e-6
            and sc["max_null_impulse_projection"]<=5e-6
            and near(sc["physical_alpha"],row["alpha"])
            for sc in row["scores"].values()
        ) and all(x<=5e-6 for vals in
                   row["calibration_max_dose_errors"].values()
                   for x in vals)
        assert dose_ok==row["dose_ok"]
        for value in row["scores"].values():
            assert len(value["per_pass_next_state_rms"])==3
            assert len(value["per_pass_terminal_logit_rms"])==3
            assert len(value["per_pass_correct_corrupted"])==3
            assert len(value["per_pass_wrong_repaired"])==3
            assert near(score_rms(value["per_pass_next_state_rms"]),
                        value["pooled_next_state_rms"])
            assert near(score_rms(value["per_pass_terminal_logit_rms"]),
                        value["pooled_terminal_logit_rms"])
            assert math.isfinite(value["pooled_next_state_rms"])
        geometry_ok=all(
            all(sham["geometry"][key]<=1e-6 for key in (
                "orthogonality_max","rank_trace_max_error",
                "idempotence_max","symmetry_max"))
            for sham in row["angular_shams"]
        ) and (
            row["clock_source"]["projector_idempotence_max"]<1e-5
            and row["clock_source"]["projector_symmetry_max"]<1e-5
            and row["clock_source"]["projector_trace_max_error"]<1e-4
            and row["static_source"]["projector_idempotence_max"]<1e-5
        )
        assert geometry_ok==row["geometry_ok"]
        gauge=row["gauge"]
        gauge_ok=(
            max(gauge[k] for k in (
                "clean_logit_error","transported_next_state_error",
                "transported_logit_error"))<=5e-5
            and max(gauge["wrong_untransported_logit_rms_by_pass"])>1e-5)
        assert gauge_ok==row["gauge_ok"]
        assert row["full_rank_logit_max_error"]<=5e-6 or not row["root_eligible"]
        competent=min(row["source_context_accuracy"].values())>=.80
        valid=bool(competent and geometry_ok and dose_ok and gauge_ok
                   and row["full_rank_logit_max_error"]<=5e-6
                   and row["full_rank_student_kl"]<=1e-7)
        assert valid==row["root_eligible"]
        aligned=row["scores"]["aligned"]["pooled_next_state_rms"]
        controls=[row["scores"]["stationary"]["pooled_next_state_rms"]]+[
            row["scores"]["time_perm_"+str(i)]["pooled_next_state_rms"]
            for i in range(7)]
        assert near(min(controls),row["strongest_control_closure"])
        assert near(aligned,row["aligned_closure"])
        assert near(min(controls)-aligned,
                    row["primary_strongest_null_minus_aligned"])
        assert near(controls[0]-aligned,row["stationary_minus_aligned"])
        assert near(min(controls[1:])-aligned,
                    row["strongest_permutation_minus_aligned"])
        if valid:eligible.append(row)
        else:excluded.append(row["root"])
        root_table.append({
            "root":row["root"],"eligible":valid,
            "aligned_closure":aligned,
            "static_closure":controls[0],
            "best_time_permutation_closure":min(controls[1:]),
            "strongest_null_gain":min(controls)-aligned,
        })
    assert len(eligible)==stats["eligible_roots"]
    assert excluded==stats["ineligible_roots"]
    effects=[r["primary_strongest_null_minus_aligned"] for r in eligible]
    assert all(near(x,y) for x,y in zip(effects,stats["root_primary_effects"]))
    assert len(effects)==len(stats["root_primary_effects"])
    mean=float(np.mean(effects)) if effects else None
    assert near(mean,stats["mean_primary_effect"]) if mean is not None else (
        stats["mean_primary_effect"] is None)
    assert stats["positive_root_count"]==sum(x>0 for x in effects)
    if len(eligible)==8:
        rng=np.random.default_rng(160505)
        draws=rng.choice(np.array(effects),size=(2000,8),replace=True).mean(1)
        ci=[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]
        assert all(near(x,y) for x,y in zip(ci,stats["bootstrap95_whole_roots"]))
    else:
        ci=None
        assert stats["bootstrap95_whole_roots"] is None
    def rm(fn):
        return float(np.mean([fn(r) for r in eligible])) if eligible else None
    kl=rm(lambda r:r["students"]["aligned"]["mean_clean_bernoulli_kl"])
    kl_diff=rm(lambda r:r["students"]["aligned"]["mean_clean_bernoulli_kl"]
                 -r["students"]["stationary"]["mean_clean_bernoulli_kl"])
    nll_diff=rm(lambda r:r["students"]["aligned"]["true_nll"]
                  -r["students"]["stationary"]["true_nll"])
    logit_diff=rm(lambda r:r["scores"]["aligned"]["pooled_terminal_logit_rms"]
                    -r["scores"]["stationary"]["pooled_terminal_logit_rms"])
    for name,val in (
        ("mean_aligned_clean_kl",kl),
        ("mean_aligned_minus_static_clean_kl",kl_diff),
        ("mean_aligned_minus_static_true_nll",nll_diff),
        ("mean_aligned_minus_static_terminal_logit_rms",logit_diff),
    ):
        if val is None:assert stats[name] is None
        else:assert near(val,stats[name])
    positive=sum(v>0 for v in effects)
    passed=bool(len(eligible)==8 and mean>=.015 and positive>=7
                and ci[0]>0 and kl<=.040 and kl_diff<=.015
                and nll_diff<=.030 and logit_diff<=.040)
    assert stats["verdict"]==(
        "ENGINEERING_SIGNAL_ONLY" if passed
        else "ENGINEERING_NULL_OR_INELIGIBLE")
    return {
        "study":"BPD-SUB-05",
        "status":"ORIGINAL_EIGHT_ROOT_INDEPENDENT_AUDIT_PASS",
        "attempted_roots":8,
        "eligible_roots":len(eligible),
        "registered_verdict":stats["verdict"],
        "mean_primary_effect":mean,
        "bootstrap95_whole_roots":ci,
        "source_sha256":source,
        "protocol_sha256":protocol,
        "future_confirmation":"UNGENERATED_SEALED",
        "roots":root_table,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    receipt=json.loads(args.receipt.read_text())
    answer=audit(receipt)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(answer,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in answer.items() if k!="roots"},sort_keys=True))


if __name__=="__main__":
    main()
