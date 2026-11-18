"""Independent original SUB-06 receipt audit; no source retraining or refitting."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
ROOTS=list(range(1700,1708))
CLOCK_MAPS=[
    [1,2,3,0],[2,3,0,1],[3,0,1,2],[1,0,3,2],
    [2,1,0,3],[0,2,1,3],[3,2,1,0],
]
STEP_MAPS=[
    [1,2,0],[2,0,1],[0,1,2],[1,0,2],
    [2,1,0],[0,2,1],[2,1,0],
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eq(a,b,tol=1e-8):
    return abs(float(a)-float(b))<=tol


def reconstruct(receipt:dict):
    assert receipt["study"]=="BPD-SUB-06"
    roots=receipt["roots"]
    summary=receipt["summary"]
    assert [r["root"] for r in roots]==ROOTS
    assert len(roots)==summary["attempted_roots"]==8
    assert summary["root_ids"]==ROOTS
    assert summary["registered_primary"]=="min_static_seven_fitted_wrongclock_minus_aligned"
    assert summary["registered_mean_effect_minimum"]==.010
    assert summary["independent_95_95_null_tail_calibrated"] is False
    assert summary["future_calibration"]=="UNGENERATED_SEALED"
    assert summary["future_confirmation"]=="UNGENERATED_SEALED"
    src=digest(ROOT/"experiment.py")
    protocol=digest(ROOT/"PROTOCOL.md")
    admitted=[]; excluded=[]; root_table=[]
    for row in roots:
        if "error_type" in row:
            assert row["root_eligible"] is False
            excluded.append(row["root"])
            root_table.append({"root":row["root"],"eligible":False,
                               "error":row["error_type"]})
            continue
        assert row["study"]=="BPD-SUB-06" and row["phase"]=="ENGINEERING"
        assert row["protocol_sha256"]==protocol
        assert row["source_sha256"]==src
        assert row["training"]["steps"]==256
        assert row["data_sizes"]=={
            "train":2048,"discovery":512,"selection":512,"heldout":1024}
        assert [s["output"] for s in row["wrong_clock_source_schedules"]]==CLOCK_MAPS
        assert [s["transition"] for s in row["wrong_clock_source_schedules"]]==STEP_MAPS
        assert set(row["fitted_source_diagnostics"])==(
            {"aligned"}|{f"wrong_{i}" for i in range(7)})
        assert len(row["fitted_source_diagnostics"])==8
        assert row["stationary_source"]["optimizer_steps"]==80
        for name,fit in row["fitted_source_diagnostics"].items():
            assert fit["optimizer_steps"]==120
            assert fit["optimizer_lr"]==.02
            assert fit["output_weight"]==.35
            assert fit["smooth_weight"]==.50
            assert len(fit["source_loss_trace"])==120
            assert fit["input_future_output_count"]==4
            assert fit["input_transition_count"]==512
            assert eq(fit["normalizer_a0"],row["shared_source_info"]["a0"])
            assert eq(fit["normalizer_b0"],row["shared_source_info"]["b0"])
            assert fit["rank_trace_max_error"]<=1e-4 or not row["source_geometry_ok"]
            assert fit["idempotence_max"]<=1e-5 or not row["source_geometry_ok"]
            assert fit["symmetry_max"]<=1e-5 or not row["source_geometry_ok"]
        assert row["fitted_source_diagnostics"]["aligned"][
            "source_output_schedule"]==[0,1,2,3]
        assert row["fitted_source_diagnostics"]["aligned"][
            "source_step_schedule"]==[0,1,2]
        for i in range(7):
            assert row["fitted_source_diagnostics"][f"wrong_{i}"][
                "source_output_schedule"]==CLOCK_MAPS[i]
            assert row["fitted_source_diagnostics"][f"wrong_{i}"][
                "source_step_schedule"]==STEP_MAPS[i]
        expected_scores=(
            {"aligned","stationary","pca","obs"}
            |{f"wrong_{i}" for i in range(7)}
            |{f"postfit_perm_{i}" for i in range(7)}
            |{f"angular_{i}" for i in range(8)}
        )
        assert set(row["scores"])==expected_scores
        assert len(row["scores"])==26
        assert set(row["students"])==(
            {"aligned","stationary","pca","obs"}
            |{f"wrong_{i}" for i in range(7)})
        assert set(row["dose_checks"])==(
            {"aligned","stationary"}|{f"wrong_{i}" for i in range(7)})
        assert len(row["angular_controls"])==8
        assert [a["index"] for a in row["angular_controls"]]==list(range(8))
        assert [a["theta"] for a in row["angular_controls"][:4]]==[
            .2,.4,.6,.8]
        assert all(a["theta"] is None for a in row["angular_controls"][4:])
        all_geom=all(all(a["geometry"][k]<=1e-6 for k in (
            "orthogonality_max","rank_trace_max_error","idempotence_max",
            "symmetry_max")) for a in row["angular_controls"])
        assert all_geom==row["angular_geometry_ok"]
        dose=all(s["max_physical_dose_error"]<=5e-6 and
                 s["max_null_impulse_projection"]<=5e-6 and
                 eq(s["physical_alpha"],row["alpha"])
                 for s in row["scores"].values())
        dose=dose and all(v<=5e-6 for checks in row["dose_checks"].values()
                          for v in checks)
        assert dose==row["dose_ok"]
        for score in row["scores"].values():
            assert len(score["per_pass_next_state_rms"])==3
            assert len(score["per_pass_terminal_logit_rms"])==3
            assert len(score["per_pass_wrong_repaired"])==3
            assert len(score["per_pass_correct_corrupted"])==3
            assert eq(math.sqrt(sum(x*x for x in score[
                "per_pass_next_state_rms"])/3),score["pooled_next_state_rms"])
            assert eq(math.sqrt(sum(x*x for x in score[
                "per_pass_terminal_logit_rms"])/3),
                score["pooled_terminal_logit_rms"])
        assert set(row["gauge"])==set(row["fitted_source_diagnostics"])|{"stationary"}
        gauge=all(g["clean_logit_error"]<=5e-5 and
                  g["transported_next_state_error"]<=5e-5 and
                  g["transported_logit_error"]<=5e-5 and
                  max(g["wrong_untransported_logit_rms_by_pass"])>1e-5
                  for g in row["gauge"].values())
        assert gauge==row["gauge_ok"]
        competent=min(row["source"]["contexts"].values())>=.80
        assert competent==row["competent"]
        source_geom=all(d["optimizer_steps"]==120 and
            d["idempotence_max"]<=1e-5 and
            d["symmetry_max"]<=1e-5 and
            d["rank_trace_max_error"]<=1e-4
            for d in row["fitted_source_diagnostics"].values())
        source_geom=source_geom and row["stationary_source"][
            "projector_idempotence_max"]<=1e-5
        assert source_geom==row["source_geometry_ok"]
        parity=(row["full_rank_parity"]<=5e-6 and
                row["full_rank_kl"]<=1e-7)
        eligible=bool(competent and source_geom and all_geom and
                      dose and gauge and parity and row["control_roster_ok"])
        assert eligible==row["root_eligible"]
        c=lambda name:row["scores"][name]["pooled_next_state_rms"]
        winner=min([c("stationary")]+[c(f"wrong_{i}") for i in range(7)])
        effect=winner-c("aligned")
        assert eq(winner,row["primary_strongest_fitted_null"])
        assert eq(c("aligned"),row["primary_aligned"])
        assert eq(effect,row["primary_strongest_null_minus_aligned"])
        assert eq(c("stationary")-c("aligned"),row["stationary_minus_aligned"])
        assert eq(min(c(f"wrong_{i}") for i in range(7))-c("aligned"),
                  row["best_wrong_fitted_minus_aligned"])
        assert eq(min(c(f"postfit_perm_{i}") for i in range(7))-
                  c("aligned"),row["postfit_strongest_minus_aligned"])
        if eligible:admitted.append(row)
        else:excluded.append(row["root"])
        root_table.append({
            "root":row["root"],"eligible":eligible,
            "aligned":c("aligned"),"stationary":c("stationary"),
            "best_fitted_wrong":min(c(f"wrong_{i}") for i in range(7)),
            "best_postfit":min(c(f"postfit_perm_{i}") for i in range(7)),
            "strongest_control_minus_aligned":effect,
        })
    assert summary["eligible_roots"]==len(admitted)
    assert summary["ineligible_roots"]==excluded
    effects=[r["primary_strongest_null_minus_aligned"] for r in admitted]
    assert all(eq(x,y) for x,y in zip(effects,summary["root_primary_effects"]))
    assert len(effects)==len(summary["root_primary_effects"])
    mean=float(np.mean(effects)) if effects else None
    if mean is None:assert summary["mean_primary_effect"] is None
    else:assert eq(mean,summary["mean_primary_effect"])
    positives=sum(x>0 for x in effects)
    assert positives==summary["positive_roots"]
    if len(admitted)==8:
        rng=np.random.default_rng(170606)
        samples=rng.choice(np.asarray(effects),
                size=(2000,8),replace=True).mean(1)
        ci=[float(np.quantile(samples,.025)),
            float(np.quantile(samples,.975))]
        assert all(eq(x,y) for x,y in zip(ci,summary["whole_root_bootstrap95"]))
    else:
        assert summary["whole_root_bootstrap95"] is None
        ci=None
    def avg(fn):
        return float(np.mean([fn(r) for r in admitted])) if admitted else None
    kl=avg(lambda r:r["students"]["aligned"]["mean_clean_bernoulli_kl"])
    kl_extra=avg(lambda r:r["students"]["aligned"]["mean_clean_bernoulli_kl"]-
                 r["students"]["stationary"]["mean_clean_bernoulli_kl"])
    nll_extra=avg(lambda r:r["students"]["aligned"]["true_nll"]-
                  r["students"]["stationary"]["true_nll"])
    logit_extra=avg(lambda r:r["scores"]["aligned"]["pooled_terminal_logit_rms"]-
                    r["scores"]["stationary"]["pooled_terminal_logit_rms"])
    for name,val in (
        ("mean_aligned_clean_kl",kl),
        ("mean_aligned_minus_static_clean_kl",kl_extra),
        ("mean_aligned_minus_static_true_nll",nll_extra),
        ("mean_aligned_minus_static_logit_response",logit_extra),
    ):
        if val is None:assert summary[name] is None
        else:assert eq(val,summary[name])
    passed=bool(len(admitted)==8 and mean>=.010 and positives>=7
            and ci[0]>0 and kl<=.040 and kl_extra<=.015
            and nll_extra<=.030 and logit_extra<=.040)
    verdict="ENGINEERING_SIGNAL_ONLY" if passed else "ENGINEERING_NULL_OR_INELIGIBLE"
    assert summary["verdict"]==verdict
    return {
        "study":"BPD-SUB-06",
        "audit":"ORIGINAL_EIGHT_ROOT_READ_ONLY_AUDIT_PASS",
        "registered_verdict":verdict,
        "eligible_roots":len(admitted),
        "root_effects":effects,
        "mean_primary":mean,
        "root_bootstrap95":ci,
        "protocol_sha256":protocol,
        "source_sha256":src,
        "science":"PILOT_ONLY_UNCALIBRATED_SHAM_TAIL",
        "roots":root_table,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    report=reconstruct(json.loads(args.receipt.read_text()))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="roots"},
                     sort_keys=True))


if __name__=="__main__":
    main()
