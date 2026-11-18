"""SUB-04 independent source-read-only auditor for an original 8-root receipt.

No training, no projector fitting, and no new evaluation. Validates per-pass
counterfactual accounting, root inference, all fixed controls and gate logic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
ROOTS=list(range(1500,1508))
ANGLES=[.20,.40,.60,.80]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a,b,tol=1e-8):
    return abs(a-b)<=tol


def rms_three(xs):
    assert len(xs)==3
    return math.sqrt(sum(v*v for v in xs)/3.)


def means(rows,key):
    return float(np.mean([r[key] for r in rows])) if rows else None


def audit(receipt:dict):
    rows=receipt["roots"]
    summary=receipt["summary"]
    assert [r["root"] for r in rows]==ROOTS
    assert len(rows)==summary["attempted_roots"]==8
    assert summary["study"]=="BPD-SUB-04"
    assert summary["phase"]=="ENGINEERING"
    assert summary["root_ids"]==ROOTS
    assert summary["independent_sham_tail_calibrated"] is False
    assert summary["future_confirmation"]=="UNGENERATED_SEALED"
    source=sha(ROOT/"experiment.py")
    protocol=sha(ROOT/"PROTOCOL.md")
    eligible=[]
    ineligible=[]
    allarms=[]
    for r in rows:
        if "error_type" in r:
            assert r["root_eligible"] is False
            ineligible.append(r["root"])
            continue
        assert r["study"]=="BPD-SUB-04" and r["phase"]=="ENGINEERING"
        assert r["source_sha256"]==source
        assert r["protocol_sha256"]==protocol
        assert r["training"]["steps"]==256
        assert r["data_sizes"]=={
            "train":2048,"discovery":512,"selection":512,"heldout":1024,
        }
        assert set(r["methods"])=={"joint","obs","pca"}
        assert len(r["shams"])==8
        assert [s["index"] for s in r["shams"]]==list(range(8))
        assert [s["theta"] for s in r["shams"][:4]]==ANGLES
        assert all(s["theta"] is None for s in r["shams"][4:])
        assert [s["type"] for s in r["shams"][:4]]==["local_joint_rotation"]*4
        assert [s["type"] for s in r["shams"][4:]]==["global_joint_rotation"]*4
        assert len(r["geometry"]["joint"]["source_loss_trace"])==80
        for key in ("projector_idempotence_max","projector_symmetry_max"):
            assert r["geometry"]["joint"][key]<=1e-5 or not r["projector_ok"]
        assert r["geometry"]["joint"]["projector_trace_error"]<=1e-4 or not r["projector_ok"]
        geometry_ok=all(all(
            s["geometry"][key]<=1e-6
            for key in ("orthogonality_max","projector_idempotence_max",
                        "projector_symmetry_max","projector_trace_error",
                        "artificial_spectrum_difference_max"))
            for s in r["shams"])
        assert geometry_ok==r["sham_geometry_ok"]
        assert set(r["check_doses"])=={"joint","obs","pca"}
        all_scored=[
            *(v["transition"] for v in r["methods"].values()),
            *(v["transition"] for v in r["shams"]),
        ]
        for scored in all_scored:
            assert len(scored["per_pass_projected_next_state_euclidean_rms"])==3
            assert len(scored["per_pass_terminal_logit_rms"])==3
            assert len(scored["per_pass_terminal_state_displacement_euclidean_rms"])==3
            assert len(scored["per_pass_correct_corrupted_count"])==3
            assert len(scored["per_pass_wrong_repaired_count"])==3
            assert close(
                rms_three(scored["per_pass_projected_next_state_euclidean_rms"]),
                scored["pooled_projected_next_state_euclidean_rms"])
            assert close(rms_three(scored["per_pass_terminal_logit_rms"]),
                         scored["pooled_terminal_logit_rms"])
            assert close(scored["alpha"],r["alpha"])
            assert all(math.isfinite(float(z)) for z in (
                scored["pooled_projected_next_state_euclidean_rms"],
                scored["pooled_terminal_logit_rms"],
                scored["max_physical_dose_error"],
                scored["max_null_projection_norm"]))
        dose_ok=all(z["max_physical_dose_error"]<=5e-6
                    and z["max_null_projection_norm"]<=5e-6
                    for z in all_scored) and all(
                    v<=5e-6 for v in r["check_doses"].values())
        assert dose_ok==r["dose_ok"]
        g=r["gauge"]
        gauge_ok=(all(g[k]<=5e-5 for k in (
            "clean_logit_max_error","clean_state_max_error",
            "intervention_logit_max_error","intervention_state_max_error"
            )) and max(g["wrong_untransported_logit_rms_by_pass"])>1e-5)
        assert gauge_ok==r["gauge_ok"]
        competent=min(r["source"]["contexts"].values())>=.80
        qualified=bool(competent and r["projector_ok"] and geometry_ok
                       and dose_ok and gauge_ok)
        assert qualified==r["root_eligible"]
        primary=(r["methods"]["obs"]["transition"]
                    ["pooled_projected_next_state_euclidean_rms"]
                -r["methods"]["joint"]["transition"]
                    ["pooled_projected_next_state_euclidean_rms"])
        near=(r["shams"][0]["transition"]
                 ["pooled_projected_next_state_euclidean_rms"]
                -r["methods"]["joint"]["transition"]
                 ["pooled_projected_next_state_euclidean_rms"])
        assert close(primary,r["primary_obs_minus_joint_state_rms"])
        assert close(near,r["direct_near_joint_sham_minus_joint_state_rms"])
        if qualified:
            eligible.append(r)
        else:
            ineligible.append(r["root"])
        allarms.append({"root":r["root"],"eligible":qualified,
            "primary":primary,"near_joint":near,
            "obs_closure":r["methods"]["obs"]["transition"]
                ["pooled_projected_next_state_euclidean_rms"],
            "joint_closure":r["methods"]["joint"]["transition"]
                ["pooled_projected_next_state_euclidean_rms"]})
    assert summary["eligible_roots"]==len(eligible)
    assert summary["ineligible_roots"]==ineligible
    effects=[r["primary_obs_minus_joint_state_rms"] for r in eligible]
    assert len(effects)==len(summary["root_primary_effects"])
    assert all(close(a,b) for a,b in zip(effects,summary["root_primary_effects"]))
    assert summary["positive_roots"]==sum(v>0 for v in effects)
    def meanroot(fn):
        return float(np.mean([fn(r) for r in eligible])) if eligible else None
    mean_p=meanroot(lambda r:r["primary_obs_minus_joint_state_rms"])
    if mean_p is None:
        assert summary["mean_primary_effect"] is None
    else:
        assert close(mean_p,summary["mean_primary_effect"])
    mean_kl=meanroot(lambda r:r["methods"]["joint"]["fidelity"]
                     ["mean_clean_to_projected_kl"])
    mean_kl_diff=meanroot(lambda r:r["methods"]["joint"]["fidelity"]
                          ["mean_clean_to_projected_kl"]
                          -r["methods"]["obs"]["fidelity"]
                          ["mean_clean_to_projected_kl"])
    mean_nll_diff=meanroot(lambda r:r["methods"]["joint"]["fidelity"]
                           ["true_label_nll"]
                           -r["methods"]["obs"]["fidelity"]
                           ["true_label_nll"])
    mean_logit_diff=meanroot(lambda r:r["methods"]["joint"]["transition"]
                             ["pooled_terminal_logit_rms"]
                             -r["methods"]["obs"]["transition"]
                             ["pooled_terminal_logit_rms"])
    for field,val in (("mean_joint_clean_kl",mean_kl),
        ("mean_joint_minus_obs_clean_kl",mean_kl_diff),
        ("mean_joint_minus_obs_true_nll",mean_nll_diff),
        ("mean_joint_minus_obs_terminal_logit_rms",mean_logit_diff)):
        if val is None:assert summary[field] is None
        else:assert close(summary[field],val)
    if len(eligible)==8:
        seed=150404
        gen=np.random.default_rng(seed)
        a=np.asarray(effects)
        resampled=gen.choice(a,size=(2000,8),replace=True).mean(1)
        ci=[float(np.quantile(resampled,.025)),
            float(np.quantile(resampled,.975))]
        assert all(close(a,b) for a,b in zip(ci,summary["bootstrap95_whole_root"]))
    else:
        assert summary["bootstrap95_whole_root"] is None
        ci=None
    signal=bool(len(eligible)==8 and mean_p>=.020
                and sum(v>0 for v in effects)>=7 and ci[0]>0
                and mean_kl<=.040 and mean_kl_diff<=.015
                and mean_nll_diff<=.030 and mean_logit_diff<=.040)
    assert summary["minimum_mechanistic_effect"]==.020
    assert summary["verdict"]==(
        "ENGINEERING_SIGNAL_ONLY" if signal
        else "ENGINEERING_NULL_OR_INELIGIBLE")
    return {
        "study":"BPD-SUB-04","receipt_audit":"ORIGINAL_EIGHT_ROOT_INDEPENDENT_AUDIT_PASS",
        "independent_roots":8,"eligible":len(eligible),
        "registered_verdict":summary["verdict"],
        "primary_mean_euclidean_rms":mean_p,
        "whole_root_bootstrap95":ci,
        "source_sha256":source,"protocol_sha256":protocol,
        "strict_evalmath":"PENDING_SEPARATE_PREFLIGHT",
        "detail":allarms,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    result=audit(json.loads(args.receipt.read_text()))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="detail"},sort_keys=True))


if __name__=="__main__":
    main()
