"""Independent read-only audit of the original SUB-03 calibration receipt.

No model training or checkpoint loading. This module checks all independent
root records, physical-dose sham accounting and numerical 95/95 order rank.
It does not change the frozen inference rule when an experiment fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parent
ROOTS=tuple(range(1400,1464))
ANGLES=(.2,.4,.6,.8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a,b,tol=1e-9):
    return abs(a-b)<=tol


def bounded(x):
    return x/(1+abs(x))


def tol_bound(vals):
    n=len(vals)
    if n<59:
        return None,None
    q=Fraction(95,100)
    conf=Fraction(95,100)
    for k in range(1,n+1):
        cdf=sum((Fraction(math.comb(n,j))*q**j*(1-q)**(n-j)
                 for j in range(k)),Fraction(0))
        if cdf>=conf:
            return sorted(vals)[k-1],k
    return None,None


def audit(j):
    roots=j["roots"]
    summary=j["summary"]
    assert [r["root"] for r in roots]==list(ROOTS)
    assert len(roots)==64
    assert summary["attempted_roots"]==64
    assert summary["root_ids"]==list(ROOTS)
    assert summary["study"]=="BPD-SUB-03"
    assert summary["phase"]=="CALIBRATION_DEVELOPMENT"
    source=sha(ROOT/"experiment.py")
    protocol=sha(ROOT/"PROTOCOL.md")
    assert all(r.get("source_sha256",source)==source
               for r in roots if "error_type" not in r)
    assert all(r.get("protocol_sha256",protocol)==protocol
               for r in roots if "error_type" not in r)
    vals=[]
    pilot=[]
    ineligible=[]
    rootdetail=[]
    for r in roots:
        if "error_type" in r:
            assert r["root_eligible"] is False
            ineligible.append(r["root"])
            rootdetail.append({"root":r["root"],"eligible":False,
                "error":r["error_type"]})
            continue
        assert r["training"]["training_steps"]==256
        assert r["data_sizes"]=={"train":2048,"discovery":512,
                                 "calibration":512,"pilot":1024}
        geom=r["obs_geometry"]
        assert geom["finite_difference_max_abs"]<.02 or not r["geometry_admitted"]
        g=r["gauge"]
        gauge_ok=all(g[key]<=5e-5 for key in (
           "max_clean_logit_error",
           "max_transported_impulse_logit_error",
           "max_clean_state_error",
           "max_intervened_state_error")) and (
           g["untransported_impulse_logit_rms"]>1e-5)
        assert bool(r["gauge_admitted"])==bool(gauge_ok)
        assert len(r["calibration"]["shams"])==8
        assert [s["index"] for s in r["calibration"]["shams"]]==list(range(8))
        assert [s["theta_radians"] for s in r["calibration"]["shams"][:4]]==list(ANGLES)
        assert all(s["theta_radians"] is None for s in r["calibration"]["shams"][4:])
        assert [s["name"] for s in r["calibration"]["shams"][:4]]==["local_spectral_rotation"]*4
        assert [s["name"] for s in r["calibration"]["shams"][4:]]==["global_haar_spectral_rotation"]*4
        f_cal=r["calibration"]
        phys_checks=[f_cal["pca"],f_cal["obs"],r["pilot"]["pca"],r["pilot"]["obs"]]
        for s in f_cal["shams"]:
            phys_checks.append(s["fiber"])
            reference=f_cal["pca"]["terminal_logit_rms"]
            response=s["fiber"]["terminal_logit_rms"]
            assert close(s["signed_pca_minus_sham_logit_rms"],reference-response)
            assert close(s["positive_false_improvement_logit_rms"],max(0,reference-response))
            assert close(s["bounded_sham_effect"],
                         bounded(max(0,reference-response)))
            geom=s["geometry"]
            assert geom["eigenvalue_law_max_error"]<1e-9 or not r["geometry_admitted"]
            assert geom["trace_error"]<1e-6 or not r["geometry_admitted"]
        dose_ok=all(f["max_dose_deviation"]<=5e-6
                    and f["max_null_projection"]<=5e-6
                    and close(f["physical_dose"],r["physical_alpha"])
                    for f in phys_checks)
        assert dose_ok==r["dose_admitted"]
        competent=min(r["contexts"].values())>=.8
        assert competent==r["competent"]
        eligible=bool(competent and gauge_ok
                      and r["dose_admitted"] and r["geometry_admitted"])
        assert eligible==r["root_eligible"]
        recorded=max(s["positive_false_improvement_logit_rms"]
                     for s in f_cal["shams"])
        assert close(recorded,f_cal["worst_positive_null_improvement_logit_rms"])
        effect=(r["pilot"]["pca"]["terminal_logit_rms"]-
                r["pilot"]["obs"]["terminal_logit_rms"])
        assert close(effect,r["pilot"]["pca_minus_obs_logit_rms"])
        assert close(bounded(effect),r["pilot"]["bounded_effect"])
        if eligible:
            vals.append(bounded(recorded))
            pilot.append(effect)
        else:ineligible.append(r["root"])
        rootdetail.append({"root":r["root"],"eligible":eligible,
                           "pilot_raw_logit_rms":effect,
                           "worst_positive_null_raw_logit_rms":recorded})
    assert summary["eligible_roots"]==len(vals)
    assert summary["ineligible_roots"]==ineligible
    assert summary["at_least_59_eligible"]==(len(vals)>=59)
    if len(vals)>=59:
        b,rank=tol_bound(vals)
        assert rank==summary["order_statistic_rank"]
        assert close(b,summary["root_level_95_95_bounded_sham_tail"])
        assert close(b/(1-b),summary["root_level_95_95_physical_sham_tail"])
        assert summary["status"]=="CALIBRATION_SUFFICIENT_FOR_PREFLIGHT"
    else:
        assert summary["status"]=="CALIBRATION_INSUFFICIENT"
        assert summary["root_level_95_95_bounded_sham_tail"] is None
    if pilot:
        assert close(mean(pilot),summary["pilot_mean_raw_logit_rms"])
    return {"study":"BPD-SUB-03",
            "decision":"ORIGINAL_CALIBRATION_RECEIPT_AUDIT_PASS",
            "attempted_roots":64,"eligible":len(vals),
            "tail_calibrated":len(vals)>=59,
            "physical_tail":summary["root_level_95_95_physical_sham_tail"],
            "mean_raw_pilot_effect":summary["pilot_mean_raw_logit_rms"],
            "source_sha256":source,"protocol_sha256":protocol,
            "roots":rootdetail}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--report",type=Path)
    a=p.parse_args()
    receipt=json.loads(a.receipt.read_text())
    out=audit(receipt)
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({key:out[key] for key in out if key!="roots"},sort_keys=True))


if __name__=="__main__":
    main()
