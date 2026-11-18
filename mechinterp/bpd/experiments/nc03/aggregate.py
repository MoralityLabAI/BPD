"""NC-03 independent-root engineering audit. No confirmation samples or actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechinterp.bpd.experiments.nc02.experiment import root_bootstrap
from .experiment import ROOTS, PROTOCOL_COMMIT, SHAMS

BASES = ("native", "rotated")


def aggregate(receipts: list[dict]) -> dict:
    if sorted(r["seed"] for r in receipts) != list(ROOTS):
        raise ValueError("aggregate requires NC-03 independent roots 300..307 exactly once")
    if len(receipts) != len(ROOTS) or len({r["seed"] for r in receipts}) != len(ROOTS):
        raise ValueError("duplicate or missing independent root")
    for r in receipts:
        if (r["study"] != "BPD-NC-03" or r["phase"] != "ENGINEERING"
                or r["protocol_commit"] != PROTOCOL_COMMIT
                or r["data"] != {"training":2048,"discovery":1024,"evaluation":1024}
                or r["training"]["training_steps"] != 128):
            raise ValueError("data/source registration mismatch")
    if len({r["source_sha256"] for r in receipts}) != 1:
        raise ValueError("mixed experiment implementation hashes")
    if len({r["protocol_sha256"] for r in receipts}) != 1:
        raise ValueError("mixed protocol file hashes")
    rs = sorted(receipts, key=lambda r:r["seed"])
    output = {
        "study":"BPD-NC-03", "phase":"ENGINEERING",
        "evidence_class":"REGISTERED_SYNTHETIC_POSITIVE_CONTROL",
        "scientific_status":"NOT_CONFIRMATORY",
        "n_independent_roots":len(rs), "roots":list(ROOTS),
        "protocol_commit":PROTOCOL_COMMIT,
        "source_sha256":rs[0]["source_sha256"],
        "protocol_sha256":rs[0]["protocol_sha256"],
        "model_task_pass_count":sum(r["task_passed"] for r in rs),
        "zero_readout_null_pass_count":sum(
            r["nulls"]["zero_readout"]["status"]=="NO_IDENTIFIABLE_SUBSPACE"
            and not r["nulls"]["zero_readout"]["projector_returned"]
            and r["nulls"]["zero_readout"]["gradient_max_abs"]==0
            for r in rs
        ),
        "bases": {},
    }
    for kind in BASES:
        eligible=[]
        failures=[]
        for r in rs:
            b=r["bases"][kind]
            reasons=[]
            if not b["closure"]["passed"]:
                reasons.append("parity")
            if b.get("status")!="ENGINEERING":
                reasons.append(str(b.get("status")))
            if b.get("oracle",{}).get("min_overlap",0) < .99:
                reasons.append("oracle_projector_overlap")
            if len(b.get("sham_keep_final_kl",[])) != SHAMS:
                reasons.append("physical_shams")
            if reasons:
                failures.append({"seed":r["seed"],"reasons":reasons})
            else:
                eligible.append(r)
        effects=[r["bases"][kind]["primary_contrast_nats"] for r in eligible]
        nll=[r["bases"][kind]["evaluation"]["arms"]["ablate"]["nll_delta_nats"]
             for r in eligible]
        keep=[r["bases"][kind]["evaluation"]["arms"]["retain"]["final_kl_nats"]
              for r in eligible]
        out={
            "eligible_root_seeds":[r["seed"] for r in eligible],
            "excluded_root_details":failures,
            "root_keep_only_KL_advantage":root_bootstrap(effects,seed=92_751),
            "root_necessity_NLL_increase":root_bootstrap(nll,seed=92_752),
            "root_keep_only_absolute_KL":root_bootstrap(keep,seed=92_753),
            "oracle_min_overlap_across_all_roots":min(
                r["bases"][kind].get("oracle",{}).get("min_overlap",0) for r in rs),
            "mean_valid_sham_candidates":sum(
                r["bases"][kind].get("sham_matching",{}).get("eligible_candidates",
                r["bases"]["native"].get("sham_matching",{}).get("eligible_candidates",0))
                for r in rs)/len(rs),
        }
        output["bases"][kind]=out
    comparisons=[r["bases"]["rotated"].get("comparison_to_native",{}) for r in rs]
    output["orthogonal_equivariance"]={
        "max_physical_map_relative_error":max(
            c.get("projector_map_relative_error",float("inf")) for c in comparisons),
        "max_heldout_retain_logit_delta":max(
            c.get("retain_logit_max_delta",float("inf")) for c in comparisons),
        "max_factor_projector_transport_abs_error":max(
            c.get("factor_projector_transport_error",float("inf")) for c in comparisons),
    }
    n=output["bases"]["native"]
    q=output["bases"]["rotated"]
    good = (
        output["model_task_pass_count"] == 8
        and output["zero_readout_null_pass_count"] == 8
        and len(n["eligible_root_seeds"]) == 8 and len(q["eligible_root_seeds"]) == 8
        and output["orthogonal_equivariance"]["max_physical_map_relative_error"] <=1e-4
        and output["orthogonal_equivariance"]["max_heldout_retain_logit_delta"] <=5e-5
    )
    for arm in (n,q):
        ci=arm["root_keep_only_KL_advantage"].get("root_bootstrap_95pct_CI")
        mean_kept=arm["root_keep_only_absolute_KL"].get("mean_effect_nats")
        good=good and bool(ci) and ci[0] >.02 and mean_kept is not None and mean_kept <=.05
    output["development_resolution"]=(
        "ENGINEERING_POSITIVE_CONTROL" if good else "NOT_RESOLVED_OR_FAILED")
    output["confirmation"]="SEALED; new roots and independent EvalMath preflight required"
    return output


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    docs=[json.loads((a.directory / f"nc03_root_{seed}.json").read_text())
          for seed in ROOTS]
    out=aggregate(docs)
    with a.out.open("x",encoding="utf-8") as f:
        json.dump(out,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("NC-03 independent roots",out["n_independent_roots"])
    print("NC-03 resolution",out["development_resolution"])
    for k in BASES:
        print(k,out["bases"][k]["root_keep_only_KL_advantage"])
    print("equivariance",out["orthogonal_equivariance"])


if __name__=="__main__":
    main()
