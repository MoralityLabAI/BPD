"""NC-05 independent-root rank-dose frontier aggregation.

Eight model roots, eight rank/gain cells, root bootstrap with a Bonferroni
familywise interval. A scientific failure is a complete usable receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .experiment import RANKS, GAINS, ROOTS, SHAMS, PROTOCOL_COMMIT

BOOTSTRAPS = 4096
FAMILY = len(RANKS) * len(GAINS)
BONFERRONI_ALPHA = 0.05 / FAMILY


def family_bootstrap(values: list[float], seed: int) -> dict:
    if not values:
        return {"n_independent_roots": 0, "status": "NO_ELIGIBLE_ROOTS",
                "familywise_ci": None, "mean_effect_nats": None}
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("nonfinite root-level effect")
    g = np.random.default_rng(seed)
    resamples = g.integers(0, len(data), size=(BOOTSTRAPS, len(data)))
    boot_means = data[resamples].mean(axis=1)
    low, high = np.quantile(boot_means, [BONFERRONI_ALPHA/2, 1-BONFERRONI_ALPHA/2])
    return {
        "n_independent_roots": len(data),
        "mean_effect_nats": float(data.mean()),
        "familywise_ci": [float(low), float(high)],
        "familywise_coverage": 1-BONFERRONI_ALPHA,
        "within_root_effects": [float(x) for x in data],
        "evidence_class": "ENGINEERING_ONLY",
    }


def aggregate(receipts: list[dict]) -> dict:
    if len(receipts) != len(ROOTS) or sorted(x["seed"] for x in receipts) != list(ROOTS):
        raise ValueError("exactly eight distinct engineering roots 500..507 required")
    if any(
        x["study"] != "BPD-NC-05" or x["phase"] != "ENGINEERING"
        or x["protocol_commit"] != PROTOCOL_COMMIT
        or x["data"] != {"training": 2048, "discovery": 1024, "evaluation": 1024}
        or x["training"]["training_steps"] != 384 for x in receipts
    ):
        raise ValueError("a root violates the preregistered source/data contract")
    if len({x["source_sha256"] for x in receipts}) != 1:
        raise ValueError("mixed run-source SHA-256")
    if len({x["protocol_sha256"] for x in receipts}) != 1:
        raise ValueError("mixed protocol SHA-256")
    ordered = sorted(receipts, key=lambda x: x["seed"])
    summary = {
        "study": "BPD-NC-05", "phase": "ENGINEERING",
        "evidence_class": "PROSPECTIVELY_REGISTERED_SYNTHETIC_DEVELOPMENT",
        "scientific_status": "NOT_CONFIRMATORY",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_sha256": ordered[0]["source_sha256"],
        "protocol_sha256": ordered[0]["protocol_sha256"],
        "seeds": list(ROOTS),
        "n_independent_training_roots": len(ordered),
        "ranks": list(RANKS), "gains": list(GAINS),
        "multiplicity_family": FAMILY,
        "bootstrap_resamples": BOOTSTRAPS,
        "familywise_per_cell_coverage": 1 - BONFERRONI_ALPHA,
        "task_success_roots": [x["seed"] for x in ordered if x["task_passed"]],
        "zero_head_success_roots": [
            x["seed"] for x in ordered
            if x["negative_controls"]["zero_head"]["max_abs_grad"] == 0.
            and x["negative_controls"]["zero_head"]["positive_hits"] == 0
            and x["negative_controls"]["zero_head"]["all_ranks_unidentifiable"]
        ],
        "cells": {},
    }
    clean_accuracy = [
        x.get("clean", {}).get("accuracy_by_pass") for x in ordered
    ]
    if all(a is not None for a in clean_accuracy):
        summary["mean_clean_accuracy_by_pass"] = [
            float(np.mean([row[t] for row in clean_accuracy])) for t in range(4)
        ]
    for rank in RANKS:
        for alpha in GAINS:
            key=f"r{rank}_g{alpha}"
            eligible = []
            excluded = []
            for root in ordered:
                why=[]
                if not root["task_passed"]:
                    why.append("task_density_gate")
                if root["seed"] not in summary["zero_head_success_roots"]:
                    why.append("zero_head_negative_control")
                if not all(root["bases"][b]["closure"]["passed"] for b in ("native","rotated")):
                    why.append("full_ones_parity")
                if not all(root["bases"][b].get("rank_diagnostics",{}).get(str(rank),{}).get("status") == "IDENTIFIABLE"
                           for b in ("native","rotated")):
                    why.append("eigenspace")
                rdoc=root["cells"].get(str(rank),{})
                match=rdoc.get("sham_selection") or {}
                if not match.get("eligible") or len(match.get("sham_proposal_indices",[])) != SHAMS:
                    why.append("discovery_dose_shams")
                gains=rdoc.get("gains",{}).get(str(alpha),{})
                native=gains.get("native",{})
                rotated=gains.get("rotated",{})
                if native.get("status") != "ENGINEERING" or rotated.get("status") != "ENGINEERING":
                    why.append("closed_loop_evaluation")
                elif (len(native.get("shams",[])) != SHAMS or len(rotated.get("shams",[])) != SHAMS):
                    why.append("eight_scored_shams")
                comparison=root["bases"]["rotated"].get("rank_diagnostics",{}).get(str(rank),{}).get("rotation_comparison",{})
                if comparison.get("operator_relative_error") is None or comparison.get("operator_relative_error",1)>1e-4:
                    why.append("physical_operator_rotation")
                if (rotated.get("rotation_max_kept_logit_delta") is None
                        or rotated.get("rotation_max_kept_logit_delta",1)>5e-5):
                    why.append("heldout_output_rotation")
                if why:
                    excluded.append({"seed":root["seed"],"reasons":why})
                else:
                    eligible.append(root)
            # Use only native scores as the primary statistical contrast. Rotated
            # factor coordinates test numerical equivariance on the same roots.
            delta_fidelity=[
                z["cells"][str(rank)]["gains"][str(alpha)]["native"]["retained"]["nll_delta_from_clean_nats"]
                for z in eligible
            ]
            delta_sham=[
                z["cells"][str(rank)]["gains"][str(alpha)]["native"]["sham_adjusted_external_nll_nats"]
                for z in eligible
            ]
            clean_kl=[
                z["cells"][str(rank)]["gains"][str(alpha)]["native"]["retained"]["final_kl_nats"]
                for z in eligible
            ]
            acc=[
                z["cells"][str(rank)]["gains"][str(alpha)]["native"]["retained"]["accuracy"]
                for z in eligible
            ]
            ci_fid=family_bootstrap(delta_fidelity, 70_000 + rank*100 + int(alpha*100))
            ci_sham=family_bootstrap(delta_sham, 80_000 + rank*100 + int(alpha*100))
            decision=(len(eligible)==len(ROOTS)
                      and ci_fid["familywise_ci"] is not None
                      and ci_sham["familywise_ci"] is not None
                      and ci_fid["familywise_ci"][1] <= 0.05
                      and ci_sham["familywise_ci"][0] > 0.02)
            summary["cells"][key] = {
                "rank":rank, "gain":alpha,
                "eligible_roots":[z["seed"] for z in eligible],
                "excluded_roots":excluded,
                "fidelity_nll_cost":ci_fid,
                "sham_adjusted_nll_advantage":ci_sham,
                "mean_clean_to_keep_KL_nats":float(np.mean(clean_kl)) if clean_kl else None,
                "mean_retained_accuracy":float(np.mean(acc)) if acc else None,
                "engineering_joint_gate_passed":bool(decision),
                "rootwise_qualified_sham_proposal_counts":{
                    str(z["seed"]): z["cells"][str(rank)]["sham_selection"]["qualifying_candidates"]
                    for z in ordered if z["cells"].get(str(rank),{}).get("sham_selection")
                },
            }
    passing=[rank for rank in RANKS if summary["cells"][f"r{rank}_g1.0"]["engineering_joint_gate_passed"]]
    summary["smallest_qualifying_full_gain_rank"]= min(passing) if passing else None
    summary["development_resolution"]=(
        "ENGINEERING_RANK_DOSE_FRONTIER" if passing else "NOT_RESOLVED_OR_FAILED"
    )
    summary["confirmation"]="SEALED; fresh roots + independent EvalMath preflight required"
    return summary


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--directory",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    receipts=[json.loads((a.directory / f"nc05_root_{seed}.json").read_text())
              for seed in ROOTS]
    result=aggregate(receipts)
    with a.out.open("x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print("NC-05 independent roots",result["n_independent_training_roots"])
    print("NC-05 development outcome",result["development_resolution"])
    print("NC-05 smallest full-gain qualifying rank",result["smallest_qualifying_full_gain_rank"])
    for key,d in result["cells"].items():
        print(key, "eligible",len(d["eligible_roots"]),
              "fidelity",d["fidelity_nll_cost"]["mean_effect_nats"],
              "advantage",d["sham_adjusted_nll_advantage"]["mean_effect_nats"],
              "gate",d["engineering_joint_gate_passed"])


if __name__=="__main__":
    main()
