"""NC-04 independent-root development aggregation, with blocked-arm transparency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechinterp.bpd.experiments.nc02.experiment import root_bootstrap
from .experiment import ROOTS, PROTOCOL_COMMIT, SHAMS

BASES = ("native", "rotated")


def aggregate(receipts: list[dict]) -> dict:
    if len(receipts) != len(ROOTS) or sorted(x["seed"] for x in receipts) != list(ROOTS):
        raise ValueError("NC-04 requires exactly independent roots 400..407")
    for x in receipts:
        if (x["study"] != "BPD-NC-04" or
                x["phase"] != "ENGINEERING" or
                x["protocol_commit"] != PROTOCOL_COMMIT or
                x["data"] != {"train": 2048, "discovery": 1024, "evaluation": 1024} or
                x["training"]["training_steps"] != 384):
            raise ValueError("receipt differs from frozen protocol")
    if len({x["source_sha256"] for x in receipts}) != 1:
        raise ValueError("different implementation hashes")
    if len({x["protocol_sha256"] for x in receipts}) != 1:
        raise ValueError("different protocol hashes")
    receipts = sorted(receipts, key=lambda x: x["seed"])
    output = {
        "study": "BPD-NC-04", "phase": "ENGINEERING",
        "evidence_class": "REGISTERED_DENSE_RECURRENT_DEVELOPMENT_ONLY",
        "n_independent_roots": len(receipts),
        "training_seeds": list(ROOTS),
        "source_sha256": receipts[0]["source_sha256"],
        "protocol_sha256": receipts[0]["protocol_sha256"],
        "protocol_commit": PROTOCOL_COMMIT,
        "model_task_success_roots": [x["seed"] for x in receipts if x["model_task_passed"]],
        "zero_readout_control_success_roots": [
            x["seed"] for x in receipts
            if x["negative_controls"]["zero_readout"]["subspace_status"] == "UNIDENTIFIABLE"
            and x["negative_controls"]["zero_readout"]["max_abs_gradient"] == 0
            and x["negative_controls"]["zero_readout"]["positive_attribution_hits"] == 0
            and not x["negative_controls"]["zero_readout"]["projector_returned"]
        ],
        "bases": {},
    }
    for kind in BASES:
        eligible, excluded = [], []
        for receipt in receipts:
            record = receipt["bases"][kind]
            reasons = []
            if not receipt["model_task_passed"]:
                reasons.append("task_or_dense_weight_gate")
            if not record["closure"]["passed"]:
                reasons.append("full_mask_parity")
            if record["projector_spectrum"].get("status") != "IDENTIFIED_RANK_TWO" if "projector_spectrum" in record else True:
                reasons.append("projector_identifiability")
            if record["status"] != "ENGINEERING":
                reasons.append(record["status"])
            if len(record.get("matched_sham_scores", [])) != SHAMS:
                reasons.append("eight_physical_shams")
            if reasons:
                excluded.append({"seed": receipt["seed"], "reasons": reasons})
            else:
                eligible.append(receipt)
        paired = [r["bases"][kind]["primary_external_nll_advantage_nats"]
                  for r in eligible]
        keep_loss = [r["bases"][kind]["evaluation"]["arms"]["retain"]["nll_change_from_clean_nats"]
                     for r in eligible]
        keep_kl = [r["bases"][kind]["evaluation"]["arms"]["retain"]["final_kl_nats"]
                   for r in eligible]
        necessity = [r["bases"][kind]["evaluation"]["arms"]["ablate"]["nll_change_from_clean_nats"]
                     for r in eligible]
        graph = [
            r["bases"][kind].get("native_style_community_diagnostic",
                                {}).get("cross_pass_community_count",0)
            for r in receipts
        ]
        output["bases"][kind] = {
            "eligible_roots": [r["seed"] for r in eligible],
            "excluded_roots": excluded,
            "n_graph_crosspass_communities_per_root": graph,
            "mean_graph_crosspass_communities": sum(graph) / len(graph),
            "primary_external_nll_advantage": root_bootstrap(paired, seed=64_040),
            "retained_minus_clean_nll": root_bootstrap(keep_loss, seed=64_041),
            "retained_final_clean_model_kl": root_bootstrap(keep_kl, seed=64_042),
            "ablated_minus_clean_nll": root_bootstrap(necessity, seed=64_043),
        }
    equiv = [r["bases"]["rotated"].get("native_rotation_equivariance", {})
             for r in receipts]
    measures = {
        "max_operator_relative_error": "relative_operator_difference",
        "max_retained_logit_abs_delta": "max_kept_logit_difference",
        "max_factor_projector_error": "max_projector_transport_difference",
    }
    output["orthogonal_equivariance"] = {
        target: (max([r[field] for r in equiv])
                 if all(field in r for r in equiv) else None)
        for target, field in measures.items()
    }
    good = (len(output["model_task_success_roots"]) == len(ROOTS)
            and len(output["zero_readout_control_success_roots"]) == len(ROOTS)
            and all(len(output["bases"][k]["eligible_roots"]) == len(ROOTS) for k in BASES)
            and output["orthogonal_equivariance"]["max_operator_relative_error"] is not None
            and output["orthogonal_equivariance"]["max_retained_logit_abs_delta"] is not None
            and output["orthogonal_equivariance"]["max_operator_relative_error"] <=1e-4
            and output["orthogonal_equivariance"]["max_retained_logit_abs_delta"] <=5e-5)
    for kind in BASES:
        report = output["bases"][kind]
        effect = report["primary_external_nll_advantage"]
        bound = effect.get("root_bootstrap_95pct_CI")
        retained = report["retained_minus_clean_nll"].get("mean_effect_nats")
        good = good and bound is not None and bound[0] > 0.02 and retained is not None and retained <=0.05
    output["development_resolution"] = (
        "ENGINEERING_POSITIVE_CONTROL" if good else "NOT_RESOLVED_OR_FAILED")
    output["confirmation"] = "SEALED; no roots registered; EvalMath power preflight required"
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    docs = [json.loads((args.directory / f"nc04_root_{seed}.json").read_text())
            for seed in ROOTS]
    result = aggregate(docs)
    with args.out.open("x", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print("NC-04",result["development_resolution"],"task_success",len(result["model_task_success_roots"]))
    for name, arm in result["bases"].items():
        print(name,"eligible",len(arm["eligible_roots"]),
              "NLL advantage",arm["primary_external_nll_advantage"])
    print("equivariance",result["orthogonal_equivariance"])


if __name__ == "__main__":
    main()
