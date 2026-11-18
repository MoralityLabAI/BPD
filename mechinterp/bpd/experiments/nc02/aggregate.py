"""Aggregate NC-02 development *independent-root* receipts; no confirmation API."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from .experiment import root_bootstrap, PROTOCOL_COMMIT

ROOTS = tuple(range(200,208))
BASES = ("native", "bpd_svd", "rotated")


def aggregate(receipts: list[dict]) -> dict:
    seeds = [r["seed"] for r in receipts]
    if sorted(seeds) != list(ROOTS) or len(set(seeds)) != 8:
        raise ValueError("NC-02 aggregate requires exactly independent engineering roots 200..207")
    if any(r["study"] != "BPD-NC-02" or r["phase"] != "ENGINEERING" or
           r["protocol_commit"] != PROTOCOL_COMMIT or
           r["data"] != {"training":2048,"discovery":1024,"evaluation":1024} or
           r["training"]["training_steps"] != 128 for r in receipts):
        raise ValueError("development input does not match preregistered population")
    if len({r["source_sha256"] for r in receipts}) != 1:
        raise ValueError("mixed NC-02 implementation hashes")
    if len({r["protocol_sha256"] for r in receipts}) != 1:
        raise ValueError("mixed NC-02 protocol hashes")
    ordered = sorted(receipts, key=lambda r:r["seed"])
    summary = {
        "study":"BPD-NC-02",
        "phase":"ENGINEERING",
        "scientific_status":"NOT_CONFIRMATORY",
        "protocol_commit":PROTOCOL_COMMIT,
        "source_sha256":ordered[0]["source_sha256"],
        "protocol_sha256":ordered[0]["protocol_sha256"],
        "seeds":list(ROOTS),
        "independent_training_roots":len(ROOTS),
        "task_success_roots":sum(r["task_passed"] for r in ordered),
        "bases":{},
    }
    for kind in BASES:
        effect_root = []
        necessity_root = []
        failed = []
        all_eligible = []
        oracle_exact = []
        null_graph = []
        for r in ordered:
            basis = r["bases"][kind]
            if not basis["closure"]["passed"]:
                failed.append({"seed":r["seed"],"reason":"parity"})
                continue
            null_graph.append(basis["shuffled_graph_null"]["mean_cross_pass_communities"])
            if kind == "native":
                oracle_exact.append(basis["oracle_recovery"]["exact_recovered_paths"])
                if basis["readout_zero_null_communities"] != 0:
                    failed.append({"seed":r["seed"],"reason":"zero-readout-null"})
                    continue
            top2 = basis["groups"][:2]
            if len(top2) < 2 or any(not g["primary_sham_matching"]["eligible"] for g in top2):
                failed.append({"seed":r["seed"],"reason":"<2 eligible discovery-ranked groups"})
                continue
            all_eligible.append(r["seed"])
            effect_root.append(sum(g["vs_primary_sham"]["retain_final_kl_advantage_nats"] for g in top2)/2)
            necessity_root.append(sum(g["evaluation"]["arms"]["ablate"]["nll_delta_nats"] for g in top2)/2)
        analysis = {
            "eligible_root_seeds":all_eligible,
            "excluded_roots":failed,
            "detected_community_counts":[r["bases"][kind].get("n_communities",0) for r in ordered],
            "mean_shuffled_cross_pass_communities":sum(null_graph)/len(null_graph) if null_graph else None,
            "root_mean_keep_kl_advantage":root_bootstrap(effect_root,seed=17_101),
            "root_mean_ablation_nll_increase":root_bootstrap(necessity_root,seed=17_102),
        }
        if kind=="native":
            analysis["exact_planted_paths_recovered_by_root"]=oracle_exact
            analysis["both_planted_paths_recovered_roots"]=sum(v==2 for v in oracle_exact)
        summary["bases"][kind]=analysis
    # The primary native causal score must use all eight roots for dev resolution.
    native = summary["bases"]["native"]
    ci = native["root_mean_keep_kl_advantage"].get("root_bootstrap_95pct_CI")
    summary["development_resolution"] = (
        "ENGINEERING_POSITIVE_CONTROL" if len(native["eligible_root_seeds"]) == 8
        and native["both_planted_paths_recovered_roots"] == 8
        and ci and ci[0] > 0.02 and summary["task_success_roots"] == 8
        else "NOT_RESOLVED_OR_FAILED"
    )
    summary["confirmation"] = "SEALED; requires fresh registration and EvalMath preflight"
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--directory",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    receipts = []
    for seed in ROOTS:
        path=args.directory / f"nc02_root_{seed}.json"
        receipts.append(json.loads(path.read_text(encoding="utf-8")))
    report=aggregate(receipts)
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(report,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    for basis,value in report["bases"].items():
        print(basis, "eligible",len(value["eligible_root_seeds"]),
              "effect",value["root_mean_keep_kl_advantage"])
    print("development",report["development_resolution"],"confirmation",report["confirmation"])


if __name__ == "__main__":
    main()
