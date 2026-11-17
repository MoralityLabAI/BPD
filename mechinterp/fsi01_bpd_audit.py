"""FSI-01 retrospective evaluation of BPD's existing public receipts.

Uses EvalMath's exact paired test and cluster-aware evidence policy. All ARG-03
data here are development-only; the protected 168-case test set is untouched.

Run with PYTHONPATH=<EvalMath>/src python mechinterp/fsi01_bpd_audit.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from spiral_ln.evaluation_finite_sample import (
    binomial_two_sided,
    evaluate_manifest,
    holm_adjust,
)

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "bundle": "mechinterp/loopscope02/results/benchmark_bundle.json",
    "trm": "mechinterp/loopscope02/results/trm_results.json",
    "qwen": "mechinterp/loopscope02/results/qwen4b_results.json",
    "grade": "supplements/loopscope04_arg03_receipts/arg03_dev96/grade_dev96.json",
}


def _load(root: Path, path: str) -> dict:
    return json.loads((root / path).read_text(encoding="utf-8"))


def _manifest(name: str, items: list[dict], candidate: list[int],
              control: list[int]) -> dict:
    if not (len(items) == len(candidate) == len(control) == 256):
        raise ValueError("LOOPSCOPE-02 paired comparison must contain all 256 items")
    return {
        "schema_version": 1,
        "experiment_id": name,
        "target_population": "LOOPSCOPE-02 fixed four-stratum Sudoku roster, retrospective",
        "evaluation_mode": "sampled",
        "alpha": "0.05",
        "minimum_effect": "0",
        "planned_comparisons": 4,
        # These four comparisons were selected for this follow-up after their
        # original outcomes were published. They are NEVER preregistered claims.
        "hypothesis_preregistered": False,
        "sealed_readout": False,
        # Each recorded source puzzle appears once, subject to the assumption
        # that the source-puzzle draws are independent within their strata.
        "clusters_independent": True,
        "exchangeable_pair_labels": True,
        "bootstrap_resamples": 1024,
        "resample_seed": 20261117,
        "cases": [
            {"case_id": item["item_id"], "cluster_id": item["origin"],
             "candidate": int(a), "baseline": int(b)}
            for item, a, b in zip(items, candidate, control)
        ],
    }


def audit(root: Path = ROOT) -> dict:
    docs = {name: _load(root, path) for name, path in SOURCES.items()}
    bundle, trm, qwen, grade = (docs[x] for x in ("bundle", "trm", "qwen", "grade"))
    items = bundle["items"]
    if not (len(items) == 256
            and len({x["item_id"] for x in items}) == 256
            and len({x["origin"] for x in items}) == 256):
        raise ValueError("item/root uniqueness changed; independence unit needs re-review")
    strata = Counter(x["stratum"] for x in items)
    if sorted(strata.values()) != [64] * 4:
        raise ValueError("expected 64 cases per Sudoku difficulty stratum")
    if trm["bundle_sha256"] != bundle["bundle_sha256"] or qwen["bundle_sha256"] != bundle["bundle_sha256"]:
        raise ValueError("incompatible LOOPSCOPE-02 bundle digests")

    contrasts = {
        "trm_target_step16_vs_step1": (
            trm["items_target_correct_by_step"][-1],
            trm["items_target_correct_by_step"][0]),
        "trm_target_step16_vs_donor_zH": (
            trm["items_target_correct_by_step"][-1],
            trm["items_target_donor_correct"]),
        "trm_board_clean16_vs_fiber_erased": (
            trm["items_board_exact_by_step"][-1],
            trm["items_board_exact_fiber_erased"]),
        "trm_target_step16_vs_qwen4b": (
            trm["items_target_correct_by_step"][-1],
            qwen["items_target_correct"]),
    }
    details = {}
    for name, (a,b) in contrasts.items():
        result = evaluate_manifest(_manifest(name, items, a, b))
        if result["verdict"] == "confirmed" or result["verdict"] == "exact_finite_result":
            raise AssertionError("retrospective unsealed comparison was misclassified")
        result["by_stratum"] = {
            stratum: {
                "n": len(indices),
                "candidate_correct": sum(a[i] for i in indices),
                "baseline_correct": sum(b[i] for i in indices),
            }
            for stratum in sorted(strata)
            for indices in [[i for i,x in enumerate(items) if x["stratum"] == stratum]]
        }
        details[name] = result

    if grade.get("split") != "development" or grade.get("rows_scored") != 96:
        raise ValueError("ARG-03 development receipt must remain 96-case development")
    tests=[]
    for arm, record in grade["results"].items():
        for label, row in record.items():
            if not label.startswith("vs_") or not isinstance(row, dict):
                continue
            if "exact_two_sided_p" not in row:
                continue
            a,b=row["gain"],row["loss"]
            p=binomial_two_sided(a,a+b)
            if abs(float(p)-row["exact_two_sided_p"]) > 1e-11:
                raise ValueError(f"paired exact tail differs for {arm}: {label}")
            tests.append({"arm":arm,"comparison":label,"gain":a,"loss":b,
                          "p_exact":str(p),"p":float(p)})
    if not tests:
        raise ValueError("ARG-03 receipt has no exact paired comparisons")
    adjustments=holm_adjust([binomial_two_sided(x["gain"], x["gain"]+x["loss"])
                             for x in tests])
    for t,adj in zip(tests,adjustments):
        t["holm_all_reported"]=float(adj)
        t["holm_exact"]=str(adj)
    tests.sort(key=lambda x:(x["p"],x["arm"],x["comparison"]))

    return {
        "schema": "fsi01.bpd.retrospective.v1",
        "status": "retrospective development evidence; held-out ARG-03 test unaccessed",
        "source_sha256": {
            key: hashlib.sha256((root/path).read_bytes()).hexdigest()
            for key,path in SOURCES.items()
        },
        "loopscope02": {
            "board_origins": len({x["origin"] for x in items}),
            "n_cases": len(items),
            "strata": dict(sorted(strata.items())),
            "contrasts": details,
        },
        "arg03_dev96": {
            "split": grade["split"],
            "n_cases": grade["rows_scored"],
            "n_arms": len(grade["results"]),
            "n_tests_reported": len(tests),
            "nominal_p_below_005": sum(x["p"]<0.05 for x in tests),
            "holm_p_below_005": sum(x["holm_all_reported"]<0.05 for x in tests),
            "smallest_tests": tests[:5],
            "scope": "Holm over all reported directional comparisons is a conservative post hoc family; repeated aliases and selected development outcomes remain exploratory.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=audit()
    payload=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload,encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
