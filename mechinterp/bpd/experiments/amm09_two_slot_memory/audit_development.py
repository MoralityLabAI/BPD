"""AMM-09 independent post-development audit of the registered four contrasts.

This is an analysis-only repair. The source-frozen trainer/actor in experiment.py
has intentionally not been modified after the first DEVELOPMENT scoring.

Read the original first-run raw cohort, plus its pre-data training policy,
original semantic development lock and corrected EvalMath receipt.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

NAME = "BPD-AMM-09"
SCHEMA = "bpd.amm09.two_slot_memory.v1"
N_ROOTS = 64
N_CASES = 512
MASKS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))
POLICY_ARMS = ("memory_only", "pooled", "memory_plus_local",
               "local_ridge", "local_count", "random_mask")
SHAM_ARMS = ("random_dark", "random_dark_b", "sham", "full_donor")
COMPARISONS = (
    ("memory_only_vs_pooled", "memory_only", "pooled"),
    ("memory_plus_local_vs_pooled", "memory_plus_local", "pooled"),
    ("memory_plus_local_vs_local_ridge", "memory_plus_local", "local_ridge"),
    ("memory_plus_local_vs_memory_only", "memory_plus_local", "memory_only"),
)
SCIENCE_MINIMUM = .08
BOOTSTRAP_REPS = 2000


def digest(value) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def seed(*parts: object) -> int:
    value = "AMM09/" + "/".join(map(str, parts))
    return int(hashlib.sha256(value.encode()).hexdigest()[:15], 16)


def one_sided_sign_p(diffs: list[int]) -> float:
    n = sum(d != 0 for d in diffs)
    if n == 0:
        return 1.
    k = sum(d > 0 for d in diffs)
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2**n


def cluster_bootstrap(diffs: list[int], comparison: str) -> list[float]:
    rng = random.Random(seed("bootstrap", comparison))
    n = len(diffs)
    draws = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / (n * N_CASES)
                   for _ in range(BOOTSTRAP_REPS))
    return [draws[50], draws[1949]]


def holm_adjustment(pvalues: dict[str, float]) -> dict[str, float]:
    adjusted = {}
    previous = 0.
    for i, key in enumerate(sorted(pvalues, key=lambda k: (pvalues[k], k))):
        previous = max(previous, min(1., pvalues[key] * (len(pvalues) - i)))
        adjusted[key] = previous
    return adjusted


def analyze(development: dict, train_lock: dict, dev_lock: dict, preflight: dict) -> dict:
    if development.get("phase") != "development" or development.get("experiment") != NAME:
        raise AssertionError("wrong development experiment/phase")
    if development.get("n_independent_roots") != N_ROOTS or development.get("total_test_cases") != N_ROOTS * N_CASES:
        raise AssertionError("independent model root or case count changed")
    if development["source_sha256"] != train_lock["source_sha256"] or development["source_sha256"] != dev_lock["source_sha256"]:
        raise AssertionError("frozen training/development model source mismatch")
    if development["training_policy_sha256"] != train_lock["training_policy_sha256"] or development["training_policy_sha256"] != dev_lock["training_policy_sha256"]:
        raise AssertionError("training policy source lock changed")
    if train_lock.get("pooled_action") != 4 or tuple(MASKS[4]) != (0, 2):
        raise AssertionError("wrong registered best fixed admissible mask")
    if preflight.get("experiment_id") != NAME or preflight.get("pilot_worlds") != N_ROOTS or preflight.get("planned_comparisons") != len(COMPARISONS):
        raise AssertionError("EvalMath input/output wrong model-root plan")
    if development.get("preflight_input", {}).get("planned_comparisons") != len(COMPARISONS):
        raise AssertionError("preregistered comparison family changed")

    rows = development["rows"]
    if len(rows) != N_ROOTS:
        raise AssertionError("missing independent development roots")
    histograms = {arm: Counter() for arm in POLICY_ARMS}
    switched_root_counts = {arm: 0 for arm in POLICY_ARMS}
    casewise_oracle_hits = 0
    semantic_items = []
    totals = {arm: 0 for arm in POLICY_ARMS + SHAM_ARMS}
    all_case_switches_vs_pooled = 0
    for i, root in enumerate(rows):
        if root.get("root") != f"development/{i:04d}":
            raise AssertionError("development root duplication or wrong phase")
        candidates = root["heldout"]["candidate_bits"]
        if set(candidates) != {f"mask_{k}" for k in range(6)} | set(SHAM_ARMS):
            raise AssertionError("unregistered seven-mask candidate leaked into the assay")
        if any(len(bits) != N_CASES or set(bits) - {"0", "1"} for bits in candidates.values()):
            raise AssertionError("malformed test candidate success bits")
        expected_oracle = sum(any(candidates[f"mask_{k}"][j] == "1" for k in range(6))
                              for j in range(N_CASES))
        if root["heldout"]["diagnostics"]["casewise_oracle_hits_audit_only"] != expected_oracle:
            raise AssertionError("casewise oracle must use only the six permitted actions")
        casewise_oracle_hits += expected_oracle
        calibrations = root["agent_noisy_feedback"]["reported_bits"]
        if set(calibrations) != {f"mask_{k}" for k in range(6)} or any(len(v) != 16 for v in calibrations.values()):
            raise AssertionError("wrong 96-label calibration budget")
        for arm in POLICY_ARMS:
            actions = root["policy_actions"][arm]
            if len(actions) != N_CASES or any(type(a) != int or not 0 <= a < 6 for a in actions):
                raise AssertionError("forbidden mask or wrong selected action count")
            reconstructed = "".join(candidates[f"mask_{a}"][j] for j, a in enumerate(actions))
            if root["policy_bits"][arm] != reconstructed:
                raise AssertionError("casewise action and reported success labels disagree")
            totals[arm] += reconstructed.count("1")
            histograms[arm].update(actions)
            switched_root_counts[arm] += len(set(actions)) > 1
        all_case_switches_vs_pooled += sum(a != b for a, b in
                      zip(root["policy_actions"]["memory_only"], root["policy_actions"]["pooled"]))
        if any(a != 4 for a in root["policy_actions"]["pooled"]):
            raise AssertionError("globally selected fixed pooled mask changed")
        for arm in SHAM_ARMS:
            if root["policy_bits"][arm] != candidates[arm]:
                raise AssertionError("sham bit traces differ from scored arm")
            totals[arm] += root["policy_bits"][arm].count("1")
        repeat = root["repeatability"]
        if repeat["pooled_bits"] != repeat["assay"]["candidate_bits"]["mask_4"]:
            raise AssertionError("repeatability sham changed pooled policy")
        if repeat["pooled_hits"] != repeat["pooled_bits"].count("1"):
            raise AssertionError("pooled repeatability bit accounting failed")
        semantic_items.append({
            "root": root["root"],
            "clean_calibration": root["calibration_clean_audit_only"]["candidate_bits"],
            "noisy_calibration": calibrations,
            "test_candidates": candidates,
            "selected_actions": root["policy_actions"],
            "selected_outcomes": root["policy_bits"],
            "repeat_local_outcomes": repeat["local_ridge_bits"],
            "repeat_pooled_outcomes": repeat["pooled_bits"],
            "fidelity_flag": root["fidelity_audit_only"],
        })
        pilot_row = development["preflight_input"]["pilot"][i]
        expected_effect = f"{root['policy_hits']['memory_only']-root['policy_hits']['pooled']}/{N_CASES}"
        expected_sham = f"{root['policy_hits']['pooled']-repeat['pooled_hits']}/{N_CASES}"
        if pilot_row != {"unit_id": root["root"], "paired_effect": expected_effect,
                         "sham_effects": [expected_sham]}:
            raise AssertionError("primary pilot or its sham differs from registered arms")
    if totals != development["totals"] or totals != dev_lock["totals"]:
        raise AssertionError("all-case policy totals changed")
    if digest(semantic_items) != dev_lock["all_candidate_action_case_bits_sha256"]:
        raise AssertionError("original locked selected-action/candidate case hash mismatch")

    pvalues = {}
    comparisons = {}
    for name, treatment, control in COMPARISONS:
        differences = [root["policy_hits"][treatment] - root["policy_hits"][control]
                       for root in rows]
        p = one_sided_sign_p(differences)
        pvalues[name] = p
        comparisons[name] = {
            "treatment": treatment, "control": control,
            "difference_successes": sum(differences),
            "mean_independent_root_effect": sum(differences) / (N_ROOTS * N_CASES),
            "positive_roots": sum(x > 0 for x in differences),
            "negative_roots": sum(x < 0 for x in differences),
            "tied_roots": sum(x == 0 for x in differences),
            "exact_one_sided_root_sign_p": p,
            "paired_root_bootstrap_95_ci": cluster_bootstrap(differences, name),
        }
    adjusted = holm_adjustment(pvalues)
    for key in comparisons:
        comparisons[key]["four_comparison_holm_p"] = adjusted[key]

    primary = comparisons["memory_only_vs_pooled"]
    if abs(preflight["mean_pilot_effect"] - primary["mean_independent_root_effect"]) > 1e-15:
        raise AssertionError("preflight did not use preregistered memory-only primary")
    floor = max(SCIENCE_MINIMUM, preflight["mechanistic_variation_floor"])
    fidelity = sum(bool(root["fidelity_audit_only"]) for root in rows)
    headroom = (casewise_oracle_hits - totals["pooled"]) / (N_ROOTS * N_CASES)
    dose_error = max(root["heldout"]["diagnostics"]["max_equal_dose_error"] for root in rows)
    visible = max(root["heldout"]["diagnostics"]["max_immediate_readout_displacement"] for root in rows)
    original_mislabeled = development["analysis"]["comparisons"]["memory_only_vs_pooled"]["mean_root_effect"]
    if abs(original_mislabeled - comparisons["memory_plus_local_vs_pooled"]["mean_independent_root_effect"]) > 1e-15:
        raise AssertionError("historical bug differs from documented numerator mistake")

    result = {
        "schema": SCHEMA + "/post_data_registered_analysis_correction",
        "evidence": "development only; explicit independent source-preserving repair",
        "experiment": NAME,
        "n_independent_model_roots": N_ROOTS,
        "n_test_cases_per_root": N_CASES,
        "training_policy_sha256": train_lock["training_policy_sha256"],
        "source_sha256": train_lock["source_sha256"],
        "locked_original_case_bits_sha256": dev_lock["all_candidate_action_case_bits_sha256"],
        "original_source_primary_mislabeled_mean": original_mislabeled,
        "historical_analysis_mistake": "Original experiment.py.analysis treated every comparison key as memory_plus_local minus control; preflight_input correctly used memory_only minus pooled",
        "comparisons": comparisons,
        "policy_action_histograms": {arm: {f"mask_{i}": histograms[arm][i]
                                       for i in range(6)} for arm in POLICY_ARMS},
        "models_selecting_multiple_masks": switched_root_counts,
        "memory_only_cases_differing_from_pooled": all_case_switches_vs_pooled,
        "casewise_six_mask_oracle_hits_audit_only": casewise_oracle_hits,
        "oracle_headroom_above_pooled_audit_only": headroom,
        "high_fidelity_models": fidelity,
        "max_active_dose_error": dose_error,
        "max_immediate_readout_drift": visible,
        "matched_sham_95_95_tolerance_bound": preflight["upper_sham_quantile"],
        "mechanistic_variation_floor": floor,
        "corrected_evalmath_decision": preflight["decision"],
        "development_admission_gates": {
            "preflight_ready": preflight["decision"] == "READY_FOR_CONFIRMATION" and not preflight["reasons"],
            "meaningful_primary_effect_and_uncertainty": primary["paired_root_bootstrap_95_ci"][0] > floor,
            "four_comparison_adjusted_sign_p": primary["four_comparison_holm_p"] < .05,
            "fidelity": fidelity >= 56,
            "oracle_headroom": headroom >= .08,
            "physical_equal_dose": dose_error <= 1e-5 and visible == 0,
        },
        "confirmation_authorized": False,
        "confirmation_authorization_reason": "A separate human-approved, source/policy/preflight-bound release manifest is still required; all 128 confirmation roots remain sealed.",
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development", required=True, type=Path)
    p.add_argument("--training-lock", required=True, type=Path)
    p.add_argument("--development-lock", required=True, type=Path)
    p.add_argument("--preflight", required=True, type=Path)
    p.add_argument("--out", required=False, type=Path)
    p.add_argument("--check-lock", type=Path)
    args = p.parse_args()
    result = analyze(json.loads(args.development.read_text()),
                     json.loads(args.training_lock.read_text()),
                     json.loads(args.development_lock.read_text()),
                     json.loads(args.preflight.read_text()))
    if args.check_lock:
        assert result == json.loads(args.check_lock.read_text()), "corrected independent audit drift"
    output = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    print("AMM09_REGISTERED_PRIMARY_CORRECTED=" + json.dumps(result["comparisons"]["memory_only_vs_pooled"], sort_keys=True))
    print("AMM09_ADMISSION=" + json.dumps(result["development_admission_gates"], sort_keys=True))
    print("AMM09_CORRECTED_ANALYSIS="+json.dumps(result,sort_keys=True))


if __name__ == "__main__":
    main()
