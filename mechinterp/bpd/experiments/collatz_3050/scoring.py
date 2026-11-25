"""Paired, parameter-clustered development scoring for BPD-COL-02.

The primary endpoint is donor-directed response under an equal-dose prefix
patch, after accounting for the stronger matched-control response. It is a
steering endpoint. A change toward an opposite-label donor is not counted as
improved correctness on the original recipient problem.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from statistics import fmean


SCHEMA = "BPD-COL-02/paired-development-score/v1"
CLEAN_ROLES = ("recipient", "effect_donor", "sham_donor")
ARMS = ("identity", "effect", "sham", "random", "full_donor_prefix")
MATCHED_ARMS = ("effect", "sham", "random")


def _number(value: object, name: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def probabilities(value: object) -> dict[str, float]:
    """Validate conditional answer probabilities without changing them."""
    if isinstance(value, dict) and "probabilities" in value:
        value = value["probabilities"]
    if not isinstance(value, dict) or set(value) != {"A", "B"}:
        raise ValueError("probabilities must contain exactly A and B")
    result = {label: _number(p, f"p({label})") for label, p in value.items()}
    if any(not 0 <= p <= 1 for p in result.values()):
        raise ValueError("answer probabilities must lie in [0,1]")
    if abs(sum(result.values()) - 1) > 1e-6:
        raise ValueError("conditional answer probabilities must sum to one")
    return result


def prediction(probs: dict[str, float]) -> str | None:
    if probs["A"] == probs["B"]:
        return None
    return "A" if probs["A"] > probs["B"] else "B"


def _expected_cases(manifest: dict | list[dict]) -> list[dict]:
    if isinstance(manifest, dict):
        if __package__:
            from .cases import validate_manifest
        else:
            from cases import validate_manifest
        validate_manifest(manifest)
    cases = manifest.get("cases") if isinstance(manifest, dict) else manifest
    if not isinstance(cases, list) or not cases:
        raise ValueError("expected manifest must contain a nonempty case list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        ident, root = case.get("case_id"), case.get("root_id")
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError("case IDs must be distinct nonempty strings")
        if type(root) is not int or root <= 0:
            raise ValueError("root_id must be a positive integer parameter")
        if case.get("d") != root or type(case.get("d")) is not int:
            raise ValueError("root_id must equal integer d")
        labels = []
        for role in CLEAN_ROLES:
            if not isinstance(case.get(role), dict) or case[role].get("label") not in ("A", "B"):
                raise ValueError("each clean role needs its exact oracle label")
            labels.append(case[role]["label"])
        if labels[0] == labels[1] or labels[0] != labels[2]:
            raise ValueError("oracle effect/sham label contract is inconsistent")
        seen.add(ident)
    return cases


def summarize(records: list[dict], manifest: dict | list[dict], *,
              protected_tolerance: float = 0.0,
              dose_relative_tolerance: float = 0.02,
              dose_absolute_tolerance: float = 1e-6,
              identity_probability_tolerance: float = 1e-6,
              scientific_floor: float = 0.05) -> dict:
    """Score all planned cases, including failures and unattempted cases.

    Completed record schema:
      {case_id, root_id, status:'completed',
       clean:{recipient:{probabilities:{A:p,B:p}}, effect_donor:{...},
              sham_donor:{...}},
       arms:{name:{probabilities:{A:p,B:p}, physical_dose:float,
                    protected_readout_max_abs:float, admission:bool}}}

    Names in ARMS are required. An error/skipped record requires a nonempty
    reason. The scorer never turns exposed development cases into a held-out
    scientific acceptance. Repeated cases are averaged within parameter d.

    The prospective engineering dose allowance is 2% of the largest delivered
    matched-arm norm plus 1e-6. Each BF16-rounded intervention is independently
    normalized within 1% of the common requested norm. Admission uses the
    measured delivered norms. This source choice precedes native model runs;
    it is not an empirically calibrated sham or scientific effect threshold.
    """
    for key, value in {
        "protected_tolerance": protected_tolerance,
        "dose_relative_tolerance": dose_relative_tolerance,
        "dose_absolute_tolerance": dose_absolute_tolerance,
        "identity_probability_tolerance": identity_probability_tolerance,
        "scientific_floor": scientific_floor,
    }.items():
        _number(value, key, nonnegative=True)
    if scientific_floor > 1:
        raise ValueError("scientific_floor is in answer-probability units")
    cases = _expected_cases(manifest)
    expected = {case["case_id"]: case for case in cases}
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    by_id = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("record must be an object")
        ident = record.get("case_id")
        if not isinstance(ident, str) or ident not in expected:
            raise ValueError("unknown or invalid record case_id")
        if ident in by_id:
            raise ValueError("duplicate record case_id")
        if (type(record.get("root_id")) is not int
                or record["root_id"] != expected[ident]["root_id"]):
            raise ValueError("record root does not match the frozen case")
        by_id[ident] = record

    per_case = []
    reason_counts = Counter()
    clean_counts = {role: {"correct": 0, "scored": 0, "ties": 0}
                    for role in CLEAN_ROLES}
    max_protected = 0.0
    max_identity_probability_error = 0.0
    max_dose_spread = 0.0
    all_arm_counts = {arm: {"admitted": 0, "rejected": 0} for arm in ARMS}
    for case in cases:
        ident = case["case_id"]
        basin_match = ("same_basin" if case.get("all_same_basin") is True else
                       "cross_basin" if case.get("all_same_basin") is False else
                       "unspecified")
        result = {"case_id": ident, "root_id": case["root_id"],
                  "basin_match": basin_match,
                  "selection_role": case.get("selection_role", "unspecified"),
                  "eligible": False, "reasons": []}
        record = by_id.get(ident)
        if record is None:
            result["status"] = "NOT_ATTEMPTED"
            result["reasons"].append("not_attempted")
            per_case.append(result)
            reason_counts.update(result["reasons"])
            continue
        status = record.get("status", "completed")
        if status in ("error", "skipped"):
            if not isinstance(record.get("reason"), str) or not record["reason"].strip():
                raise ValueError("error/skipped records need a nonempty reason")
            result.update(status=status.upper(), reason=record["reason"])
            result["reasons"].append(status)
            per_case.append(result)
            reason_counts.update(result["reasons"])
            continue
        if status != "completed":
            raise ValueError(f"unknown record status {status!r}")
        clean, arms = record.get("clean"), record.get("arms")
        if not isinstance(clean, dict) or set(clean) != set(CLEAN_ROLES):
            raise ValueError("completed records require all three clean roles")
        if not isinstance(arms, dict) or set(arms) != set(ARMS):
            raise ValueError("completed records require exactly the declared five arms")
        clean_probs = {role: probabilities(clean[role]) for role in CLEAN_ROLES}
        clean_correct = {}
        for role in CLEAN_ROLES:
            pred = prediction(clean_probs[role])
            clean_correct[role] = pred == case[role]["label"]
            clean_counts[role]["scored"] += 1
            clean_counts[role]["correct"] += clean_correct[role]
            clean_counts[role]["ties"] += pred is None
        if not all(clean_correct.values()):
            result["reasons"].append("base_task_incompetence")

        arm_probs, dose = {}, {}
        for arm in ARMS:
            data = arms[arm]
            if not isinstance(data, dict) or type(data.get("admission")) is not bool:
                raise ValueError(f"{arm} needs an explicit boolean admission")
            arm_probs[arm] = probabilities(data)
            dose[arm] = _number(data.get("physical_dose"), f"{arm}.physical_dose",
                                nonnegative=True)
            protected = _number(data.get("protected_readout_max_abs"),
                                f"{arm}.protected_readout_max_abs", nonnegative=True)
            max_protected = max(max_protected, protected)
            all_arm_counts[arm]["admitted" if data["admission"] else "rejected"] += 1
            if not data["admission"]:
                result["reasons"].append(f"rejected_{arm}")
            if protected > protected_tolerance:
                result["reasons"].append(f"current_readout_changed_{arm}")
        if dose["identity"] != 0.0:
            result["reasons"].append("identity_nonzero_dose")
        identity_error = max(abs(arm_probs["identity"][label] - clean_probs["recipient"][label])
                             for label in ("A", "B"))
        max_identity_probability_error = max(max_identity_probability_error, identity_error)
        if identity_error > identity_probability_tolerance:
            result["reasons"].append("identity_probability_drift")
        matched_doses = [dose[arm] for arm in MATCHED_ARMS]
        spread = max(matched_doses) - min(matched_doses)
        max_dose_spread = max(max_dose_spread, spread)
        if min(matched_doses) <= 0:
            result["reasons"].append("zero_matched_dose")
        if spread > dose_absolute_tolerance + dose_relative_tolerance * max(matched_doses):
            result["reasons"].append("unmatched_physical_doses")

        target = case["effect_donor"]["label"]
        baseline = clean_probs["recipient"][target]
        shifts = {arm: arm_probs[arm][target] - baseline for arm in ARMS}
        strongest_control = max(abs(shifts["sham"]), abs(shifts["random"]))
        effect = shifts["effect"] - strongest_control
        result.update(
            status="COMPLETED", clean_correct=clean_correct,
            clean_predictions={role: prediction(clean_probs[role]) for role in CLEAN_ROLES},
            donor_label=target, recipient_label=case["recipient"]["label"],
            donor_probability_shifts=shifts,
            matched_control_absolute_envelope=strongest_control,
            primary_contrast=effect,
            physical_doses=dose, matched_dose_spread=spread,
            identity_probability_error=identity_error,
            donor_recovery={arm: prediction(arm_probs[arm]) == target for arm in ARMS},
            recipient_task_correct={arm: prediction(arm_probs[arm]) == case["recipient"]["label"]
                                    for arm in ARMS},
        )
        result["eligible"] = not result["reasons"]
        if "base_task_incompetence" in result["reasons"]:
            result["status"] = "INELIGIBLE_BASE_TASK"
        elif result["reasons"]:
            result["status"] = "INELIGIBLE_INSTRUMENT"
        reason_counts.update(result["reasons"])
        per_case.append(result)

    clusters = defaultdict(list)
    for case in per_case:
        clusters[case["root_id"]].append(case)
    per_root = []
    for root_id, entries in sorted(clusters.items()):
        eligible = [entry for entry in entries if entry["eligible"]]
        # Root inference uses complete case sets. Partial root eligibility is
        # reported, while its primary effect remains outside the root average.
        complete = len(eligible) == len(entries)
        basin_matches = {entry["basin_match"] for entry in entries}
        root = {"root_id": root_id, "planned_cases": len(entries),
                "basin_match": next(iter(basin_matches)) if len(basin_matches) == 1 else "mixed",
                "eligible_cases": len(eligible), "complete_root": complete,
                "primary_contrast": fmean(x["primary_contrast"] for x in entries) if complete else None,
                "effect_shift": fmean(x["donor_probability_shifts"]["effect"] for x in entries)
                                if complete else None,
                "control_envelope": fmean(x["matched_control_absolute_envelope"] for x in entries)
                                    if complete else None}
        per_root.append(root)
    eligible_roots = [root for root in per_root if root["complete_root"]]
    primary = fmean(root["primary_contrast"] for root in eligible_roots) if eligible_roots else None
    basin_strata = {}
    for basin_match in sorted({root["basin_match"] for root in per_root}):
        roots = [root for root in per_root if root["basin_match"] == basin_match]
        complete_roots = [root for root in roots if root["complete_root"]]
        basin_strata[basin_match] = {
            "n_units": len(roots), "n_eligible_units": len(complete_roots),
            "n_cases": sum(root["planned_cases"] for root in roots),
            "n_eligible_cases": sum(root["eligible_cases"] for root in roots),
            "primary_root_mean": fmean(root["primary_contrast"] for root in complete_roots)
                                 if complete_roots else None,
        }
    completed_cases = [case for case in per_case if "clean_correct" in case]
    competent_cases = sum(all(case["clean_correct"].values()) for case in completed_cases)
    return {
        "schema": SCHEMA, "verdict": "PILOT_ONLY",
        "evidence_class": "public development; conditional on the tested checkpoint",
        "score_semantics": "conditional next-token probability over the two declared answers A/B",
        "unit": "parameter d; repeated horizons, doses and token observations share one root",
        "unit_selection": "deterministic public parameter selection within one checkpoint",
        "root_weighting": "equal weight across complete eligible roots; equal case weight within each root",
        "n_units": len(clusters), "n_eligible_units": len(eligible_roots),
        "n_cases": len(cases), "n_attempted_cases": len(records),
        "n_completed_cases": len(completed_cases),
        "n_eligible_cases": sum(case["eligible"] for case in per_case),
        "case_status_counts": dict(sorted(Counter(case["status"] for case in per_case).items())),
        "ineligibility_reason_counts": dict(sorted(reason_counts.items())),
        "clean_task": clean_counts, "arm_admission": all_arm_counts,
        "competence_coverage": {
            "all_clean_correct_cases": competent_cases,
            "fraction_of_completed_cases": competent_cases / len(completed_cases)
                                           if completed_cases else None,
            "fraction_of_planned_cases": competent_cases / len(cases),
            "scope": "all three clean roles correct in completed records; partial error records remain diagnostic",
        },
        "primary_contrast": "effect donor-probability shift minus maximum absolute matched sham/random shift",
        "primary_interpretation": "donor-directed steering conditional on clean competence and instrument admission",
        "primary_root_mean": primary,
        "basin_strata": basin_strata,
        "scientific_variation_floor": scientific_floor,
        "scientific_floor_units": "conditional answer probability",
        "scientific_floor_status": "prospective effect scale; the development verdict remains PILOT_ONLY",
        "matched_sham_floor": None,
        "matched_sham_floor_status": "uncalibrated; a 95th-percentile/95%-confidence sample maximum needs at least 59 independent draws from the intended root-level null distribution",
        "instrument": {
            "max_protected_readout_abs": max_protected,
            "protected_readout_tolerance": protected_tolerance,
            "protected_readout_scope": "complete native vocabulary head at the final prompt position after the first native pass",
            "protected_readout_evidence": "runner checks exact first-pass final-vector equality; scorer validates its scalar certificate",
            "max_identity_probability_error": max_identity_probability_error,
            "identity_probability_tolerance": identity_probability_tolerance,
            "max_matched_dose_spread": max_dose_spread,
            "dose_relative_tolerance": dose_relative_tolerance,
            "dose_absolute_tolerance": dose_absolute_tolerance,
            "dose_tolerance_basis": "prospective engineering allowance for measured BF16-rounded norms; no native calibration",
        },
        "planned_comparisons_per_case": list(ARMS),
        "actual_completed_comparisons": sum(sum(counts.values()) for counts in all_arm_counts.values()),
        "estimated_resolution_power": "unavailable; native independent-root sham pilot required",
        "readout_selection_exposure": "all case parameters are public exposed development sources",
        "causal_controls": "identity, equal-dose semantic sham, equal-dose seeded random, separate full-donor prefix",
        "further_data": "freeze model/site/arm selection after development, calibrate independent-root sham variation, and register fresh confirmation roots",
        "per_root": per_root, "per_case": per_case,
    }


score_records = summarize
