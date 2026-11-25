"""Independent, root-level statistics for the RP-02 training experiment.

This module consumes recorded predictions and labels. It imports neither the
training implementation nor the model. Every inferential contrast uses the
independently trained model root as its unit. Its verdict is DEVELOPMENT_ONLY.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from math import lcm
from numbers import Integral

import numpy as np


ARMS = ("A_task", "B_causal", "C_rotated")
EVALUATIONS = ("iid", "long")
MEANINGFUL_FLOOR = Fraction(1, 20)
ALPHA = Fraction(1, 20)
PLANNED_COMPARISONS = 2
PROTOCOL_ROOTS = 8


def _integer(value, where: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{where} must be an integer")
    value = int(value)
    if nonnegative and value < 0:
        raise ValueError(f"{where} must be nonnegative")
    return value


def _sequence(value, where: str) -> list:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{where} must be a list or tuple")
    return list(value)


def _prediction_record(value, where: str, *, causal: bool = False) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    required = {"correct", "predictions", "labels"}
    if causal:
        required.add("targeted")
    if not required.issubset(value):
        raise ValueError(f"{where} is missing {sorted(required - set(value))}")
    labels = [_integer(v, f"{where}.labels[{i}]")
              for i, v in enumerate(_sequence(value["labels"], f"{where}.labels"))]
    predictions = [_integer(v, f"{where}.predictions[{i}]")
                   for i, v in enumerate(_sequence(value["predictions"], f"{where}.predictions"))]
    raw_correct = _sequence(value["correct"], f"{where}.correct")
    if not labels or len(predictions) != len(labels) or len(raw_correct) != len(labels):
        raise ValueError(f"{where} needs nonempty, equal-length labels/predictions/correct")
    correct = []
    for i, bit in enumerate(raw_correct):
        if not isinstance(bit, (Integral, bool, np.bool_)) or int(bit) not in (0, 1):
            raise ValueError(f"{where}.correct[{i}] must be binary")
        correct.append(int(bit))
    expected = [int(prediction == label) for prediction, label in zip(predictions, labels)]
    if correct != expected:
        raise ValueError(f"{where}.correct disagrees with predictions and labels")
    record = {"correct": correct, "predictions": predictions, "labels": labels}
    if causal:
        targeted = _sequence(value["targeted"], f"{where}.targeted")
        if len(targeted) != len(labels) or any(not isinstance(v, (bool, np.bool_)) for v in targeted):
            raise ValueError(f"{where}.targeted must contain one boolean per case")
        record["targeted"] = [bool(v) for v in targeted]
    return record


def _validate(results) -> list[dict]:
    results = _sequence(results, "results")
    if not results:
        raise ValueError("At least one root is required")
    if len(results) > 18:
        raise ValueError("This development analyzer supports at most 18 roots for exact sign flips")
    roots, seen = [], set()
    for index, source in enumerate(results):
        if not isinstance(source, Mapping) or "root" not in source or "arms" not in source:
            raise ValueError(f"results[{index}] needs root and arms")
        root = _integer(source["root"], f"results[{index}].root")
        if root in seen:
            raise ValueError(f"Duplicate root {root}")
        seen.add(root)
        if not isinstance(source["arms"], Mapping) or set(source["arms"]) != set(ARMS):
            raise ValueError(f"Root {root} must contain exactly arms {ARMS}")
        arms = {}
        for arm in ARMS:
            value = source["arms"][arm]
            if not isinstance(value, Mapping) or not isinstance(value.get("eval"), Mapping):
                raise ValueError(f"Root {root}, {arm} needs an eval mapping")
            if any(split not in value["eval"] for split in EVALUATIONS):
                raise ValueError(f"Root {root}, {arm} needs iid and long evaluations")
            arms[arm] = {
                "eval": {split: _prediction_record(value["eval"][split], f"root {root}.{arm}.eval.{split}")
                         for split in EVALUATIONS},
                "causal": _prediction_record(value.get("causal"), f"root {root}.{arm}.causal", causal=True),
            }
        reference = arms[ARMS[0]]
        for arm in ARMS[1:]:
            for split in EVALUATIONS:
                if arms[arm]["eval"][split]["labels"] != reference["eval"][split]["labels"]:
                    raise ValueError(f"Root {root}, {split}: labels/order/length differ across arms")
            if arms[arm]["causal"]["labels"] != reference["causal"]["labels"]:
                raise ValueError(f"Root {root}: causal labels/order/length differ across arms")
            if arms[arm]["causal"]["targeted"] != reference["causal"]["targeted"]:
                raise ValueError(f"Root {root}: causal targeted masks differ across arms")
        roots.append({"root": root, "arms": arms})
    return sorted(roots, key=lambda row: row["root"])


def _accuracy(roots: list[dict], arm: str, endpoint: str, subgroup: str = "all") -> dict:
    rows = []
    for root in roots:
        record = (root["arms"][arm]["causal"] if endpoint == "causal"
                  else root["arms"][arm]["eval"][endpoint])
        if subgroup == "all":
            bits = record["correct"]
        else:
            wanted = subgroup == "targeted"
            bits = [bit for bit, target in zip(record["correct"], record["targeted"]) if target == wanted]
        total, correct = len(bits), sum(bits)
        rows.append({"root": root["root"], "n_cases": total, "correct": correct,
                     "accuracy": correct / total if total else None})
    total = sum(row["n_cases"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    complete = all(row["n_cases"] for row in rows)
    return {
        "n_units": len(roots),
        "n_units_with_cases": sum(row["n_cases"] > 0 for row in rows),
        "n_cases": total,
        "correct": correct,
        "macro_accuracy": (sum(row["accuracy"] for row in rows) / len(rows)) if complete else None,
        "pooled_accuracy": correct / total if total else None,
        "per_root": rows,
        "scope": "descriptive; repeated cases remain clustered by model root",
    }


def exact_sign_flip(effects: list[Fraction]) -> dict:
    """Two-sided mean test under independent, rootwise sign symmetry.

    Integer weights preserve ties exactly even when root case counts differ.
    The Counter enumerates the 2**n sign assignments with multiplicities.
    """
    if not effects or len(effects) > 18 or any(not isinstance(x, Fraction) for x in effects):
        raise ValueError("Exact sign flips require 1..18 rational root effects")
    scale = lcm(*(effect.denominator for effect in effects))
    weights = [effect.numerator * (scale // effect.denominator) for effect in effects]
    observed = abs(sum(weights))
    distribution = Counter({0: 1})
    for weight in weights:
        next_distribution = Counter()
        for value, multiplicity in distribution.items():
            next_distribution[value + weight] += multiplicity
            next_distribution[value - weight] += multiplicity
        distribution = next_distribution
    assignments = 1 << len(effects)
    tail = sum(count for value, count in distribution.items() if abs(value) >= observed)
    p = Fraction(tail, assignments)
    return {
        "method": "exact_two_sided_root_sign_flip_mean",
        "p_value": float(p),
        "p_fraction": str(p),
        "tail_assignments": tail,
        "total_assignments": assignments,
        "n_units": len(effects),
        "n_zero_effect_units": sum(effect == 0 for effect in effects),
        "assumptions": "independent roots and joint null invariant to independently reversing each root effect",
    }


def _holm(p_values: list[Fraction]) -> list[Fraction]:
    adjusted = [Fraction(1)] * len(p_values)
    running = Fraction(0)
    for rank, index in enumerate(sorted(range(len(p_values)), key=lambda i: p_values[i])):
        running = max(running, min(Fraction(1), p_values[index] * (len(p_values) - rank)))
        adjusted[index] = running
    return adjusted


def _bootstrap(effects: list[Fraction], draws: int, seed: int) -> dict:
    values = np.array([float(effect) for effect in effects], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975], method="linear")
    return {
        "method": "seeded_percentile_bootstrap_of_model_roots",
        "confidence_level": 0.95,
        "lower": float(lower),
        "upper": float(upper),
        "draws": draws,
        "seed": seed,
        "approximate": True,
        "familywise_adjusted": False,
        "scope": "descriptive development interval; small-root coverage is unestablished",
    }


def _contrast(roots: list[dict], baseline: str, role: str, draws: int, seed: int) -> tuple[dict, Fraction]:
    candidate, split = "B_causal", "long"
    rows, effects = [], []
    for root in roots:
        a = root["arms"][baseline]["eval"][split]["correct"]
        b = root["arms"][candidate]["eval"][split]["correct"]
        recovered = sum(right == 1 and left == 0 for left, right in zip(a, b))
        corrupted = sum(right == 0 and left == 1 for left, right in zip(a, b))
        effect = Fraction(recovered - corrupted, len(a))
        effects.append(effect)
        rows.append({"root": root["root"], "n_cases": len(a),
                     "baseline_correct": sum(a), "candidate_correct": sum(b),
                     "recovery": recovered, "corruption": corrupted,
                     "delta": float(effect), "delta_fraction": str(effect)})
    n_cases = sum(row["n_cases"] for row in rows)
    recovered = sum(row["recovery"] for row in rows)
    corrupted = sum(row["corruption"] for row in rows)
    effect = sum(effects, Fraction(0)) / len(effects)
    test = exact_sign_flip(effects)
    output = {
        "id": f"{candidate}_minus_{baseline}_{split}",
        "role": role,
        "candidate": candidate,
        "baseline": baseline,
        "endpoint": "ordinary long-delay answer accuracy with steering disabled",
        "n_units": len(roots),
        "n_cases": n_cases,
        "macro_effect": float(effect),
        "macro_effect_fraction": str(effect),
        "pooled_effect": (recovered - corrupted) / n_cases,
        "recovery": recovered,
        "corruption": corrupted,
        "net_recovery": recovered - corrupted,
        "scientific_variation_floor": float(MEANINGFUL_FLOOR),
        "per_root": rows,
        "sign_flip": test,
        "bootstrap_95": _bootstrap(effects, draws, seed),
        "evidence_class": "DEVELOPMENT_ONLY",
        "scientific_verdict": "DEVELOPMENT_ONLY",
    }
    return output, Fraction(test["p_fraction"])


def summarize(results, *, bootstrap_draws: int = 10000, bootstrap_seed: int = 20261126) -> dict:
    """Validate all paired records and summarize the fixed two-contrast family."""
    bootstrap_draws = _integer(bootstrap_draws, "bootstrap_draws")
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap_seed")
    if bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be positive")
    roots = _validate(results)
    contrasts, p_values = [], []
    for baseline, role in (("A_task", "primary"), ("C_rotated", "secondary")):
        contrast, p = _contrast(roots, baseline, role, bootstrap_draws, bootstrap_seed)
        contrasts.append(contrast)
        p_values.append(p)
    for contrast, adjusted in zip(contrasts, _holm(p_values)):
        effect = Fraction(contrast["macro_effect_fraction"])
        flags = {
            "positive_mean": effect > 0,
            "mean_exceeds_meaningful_floor": effect > MEANINGFUL_FLOOR,
            "holm_p_at_most_0_05": adjusted <= ALPHA,
            "descriptive_bootstrap_lower_above_zero": contrast["bootstrap_95"]["lower"] > 0,
        }
        contrast["holm_adjusted_p"] = float(adjusted)
        contrast["holm_adjusted_p_fraction"] = str(adjusted)
        contrast["descriptive_flags"] = flags
        contrast["positive_development_signal"] = all(flags.values())
        contrast["confirmation_claim"] = False
    arms = {
        arm: {
            "iid": _accuracy(roots, arm, "iid"),
            "long": _accuracy(roots, arm, "long"),
            "causal": {group: _accuracy(roots, arm, "causal", group)
                       for group in ("all", "targeted", "untargeted")},
        }
        for arm in ARMS
    }
    return {
        "schema_version": 1,
        "experiment": "BPD-RP-02 training steering",
        "evidence_class": "DEVELOPMENT_ONLY",
        "scientific_verdict": "DEVELOPMENT_ONLY",
        "confirmation_claim": False,
        "unit": "independently initialized model root, paired across training arms",
        "n_units": len(roots),
        "protocol_expected_roots": PROTOCOL_ROOTS,
        "protocol_root_count_complete": len(roots) == PROTOCOL_ROOTS,
        "roots": [root["root"] for root in roots],
        "all_roots_retained": True,
        "n_cases": {"iid": arms["A_task"]["iid"]["n_cases"],
                    "long": arms["A_task"]["long"]["n_cases"],
                    "causal": arms["A_task"]["causal"]["all"]["n_cases"]},
        "case_count_scope": "common paired observations per arm, clustered within roots",
        "planned_comparisons": PLANNED_COMPARISONS,
        "actual_inferential_comparisons": len(contrasts),
        "multiplicity": "Holm adjustment across both ordinary long-delay contrasts",
        "family_alpha": float(ALPHA),
        "scientific_variation_floor": float(MEANINGFUL_FLOOR),
        "readout_scope": "development evaluation; all inference is descriptive",
        "subgroup_scope": "IID and causal subgroups have descriptive accuracies only; undefined root subgroups yield null macro accuracy",
        "arms": arms,
        "contrasts": contrasts,
    }
