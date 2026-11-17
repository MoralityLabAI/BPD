"""BPD-PD-02 posthoc omission audit from the *unchanged* full JSON receipt.

This computes a mechanistic diagnostic after the registered FAIL. It neither
changes the model nor re-runs any train/held-out example. The four-pass
independence product is a prediction from independent input draws; the
reported agreement is descriptive, not a preregistered hypothesis test.

Run: python -m mechinterp.bpd.theory.pd02.omission_audit <receipt.json>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def analyze(receipt: dict) -> dict:
    if receipt["study_id"] != "BPD-PD-02":
        raise ValueError("BPD-PD-02 source receipt required")
    arms = [r for r in receipt["arms"] if r["arm"] == "hard_disjoint"]
    if len(arms) != 24:
        raise ValueError("complete 24 hard-disjoint arms required")
    per_seed = []
    false_off = 0
    wrong_on = 0
    required = 0
    expected = []
    observed = []
    for r in sorted(arms, key=lambda x: x["seed"]):
        vals = r["stage1_primary_heldout"]["mean_hard_gate_by_pass"]
        n = r["heldout_count"]
        if len(vals) != 4 or any(len(v) != 2 for v in vals):
            raise ValueError("expected four passes and two gate directions")
        index0 = 0 if vals[0][0] + vals[1][0] > vals[0][1] + vals[1][1] else 1
        indices = [index0, index0, 1 - index0, 1 - index0]
        intended = [vals[p][indices[p]] for p in range(4)]
        opposite = [vals[p][1 - indices[p]] for p in range(4)]
        for rate, wrong_rate in zip(intended, opposite):
            missing = n - round(n * rate)
            wrong = round(n * wrong_rate)
            if abs(missing - n * (1 - rate)) > 1e-6:
                raise ValueError("rate incompatible with integer mask count")
            false_off += missing
            wrong_on += wrong
            required += n
        predicted = math.prod(intended)
        actual = r["stage1_primary_heldout"]["phase_assignment_fraction"]
        expected.append(predicted)
        observed.append(actual)
        per_seed.append({
            "seed": r["seed"],
            "per_pass_required_gate_frequencies": intended,
            "per_pass_incorrect_gate_frequencies": opposite,
            "independence_predicted_full_board_fraction": predicted,
            "measured_full_board_fraction": actual,
            "difference": actual - predicted,
        })
    return {
        "required_gate_invocations": required,
        "false_off_count": false_off,
        "wrong_on_count": wrong_on,
        "false_off_fraction": false_off / required,
        "measured_full_board_fraction_mean": statistics.mean(observed),
        "independent_four_pass_product_mean": statistics.mean(expected),
        "mean_absolute_prediction_error": statistics.mean(abs(a - b) for a, b in zip(expected, observed)),
        "prediction_rmse": math.sqrt(statistics.mean((a - b) ** 2 for a, b in zip(expected, observed))),
        "cross_seed_pearson_correlation": statistics.correlation(expected, observed),
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    ans = analyze(json.loads(args.receipt.read_text(encoding="utf-8")))
    s = json.dumps(ans, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(s, encoding="utf-8")
    else:
        print(s)


if __name__ == "__main__":
    main()
