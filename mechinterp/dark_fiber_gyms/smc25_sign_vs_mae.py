"""BPD-OUT05: sign-sensitive causal decisions vs MAE after SMC-25.

No trained neural model is imported. Theoretical results use exact Fraction
arithmetic; the optional receipt CLI replays SMC-25's eight *exploratory*
sign-difference contrasts with whole-puzzle bootstrap resampling. Those
contrasts were observed before this analysis was designed.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import random
import statistics


SMC25_SCORED_SHA256 = "6e58461acd9c012f3ff9324be1420dc693cb3c318419c9a9bebb2376af2c878c"
SMC25_FINAL_RUN = "https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44113889861"
DEFAULT_BOOT_SEED = 20261117
DEFAULT_BOOT_DRAWS = 4000
FAMILIES = ("smc25_novel30", "smc25_novel34")
COMPONENTS = ("newB_501_SAE_71", "newB_501_SAE_89",
              "newB_701_SAE_71", "newB_701_SAE_89")


def sign(value):
    x = F(value)
    return 1 if x > 0 else -1 if x < 0 else 0


def weighted_median_exact(effects, weights):
    """Deterministic left weighted median; tie resolves to smaller effect."""
    d = [F(x) for x in effects]
    p = [F(x) for x in weights]
    if len(d) < 2 or len(d) != len(p) or any(x < 0 for x in p) or sum(p) != 1:
        raise ValueError("two or more effects and valid probability weights required")
    items = sorted(zip(d, range(len(d)), p))
    c = F(0)
    for value, _, mass in items:
        c += mass
        if c >= F(1, 2):
            return value
    raise ArithmeticError("median distribution misnormalized")


def sign_stability_certificate(effects, p, q):
    """For class effects d_i != 0, |E_p sign(d)-E_q sign(d)| <= 2 TV.

    A nonzero true sign margin strictly exceeding 2 TV guarantees the
    calibrated/perturbed probability vector gives the same sign decision.
    This is a bound on conditional predicted effects, not a real-world
    generalization guarantee for a learned probability model.
    """
    d = [F(x) for x in effects]
    a = [F(x) for x in p]
    b = [F(x) for x in q]
    if len(d) < 2 or len(d) != len(a) or len(d) != len(b):
        raise ValueError("candidate effects and probabilities must align")
    if any(x == 0 for x in d):
        raise ValueError("exact sign decision requires nonzero effects")
    if any(x < 0 for x in a+b) or sum(a) != 1 or sum(b) != 1:
        raise ValueError("probability mass must sum to one")
    s_a = sum(x*sign(y) for x,y in zip(a,d))
    s_b = sum(x*sign(y) for x,y in zip(b,d))
    tv = sum(abs(x-y) for x,y in zip(a,b))/2
    assert abs(s_a-s_b) <= 2*tv
    guaranteed = abs(s_a) > 2*tv
    if guaranteed:
        assert sign(s_a) == sign(s_b)
    return {"true_signed_mass":str(s_a),"proxy_signed_mass":str(s_b),
            "total_variation":str(tv),"guaranteed_same_sign":guaranteed,
            "actual_same_sign":sign(s_a) == sign(s_b)}


def exact_sign_mae_counterexample():
    """A label distribution with improved sign classification and worse MAE.

    Nine candidate effects and calibrated vs uniform weights are fixed.
    Actual positive cases concentrate at +1 and negative cases at -10.
    A badly biased calibrated distribution chooses +10. Its sign
    classification is better than the uniform median's sign, while its
    absolute error is larger on the same cases.
    """
    d = [F(x) for x in (-10,-10,-1,-1,-1,1,1,10,10)]
    p = [F(0) for _ in d]
    p[0] = F(3,10)
    p[5] = F(7,10)
    q = [F(0) for _ in d]
    q[7] = F(1)
    uniform = [F(1,9) for _ in d]
    med_cal = weighted_median_exact(d,q)
    med_unif = weighted_median_exact(d,uniform)
    mae = lambda guess: sum(weight*abs(truth-guess) for truth,weight in zip(d,p))
    accuracy = lambda guess: sum(weight for truth,weight in zip(d,p) if sign(truth)==sign(guess))
    assert med_cal == 10 and med_unif == -1
    assert mae(med_cal) == F(123,10) and mae(med_unif) == F(41,10)
    assert accuracy(med_cal) == F(7,10) and accuracy(med_unif) == F(3,10)
    return {
        "candidate_effects":[int(x) for x in d],
        "actual_class_probabilities":[str(x) for x in p],
        "calibrated_class_probabilities":[str(x) for x in q],
        "calibrated_median":str(med_cal),"uniform_median":str(med_unif),
        "calibrated_sign_accuracy":str(accuracy(med_cal)),
        "uniform_sign_accuracy":str(accuracy(med_unif)),
        "calibrated_expected_MAE":str(mae(med_cal)),
        "uniform_expected_MAE":str(mae(med_unif)),
    }


def _correct_signed_effect(truth, prediction):
    return int(truth != 0.0 and prediction != 0.0 and
               (truth > 0.0) == (prediction > 0.0))


def replay_exploratory_sign_effects(scored, draws=DEFAULT_BOOT_DRAWS,
                                    seed=DEFAULT_BOOT_SEED):
    """Post-result sign-gap bootstrap; cluster by 64 independent puzzle IDs.

    Uses per-puzzle eligible-goal sign advantage as the observation. Each
    bootstrap resamples all 64 puzzle IDs and excludes empty puzzle
    clusters from the mean. This was NOT the preregistered SMC-25 gate.
    """
    if scored.get("schema") != "SMC-25/finite-native-all-candidate-scored-20261117-v1":
        raise ValueError("expected frozen original SMC-25 scored receipt")
    if scored.get("n_unique_puzzles") != 128 or scored.get("n_goal_slots") != 4096:
        raise ValueError("source split or total goal count changed")
    if set(scored.get("test_data", {})) != set(FAMILIES):
        raise ValueError("source family split changed")
    output = []
    for family in FAMILIES:
        if set(scored["test_data"][family]) != set(COMPONENTS):
            raise ValueError("original independent teacher/SAE roster changed")
        for component in COMPONENTS:
            item = scored["test_data"][family][component]
            rows = item["per_puzzle"]
            if len(rows) != 64 or sorted(row["puzzle_index"] for row in rows) != list(range(64)):
                raise ValueError("puzzle cluster identities missing")
            deltas = []
            n = right_cal = right_uniform = 0
            for puzzle in rows:
                differences = []
                if len(puzzle["goals"]) != 8:
                    raise ValueError("eight fixed source goal positions required")
                for goal in puzzle["goals"]:
                    if goal["status"] != "eligible":
                        continue
                    obs = float(goal["observed"]["true_D"])
                    pred = goal["predictions"]
                    if set(("calibrated_median","uniform_median")) - set(pred):
                        raise ValueError("frozen sign predictor arm missing")
                    ca = _correct_signed_effect(obs, float(pred["calibrated_median"]))
                    un = _correct_signed_effect(obs, float(pred["uniform_median"]))
                    right_cal += ca; right_uniform += un; n += 1
                    differences.append(ca-un)
                deltas.append(statistics.mean(differences) if differences else None)
            if n != item["summary"]["eligible_goals"]:
                raise ValueError("original eligible count changed")
            if right_cal != item["summary"]["calibrated_median_sign_correct"] or right_uniform != item["summary"]["uniform_median_sign_correct"]:
                raise ValueError("registered sign tally fails independent replay")
            nonempty = [d for d in deltas if d is not None]
            observed = statistics.mean(nonempty)
            rng = random.Random(seed)
            boots = []
            for _ in range(draws):
                sampled = [deltas[rng.randrange(64)] for _ in range(64)]
                cohort = [x for x in sampled if x is not None]
                if cohort:
                    boots.append(statistics.mean(cohort))
            boots.sort()
            if len(boots) != draws:
                raise ValueError("sign bootstrap lost its source denominator")
            result = {
                "family":family,"component":component,
                "eligible_goals":n,"eligible_puzzles":len(nonempty),
                "calibrated_sign_correct":right_cal,
                "uniform_sign_correct":right_uniform,
                "calibrated_minus_uniform_goal_count":right_cal-right_uniform,
                "puzzle_mean_sign_gain":observed,
                "posthoc_puzzle_bootstrap95":[boots[int(.025*draws)],
                                            boots[int(.975*draws)-1]],
                "posthoc_status":"EXPLORATORY_AFTER_SMC25_RESULT",
            }
            output.append(result)
    return {
        "schema":"BPD-OUT05/source-locked-SMC25-exploratory-sign-audit-v1",
        "source_run":SMC25_FINAL_RUN,
        "source_scored_sha256":SMC25_SCORED_SHA256,
        "n_source_unique_puzzles":128,
        "n_eligible_goal_SAE_measurements":sum(x["eligible_goals"] for x in output),
        "bootstrap_seed":seed,"bootstrap_draws":draws,
        "original_primary_MAE_scientific_gate":"FAIL_0_OF_4",
        "exploratory_sign_by_cell":output,
        "inferential_limit":"8 cells reuse 2 independently trained teachers and 128 puzzle sources. 95% intervals are retrospective, unadjusted, and do not replace the registered MAE gate.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-other-source", action="store_true")
    args = parser.parse_args()
    raw = args.scored.read_bytes()
    if not args.allow_other_source and hashlib.sha256(raw).hexdigest() != SMC25_SCORED_SHA256:
        raise ValueError("original published SMC25 scored.json SHA-256 mismatch")
    report = replay_exploratory_sign_effects(json.loads(raw))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status":"EXPLORATORY_REPLAY_PASS",
                      "source_puzzles":report["n_source_unique_puzzles"],
                      "goal_measurements":report["n_eligible_goal_SAE_measurements"],
                      "out":str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
