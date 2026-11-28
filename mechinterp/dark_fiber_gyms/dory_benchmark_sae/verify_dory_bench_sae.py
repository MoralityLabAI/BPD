"""Verify the local, compact Dory SAE benchmark archive without loading the model."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean
from zipfile import ZipFile


EXPECTED = {"i3_math", "i3_logic", "gsm8k", "math500", "rlt_revision"}


def require(condition, error):
    if not condition:
        raise ValueError(error)


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-10, rel_tol=0)


def check_score_rows(scores, roster, count, noops):
    require(len(scores) == len({row["id"] for row in scores}) == count, "Wrong native score count")
    require(sum(row["same_value_full_head_match"] is True for row in scores) == noops, "Same-value control lost")
    for row in scores:
        source = roster[row["root"]]
        require(source["split"] == "development" and source["suite"] == row["suite"], "Readout root custody changed")
        require(close(row["after"] - row["before"], row["delta"]), "Inconsistent logit margin")
        if row["arm"] == "same_value":
            require(row["dose"] == 0 and row["clean_head"] == row["patched_head"] and row["delta"] == 0,
                    "Same-value full-head replay changed")
        else:
            require(abs(row["dose"] - 0.05) < 1e-6 and row["radius_error"] <= 1e-6, "Physical dose or radius changed")


def audit(path):
    with ZipFile(path) as archive:
        require(archive.testzip() is None and len(set(archive.namelist())) == len(archive.namelist()), "Invalid ZIP")

        def load(name):
            return json.loads(archive.read(name))

        def rows(name):
            return [json.loads(x) for x in archive.read(name).splitlines()]

        report = load("dory_bench_sae_results.json")
        roster = {row["id"]: row for row in load("source_roster.json")}
        custody = Counter((r["suite"], r["split"]) for r in roster.values())
        require(len(roster) == 62 and {x[0] for x in custody} == EXPECTED
                and all(custody[suite, "development"] == 6 for suite in EXPECTED)
                and sum(custody[suite, "fit"] for suite in EXPECTED) == 32 and not custody["math500", "fit"],
                "Source cohort changed")
        primary = rows("primary_scores.jsonl")
        one, two = rows("sweep_l1_scores.jsonl"), rows("sweep_l2_scores.jsonl")
        for batch, count, noops in ((primary, 180, 60), (one, 270, 30), (two, 510, 30)):
            check_score_rows(batch, roster, count, noops)
        clean = rows("clean_scores.jsonl")
        require(len(clean) == 120 and set(Counter((r["suite"], r["loops"]) for r in clean).values()) == {12},
                "Clean selector cohort changed")
        open_raw = load("open_grades.json")["rows"]
        geometry = load("open_activation_geometry.json")
        require(len(open_raw) == len(geometry) == 20 and {r["id"] for r in open_raw} == {r["trace_id"] for r in geometry},
                "An open response lost its activation replay receipt")
        require(set(Counter((r["suite"], r["loops"]) for r in open_raw).values()) == {2}, "Open response cohort changed")
        future_effects = defaultdict(list)
        for row in one + two:
            if row["arm"] != "same_value":
                future_effects[row["suite"], row["loops"], row["layer"], row["arm"]].append(row)
        require(len(report["causal_arms"]) == len(future_effects) == 120, "Missing descriptive effect cell")
        labels = report["registration"]["answer_tokens"]
        for cell in report["causal_arms"]:
            key = tuple(cell[name] for name in ("suite", "loops", "site", "arm"))
            data = future_effects[key]
            require(len(data) == 6 and len({r["root"] for r in data}) == 3 and close(mean(r["delta"] for r in data), cell["mean_donor_margin_delta"]),
                    "Inconsistent root-balanced effect summary")
            before = [r for r in data if r["clean_argmax"] == labels[r["expected"]]]
            require(len(before) == cell["native_clean_correct"] and len(data) - len(before) == cell["native_clean_other"]
                    and sum(r["argmax"] != labels[r["expected"]] for r in before) == cell["native_correct_to_any_other"]
                    and sum(r["argmax"] == labels[r["expected"]] and r["clean_argmax"] != labels[r["expected"]] for r in data)
                    == cell["native_other_to_correct"], "Native recovery/corruption accounting changed")
        fingerprints = report["first_pass_write_pairs"]
        require(len(fingerprints) == 240 and sum(a == b for a, b in fingerprints.values()) == report["identical_first_pass_writes"],
                "First-pass write digest summary changed")
        effects = {(row["root"], row["expected"], row["arm"]): row for row in primary}
        for item in load("preflight_input.json")["pilot"]:
            root = item["unit_id"]
            contrasts = []
            for label in ("A", "B"):
                a, b = (effects[root, label, arm]["after"] for arm in ("action_sae", "sham"))
                contrasts.append(0.5 * (math.tanh(a / 4) - math.tanh(b / 4)))
            require(close(mean(contrasts), float(item["paired_effect"])), "Primary contrast does not match its raw rows")
        require(report["preflight"]["pilot_worlds"] == 30 and report["intervention_receipts"] == 960
                and rows("dory_bench_sae_decisions.jsonl")[-1]["phase"] == "close_panel", "Run closure changed")
    return {"source_roots": len(roster), "suites": sorted(EXPECTED), "verified_intervention_receipts": 960,
            "open_responses": len(open_raw),
            "matched_open_activation_replays": sum(x["status"] == "native_prefix_and_next_token_matched" for x in geometry),
            "identical_first_pass_writes": report["identical_first_pass_writes"], "preflight": report["preflight"]["decision"]}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.bundle), indent=2))
