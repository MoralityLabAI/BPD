"""Replay the released Oakhaven score arithmetic and audit its source locks."""

from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
FILES = {"development": (48, 16), "story_one_loop": (64, 4),
         "story_two_loops": (124, 4), "refit_scores": (140, 20)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def close(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-10)


def public_roster(raw):
    keys = ("id", "root", "group", "expected", "side", "context")
    cases = [json.loads(line) for line in raw.splitlines()]
    return [{key: row[key] for key in keys if key in row} for row in cases]


def check_locks(root, source_bundle=None):
    manifest = load(root / "receipts/manifest.json")
    for name, spec in manifest["files"].items():
        require(sha256((root / name).read_bytes()).hexdigest() == spec["sha256"], f"Changed receipt {name}")
    if source_bundle is not None:
        require(sha256(source_bundle.read_bytes()).hexdigest() == manifest["source_bundle_sha256"],
                "Source bundle digest changed")
        with ZipFile(source_bundle) as bundle:
            for name, spec in manifest["files"].items():
                raw = bundle.read(spec["bundle_member"])
                if spec.get("projection") == "public_case_roster":
                    require(load(root / name) == public_roster(raw), "Source roster projection changed")
                else:
                    require((root / name).read_bytes() == raw, f"Bundle member mismatch for {name}")
            for name, digest in manifest["local_only_source_sha256"].items():
                require(sha256(bundle.read(name)).hexdigest() == digest, f"Bundle source mismatch for {name}")
    return manifest


def check_rows(path, expected, expected_noops, roster, action_tokens, cohort):
    records = rows(path)
    require(len(records) == len({row["id"] for row in records}) == expected, f"Wrong receipt count at {path.name}")
    noops = 0
    for row in records:
        source, donor = (roster[row[key]] for key in ("source", "donor"))
        require(source["root"] == donor["root"] == row["root"] and source["group"] == donor["group"]
                and source["group"] in cohort and source["expected"] == row["expected"] != donor["expected"]
                and source.get("context", "fixture") == donor.get("context", "fixture") == row["context"],
                f"Bad custody on {row['id']}")
        require(close(row["donor_margin_after"] - row["donor_margin_before"], row["donor_margin_delta"]),
                f"Bad margin on {row['id']}")
        require(row["clean_exact_action_token"] is (row["clean_top_id"] == action_tokens[source["expected"]])
                and row["became_donor_token"] is (row["clean_exact_action_token"]
                                                  and row["top_id"] == action_tokens[donor["expected"]]),
                f"Bad action outcome on {row['id']}")
        if row["arm"] == "same_value":
            noops += 1
            require(row["native_head"] == row["clean_head"] and row["same_value_head_parity"] is True
                    and row["actual_relative_dose"] == 0 and row["donor_margin_delta"] == 0,
                    f"Bad same-value parity on {row['id']}")
        else:
            require(abs(row["actual_relative_dose"] - 0.05) <= 1e-6 and 0 <= row["relative_radius_error"] <= 1e-6,
                    f"Bad physical intervention on {row['id']}")
    require(noops == expected_noops, f"Wrong same-value count at {path.name}")
    return records


def check_effects(records, comparisons, key, report_key, margin_name, flip_name):
    groups = defaultdict(list)
    for row in records:
        if row["arm"] != "same_value":
            groups[key(row)].append(row)
    require(set(groups) == {report_key(row) for row in comparisons} and len(groups) == len(comparisons),
            "Effect summary has different cells than the receipts")
    for cell in comparisons:
        source = groups[report_key(cell)]
        require(close(mean(r["donor_margin_delta"] for r in source), cell[margin_name])
                and sum(r["became_donor_token"] for r in source) == cell[flip_name], "Effect summary has changed")
        if "case_count" in cell:
            require(len(source) == cell["case_count"] and len({r["root"] for r in source}) == cell["source_roots"],
                    "Refit case count or root count changed")
        else:
            require(len(source) == 2 and {r["expected"]: r["donor_margin_delta"] for r in source}
                    == cell["donor_margin_delta_by_action"], "Storyworld pair changed")
    return groups


def check_power(records, spec, report):
    arms = {(row["root"], row["expected"], row["arm"]): row for row in records}
    paired, shams = [], []
    local_roots = {row["root"] for row in records}
    require(local_roots == {row["unit_id"] for row in spec["pilot"]} and len(local_roots) == 8,
            "Pilot root set changed")
    for root in spec["pilot"]:
        unit, effects, controls = root["unit_id"], [], []
        for expected in ("deploy", "hold"):
            sae, sham = (arms[unit, expected, name] for name in ("sae", "sham"))
            effects.append(0.5 * (math.tanh(sae["donor_margin_after"] / 4) - math.tanh(sham["donor_margin_after"] / 4)))
            controls.append(0.5 * (math.tanh(sham["donor_margin_after"] / 4) - math.tanh(sham["donor_margin_before"] / 4)))
        require(close(mean(effects), float(root["paired_effect"]))
                and all(close(a, b) for a, b in zip(sorted(controls), sorted(map(float, root["sham_effects"])), strict=True)),
                "Power preflight input drifted")
        paired.append(mean(effects))
        shams.extend(controls)
    require(close(mean(paired), report["mean_pilot_effect"]) and close(max(map(abs, shams)), report["observed_worst_sham"])
            and report["decision"] == "REDESIGN_OR_PILOT_ONLY" and report["upper_sham_quantile"] is None
            and spec["heldout_sealed"] is False, "Power disposition changed")


def audit(root=ROOT, source_bundle=None):
    manifest = check_locks(root, source_bundle)
    directory = root / "receipts"
    cases = load(directory / "roster.json")
    roster = {row["id"]: row for row in cases}
    groups = {group: {row["root"] for row in cases if row["group"] == group}
              for group in ("fit", "development", "story")}
    require(len(cases) == len(roster) == 52 and len(set().union(*groups.values())) == 25
            and tuple(map(len, groups.values())) == (16, 8, 1), "Split custody changed")
    primary = load(directory / "story_results.json")
    refit = load(directory / "action_refit.json")
    clean = primary["clean_action"]
    tokens = clean[0]["action_token_ids"]
    require(len(clean) == 4 and all(row["action_token_ids"] == tokens
            and row["two_loops"]["original_prefill_full_head_matches"]
            and row["two_loops"]["original_format_prefix_and_action_greedy"] for row in clean),
            "Clean storyworld replay changed")
    cohorts = {"development": {"development"}, "story_one_loop": {"story"}, "story_two_loops": {"story"},
               "refit_scores": {"development", "story"}}
    records = {name: check_rows(directory / f"{name}.jsonl", count, noops, roster, tokens, cohorts[name])
               for name, (count, noops) in FILES.items()}
    cells = check_effects(records["story_one_loop"] + records["story_two_loops"], primary["case_pair_action_effects"],
                          lambda r: (r["loops"], r["context"], r["layer"], r["arm"]),
                          lambda r: (r["native_loops"], r["context"], r["physical_site"], r["arm"]),
                          "donor_margin_delta_mean", "donor_action_flips")
    check_effects(records["refit_scores"], refit["causal"], lambda r: (r["context"], r["layer"], r["arm"]),
                  lambda r: (r["context"], r["site"], r["arm"]), "average_donor_margin_shift", "donor_action_flips")
    digests = primary["first_loop_write_digests"]
    require(set(digests) == {r["id"] for r in records["story_one_loop"] if r["arm"] != "same_value"}
            and len(digests) == primary["identical_first_loop_writes_across_budgets"] == 60
            and all(row["one_loop"] == row["two_loops"] for row in digests.values()), "Matched first-pass writes changed")
    for row in primary["matched_first_loop_edits"]:
        means = [mean(r["donor_margin_delta"] for r in cells[budget, row["context"], row["first_loop_site"], row["arm"]])
                 for budget in (1, 2)]
        require(close(row["one_loop_margin_shift"], means[0]) and close(row["two_loop_margin_shift"], means[1])
                and close(row["added_loop_change_in_shift"], means[1] - means[0]), "Matched loop effect changed")
    check_power(records["development"], load(directory / "power_input.json"), primary["power_preflight"])
    decisions = rows(directory / "decisions.jsonl")
    require(len(decisions) == 10 and decisions[-1]["phase"] == "close_action_aligned_refit", "Decision roster changed")
    return {"source_bundle_sha256": manifest["source_bundle_sha256"], "fit_worlds": len(groups["fit"]),
            "development_worlds": len(groups["development"]), "storyworlds": len(groups["story"]),
            "verified_score_rows": sum(map(len, records.values())), "decision_records": len(decisions),
            "identical_first_pass_writes": len(digests), "power": primary["power_preflight"]["decision"]}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(source_bundle=args.source_bundle), indent=2, sort_keys=True))
