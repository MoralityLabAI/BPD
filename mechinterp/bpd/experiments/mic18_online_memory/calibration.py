"""Exact physical donor/mask calibration for a constructed recurrent register."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import finite_memory
from register import MemoryProgram, bits, observer_before_query, pack, word_bits

HERE = Path(__file__).resolve().parent
ARM_NAMES = ("clean", "whole_donor", "target_patch", "nuisance_patch", "self_sham")


def patch(recipient: tuple[int, ...], donor: tuple[int, ...], mask: tuple[int, ...]) -> tuple[int, ...]:
    """Copy exactly the selected coordinates from a fixed physical donor."""
    recipient = bits(recipient, len(recipient))
    donor = bits(donor, len(recipient))
    if len(set(mask)) != len(mask) or any(type(i) is not int or not 0 <= i < len(recipient) for i in mask):
        raise ValueError("mask must contain distinct valid coordinate indices")
    chosen = set(mask)
    return tuple(donor[i] if i in chosen else recipient[i] for i in range(len(recipient)))


def squared_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    if len(a) != len(b):
        raise ValueError("physical state widths differ")
    return sum((x-y)**2 for x, y in zip(a, b))


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def verify_lock() -> dict:
    lock = json.loads((HERE / "SOURCE_LOCK.json").read_text())
    for filename, expected in lock["files"].items():
        if hashlib.sha256((HERE / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source hash mismatch: {filename}")
    return lock


def reachable_states(program: MemoryProgram) -> dict[int, tuple[int, ...]]:
    states = {}
    for word in range(2**program.width):
        state = program.commit(word_bits(word, program.width))
        if state != word_bits(word, program.width):
            raise AssertionError("physical recurrent stream failed")
        states[word] = state
    return states


def make_pair(program: MemoryProgram, states: dict[int, tuple[int, ...]], word: int,
              target: int, nuisance: int) -> dict:
    width = program.width
    if target == nuisance or not (0 <= target < width and 0 <= nuisance < width):
        raise ValueError("distinct target and nuisance coordinates required")
    recipient = states[word]
    donor_word = word ^ (1 << (width-1-target)) ^ (1 << (width-1-nuisance))
    donor = states[donor_word]
    masks = {"clean": (), "whole_donor": tuple(range(width)),
             "target_patch": (target,), "nuisance_patch": (nuisance,), "self_sham": (target,)}
    targets = {"recipient": word_bits(word, width), "donor": word_bits(donor_word, width)}
    answers = {"recipient": tuple(program.query(recipient, q) for q in range(width)),
               "donor": tuple(program.query(donor, q) for q in range(width))}
    arms = {}
    rejected = []
    if answers != targets:
        rejected.append("base_task_error")
    if observer_before_query(recipient) != observer_before_query(donor):
        rejected.append("current_observer_mismatch")
    for name, mask in masks.items():
        source = recipient if name == "self_sham" else donor
        state = patch(recipient, source, mask)
        witness = pack(state)
        reachable = states.get(witness) == state
        if not reachable:
            rejected.append("unreachable_" + name)
        if any(state[i] != recipient[i] for i in range(width) if i not in mask):
            rejected.append("changed_outside_mask_" + name)
        outputs = tuple(program.query(state, q) for q in range(width))
        recipient_correct = tuple(int(a == b) for a, b in zip(outputs, targets["recipient"]))
        donor_agreement = tuple(int(a == b) for a, b in zip(outputs, answers["donor"]))
        eligible = tuple(int(a != b) for a, b in zip(answers["recipient"], answers["donor"]))
        recovery = tuple(donor_agreement[q] if eligible[q] else None for q in range(width))
        arms[name] = {"mask": mask, "mask_cardinality": len(mask), "state": state,
                      "source": "recipient" if name == "self_sham" else "same_donor",
                      "squared_displacement": squared_distance(recipient, state),
                      "reachable": reachable, "reachable_word_witness": witness,
                      "observer": observer_before_query(state), "outputs": outputs,
                      "recipient_correct": recipient_correct, "donor_agreement": donor_agreement,
                      "donor_effect_recovery": recovery}
    if arms["target_patch"]["squared_displacement"] != 1 or arms["nuisance_patch"]["squared_displacement"] != 1:
        rejected.append("singleton_dose_mismatch")
    if arms["whole_donor"]["squared_displacement"] != 2:
        rejected.append("whole_donor_dose_mismatch")
    if any(arm["observer"] != 0 for arm in arms.values()):
        rejected.append("prequery_observer_changed")
    return {"width": width, "recipient_word": word, "donor_word": donor_word,
            "target": target, "nuisance": nuisance, "phase": width,
            "recipient_state": recipient, "donor_state": donor,
            "recipient_targets": targets["recipient"], "donor_targets": targets["donor"],
            "recipient_answers": answers["recipient"], "donor_answers": answers["donor"],
            "admitted": not rejected, "rejection_reasons": rejected, "arms": arms}


def empty_counts() -> dict[str, int]:
    return {"queries": 0, "recipient_correct": 0, "donor_agreement": 0,
            "effect_eligible_queries": 0, "donor_effect_recovered": 0,
            "primary_queries": 0, "primary_recipient_correct": 0,
            "primary_donor_effect_recovered": 0,
            "nuisance_queries": 0, "nuisance_recipient_correct": 0,
            "nuisance_donor_effect_recovered": 0}


def alias_census(program: MemoryProgram, states: dict[int, tuple[int, ...]]) -> dict:
    width = program.width
    signatures = {word: tuple(program.query(state, q) for q in range(width))
                  for word, state in states.items()}
    pairs = disagreements = 0
    for a in range(2**width):
        for b in range(a+1, 2**width):
            if observer_before_query(states[a]) != observer_before_query(states[b]):
                raise AssertionError("constant observer changed")
            conflicts = sum(x != y for x, y in zip(signatures[a], signatures[b]))
            if not conflicts:
                raise AssertionError("distinct stored words lost a future query distinction")
            pairs += 1
            disagreements += conflicts
    return {"current_observer_classes": 1,
            "future_answer_signatures": len(set(signatures.values())),
            "equal_observer_pairs_with_future_conflict": pairs,
            "pair_query_disagreements": disagreements,
            "witness": {"recipient_word": 0, "donor_word": 1, "query": width-1,
                        "current_observer_pair": [0, 0], "future_answer_pair": [0, 1]}}


def exhaustive(records: list | None = None) -> dict:
    protocol = json.loads((HERE / "PROTOCOL.json").read_text())
    lock = verify_lock()
    by_width = []
    all_counts = {name: empty_counts() for name in ARM_NAMES}
    digest = hashlib.sha256()
    pairs = admitted = rejected = query_cases = 0
    physical_pairs = set()
    for width in protocol["widths"]:
        program = MemoryProgram.build(width)
        states = reachable_states(program)
        core = finite_memory.evaluate(finite_memory.make_register(width))
        if (core["score"], core["total"]) != (width * 2**width, width * 2**width):
            raise AssertionError("shared finite-register ceiling mismatch")
        counts = {name: empty_counts() for name in ARM_NAMES}
        width_pairs = 0
        width_physical_pairs = set()
        for word in range(2**width):
            for target in range(width):
                for nuisance in range(width):
                    if nuisance == target:
                        continue
                    pair = make_pair(program, states, word, target, nuisance)
                    physical_pairs.add((width, word, pair["donor_word"]))
                    width_physical_pairs.add((word, pair["donor_word"]))
                    pairs += 1
                    width_pairs += 1
                    query_cases += width
                    admitted += int(pair["admitted"])
                    rejected += int(not pair["admitted"])
                    digest.update(canonical(pair) + b"\n")
                    if records is not None:
                        records.append(pair)
                    for name, arm in pair["arms"].items():
                        row = counts[name]
                        row["queries"] += width
                        row["recipient_correct"] += sum(arm["recipient_correct"])
                        row["donor_agreement"] += sum(arm["donor_agreement"])
                        row["effect_eligible_queries"] += sum(x is not None for x in arm["donor_effect_recovery"])
                        row["donor_effect_recovered"] += sum(x or 0 for x in arm["donor_effect_recovery"])
                        row["primary_queries"] += 1
                        row["primary_recipient_correct"] += arm["recipient_correct"][target]
                        row["primary_donor_effect_recovered"] += arm["donor_effect_recovery"][target] or 0
                        row["nuisance_queries"] += 1
                        row["nuisance_recipient_correct"] += arm["recipient_correct"][nuisance]
                        row["nuisance_donor_effect_recovered"] += arm["donor_effect_recovery"][nuisance] or 0
        for name in ARM_NAMES:
            for key, value in counts[name].items():
                all_counts[name][key] += value
        if width_pairs != 2**width * width * (width-1):
            raise AssertionError("incomplete pair census")
        for name, (task_errors, recovered) in {
                "clean": (0, 0), "whole_donor": (2, 2),
                "target_patch": (1, 1), "nuisance_patch": (1, 1), "self_sham": (0, 0)}.items():
            if counts[name]["recipient_correct"] != width_pairs * (width-task_errors):
                raise AssertionError("recipient task census differs from coordinate-level oracle")
            if counts[name]["donor_effect_recovered"] != width_pairs * recovered:
                raise AssertionError("donor effect census differs from coordinate-level oracle")
        by_width.append({"width": width, "pairs": width_pairs,
                         "unique_ordered_physical_pairs": len(width_physical_pairs),
                         "cross_query_cases": width_pairs*width,
                         "arms": counts, "aliases": alias_census(program, states),
                         "physical_resources": program.resource_account(),
                         "patch_coordinate_copies": {"clean": 0, "whole_donor": width,
                                                      "target_patch": 1, "nuisance_patch": 1, "self_sham": 1}})
    if (rejected or pairs != protocol["pair_count"] or query_cases != protocol["cross_query_count"]
            or len(physical_pairs) != protocol["unique_ordered_physical_pairs"]):
        raise AssertionError({"pairs": pairs, "admitted": admitted, "rejected": rejected, "queries": query_cases})
    return {"schema": "MIC-18/BPD-register-calibration-results/v1",
            "status": "PASS_EXACT_CONSTRUCTED_CALIBRATION", "evidence_class": "exact finite domain",
            "generated_pairs": pairs, "admitted_pairs": admitted, "rejected_pairs": rejected,
            "pair_count_unit": "recipient word with ordered target and nuisance roles",
            "unique_ordered_physical_pairs": len(physical_pairs),
            "cross_query_cases": query_cases, "arm_query_scores": query_cases * len(ARM_NAMES),
            "prequery_observer": "engineered constant zero", "optimizer_updates": 0,
            "trained_models": 0, "independent_trained_roots": 0,
            "population_gate": "NO_TRAINED_ROOTS_OR_SHAM_CALIBRATION",
            "source_files": lock["files"], "provenance": lock["provenance"],
            "raw_cases_sha256": digest.hexdigest(), "arms": all_counts, "by_width": by_width}


def archive(records: list) -> bytes:
    raw = b"".join(canonical(row) + b"\n" for row in records)
    return base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)) + b"\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    records = []
    result = exhaustive(records)
    encoded = archive(records)
    receipt, raw = HERE / "RESULTS.json", HERE / "cases.jsonl.gz.b64"
    if args.check:
        if json.loads(receipt.read_text()) != result:
            raise SystemExit("stored BPD receipt differs from exhaustive replay")
        stored_cases = gzip.decompress(base64.b64decode(raw.read_bytes()))
        replay_cases = b"".join(canonical(row) + b"\n" for row in records)
        if stored_cases != replay_cases:
            raise SystemExit("stored BPD paired cases differ from exhaustive replay")
    if args.write:
        receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raw.write_bytes(encoded)
    print(json.dumps({"status": result["status"], "pairs": result["generated_pairs"],
                      "cross_query_cases": result["cross_query_cases"],
                      "rejected_pairs": result["rejected_pairs"],
                      "raw_cases_sha256": result["raw_cases_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
