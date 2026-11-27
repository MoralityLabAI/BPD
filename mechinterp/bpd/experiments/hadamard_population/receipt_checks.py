"""Finite integrity checks for the complete HAD-05 result receipt roster.

Certificate algebra is checked by audit.py and archive_audit.py. This module
checks identities, declared budgets, event ordering, and the lineage references
that the historical receipts can support. It also accepts .json.gz files.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


ARMS = {
    "allocation": (
        "serial", "round_robin", "best", "ldt_best", "ucb", "ucb_transfer",
        "mlp", "no_ldt", "trm", "fixed", "sham", "live", "trm_transfer",
    ),
    "exact": ("best", "archive16", "archive64", "archive128", "random64"),
    "near": ("best", "exact64", "near64_noinject", "near64", "random_near64"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def starts(seed, m, population):
    """Reconstruct the original start bytes independently of the search code."""
    rng = np.random.default_rng(seed)
    xs = rng.choice([-1, 1], size=(population, 4, m))
    ss = rng.integers(0, 2**62, size=population, dtype=np.int64)
    digest = hashlib.sha256(xs.astype("int8").tobytes() + ss.tobytes()).hexdigest()
    return xs, digest


def matrix(x, m, name):
    require(isinstance(x, list) and len(x) == 4, f"{name} must contain four rows")
    for row in x:
        require(isinstance(row, list) and len(row) == m, f"{name} row length mismatch")
        require(all(type(v) is int and v in (-1, 1) for v in row), f"{name} entries must be signs")


def source_rows(sources, child, prior, xs, m, population, arity, randomized, name):
    """Validate four sources and any source sequence available at block zero."""
    require(isinstance(sources, list) and len(sources) == 4, f"{name} requires four source references")
    verified = 0
    for row, source in zip(child, sources):
        require(isinstance(source, (list, tuple)) and len(source) == arity, f"{name} source arity mismatch")
        slot, block, row_id = source[:3]
        count(slot, f"{name} source slot"); count(block, f"{name} source block"); count(row_id, f"{name} source row")
        require(slot < population and row_id < 4, f"{name} source index outside population")
        generation = count(source[3], f"{name} source generation") if arity == 4 else 0
        require(generation <= len(prior), f"{name} source refers to a future generation")
        if generation:
            parent = prior[generation - 1]
            require(parent["generation"] == generation and parent["slot"] == slot, f"{name} source generation/slot mismatch")
            origin = parent["sequences"][row_id]
        else:
            origin = xs[slot, row_id].tolist()
        if block == 0:
            if randomized:
                require(sum(row) == sum(origin), f"{name} randomized genesis row changed signed sum")
            else:
                require(row == origin, f"{name} genesis source row mismatch")
                verified += 1
    return verified


def validate(path, family):
    require(family in ARMS, f"unknown family {family!r}")
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    require(isinstance(data, dict), "receipt must be an object")
    a = data["args"]
    root = count(a["root"], "root")
    roots = count(a["roots"], "roots", 1)
    cases = count(a["cases"], "cases", 1)
    population = count(a["population"], "population", 1)
    rounds = count(a["rounds"], "rounds")
    chunk = count(a["chunk"], "chunk", 1)
    lengths = a["lengths"]
    require(isinstance(lengths, list) and len(lengths) > 0, "lengths must be a nonempty list")
    for m in lengths:
        count(m, "length", 3)
        require(m % 2 == 1, "length must be odd")
    require(len(lengths) == len(set(lengths)), "duplicate lengths in roster")
    expected = set(itertools.product(range(root, root + roots), lengths, range(cases), ARMS[family]))
    rows = data["rows"]
    require(isinstance(rows, list) and len(rows) == len(expected), "row count differs from full declared roster")
    seen = set()
    populations = {}
    injections_per_arm = Counter({arm: 0 for arm in ARMS[family]})
    rows_with_injections = max_injections = verified_sources = certificates = 0
    for index, z in enumerate(rows):
        name = f"row {index}"
        key = (z["root"], z["m"], z["case"], z["arm"])
        require(key in expected, f"{name} has an undeclared root/order/case/arm")
        require(key not in seen, f"{name} duplicates a roster entry")
        seen.add(key)
        unit = key[:3]
        m = key[1]
        seed = count(z["seed"], f"{name} seed")
        require(seed == key[0] * 100000 + m * 100 + key[2], f"{name} seed differs from declared schedule")
        if unit not in populations:
            populations[unit] = starts(seed, m, population)
        xs, digest = populations[unit]
        require(z["start_hash"] == digest, f"{name} start hash differs from regenerated starts")
        budget = rounds * chunk * 4 * m
        work_key = "candidate_scores" if family == "allocation" else "candidate_units"
        work = count(z[work_key], f"{name} total work")
        require(work <= budget, f"{name} exceeds declared budget")
        best = count(z["best"], f"{name} best energy")
        require(type(z["solved"]) is bool, f"{name} solved must be Boolean")
        require(z["solved"] == (z["certificate"] is not None), f"{name} solved/certificate mismatch")
        require(z["solved"] == (best == 0), f"{name} solved/best mismatch")
        require(isinstance(z["seconds"], (int, float)) and math.isfinite(z["seconds"]) and z["seconds"] >= 0, f"{name} invalid elapsed time")
        if z["solved"]:
            matrix(z["certificate"], m, f"{name} certificate")
            certificates += 1
        if family != "allocation":
            aw = count(z["archive_candidate_units"], f"{name} archive work")
            sw = count(z["search_candidate_scores"], f"{name} search work")
            require(work == aw + sw, f"{name} costs do not add up")
            require(sw % (4 * m) == 0, f"{name} search cost is not a whole flip")
            kind = z["solution_kind"]
            require((kind in ("archive", "native")) if z["solved"] else kind is None, f"{name} solution kind mismatch")
        else:
            aw = 0
            require(work % (4 * m) == 0, f"{name} candidate score cost is not a whole batch")
            offers = count(z["offers"], f"{name} transfer offers")
            transfers = count(z["transfers"], f"{name} transfers")
            require(transfers <= offers and offers * 4 * m <= work, f"{name} transfer counts exceed work")
            allocation = z["allocation"]
            require(len(allocation) == population, f"{name} allocation length mismatch")
            for value in allocation:
                count(value, f"{name} allocated blocks")
        trace = z["trace"]
        require(isinstance(trace, list) and len(trace) <= rounds, f"{name} trace exceeds round ceiling")
        last_work = last_aw = last_sw = 0
        for event in trace:
            require(isinstance(event, list) and len(event) == (4 if family == "allocation" else 5), f"{name} trace shape mismatch")
            slot = count(event[0], f"{name} trace slot")
            require(slot < population, f"{name} trace slot outside population")
            count(event[1], f"{name} trace best"); count(event[2], f"{name} trace energy")
            event_work = count(event[3], f"{name} trace work")
            require(last_work <= event_work <= work, f"{name} trace work is not monotone within total")
            if family != "allocation":
                event_aw = count(event[4], f"{name} trace archive work")
                require(last_aw <= event_aw <= min(aw, event_work), f"{name} trace archive work is not monotone within total")
                require((event_work - event_aw) % (4 * m) == 0, f"{name} trace search cost is not a whole flip")
                require(event_work - event_aw >= last_sw, f"{name} trace search work is not monotone")
                last_aw = event_aw
                last_sw = event_work - event_aw
            last_work = event_work
        if trace:
            require(last_work == work, f"{name} final trace work differs from total")
            if family != "allocation":
                require(last_aw == aw, f"{name} final trace archive work differs from total")
        if family == "allocation":
            require(count(z["rounds"], f"{name} rounds") == len(trace), f"{name} round count differs from trace")
            require(sum(allocation) <= len(trace), f"{name} allocations exceed trace decisions")
            require(bool(trace) or work == 0, f"{name} work lacks an allocation trace")
        for field in ("archive_rows", "archive_pairs", "archive_offers", "archive_duplicates", "archive_row_rejected", "archive_pair_checked", "archive_pair_rejected", "archive_cap_skips", "archive_payload_bytes", "archive_max_payload", "archive_heap_estimate", "near_checks", "buckets", "bucket_refs", "bucket_max_payload"):
            if field in z:
                count(z[field], f"{name} {field}")
        if "archive_max_payload" in z:
            require(z["archive_payload_bytes"] <= z["archive_max_payload"], f"{name} encoded archive exceeds modeled cap")
            require(z["archive_max_payload"] + z.get("bucket_max_payload", 0) <= a["archive_mib"] * (1 << 20), f"{name} modeled archive exceeds configured cap")
        injections = z["injections"] if family == "near" else []
        require(isinstance(injections, list), f"{name} injections must be a list")
        if family == "near":
            require(z["arm"] in ("near64", "random_near64") or len(injections) == 0, f"{name} injection appears in a non-injecting arm")
            require(len(injections) * 4 * m <= aw, f"{name} injections lack rebuild cost")
        previous_injection_work = 0
        for number, injection in enumerate(injections, 1):
            label = f"{name} injection {number}"
            require(count(injection["generation"], f"{label} generation", 1) == number, f"{label} generation is not sequential")
            slot = count(injection["slot"], f"{label} slot")
            require(slot < population, f"{label} slot outside population")
            count(injection["energy"], f"{label} energy")
            require(injection["energy"] < count(injection["replaced_best"], f"{label} replaced best"), f"{label} fails the replacement guard")
            injection_work = count(injection["work"], f"{label} work")
            require(previous_injection_work + 4 * m <= injection_work <= work, f"{label} work is out of order or lacks rebuild charge")
            previous_injection_work = injection_work
            matrix(injection["sequences"], m, f"{label} sequences")
            verified_sources += source_rows(injection["sources"], injection["sequences"], injections[:number - 1], xs, m, population, 4, z["arm"] == "random_near64", label)
        injections_per_arm[z["arm"]] += len(injections)
        rows_with_injections += int(bool(injections))
        max_injections = max(max_injections, len(injections))
        if "injection_count" in z:
            require(count(z["injection_count"], f"{name} injection count") == len(injections), f"{name} injection count differs from serialized events")
        if family == "near":
            generation = z["solution_injection"]
            if z["solution_kind"] == "native":
                count(generation, f"{name} solution generation")
                require(generation <= len(injections), f"{name} solution refers to a future generation")
                if generation:
                    slot = injections[generation - 1]["slot"]
                    require(all(t["slot"] != slot for t in injections[generation:]), f"{name} solved generation was later replaced")
            else:
                require(generation is None, f"{name} solution generation lacks a native solution")
        if family != "allocation" and z["solution_kind"] == "archive":
            arity = 4 if family == "near" and z["arm"] != "exact64" else 3
            verified_sources += source_rows(z["archive_sources"], z["certificate"], injections, xs, m, population, arity, z["arm"] in ("random64", "random_near64"), f"{name} archive certificate")
    require(seen == expected, "receipt misses declared roster entries")
    return {
        "status": "finite_integrity_pass",
        "family": family,
        "rows": len(rows),
        "paired_populations": len(populations),
        "certificates": certificates,
        "injections": sum(injections_per_arm.values()),
        "injections_per_arm": dict(injections_per_arm),
        "rows_with_injections": rows_with_injections,
        "maximum_injections_per_row": max_injections,
        "genesis_source_rows_verified": verified_sources,
        "lineage_limit": "Generation and slot references are checked. Block-zero source rows can be reconstructed. Later source rows need archived snapshots or flip logs, which these receipts omit; full source-row provenance and independent ancestry remain unverified.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("family", choices=tuple(ARMS))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = validate(args.path, args.family)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
