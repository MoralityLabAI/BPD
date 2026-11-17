"""AMM-01: synthetic BPD-style recurrent causal-probe memory transfer.

Pure Python 3.11+; no model checkpoints, sealed data, or external services.
See PROTOCOL.md, committed before source or prospective scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ID = "BPD-AMM-01"
VERSION = "AMM01/v1"
FAMILIES = ("wren", "basalt")
HORIZONS = (1, 2)
TRAIN_ROOTS = 64
DEV_ROOTS = 64
CONFIRM_ROOTS = 512
CASES = 256
SIGMA = 0.55
EPSILON = 0.035
BOOTSTRAPS = 2000
MIN_EFFECT = 0.15
ALPHA = 0.05
COMPARISONS = ("memory_vs_pooled", "memory_vs_swapped", "memory_vs_random")
HERE = Path(__file__).resolve().parent


def canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(data: Any) -> str:
    return hashlib.sha256(canonical(data).encode("utf-8")).hexdigest()


def rng_for(*parts: object) -> random.Random:
    key = "AMM01/" + "/".join(map(str, parts))
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


@dataclass(frozen=True)
class World:
    root: str
    family: str
    sign: int
    gain: float


def world(phase: str, index: int) -> World:
    if phase not in ("train", "development", "confirmation", "confirm_fresh"):
        raise ValueError("unknown phase")
    cap = {"train": TRAIN_ROOTS, "development": DEV_ROOTS, "confirmation": CONFIRM_ROOTS, "confirm_fresh": CONFIRM_ROOTS}[phase]
    if not 0 <= index < cap:
        raise ValueError("world index out of range")
    r = rng_for("world", phase, index)
    return World(f"{phase}/{index:04d}", FAMILIES[index % 2], r.choice((-1, 1)), r.uniform(0.45, 0.85))


def operator(w: World) -> tuple[tuple[float, ...], ...]:
    """A rank <= 2, physical, shared 4x4 recurrent operator."""
    s, g = w.sign, w.gain
    if w.family == "wren":
        return ((0., s*g, -s*g, 0.), (0., 0., 0., 0.),
                (0., 0., 0., 0.), (0., 0., 0., 0.))
    if w.family == "basalt":
        return ((0., s*EPSILON, -s*EPSILON, 1.), (0., 0., 0., 0.),
                (0., 0., 0., 0.), (0., s*g, -s*g, 0.))
    raise ValueError("invalid family")


def matvec(m: tuple[tuple[float, ...], ...], v: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sum(a*b for a, b in zip(row, v)) for row in m)


def exact_probe(w: World, horizon: int) -> float:
    if horizon not in HORIZONS:
        raise ValueError("one of two registered horizons required")
    d = (0., 1 / math.sqrt(2), -1 / math.sqrt(2), 0.)
    m = operator(w)
    h = d
    for _ in range(horizon):
        h = matvec(m, h)
    return h[0]


def observations(w: World, case: int) -> tuple[float, float, float]:
    """Paired common noise for the two horizons plus independent sham noise."""
    if not 0 <= case < CASES:
        raise ValueError("case index out of range")
    r = rng_for("observation", w.root, case)
    return (r.gauss(0, SIGMA), r.gauss(0, SIGMA), r.gauss(0, SIGMA))


def correct(signal: float, noise: float, truth: int) -> int:
    return int((1 if signal + noise > 0.0 else -1) == truth)


def learn_memory() -> dict[str, Any]:
    """Fit routing only from independent training worlds, including failure counts."""
    counts = {family: {"1": 0, "2": 0, "cases": 0, "roots": 0} for family in FAMILIES}
    for index in range(TRAIN_ROOTS):
        w = world("train", index)
        t1, t2 = exact_probe(w, 1), exact_probe(w, 2)
        row = counts[w.family]
        row["roots"] += 1
        for case in range(CASES):
            n1, n2, _ = observations(w, case)
            row["1"] += correct(t1, n1, w.sign)
            row["2"] += correct(t2, n2, w.sign)
            row["cases"] += 1
    mapping = {f: (1 if counts[f]["1"] >= counts[f]["2"] else 2) for f in FAMILIES}
    pooled1 = sum(counts[f]["1"] for f in FAMILIES)
    pooled2 = sum(counts[f]["2"] for f in FAMILIES)
    memory = {"schema": "BPD-AMM-01/training-memory/v1", "source_partition": "train",
              "independent_roots": TRAIN_ROOTS, "cases_per_root": CASES,
              "training_correct": counts, "family_horizon": mapping,
              "pooled_horizon": 1 if pooled1 >= pooled2 else 2,
              "selection": "maximum training hits; ties favor horizon 1"}
    memory["sha256"] = digest(memory)
    return memory


def choose_horizon(arm: str, family: str, memory: dict[str, Any], root: str, case: int) -> int:
    """The policy boundary: only family, frozen training memory and local coin."""
    if family not in FAMILIES:
        raise ValueError("unknown family")
    if arm == "memory":
        return memory["family_horizon"][family]
    if arm in ("pooled", "sham"):
        return memory["pooled_horizon"]
    if arm == "swapped":
        return memory["family_horizon"][FAMILIES[1 - FAMILIES.index(family)]]
    if arm == "random":
        return rng_for("policy", root, case).choice(HORIZONS)
    raise ValueError("unregistered arm")


def evaluate_root(w: World, memory: dict[str, Any]) -> dict[str, Any]:
    arms = ("memory", "pooled", "swapped", "random", "sham")
    traces = {a: [] for a in arms}
    probes = {a: [0, 0] for a in arms}
    y = {1: exact_probe(w, 1), 2: exact_probe(w, 2)}
    for case in range(CASES):
        n1, n2, ns = observations(w, case)
        for arm in arms:
            horizon = choose_horizon(arm, w.family, memory, w.root, case)
            probes[arm][horizon - 1] += 1
            noise = ns if arm == "sham" else (n1 if horizon == 1 else n2)
            traces[arm].append(str(correct(y[horizon], noise, w.sign)))
    bits = {a: "".join(traces[a]) for a in arms}
    return {"root": w.root, "family": w.family, "latent_sign": w.sign,
            "gain": w.gain, "exact_probe": {"1": y[1], "2": y[2]},
            "hits": {a: bits[a].count("1") for a in arms},
            "horizon_counts": {a: probes[a] for a in arms},
            "case_bits": bits}


def cohort(phase: str, memory: dict[str, Any] | None = None) -> dict[str, Any]:
    if phase not in ("development", "confirm_fresh"):
        raise ValueError("cohort must be development or fresh confirmation")
    memory = memory or learn_memory()
    n = DEV_ROOTS if phase == "development" else CONFIRM_ROOTS
    rows = [evaluate_root(world(phase, i), memory) for i in range(n)]
    root_ids = [r["root"] for r in rows]
    if len(root_ids) != len(set(root_ids)) or any(not x.startswith(phase + "/") for x in root_ids):
        raise AssertionError("duplicate/cross-partition test root")
    totals = {a: sum(r["hits"][a] for r in rows) for a in ("memory", "pooled", "swapped", "random", "sham")}
    return {"schema": VERSION, "experiment": ID, "phase": phase,
            "independent_units": n, "cases_per_unit": CASES,
            "total_scored_cases": n * CASES,
            "physical_intervention_dose_l2": 1.0,
            "family_balance": {f: sum(r["family"] == f for r in rows) for f in FAMILIES},
            "training_memory": memory, "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "totals": totals, "rows": rows}


def preflight_spec(dev: dict[str, Any]) -> dict[str, Any]:
    if dev["phase"] != "development" or dev["independent_units"] != DEV_ROOTS:
        raise ValueError("development cohort only")
    pilot = []
    for r in dev["rows"]:
        h = r["hits"]
        pilot.append({"unit_id": r["root"],
                      "paired_effect": f"{h['memory'] - h['pooled']}/{CASES}",
                      "sham_effects": [f"{h['sham'] - h['pooled']}/{CASES}"]})
    return {"schema_version": 1, "experiment_id": ID,
            "independent_unit": "one phase-disjoint synthetic recurrent world/root",
            "alpha": "0.05", "scientific_min_effect": "0.15",
            "target_effect": "0.25", "measurement_resolution": f"1/{CASES}",
            "sham_quantile": "0.95", "sham_confidence": "0.95",
            "planned_comparisons": len(COMPARISONS), "confirm_units_planned": CONFIRM_ROOTS,
            "pilot_from_development_only": True, "heldout_sealed": True,
            "units_independent": True, "pilot": pilot}


def sign_p_one_sided(differences: list[int]) -> float:
    positive = sum(d > 0 for d in differences)
    nonzero = sum(d != 0 for d in differences)
    if not nonzero:
        return 1.0
    return sum(math.comb(nonzero, k) for k in range(positive, nonzero+1)) / 2**nonzero


def holm(ps: dict[str, float]) -> dict[str, float]:
    ordered = sorted(ps, key=lambda name: (ps[name], name))
    ans: dict[str, float] = {}
    floor = 0.0
    for index, name in enumerate(ordered):
        floor = max(floor, min(1.0, (len(ordered)-index)*ps[name]))
        ans[name] = floor
    return ans


def bootstrap_interval(diffs: list[int], name: str) -> list[float]:
    r = rng_for("bootstrap", name)
    n = len(diffs)
    vals = []
    for _ in range(BOOTSTRAPS):
        vals.append(sum(diffs[r.randrange(n)] for _ in range(n)) / (CASES * n))
    vals.sort()
    return [vals[50], vals[1949]]


def analyze(confirm: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    if confirm["phase"] != "confirm_fresh" or confirm["independent_units"] != CONFIRM_ROOTS:
        raise ValueError("wrong confirmation cohort")
    if preflight.get("decision") != "READY_FOR_CONFIRMATION" or preflight.get("experiment_id") != ID:
        raise ValueError("EvalMath strict preflight has not authorized confirmation")
    arms = {"memory_vs_pooled": "pooled", "memory_vs_swapped": "swapped", "memory_vs_random": "random"}
    diffs = {name: [r["hits"]["memory"] - r["hits"][other] for r in confirm["rows"]]
             for name, other in arms.items()}
    ps = {name: sign_p_one_sided(diffs[name]) for name in COMPARISONS}
    adj = holm(ps)
    results = {}
    for name in COMPARISONS:
        a = diffs[name]
        results[name] = {"mean_gain": sum(a)/(CASES*len(a)),
                         "positive_roots": sum(x>0 for x in a),
                         "negative_roots": sum(x<0 for x in a),
                         "tied_roots": sum(x==0 for x in a),
                         "one_sided_exact_sign_p": ps[name],
                         "holm_adjusted_p": adj[name],
                         "root_cluster_bootstrap_95_ci": bootstrap_interval(a, name)}
    floor = max(MIN_EFFECT, preflight["mechanistic_variation_floor"])
    primary = results[COMPARISONS[0]]
    passed = primary["root_cluster_bootstrap_95_ci"][0] > floor and primary["holm_adjusted_p"] < ALPHA
    return {"schema": "BPD-AMM-01/analysis/v1", "evidence_class": "prospective synthetic strategy-memory transfer",
            "independent_confirmation_worlds": CONFIRM_ROOTS,
            "repeated_cases_per_world": CASES,
            "physical_dose_l2": 1.0, "scientific_min_effect": MIN_EFFECT,
            "matched_sham_95q_95confidence_floor": preflight["upper_sham_quantile"],
            "mechanistic_variation_floor": floor,
            "planned_actual_comparisons": len(COMPARISONS),
            "contrasts": results, "primary_gate_pass": passed,
            "boundary": "Engineered rank<=2 synthetic tied linear systems and fixed policy agents; no LLM or trained neural BPD mechanism."}


def bundle(phase: str, preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    # Hard preflight admission gate. Do not touch the fresh generator when absent.
    if phase not in ("development", "confirm_fresh"):
        raise ValueError("historical confirmation is engineering-only; use confirm_fresh")
    if phase == "confirm_fresh" and (
        preflight is None
        or preflight.get("decision") != "READY_FOR_CONFIRMATION"
        or preflight.get("experiment_id") != ID
        or preflight.get("planned_confirmation_worlds") != CONFIRM_ROOTS
    ):
        raise ValueError("fresh confirmation requires a matching strict EvalMath preflight")
    m = learn_memory()
    run = cohort(phase, m)
    if phase == "development":
        run["evalmath_preflight_input"] = preflight_spec(run)
    else:
        assert preflight is not None
        run["evalmath_preflight_sha256"] = digest(preflight)
        run["analysis"] = analyze(run, preflight)
    return run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("development", "confirm_fresh"), required=True)
    p.add_argument("--preflight", type=Path, help="EvalMath strict preflight JSON for confirmation")
    p.add_argument("--out", type=Path, help="write reproducible JSON; refuse overwrite")
    p.add_argument("--check", type=Path, help="recompute and verify an existing receipt")
    args = p.parse_args()
    pf = json.loads(args.preflight.read_text()) if args.preflight else None
    obj = bundle(args.phase, pf)
    if args.check is not None:
        locked = json.loads(args.check.read_text())
        if canonical(obj) != canonical(locked):
            raise AssertionError("locked receipt differs from source replay")
        print(f"{ID} {args.phase} locked replay PASS")
    elif args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as file:
            json.dump(obj, file, sort_keys=True, indent=2)
            file.write("\n")
        print(f"{ID} {args.phase}: {obj['totals']} ({obj['independent_units']} independent roots)")
    else:
        print(canonical({"experiment": ID, "phase": args.phase, "totals": obj["totals"],
                         "independent_roots": obj["independent_units"]}))


if __name__ == "__main__":
    main()
