"""Exact public-development triplets for the BPD-COL-02 local gym.

The arithmetic prompt includes the seed. This is a bounded-computation task,
not an information-theoretic hidden-seed test. The optional trace tier supplies
the exact trajectory and measures readout/routing competence separately.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from math import gcd
from pathlib import Path
import sys

# Import the standalone finite oracle without importing BPD's neural package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collatz_observability.collatz_core import analyze, modular_labels, odd_step


SCHEMA = "BPD-COL-02/public-development-cases/v1"
TIERS = ("arithmetic", "trace")
ROLES = ("recipient", "effect_donor", "sham_donor")
SOURCE_COMMIT = "b3eb33ed10485264fe7a593e3b0ace418d9f9a6d"
SOURCE_SHA256 = "24ff52171aedf890c33439896ab5e71de65d9b05d557ba61d162f3623aa5dbca"


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _integer(value: object, name: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
    return value


def trajectory(d: int, n: int, horizon: int) -> list[int]:
    _integer(d, "d", 1, 1025)
    _integer(n, "n", 1, d)
    _integer(horizon, "horizon", 1, 2)
    if d % 4 != 1 or gcd(d, 6) != 1 or n % 2 != 1:
        raise ValueError("require d=1 mod4, gcd(d,6)=1, and odd n")
    result = [n]
    for _ in range(horizon):
        n = odd_step(n, d)[0]
        result.append(n)
    return result


def branch(n: int) -> int:
    return 3 if n % 4 == 1 else 1


def render_prompt(d: int, n: int, horizon: int, label_map: dict[str, str],
                  tier: str = "arithmetic") -> str:
    """Public task text; gold labels and matching metadata stay outside it."""
    trace = trajectory(d, n, horizon)
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    if (not isinstance(label_map, dict) or set(label_map) != {"1", "3"}
            or set(label_map.values()) != {"A", "B"}):
        raise ValueError("label_map must biject branch strings 1,3 onto A,B")
    if tier == "arithmetic":
        prefix = (
            "For odd x, set t=3*x+d when x mod 4=1, otherwise t=x+d. "
            "Divide t by 2 until odd; this is one step. "
            f"Use d={d:06d}, start={n:06d}, steps={horizon}. "
        )
    else:
        prefix = (
            "Read the recorded odd-state trace. "
            f"d={d:06d}; steps={horizon}; trace="
            + " -> ".join(f"{state:06d}" for state in trace) + ". "
        )
    return (prefix + f"For the final x, answer {label_map['3']} if x mod 4=1, "
            f"or {label_map['1']} if x mod 4=3. Reply with one letter. Answer:")


def _future(core, n: int, horizon: int) -> int:
    for _ in range(horizon):
        n = core.successors[n]
    return n


def _select_triplet(core, labels: dict[int, int], horizon: int,
                    same_basin: bool = True) -> tuple[int, int, int] | None:
    groups = defaultdict(list)
    for n in core.states:
        key = (labels[n % core.d], branch(n))
        if same_basin:
            key += (core.basin[n],)
        groups[key].append(n)
    for key in sorted(groups):
        states = groups[key]
        futures = {n: branch(_future(core, n, horizon)) for n in states}
        for recipient in states:
            effects = [n for n in states if futures[n] != futures[recipient]]
            shams = [n for n in states if n != recipient and futures[n] == futures[recipient]]
            if effects and shams:
                return recipient, effects[0], shams[0]
    return None


def make_case(d: int, horizon: int, triplet: tuple[int, int, int], *,
              index: int, tier: str = "arithmetic", role: str = "development") -> dict:
    core = analyze(d)
    labels = modular_labels(d)
    if len(set(triplet)) != 3:
        raise ValueError("recipient, effect donor and sham donor must differ")
    for n in triplet:
        trajectory(d, n, horizon)
    observations = {(labels[n % d], branch(n)) for n in triplet}
    if len(observations) != 1:
        raise ValueError("triplet does not share current arithmetic observer")
    futures = [branch(_future(core, n, horizon)) for n in triplet]
    if futures[0] == futures[1] or futures[0] != futures[2]:
        raise ValueError("effect and sham future-label contract failed")
    # Balance recipient labels by alternating their letter assignment. This
    # mapping is printed in every prompt and frozen before any model is used.
    recipient_label = "A" if index % 2 == 0 else "B"
    other_label = "B" if recipient_label == "A" else "A"
    label_map = {str(futures[0]): recipient_label, str(futures[1]): other_label}
    case = {
        "case_id": f"COL3050-d{d:06d}-h{horizon}-{tier}",
        "root_id": d, "d": d, "horizon": horizon, "tier": tier,
        "selection_role": role, "label_map": label_map,
        "current_observer": {"quotient": labels[triplet[0] % d],
                             "branch_multiplier": branch(triplet[0])},
        "all_same_basin": len({core.basin[n] for n in triplet}) == 1,
    }
    for name, n, future in zip(ROLES, triplet, futures):
        case[name] = {
            "n": n, "trace": trajectory(d, n, horizon),
            "basin": core.basin[n], "future_branch": future,
            "label": label_map[str(future)],
            "prompt": render_prompt(d, n, horizon, label_map, tier),
        }
    return case


def generate_manifest(max_roots: int = 4, horizons: tuple[int, ...] = (1,),
                      tier: str = "arithmetic", max_parameter: int = 1025,
                      cases_per_root: int = 1) -> dict:
    """Freeze deterministic public triplets, before seeing model behavior.

    d13 is a cross-basin calibration fixture. Other roots require every donor
    to stay in the recipient basin. d73,h1 uses the already proved infinite-
    trace sham (recipient1, effect9, sham37). Model runs remain development.
    """
    _integer(max_roots, "max_roots", 1, 32)
    _integer(max_parameter, "max_parameter", 1, 1025)
    _integer(cases_per_root, "cases_per_root", 1, 2)
    if not isinstance(horizons, (tuple, list)) or not horizons:
        raise ValueError("horizons must contain distinct integers")
    for h in horizons:
        _integer(h, "horizon", 1, 2)
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must contain distinct integers")
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    oracle_path = Path(__file__).resolve().parents[1] / "collatz_observability" / "collatz_core.py"
    if hashlib.sha256(oracle_path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("canonical arithmetic oracle source hash changed")
    parameters = [d for d in range(1, max_parameter + 1, 4) if gcd(d, 6) == 1]
    priority = [d for d in (13, 73) if d in parameters]
    ordered = priority + [d for d in parameters if d not in priority]
    cases, screening, selected_roots = [], [], []
    for d in ordered:
        if len(selected_roots) >= max_roots:
            break
        core, labels = analyze(d), modular_labels(d)
        root_cases = []
        for horizon in horizons:
            if len(root_cases) >= cases_per_root:
                break
            if d == 13:
                triplet = (1, 5, 9) if horizon == 1 else (1, 9, 5)
                role = "cross_basin_calibration"
            elif d == 73 and horizon == 1:
                triplet, role = (1, 9, 37), "same_basin_exact_trace_sham"
            else:
                triplet = _select_triplet(core, labels, horizon, same_basin=True)
                role = "same_basin_development"
            if triplet is None:
                screening.append({"d": d, "horizon": horizon,
                                  "eligible": False, "reason": "no_same_basin_triplet"})
                continue
            root_cases.append(make_case(d, horizon, triplet,
                                        index=len(cases) + len(root_cases), tier=tier, role=role))
            screening.append({"d": d, "horizon": horizon, "eligible": True, "role": role})
        if root_cases:
            selected_roots.append(d)
            cases.extend(root_cases)
    if len(selected_roots) != max_roots:
        raise ValueError(f"only {len(selected_roots)} eligible roots below max_parameter")
    manifest = {
        "schema": SCHEMA,
        "specification": {
            "requested_roots": max_roots, "max_parameter": max_parameter,
            "horizons": list(horizons), "cases_per_root": cases_per_root, "tier": tier,
            "unit": "parameter d; repeated cases and intervention arms share that root",
            "source_population": "public previously exposed d<=1025 development",
            "information_contract": "seed is supplied; bounded computation or explicit trace readout",
            "split": "development_only", "confirmation_authorized": False,
            "selection": "deterministic exact oracle, before any model outcomes",
            "label_assignment": "alternate recipient A/B; displayed mapping in prompt",
        },
        "source": {"repository": "patrickdugan/MathCan", "commit": SOURCE_COMMIT,
                   "oracle_sha256": SOURCE_SHA256},
        "n_roots": len(selected_roots), "n_cases": len(cases),
        "root_ids": selected_roots, "screening": screening, "cases": cases,
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    return manifest


def validate_manifest(manifest: dict) -> None:
    """Verify the frozen generator contract and all oracle/prompt fields."""
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unknown case manifest schema")
    spec = manifest.get("specification", {})
    expected = generate_manifest(
        max_roots=spec.get("requested_roots"), horizons=spec.get("horizons"),
        tier=spec.get("tier"), max_parameter=spec.get("max_parameter"),
        cases_per_root=spec.get("cases_per_root"))
    if manifest != expected:
        raise ValueError("manifest differs from deterministic oracle generation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, default=4)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1])
    parser.add_argument("--cases-per-root", type=int, default=1)
    parser.add_argument("--tier", choices=TIERS, default="arithmetic")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = generate_manifest(args.roots, tuple(args.horizons), args.tier,
                                 cases_per_root=args.cases_per_root)
    text = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}: {manifest['n_roots']} roots, {manifest['n_cases']} cases")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
