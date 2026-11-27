"""Descriptive, root-weighted reductions of saved RP04 physical receipts.

This module reads receipt JSON only. Full per-case arrays stay in the input
files, whose hashes bind every reduction. An indexed factor or atom array
never acquires a case axis merely because its length matches the batch size.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from statistics import fmean

from .common import CHECKPOINTS, CONDITIONS, ROOTS, SMOKE_ROOTS, Recipe, sha


_INDEX_AXES = {
    "shape": "factor_dimension",
    "singular_values": "singular_value_order",
    "selected_group_ranks": "edited_slot",
    "selected_U_group_ranks": "edited_slot",
    "projector_chordal_by_slot": "edited_slot",
    "atom_energy_by_component_mean": "component_index",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _moments(values):
    """Count every input; undefined quantities retain explicit null counts."""
    _require(all(value is None or isinstance(value, (int, float)) for value in values),
             "Physical diagnostics must contain numeric values or null")
    present = [value for value in values if value is not None]
    _require(all(math.isfinite(value) for value in present),
             "Physical diagnostics contain a nonfinite value")
    result = {
        "count": len(values), "valid_count": len(present),
        "null_count": len(values) - len(present),
        "mean": fmean(present) if present else None,
        "min": min(present) if present else None,
        "max": max(present) if present else None,
    }
    if present and all(type(value) is bool for value in present):
        result["true_count"] = sum(present)
        result["false_count"] = len(present) - sum(present)
    return result


def _summarize(value, edited_slots, slots, components, key="", axis=None):
    if isinstance(value, dict):
        return {name: _summarize(item, edited_slots, slots, components, name)
                for name, item in sorted(value.items())}
    if value is None or isinstance(value, (int, float)):
        return {"axis": "scalar", **_moments([value])}
    _require(isinstance(value, list), "Unexpected physical diagnostic type: " + key)

    if key == "atom_energy_by_slot_and_component_mean":
        _require(len(value) == slots, "Atom-energy slot inventory differs")
        rows = []
        for row in value:
            if row is None:
                rows.append([None] * components)
            else:
                _require(isinstance(row, list) and len(row) == components,
                         "Atom-energy component inventory differs")
                rows.append(row)
        flat = [item for row in rows for item in row]
        return {
            "axis": "edited_slot_by_component_index", "shape": [slots, components],
            **_moments(flat),
            "by_edited_slot": {
                str(slot): _summarize(row, edited_slots, slots, components,
                                     axis="component_index")
                for slot, row in enumerate(rows)
            },
        }

    _require(all(item is None or isinstance(item, (int, float)) for item in value),
             "Unexpected nested physical diagnostic: " + key)
    selected_axis = axis or _INDEX_AXES.get(key)
    if selected_axis is None:
        selected_axis = "base_pair" if len(value) == len(edited_slots) else "index"
    result = {"axis": selected_axis, **_moments(value)}
    if selected_axis == "base_pair":
        result["by_edited_slot"] = {
            str(slot): {"axis": "base_pair", **_moments([
                item for item, edited in zip(value, edited_slots, strict=True) if edited == slot
            ])} for slot in range(slots)
        }
    else:
        result["by_index"] = {
            str(index): {"axis": "scalar", **_moments([item])}
            for index, item in enumerate(value)
        }
    return result


def _aggregate_roots(records):
    """Average root means, preserving observation counts and the raw range."""
    _require(bool(records), "No model-root diagnostics to aggregate")
    first = records[0]
    if "axis" not in first:
        keys = set(first)
        _require(all(set(record) == keys for record in records),
                 "Physical diagnostic keys differ across roots")
        return {key: _aggregate_roots([record[key] for record in records])
                for key in sorted(keys)}
    _require(all(record["axis"] == first["axis"] for record in records),
             "Physical diagnostic axes differ across roots")
    root_means = [record["mean"] for record in records]
    minima = [record["min"] for record in records if record["min"] is not None]
    maxima = [record["max"] for record in records if record["max"] is not None]
    result = {
        "axis": first["axis"],
        "root_weighted": _moments(root_means),
        "pooled_observation_counts": {
            name: sum(record[name] for record in records)
            for name in ("count", "valid_count", "null_count")
        },
        "observed_min": min(minima) if minima else None,
        "observed_max": max(maxima) if maxima else None,
    }
    if any("true_count" in record for record in records):
        result["pooled_observation_counts"].update({
            "true_count": sum(record.get("true_count", 0) for record in records),
            "false_count": sum(record.get("false_count", 0) for record in records),
        })
    if "shape" in first:
        _require(all(record.get("shape") == first["shape"] for record in records),
                 "Physical diagnostic shapes differ across roots")
        result["shape"] = first["shape"]
    for child in ("by_edited_slot", "by_index"):
        if child not in first:
            _require(all(child not in record for record in records),
                     "Physical diagnostic index inventory differs")
            continue
        if child == "by_index" and first["axis"] == "component_index":
            result["component_index_scope"] = (
                "Component identities are local to each trained root; elementwise "
                "atom means are retained in per_root."
            )
            continue
        keys = set(first[child])
        _require(all(child in record and set(record[child]) == keys for record in records),
                 "Physical diagnostic index inventory differs")
        result[child] = {
            key: _aggregate_roots([record[child][key] for record in records])
            for key in sorted(keys, key=int)
        }
    return result


def summarize_diagnostics(folder):
    """Return the reproducible descriptive summary for a completed root roster.

    Per-root means use defined observations. Every checkpoint-level mean then
    weights each root with a defined mean equally. Roots with no defined value
    remain in the root-level null counts. Component indices stay local to each
    root; semantic edited-slot summaries can be averaged across roots.
    """
    folder = Path(folder)
    hashes = {}

    def read(relative):
        path = folder / relative
        hashes[str(relative)] = sha(path)
        return json.loads(path.read_text())

    manifest = read("run_manifest.json")
    phase = manifest["phase"]
    _require(phase in ("smoke", "development"), "Unknown diagnostic phase")
    expected_roots = SMOKE_ROOTS if phase == "smoke" else ROOTS
    _require(manifest["roots"] == list(expected_roots), "Incomplete frozen root roster")
    recipe = Recipe(**manifest["recipe"])
    expected_pairs = recipe.smoke_base_pairs if phase == "smoke" else recipe.base_pairs
    per_root = []
    for root in expected_roots:
        directory = Path(f"root_{root}")
        cases = read(directory / "cases.json")
        _require(cases["root"] == root and cases["phase"] == phase,
                 "Diagnostic case labels differ")
        _require(cases["n_base_pairs"] == expected_pairs, "Diagnostic pair roster differs")
        edited_slots = cases["edit_slot_by_pair"]
        _require(len(edited_slots) == expected_pairs
                 and all(type(slot) is int and 0 <= slot < recipe.slots for slot in edited_slots),
                 "Diagnostic edited-slot inventory differs")
        slot_counts = Counter(edited_slots)
        row = {
            "root": root, "n_base_pairs": expected_pairs,
            "edit_slot_counts": {str(slot): slot_counts[slot] for slot in range(recipe.slots)},
            "batch_sha256": cases["batch_sha256"],
            "checkpoint_sha256": {}, "checkpoints": {},
        }
        for checkpoint in CHECKPOINTS:
            snapshot = read(directory / (checkpoint + ".json"))
            _require(snapshot["root"] == root and snapshot["checkpoint"] == checkpoint
                     and snapshot["phase"] == phase, "Diagnostic checkpoint labels differ")
            _require(snapshot["batch_sha256"] == cases["batch_sha256"],
                     "Diagnostic paired input hash differs")
            _require(set(snapshot["conditions"]) == set(CONDITIONS),
                     "Diagnostic intervention inventory differs")
            diagnostic = snapshot["direction_diagnostics"]
            components = diagnostic["factor"]["shape"][1]
            sections = {
                "factor": diagnostic["factor"],
                "geometry": diagnostic["geometry"],
                "directions": diagnostic["directions"],
                "rotated_directions": snapshot["rotated_diagnostics"],
                "pathways": snapshot["pathway_diagnostics"],
                "condition_physical": {
                    name: snapshot["conditions"][name]["physical"] for name in CONDITIONS
                },
                "checks": snapshot["checks"],
            }
            row["checkpoint_sha256"][checkpoint] = snapshot["checkpoint_sha256"]
            row["checkpoints"][checkpoint] = _summarize(
                sections, edited_slots, recipe.slots, components)
        per_root.append(row)

    return {
        "schema_version": 1, "experiment": "BPD-RP-04", "phase": phase,
        "evidence_class": "DEVELOPMENT_ONLY", "analysis_role": "descriptive",
        "unit": "Previously exposed model root; snapshots and cases are repeated observations",
        "roots": list(expected_roots), "n_units": len(expected_roots),
        "checkpoints": list(CHECKPOINTS), "input_hashes": hashes,
        "root_weighting": (
            "Within a root, means use defined observations. Checkpoint summaries "
            "give equal weight to each defined root mean. Null counts retain "
            "undefined observations and roots; pooled counts do not set weights."
        ),
        "array_axes": (
            "base_pair vectors receive edited-slot summaries. Factor spectra, "
            "slot ranks and atom means retain their index axes. Component indices "
            "are local to each root, so their elementwise means stay in per_root."
        ),
        "gauge_scope": {
            "grouped_physical_writes": (
                "Invariant under invertible factor basis changes preserving the "
                "selected/outside partition of the measured case."
            ),
            "atom_energy": (
                "Invariant under reciprocal V-column/U-row scaling and sign changes; "
                "within-block mixing can redistribute atom energy."
            ),
            "coordinate_reads": "Read-coordinate norms depend on the factor gauge.",
            "solver": (
                "Condition numbers and numerical rank use the supplied factor "
                "representation and declared singular-value cutoff."
            ),
        },
        "measurement_scope": {
            "directions": "Cast state edits at the declared physical state dose",
            "pathways": "Parent state edit determines each route's inherited write dose",
            "pathway_final_all_head_delta_norm": (
                "Norm across value logits for each base receiver's original query; "
                "expanded all-address outcomes are recorded separately."
            ),
            "atom_means": "Means of component squared write norms in the saved factor gauge",
            "counterfactual_state": "Hybrid-derived directions use privileged counterfactual states",
            "calculation": "Reductions of hashed saved receipts; raw arrays remain in those receipts",
        },
        "per_checkpoint": {
            checkpoint: _aggregate_roots([row["checkpoints"][checkpoint] for row in per_root])
            for checkpoint in CHECKPOINTS
        },
        "per_root": per_root,
    }
