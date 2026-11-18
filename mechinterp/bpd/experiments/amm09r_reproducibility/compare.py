"""Compare AMM-09R engineering receipts without importing torch or fitting models.

The registered gate requires complete hosted repeats and exact pinned32 payloads.
Baseline and historical numerical differences remain explicit report entries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any


SCHEMA = "bpd.amm09r.engineering.v1"
REGISTRATION_COMMIT = "ffc7143e3b597d840c5000a94a45b45d964931cf"
WORKERS = ("ubuntu-22.04", "ubuntu-24.04")
ORDINALS = (1, 2)
GROUPS = (("engineering", "baseline32"), ("engineering", "pinned32"),
          ("historical", "baseline32"))
VERSIONS = {"python": "3.12.12", "torch": "2.10.0+cpu", "numpy": "2.2.6"}
PINNED_ENV = {
    "ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "COMPATIBLE",
    "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE", "OMP_DYNAMIC": "FALSE", "PYTHONHASHSEED": "0",
}
HEX = re.compile(r"[0-9a-f]{64}\Z")
MASKS = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]
PHYSICAL_GATES = {
    "six_frozen_nonempty_two_slot_masks", "finite", "gauge", "equal_dose",
    "current_readout_unchanged", "projection_and_normalization",
    "native_float32_export_parity", "native_reference_parity_at_registered_tolerance",
    "native_pass_counts", "native_pass_reset", "native_context_cleanup",
    "full_donor_equal_dose",
}
STAGES = ("rows", "root", "initial", "train_input_sha256", "teacher",
          "factor_before_svd", "svd", "factor", "fit", "assays", "physical", "fixture")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _pointer(path) -> str:
    return "/" + "/".join(str(x).replace("~", "~0").replace("/", "~1") for x in path)


def _brief(value):
    if isinstance(value, str) and len(value) > 120:
        return {"length": len(value), "prefix": value[:80], "sha256": hashlib.sha256(value.encode()).hexdigest()}
    if isinstance(value, (dict, list)):
        return {"type": type(value).__name__, "length": len(value)}
    return value


def _flatten(values, shape):
    if not shape:
        if type(values) not in (int, float) or not math.isfinite(values):
            raise ValueError("tensor scalar must be finite and numeric")
        yield values
        return
    if not isinstance(values, list) or len(values) != shape[0]:
        raise ValueError("tensor values disagree with shape")
    for value in values:
        yield from _flatten(value, shape[1:])


def tensor_errors(value, path=()):
    """Verify tagged raw little-endian tensors independently of their JSON hash."""
    if isinstance(value, dict):
        if {"dtype", "shape", "values", "sha256"}.issubset(value):
            try:
                if value["dtype"] not in ("float32", "float64"):
                    raise ValueError("unsupported tensor dtype")
                shape = value["shape"]
                if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError("invalid tensor shape")
                scalars = list(_flatten(value["values"], shape))
                code = "f" if value["dtype"] == "float32" else "d"
                raw = struct.pack("<" + str(len(scalars)) + code, *scalars)
                if hashlib.sha256(raw).hexdigest() != value["sha256"]:
                    raise ValueError("tensor byte hash mismatch")
            except (ValueError, TypeError, OverflowError, struct.error) as error:
                yield f"{_pointer(path)}: {error}"
            return
        for key, item in value.items():
            yield from tensor_errors(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from tensor_errors(item, (*path, index))


def _hash_map(value):
    return isinstance(value, dict) and bool(value) and all(
        isinstance(key, str) and isinstance(item, str) and HEX.fullmatch(item)
        for key, item in value.items())


def _provenance(receipt):
    value = receipt.get("provenance")
    return value if isinstance(value, dict) else {}


def validate_receipt(receipt):
    errors = []
    if not isinstance(receipt, dict):
        return ["receipt must be an object"]
    if receipt.get("schema") != SCHEMA:
        errors.append("wrong receipt schema")
    if receipt.get("execution_status") != "completed":
        errors.append("execution did not complete")
    if receipt.get("confirmation_accessed") is not False:
        errors.append("missing protected-population exclusion receipt")
    if (receipt.get("mode"), receipt.get("configuration")) not in GROUPS:
        errors.append("unregistered mode/configuration")
    provenance = receipt.get("provenance", {})
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
        provenance = {}
    if provenance.get("worker") not in WORKERS:
        errors.append("unregistered worker label")
    if type(provenance.get("process_ordinal")) is not int or provenance.get("process_ordinal") not in ORDINALS:
        errors.append("unregistered process ordinal")
    if not isinstance(provenance.get("host_id"), str) or not provenance.get("host_id"):
        errors.append("missing host identity")
    if not _hash_map(receipt.get("source_sha256")):
        errors.append("missing or invalid source hashes")
    payload = receipt.get("payload")
    try:
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        if digest(payload) != receipt.get("payload_sha256"):
            errors.append("payload digest mismatch")
        errors.extend(tensor_errors(payload))
    except (ValueError, TypeError, OverflowError) as error:
        errors.append(str(error))
        return errors
    rows = payload.get("rows")
    mode = receipt.get("mode")
    count, phase = (8, "engineering") if mode == "engineering" else (64, "training")
    expected = [f"{phase}/{index:04d}" for index in range(count)]
    if not isinstance(rows, list) or [row.get("root") if isinstance(row, dict) else None for row in rows] != expected:
        errors.append("root count, order, or membership differs from registration")
        return errors
    if mode == "engineering":
        if (payload.get("phase") != "engineering" or payload.get("n_roots") != 8
                or payload.get("namespace") != "AMM09R-ENGINEERING-V1/"):
            errors.append("incorrect excluded engineering namespace or phase")
        if not isinstance(payload.get("fixture"), dict) or not payload["fixture"]:
            errors.append("missing engineering ridge/policy fixture")
        for row in rows:
            root = row["root"]
            for stage in ("initial", "teacher", "factor_before_svd", "factor", "fit"):
                if not isinstance(row.get(stage), dict) or not row[stage]:
                    errors.append(f"{root}: missing {stage}")
            if not _hash_map(row.get("train_input_sha256")) or set(row["train_input_sha256"]) != {"teacher_train", "factor_train"}:
                errors.append(f"{root}: incomplete training-input hashes")
            assays = row.get("assays", {})
            if not isinstance(assays, dict):
                assays = {}
            for split, n in (("training_pairs", 128), ("engineering", 512)):
                assay = assays.get(split, {})
                if not isinstance(assay, dict):
                    assay = {}
                bits = assay.get("candidate_bits", {})
                for action in range(6):
                    bitstring = bits.get(f"mask_{action}") if isinstance(bits, dict) else None
                    if not isinstance(bitstring, str) or len(bitstring) != n or set(bitstring) - {"0", "1"}:
                        errors.append(f"{root}/{split}: invalid mask_{action} candidate bits")
                for field, shape in (("features", [n, 6, 18]), ("probes", [n, 6, 2]),
                                     ("patches", [n, 6, 6]), ("margins", [n, 6])):
                    packed = assay.get(field, {})
                    if (not isinstance(packed, dict) or packed.get("shape") != shape
                            or not {"dtype", "shape", "sha256", "values"}.issubset(packed)):
                        errors.append(f"{root}/{split}: missing or incorrect {field} shape")
            physical = row.get("physical", {})
            gates = physical.get("gates", {}) if isinstance(physical, dict) else {}
            if (not isinstance(physical, dict) or physical.get("passed") is not True
                    or not isinstance(gates, dict) or not PHYSICAL_GATES.issubset(gates)
                    or any(value is not True for value in gates.values())):
                errors.append(f"{root}: physical admission failed or incomplete")
            if isinstance(physical, dict) and physical.get("masks") != MASKS:
                errors.append(f"{root}: wrong physical mask set")
    return errors


def _margin(payload, path, case):
    try:
        if len(path) != 6 or path[0] != "rows" or path[2] != "assays" or path[4] != "candidate_bits":
            return None
        action = int(path[5].removeprefix("mask_"))
        return payload["rows"][path[1]]["assays"][path[3]]["margins"]["values"][case][action]
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def differences(left, right, limit=48):
    report = {"exact": digest(left) == digest(right), "different_values": 0,
              "numeric_differences": 0, "max_absolute_numeric_difference": 0.,
              "candidate_bit_flips": 0, "first_differing_recorded_stage": None,
              "examples": [], "candidate_bit_examples": []}

    def record(path, a, b):
        report["different_values"] += 1
        if report["first_differing_recorded_stage"] is None:
            report["first_differing_recorded_stage"] = _pointer(path[:3] if path[:1] == ("rows",) else path[:1])
        if len(report["examples"]) < limit:
            report["examples"].append({"path": _pointer(path), "left": _brief(a), "right": _brief(b)})
        if type(a) in (int, float) and type(b) in (int, float):
            report["numeric_differences"] += 1
            report["max_absolute_numeric_difference"] = max(report["max_absolute_numeric_difference"], abs(a-b))
        if "candidate_bits" in path and isinstance(a, str) and isinstance(b, str) and len(a) == len(b):
            for case, (aa, bb) in enumerate(zip(a, b)):
                if aa != bb:
                    report["candidate_bit_flips"] += 1
                    if len(report["candidate_bit_examples"]) < limit:
                        report["candidate_bit_examples"].append({
                            "path": _pointer(path), "case": case, "left_bit": aa, "right_bit": bb,
                            "left_margin": _margin(left, path, case), "right_margin": _margin(right, path, case)})

    def visit(a, b, path=()):
        if type(a) is not type(b):
            record(path, a, b)
        elif isinstance(a, dict):
            keys = sorted(a.keys() | b.keys(), key=lambda key: (STAGES.index(key) if key in STAGES else len(STAGES), key))
            for key in keys:
                if key not in a or key not in b:
                    record((*path, key), a.get(key, "<missing>"), b.get(key, "<missing>"))
                else:
                    visit(a[key], b[key], (*path, key))
        elif isinstance(a, list):
            if len(a) != len(b):
                record((*path, "length"), len(a), len(b))
            for index, (aa, bb) in enumerate(zip(a, b)):
                visit(aa, bb, (*path, index))
        elif isinstance(a, float):
            if struct.pack("<d", a) != struct.pack("<d", b):
                record(path, a, b)
        elif a != b:
            record(path, a, b)

    if not report["exact"]:
        visit(left, right)
    report["examples_truncated"] = report["different_values"] > len(report["examples"])
    report["candidate_bit_examples_truncated"] = report["candidate_bit_flips"] > len(report["candidate_bit_examples"])
    return report


def compare_receipts(named_receipts):
    """Return a report; malformed, missing, or duplicate receipts fail closed."""
    groups = {group: [] for group in GROUPS}
    errors = []
    for name, receipt in named_receipts:
        issues = validate_receipt(receipt)
        if issues:
            errors.append({"file": name, "errors": issues})
        if isinstance(receipt, dict):
            group = (receipt.get("mode"), receipt.get("configuration"))
            if all(isinstance(value, str) for value in group) and group in groups:
                groups[group].append((name, receipt))
    reports = {}
    complete = True
    for group, entries in groups.items():
        name = "/".join(group)
        slots = [(_provenance(r).get("worker"), _provenance(r).get("process_ordinal")) for _, r in entries]
        expected_slots = [(worker, ordinal) for worker in WORKERS for ordinal in ORDINALS]
        coverage = (len(slots) == 4
                    and all(worker in WORKERS and type(ordinal) is int and ordinal in ORDINALS for worker, ordinal in slots)
                    and set(slots) == set(expected_slots))
        complete = complete and coverage
        source_equal = bool(entries) and all(r.get("source_sha256") == entries[0][1].get("source_sha256") for _, r in entries)
        payloads = [(file, r) for file, r in entries if isinstance(r.get("payload"), dict)]
        comparisons = []
        for file, receipt in payloads[1:]:
            try:
                diff = differences(payloads[0][1]["payload"], receipt["payload"])
                comparisons.append({"left_file": payloads[0][0], "right_file": file, **diff})
            except (ValueError, TypeError, OverflowError) as error:
                comparisons.append({"left_file": payloads[0][0], "right_file": file, "exact": False, "error": str(error)})
        report = {
            "complete": coverage, "receipt_count": len(entries), "observed_slots": [list(slot) for slot in slots],
            "matching_source_hashes": source_equal,
            "all_payloads_exact": coverage and len(payloads) == 4 and all(c["exact"] for c in comparisons),
            "comparisons": comparisons,
            "historical_comparisons_to_original": [
                {"file": file, "comparison": r["comparison_to_original"]}
                for file, r in entries if "comparison_to_original" in r],
        }
        reports[name] = report

    pinned = groups[("engineering", "pinned32")]
    provenance = [_provenance(r) for _, r in pinned]
    software = len(provenance) == 4 and all(
        p.get("registered_software") is True and all(p.get(key) == expected for key, expected in VERSIONS.items())
        for p in provenance)
    runtime = len(provenance) == 4 and all(
        isinstance(p.get("environment"), dict)
        and all(p["environment"].get(key) == expected for key, expected in PINNED_ENV.items())
        and p.get("deterministic_algorithms") is True
        and all(type(p.get(key)) is int and p[key] == 1 for key in ("num_threads", "num_interop_threads"))
        and p.get("cpu_capability") in ("NO AVX", "DEFAULT")
        for p in provenance)
    binaries = len(provenance) == 4 and all(
        _hash_map(p.get("torch_binary_sha256"))
        and {"libtorch_cpu.so", "_C.so"}.issubset(p["torch_binary_sha256"])
        for p in provenance)
    binaries = bool(binaries and all(p["torch_binary_sha256"] == provenance[0]["torch_binary_sha256"] for p in provenance))
    host_sets = {worker: {p.get("host_id") for p in provenance
                         if p.get("worker") == worker and isinstance(p.get("host_id"), str)} for worker in WORKERS}
    distinct_hosts = all(len(values) == 1 and None not in values and "" not in values for values in host_sets.values())
    distinct_hosts = bool(distinct_hosts and host_sets[WORKERS[0]] != host_sets[WORKERS[1]])
    cpu_models = sorted({p.get("cpu_model") for p in provenance if isinstance(p.get("cpu_model"), str) and p["cpu_model"]})
    pinned_report = reports["engineering/pinned32"]
    sources = [r.get("source_sha256") for entries in groups.values() for _, r in entries]
    all_sources_equal = len(sources) == 12 and all(source == sources[0] for source in sources)
    gates = {
        "all_registered_repeats_present": complete,
        "all_receipts_valid": not errors,
        "pinned_payloads_bit_exact": pinned_report["all_payloads_exact"],
        "pinned_sources_equal": pinned_report["matching_source_hashes"],
        "all_registered_sources_equal": all_sources_equal,
        "registered_software": software,
        "pinned_runtime_settings": runtime,
        "torch_binaries_equal": binaries,
        "two_distinct_hosts": distinct_hosts,
    }
    return {
        "schema": "bpd.amm09r.comparison.v1", "registration_commit": REGISTRATION_COMMIT,
        "registered_gate_passed": all(gates.values()), "gates": gates,
        "confirmation_authorized": False, "scientific_evidence": False,
        "cpu_models": cpu_models, "distinct_cpu_models_observed": len(cpu_models) > 1,
        "hardware_scope": "different recorded CPU models" if len(cpu_models) > 1 else "recorded CPU models do not establish hardware diversity",
        "receipt_errors": errors, "groups": reports,
    }


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON object key: {key}")
        out[key] = value
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="write a descriptive local report with a failing gate and return zero")
    args = parser.parse_args(argv)
    receipts, load_errors = [], []
    for path in sorted(args.directory.rglob("receipt*.json")):
        try:
            def invalid_constant(value):
                raise ValueError(f"nonfinite JSON constant: {value}")
            value = json.loads(path.read_text(), object_pairs_hook=_unique_object, parse_constant=invalid_constant)
            receipts.append((str(path.relative_to(args.directory)), value))
        except (OSError, ValueError) as error:
            load_errors.append({"file": str(path), "errors": [str(error)]})
    report = compare_receipts(receipts)
    if load_errors:
        report["receipt_errors"].extend(load_errors)
        report["gates"]["all_receipts_valid"] = False
        report["registered_gate_passed"] = False
    report["descriptive_only"] = args.allow_incomplete
    if args.allow_incomplete:
        report["registered_gate_passed"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"registered_gate_passed": report["registered_gate_passed"], "gates": report["gates"],
                      "distinct_cpu_models_observed": report["distinct_cpu_models_observed"]}, sort_keys=True))
    return 0 if report["registered_gate_passed"] or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
