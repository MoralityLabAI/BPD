"""Independent, read-only SUB-08 receipt audit; this module never trains models.

The task is reconstructed from literal state-transition permutations.  No
oracle, training, runner, or saved aggregate-metric implementation is imported.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


SCHEMA = 1
DESIGN_SEEDS = (1890, 1891, 1892)
PROSPECTIVE_SEEDS = tuple(range(1900, 1908))
CONFIGS = ((48, 2400), (48, 3600), (64, 2400), (64, 3600))
TOLERANCE = 1e-6
TRANSITIONS = ((2, 3, 0, 1), (0, 2, 1, 3), (0, 1, 3, 2), (0, 3, 2, 1))
SOURCE_PREFIX = "mechinterp/bpd/experiments/sub08/"
REQUIRED_SOURCES = tuple(SOURCE_PREFIX + name for name in (
    "DESIGN_PROTOCOL.md", "oracle.py", "source.py", "run.py", "audit.py",
))
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


class AuditError(ValueError):
    """An original receipt does not satisfy the registered integrity contract."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _integer(value: Any, name: str) -> int:
    _need(type(value) is int, f"{name} must be an integer")
    return value


def _finite(value: Any, name: str) -> float:
    _need(isinstance(value, (int, float)) and not isinstance(value, bool),
          f"{name} must be numeric")
    result = float(value)
    _need(math.isfinite(result), f"{name} must be finite")
    return result


def _safe_file(directory: Path, relative: Any) -> Path:
    _need(isinstance(relative, str) and bool(relative), "file path is missing")
    rel = Path(relative)
    _need(not rel.is_absolute() and ".." not in rel.parts, f"unsafe path: {relative}")
    base = directory.resolve()
    path = (base / rel).resolve()
    _need(path != base and path.is_relative_to(base), f"path escapes directory: {relative}")
    _need(path.is_file(), f"missing file: {relative}")
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_sha256(ids: Any) -> str:
    """Digest the ordered IDs as explicitly little-endian signed 64-bit data."""
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def exact_universe() -> tuple[np.ndarray, np.ndarray]:
    ids = np.arange(4096, dtype=np.int64)
    sequences = (ids[:, None] // (4 ** np.arange(5, -1, -1))) % 4
    labels = np.ones(4096, dtype=np.int64)
    table = np.asarray(TRANSITIONS, dtype=np.int64)
    for position in range(6):
        labels = table[sequences[:, position], labels]
    return sequences, labels


def exact_partitions() -> dict[str, np.ndarray]:
    ids = np.random.default_rng(246810).permutation(4096)
    return {
        "train": ids[:2048], "validation": ids[2048:2560],
        "reserved_dose": ids[2560:3072], "heldout": ids[3072:],
    }


def feature_ceiling(ids: Any, feature: str) -> int:
    sequences, labels = exact_universe()
    groups: dict[tuple[int, ...], list[int]] = {}
    for index in np.asarray(ids, dtype=np.int64).tolist():
        sequence = sequences[index]
        if feature.startswith("prefix"):
            size = int(feature[len("prefix"):])
            _need(0 <= size <= 6, "invalid prefix length")
            key = tuple(int(x) for x in sequence[:size])
        elif feature == "counts":
            key = tuple(int(x) for x in np.bincount(sequence, minlength=4))
        elif feature == "last":
            key = (int(sequence[-1]),)
        else:
            raise AuditError(f"unknown restricted feature: {feature}")
        counts = groups.setdefault(key, [0, 0, 0, 0])
        counts[int(labels[index])] += 1
    return sum(max(counts) for counts in groups.values())


def exact_certificate() -> dict[str, Any]:
    _, labels = exact_universe()
    features = ("prefix2", "prefix4", "last", "counts")
    population = {key: feature_ceiling(np.arange(4096), key) for key in features}
    heldout = {key: feature_ceiling(exact_partitions()["heldout"], key) for key in features}
    _need(np.bincount(labels, minlength=4).tolist() == [781, 1146, 1024, 1145],
          "literal oracle population counts disagree with registration")
    _need(population == dict(zip(features, (1232, 1556, 1188, 1603))),
          "population information certificate disagrees with registration")
    _need(heldout == dict(zip(features, (347, 531, 304, 449))),
          "heldout information certificate disagrees with registration")
    return {
        "class_counts": np.bincount(labels, minlength=4).tolist(),
        "population_n": 4096, "population_correct_ceilings": population,
        "heldout_n": 1024, "heldout_correct_ceilings": heldout,
        "partition_hashes": {name: ids_sha256(ids) for name, ids in exact_partitions().items()},
    }


def _arrays(ids: Any, labels: Any, logits: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _need(isinstance(ids, (list, np.ndarray)), "sequence IDs must be an array")
    _need(isinstance(labels, (list, np.ndarray)), "labels must be an array")
    _need(all(type(x) is int or isinstance(x, np.integer) for x in ids), "sequence IDs must be integers")
    _need(all(type(x) is int or isinstance(x, np.integer) for x in labels), "labels must be integers")
    indices = np.asarray(ids, dtype=np.int64)
    targets = np.asarray(labels, dtype=np.int64)
    values = np.asarray(logits, dtype=np.float64)
    _need(indices.ndim == 1 and len(indices) > 0, "invalid sequence ID shape")
    _need(len(set(indices.tolist())) == len(indices), "duplicate sequence IDs")
    _need(bool(((indices >= 0) & (indices < 4096)).all()), "sequence ID outside universe")
    _need(targets.shape == indices.shape, "target shape mismatch")
    _need(values.shape == (len(indices), 4), "logits must have four classes per sequence")
    _need(bool(np.isfinite(values).all()), "nonfinite logits")
    _, oracle = exact_universe()
    _need(np.array_equal(targets, oracle[indices]), "saved target disagrees with literal oracle")
    return indices, targets, values


def metrics_from_logits(ids: Any, labels: Any, logits: Any) -> dict[str, Any]:
    indices, targets, values = _arrays(ids, labels, logits)
    predicted = values.argmax(axis=1)
    correct = predicted == targets
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    supports = confusion.sum(axis=1)
    class_correct = np.diag(confusion)
    recalls = [float(class_correct[c] / supports[c]) if supports[c] else None for c in range(4)]
    shifted = values - values.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(values)), targets]
    per_last = []
    for token in range(4):
        selected = indices % 4 == token
        per_last.append({
            "token": token, "n": int(selected.sum()),
            "correct": int(correct[selected].sum()),
            "accuracy": float(correct[selected].mean()) if selected.any() else None,
            "class_recall": [
                float(correct[selected & (targets == c)].mean())
                if (selected & (targets == c)).any() else None for c in range(4)
            ],
        })
    return {
        "n": len(indices), "correct": int(correct.sum()), "accuracy": float(correct.mean()),
        "nll": float(losses.mean()), "confusion": confusion.tolist(),
        "class_support": supports.tolist(), "class_correct": class_correct.tolist(),
        "class_recall": recalls, "per_last_token": per_last,
        "ids_sha256": ids_sha256(indices),
    }


def _prefix_consistency(ids: np.ndarray, values: np.ndarray, length: int) -> float:
    keys = ids // (4 ** (6 - length))
    first: dict[int, np.ndarray] = {}
    maximum = 0.0
    for key, logits in zip(keys.tolist(), values):
        reference = first.setdefault(key, logits)
        maximum = max(maximum, float(np.abs(logits - reference).max()))
    return maximum


def accuracy_recall_gates(measured: dict[str, Any], role: str) -> dict[str, bool]:
    """Registered thresholds applied directly to integer outcome counts."""
    _need(role in ("design", "prospective"), "invalid performance-gate role")
    if role == "design":
        return {
            "validation_accuracy": 20 * measured["correct"] >= 19 * measured["n"],
            "validation_class_recall": all(
                support > 0 and 10 * correct >= 9 * support
                for correct, support in zip(measured["class_correct"], measured["class_support"])
            ),
        }
    return {
        "heldout_accuracy": 10 * measured["correct"] >= 9 * measured["n"],
        "heldout_class_recall": all(
            support > 0 and 20 * correct >= 17 * support
            for correct, support in zip(measured["class_correct"], measured["class_support"])
        ),
    }


def clopper_pearson(k: int, n: int, alpha: float = .05) -> tuple[float, float]:
    """Two-sided exact binomial interval, using only the standard library."""
    _need(type(k) is int and type(n) is int and 0 <= k <= n and n > 0,
          "binomial counts must satisfy 0 <= k <= n and n > 0")
    _need(0 < alpha < 1, "alpha must lie strictly between zero and one")

    def probability(p: float, start: int, end: int) -> float:
        return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(start, end + 1))

    def solve(start: int, end: int, increasing: bool) -> float:
        lo, hi = 0.0, 1.0
        target = alpha / 2
        for _ in range(100):
            middle = (lo + hi) / 2
            below = probability(middle, start, end) < target
            if below == increasing:
                lo = middle
            else:
                hi = middle
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(k, n, True)
    upper = 1.0 if k == n else solve(0, k, False)
    return lower, upper


def _runtime(runtime: Any) -> None:
    _need(isinstance(runtime, dict), "runtime receipt is missing")
    _need(isinstance(runtime.get("python"), str) and runtime["python"].startswith("3.12."),
          "Python 3.12 runtime is required")
    required = {
        "torch": "2.5.1+cpu", "numpy": "2.3.5", "device": "cpu", "dtype": "float32",
        "torch_threads": 1, "deterministic_algorithms": True,
    }
    for name, expected in required.items():
        _need(runtime.get(name) == expected and type(runtime.get(name)) is type(expected),
              f"runtime mismatch: {name}")


def _sources(hashes: Any, repo_root: Path) -> None:
    _need(isinstance(hashes, dict) and bool(hashes), "source hashes are missing")
    _need(set(REQUIRED_SOURCES).issubset(hashes), "required source files are absent from source lock")
    for relative, expected in hashes.items():
        _need(isinstance(expected, str) and bool(HEX64.fullmatch(expected)), f"invalid source digest: {relative}")
        path = _safe_file(repo_root, relative)
        _need(sha256_file(path) == expected, f"source hash mismatch: {relative}")


def _checkpoint(row: dict[str, Any], directory: Path) -> tuple[dict[str, Any], Path]:
    import torch

    checkpoint = row.get("checkpoint")
    _need(isinstance(checkpoint, dict), "checkpoint receipt is missing")
    expected = checkpoint.get("sha256")
    _need(isinstance(expected, str) and bool(HEX64.fullmatch(expected)), "invalid checkpoint digest")
    path = _safe_file(directory, checkpoint.get("path"))
    _need(sha256_file(path) == expected, "checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _need(isinstance(payload, dict), "invalid checkpoint payload")
    for name in ("seed", "width", "updates", "role"):
        _need(payload.get(name) == row[name], f"checkpoint metadata mismatch: {name}")
    state = payload.get("state_dict")
    _need(isinstance(state, dict), "checkpoint state_dict is missing")
    width = row["width"]
    shapes = {
        "cell.weight_ih": (3 * width, 4), "cell.weight_hh": (3 * width, width),
        "cell.bias_ih": (3 * width,), "cell.bias_hh": (3 * width,),
        "head.weight": (4, width), "head.bias": (4,),
    }
    _need(set(state) == set(shapes), "checkpoint parameter names differ from registered model")
    for name, shape in shapes.items():
        tensor = state[name]
        _need(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == shape, f"parameter shape mismatch: {name}")
        _need(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()),
              f"nonfinite or non-float32 parameter: {name}")
    count = sum(tensor.numel() for tensor in state.values())
    _need(count == 3 * width ** 2 + 22 * width + 4 == row["parameter_count"], "parameter count mismatch")
    return state, path


def _replay(state: dict[str, Any], width: int, split_rows: dict[str, Any]) -> dict[str, float]:
    """Read-only inference with separately assembled modules and an explicit loop."""
    import torch

    _need(torch.__version__ == "2.5.1+cpu" and np.__version__ == "2.3.5",
          "checkpoint replay requires the pinned numerical runtime")
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1889)  # Excluded smoke initialization, overwritten below.
        cell = torch.nn.GRUCell(4, width, device="cpu", dtype=torch.float32)
        head = torch.nn.Linear(width, 4, device="cpu", dtype=torch.float32)
    cell.load_state_dict({name.removeprefix("cell."): value for name, value in state.items() if name.startswith("cell.")})
    head.load_state_dict({name.removeprefix("head."): value for name, value in state.items() if name.startswith("head.")})
    cell.eval()
    head.eval()
    universe, _ = exact_universe()
    errors = {}
    with torch.no_grad():
        for name, saved in split_rows.items():
            tokens = torch.tensor(universe[np.asarray(saved["ids"], dtype=np.int64)], dtype=torch.int64, device="cpu")
            inputs = torch.nn.functional.one_hot(tokens, num_classes=4).to(torch.float32)
            hidden = torch.zeros(len(tokens), width, dtype=torch.float32, device="cpu")
            for position in range(6):
                hidden = cell(inputs[:, position], hidden)
                field = f"prefix{position + 1}_logits"
                if field in saved:
                    replay = head(hidden).to(torch.float64).numpy()
                    difference = float(np.abs(replay - np.asarray(saved[field], dtype=np.float64)).max())
                    errors[f"{name}.{field}"] = difference
                    _need(difference <= TOLERANCE, f"checkpoint replay mismatch: {name}.{field}")
            replay = head(hidden).to(torch.float64).numpy()
            difference = float(np.abs(replay - np.asarray(saved["logits"], dtype=np.float64)).max())
            errors[f"{name}.logits"] = difference
            _need(difference <= TOLERANCE, f"checkpoint replay mismatch: {name}.logits")
    return errors


def audit_root(row: dict[str, Any], output_dir: Path, role: str,
               expected_config: tuple[int, int] | None,
               expected_source_hashes: dict[str, str], expected_runtime: dict[str, Any],
               repo_root: Path | None = None) -> dict[str, Any]:
    """Validate one original record and reconstruct its competence gates."""
    _need(isinstance(row, dict) and type(row.get("schema")) is int and row["schema"] == SCHEMA, "invalid root schema")
    _need(row.get("role") == role, "root role mismatch")
    seed = _integer(row.get("seed"), "seed")
    width = _integer(row.get("width"), "width")
    updates = _integer(row.get("updates"), "updates")
    completed = _integer(row.get("completed_updates"), "completed_updates")
    _need(seed in (DESIGN_SEEDS if role == "design" else PROSPECTIVE_SEEDS), "seed outside registered roster")
    _need((width, updates) in CONFIGS, "unregistered candidate configuration")
    _need(expected_config is None or (width, updates) == expected_config, "root differs from locked candidate")
    _need(row.get("source_hashes") == expected_source_hashes, "root source lock differs from manifest")
    _need(row.get("runtime") == expected_runtime, "root runtime differs from manifest")
    _need(0 <= completed <= updates, "invalid completed update count")
    summary = {"seed": seed, "width": width, "updates": updates, "completed_updates": completed}
    if row.get("status") == "runtime_failure":
        _need(isinstance(row.get("exception"), str) and bool(row["exception"].strip()), "failed root lacks exception")
        _need(not row.get("splits"), "runtime-failed root unexpectedly contains scored splits")
        preserved_checkpoint = None
        if "checkpoint" in row:
            checkpoint = row["checkpoint"]
            _need(isinstance(checkpoint, dict), "invalid failed-root checkpoint receipt")
            expected = checkpoint.get("sha256")
            _need(isinstance(expected, str) and bool(HEX64.fullmatch(expected)), "invalid failed-root checkpoint digest")
            preserved_checkpoint = sha256_file(_safe_file(output_dir, checkpoint.get("path")))
            _need(preserved_checkpoint == expected, "failed-root checkpoint hash mismatch")
        return {**summary, "status": "runtime_failure", "passed": False,
                "exception": row["exception"], "gates": {"completed": False},
                "checkpoint_sha256": preserved_checkpoint}
    _need(row.get("status") == "complete", "unknown root status")
    _need(completed == updates, "completed root has wrong update budget")
    _need(isinstance(row.get("initial_state_sha256"), str) and bool(HEX64.fullmatch(row["initial_state_sha256"])),
          "initialization digest is missing or malformed")
    _need(_integer(row.get("parameter_count"), "parameter_count") == 3 * width ** 2 + 22 * width + 4, "wrong model parameter count")
    trace = row.get("trace")
    _need(isinstance(trace, list), "training trace is missing")
    _need([entry.get("updates") for entry in trace] == list(range(200, updates + 1, 200)), "training trace update roster mismatch")
    for entry in trace:
        _need(_finite(entry.get("loss"), "trace loss") >= 0, "negative training loss")
    _need(_finite(row.get("final_training_loss"), "final_training_loss") >= 0, "negative final training loss")
    split_rows = row.get("splits")
    expected_names = {"train", "validation"} | ({"heldout"} if role == "prospective" else set())
    _need(isinstance(split_rows, dict) and set(split_rows) == expected_names, "unexpected or missing scored partition")
    partitions = exact_partitions()
    calculated = {}
    for name, saved in split_rows.items():
        _need(isinstance(saved, dict), f"invalid partition receipt: {name}")
        ids, targets, values = _arrays(saved.get("ids"), saved.get("labels"), saved.get("logits"))
        _need(np.array_equal(ids, partitions[name]), f"ordered partition IDs mismatch: {name}")
        if "ids_sha256" in saved:
            _need(saved["ids_sha256"] == ids_sha256(ids), f"partition digest mismatch: {name}")
        _need(set(saved).issubset({"ids", "labels", "logits", "predictions", "prediction", "ids_sha256", "prefix2_logits", "prefix4_logits", "prefix2_predictions", "prefix4_predictions"}),
              f"unknown per-sequence field in {name}")
        for field in ("prediction", "predictions"):
            if field in saved:
                _need(np.array_equal(np.asarray(saved[field]), values.argmax(axis=1)), f"saved predictions disagree with logits: {name}")
        if name != "heldout":
            _need(not any(field.startswith("prefix") for field in saved),
                  f"unexpected prefix readout in {name}")
        else:
            for length in (2, 4):
                _, _, prefix_values = _arrays(ids, targets, saved.get(f"prefix{length}_logits"))
                field = f"prefix{length}_predictions"
                if field in saved:
                    _need(np.array_equal(np.asarray(saved[field]), prefix_values.argmax(axis=1)),
                          f"saved predictions disagree with logits: {name}.{field}")
        calculated[name] = metrics_from_logits(ids, targets, values)
    if "training_ids_sha256" in row:
        _need(row["training_ids_sha256"] == ids_sha256(partitions["train"]), "training IDs digest mismatch")
    checks = row.get("checks")
    _need(isinstance(checks, dict) and checks.get("weights_finite") is True, "finite-weight check failed")
    full_parity = _finite(checks.get("full_width_parity_max"), "full_width_parity_max")
    _need(0 <= full_parity <= TOLERANCE, "full-width parity check failed")
    state, checkpoint_path = _checkpoint(row, output_dir)
    replay_errors = _replay(state, width, split_rows)
    gates = {"receipt_integrity": True, "finite_weights_and_logits": True,
             "exact_updates": True, "full_width_parity": True, "checkpoint_replay": True}
    prefix = {}
    if role == "design":
        measured = calculated["validation"]
        gates.update(accuracy_recall_gates(measured, role))
    else:
        measured = calculated["heldout"]
        gates.update(accuracy_recall_gates(measured, role))
        ceilings = {key: feature_ceiling(partitions["heldout"], key) for key in ("prefix2", "prefix4", "counts", "last")}
        gates["counts_gap"] = 5 * (measured["correct"] - ceilings["counts"]) >= 2 * measured["n"]
        gates["prefix4_gap"] = 20 * (measured["correct"] - ceilings["prefix4"]) >= 7 * measured["n"]
        for size in (2, 4):
            field = f"prefix{size}_logits"
            saved = split_rows["heldout"]
            _need(field in saved, f"missing {field}")
            ids, targets, values = _arrays(saved["ids"], saved["labels"], saved[field])
            metric = metrics_from_logits(ids, targets, values)
            consistency = _prefix_consistency(ids, values, size)
            reported = _finite(checks.get(f"prefix{size}_consistency_max"), f"prefix{size}_consistency_max")
            _need(0 <= reported <= TOLERANCE and consistency <= TOLERANCE,
                  f"prefix{size} consistency check failed")
            gates[f"prefix{size}_bayes_ceiling"] = metric["correct"] <= ceilings[f"prefix{size}"]
            prefix[f"prefix{size}"] = {**metric, "consistency_max": consistency,
                                       "correct_ceiling": ceilings[f"prefix{size}"]}
        measured["correct_ceilings"] = ceilings
        measured["counts_margin"] = (measured["correct"] - ceilings["counts"]) / measured["n"]
        measured["prefix4_margin"] = (measured["correct"] - ceilings["prefix4"]) / measured["n"]
    return {**summary, "status": "complete", "passed": all(gates.values()), "gates": gates,
            "initial_state_sha256": row["initial_state_sha256"], "metrics": calculated,
            "prefix": prefix, "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_replay_max": max(replay_errors.values(), default=0.0),
            "replay_errors": replay_errors, "copied_aggregate_metrics_used": False}


def _lock(lock: Any, manifest: dict[str, Any]) -> tuple[int, int]:
    _need(isinstance(lock, dict) and type(lock.get("schema")) is int and lock["schema"] == SCHEMA, "invalid competence lock")
    config = (_integer(lock.get("width"), "locked width"), _integer(lock.get("updates"), "locked updates"))
    _need(config in CONFIGS, "locked configuration was not registered")
    _need(lock.get("roots") == list(PROSPECTIVE_SEEDS), "competence lock roster mismatch")
    _need(lock.get("source_hashes") == manifest["source_hashes"], "lock source hashes differ from manifest")
    _need(lock.get("runtime") == manifest["runtime"], "lock runtime differs from manifest")
    _need(lock.get("partition_hashes") == {name: ids_sha256(ids) for name, ids in exact_partitions().items()},
          "lock partition hashes disagree with exact partition")
    for field in ("protocol_commit", "source_commit"):
        _need(isinstance(lock.get(field), str) and bool(HEX40.fullmatch(lock[field])), f"invalid {field}")
    _need(lock["source_commit"] == manifest.get("source_commit"), "lock source commit differs from manifest")
    for field in ("design_manifest_sha256", "design_audit_sha256", "design_selection_sha256"):
        _need(isinstance(lock.get(field), str) and bool(HEX64.fullmatch(lock[field])), f"invalid {field}")
    return config


def audit_role(output_dir: str | Path, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Audit a complete role directory; integrity failure always produces HOLD.

    Failures are returned as data so the original cohort can be archived even
    when a record is missing or corrupt.  No failure is omitted from its roster.
    """
    directory = Path(output_dir)
    repository = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[4]
    errors: list[str] = []
    summaries: list[dict[str, Any]] = []
    role = "unknown"
    manifest: dict[str, Any] = {}
    config = None
    seen: set[tuple[int, int, int]] = set()
    try:
        manifest_path = _safe_file(directory, "manifest.json")
        manifest = json.loads(manifest_path.read_text())
        _need(isinstance(manifest, dict) and type(manifest.get("schema")) is int and manifest["schema"] == SCHEMA, "invalid manifest schema")
        role = manifest.get("role")
        _need(role in ("design", "prospective"), "invalid manifest role")
        _runtime(manifest.get("runtime"))
        _sources(manifest.get("source_hashes"), repository)
        _need(isinstance(manifest.get("source_commit"), str) and bool(HEX40.fullmatch(manifest["source_commit"])), "invalid manifest source commit")
        _need(manifest.get("partition_hashes") == {name: ids_sha256(ids) for name, ids in exact_partitions().items()},
              "manifest partition hashes disagree with exact partition")
        paths = manifest.get("roots")
        _need(isinstance(paths, list) and all(isinstance(path, str) for path in paths), "invalid root-file roster")
        _need(len(paths) == len(set(paths)), "duplicate root-file paths")
        if role == "prospective":
            lock_path = _safe_file(directory, "lock.json")
            registered_lock_path = _safe_file(repository, SOURCE_PREFIX + "COMPETENCE_LOCK.json")
            _need(sha256_file(lock_path) == sha256_file(registered_lock_path), "copied competence lock differs from repository lock")
            config = _lock(json.loads(lock_path.read_text()), manifest)
        resolved = set()
        for relative in paths:
            row: dict[str, Any] = {}
            try:
                path = _safe_file(directory, relative)
                _need(path not in resolved, "root-file paths alias the same file")
                resolved.add(path)
                row = json.loads(path.read_text())
                _need(isinstance(row, dict), "invalid root JSON object")
                key = tuple(_integer(row.get(name), name) for name in ("seed", "width", "updates"))
                _need(key not in seen, "duplicate seed/configuration record")
                seen.add(key)
                summary = audit_root(row, directory, role, config, manifest["source_hashes"], manifest["runtime"], repository)
                summaries.append({"path": relative, **summary})
            except Exception as exc:
                message = f"{relative}: {type(exc).__name__}: {exc}"
                errors.append(message)
                summaries.append({"path": relative, "seed": row.get("seed"), "width": row.get("width"),
                                  "updates": row.get("updates"), "status": "invalid_receipt",
                                  "passed": False, "error": message})
        expected = ({(seed, width, updates) for seed in DESIGN_SEEDS for width, updates in CONFIGS}
                    if role == "design" else {(seed, *config) for seed in PROSPECTIVE_SEEDS})
        _need(seen == expected, f"root roster mismatch; missing={sorted(expected - seen)}, unexpected={sorted(seen - expected)}")
        initialization_groups: dict[tuple[int, int], set[str]] = defaultdict(set)
        for row in summaries:
            if row.get("status") == "complete":
                initialization_groups[(row["seed"], row["width"])].add(row["initial_state_sha256"])
        _need(all(len(values) == 1 for values in initialization_groups.values()),
              "design checkpoints disagree on initialization digest")
        hashes = [next(iter(values)) for values in initialization_groups.values()]
        _need(len(hashes) == len(set(hashes)), "different registered initializations share a digest")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    result: dict[str, Any] = {
        "schema": SCHEMA, "role": role, "audit_valid": not errors,
        "errors": errors, "root_results": summaries, "n_records": len(summaries),
        "n_expected_records": 12 if role == "design" else 8 if role == "prospective" else None,
        "oracle_certificate": exact_certificate(), "causal_units_measured": 0,
        "physical_dose": None, "causal_effect_floor": None, "matched_sham_floor": None,
        "causal_power": None, "scientific_status": "CAUSAL_STUDY_NOT_RUN",
    }
    manifest_file = directory / "manifest.json"
    if manifest_file.is_file():
        result["manifest_sha256"] = sha256_file(manifest_file)
    if role == "design":
        candidates = []
        for width, updates in CONFIGS:
            matches = [row for row in summaries if (row.get("width"), row.get("updates")) == (width, updates)]
            passed = not errors and len(matches) == 3 and all(row.get("passed") for row in matches)
            candidates.append({"width": width, "updates": updates,
                               "passing_seeds": [row["seed"] for row in matches if row.get("passed")],
                               "passed": passed, "cost": updates * (3 * width ** 2 + 22 * width + 4)})
        accepted = sorted((row for row in candidates if row["passed"]), key=lambda row: (row["cost"], row["width"], row["updates"]))
        selected = {key: accepted[0][key] for key in ("width", "updates")} if accepted else None
        result.update({"n_roots": 6, "n_seed_ids": 3, "candidate_results": candidates,
                       "selected_config": selected, "verdict": "DESIGN_PASS" if selected else "DESIGN_HOLD"})
        copied = manifest.get("selection")
        if copied is not None:
            expected_selection = copied.get("selected_config") if isinstance(copied, dict) else None
            if expected_selection != selected:
                result["errors"].append("saved candidate selection differs from independent audit")
                result.update({"audit_valid": False, "selected_config": None, "verdict": "DESIGN_HOLD"})
    elif role == "prospective":
        passed = sum(bool(row.get("passed")) for row in summaries)
        result.update({"n_roots": 8, "passed_roots": passed, "failed_roots": 8 - min(passed, 8),
                       "root_success_fraction": min(passed, 8) / 8,
                       "root_success_exact95": list(clopper_pearson(min(passed, 8), 8)),
                       "selected_config": {"width": config[0], "updates": config[1]} if config else None,
                       "verdict": "SOURCE_COMPETENCE_PASS" if not errors and passed >= 7 else "SOURCE_COMPETENCE_HOLD"})
    else:
        result["verdict"] = "AUDIT_HOLD"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_role(args.directory, args.repo_root)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if result["audit_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
