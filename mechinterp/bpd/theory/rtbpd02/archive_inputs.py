"""Hash-gated access to the registered COMP14 CAL vectors and IO18 factor banks."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np


COMP = "experiments/nan_comp14"
FITS = "experiments/nan_fit16_io18"
RUN_COMP = "runs/nan_comp14"
RUN_FITS = "runs/nan_fit16_io18"
IDENTITY = ("id", "root", "split", "arm", "world")
WORLDS = ("A", "B")
SITES = ("down_proj_input", "down_proj_output", "residual_after_layer")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    require(sha256(data).hexdigest() == expected, f"archive hash mismatch for {path.name}")
    return data


def bound_json(path: Path, expected: str):
    return json.loads(bound(path, expected))


def select_cal(payload: bytes, spec: dict):
    entries = [
        (i, row) for i, line in enumerate(payload.splitlines())
        if ((row := json.loads(line))["split"], row["arm"]) == (spec["fit_split"], spec["arm"])
    ]
    pairs = {(row["root"], row["world"]) for _, row in entries}
    roots = tuple(sorted({root for root, _ in pairs}))
    require(
        len(roots) == spec["expected_roots"] and len(entries) == 2 * len(roots)
        and pairs == {(root, w) for root in roots for w in WORLDS}
        and len({row["id"] for _, row in entries}) == len(entries),
        "registered CAL roots must have exactly one member per world",
    )
    return entries, roots


def verify_case(case, audit, public, tokens, case_stem, passes):
    require(all(case[k] == public[k] == audit[k] for k in IDENTITY), "case identity mismatch")
    ids = case["initial_prompt_ids"]
    digest = sha256(json.dumps(ids).encode()).hexdigest()
    require(
        case["status"] == "completed" and case["state_file"] == case_stem + "_states.npz"
        and case["state_file_sha256"] == audit["state_sha256"]
        and all(type(x) is int and x >= 0 for x in ids) and len(ids) == tokens["tokens"] > 0
        and digest == case["initial_prompt_ids_sha256"] == tokens["ids_sha256"]
        and case["steps"][0]["step"] == 0 and case["steps"][0]["history_tokens"] == len(ids)
        and [p["loop"] for p in case["initial_phases"]] == passes,
        "case token, pass or state custody mismatch",
    )


def state_arrays(case, content, layer, passes, input_width, output_width):
    rows = case["state_index"]
    keys = [r["array_key"] for r in rows]
    matches = [
        ((r["loop"], r["site"]), r["array_key"]) for r in rows
        if r["layer_zero_based"] == layer and r["site"] in SITES and r["loop"] in passes
    ]
    index = dict(matches)
    require(len(matches) == len(index) == len(passes) * len(SITES), "missing or duplicate physical site")
    require(set(index) == {(p, site) for p in passes for site in SITES}, "wrong archived sites")
    vectors = {}
    with np.load(BytesIO(content), allow_pickle=False) as arrays:
        require(len(keys) == len(set(keys)) == len(arrays.files) and set(keys) == set(arrays.files),
                "state payload layout disagrees with its case index")
        for p in passes:
            parts = []
            for site, width in zip(SITES, (input_width, output_width, output_width), strict=True):
                vector = arrays[index[p, site]]
                require(vector.shape == (width,) and vector.dtype == np.float32 and np.isfinite(vector).all(),
                        "physical vector has an unexpected shape, dtype or nonfinite value")
                parts.append(vector)
            vectors[p] = tuple(parts)
    return vectors


@dataclass(frozen=True)
class Occurrence:
    x: np.ndarray
    native: np.ndarray
    carrier: np.ndarray


@dataclass(frozen=True)
class Bank:
    v: np.ndarray
    u: np.ndarray

    def apply(self, rows):
        return (rows @ self.v) @ self.u

    def atoms(self, rows):
        return (rows @ self.v)[..., None] * self.u


def read_factors(path, digest, keys, dimensions):
    with np.load(BytesIO(bound(path, digest)), allow_pickle=False) as archive:
        v, u = (np.asarray(archive[name], dtype=np.float64) for name in keys)
    require(
        (v.shape, u.shape) == dimensions and np.isfinite(v).all() and np.isfinite(u).all()
        and np.linalg.matrix_rank(v) == np.linalg.matrix_rank(u) == dimensions[0][1],
        "physical factor bank shape, rank or finite numerics changed",
    )
    return Bank(v, u)


def read_captures(repo, backend, data, base, roster, roots, spec, check):
    pins = spec["source_file_sha256"]
    source_sha = pins[f"{COMP}/SOURCE_MANIFEST.json"]
    cfg = base["backends"][backend]
    archive = repo / RUN_COMP / backend
    audit = data[f"{RUN_COMP}/audits/{backend}.json"]
    require(pins[f"{RUN_COMP}/audits/{backend}.json"] == cfg["audit"], "inconsistent independent COMP14 audit pin")
    manifest = bound_json(archive / "manifest.json", cfg["manifest"])
    require(
        audit["status"] == "passed" and audit["worker_status"] == "completed"
        and audit["source_manifest_sha256"] == manifest["source_manifest_sha256"] == source_sha
        and audit["backend"] == manifest["backend"] == backend
        and not audit["sealed_test_accessed"] and not manifest["sealed_test_accessed"]
        and audit["missing_contexts"] == 0
        and audit["contexts_checked"] == audit["registered_contexts"] == len(audit["records"]) == 240
        and audit["physical_vectors_checked"] == 40 * 240
        and manifest["checkpoint_revision"] == base["revision"]
        and manifest["target_parameter_sha256"] == cfg["target"],
        "original COMP14 closure or native identity does not match the registration",
    )
    token_receipt = data[f"{COMP}/fixtures/tokenizer_receipt.json"]
    require(not token_receipt["sealed_test_accessed"], "token receipt scope changed")
    cases = {}
    custody = []
    for slot, public in roster:
        check()
        receipt = audit["records"][slot]
        stem = f"case_{slot:04d}"
        case = bound_json(archive / (stem + ".json"), receipt["case_sha256"])
        verify_case(case, receipt, public, token_receipt["contexts"][public["id"]], stem, spec["passes"])
        payload = bound(archive / (stem + "_states.npz"), receipt["state_sha256"])
        cases[public["root"], public["world"]] = state_arrays(
            case, payload, spec["layer_zero_based"], spec["passes"], base["input_width"], base["width"],
        )
        custody.append((receipt["case_sha256"], receipt["state_sha256"], case["initial_prompt_ids_sha256"]))
    result = {}
    for p in spec["passes"]:
        result[p] = Occurrence(*[
            np.stack([np.stack([cases[root, world][p][i] for world in WORLDS]) for root in roots])
            for i in range(len(SITES))
        ])
    return result, sha256(json.dumps(custody, separators=(",", ":")).encode()).hexdigest()


def read_banks(repo, backend, data, base, spec):
    pins = spec["source_file_sha256"]
    source_sha = pins[f"{FITS}/SOURCE_MANIFEST.json"]
    source = data[f"{FITS}/SOURCE_MANIFEST.json"]
    old = data["reports/nanbeige_bpd_gym/nan_fit16_io18_progress.json"]
    init_pin = source["initialization"][backend]
    cfg = base["backends"][backend]
    old_backend = old["backends"][backend]
    audit_path = f"{RUN_FITS}/audits/{backend}.json"
    audit = data[audit_path]
    dimensions = ((base["input_width"], base["rank"]), (base["rank"], base["width"]))
    require(
        old["source_manifest_sha256"] == audit["source_manifest_sha256"] == source_sha
        and old_backend["status"] == "completed_audited" and audit["status"] == "passed"
        and audit["backend"] == backend and audit["worker_status"] == "completed" and not audit["sealed_test_accessed"]
        and old_backend["receipt_sha256"] == audit["receipt_sha256"]
        and old_backend["audit_sha256"] == pins[audit_path]
        and sorted(c["passes"] for c in audit["cells"]) == spec["passes"],
        "IO18 fit group has no matching independent audit",
    )
    initial = read_factors(
        repo / f"runs/nan_gym05/{backend}/linear_baselines/layer21_down_proj.npz",
        init_pin["file_sha256"], ("V8", "U8"), dimensions,
    )
    banks, receipts = {"initializer": initial}, {"initializer_sha256": init_pin["file_sha256"]}
    for cell in audit["cells"]:
        phase = cell["passes"]
        folder = repo / RUN_FITS / backend / f"t{phase}_layer21_down_proj_rank8"
        receipt = bound_json(folder / "receipt.json", cell["receipt_sha256"])
        require(
            cell["status"] == "passed" and receipt["status"] == "completed"
            and receipt["backend"] == backend and receipt["passes"] == phase
            and receipt["layer_zero_based"] == spec["layer_zero_based"] and receipt["module"] == "down_proj"
            and receipt["rank"] == base["rank"] and not receipt["sealed_test_accessed"]
            and receipt["source_manifest_sha256"] == source_sha
            and receipt["target_parameter_sha256_unchanged"] == cfg["target"]
            and receipt["initial_svd_file_sha256"] == init_pin["file_sha256"]
            and receipt["effective_operator_sha256"] == init_pin["effective_operator_sha256"]
            and receipt["native_all_one_full_head_bit_exact"] is True
            and receipt["native_all_one_boundary_states_bit_exact"] is True
            and receipt["optimizer_steps_completed"] == 16,
            "IO18 receipt changed native target, same-initialization, or exact parity qualification",
        )
        bank_pair = []
        for stage, steps, filename in (("initial_checkpoint", 0, "initial.npz"), ("final_checkpoint", 16, "final.npz")):
            checkpoint = receipt[stage]
            require(checkpoint["optimizer_steps_completed"] == steps and checkpoint["file"] == filename,
                    "IO18 checkpoint path or optimizer step changed")
            if steps:
                require(checkpoint["sha256"] == cell["checkpoint_sha256"], "independent fit payload audit mismatch")
            bank_pair.append(read_factors(folder / filename, checkpoint["sha256"], ("V", "U"), dimensions))
        require(np.array_equal(initial.v, bank_pair[0].v) and np.array_equal(initial.u, bank_pair[0].u),
                "IO18 did not start from the frozen SVD initializer")
        name = f"t{phase}"
        if phase == 2:
            require(cell["receipt_sha256"] == cfg["factor_receipt"] and cell["checkpoint_sha256"] == cfg["factor_bank"],
                    "T2 target differs from the previously pinned physical bank")
        banks[name] = bank_pair[1]
        receipts[name] = {
            "receipt_sha256": cell["receipt_sha256"],
            "factor_sha256": cell["checkpoint_sha256"],
            "original_native_all_one_full_head_bit_exact": True,
            "original_native_all_one_boundary_states_bit_exact": True,
            "original_backend_effective_recomposition_relative_l2": receipt["final_effective_operator_closure"]["relative_l2"],
        }
    return banks, receipts


def load_inputs(repo, spec, check):
    files = spec["source_file_sha256"]
    roster_path = f"{COMP}/fixtures/public.jsonl"
    roster, roots = select_cal(bound(repo / roster_path, files[roster_path]), spec)
    data = {path: bound_json(repo / path, digest) for path, digest in files.items() if path != roster_path}
    base = data["experiments/nan_readpath37/registration.json"]
    require(
        base["revision"] == spec["snapshot_revision"] == data[f"{COMP}/SOURCE_MANIFEST.json"]["checkpoint_revision"]
        and base["width"] == 3072 and base["input_width"] == 10752 and base["rank"] == 8,
        "parent dimensions or checkpoint revision changed",
    )
    states, banks, custody = {}, {}, {}
    for backend in spec["backends"]:
        states[backend], root_hash = read_captures(repo, backend, data, base, roster, roots, spec, check)
        banks[backend], qualified = read_banks(repo, backend, data, base, spec)
        custody[backend] = {
            "ordered_case_state_token_custody_sha256": root_hash,
            "native_manifest_sha256": base["backends"][backend]["manifest"],
            "native_target_parameter_sha256": base["backends"][backend]["target"],
            "fit_qualification": qualified,
            "cases_verified": 2 * len(roots),
            "vectors_verified": 2 * len(roots) * len(spec["passes"]) * len(SITES),
        }
    return states, banks, custody, data["experiments/nan_bpd01/receipts/checkpoint_verification.json"], (
        data[f"{COMP}/fixtures/tokenizer_receipt.json"]["tokenizer_config_sha256"]
    )
