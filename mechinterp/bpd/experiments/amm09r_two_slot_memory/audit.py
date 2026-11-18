"""Independent AMM09R1 receipt audit and pinned EvalMath execution.

Standard library only. No model imports, training, donor generation, or
confirmation access. Counts are reconstructed before root-level inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
NAME = "BPD-AMM-09R1"
SCHEMA = "bpd.amm09r1.two_slot_memory.v1"
SOURCE_FILES = ("__init__.py", "models.py", "case.py", "experiment.py", "run.py",
                "test_experiment.py", "RUNTIME.json", "PROTOCOL.md", "audit.py",
                "../../../../.github/workflows/bpd-amm09r1-fresh-development.yml")
PLANNER_COMMIT = "6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a"
PLANNER_SHA256 = "c2621a7900930f81ed53799ce439ee1c8f2d7006918a04ef57b5381630d15494"
PLANNER_GIT_BLOB = "0ff062f76f5534d9f18425f986ae3e750397f50c"
ENGINEERING_REGISTRATION = "ffc7143e3b597d840c5000a94a45b45d964931cf"
ENGINEERING_GATES = {
    "all_registered_repeats_present", "all_receipts_valid", "pinned_payloads_bit_exact",
    "pinned_sources_equal", "all_registered_sources_equal", "registered_software",
    "pinned_runtime_settings", "torch_binaries_equal", "two_distinct_hosts",
}
N_ROOTS, N_CASES, N_TRAIN, N_CAL = 64, 512, 128, 16
MASKS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))
CANDIDATES = tuple(f"mask_{i}" for i in range(6))
CONTROLS = ("random_dark", "random_dark_b", "sham", "full_donor")
CHOICES = ("memory_plus_local", "local_ridge", "memory_only", "local_count", "pooled", "random_mask")
ARMS = CHOICES + CONTROLS
COMPARISONS = (
    ("memory_only_vs_pooled", "memory_only", "pooled"),
    ("memory_plus_local_vs_pooled", "memory_plus_local", "pooled"),
    ("memory_plus_local_vs_local_ridge", "memory_plus_local", "local_ridge"),
    ("memory_plus_local_vs_memory_only", "memory_plus_local", "memory_only"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _pairs(values):
    result = {}
    for key, value in values:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def read_json(path):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs,
                      parse_constant=reject_constant)


def source_hashes():
    return {name: file_sha(ROOT / name) for name in SOURCE_FILES}


def number(value, context, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value), context + ": finite number required")
    require(not nonnegative or value >= 0, context + ": negative value")
    return value


def integer(value, context, lower=0, upper=None):
    require(type(value) is int and value >= lower and (upper is None or value <= upper),
            context + ": invalid integer")
    return value


def counts_from_bits(bits, n, keys, context):
    require(isinstance(bits, dict) and set(bits) == set(keys), context + ": wrong outcome set")
    for key, value in bits.items():
        require(isinstance(value, str) and len(value) == n and not set(value) - {"0", "1"},
                context + ": malformed " + key)
    return {key: bits[key].count("1") for key in keys}


def check_counts(recorded, rebuilt, context):
    require(isinstance(recorded, dict) and set(recorded) == set(rebuilt), context + ": wrong count set")
    require(all(type(recorded[key]) is int and recorded[key] == value for key, value in rebuilt.items()),
            context + ": counts disagree with bits")


def weights(values, context, quantized=False):
    require(isinstance(values, list) and len(values) == 18, context + ": expected 18 coefficients")
    for value in values:
        number(value, context)
        require(not quantized or round(value, 4) == value, context + ": unquantized coefficient")


def fidelity(fit):
    required = ("teacher_task_mse", "factor_task_mse", "weight_reconstruction_relative", "min_rank3_gap")
    require(isinstance(fit, dict), "missing fitted-model diagnostics")
    for key in required:
        number(fit.get(key), "fit/" + key, nonnegative=True)
    return (fit["teacher_task_mse"] <= 1e-4 and fit["factor_task_mse"] <= .02
            and fit["weight_reconstruction_relative"] <= .10 and fit["min_rank3_gap"] > .01)


def inspect_assay(assay, root, split, n, *, full_features=False):
    context = root + "/" + split
    require(isinstance(assay, dict) and assay.get("root") == root
            and assay.get("split") == split and type(assay.get("n")) is int and assay["n"] == n,
            context + ": wrong assay membership")
    counts = counts_from_bits(assay.get("candidate_bits"), n, CANDIDATES + CONTROLS, context)
    check_counts(assay.get("candidate_hits"), counts, context)
    bits = assay["candidate_bits"]
    oracle = sum(any(bits[key][i] == "1" for key in CANDIDATES) for i in range(n))
    diagnostics = assay.get("diagnostics")
    require(isinstance(diagnostics, dict), context + ": missing diagnostics")
    require(type(diagnostics.get("casewise_oracle_hits_audit_only")) is int
            and diagnostics["casewise_oracle_hits_audit_only"] == oracle,
            context + ": six-mask oracle mismatch")
    require(type(diagnostics.get("full_donor_hits_audit_only")) is int
            and diagnostics["full_donor_hits_audit_only"] == counts["full_donor"],
            context + ": full donor accounting mismatch")
    for key in ("max_equal_dose_error", "max_immediate_readout_displacement",
                "full_donor_equal_dose_error", "full_donor_immediate_readout_displacement",
                "min_singular_gap", "mean_dose"):
        number(diagnostics.get(key), context + "/" + key, nonnegative=True)
    integer(diagnostics.get("zero_dose_pairs"), context + "/zero_dose_pairs", upper=n)
    degenerate = diagnostics.get("degenerate_mask_projection_pairs")
    require(isinstance(degenerate, list) and len(degenerate) == 6, context + ": wrong fallback counts")
    for value in degenerate:
        integer(value, context + "/fallback", upper=n)
    if full_features or "features" in assay:
        features = assay.get("features")
        require(isinstance(features, list) and len(features) == n, context + ": missing feature rows")
        for row in features:
            require(isinstance(row, list) and len(row) == 6, context + ": wrong feature action count")
            for vector in row:
                require(isinstance(vector, list) and len(vector) == 18, context + ": wrong feature width")
                for value in vector:
                    number(value, context + "/feature")
        require(digest(features) == assay.get("features_sha256"), context + ": feature digest mismatch")
    feature_hash = assay.get("features_sha256")
    require(isinstance(feature_hash, str) and len(feature_hash) == 64
            and not set(feature_hash) - set("0123456789abcdef"), context + ": missing feature digest")
    return counts, oracle, diagnostics


def validate_runtime(signature, manifest):
    require(isinstance(signature, dict), "missing runtime signature")
    for key, value in manifest["versions"].items():
        require(signature.get(key) == value, "runtime version mismatch: " + key)
    require(signature.get("environment") == manifest["environment"], "runtime environment mismatch")
    require(signature.get("torch_git") == manifest["torch_git"], "accepted torch source mismatch")
    require(signature.get("torch_binary_sha256") == manifest["torch_binary_sha256"], "accepted torch binary mismatch")
    require(type(signature.get("num_threads")) is int and signature["num_threads"] == 1
            and type(signature.get("num_interop_threads")) is int and signature["num_interop_threads"] == 1,
            "runtime thread count mismatch")
    require(signature.get("default_dtype") == "torch.float32"
            and signature.get("cpu_capability") in ("NO AVX", "DEFAULT")
            and signature.get("deterministic_algorithms") is True, "unregistered numerical runtime")


def inspect_training(training, sources, runtime, acceptance_sha):
    require(training.get("schema") == SCHEMA + "/training_memory" and training.get("phase") == "training"
            and training.get("experiment") == NAME, "wrong training schema/phase/experiment")
    require(training.get("source_sha256") == sources, "training source mismatch")
    require(training.get("n_independent_training_roots") == N_ROOTS
            and training.get("training_cases_per_root") == N_TRAIN and training.get("ridge_lambda") == 10.,
            "training budget or ridge penalty mismatch")
    require(training.get("engineering_acceptance_sha256") == acceptance_sha == runtime["engineering_acceptance_sha256"],
            "training engineering acceptance digest mismatch")
    validate_runtime(training.get("runtime_signature"), runtime)
    rows = training.get("rows")
    require(isinstance(rows, list) and [row.get("root") for row in rows]
            == [f"training/{index:04d}" for index in range(N_ROOTS)], "wrong training root order/membership")
    totals = {key: 0 for key in CANDIDATES}
    physical = []
    for index, row in enumerate(rows):
        require(row.get("family") == ("wren" if index % 2 == 0 else "basalt"), "training family mismatch")
        fidelity(row.get("fit"))
        counts, _, diagnostics = inspect_assay(row.get("assay"), row["root"], "training_pairs", N_TRAIN,
                                               full_features=True)
        for key in CANDIDATES:
            totals[key] += counts[key]
        physical.append(diagnostics)
    check_counts(training.get("candidate_hits"), totals, "training aggregate")
    pooled = min(range(6), key=lambda action: (-totals[f"mask_{action}"], action))
    require(type(training.get("pooled_action")) is int and training["pooled_action"] == pooled,
            "pooled action differs from training-best permitted mask")
    weights(training.get("global_coefficients"), "training prior", quantized=True)
    bit_rows = [{"root": row["root"], "candidate_bits": row["assay"]["candidate_bits"]} for row in rows]
    require(digest(bit_rows) == training.get("train_bits_sha256"), "training bit digest mismatch")
    require(digest(rows) == training.get("training_rows_sha256"), "training full-row digest mismatch")
    content = {key: value for key, value in training.items() if key not in ("sha256", "rows")}
    require(digest(content) == training.get("sha256"), "training policy digest mismatch")
    lock = {
        "schema": SCHEMA + "/training_lock", "source_sha256": sources,
        "training_policy_sha256": training["sha256"], "independent_roots": N_ROOTS,
        "pooled_action": pooled, "train_candidate_hits": totals,
        "training_case_bits_sha256": training["train_bits_sha256"],
        "training_rows_sha256": training["training_rows_sha256"],
        "runtime_signature": training["runtime_signature"],
        "engineering_acceptance_sha256": acceptance_sha,
        "global_coefficients": training["global_coefficients"],
    }
    return lock, physical


def selected_bits(candidates, actions, n, context):
    require(isinstance(actions, list) and len(actions) == n
            and all(type(action) is int and 0 <= action < 6 for action in actions),
            context + ": invalid or inadmissible action")
    return "".join(candidates[f"mask_{action}"][i] for i, action in enumerate(actions))


def seed(*parts):
    text = "AMM09R1/" + "/".join(map(str, parts))
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


def inspect_development(development, training, sources):
    require(development.get("schema") == SCHEMA + "/cohort" and development.get("experiment") == NAME
            and development.get("phase") == "development", "wrong development identity")
    for field, expected in (("n_independent_roots", N_ROOTS), ("n_calibration_pairs_per_root", N_CAL),
                            ("n_test_pairs_per_root", N_CASES), ("n_repeatability_pairs_per_root", N_CASES),
                            ("total_test_cases", N_ROOTS * N_CASES)):
        require(type(development.get(field)) is int and development[field] == expected, "wrong " + field)
    require(development.get("source_sha256") == sources, "development source mismatch")
    require(development.get("training_policy_sha256") == training["sha256"], "development training-policy mismatch")
    require(development.get("runtime_signature") == training["runtime_signature"], "development runtime mismatch")
    require(development.get("engineering_acceptance_sha256") == training["engineering_acceptance_sha256"],
            "development engineering acceptance mismatch")
    require(development.get("pooled_action") == training["pooled_action"]
            and development.get("global_coefficients") == training["global_coefficients"], "development prior mismatch")
    rows = development.get("rows")
    require(isinstance(rows, list) and [row.get("root") for row in rows]
            == [f"development/{index:04d}" for index in range(N_ROOTS)], "wrong development root order/membership")
    totals = {arm: 0 for arm in ARMS}
    histograms = {arm: Counter() for arm in CHOICES}
    derived, physical = [], []
    flips = 0
    pooled = training["pooled_action"]
    for index, row in enumerate(rows):
        root = row["root"]
        require(row.get("family") == ("wren" if index % 2 == 0 else "basalt"), root + ": family mismatch")
        _, oracle, diagnostics = inspect_assay(row.get("heldout"), root, "heldout", N_CASES)
        physical.append(diagnostics)
        candidates = row["heldout"]["candidate_bits"]
        actions = row.get("policy_actions")
        require(isinstance(actions, dict) and set(actions) == set(CHOICES), root + ": wrong policy action set")
        rebuilt = {arm: selected_bits(candidates, actions[arm], N_CASES, root + "/" + arm) for arm in CHOICES}
        rebuilt.update({arm: candidates[arm] for arm in CONTROLS})
        policy_counts = counts_from_bits(row.get("policy_bits"), N_CASES, ARMS, root + "/policy")
        require(row["policy_bits"] == rebuilt, root + ": selected outcome bits mismatch")
        rebuilt_counts = {arm: value.count("1") for arm, value in rebuilt.items()}
        check_counts(row.get("policy_hits"), rebuilt_counts, root + "/per-root policy")
        require(policy_counts == rebuilt_counts, root + ": policy reconstruction mismatch")
        policy = row.get("local_policy")
        require(isinstance(policy, dict) and policy.get("pooled_action") == pooled
                and policy.get("prior_weights") == training["global_coefficients"]
                and policy.get("noisy_label_count") == N_CAL * 6, root + ": local policy metadata mismatch")
        weights(policy.get("residual_weights"), root + "/residual")
        weights(policy.get("local_weights"), root + "/local")
        require(actions["pooled"] == [pooled] * N_CASES, root + ": pooled mask changed")
        require(actions["random_mask"] == [seed(root, "random_mask", i) % 6 for i in range(N_CASES)],
                root + ": random-mask action stream mismatch")
        _, _, cal_diagnostics = inspect_assay(row.get("calibration_clean_audit_only"), root, "calibration", N_CAL)
        physical.append(cal_diagnostics)
        feedback = row.get("agent_noisy_feedback")
        require(isinstance(feedback, dict) and feedback.get("schema") == SCHEMA + "/agent_noisy_feedback"
                and feedback.get("noisy_labels") == N_CAL * 6 and feedback.get("n_local_cases") == N_CAL
                and feedback.get("known_judge_error") == .20, root + ": calibration budget/schema mismatch")
        reported = counts_from_bits(feedback.get("reported_bits"), N_CAL, CANDIDATES, root + "/noisy labels")
        check_counts(feedback.get("reported_hits"), reported, root + "/noisy labels")
        expected_local = min(range(6), key=lambda action: (-reported[f"mask_{action}"], action))
        require(type(policy.get("local_count_action")) is int and policy["local_count_action"] == expected_local
                and actions["local_count"] == [expected_local] * N_CASES, root + ": local-count policy mismatch")
        clean = row["calibration_clean_audit_only"]["candidate_bits"]
        root_flips = sum(a != b for key in CANDIDATES for a, b in zip(clean[key], feedback["reported_bits"][key]))
        flips += root_flips
        repeat = row.get("repeatability")
        require(isinstance(repeat, dict), root + ": missing repeatability")
        _, _, repeat_diagnostics = inspect_assay(repeat.get("assay"), root, "repeatability", N_CASES)
        physical.append(repeat_diagnostics)
        repeat_bits = repeat["assay"]["candidate_bits"]
        repeat_pooled = repeat_bits[f"mask_{pooled}"]
        repeat_local = selected_bits(repeat_bits, repeat.get("local_ridge_actions"), N_CASES, root + "/repeat-local")
        require(repeat.get("pooled_bits") == repeat_pooled and repeat.get("local_ridge_bits") == repeat_local,
                root + ": repeatability selected bits mismatch")
        repeat_hits = {"pooled_hits": repeat_pooled.count("1"), "local_ridge_hits": repeat_local.count("1")}
        check_counts({key: repeat.get(key) for key in repeat_hits}, repeat_hits, root + "/repeatability")
        faithful = fidelity(row.get("fit"))
        require(type(row.get("fidelity_audit_only")) is bool and row["fidelity_audit_only"] == faithful,
                root + ": fidelity flag mismatch")
        for arm in ARMS:
            totals[arm] += rebuilt_counts[arm]
        for arm in CHOICES:
            histograms[arm].update(actions[arm])
        derived.append({"root": root, "family": row["family"], "policy_hits": rebuilt_counts,
                        "repeat_pooled_hits": repeat_hits["pooled_hits"], "fidelity": faithful,
                        "oracle_hits": oracle, "noisy_label_flips": root_flips,
                        "differences": {name: rebuilt_counts[treatment] - rebuilt_counts[control]
                                        for name, treatment, control in COMPARISONS}})
    check_counts(development.get("totals"), totals, "development aggregate")
    high_fidelity = sum(row["fidelity"] for row in derived)
    oracle_gain = sum(row["oracle_hits"] - row["policy_hits"]["pooled"] for row in derived)
    require(development.get("high_fidelity_roots") == high_fidelity, "development fidelity summary mismatch")
    require(development.get("oracle_headroom_hits_audit_only") == oracle_gain
            and development.get("oracle_headroom_rate_audit_only") == oracle_gain / (N_ROOTS * N_CASES),
            "development oracle summary mismatch")
    held_diagnostics = [row["heldout"]["diagnostics"] for row in rows]
    require(development.get("max_equal_dose_error") == max(row["max_equal_dose_error"] for row in held_diagnostics)
            and development.get("max_immediate_readout_displacement") == max(row["max_immediate_readout_displacement"] for row in held_diagnostics),
            "development physical summary mismatch")
    return derived, totals, physical, histograms, flips


def root_statistics(derived):
    comparisons, pvalues = {}, {}
    for name, treatment, control in COMPARISONS:
        differences = [row["policy_hits"][treatment] - row["policy_hits"][control] for row in derived]
        n = sum(value != 0 for value in differences)
        positive = sum(value > 0 for value in differences)
        probability = sum(math.comb(n, j) for j in range(positive, n + 1)) / (2 ** n) if n else 1.
        rng = random.Random(seed("bootstrap", name))
        draws = sorted(sum(differences[rng.randrange(N_ROOTS)] for _ in range(N_ROOTS)) / (N_ROOTS * N_CASES)
                       for _ in range(2000))
        comparisons[name] = {
            "mean_root_effect": sum(differences) / (N_ROOTS * N_CASES),
            "root_bootstrap_95_ci": [draws[50], draws[1949]],
            "positive_roots": positive, "negative_roots": sum(value < 0 for value in differences),
            "tied_roots": sum(value == 0 for value in differences),
            "exact_one_sided_root_sign_p": probability,
        }
        pvalues[name] = probability
    previous = 0.
    for index, name in enumerate(sorted(pvalues, key=lambda key: (pvalues[key], key))):
        previous = max(previous, min(1., (len(pvalues) - index) * pvalues[name]))
        comparisons[name]["holm_adjusted_p"] = previous
    return comparisons


def rebuilt_pilot(derived):
    return {
        "schema_version": 1, "experiment_id": NAME,
        "independent_unit": "one 6D rank-three three-pass trained physical and BPD model root",
        "alpha": "0.05", "scientific_min_effect": "0.08", "target_effect": "0.16",
        "measurement_resolution": "1/512", "sham_quantile": "0.95", "sham_confidence": "0.95",
        "planned_comparisons": 4, "confirm_units_planned": 128,
        "pilot_from_development_only": True, "heldout_sealed": True, "units_independent": True,
        "pilot": [{"unit_id": row["root"],
                   "paired_effect": f"{row['policy_hits']['memory_only'] - row['policy_hits']['pooled']}/512",
                   "sham_effects": [f"{row['policy_hits']['pooled'] - row['repeat_pooled_hits']}/512"]}
                  for row in derived],
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + chr(10))


def execute_planner(planner, pilot):
    require(file_sha(planner) == PLANNER_SHA256, "pinned EvalMath executable hash mismatch")
    with tempfile.TemporaryDirectory(prefix="amm09r1-audit-") as directory:
        input_path, output_path = Path(directory) / "pilot.json", Path(directory) / "preflight.json"
        write_json(input_path, pilot)
        execution = subprocess.run([sys.executable, str(Path(planner).resolve()), "--pilot", str(input_path),
                                    "--out", str(output_path)], capture_output=True, text=True, timeout=60, check=False)
        require(execution.returncode == 0, "pinned EvalMath execution failed: " + execution.stderr[-1000:])
        require(output_path.exists(), "pinned EvalMath did not produce a receipt")
        result = read_json(output_path)
    require(result.get("schema") == "EvalMath/mechinterp-power-preflight/v1"
            and result.get("experiment_id") == NAME and result.get("pilot_worlds") == N_ROOTS
            and result.get("planned_confirmation_worlds") == 128 and result.get("planned_comparisons") == 4,
            "pinned EvalMath returned the wrong study identity")
    require(result.get("decision") in ("READY_FOR_CONFIRMATION", "REDESIGN_OR_PILOT_ONLY"),
            "unknown pinned EvalMath decision")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot-out", type=Path, required=True)
    parser.add_argument("--preflight-out", type=Path, required=True)
    parser.add_argument("--check-lock", type=Path)
    args = parser.parse_args(argv)
    outputs = (args.out, args.pilot_out, args.preflight_out)
    require(len({path.resolve() for path in outputs}) == 3, "audit output paths must be distinct")
    require(all(not path.exists() for path in outputs), "audit outputs must be new paths")
    require(file_sha(args.planner) == PLANNER_SHA256, "pinned EvalMath executable hash mismatch")
    sources = source_hashes()
    runtime = read_json(ROOT / "RUNTIME.json")
    acceptance_path = ROOT / "results" / "ENGINEERING_ACCEPTANCE.json"
    acceptance_sha = file_sha(acceptance_path)
    require(acceptance_sha == runtime["engineering_acceptance_sha256"], "frozen engineering acceptance mismatch")
    acceptance = read_json(acceptance_path)
    require(acceptance.get("schema") == "bpd.amm09r.comparison.v1"
            and acceptance.get("registration_commit") == ENGINEERING_REGISTRATION
            and acceptance.get("registered_gate_passed") is True
            and acceptance.get("descriptive_only") is False
            and acceptance.get("confirmation_authorized") is False
            and acceptance.get("scientific_evidence") is False
            and acceptance.get("receipt_errors") == []
            and isinstance(acceptance.get("gates"), dict)
            and ENGINEERING_GATES.issubset(acceptance["gates"])
            and all(value is True for value in acceptance["gates"].values()),
            "engineering acceptance is not a complete passing receipt")
    training, development = read_json(args.training), read_json(args.development)
    training_lock, training_physical = inspect_training(training, sources, runtime, acceptance_sha)
    require(training_lock == read_json(ROOT / "results" / "TRAINING_LOCK.json"), "committed training lock mismatch")
    derived, totals, development_physical, histograms, flips = inspect_development(development, training, sources)
    comparisons = root_statistics(derived)
    source_analysis = {"schema": SCHEMA + "/analysis", "status": "development_only", "comparisons": comparisons,
                       "confirmation_authorized": False,
                       "comparison_arms": {name: {"treatment": treatment, "control": control}
                                           for name, treatment, control in COMPARISONS}}
    require(development.get("analysis") == source_analysis, "independent root statistics disagree with source analysis")
    pilot = rebuilt_pilot(derived)
    require(development.get("preflight_input") == pilot, "recorded pilot differs from reconstructed root effects/shams")
    preflight = execute_planner(args.planner, pilot)
    primary = comparisons["memory_only_vs_pooled"]
    require(preflight.get("mean_pilot_effect") == primary["mean_root_effect"], "planner primary mean mismatch")
    sham_max = max(abs(row["policy_hits"]["pooled"] - row["repeat_pooled_hits"]) / N_CASES for row in derived)
    require(preflight.get("upper_sham_quantile") == sham_max and preflight.get("order_statistic_rank") == N_ROOTS,
            "planner 64-root 95/95 sham envelope mismatch")
    floor = max(.08, 1 / N_CASES, sham_max)
    require(preflight.get("mechanistic_variation_floor") == floor, "planner physical variation floor mismatch")
    all_physical = training_physical + development_physical
    max_dose = max(row["max_equal_dose_error"] for row in all_physical)
    max_visible = max(row["max_immediate_readout_displacement"] for row in all_physical)
    full_donor_dose = max(row["full_donor_equal_dose_error"] for row in all_physical)
    faithful = sum(row["fidelity"] for row in derived)
    oracle_hits = sum(row["oracle_hits"] for row in derived)
    headroom = (oracle_hits - totals["pooled"]) / (N_ROOTS * N_CASES)
    gates = {
        "primary_lower_bound_above_floor": primary["root_bootstrap_95_ci"][0] > floor,
        "primary_holm_significant": primary["holm_adjusted_p"] < .05,
        "source_fidelity": faithful >= 56, "allowed_oracle_headroom": headroom >= .08,
        "all_assays_equal_dose": max_dose <= 1e-5, "all_assays_current_readout_unchanged": max_visible == 0,
        "full_donor_equal_dose": full_donor_dose <= 1e-5,
        "evalmath_ready": preflight["decision"] == "READY_FOR_CONFIRMATION" and not preflight.get("reasons"),
    }
    result = {
        "schema": SCHEMA + "/independent_development_audit", "experiment": NAME,
        "evidence": "prospective development on synthetic trained linear recurrent models",
        "n_independent_training_roots": N_ROOTS, "n_independent_development_roots": N_ROOTS,
        "n_heldout_cases_per_root": N_CASES, "masks": [list(mask) for mask in MASKS],
        "source_sha256": sources, "training_file_sha256": file_sha(args.training),
        "development_file_sha256": file_sha(args.development),
        "training_lock_file_sha256": file_sha(ROOT / "results" / "TRAINING_LOCK.json"),
        "training_policy_sha256": training["sha256"], "training_rows_sha256": training["training_rows_sha256"],
        "development_rows_sha256": digest(development["rows"]),
        "engineering_acceptance_sha256": acceptance_sha, "runtime_signature": training["runtime_signature"],
        "planner": {"repository_commit": PLANNER_COMMIT, "git_blob": PLANNER_GIT_BLOB,
                    "source_sha256": PLANNER_SHA256, "reexecuted_from_reconstructed_rows": True,
                    "pilot_sha256": digest(pilot), "preflight_sha256": digest(preflight)},
        "comparison_arms": source_analysis["comparison_arms"], "comparisons": comparisons,
        "totals": totals, "roots": derived,
        "policy_action_histograms": {arm: {f"mask_{i}": counter[i] for i in range(6)}
                                     for arm, counter in histograms.items()},
        "noisy_label_flips": flips, "noisy_label_budget": N_ROOTS * N_CAL * 6,
        "noise_stream_replayed_by_this_auditor": False,
        "training_ridge_refit_performed_by_this_auditor": False,
        "high_fidelity_roots": faithful, "six_mask_oracle_hits": oracle_hits,
        "allowed_oracle_headroom_rate": headroom, "matched_sham_95_95_bound": sham_max,
        "mechanistic_variation_floor": floor, "max_dose_error_all_assays": max_dose,
        "max_immediate_readout_displacement_all_assays": max_visible,
        "max_full_donor_dose_error_all_assays": full_donor_dose,
        "evalmath": preflight, "development_gates": gates,
        "development_admitted": all(gates.values()),
        "confirmation_authorized": False,
        "confirmation_status": "all 128 AMM09R1 confirmation roots remain sealed; no confirmation runner exists",
    }
    if args.check_lock:
        require(result == read_json(args.check_lock), "independent audit differs from its locked receipt")
    write_json(args.pilot_out, pilot)
    write_json(args.preflight_out, preflight)
    write_json(args.out, result)
    print(json.dumps({"experiment": NAME, "development_admitted": result["development_admitted"],
                      "primary": primary, "evalmath_decision": preflight["decision"],
                      "confirmation_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
