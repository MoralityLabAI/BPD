"""Run only registered excluded engineering or exposed historical TRAIN replay.

Invoke this file directly so CPU environment settings precede importing torch.
No development/confirmation CLI or release-manifest interface exists here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ORIGINAL = HERE.parent / "amm09_two_slot_memory"
SCHEMA = "bpd.amm09r.engineering.v1"
NAMESPACE = "AMM09R-ENGINEERING-V1/"
PROGRESS = {"stage": "startup", "root": None, "rows": []}
ORIGINAL_HASHES = {
    "models.py": "2fec8382c067147d60d9f2d4fe8d171ce3d5e15f405afffbbab5389023bc1a03",
    "case.py": "f72788b058d7eff58218ed4e0c729437820a8cfdd2518609409f2fc7fec9132a",
    "experiment.py": "57af128ca48ef2dcd527abfcd443373705ecd046a460d53bafe2a0222df56897",
}
PINNED_ENV = {
    "ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "COMPATIBLE",
    "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "MKL_DYNAMIC": "FALSE",
    "OMP_DYNAMIC": "FALSE", "PYTHONHASHSEED": "0",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor(value):
    array = value.detach().cpu().contiguous().numpy()
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return {"dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "values": array.tolist()}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_original(engineering):
    for name, expected in ORIGINAL_HASHES.items():
        if file_sha(ORIGINAL / name) != expected:
            raise ValueError("frozen original source mismatch: " + name)
    package = types.ModuleType("_amm09r_frozen")
    package.__path__ = [str(ORIGINAL)]
    sys.modules[package.__name__] = package
    m = load_module(package.__name__ + ".models", ORIGINAL / "models.py")
    if engineering:
        def engineering_seed(*parts):
            value = NAMESPACE + "/".join(map(str, parts))
            return int(hashlib.sha256(value.encode()).hexdigest()[:15], 16)
        m.seed = engineering_seed
        m.PHASE_N = {"engineering": 8}
    case = load_module(package.__name__ + ".case", ORIGINAL / "case.py")
    experiment = load_module(package.__name__ + ".experiment", ORIGINAL / "experiment.py")
    return m, case, experiment


def source_hashes():
    paths = set(ORIGINAL / name for name in ORIGINAL_HASHES)
    paths.update(HERE.glob("*.py"))
    paths.add(HERE / "ENGINEERING_PROTOCOL.md")
    paths.update(REPO / "mechinterp" / "bpd" / name for name in ("components.py", "config.py"))
    return {str(p.relative_to(REPO)): file_sha(p) for p in sorted(paths)}


def provenance(torch, args):
    import numpy
    cpu = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    fields = {}
    for line in cpu.splitlines():
        if ":" in line:
            key, value = (x.strip() for x in line.split(":", 1))
            fields.setdefault(key, value)
    torch_dir = Path(torch.__file__).parent
    binaries = {"libtorch_cpu.so": file_sha(torch_dir / "lib" / "libtorch_cpu.so"),
                "_C.so": file_sha(torch._C.__file__)}
    records = {}
    for package in ("torch", "numpy"):
        dist = importlib.metadata.distribution(package)
        records[package] = hashlib.sha256((dist.read_text("RECORD") or "").encode()).hexdigest()
    versions = platform.python_version(), torch.__version__, numpy.__version__
    checkout = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    return {
        "checkout_commit": checkout,
        "worker": args.worker, "process_ordinal": args.process_ordinal,
        "host_id": socket.gethostname(), "process_id": os.getpid(),
        "python": versions[0], "torch": versions[1], "numpy": versions[2],
        "torch_git": torch.version.git_version,
        "registered_software": versions == ("3.12.12", "2.10.0+cpu", "2.2.6"),
        "python_executable_sha256": file_sha(sys.executable),
        "torch_binary_sha256": binaries, "distribution_record_sha256": records,
        "cpu_model": fields.get("model name", platform.processor()),
        "cpu_vendor": fields.get("vendor_id"), "cpu_flags": fields.get("flags"),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "platform": platform.platform(), "machine": platform.machine(),
        "torch_config": torch.__config__.show(),
        "parallel_info": torch.__config__.parallel_info(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment": {key: os.environ.get(key) for key in PINNED_ENV},
        "runner": {key: os.environ.get(key) for key in (
            "ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH", "RUNNER_NAME",
            "GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB")},
        "container_digest": None, "execution_environment": "native-hosted-worker",
        "packages": sorted(f"{d.metadata['Name']}=={d.version}"
                           for d in importlib.metadata.distributions() if d.metadata.get("Name")),
    }


def input_hashes(m, index, torch):
    out = {}
    for stream, steps in (("teacher_train", m.TEACHER_STEPS), ("factor_train", m.FACTOR_STEPS)):
        g = m.gen(m.root_key("engineering", index), stream)
        h = hashlib.sha256()
        for _ in range(steps):
            batch = torch.rand((96, m.R), generator=g) * 1.6 - .8
            h.update(batch.numpy().astype("<f4", copy=False).tobytes())
        out[stream] = h.hexdigest()
    return out


def assay(m, case, factor, index, split, n, torch):
    rec, donor, target = m.donor_pairs("engineering", index, n, split)
    raw = case.candidate_features(factor, rec, donor, m.family_for(index))
    scored = case.score("engineering", index, factor, split, n)
    if raw["features"].tolist() != scored.pop("features"):
        raise AssertionError("assay instrumentation changed feature values")
    margin = m.TOL - (raw["probes"] - target[:, None, :]).abs().amax(dim=-1)
    success = ((raw["probes"] - target[:, None, :]).abs() <= m.TOL).all(dim=2)
    for a in range(6):
        bits = "".join(str(int(v)) for v in success[:, a])
        if bits != scored["candidate_bits"][f"mask_{a}"]:
            raise AssertionError("instrumented label differs from frozen source")
    scored.update({
        "recipient": tensor(rec), "donor": tensor(donor), "target_audit_only": tensor(target),
        "features": tensor(raw["features"]), "probes": tensor(raw["probes"]),
        "patches": tensor(torch.stack(raw["patches"], dim=1)), "margins": tensor(margin),
        "dose": tensor(raw["dose"]),
    })
    return scored


def engineering_run(m, case, e, torch):
    from unittest.mock import patch
    physical = load_module("_amm09r_physical", HERE / "physical_audit.py")
    rows, xs, ys = [], [], []
    PROGRESS["rows"] = rows
    total = [0] * 6
    for index in range(8):
        PROGRESS.update(stage="initialization", root=m.root_key("engineering", index))
        initial_teacher = m.Physical("engineering", index)
        initial_factor = m.Factors("engineering", index)
        initial = {name: tensor(value) for name, value in (
            ("I", initial_teacher.I), ("O", initial_teacher.O),
            ("V", initial_factor.V), ("U", initial_factor.U))}
        inputs = input_hashes(m, index, torch)
        original_factor_class = m.Factors
        captured = {}

        class TrackedFactors(original_factor_class):
            def __init__(self, phase, root_index):
                super().__init__(phase, root_index)
                captured["instance"] = self

        original_svd = torch.linalg.svd

        def trace_svd(value, *args, **kwargs):
            f = captured["instance"]
            captured["before"] = {"V": tensor(f.V), "U": tensor(f.U), "full_product": tensor(value)}
            result = original_svd(value, *args, **kwargs)
            captured["svd"] = {name: tensor(t) for name, t in zip(("left", "singular", "right"), result)}
            return result

        with patch.object(m, "Factors", TrackedFactors), patch.object(torch.linalg, "svd", trace_svd):
            PROGRESS["stage"] = "model_fit"
            teacher, factor, metrics = m.fit("engineering", index)
        fitted_teacher = {"I": tensor(teacher.I), "O": tensor(teacher.O), "weight": tensor(teacher.weight())}
        fitted_factor = {"V": tensor(factor.V), "U": tensor(factor.U), "weight": tensor(factor.weight()),
                         "rankone": [tensor(factor.V[:, j:j+1] @ factor.U[j:j+1, :]) for j in range(3)]}
        PROGRESS["stage"] = "assays_and_physical_audit"
        train = assay(m, case, factor, index, "training_pairs", 128, torch)
        held = assay(m, case, factor, index, "engineering", 512, torch)
        x = torch.tensor(train["features"]["values"], dtype=torch.float64).reshape(-1, 18)
        y = torch.tensor([[int(train["candidate_bits"][f"mask_{a}"][i]) for a in range(6)]
                          for i in range(128)], dtype=torch.float64).reshape(-1)
        xs.append(x); ys.append(y)
        for a in range(6):
            total[a] += train["candidate_hits"][f"mask_{a}"]
        rows.append({"root": m.root_key("engineering", index), "initial": initial,
                     "train_input_sha256": inputs, "teacher": fitted_teacher,
                     "factor_before_svd": captured["before"], "svd": captured["svd"],
                     "factor": fitted_factor, "fit": metrics,
                     "assays": {"training_pairs": train, "engineering": held},
                     "physical": physical.audit(m, case, teacher, factor, "engineering", index)})
        print(f"engineering root {index + 1}/8 complete", flush=True)
    PROGRESS["stage"] = "engineering_ridge_fixture"
    raw = e.ridge(torch.cat(xs), torch.cat(ys), 10., False)
    rounded = [round(value, 4) for value in raw]
    pooled = min(range(6), key=lambda a: (-total[a], a))
    decisions = {}
    for row in rows:
        features = row["assays"]["engineering"]["features"]["values"]
        values = e.regression_prediction(features, rounded).clamp(0, 1)
        decisions[row["root"]] = {"scores": tensor(values), "actions": values.argmax(dim=1).tolist()}
    return {"phase": "engineering", "namespace": NAMESPACE, "n_roots": 8,
            "rows": rows, "fixture": {"raw_coefficients": raw, "global_coefficients": rounded,
            "pooled_action": pooled, "candidate_hits": total, "decisions": decisions}}


def compare_original(run, path):
    if file_sha(path) != "a55080a1d09f99cbea0d99182cbe4ca38214b90b65ee0a3adc1af2d828858313":
        raise ValueError("original TRAIN archive hash mismatch")
    original = json.loads(path.read_text())
    mismatch = {"teacher_weight_sha256": [], "factor_weight_sha256": []}
    bits = []
    for before, after in zip(original["rows"], run["rows"]):
        if before["root"] != after["root"]:
            raise ValueError("historical root membership mismatch")
        for key in mismatch:
            if before["fit"][key] != after["fit"][key]:
                mismatch[key].append(before["root"])
        counts = {f"mask_{a}": sum(x != y for x, y in zip(
                    before["assay"]["candidate_bits"][f"mask_{a}"],
                    after["assay"]["candidate_bits"][f"mask_{a}"])) for a in range(6)}
        if any(counts.values()):
            bits.append({"root": before["root"], "hamming_by_mask": counts})
    return {"original_file_sha256": file_sha(path), "entire_original_payload_equal": original == run,
            "policy_sha256_equal": original["sha256"] == run["sha256"],
            "original_policy_sha256": original["sha256"], "current_policy_sha256": run["sha256"],
            "weight_mismatch_roots": mismatch,
            "candidate_bit_changes": sum(sum(row["hamming_by_mask"].values()) for row in bits),
            "changed_roots": bits, "original_pooled": original["pooled_action"],
            "current_pooled": run["pooled_action"],
            "max_abs_rounded_coefficient_difference": max(abs(a-b) for a, b in zip(
                original["global_coefficients"], run["global_coefficients"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("engineering", "historical"), required=True)
    parser.add_argument("--configuration", choices=("baseline32", "pinned32"), required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--process-ordinal", type=int, choices=(1, 2), required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-unpinned", action="store_true", help="local engineering diagnostics only")
    args = parser.parse_args()
    if args.mode == "historical" and (args.configuration != "baseline32" or args.original is None):
        parser.error("historical mode requires baseline32 and the verified original training receipt")
    if args.mode == "historical":
        if file_sha(args.original) != "a55080a1d09f99cbea0d99182cbe4ca38214b90b65ee0a3adc1af2d828858313":
            raise ValueError("original TRAIN archive hash mismatch before any model access")
    if args.out.exists():
        raise FileExistsError(args.out)
    if "torch" in sys.modules:
        raise RuntimeError("run directly in a fresh Python process before importing torch")
    # PYTHONHASHSEED affects interpreter startup; re-exec once when needed.
    if args.configuration == "pinned32":
        if any(os.environ.get(k) != v for k, v in PINNED_ENV.items()):
            os.environ.update(PINNED_ENV)
            os.execv(sys.executable, [sys.executable, *sys.argv])
    sys.path.insert(0, str(REPO))
    import torch
    torch.set_num_threads(1)
    if args.configuration == "pinned32":
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("registered float32 default required")
    prov = provenance(torch, args)
    if not prov["registered_software"] and not args.allow_unpinned:
        raise ValueError("registered Python3.12.12/torch2.10.0+cpu/numpy2.2.6 required")
    m, case, e = load_original(args.mode == "engineering")
    original_comparison = None
    failure = None
    try:
        if args.mode == "engineering":
            payload = engineering_run(m, case, e, torch)
        else:
            # Read-only snapshots preserve partial receipts if a later root fails.
            from unittest.mock import patch
            original_fit, original_score = m.fit, case.score

            def tracked_fit(phase, index):
                PROGRESS.update(stage="historical_model_fit", root=m.root_key(phase, index))
                result = original_fit(phase, index)
                PROGRESS["rows"].append({"root": m.root_key(phase, index), "fit": result[2]})
                return result

            def tracked_score(*values, **kwargs):
                PROGRESS["stage"] = "historical_training_assay"
                scored = original_score(*values, **kwargs)
                PROGRESS["rows"][-1]["assay"] = dict(scored)
                return scored

            with patch.object(m, "fit", tracked_fit), patch.object(case, "score", tracked_score):
                payload = e.train()
            original_comparison = compare_original(payload, args.original)
    except Exception as error:
        failure = error
        payload = {"phase": args.mode, "rows": PROGRESS["rows"],
                   "failure": {"type": type(error).__name__, "message": str(error),
                               "root": PROGRESS["root"], "stage": PROGRESS["stage"]}}
    result = {"schema": SCHEMA, "mode": args.mode, "configuration": args.configuration,
              "provenance": prov, "source_sha256": source_hashes(), "payload": payload,
              "payload_sha256": digest(payload), "comparison_to_original": original_comparison,
              "confirmation_accessed": False, "execution_status": "failed" if failure else "completed"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps({"mode": args.mode, "configuration": args.configuration,
                      "payload_sha256": result["payload_sha256"],
                      "original_comparison": original_comparison}, sort_keys=True), flush=True)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
