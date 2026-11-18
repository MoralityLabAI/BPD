"""FC05 independent calibration with archived source and task evidence.

The FC04 arithmetic and source selection order are preserved. FC05 adds
audits after selection, with all original model roots retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import (
    PASSES, WIDTH, sample_x, external_labels, parity,
)
from mechinterp.bpd.experiments.nc04.experiment import (
    DenseLooper, factor_basis, self_score_discovery, context_accuracies,
)
from mechinterp.bpd.experiments.nc05.experiment import clean_dose_rms
from mechinterp.bpd.experiments.nc08.experiment import passwise_source_candidates
from mechinterp.bpd.experiments.nc09_fixed12 import experiment as nc09
from mechinterp.bpd.experiments.nc09_fixed12.grassmann_fc03.solver import projector_digest
from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_fc04.solver import complementary_basis
from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_calibration_fc05.solver import fit_full_tangent_shams
from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_calibration_fc05.audit import (
    save_npz, source_audit, task_audit,
)

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[4]
PROTO_SHA = hashlib.sha256((ROOT / "PROTOCOL.md").read_bytes()).hexdigest()
PROTOCOL_COMMIT = "8eb34f3533591513b46f14b902721d7f78bc0d43"
PARENT_PROTO_SHA = "dc9423f1fd9e308221c41ed2af83e7398a59dd0eb0beefa8394091e373156e02"
PARENT_SOURCE_SHA = "8d1b88eea8518a83669f03f9126136ca3d56ac5bac5fe041e2e03fb12922d380"
ROOTS = tuple(range(16000, 16059))
SMOKE = 15999
STUDY = "BPD-NC09-FC05"
PHASE = "FULL_TANGENT_CALIBRATION_DEVELOPMENT"
ARMS = ("dp12", "flat3", "flat3_dose", "flat4", "reverse")
MANDATORY = ("dp12", "flat3", "flat3_dose", "flat4")
FULL_BUDGET = {
    "steps": 384, "discovery_n": 1024, "selector_n": 512,
    "heldout_n": 1024, "shams": 8, "solver_starts": 32, "solver_steps": 64,
}


def validate_budget(seed: int, **budget):
    if seed not in ROOTS and seed != SMOKE:
        raise ValueError("Only excluded root15999 and registered roots16000..16058 are allowed")
    if set(budget) != set(FULL_BUDGET):
        raise ValueError("Complete FC05 budget specification required")
    if any(not isinstance(v, int) or isinstance(v, bool) for v in budget.values()):
        raise ValueError("FC05 budgets must be integers")
    if any(v < 1 for k, v in budget.items() if k != "solver_steps") or budget["solver_steps"] < 0:
        raise ValueError("Invalid FC05 budget")
    if seed in ROOTS and budget != FULL_BUDGET:
        raise ValueError("Calibration roots require the complete registered budget")


def protect_source_manifest():
    """Verify committed dependencies before creating any model or source input."""
    path = ROOT / "SOURCE_MANIFEST.json"
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("protocol_commit_sha") != PROTOCOL_COMMIT:
        raise AssertionError("FC05 protocol commit differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise AssertionError("FC05 source manifest is empty")
    required = {
        str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        str((ROOT / "PROTOCOL.md").relative_to(REPO_ROOT)),
        str((ROOT / "solver.py").relative_to(REPO_ROOT)),
        str((ROOT / "audit.py").relative_to(REPO_ROOT)),
        "mechinterp/bpd/experiments/nc09_fixed12/experiment.py",
        "mechinterp/bpd/experiments/nc09_fixed12/full_tangent_fc04/solver.py",
    }
    if not required <= set(files):
        raise AssertionError("FC05 source manifest omits required source")
    for relative, expected in files.items():
        local = (REPO_ROOT / relative).resolve()
        if not local.is_relative_to(REPO_ROOT) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise AssertionError("Malformed source manifest path or SHA256")
        if hashlib.sha256(local.read_bytes()).hexdigest() != expected:
            raise AssertionError("FC05 dependency hash mismatch: " + relative)
    parent = Path(nc09.__file__).resolve()
    if hashlib.sha256(parent.read_bytes()).hexdigest() != PARENT_SOURCE_SHA:
        raise AssertionError("Original NC09 experiment changed")
    if hashlib.sha256(parent.with_name("PROTOCOL.md").read_bytes()).hexdigest() != PARENT_PROTO_SHA:
        raise AssertionError("Original NC09 protocol changed")
    return {
        "passed": True,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "source_manifest_file": path.name,
        "verified_files": files,
        "nc09_original_experiment_sha256": PARENT_SOURCE_SHA,
        "nc09_original_protocol_sha256": PARENT_PROTO_SHA,
    }


def require_calibration_release(seed: int):
    """A full smoke never opens a calibration root through this entry point."""
    if seed == SMOKE:
        return
    if seed not in ROOTS:
        raise ValueError("Unregistered model root")
    lock = json.loads((ROOT / "EXECUTE_LOCK.json").read_bytes())
    if (lock.get("study") != STUDY
            or lock.get("phase") != "FIRST_ORIGINAL_59_ROOT_CALIBRATION"
            or lock.get("model_roots") != list(ROOTS)
            or lock.get("protocol_commit_sha") != PROTOCOL_COMMIT):
        raise AssertionError("FC05 execution lock does not release this exact cohort")
    actual = hashlib.sha256((ROOT / "SOURCE_MANIFEST.json").read_bytes()).hexdigest()
    if lock.get("source_manifest_sha256") != actual:
        raise AssertionError("FC05 execution lock source manifest mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", lock.get("source_commit_sha", "")):
        raise AssertionError("FC05 execution lock requires its full source commit")


def train_model(seed: int, *, steps: int = 384):
    if seed not in ROOTS and seed != SMOKE:
        raise ValueError("Only excluded15999 and registered16000..16058 are allowed")
    if not isinstance(steps, int) or steps < 1 or (seed in ROOTS and steps != 384):
        raise ValueError("Invalid or reduced calibration training budget")
    protect_source_manifest()
    require_calibration_release(seed)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = DenseLooper()
    source = sample_x(seed, 1, 2048)
    signs = torch.where(external_labels(source), 1., -1.)
    gen = torch.Generator().manual_seed(seed + 1_737_701)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    for _ in range(steps):
        ix = torch.randint(len(source), (128,), generator=gen)
        logit = model(source[ix])[0][:, -1]
        loss = F.softplus(-2 * signs[ix] * logit).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        out = model(source)[0][:, -1]
        acc = float(((out >= 0) == external_labels(source)).float().mean())
        density = float((model.mix.weight.detach().abs() >= 1e-4).float().mean())
    return model, {
        "root": seed, "steps": steps, "train_size": len(source),
        "train_accuracy": acc, "mix_density": density,
        "last_batch_loss": float(loss.detach()),
        "batch_size": 128, "learning_rate": .005,
    }


def _array(tensor: torch.Tensor, *, double: bool = False):
    value = tensor.detach().cpu()
    if double:
        value = value.double()
    return value.numpy().copy()


@torch.no_grad()
def evaluate_task_with_logits(bpd, model, x, labels, clean, clean_writes, arm):
    """Use the original rollout and metric functions; retain their final logits."""
    kept, writes = nc09.closed_loop_rollout(bpd, model, x, arm["projector"], arm["gain"])
    metrics = nc09.score_outputs(clean, clean_writes, kept, writes, labels)
    return metrics, _array(kept[:, -1])


def _gauge_passed(item):
    return bool(item and item.get("physical_map_relative_error", float("inf")) <= 1e-4
                and item.get("max_complete_rollout_logit_error", float("inf")) <= 5e-5)


def task_inputs_after_source_audit(record, seed, count):
    if (not record.get("source_audit_before_task_label_access")
            or not isinstance(record.get("source_archive"), dict)
            or not isinstance(record.get("independent_source_audit"), dict)):
        raise AssertionError("Source archive and audit must precede task inputs and labels")
    inputs = sample_x(seed, 3, count)
    return inputs, external_labels(inputs)


def _effects(result, shams):
    if (len(result["sham_nlls"]) != shams or shams < 2
            or any(result["task_nll_by_arm"].get(k) is None for k in MANDATORY)):
        return None
    nll = {k: result["task_nll_by_arm"][k]["nll_nats"] for k in MANDATORY}
    values = result["sham_nlls"]
    median = float(np.median(values))
    null = [float(v - median) for v in values]
    pseudo = [float((sum(values) - v) / (len(values) - 1) - v) for v in values]
    index = int(np.random.default_rng(18_180_000 + result["root"]).integers(0, 8)) if shams == 8 else None
    e = {
        "A_flat3_minus_dp12": float(nll["flat3"] - nll["dp12"]),
        "D_dose_flat3_minus_dp12": float(nll["flat3_dose"] - nll["dp12"]),
        "S_median_sham_minus_dp12": float(median - nll["dp12"]),
        "S_mean_sham_minus_dp12": float(np.mean(values) - nll["dp12"]),
        "C_dp12_minus_clean": float(nll["dp12"] - result["clean_nll"]),
        "F_dp12_minus_flat4": float(nll["dp12"] - nll["flat4"]),
        "null_sham_signed_effects": null, "T_null_worst_abs": float(max(abs(x) for x in null)),
        "Z_pseudo_null_by_sham": pseudo, "pseudo_selected_sham_index": index,
    }
    if index is not None:
        e["Z_preselected_pseudo_sham"] = pseudo[index]
        if abs(sum(pseudo)) >= 1e-10:
            raise AssertionError("Pseudo-null conditional mean changed")
    return e


def evaluate_calibration_root(seed: int, *, out_dir: Path,
                              steps: int = 384, discovery_n: int = 1024,
                              selector_n: int = 512, heldout_n: int = 1024,
                              shams: int = 8, solver_starts: int = 32,
                              solver_steps: int = 64):
    budget = dict(steps=steps, discovery_n=discovery_n, selector_n=selector_n,
                  heldout_n=heldout_n, shams=shams,
                  solver_starts=solver_starts, solver_steps=solver_steps)
    validate_budget(seed, **budget)
    source_integrity = protect_source_manifest()
    require_calibration_release(seed)
    out_dir = Path(out_dir)
    stem = "nc09_fc05_root_%d" % seed
    source_path = out_dir / (stem + "_source.npz")
    task_path = out_dir / (stem + "_task.npz")
    if any(path.exists() for path in (source_path, task_path, out_dir / (stem + ".json"))):
        raise FileExistsError("Original root evidence already exists; retraining is blocked")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model, training = train_model(seed, steps=steps)
    model_state = {"model__" + key: _array(value) for key, value in model.state_dict().items()}
    xd = sample_x(seed, 2, discovery_n)
    xs = sample_x(seed, 4, selector_n)
    bpd = BPDModel(model, {"mix": WIDTH}, max_passes=PASSES,
                   ablation=AblationConfig(mode="zero", sphere_rescale=False))
    factor_basis(bpd, "native", seed)
    closure = parity(bpd, model, xs[:min(64, len(xs))])
    z, g, _hits, modelscore = self_score_discovery(bpd, model, xd)
    with bpd.target_capture():
        model(xd, capture=True)
        acts = torch.stack(model.acts, 1).detach()
    candidate = passwise_source_candidates(z, g)
    optimum = nc09.optimal_fixed12(candidate)
    arms = nc09.fixed_arm_projectors(candidate, optimum)
    U = bpd.wrappers["mix"].U.detach()
    dpdose = clean_dose_rms(acts, U, arms["dp12"]["projector"]) if arms["dp12"]["eligible"] else None
    flatdose = clean_dose_rms(acts, U, arms["flat3"]["projector"]) if arms["flat3"]["eligible"] else None
    gain = dpdose / flatdose if dpdose is not None and flatdose is not None and flatdose > 1e-12 else None
    source_gain_ok = bool(gain is not None and .5 <= gain <= 1.5)
    for name, arm in arms.items():
        arm["gain"] = float(gain) if name == "flat3_dose" and gain is not None else 1.
        arm["source_write_rms"] = (clean_dose_rms(acts, U, arm["projector"]) * arm["gain"]
                                   if arm["eligible"] else None)
        arm["model_only_validation"] = nc09.model_only_diagnostic(bpd, model, xs, arm) if arm["eligible"] else None
    if not source_gain_ok:
        arms["flat3_dose"]["eligible"] = False
    selected, sham_admission, source_basis = [], None, []
    if optimum is not None:
        source_basis = nc09.tuple_basis(candidate, optimum["ranks"])
        selected, sham_admission = fit_full_tangent_shams(
            acts, U, z, g, source_basis, seed,
            starts=solver_starts, steps=solver_steps, want=shams,
        )
    result = {
        "study": STUDY, "phase": PHASE, "root": seed,
        "protocol_sha256": PROTO_SHA,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_manifest_sha256": source_integrity["source_manifest_sha256"],
        "source_integrity": source_integrity, "parent_integrity": source_integrity,
        "training": training, "budget": budget,
        "splits": {"train": 2048, "derivative_discovery": len(xd),
                   "model_validation": len(xs), "labeled_calibration": heldout_n},
        "full_mask_closure": closure,
        "modelscore_nonzero_count": modelscore["model_score_nonzero_count"],
        "source_rank12_optimum": optimum,
        "passwise_source_candidates": {
            str(t): {str(rank): {k: v for k, v in item.items()
                                 if k not in ("basis", "projector")}
                     for rank, item in by_rank.items()}
            for t, by_rank in candidate.items()
        },
        "source_dp12_projector_receipt": projector_digest(arms["dp12"]["projector"]) if arms["dp12"]["eligible"] else None,
        "physical_source_gain": gain, "dp12_source_write_rms": dpdose,
        "flat3_source_write_rms": flatdose, "source_gain_eligible": source_gain_ok,
        "model_only_diagnostics": {k: v["model_only_validation"] for k, v in arms.items()},
        "source_arms": {k: {field: value for field, value in v.items() if field != "projector"}
                        for k, v in arms.items()},
        "sham_admission": sham_admission,
        "selected_sham_score_source_only": True,
        "native_rotated_gauge": {}, "native_rotated_sham_gauge": [],
        "task_nll_by_arm": {}, "sham_task_metrics": [], "sham_nlls": [],
        "evaluation_errors": [], "eligible": False,
        "original_fc01_hold_unchanged": "CALIBRATION_REDESIGN_OR_PILOT_ONLY",
        "original_fc02_hold_unchanged": "CALIBRATION_INCOMPLETE",
        "original_fc03_hold_unchanged": "ENGINEERING_NULL_OR_INELIGIBLE",
        "scientific_confirmation_roots_evaluated": 0,
    }
    source_arrays = {
        "a": _array(acts, double=True), "z": _array(z, double=True),
        "g": _array(g, double=True), "U": _array(U, double=True),
        "native_factor_V": _array(bpd.wrappers["mix"].V, double=True),
        "discovery_inputs": _array(xd),
        "sham_projectors": np.stack([_array(p, double=True) for p in selected])
            if selected else np.empty((0, PASSES, WIDTH, WIDTH), dtype=np.float64),
        **model_state,
    }
    if optimum is not None:
        source_arrays["dp12_projector"] = _array(arms["dp12"]["projector"], double=True)
        for t, E in enumerate(source_basis):
            source_arrays["E_%d" % t] = _array(E, double=True)
            source_arrays["C_%d" % t] = _array(complementary_basis(E), double=True)
    for name, arm in arms.items():
        if arm["eligible"]:
            source_arrays["arm_projector__" + name] = _array(arm["projector"], double=True)
    result["source_archive"] = save_npz(source_path, source_arrays)
    result["independent_source_audit"] = source_audit(result, source_path)
    result["source_audit_before_task_label_access"] = True
    for name, arm in arms.items():
        if arm["eligible"]:
            try:
                result["native_rotated_gauge"][name] = nc09.gauge_audit(
                    bpd, model, xs, seed, arm["projector"], arm["gain"])
            except (RuntimeError, ValueError, FloatingPointError) as error:
                result["native_rotated_gauge"][name] = None
                result["evaluation_errors"].append({"stage": "fixed_arm_gauge", "arm": name,
                                                     "type": type(error).__name__, "message": str(error)})
    for index, p in enumerate(selected):
        try:
            gauge = nc09.gauge_audit(bpd, model, xs, seed, p, 1.)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            gauge = None
            result["evaluation_errors"].append({"stage": "sham_gauge", "sham_index": index,
                                                 "type": type(error).__name__, "message": str(error)})
        result["native_rotated_sham_gauge"].append(gauge)
    factor_basis(bpd, "native", seed)
    # This is the first split-3 input or task-label access. The saved source
    # archive and independent NumPy audit are complete before reaching it.
    x_eval, labels = task_inputs_after_source_audit(result, seed, heldout_n)
    context = context_accuracies(model, x_eval, labels)
    with torch.no_grad(), bpd.target_capture():
        clean, _ = model(x_eval, capture=True)
        clean_writes = torch.stack(model.writes, 1)
    true_sign = torch.where(labels, 1., -1.)
    result["clean_nll"] = float(F.softplus(-2 * true_sign * clean[:, -1]).mean())
    result["clean_task_context"] = context
    result["clean_task_metrics"] = nc09.score_outputs(clean, clean_writes, clean, clean_writes, labels)
    task_arrays = {"inputs": _array(x_eval), "labels": _array(labels), "clean_logits": _array(clean[:, -1])}
    for name, arm in arms.items():
        score = None
        if arm["eligible"]:
            try:
                score, logits = evaluate_task_with_logits(bpd, model, x_eval, labels, clean, clean_writes, arm)
                task_arrays["arm__" + name] = logits
            except (RuntimeError, ValueError, FloatingPointError) as error:
                result["evaluation_errors"].append({"stage": "fixed_arm_task", "arm": name,
                                                     "type": type(error).__name__, "message": str(error)})
        result["task_nll_by_arm"][name] = score
    for index, p in enumerate(selected):
        score = None
        try:
            score, logits = evaluate_task_with_logits(
                bpd, model, x_eval, labels, clean, clean_writes, {"projector": p, "gain": 1.})
            task_arrays["sham__%d" % index] = logits
        except (RuntimeError, ValueError, FloatingPointError) as error:
            result["evaluation_errors"].append({"stage": "sham_task", "sham_index": index,
                                                 "type": type(error).__name__, "message": str(error)})
        result["sham_task_metrics"].append(score)
        result["sham_nlls"].append(float(score["nll_nats"]) if score is not None else None)
    result["task_archive"] = save_npz(task_path, task_arrays)
    result["independent_task_audit"] = task_audit(result, task_path)
    result["effects"] = (_effects(result, shams)
                          if all(v is not None for v in result["sham_nlls"]) else None)
    result["eligible"] = bool(
        min(context.values()) >= .85 and training["mix_density"] >= .9
        and closure["passed"] and optimum is not None and source_gain_ok
        and all(arms[k]["eligible"] for k in MANDATORY)
        and sham_admission is not None and sham_admission["selected"] == shams
        and all(x["eligible"] for x in sham_admission["projectors"])
        and len(result["sham_nlls"]) == shams
        and all(v is not None for v in result["sham_nlls"])
        and all(result["task_nll_by_arm"][k] is not None for k in MANDATORY)
        and all(_gauge_passed(result["native_rotated_gauge"].get(k)) for k in MANDATORY)
        and len(result["native_rotated_sham_gauge"]) == shams
        and all(_gauge_passed(x) for x in result["native_rotated_sham_gauge"])
        and result["independent_source_audit"]["passed"]
        and result["independent_source_audit"]["complete"]
        and result["independent_task_audit"]["passed"]
        and result["independent_task_audit"]["complete"]
        and not result["evaluation_errors"]
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=int, default=SMOKE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Original receipt already exists")
    record = evaluate_calibration_root(args.root, out_dir=args.out.parent)
    with args.out.open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"root": args.root, "eligible": record["eligible"],
                      "selected_shams": len(record["sham_nlls"]),
                      "source_audit": record["independent_source_audit"]["passed"],
                      "task_audit": record["independent_task_audit"]["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
