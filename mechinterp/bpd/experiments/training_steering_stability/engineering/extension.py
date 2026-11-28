"""Continue the exact exposed RP03 checkpoints under EXTENSION_PLAN.md."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from torch import Tensor
import torch.nn.functional as F

from . import learning_curve as lc
from mechinterp.bpd.experiments.training_steering.data import partition_mask, root_encoding
from mechinterp.bpd.experiments.training_steering.model import component_patch

HERE = Path(__file__).resolve().parent
PRIOR = HERE / "results"
CHECKPOINTS = (3000, 6000)
ORDINARY_FLOOR = .80
BINDING_FLOOR = .75


@torch.no_grad()
def discover(model, root: int, checkpoint: int, recipe) -> tuple[Tensor, dict]:
    cohort = lc.batch(root, "extension_attribution_train", checkpoint,
                      recipe.calibration_n, 2, recipe)
    n = len(cohort.query)
    clean = model.rollout(torch.cat((cohort.receiver, cohort.donor)))
    recipient = clean["states"][:n, cohort.prefix_steps - 1]
    donor = clean["states"][n:, cohort.prefix_steps - 1]
    candidates = torch.arange(recipe.components).reshape(-1, recipe.group_size)
    base_ce = F.cross_entropy(clean["logits"][:n], cohort.labels_h, reduction="none")
    losses, dose_errors, invalid = [], [], []
    for group in candidates:
        delta, info = component_patch(model, recipient, donor,
                                      group.expand(recipe.slots, -1), cohort.edit_slot,
                                      dose=recipe.intervention_l2)
        logits = model.rollout(cohort.receiver, patch=delta,
                               patch_step=cohort.prefix_steps)["logits"]
        ce = F.cross_entropy(logits, cohort.labels_h, reduction="none")
        losses.append([float(ce[cohort.edit_slot == slot].mean())
                       for slot in range(recipe.slots)])
        valid_norm = info["delivered_norm"][info["valid"]]
        dose_errors.append(float((valid_norm - recipe.intervention_l2).abs().max())
                           if len(valid_norm) else None)
        invalid.append(int((~info["valid"]).sum()))
    chosen = torch.tensor(losses).argmin(dim=0)
    groups = candidates[chosen].clone()
    return groups, {
        "scope": "Exposed-root extension; TRAIN-only fixed RP02 group scan",
        "seed": lc.seed_for(root, "extension_attribution_train", checkpoint, 2),
        "batch_sha256": cohort.fingerprint(), "calibration_pairs": n,
        "candidate_groups": candidates.tolist(), "candidate_ce_by_slot": losses,
        "selected_candidate": chosen.tolist(), "groups": groups.tolist(),
        "dose_errors_by_candidate": dose_errors, "invalid_by_candidate": invalid,
        "base_counterfactual_ce_by_slot": [
            float(base_ce[cohort.edit_slot == slot].mean()) for slot in range(recipe.slots)],
        "cases_by_slot": [int((cohort.edit_slot == slot).sum()) for slot in range(recipe.slots)],
    }


@torch.no_grad()
def standard_diagnostic(model, root: int, checkpoint: int, delay: int,
                        groups: Tensor, recipe) -> dict:
    cohort = lc.batch(root, "extension_diagnostic_pairs", checkpoint,
                      lc.DIAGNOSTIC_PAIRS, delay, recipe)
    n = len(cohort.query)
    clean = model.rollout(torch.cat((cohort.receiver, cohort.donor)))
    recipient = clean["states"][:n, cohort.prefix_steps - 1]
    donor = clean["states"][n:, cohort.prefix_steps - 1]
    delta, physical = component_patch(model, recipient, donor, groups, cohort.edit_slot,
                                       dose=recipe.intervention_l2)
    patched = model.rollout(cohort.receiver, patch=delta, patch_step=cohort.prefix_steps)
    ordinary = lc.prediction_record(clean["logits"][:n], cohort.labels_r)
    causal = lc.prediction_record(patched["logits"], cohort.labels_h)
    targeted = cohort.query == cohort.edit_slot
    correct = torch.tensor(causal["correct"], dtype=torch.bool)
    causal.update({"targeted": targeted.tolist(), "targeted_n": int(targeted.sum()),
                   "unaffected_n": int((~targeted).sum()),
                   "targeted_accuracy": float(correct[targeted].float().mean()),
                   "unaffected_accuracy": float(correct[~targeted].float().mean())})
    modal = lc.modal_predictions(cohort.values_r, recipe.values)
    write_delta = (patched["recurrent_writes"][:, cohort.prefix_steps]
                   - clean["recurrent_writes"][:n, cohort.prefix_steps])
    return {
        "scope": "TRAIN-only extension diagnostic; ordinary steering disabled",
        "seed": lc.seed_for(root, "extension_diagnostic_pairs", checkpoint, delay),
        "delay": delay, "pairs": n, "batch_sha256": cohort.fingerprint(),
        "ordinary": ordinary,
        "modal_value_baseline": {
            "scope": "Host-only semantic memory; query-blind; smallest-value tie",
            "accuracy": float((modal == cohort.labels_r).float().mean()),
            "correct": (modal == cohort.labels_r).int().tolist(),
            "predictions": modal.tolist()},
        "causal": causal,
        "physical": {
            "invalid_interventions": int((~physical["valid"]).sum()),
            "delivered_l2_min": float(delta.norm(dim=-1).min()),
            "delivered_l2_max": float(delta.norm(dim=-1).max()),
            "native_write_max_error": float((write_delta - delta).abs().max()),
            "prefix_state_max_error": float((patched["states"][:, :cohort.prefix_steps]
                                               - clean["states"][:n, :cohort.prefix_steps]).abs().max()),
            "recipient_state_rms": float(recipient.square().mean().sqrt()),
            "all_head_max_change": float((model.output(patched["states"])
                                            - model.output(clean["states"][:n])).abs().max()),
            "terminal_state_distance_mean": float((patched["states"][:, -1]
                                                      - clean["states"][:n, -1]).norm(dim=-1).mean())},
    }


@torch.no_grad()
def address_binding_diagnostic(model, root: int, checkpoint: int, delay: int,
                               recipe) -> dict:
    chosen_features, chosen_values, chosen_queries, fingerprints = [], [], [], []
    accepted = 0
    for attempt in range(16):
        candidates = lc.batch(root, "extension_address_binding_candidates",
                              checkpoint * 100 + attempt, 2048, delay, recipe)
        fingerprints.append(candidates.fingerprint())
        distinct = (candidates.values_r.sort(dim=1).values
                    == torch.arange(recipe.values)[None, :]).all(dim=1)
        take = torch.where(distinct)[0][:lc.BINDING_BASES - accepted]
        if len(take):
            chosen_features.append(candidates.receiver[take])
            chosen_values.append(candidates.values_r[take])
            chosen_queries.append(candidates.query[take])
            accepted += len(take)
        if accepted == lc.BINDING_BASES:
            break
    if accepted != lc.BINDING_BASES:
        raise RuntimeError("Binding diagnostic exhausted its fixed admission cap")
    features = torch.cat(chosen_features).repeat_interleave(recipe.slots, dim=0)
    memories = torch.cat(chosen_values).repeat_interleave(recipe.slots, dim=0)
    old_queries = torch.cat(chosen_queries).repeat_interleave(recipe.slots)
    queries = torch.arange(recipe.slots).repeat(lc.BINDING_BASES)
    address_encoding, _ = root_encoding(root, recipe.slots, recipe.values)
    row = torch.arange(len(features))
    features[row, -1, 3 + address_encoding[old_queries]] -= 1.0
    features[row, -1, 3 + address_encoding[queries]] += 1.0
    labels = memories.gather(1, queries[:, None]).squeeze(1)
    modal = lc.modal_predictions(memories, recipe.values)
    if int((modal == labels).sum()) != lc.BINDING_BASES:
        raise AssertionError("All-query modal baseline must equal 25 percent")
    if not bool(partition_mask(memories, "train", recipe.values).all()):
        raise AssertionError("Binding diagnostic left the TRAIN partition")
    digest = hashlib.sha256()
    for tensor in (features, memories, queries, labels):
        digest.update(tensor.contiguous().numpy().tobytes())
    return {
        "scope": "TRAIN-only extension address binding; ordinary steering disabled",
        "base_sequences": lc.BINDING_BASES, "cases": len(features), "delay": delay,
        "candidate_batches": len(fingerprints), "candidate_batch_sha256": fingerprints,
        "expanded_sha256": digest.hexdigest(),
        "query_counts": queries.bincount(minlength=recipe.slots).tolist(),
        "label_counts": labels.bincount(minlength=recipe.values).tolist(),
        "canonical_memories": memories[::recipe.slots].tolist(),
        "only_query_address_changes_within_group": True,
        "ordinary": lc.prediction_record(model(features), labels),
        "modal_value_baseline": {"accuracy": float((modal == labels).float().mean()),
                                 "correct": (modal == labels).int().tolist(),
                                 "predictions": modal.tolist()},
    }


def source_hashes() -> dict[str, str]:
    hashes = lc.source_hashes()
    for path in (Path(__file__), HERE / "EXTENSION_PLAN.md", PRIOR / "run_manifest.json",
                 PRIOR / "completion.json"):
        hashes[str(path.relative_to(lc.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for root in lc.ROOTS:
        for name in ("checkpoint_2000.pt", "checkpoint_2000.json"):
            path = PRIOR / f"root_{root}" / name
            hashes[str(path.relative_to(lc.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def run_root(root: int, out: Path, recipe) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    parent_path = PRIOR / f"root_{root}" / "checkpoint_2000.pt"
    parent_record = json.loads((parent_path.with_suffix(".json")).read_text())
    parent_bytes_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    if parent_bytes_hash != parent_record["checkpoint_sha256"]:
        raise AssertionError("Parent checkpoint bytes differ from the saved receipt")
    saved = torch.load(parent_path, map_location="cpu", weights_only=True)
    if saved["root"] != root or saved["updates"] != 2000:
        raise AssertionError("Parent checkpoint root or update count differs")
    model = lc.new_model(recipe)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                                  weight_decay=recipe.weight_decay)
    optimizer.load_state_dict(saved["optimizer"])
    if lc.state_hash(model) != parent_record["model_state_sha256"]:
        raise AssertionError("Loaded model differs from the parent state hash")
    for group in optimizer.param_groups:
        if group["lr"] != recipe.learning_rate or group["weight_decay"] != recipe.weight_decay:
            raise AssertionError("Loaded AdamW settings differ from the fixed recipe")
    start = time.time()
    trace, checkpoints = [], {}
    stream = hashlib.sha256()
    for step in range(2001, 6001):
        model.train()
        delay = recipe.train_delays[(step - 1) % len(recipe.train_delays)]
        cohort = lc.batch(root, "task_training", step - 1, recipe.batch_size, delay, recipe)
        fingerprint = cohort.fingerprint()
        stream.update(fingerprint.encode())
        features, labels = lc.full_task_batch(cohort)
        loss = F.cross_entropy(model(features), labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite task loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_l2 = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
        if not bool(torch.isfinite(gradient_l2)):
            raise ValueError("Non-finite task gradient")
        optimizer.step()
        if step % 50 == 0:
            trace.append({"total_updates": step, "task_ce": float(loss.detach()),
                          "preclip_gradient_l2": float(gradient_l2),
                          "batch_sha256": fingerprint,
                          "extension_training_stream_sha256": stream.hexdigest()})
            lc.dump(out / "training_trace.json", trace)
            if step % 500 == 0:
                print(json.dumps({"root": root, "total_updates": step,
                                  "task_ce": trace[-1]["task_ce"]}), flush=True)
        if step in CHECKPOINTS:
            model.eval()
            path = out / f"checkpoint_{step}.pt"
            torch.save({"root": root, "updates": step, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, path)
            groups, attribution = discover(model, root, step, recipe)
            diagnostics = {str(delay): {
                "standard": standard_diagnostic(model, root, step, delay, groups, recipe),
                "address_binding": address_binding_diagnostic(model, root, step, delay, recipe)}
                for delay in lc.DELAYS}
            record = {"updates": step, "model_state_sha256": lc.state_hash(model),
                      "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "extension_training_stream_sha256": stream.hexdigest(),
                      "attribution": attribution, "diagnostics": diagnostics}
            checkpoints[str(step)] = record
            lc.dump(out / f"checkpoint_{step}.json", record)
            print(json.dumps({"root": root, "checkpoint": step,
                              "ordinary": {d: diagnostics[d]["standard"]["ordinary"]["accuracy"] for d in diagnostics},
                              "binding": {d: diagnostics[d]["address_binding"]["ordinary"]["accuracy"] for d in diagnostics}}), flush=True)
    result = {"root": root, "scope": "EXPOSED_TRAIN_ONLY_ENGINEERING_EXTENSION",
              "recipe": asdict(recipe), "parent_checkpoint_sha256": parent_bytes_hash,
              "parent_state_sha256": parent_record["model_state_sha256"],
              "parent_training_stream_sha256": parent_record["training_stream_sha256"],
              "parent_optimizer_restored": True, "start_update": 2000, "end_update": 6000,
              "extension_training_trajectories": 4000 * recipe.batch_size * 3,
              "training_trace": trace, "checkpoints": checkpoints,
              "elapsed_seconds": time.time() - start}
    lc.dump(out / "result.json", result)
    lc.dump(out / "artifact_hashes.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted(out.iterdir()) if p.is_file()})
    return result


def selection_rule(results: list[dict]) -> dict:
    if {r["root"] for r in results} != set(lc.ROOTS):
        raise ValueError("Selection requires both fixed roots")
    checks = {}
    qualifying = []
    for checkpoint in CHECKPOINTS:
        rows = []
        for result in results:
            for delay in lc.DELAYS:
                data = result["checkpoints"][str(checkpoint)]["diagnostics"][str(delay)]
                ordinary = data["standard"]["ordinary"]["accuracy"]
                binding = data["address_binding"]["ordinary"]["accuracy"]
                rows.append({"root": result["root"], "delay": delay,
                             "ordinary_accuracy": ordinary, "address_binding_accuracy": binding,
                             "pass": ordinary >= ORDINARY_FLOOR and binding >= BINDING_FLOOR})
        passes = all(row["pass"] for row in rows)
        checks[str(checkpoint)] = {"pass": passes, "rows": rows}
        if passes:
            qualifying.append(checkpoint)
    selected = min(qualifying) if qualifying else 6000
    return {"selection_exposure": "Two engineering roots exposed before this extension; thresholds fixed in EXTENSION_PLAN.md",
            "ordinary_floor": ORDINARY_FLOOR, "address_binding_floor": BINDING_FLOOR,
            "candidate_checks": checks, "selected_total_updates": selected,
            "weak_competence": not bool(qualifying),
            "verdict": "ENGINEERING_COMPETENCE_GATE_PASSED" if qualifying else "WEAK_COMPETENCE"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "extension_results")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(HERE):
        parser.error("Output must remain within the engineering directory")
    original_sources = json.loads((PRIOR / "run_manifest.json").read_text())["source_hashes"]
    if original_sources != lc.source_hashes():
        raise RuntimeError("The completed 2000-update source has changed")
    recipe = replace(lc.Recipe(), warmup_steps=6000, training_steps=0)
    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    sources = source_hashes()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    lc.dump(args.out / "run_manifest.json", {
        "scope": "EXPOSED_TRAIN_ONLY_ENGINEERING_EXTENSION", "roots": list(lc.ROOTS),
        "start_update": 2000, "end_update": 6000, "checkpoints": list(CHECKPOINTS),
        "recipe": asdict(recipe), "source_hashes": sources, "started_unix": start,
        "python": platform.python_version(), "torch": torch.__version__})
    results = [run_root(root, args.out / f"root_{root}", recipe) for root in lc.ROOTS]
    if sources != source_hashes():
        raise RuntimeError("Extension source changed during the run")
    lc.dump(args.out / "selection.json", selection_rule(results))
    rows = []
    for result in results:
        for checkpoint, record in result["checkpoints"].items():
            for delay, diagnostic in record["diagnostics"].items():
                rows.append({"root": result["root"], "updates": int(checkpoint), "delay": int(delay),
                             "ordinary_accuracy": diagnostic["standard"]["ordinary"]["accuracy"],
                             "modal_accuracy": diagnostic["standard"]["modal_value_baseline"]["accuracy"],
                             "causal_targeted_accuracy": diagnostic["standard"]["causal"]["targeted_accuracy"],
                             "causal_unaffected_accuracy": diagnostic["standard"]["causal"]["unaffected_accuracy"],
                             "address_binding_accuracy": diagnostic["address_binding"]["ordinary"]["accuracy"],
                             "address_binding_modal_accuracy": diagnostic["address_binding"]["modal_value_baseline"]["accuracy"]})
    lc.dump(args.out / "summary.json", {"scope": "EXPOSED_TRAIN_ONLY_ENGINEERING_EXTENSION", "rows": rows})
    lc.dump(args.out / "completion.json", {
        "completed": True, "roots": list(lc.ROOTS), "elapsed_seconds": time.time() - start,
        "result_sha256": {str(r): hashlib.sha256((args.out / f"root_{r}" / "result.json").read_bytes()).hexdigest()
                          for r in lc.ROOTS}})


if __name__ == "__main__":
    main()
