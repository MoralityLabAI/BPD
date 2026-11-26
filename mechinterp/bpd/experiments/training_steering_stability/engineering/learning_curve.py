"""Fixed RP03 task-only learning diagnostic on the RP02 TRAIN partition."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
from torch import Tensor
import torch.nn.functional as F

from mechinterp.bpd.experiments.training_steering.data import (
    PairBatch, make_pairs, partition_mask, root_encoding,
)
from mechinterp.bpd.experiments.training_steering.model import component_patch
from mechinterp.bpd.experiments.training_steering.experiment import (
    Recipe, full_task_batch, new_model, state_hash,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RP02 = HERE.parents[1] / "training_steering"
ROOTS = (27000, 27001)
CHECKPOINTS = (200, 800, 2000)
DELAYS = (2, 7)
DIAGNOSTIC_PAIRS = 512
BINDING_BASES = 128


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def seed_for(root: int, domain: str, index: int = 0, delay: int = 0) -> int:
    raw = f"BPD-RP-03:engineering:{root}:{domain}:{index}:{delay}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def batch(root: int, domain: str, index: int, n: int, delay: int,
          recipe: Recipe) -> PairBatch:
    result = make_pairs(seed_for(root, domain, index, delay), n, root_seed=root,
                        delay=delay, split="train", slots=recipe.slots,
                        values=recipe.values, noise=recipe.noise)
    for values in (result.values_r, result.values_d, result.values_h):
        if not bool(partition_mask(values, "train", recipe.values).all()):
            raise AssertionError("Engineering generation left the TRAIN partition")
    return result


def modal_predictions(memory: Tensor, values: int = 4) -> Tensor:
    counts = F.one_hot(memory, num_classes=values).sum(dim=1)
    return counts.argmax(dim=1)


def prediction_record(logits: Tensor, labels: Tensor) -> dict:
    predicted = logits.argmax(dim=-1)
    correct = predicted == labels
    return {"accuracy": float(correct.float().mean()),
            "cross_entropy": float(F.cross_entropy(logits, labels)),
            "correct": correct.int().tolist(), "predictions": predicted.tolist(),
            "labels": labels.tolist()}


@torch.no_grad()
def discover(model, root: int, checkpoint: int, recipe: Recipe) -> tuple[Tensor, dict]:
    """The fixed RP02 eight-group scan, using an RP03 engineering seed domain."""
    cohort = batch(root, "attribution_train", checkpoint, recipe.calibration_n, 2, recipe)
    n = len(cohort.query)
    clean = model.rollout(torch.cat((cohort.receiver, cohort.donor)))
    receiver = clean["states"][:n, cohort.prefix_steps - 1]
    donor = clean["states"][n:, cohort.prefix_steps - 1]
    base_ce = F.cross_entropy(clean["logits"][:n], cohort.labels_h, reduction="none")
    candidates = torch.arange(recipe.components).reshape(-1, recipe.group_size)
    losses, dose_errors, invalid = [], [], []
    for group in candidates:
        groups = group.expand(recipe.slots, -1)
        delta, info = component_patch(model, receiver, donor, groups, cohort.edit_slot,
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
        "scope": "TRAIN-only engineering attribution; fixed RP02 candidate family",
        "seed": seed_for(root, "attribution_train", checkpoint, 2),
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
                        groups: Tensor, recipe: Recipe) -> dict:
    cohort = batch(root, "diagnostic_pairs", checkpoint, DIAGNOSTIC_PAIRS, delay, recipe)
    n = len(cohort.query)
    clean = model.rollout(torch.cat((cohort.receiver, cohort.donor)))
    receiver = clean["states"][:n, cohort.prefix_steps - 1]
    donor = clean["states"][n:, cohort.prefix_steps - 1]
    delta, physical = component_patch(model, receiver, donor, groups, cohort.edit_slot,
                                       dose=recipe.intervention_l2)
    patched = model.rollout(cohort.receiver, patch=delta, patch_step=cohort.prefix_steps)
    ordinary = prediction_record(clean["logits"][:n], cohort.labels_r)
    causal = prediction_record(patched["logits"], cohort.labels_h)
    targeted = cohort.query == cohort.edit_slot
    correct = torch.tensor(causal["correct"], dtype=torch.bool)
    causal.update({"targeted": targeted.tolist(),
                   "targeted_n": int(targeted.sum()),
                   "unaffected_n": int((~targeted).sum()),
                   "targeted_accuracy": float(correct[targeted].float().mean()),
                   "unaffected_accuracy": float(correct[~targeted].float().mean())})
    modal = modal_predictions(cohort.values_r, recipe.values)
    write_delta = (patched["recurrent_writes"][:, cohort.prefix_steps]
                   - clean["recurrent_writes"][:n, cohort.prefix_steps])
    return {
        "scope": "TRAIN-partition diagnostic; ordinary steering disabled",
        "seed": seed_for(root, "diagnostic_pairs", checkpoint, delay),
        "delay": delay, "pairs": n, "batch_sha256": cohort.fingerprint(),
        "ordinary": ordinary,
        "modal_value_baseline": {
            "scope": "host-only semantic memory values; query-blind; smallest-value tie",
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
            "recipient_state_rms": float(receiver.square().mean().sqrt()),
            "all_head_max_change": float((model.output(patched["states"])
                                            - model.output(clean["states"][:n])).abs().max()),
            "terminal_state_distance_mean": float((patched["states"][:, -1]
                                                      - clean["states"][:n, -1]).norm(dim=-1).mean())},
    }


def binding_batch(root: int, checkpoint: int, delay: int, recipe: Recipe) -> tuple[Tensor, Tensor, Tensor, dict]:
    """Admit all-distinct TRAIN tuples, then query every address on each sequence."""
    chosen_features, chosen_values, chosen_queries = [], [], []
    input_fingerprints = []
    accepted = 0
    for attempt in range(16):
        index = checkpoint * 100 + attempt
        candidates = batch(root, "address_binding_candidates", index, 2048, delay, recipe)
        input_fingerprints.append(candidates.fingerprint())
        distinct = (candidates.values_r.sort(dim=1).values
                    == torch.arange(recipe.values)[None, :]).all(dim=1)
        take = torch.where(distinct)[0][:BINDING_BASES - accepted]
        if len(take):
            chosen_features.append(candidates.receiver[take])
            chosen_values.append(candidates.values_r[take])
            chosen_queries.append(candidates.query[take])
            accepted += len(take)
        if accepted == BINDING_BASES:
            break
    if accepted != BINDING_BASES:
        raise RuntimeError("All-distinct binding diagnostic exhausted its fixed admission cap")
    features = torch.cat(chosen_features).repeat_interleave(recipe.slots, dim=0)
    memories = torch.cat(chosen_values).repeat_interleave(recipe.slots, dim=0)
    original_queries = torch.cat(chosen_queries).repeat_interleave(recipe.slots)
    queries = torch.arange(recipe.slots).repeat(BINDING_BASES)
    address_encoding, _ = root_encoding(root, recipe.slots, recipe.values)
    row = torch.arange(len(features))
    features[row, -1, 3 + address_encoding[original_queries]] -= 1.0
    features[row, -1, 3 + address_encoding[queries]] += 1.0
    labels = memories.gather(1, queries[:, None]).squeeze(1)
    modal = modal_predictions(memories, recipe.values)
    if int((modal == labels).sum()) != BINDING_BASES:
        raise AssertionError("All-query modal baseline must equal 25 percent")
    if not bool(partition_mask(memories, "train", recipe.values).all()):
        raise AssertionError("Binding diagnostic left the TRAIN partition")
    digest = hashlib.sha256()
    for tensor in (features, memories, queries, labels):
        digest.update(tensor.contiguous().numpy().tobytes())
    return features, labels, modal, {
        "scope": "TRAIN-only address-binding diagnostic; ordinary steering disabled",
        "base_sequences": BINDING_BASES, "cases": len(features), "delay": delay,
        "candidate_batches": len(input_fingerprints),
        "candidate_batch_sha256": input_fingerprints, "expanded_sha256": digest.hexdigest(),
        "query_counts": queries.bincount(minlength=recipe.slots).tolist(),
        "label_counts": labels.bincount(minlength=recipe.values).tolist(),
        "canonical_memories": memories[::recipe.slots].tolist(),
        "only_query_address_changes_within_group": True,
    }


@torch.no_grad()
def address_binding_diagnostic(model, root: int, checkpoint: int, delay: int,
                               recipe: Recipe) -> dict:
    features, labels, modal, metadata = binding_batch(root, checkpoint, delay, recipe)
    return {**metadata, "ordinary": prediction_record(model(features), labels),
            "modal_value_baseline": {"accuracy": float((modal == labels).float().mean()),
                                     "correct": (modal == labels).int().tolist(),
                                     "predictions": modal.tolist()}}


def source_hashes() -> dict[str, str]:
    paths = [HERE / "PLAN.md", Path(__file__),
             *[RP02 / name for name in ("__init__.py", "data.py", "model.py", "experiment.py", "stats.py")],
             RP02.parents[1] / "components.py", RP02.parents[1] / "config.py",
             HERE.parent / "planning/rp02_preflight_input.json",
             HERE.parent / "planning/rp02_preflight_result.json",
             HERE.parent / "planning/provenance.json"]
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_root(root: int, out: Path, recipe: Recipe) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed_for(root, "initialization"))
    model = new_model(recipe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                                  weight_decay=recipe.weight_decay)
    start = time.time()
    trace, checkpoints = [], {}
    stream = hashlib.sha256()
    for step in range(1, recipe.warmup_steps + 1):
        model.train()
        delay = recipe.train_delays[(step - 1) % len(recipe.train_delays)]
        cohort = batch(root, "task_training", step - 1, recipe.batch_size, delay, recipe)
        fingerprint = cohort.fingerprint()
        stream.update(fingerprint.encode())
        features, labels = full_task_batch(cohort)
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite task loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_l2 = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
        if not bool(torch.isfinite(gradient_l2)):
            raise ValueError("Non-finite task gradient")
        optimizer.step()
        if step % 50 == 0:
            trace.append({"updates": step, "task_ce": float(loss.detach()),
                          "preclip_gradient_l2": float(gradient_l2),
                          "batch_sha256": fingerprint,
                          "training_stream_sha256": stream.hexdigest()})
            dump(out / "training_trace.json", trace)
            print(json.dumps({"root": root, "updates": step,
                              "task_ce": trace[-1]["task_ce"]}), flush=True)
        if step in CHECKPOINTS:
            model.eval()
            checkpoint_path = out / f"checkpoint_{step}.pt"
            torch.save({"root": root, "updates": step, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, checkpoint_path)
            groups, attribution = discover(model, root, step, recipe)
            diagnostics = {str(delay): {
                "standard": standard_diagnostic(model, root, step, delay, groups, recipe),
                "address_binding": address_binding_diagnostic(model, root, step, delay, recipe)}
                for delay in DELAYS}
            record = {"updates": step, "model_state_sha256": state_hash(model),
                      "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                      "training_stream_sha256": stream.hexdigest(),
                      "attribution": attribution, "diagnostics": diagnostics}
            checkpoints[str(step)] = record
            dump(out / f"checkpoint_{step}.json", record)
            print(json.dumps({"root": root, "checkpoint": step,
                              "ordinary": {d: diagnostics[d]["standard"]["ordinary"]["accuracy"] for d in diagnostics},
                              "binding": {d: diagnostics[d]["address_binding"]["ordinary"]["accuracy"] for d in diagnostics}}), flush=True)
    result = {"root": root, "scope": "TRAIN_ONLY_ENGINEERING", "recipe": asdict(recipe),
              "training_trajectories": 3 * recipe.batch_size * recipe.warmup_steps,
              "training_trace": trace, "checkpoints": checkpoints,
              "elapsed_seconds": time.time() - start}
    dump(out / "result.json", result)
    dump(out / "artifact_hashes.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(out.iterdir()) if p.is_file()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--roots", type=int, nargs="+", default=list(ROOTS))
    args = parser.parse_args()
    if len(set(args.roots)) != len(args.roots) or any(r not in ROOTS for r in args.roots):
        parser.error("Root selection must be a unique subset of 27000 and 27001")
    if not args.out.resolve().is_relative_to(HERE):
        parser.error("Engineering output must remain inside this engineering directory")
    recipe = replace(Recipe(), warmup_steps=2000, training_steps=0)
    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    sources = source_hashes()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                          capture_output=True, check=True).stdout.strip()
    dump(args.out / "run_manifest.json", {
        "scope": "TRAIN_ONLY_ENGINEERING", "roots": args.roots,
        "checkpoints": list(CHECKPOINTS), "diagnostic_delays": list(DELAYS),
        "recipe": asdict(recipe), "source_hashes": sources, "git_head": head,
        "started_unix": start, "python": platform.python_version(), "torch": torch.__version__,
        "preflight_scope": "Prior RP02 evidence requires redesign; this is the bounded TRAIN-only diagnostic in PLAN.md"})
    results = [run_root(root, args.out / f"root_{root}", recipe) for root in args.roots]
    if sources != source_hashes():
        raise RuntimeError("Source changed during engineering run")
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
    dump(args.out / "summary.json", {"scope": "TRAIN_ONLY_ENGINEERING", "rows": rows})
    dump(args.out / "completion.json", {
        "completed": True, "roots": args.roots, "elapsed_seconds": time.time() - start,
        "result_sha256": {str(r): hashlib.sha256((args.out / f"root_{r}" / "result.json").read_bytes()).hexdigest()
                          for r in args.roots}})


if __name__ == "__main__":
    main()
