"""Prospective bounded TRAIN-only RP03 delay and state-edit dose admission."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
import torch.nn.functional as F

from mechinterp.bpd.experiments.training_steering.data import (
    make_pairs, partition_mask, root_encoding,
)
from mechinterp.bpd.experiments.training_steering.experiment import Recipe, state_hash
from mechinterp.bpd.experiments.training_steering.model import ComponentMemory
from mechinterp.bpd.experiments.training_steering_stability.model import (
    ComponentStateMemory, projectors, state_patch,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RP02 = HERE.parents[1] / "training_steering"
ROOTS = (27000, 27001)
DELAYS = (31, 63, 127)
DOSES = (.75, 1.5, 3.)
ORDINARY_N, BINDING_N, CALIBRATION_N, CAUSAL_N = 512, 256, 512, 256


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(root, domain, index=0, delay=0):
    raw = f"BPD-RP-03:admission:{root}:{domain}:{index}:{delay}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def batch(root, domain, n, delay, recipe, index=0):
    result = make_pairs(seed_for(root, domain, index, delay), n, root_seed=root,
                        delay=delay, split="train", slots=recipe.slots,
                        values=recipe.values, noise=recipe.noise)
    for memory in (result.values_r, result.values_d, result.values_h):
        if not bool(partition_mask(memory, "train", recipe.values).all()):
            raise AssertionError("Admission generation left the TRAIN partition")
    return result


def prediction_record(logits, labels):
    predictions = logits.argmax(-1)
    correct = predictions.eq(labels)
    return {"n": len(labels), "accuracy": float(correct.float().mean()),
            "cross_entropy": float(F.cross_entropy(logits, labels)),
            "predictions": predictions.tolist(), "labels": labels.tolist(),
            "correct": correct.int().tolist()}


def modal(memory, values=4):
    return F.one_hot(memory, num_classes=values).sum(1).argmax(1)


def binding_batch(root, delay, recipe):
    wanted = BINDING_N // recipe.slots
    features, memories, old_queries, candidate_hashes, seeds = [], [], [], [], []
    admitted = 0
    for attempt in range(16):
        pairs = batch(root, "binding_candidates", 2048, delay, recipe, attempt)
        candidate_hashes.append(pairs.fingerprint())
        seeds.append(seed_for(root, "binding_candidates", attempt, delay))
        distinct = pairs.values_r.sort(dim=1).values.eq(torch.arange(recipe.values)).all(1)
        selected = torch.where(distinct)[0][:wanted - admitted]
        if len(selected):
            features.append(pairs.receiver[selected])
            memories.append(pairs.values_r[selected])
            old_queries.append(pairs.query[selected])
            admitted += len(selected)
        if admitted == wanted:
            break
    if admitted != wanted:
        raise RuntimeError("Bounded binding admission exhausted")
    x = torch.cat(features).repeat_interleave(recipe.slots, 0)
    memory = torch.cat(memories).repeat_interleave(recipe.slots, 0)
    original_queries = torch.cat(old_queries).repeat_interleave(recipe.slots)
    queries = torch.arange(recipe.slots).repeat(wanted)
    address, _ = root_encoding(root, recipe.slots, recipe.values)
    rows = torch.arange(BINDING_N)
    x[rows, -1, 3 + address[original_queries]] -= 1.
    x[rows, -1, 3 + address[queries]] += 1.
    labels = memory.gather(1, queries[:, None]).squeeze(1)
    modal_predictions = modal(memory, recipe.values)
    if int(modal_predictions.eq(labels).sum()) != wanted:
        raise AssertionError("Binding modal accuracy must equal 25 percent")
    if not bool(partition_mask(memory, "train", recipe.values).all()):
        raise AssertionError("Binding generation left TRAIN")
    digest = hashlib.sha256()
    for value in (x, memory, queries, labels):
        digest.update(value.contiguous().numpy().tobytes())
    return x, labels, modal_predictions, {
        "split": "train", "base_sequences": wanted, "cases": BINDING_N,
        "candidate_seeds": seeds, "candidate_batch_sha256": candidate_hashes,
        "expanded_sha256": digest.hexdigest(), "canonical_memories": memory[::4].tolist(),
        "query_counts": queries.bincount(minlength=recipe.slots).tolist(),
        "label_counts": labels.bincount(minlength=recipe.values).tolist(),
        "only_query_address_changes_within_group": True,
    }


def checkpoint_path(root):
    return HERE / "extension_results" / f"root_{root}" / "checkpoint_6000.pt"


def load_model(root, recipe, original=False):
    checkpoint = torch.load(checkpoint_path(root), map_location="cpu", weights_only=True)
    if checkpoint["root"] != root or checkpoint["updates"] != 6000:
        raise AssertionError("Wrong checkpoint")
    cls = ComponentMemory if original else ComponentStateMemory
    model = cls(width=recipe.width, components=recipe.components,
                input_dim=3 + recipe.slots + recipe.values, values=recipe.values)
    model.load_state_dict(checkpoint["model"], strict=True)
    expected = json.loads(checkpoint_path(root).with_suffix(".json").read_text())
    if file_hash(checkpoint_path(root)) != expected["checkpoint_sha256"]:
        raise AssertionError("Saved checkpoint bytes changed")
    if state_hash(model) != expected["model_state_sha256"]:
        raise AssertionError("Saved model state changed")
    model.eval()
    return model


@torch.no_grad()
def delay_readout(model, root, delay, recipe, original_model=None):
    pairs = batch(root, "ordinary", ORDINARY_N, delay, recipe)
    logits = model(pairs.receiver)
    baseline = modal(pairs.values_r, recipe.values)
    x, labels, binding_modal, metadata = binding_batch(root, delay, recipe)
    result = {
        "root": root, "delay": delay, "split": "train", "steering_enabled": False,
        "seed": seed_for(root, "ordinary", delay=delay),
        "batch_sha256": pairs.fingerprint(), "ordinary": prediction_record(logits, pairs.labels_r),
        "modal_value_baseline": {"accuracy": float(baseline.eq(pairs.labels_r).float().mean()),
                                  "predictions": baseline.tolist()},
        "binding": {**metadata, "ordinary": prediction_record(model(x), labels),
                    "modal_value_baseline": {"accuracy": float(binding_modal.eq(labels).float().mean()),
                                              "predictions": binding_modal.tolist()}},
    }
    if original_model is not None:
        difference = float((original_model(pairs.receiver) - logits).abs().max())
        if difference != 0:
            raise AssertionError("Unpatched RP03 and saved RP02 ordinary paths differ")
        result["original_class_logits_max_error"] = difference
    return result


def select_delay(results):
    rows = []
    for delay in DELAYS:
        accuracies = {str(root): results[root][str(delay)]["ordinary"]["accuracy"] for root in ROOTS}
        rows.append({"delay": delay, "ordinary_accuracy_by_root": accuracies,
                     "passes": all(.55 <= value <= .90 for value in accuracies.values())})
    accepted = [row["delay"] for row in rows if row["passes"]]
    return {"selected_delay": accepted[0] if accepted else 127,
            "headroom_gate_passed": bool(accepted),
            "flags": [] if accepted else ["HEADROOM_GATE_FAILED"], "grid": rows}


@torch.no_grad()
def discover(model, root, dose, recipe):
    cohort = batch(root, "calibration", CALIBRATION_N, 2, recipe)
    n, step = len(cohort.query), cohort.prefix_steps
    clean = model.rollout(torch.cat((cohort.receiver, cohort.donor)))
    receiver, donor = clean["states"][:n, step - 1], clean["states"][n:, step - 1]
    candidates = torch.arange(recipe.components).reshape(-1, recipe.group_size)
    all_projectors = projectors(model, candidates)
    losses, targeted_accuracy, invalid, norm_errors = [], [], [], []
    for projector in all_projectors:
        delta, info = state_patch(receiver, donor, projector.expand(recipe.slots, -1, -1),
                                   cohort.edit_slot, dose)
        logits = model.rollout(cohort.receiver, patch=delta, patch_step=step)["logits"]
        ce = F.cross_entropy(logits, cohort.labels_h, reduction="none")
        correct = logits.argmax(-1).eq(cohort.labels_h)
        losses.append([float(ce[cohort.edit_slot == slot].mean()) for slot in range(recipe.slots)])
        targeted_accuracy.append([float(correct[(cohort.edit_slot == slot) &
                                                (cohort.query == slot)].float().mean())
                                  for slot in range(recipe.slots)])
        invalid.append(int((~info["valid"]).sum()))
        norm_errors.append(float((info["delivered_norm"][info["valid"]] - dose).abs().max())
                           if bool(info["valid"].any()) else None)
    chosen = torch.tensor(losses).argmin(0)
    groups, fixed = candidates[chosen].clone(), all_projectors[chosen].clone()
    return groups, fixed, {
        "seed": seed_for(root, "calibration", delay=2), "batch_sha256": cohort.fingerprint(),
        "n": n, "delay": 2, "dose": dose, "split": "train",
        "candidate_groups": candidates.tolist(), "candidate_ce_by_slot": losses,
        "candidate_targeted_accuracy_by_slot": targeted_accuracy,
        "invalid_by_candidate": invalid, "dose_error_by_candidate": norm_errors,
        "selected_candidate": chosen.tolist(), "groups": groups.tolist(),
        "selection_scope": "minimum mean hybrid CE per edited slot; eight groups; overlap allowed",
        "clean_calibration_accuracy": float(clean["logits"][:n].argmax(-1).eq(cohort.labels_r).float().mean()),
    }


def rotation_for(root, width):
    generator = torch.Generator().manual_seed(seed_for(root, "rotation"))
    return (torch.randperm(width, generator=generator),
            2 * torch.randint(0, 2, (width,), generator=generator).float() - 1)


def outcome(logits, cohort, clean_correct):
    result = prediction_record(logits, cohort.labels_h)
    target = cohort.query.eq(cohort.edit_slot)
    correct = logits.argmax(-1).eq(cohort.labels_h)
    original_correct = logits.argmax(-1).eq(cohort.labels_r)
    eligible = ~target & clean_correct
    corruptions = int((eligible & ~original_correct).sum())
    result.update({
        "targeted": target.tolist(), "original_labels": cohort.labels_r.tolist(),
        "targeted_n": int(target.sum()), "targeted_correct": int((target & correct).sum()),
        "targeted_accuracy": float(correct[target].float().mean()),
        "targeted_original_retained": int((target & original_correct).sum()),
        "unaffected_n": int((~target).sum()), "unaffected_correct": int((~target & correct).sum()),
        "unaffected_accuracy": float(correct[~target].float().mean()),
        "unaffected_originally_correct": int(eligible.sum()),
        "unaffected_original_corruptions": corruptions,
        "unaffected_original_recoveries": int((~target & ~clean_correct & original_correct).sum()),
        "unaffected_original_corruption_rate": corruptions / int(eligible.sum()) if bool(eligible.any()) else 1.,
        "unaffected_corruptions_per_all_cases": corruptions / int((~target).sum()),
    })
    return result


def physical_record(model, run, ordinary, receiver, donor, delta, step):
    n = len(receiver)
    incoming = run["incoming_states"][:, step]
    actual = incoming - receiver
    radius = lambda tensor: tensor.norm(dim=-1).tolist()
    return {
        "recipient_radius": radius(receiver), "donor_radius": radius(donor),
        "edited_incoming_radius": radius(incoming), "delivered_l2": radius(delta),
        "actual_incoming_displacement_l2": radius(actual),
        "incoming_delta_max_error": float((actual - delta).abs().max()),
        "prefix_state_max_error": float((run["states"][:, :step] - ordinary["states"][:n, :step]).abs().max()),
        "recipient_state_rms": float(receiver.square().mean().sqrt()),
        "edited_incoming_state_rms": float(incoming.square().mean().sqrt()),
        "terminal_state_distance": radius(run["states"][:, -1] - ordinary["states"][:n, -1]),
        "all_head_max_change": float((model.output(run["states"]) - model.output(ordinary["states"][:n])).abs().max()),
    }


@torch.no_grad()
def causal_readout(model, root, delay, dose, fixed, rotation, recipe):
    cohort = batch(root, "causal", CAUSAL_N, delay, recipe)
    n, step = len(cohort.query), cohort.prefix_steps
    ordinary = model.rollout(torch.cat((cohort.receiver, cohort.donor, cohort.hybrid)))
    receiver, donor, hybrid = (ordinary["states"][i*n:(i+1)*n, step - 1] for i in range(3))
    delta, info = state_patch(receiver, donor, fixed, cohort.edit_slot, dose)
    sham = delta[:, rotation[0]] * rotation[1]
    clean_correct = ordinary["logits"][:n].argmax(-1).eq(cohort.labels_r)
    result = {
        "root": root, "delay": delay, "dose": dose, "split": "train",
        "seed": seed_for(root, "causal", delay=delay), "batch_sha256": cohort.fingerprint(),
        "ordinary": prediction_record(ordinary["logits"][:n], cohort.labels_r),
        "raw_projected_l2": info["raw_norm"].tolist(), "valid": info["valid"].tolist(),
        "fixed_sham_norm_max_error": float((delta.norm(dim=-1) - sham.norm(dim=-1)).abs().max()),
        "arms": {},
    }
    for name, intervention in (("fixed", delta), ("matched_rotated", sham),
                               ("hybrid_state_oracle", hybrid - receiver)):
        if name == "hybrid_state_oracle":
            run = model.rollout(cohort.receiver, state_replacement=hybrid, patch_step=step)
        else:
            run = model.rollout(cohort.receiver, patch=intervention, patch_step=step)
        record = outcome(run["logits"], cohort, clean_correct)
        record["physical"] = physical_record(model, run, ordinary, receiver, donor, intervention, step)
        if name == "hybrid_state_oracle":
            difference = float((run["logits"] - ordinary["logits"][2*n:]).abs().max())
            if difference > 1e-5:
                raise AssertionError("Hybrid-state oracle failed native continuation parity")
            record["native_hybrid_logits_max_error"] = difference
            record["scope"] = "privileged full hybrid prefix; unmatched natural radius"
        if record["physical"]["prefix_state_max_error"] != 0:
            raise AssertionError("Intervention changed a prior state")
        if record["physical"]["incoming_delta_max_error"] > 1e-5:
            raise AssertionError("Physical hidden-state displacement mismatch")
        result["arms"][name] = record
    if result["fixed_sham_norm_max_error"] > 1e-5:
        raise AssertionError("Signed permutation changed physical intervention norm")
    return result


def select_dose(results):
    rows = []
    scores = {}
    for dose in DOSES:
        arms = {}
        for arm in ("fixed", "matched_rotated", "hybrid_state_oracle"):
            records = [results[root][str(dose)]["causal"]["arms"][arm] for root in ROOTS]
            fields = ("targeted_n", "targeted_correct", "unaffected_n", "unaffected_correct",
                      "unaffected_originally_correct", "unaffected_original_corruptions",
                      "unaffected_original_recoveries")
            pooled = {field: sum(record[field] for record in records) for field in fields}
            accuracy = Fraction(pooled["targeted_correct"], pooled["targeted_n"])
            denominator = pooled["unaffected_originally_correct"]
            corruption = Fraction(pooled["unaffected_original_corruptions"], denominator) if denominator else Fraction(1)
            pooled.update({"targeted_accuracy": float(accuracy),
                           "unaffected_accuracy": pooled["unaffected_correct"] / pooled["unaffected_n"],
                           "unaffected_original_corruption_rate": float(corruption),
                           "targeted_minus_corruption": float(accuracy - corruption)})
            if arm == "fixed":
                scores[dose] = accuracy - corruption
                pooled["passes"] = accuracy >= Fraction(1, 4) and corruption <= Fraction(1, 5)
            arms[arm] = pooled
        rows.append({"dose": dose, "arms": arms})
    accepted = [row["dose"] for row in rows if row["arms"]["fixed"]["passes"]]
    selected = accepted[0] if accepted else max(DOSES, key=lambda dose: (scores[dose], -dose))
    return {"selected_dose": selected, "transport_gate_passed": bool(accepted),
            "flags": [] if accepted else ["TRANSPORT_GATE_FAILED"], "grid": rows}


def source_hashes():
    paths = [Path(__file__), HERE / "ADMISSION_PLAN.md", HERE.parent / "model.py",
             HERE.parent / "__init__.py", RP02.parents[1] / "components.py",
             RP02.parents[1] / "config.py",
             *[RP02 / name for name in ("__init__.py", "model.py", "data.py", "experiment.py", "stats.py")],
             *[HERE.parent / "planning" / name for name in (
                 "rp02_preflight_input.json", "rp02_preflight_result.json", "provenance.json")],
             *[HERE / "extension_results" / name for name in (
                 "run_manifest.json", "completion.json", "selection.json", "replay_audit.json")]]
    for root in ROOTS:
        paths.extend((checkpoint_path(root), checkpoint_path(root).with_suffix(".json")))
    return {str(path.relative_to(REPO)): file_hash(path) for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "admission_results")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(HERE):
        raise ValueError("Admission output must remain in the engineering directory")
    out.mkdir(parents=True, exist_ok=False)
    recipe = replace(Recipe(), warmup_steps=6000, calibration_n=CALIBRATION_N)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sources = source_hashes()
    manifest = {"scope": "TRAIN-only engineering selection on two exposed roots",
                "roots": ROOTS, "checkpoints": 6000, "delays": DELAYS, "doses": DOSES,
                "ordinary_n": ORDINARY_N, "binding_n": BINDING_N,
                "calibration_n": CALIBRATION_N, "causal_n": CAUSAL_N,
                "recipe": asdict(recipe), "source_hashes": sources,
                "python": platform.python_version(), "torch": str(torch.__version__),
                "threads": 1, "seed_namespace": "BPD-RP-03:admission", "started_unix": time.time()}
    dump(out / "run_manifest.json", manifest)
    models, delays = {}, {}
    for root in ROOTS:
        models[root] = load_model(root, recipe)
        original = load_model(root, recipe, original=True)
        delays[root] = {}
        for delay in DELAYS:
            record = delay_readout(models[root], root, delay, recipe,
                                   original_model=original if delay == DELAYS[0] else None)
            delays[root][str(delay)] = record
            dump(out / f"root_{root}" / f"delay_{delay}.json", record)
            print(json.dumps({"root": root, "delay": delay, "ordinary": record["ordinary"]["accuracy"],
                              "binding": record["binding"]["ordinary"]["accuracy"]}), flush=True)
    delay_choice = select_delay(delays)
    dump(out / "delay_selection.json", delay_choice)
    doses = {}
    for root in ROOTS:
        doses[root] = {}
        rotation = rotation_for(root, recipe.width)
        for dose in DOSES:
            groups, fixed, attribution = discover(models[root], root, dose, recipe)
            causal = causal_readout(models[root], root, delay_choice["selected_delay"], dose,
                                    fixed, rotation, recipe)
            record = {"root": root, "dose": dose, "attribution": attribution, "causal": causal}
            doses[root][str(dose)] = record
            dump(out / f"root_{root}" / f"dose_{dose}.json", record)
            torch.save({"groups": groups, "fixed_projectors": fixed,
                        "permutation": rotation[0], "signs": rotation[1]},
                       out / f"root_{root}" / f"geometry_{dose}.pt")
            print(json.dumps({"root": root, "dose": dose,
                              "fixed_targeted": causal["arms"]["fixed"]["targeted_accuracy"],
                              "fixed_corruption": causal["arms"]["fixed"]["unaffected_original_corruption_rate"]}), flush=True)
        if state_hash(models[root]) != json.loads(checkpoint_path(root).with_suffix(".json").read_text())["model_state_sha256"]:
            raise AssertionError("Admission changed saved model state")
    dose_choice = select_dose(doses)
    selection = {"selected_delay": delay_choice["selected_delay"],
                 "selected_dose": dose_choice["selected_dose"],
                 "headroom_gate_passed": delay_choice["headroom_gate_passed"],
                 "transport_gate_passed": dose_choice["transport_gate_passed"],
                 "flags": delay_choice["flags"] + dose_choice["flags"],
                 "delay_selection": delay_choice, "dose_selection": dose_choice,
                 "scope": "TRAIN-only selection on exposed engineering roots 27000 and 27001",
                 "confirmation_authorized": False}
    dump(out / "selection.json", selection)
    if source_hashes() != sources:
        raise AssertionError("Admission sources changed during execution")
    artifacts = {str(path.relative_to(out)): file_hash(path) for path in sorted(out.rglob("*")) if path.is_file()}
    dump(out / "completion.json", {"elapsed_seconds": time.time() - manifest["started_unix"],
                                    "source_hashes_match": True, "artifacts": artifacts,
                                    "verdict": "BOUNDED_TRAIN_ADMISSION_COMPLETE"})
    print(json.dumps({"selected_delay": selection["selected_delay"],
                      "selected_dose": selection["selected_dose"], "flags": selection["flags"]}), flush=True)


if __name__ == "__main__":
    main()
