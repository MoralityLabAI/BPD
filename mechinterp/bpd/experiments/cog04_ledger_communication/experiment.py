"""Run COG-04 on the source-pinned COG-02 finite proof-search worlds."""
from __future__ import annotations

import argparse
import hashlib
import gzip
import struct
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mechinterp.bpd.components import PassMaskSchedule, TiedComponentLinear
from .transport import action_reply, decode, encode
from .vendor import ledger

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    body = "".join(json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for x in rows)
    if str(path).endswith(".gz"):
        Path(path).write_bytes(gzip.compress(body.encode(), mtime=0))
    else:
        Path(path).write_text(body)


def root_seed(task, arm):
    return int(hashlib.sha256((task["root_id"] + arm).encode()).hexdigest()[:8], 16)


def public_features(task, known):
    return np.asarray([ledger.candidate_features(task, known, i)
                       for i in range(len(task["world"]["rules"]))], dtype=np.float32)


def training_data(tasks):
    xs, ys = [], []
    roots, collected = set(), 0
    for task in tasks:
        if task["split"] != "train":
            continue
        roots.add(task["root_id"])
        known, seen = ledger.initial_state(task), set()
        for _ in range(64):
            key = tuple(known)
            if key in seen:
                raise ValueError("teacher repeated a state")
            seen.add(key)
            row = ledger.ranking_row(task, known)
            xs.extend(c["features"] for c in row["candidates"])
            ys.extend(row["labels"]["useful"])
            collected += row["collection_cost"]["verifier_calls"]
            action = ledger.reference_action(task, known)
            if action is None:
                break
            receipt = ledger.check_step(task, known, action)
            collected += 1
            known = receipt["next_known"]
            if receipt["solved"]:
                break
    if not xs or len(roots) < 2:
        raise ValueError("need at least two training roots")
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), collected


class Scorer(nn.Module):
    def __init__(self, n_features, hidden=16):
        super().__init__()
        self.input = nn.Linear(n_features, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(torch.tanh(self.input(x))).squeeze(-1)


def fit(tasks, protocol):
    torch.manual_seed(protocol["training_seed"])
    np.random.seed(protocol["training_seed"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x, y, collection_calls = training_data(tasks)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 0.1)
    normalized = (x - mean) / std
    xt, yt = torch.from_numpy(normalized), torch.from_numpy(y)
    cfg = protocol["training"]
    model = Scorer(x.shape[1], cfg["hidden_width"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    pos_weight = torch.tensor((len(y) - y.sum()) / max(1, y.sum()), dtype=torch.float32)
    losses = []
    for epoch in range(cfg["epochs"] + 1):
        logits = model(xt)
        loss = F.binary_cross_entropy_with_logits(logits, yt, pos_weight=pos_weight)
        if epoch % 10 == 0 or epoch == cfg["epochs"]:
            losses.append({"epoch": epoch, "weighted_bce": float(loss.detach())})
        if epoch < cfg["epochs"]:
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    checkpoint = {"schema": "cog04.scorer.v1", "seed": protocol["training_seed"],
                  "feature_names": list(ledger.FEATURE_NAMES), "mean": mean.tolist(), "std": std.tolist(),
                  "weights": {k: v.detach().tolist() for k, v in model.state_dict().items()},
                  "n_parameters": sum(p.numel() for p in model.parameters()), "losses": losses,
                  "training_candidates": len(y), "positive_candidates": int(y.sum()),
                  "training_roots": len([t for t in tasks if t["split"] == "train"]),
                  "offline_training_collection_verifier_calls": collection_calls,
                  "training_data_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()}
    return model, mean, std, normalized, checkpoint


class FactorProtocol:
    def __init__(self, model, mean, std, train_x, rank, bits=32, name=None,
                 shuffle=False, weight_override=None, identity_factor=False):
        self.name = name or f"rank{rank}_{bits}bit"
        self.mean, self.std = mean.copy(), std.copy()
        linear = nn.Linear(model.input.in_features, model.input.out_features)
        with torch.no_grad():
            linear.weight.copy_(model.input.weight if weight_override is None else weight_override)
            linear.bias.copy_(model.input.bias)
        self.component = TiedComponentLinear(linear, c=rank, max_passes=64, init_svd=True)
        if identity_factor:
            if rank != linear.in_features:
                raise ValueError("identity factor requires full input rank")
            with torch.no_grad():
                self.component.V.copy_(torch.eye(rank))
                self.component.U.copy_(linear.weight.T)
        self.identity_factor = identity_factor
        self.component.eval()
        self.rank, self.bits, self.shuffle = rank, bits, shuffle
        self.u = self.component.U.detach().numpy().copy()
        self.bias = model.input.bias.detach().numpy().copy()
        self.head = model.head.weight.detach().numpy().reshape(-1).copy()
        self.head_bias = float(model.head.bias.detach()[0])
        with torch.no_grad():
            ztrain = self.component.component_acts(torch.from_numpy(train_x)).numpy()
        self.scales = np.maximum(np.max(np.abs(ztrain), axis=0), 1e-4)
        # Sender and receiver each get their own quantizer scales. Their two
        # framed streams charge both physical deliveries.
        shape = struct.pack("<HHHB", mean.size, self.u.shape[1], rank, bits)
        sender_arrays = [mean, std, self.component.V.detach().numpy(), self.scales]
        receiver_arrays = [self.u, self.bias, self.head, np.array([self.head_bias]), self.scales]
        self.sender_setup = shape + b"".join(np.asarray(a, dtype="<f4").tobytes() for a in sender_arrays)
        self.receiver_setup = shape + b"".join(np.asarray(a, dtype="<f4").tobytes() for a in receiver_arrays)
        self.setup_bytes = len(self.sender_setup) + len(self.receiver_setup)
        target = model.input.weight.detach().numpy()
        approx = self.component.reconstructed_weight().detach().numpy()
        self.relative_weight_error = float(np.linalg.norm(target - approx) / max(1e-12, np.linalg.norm(target)))
        self.weight_displacement = float(np.linalg.norm(target - approx))
        self.input_weight = target.copy()
        self.full_bias = model.input.bias.detach().numpy().copy()

    def reset(self):
        self.component.reset_pass_state()

    def scores(self, features, rng):
        x = (features - self.mean) / self.std
        with torch.no_grad():
            z = self.component.component_acts(torch.from_numpy(x)).numpy()
        if self.shuffle:
            z = z[:, rng.permutation(self.rank)]
        message = encode(z, self.scales, self.bits)
        received = decode(message.payload, self.scales, expected_rank=self.rank)
        pre = received @ self.u + self.bias
        scores = np.tanh(pre) @ self.head + self.head_bias
        full_pre = x @ self.input_weight.T + self.full_bias
        full_scores = np.tanh(full_pre) @ self.head + self.head_bias
        # tanh is 1-Lipschitz. This is an offline certificate against the full
        # scorer, calculated after policy input construction. It never ranks
        # candidates or changes the transmitted payload.
        error_bounds = np.abs(pre - full_pre) @ np.abs(self.head) + 2e-5
        best = int(np.argmax(full_scores))
        competitor = np.max(np.delete(full_scores + error_bounds, best)) if len(scores) > 1 else -float("inf")
        certificate = bool(full_scores[best] - error_bounds[best] > competitor)
        if certificate and int(np.argmax(scores)) != best:
            raise AssertionError("argmax certificate violated")
        return scores, {"forward_bits": message.nbits, "reply_bits": 16,
                        "clipped_values": message.clipped_values, "component_values": message.values,
                        "logit_mse": float(np.mean((scores - full_scores) ** 2)),
                        "hidden_displacement_l2_mean": float(np.linalg.norm(pre - full_pre, axis=1).mean()),
                        "full_hidden_radius_l2_mean": float(np.linalg.norm(full_pre, axis=1).mean()),
                        "argmax_certified": certificate, "full_action": best,
                        "logit_error_bound_max": float(error_bounds.max())}

    def manifest(self):
        return {"name": self.name, "rank": self.rank, "bits_per_component": self.bits,
                "setup_bytes": self.setup_bytes, "relative_weight_error": self.relative_weight_error,
                "weight_displacement_frobenius": self.weight_displacement,
                "scales": self.scales.tolist(), "shuffle": self.shuffle, "identity_factor": self.identity_factor,
                "sender_setup_bytes": len(self.sender_setup), "receiver_setup_bytes": len(self.receiver_setup),
                "sender_setup_sha256": hashlib.sha256(self.sender_setup).hexdigest(),
                "receiver_setup_sha256": hashlib.sha256(self.receiver_setup).hexdigest()}


def make_arms(model, mean, std, train_x, protocol):
    full = min(model.input.in_features, model.input.out_features)
    ranks = sorted(set(full if r == "full" else int(r) for r in protocol["rank_grid"]))
    arms = {"full_rank_float32": FactorProtocol(model, mean, std, train_x, full, 32,
                                                "full_rank_float32", identity_factor=True)}
    for rank in ranks:
        for bits in protocol["bits_grid"]:
            arm = FactorProtocol(model, mean, std, train_x, rank, bits)
            arms[arm.name] = arm
    arms["shuffled_rank4_float32"] = FactorProtocol(model, mean, std, train_x, 4, 32,
                                                  "shuffled_rank4_float32", shuffle=True)
    target = model.input.weight.detach().numpy()
    delta = arms["rank4_32bit"].component.reconstructed_weight().detach().numpy() - target
    for i in range(3):
        rng = np.random.default_rng(protocol["training_seed"] + 100 + i)
        left = np.linalg.qr(rng.normal(size=(target.shape[0], target.shape[0])))[0]
        right = np.linalg.qr(rng.normal(size=(target.shape[1], target.shape[1])))[0]
        perturbation = left @ delta @ right.T
        weight = torch.tensor(target + perturbation, dtype=torch.float32)
        arm = FactorProtocol(model, mean, std, train_x, full, 32,
                             f"dose_matched_sham_{i}", weight_override=weight)
        arms[arm.name] = arm
    return arms


def rollout(task, arm, name, budget):
    known = ledger.initial_state(task)
    needed = ledger.useful_atoms(task)
    rng = np.random.default_rng(root_seed(task, name))
    totals = Counter()
    predictions = []
    rejected = set()
    if arm:
        arm.reset()
    for step in range(budget):
        available = [i for i, rule in enumerate(task["world"]["rules"])
                     if rule["conclusion"] not in known and i not in rejected]
        if not available:
            break
        features = public_features(task, known)[available]
        if arm:
            scores, metadata = arm.scores(features, rng)
            selected_position = int(np.argmax(scores))
            action = available[selected_position]
            metadata["full_action"] = available[metadata["full_action"]]
        elif name == "public_structural_heuristic":
            action = ledger.reference_action(task, known)
            if action is None:
                break
            metadata = {"forward_bits": 0, "reply_bits": 0}
        elif name == "uniform_random":
            action = available[int(rng.integers(0, len(features)))]
            metadata = {"forward_bits": 0, "reply_bits": 0}
        else:
            raise ValueError("unknown arm")
        if arm:
            if len(action_reply(selected_position, len(features))) * 8 != metadata["reply_bits"]:
                raise AssertionError("reply accounting failure")
        receipt = ledger.check_step(task, known, action)
        useful = receipt["new_fact"] is not None and receipt["new_fact"] in needed
        full_useful = False
        if arm:
            ref_receipt = ledger.check_step(task, known, metadata["full_action"])
            full_useful = ref_receipt["new_fact"] is not None and ref_receipt["new_fact"] in needed
            # Instrumentation calls are separate from the policy's budget and
            # are fully charged in the audit collection total.
            totals["offline_audit_verifier_calls"] += 1
            totals["correct_to_wrong"] += int(full_useful and not useful)
            totals["wrong_to_correct"] += int(useful and not full_useful)
            totals["full_useful"] += int(full_useful)
        totals["verifier_calls"] += receipt["verifier_calls"]
        totals["accepted"] += int(receipt["accepted"])
        totals["useful"] += int(useful)
        totals["new_lemmas"] += int(receipt["new_fact"] is not None)
        for key, val in metadata.items():
            if key != "full_action":
                totals[key] += val
        predictions.append({"root_id": task["root_id"], "split": task["split"], "arm": name,
                            "step": step, "action": action, "candidate_count": len(available), "accepted": receipt["accepted"],
                            "useful": useful, "new_fact": receipt["new_fact"], "solved": receipt["solved"],
                            "code": receipt["code"], "verifier_calls": 1,
                            "known_before_sha256": ledger.digest(known), **metadata})
        if receipt["new_fact"] is not None:
            rejected.clear()
        elif not receipt["accepted"]:
            rejected.add(action)
        known = receipt["next_known"]
        if receipt["solved"]:
            break
    n = max(1, totals["verifier_calls"])
    result = {"root_id": task["root_id"], "task_id": task["task_id"], "split": task["split"],
              "family": task["family"], "arm": name, "budget": budget,
              "solved": task["world"]["goal"] in known, **dict(totals),
              "useful_fraction": totals["useful"] / n,
              "accepted_fraction": totals["accepted"] / n,
              "known_final_sha256": ledger.digest(known)}
    return result, predictions


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["arm"])].append(row)
    out = []
    for (split, arm), values in sorted(groups.items()):
        calls = sum(v["verifier_calls"] for v in values)
        items = {"split": split, "arm": arm, "roots": len(values),
                 "solved": sum(v["solved"] for v in values),
                 "solve_fraction": sum(v["solved"] for v in values) / len(values)}
        for key in ("verifier_calls", "offline_audit_verifier_calls", "forward_bits", "reply_bits", "clipped_values",
                    "component_values", "accepted", "useful", "new_lemmas", "correct_to_wrong", "wrong_to_correct",
                    "full_useful", "argmax_certified"):
            items[key] = sum(v.get(key, 0) for v in values)
        for key in ("logit_mse", "hidden_displacement_l2_mean", "full_hidden_radius_l2_mean", "logit_error_bound_max"):
            items[key] = sum(v.get(key, 0) for v in values) / max(1, calls)
        items["useful_per_verifier_call"] = items["useful"] / max(1, calls)
        items["solve_per_100_verifier_calls"] = 100 * items["solved"] / max(1, calls)
        items["certified_step_fraction"] = items["argmax_certified"] / max(1, calls)
        items["clipping_fraction"] = items["clipped_values"] / max(1, items["component_values"])
        items["mean_root_message_bytes"] = (items["forward_bits"] + items["reply_bits"]) / (8 * len(values))
        out.append(items)
    return out


def build_preflight(calibration, out, comparisons):
    by_root = defaultdict(dict)
    for r in calibration:
        by_root[r["root_id"]][r["arm"]] = r
    full_name = "full_rank_float32"
    pilot = []
    for root, rows in sorted(by_root.items()):
        full = int(rows[full_name]["solved"])
        pilot.append({"unit_id": root,
                      "paired_effect": str(int(rows["rank4_4bit"]["solved"]) - full),
                      "sham_effects": [str(int(rows[f"dose_matched_sham_{i}"]["solved"]) - full) for i in range(3)]})
    spec = {"schema_version": 1, "experiment_id": "COG-04", "independent_unit": "generated DAG root conditional on checkpoint",
            "alpha": "0.05", "scientific_min_effect": "0.05", "target_effect": "0.10",
            "measurement_resolution": "0", "sham_quantile": "0.95", "sham_confidence": "0.95",
            "planned_comparisons": comparisons, "confirm_units_planned": 64,
            "pilot_from_development_only": True, "heldout_sealed": False, "units_independent": True, "pilot": pilot}
    write_json(out / "preflight_input.json", spec)
    proc = subprocess.run([sys.executable, str(HERE / "vendor/preflight.py"), "--pilot", str(out / "preflight_input.json"),
                           "--out", str(out / "preflight.json"), "--strict"], text=True, capture_output=True)
    if proc.returncode not in (0, 2):
        raise RuntimeError(proc.stderr or proc.stdout)
    return json.loads((out / "preflight.json").read_text())


def run(out, per_split=64, budget=24):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE / "protocol.json").read_text())
    tasks = ledger.generate_tasks(per_split, seed=20261126)
    roots = defaultdict(set)
    for task in tasks:
        ledger.validate_task(task)
        roots[task["root_id"]].add(task["split"])
    if any(len(s) != 1 for s in roots.values()):
        raise ValueError("root leakage across splits")
    train_tasks = [t for t in tasks if t["split"] == "train"]
    model, mean, std, train_x, checkpoint = fit(train_tasks, protocol)
    write_json(out / "checkpoint.json", checkpoint)
    arms = make_arms(model, mean, std, train_x, protocol)
    full_name = "full_rank_float32"
    write_json(out / "arms.json", [a.manifest() for a in arms.values()])
    write_json(out / "setup_streams.json", {name: {"sender_hex": arm.sender_setup.hex(), "receiver_hex": arm.receiver_setup.hex()}
                                           for name, arm in arms.items()})
    write_json(out / "split_manifest.json", [{k: t[k] for k in ("root_id", "task_id", "split", "family", "task_digest")} for t in tasks])
    calibration, cal_decisions = [], []
    for task in tasks:
        if task["split"] == "calibration":
            for name in [full_name, "rank4_4bit"] + [f"dose_matched_sham_{i}" for i in range(3)]:
                result, decisions = rollout(task, arms[name], name, budget)
                calibration.append(result)
                cal_decisions.extend(decisions)
    write_jsonl(out / "calibration_roots.jsonl", calibration)
    preflight = build_preflight(calibration, out, len(arms) + 1)
    # A preflight result is recorded before any ordinary/ood readout. This
    # public capability pilot does not release a mechanistic confirmation.
    records, decisions = [], []
    smoke_exposed = {t["root_id"] for t in ledger.generate_tasks(4, seed=20261126)
                     if t["split"] in ("ordinary_test", "ood_test")}
    for task in tasks:
        if task["split"] in ("ordinary_test", "ood_test"):
            for name, arm in list(arms.items()) + [("uniform_random", None), ("public_structural_heuristic", None)]:
                row, detail = rollout(task, arm, name, budget)
                row["earlier_transport_smoke_exposure"] = task["root_id"] in smoke_exposed
                records.append(row)
                decisions.extend(detail)
    write_jsonl(out / "per_root.jsonl", records)
    write_jsonl(out / "decisions.jsonl.gz", decisions)
    summary = {"experiment": "COG-04", "schema_version": 1,
               "evidence_class": "synthetic observational engineering pilot",
               "independent_unit": "generated proof DAG root, conditional on one learned checkpoint",
               "model_seeds": [protocol["training_seed"]], "verifier_budget": budget,
               "per_split": per_split, "fitted_train_roots": len(train_tasks),
               "feature_names": list(ledger.FEATURE_NAMES), "n_parameters": checkpoint["n_parameters"],
               "fit_initial_loss": checkpoint["losses"][0]["weighted_bce"],
               "fit_final_loss": checkpoint["losses"][-1]["weighted_bce"],
               "offline_training_collection_verifier_calls": checkpoint["offline_training_collection_verifier_calls"],
               "calibration_policy_verifier_calls": sum(r["verifier_calls"] for r in calibration),
               "calibration_audit_verifier_calls": sum(r.get("offline_audit_verifier_calls", 0) for r in calibration),
               "readout_audit_verifier_calls": sum(r.get("offline_audit_verifier_calls", 0) for r in records),
               "preflight": preflight, "readouts": summarize(records),
               "fresh_root_readouts": summarize([r for r in records if not r["earlier_transport_smoke_exposure"]]),
               "smoke_exposed_readout_roots": sorted(smoke_exposed),
               "selection_exposure": "First four ordinary and four OOD roots were used in a transport smoke with four training roots. Architecture and arm grid were fixed before smoke. Fresh-root readouts exclude them.",
               "root_overlap_across_splits": 0, "source_module_sha256": sha(HERE / "vendor/ledger.py"),
               "protocol_sha256": sha(HERE / "protocol.json"), "checkpoint_sha256": sha(out / "checkpoint.json"),
               "implementation_sha256": {str(p.relative_to(HERE)) if p.is_relative_to(HERE) else "bpd_components.py": sha(p)
                                           for p in [HERE / "experiment.py", HERE / "transport.py", HERE / "vendor/preflight.py",
                                                     HERE.parents[1] / "components.py"]},
               "torch_version": torch.__version__, "numpy_version": np.__version__}
    write_json(out / "summary.json", summary)
    write_json(out / "file_hashes.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "file_hashes.json"})
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=HERE / "results")
    p.add_argument("--per-split", type=int, default=64)
    p.add_argument("--budget", type=int, default=24)
    a = p.parse_args()
    s = run(a.out, a.per_split, a.budget)
    print(json.dumps({k: s[k] for k in ("experiment", "n_parameters", "fit_initial_loss", "fit_final_loss", "preflight")}, indent=2))
    for r in s["readouts"]:
        print(r["split"], r["arm"], r["solved"], r["roots"], r["verifier_calls"], round(r["mean_root_message_bytes"], 2))


if __name__ == "__main__":
    main()
