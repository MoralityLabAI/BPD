"""Independent, read-only RP-01 receipt audit.

This file imports no RP-01 or BPD implementation. It does not train models,
choose new policies, or generate new task roots. Candidate retention is rebuilt
from the score ledger. Archived gates are replayed through explicit tensor
operations to check the native recurrence and calibration scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ARMS = ("greedy", "multistart", "beam")
ZERO = (0, 0, 0, 0)
REQUIRED_HASHED = {
    "audit.json", "calibration_traces.npz", "checkpoint.pt", "events.json",
    "search.json", "terminal_traces.npz", "training.json",
}


class AuditFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def near(actual, expected, message, tolerance=2e-5):
    require(math.isfinite(float(actual)) and
            abs(float(actual) - float(expected)) <= tolerance,
            f"{message}: {actual!r} versus {expected!r}")


def read_json(path):
    return json.loads(path.read_text())


def rebuild_search(receipt):
    """Rebuild nine slots and stable distinct retention using no search imports."""
    candidates, retentions = receipt["candidates"], receipt["retentions"]
    cursor = 0
    selection_cursor = 0
    used = Counter()

    def consume(code, arm, parent, stage, slot=0, edit="initial", padded=False):
        nonlocal cursor
        require(cursor < len(candidates), "Search candidate ledger ended early")
        row = candidates[cursor]
        expected = dict(id=f"c{cursor:04d}", code=list(code), arm=arm, parent=parent,
                        stage=stage, slot=slot, edit=edit, padded=padded)
        for field, value in expected.items():
            require(row.get(field) == value,
                    f"Candidate {cursor} field {field}: {row.get(field)!r} != {value!r}")
        require(isinstance(row["score"], (int, float)) and
                math.isfinite(row["score"]), f"Candidate {cursor} score is invalid")
        used[arm] += 1
        cursor += 1
        return row

    def expand(parent, arm, stage):
        code = tuple(parent["code"])
        out = [consume(code, arm, parent["id"], stage, 0, "noop", False)]
        slot = 1
        for coordinate in range(4):
            for direction in (1, -1):
                child = list(code)
                child[coordinate] += direction
                padded = abs(child[coordinate]) > 2
                out.append(consume(code if padded else child, arm, parent["id"],
                                   stage, slot, f"group_{coordinate}:{direction:+d}", padded))
                slot += 1
        return out

    def select(pool, width, arm, stage):
        nonlocal selection_cursor
        ordered = sorted(enumerate(pool), key=lambda pair: (pair[1]["score"], pair[0]))
        seen, chosen = set(), []
        for _, row in ordered:
            code = tuple(row["code"])
            if code not in seen:
                seen.add(code)
                chosen.append(row)
            if len(chosen) == width:
                break
        require(len(chosen) == width, f"{arm}/{stage} has too few distinct candidates")
        require(selection_cursor < len(retentions), "Retention ledger ended early")
        expected = dict(arm=arm, stage=stage, pool=[r["id"] for r in pool],
                        selected=[r["id"] for r in chosen])
        require(retentions[selection_cursor] == expected,
                f"Retention {selection_cursor} does not match independent rebuild")
        selection_cursor += 1
        return chosen

    initial = consume(ZERO, "common", None, "initial")
    starters = select([initial] + expand(initial, "common", "bootstrap"),
                      3, "common", "bootstrap")
    greedy = starters[0]
    for depth in range(18):
        greedy = select([greedy] + expand(greedy, "greedy", depth),
                        1, "greedy", depth)[0]
    paths = list(starters)
    for depth in range(6):
        for branch in range(3):
            stage = f"round_{depth}:path_{branch}"
            parent = paths[branch]
            paths[branch] = select([parent] + expand(parent, "multistart", stage),
                                   1, "multistart", stage)[0]
    multistart = select(paths, 1, "multistart", "terminal")[0]
    frontier = list(starters)
    for depth in range(6):
        pool = list(frontier)
        for parent in tuple(frontier):
            pool.extend(expand(parent, "beam", depth))
        frontier = select(pool, 3, "beam", depth)
    selected = dict(greedy=greedy, multistart=multistart, beam=frontier[0])
    require(cursor == len(candidates) == 496, "Physical candidate count differs from 496")
    require(selection_cursor == len(retentions) == 44, "Retention count differs from 44")
    counts = dict(common=10, greedy=162, multistart=162, beam=162)
    require(dict(used) == counts == receipt["physical_counts"], "Per-arm physical count mismatch")
    require(receipt["total_physical_calls"] == 496, "Total physical count mismatch")
    require(receipt["logical_calls_per_arm"] == {arm: 172 for arm in ARMS},
            "Logical budget mismatch")
    require(receipt["selected"] == selected, "Selected candidate receipts differ from rebuild")
    return selected


def tensor(array):
    return torch.from_numpy(np.asarray(array).copy())


def inspect_trace(arrays, prefix, code, count, permutation=None, identities=None):
    stem = prefix + "_"
    fields = ("pre_input", "ci_input", "raw_gate", "gate", "invocation")
    values = {field: tensor(arrays[stem + field]) for field in fields}
    for field in fields[:-1]:
        value = values[field]
        require(tuple(value.shape) == (3, count, 16), f"{prefix}/{field} shape mismatch")
        require(bool(torch.isfinite(value).all()), f"{prefix}/{field} contains nonfinite data")
    require(torch.equal(values["pre_input"], values["ci_input"]),
            f"{prefix}: CI inputs differ from reached inputs")
    inv = values["invocation"]
    require(tuple(inv.shape) == (3, 4), f"{prefix}: invocation shape")
    require(inv[:, 0].tolist() == [0, 1, 2], f"{prefix}: pass counter mismatch")
    require(torch.equal(inv[:, 1:], inv[0, 1:].expand(3, 3)),
            f"{prefix}: physical wrapper or factor identities changed")
    these_ids = tuple(inv[0, 1:].tolist())
    if identities is not None:
        require(these_ids == identities, f"{prefix}: factor identities changed across rollouts")
    bias = torch.tensor(code).repeat_interleave(4).float() * 0.25
    expected_gate = (values["raw_gate"] + bias).clamp(0, 1)
    if permutation is not None:
        expected_gate = expected_gate[..., permutation]
    require(torch.equal(expected_gate, values["gate"]),
            f"{prefix}: deployed gate differs from recorded CI and selected code")
    return values, these_ids


def native_replay(features, weights, v, u, gates=None, native=False):
    """Explicit three-pass source recurrence, using archived gates for BPD."""
    bits, query, noise = features
    bit = bits.float() * 2 - 1
    z, one = torch.zeros_like(bit), torch.ones_like(bit)
    first = torch.stack((bit[:, 0] + noise[:, 0], one[:, 0], z[:, 0], one[:, 0]), -1)
    second = torch.stack((bit[:, 1] + noise[:, 1], z[:, 1], one[:, 1], one[:, 1]), -1)
    src = torch.stack((first, second), 1)
    n = len(bits)
    state = torch.zeros((n, 16), dtype=torch.float32)
    q = weights["reader.weight"][query]
    pres, states, logits, betas, writes = [], [], [], [], []
    for tau in range(3):
        written = F.linear(src, weights["writer.weight"], weights["writer.bias"])
        written = written + 0.12 * F.linear(state, weights["feedback.weight"]).unsqueeze(1)
        normalized = F.normalize(written, p=2, dim=-1) * math.sqrt(16)
        scores = (normalized * q[:, None, :]).sum(-1) / math.sqrt(16)
        gate = F.softmax(scores, dim=-1)
        message = (written * gate[:, :, None]).sum(1)
        write = F.linear(message, weights["mix.W_target"], weights["mix.bias"]) if native else (
            ((message @ v) * gates[tau]) @ u + weights["mix.bias"])
        state = torch.tanh(0.35 * F.linear(state, weights["core.weight"], weights["core.bias"]) + write)
        pres.append(message)
        states.append(state)
        writes.append(write)
        logits.append(F.linear(state, weights["head.weight"], weights["head.bias"]))
        betas.append(gate[:, 1])
    return dict(pre_input=torch.stack(pres), states=torch.stack(states),
                intermediate_logits=torch.stack(logits), logits=logits[-1],
                beta3=torch.stack(betas), writes=torch.stack(writes))


def max_error(actual, expected, label, tolerance=2e-5):
    require(actual.shape == expected.shape, f"{label}: shape mismatch")
    error = float((actual - expected).abs().max())
    require(math.isfinite(error) and error <= tolerance,
            f"{label}: max error {error} exceeds {tolerance}")
    return error


def summary(value):
    flat = value.double().flatten()
    return dict(mean=float(flat.mean()), rms=float(flat.square().mean().sqrt()),
                maximum=float(flat.max()), minimum=float(flat.min()))


def check_summary(recorded, actual, label):
    for field, expected in summary(actual).items():
        near(recorded[field], expected, f"{label}/{field}")


def audit_root(path):
    result = read_json(path / "result.json")
    require(path.name == f'root_{result["root"]}', "Root directory and assigned seed disagree")
    receipt = read_json(path / "search.json")
    audit = read_json(path / "audit.json")
    training = read_json(path / "training.json")
    events = read_json(path / "events.json")
    require(result["audit"] == audit, "Result and audit JSON disagree")
    require(result["evidence_class"] == "PILOT_ONLY_ENGINEERING", "Unexpected evidence class")
    hashes = result["artifact_hashes"]
    require(REQUIRED_HASHED.issubset(hashes), "Missing required artifact hashes")
    for name, expected in hashes.items():
        target = (path / name).resolve()
        require(target.parent == path.resolve(), f"Artifact hash path escapes root: {name}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        require(actual == expected, f"Artifact hash mismatch: {name}")
    selected = rebuild_search(receipt)
    selected_codes = {arm: selected[arm]["code"] for arm in ARMS}
    require(result["selected_codes"] == selected_codes, "Result selected codes mismatch")
    for arm in ARMS:
        near(result["selected_calibration_ce"][arm], selected[arm]["score"],
             f"{arm} calibration CE", 0)
    sequence = [row["event"] for row in events]
    required = ("calibration_labels_accessed", "all_policies_selected",
                "terminal_created_after_selection", "terminal_labels_accessed_after_selection")
    require(all(sequence.count(name) == 1 for name in required), "Event missing or repeated")
    indices = [sequence.index(name) for name in required]
    require(indices == sorted(indices), "Terminal data/labels preceded completed selection")
    selection_event = events[indices[1]]
    require(selection_event["codes"] == selected_codes and selection_event["candidate_calls"] == 496,
            "Selection event disagrees with search")
    cfg = training["config"]
    require(cfg["rollout_mode"] == "closed_loop", "Unexpected rollout mode")
    require(cfg["ablation"]["mode"] == "zero" and not cfg["ablation"]["sphere_rescale"],
            "Residual mean or sphere path enabled")
    require(cfg["coeff_adv"] > 0 and cfg["coeff_stoch"] > 0, "Training branch disabled")
    require(cfg["minimality"]["pool_mode"] == "max", "Temporal pooling differs from max")
    require(cfg["adversary"]["scope"] == "shared_batch_and_pass", "PGD sharing changed")
    require(training["pgd_steps"] == 240 and training["pgd_source_shapes"]["mix"] == [1, 1, 1, 17],
            "PGD execution receipt mismatch")
    require(all(float(training["parameter_change"][name]) > 0 for name in ("v_l2", "u_l2", "ci_l2")),
            "Learned parameter block has zero update")
    require(any(row["adv_loss"] > 0 for row in training["steps"]), "No nonzero adversarial loss")

    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = checkpoint["source_and_bpd"]
    weights = {key[len("target_model."):]: value for key, value in state.items()
               if key.startswith("target_model.")}
    v = checkpoint["quant_v_codes"].float() * checkpoint["scale_v"]
    u = checkpoint["quant_u_codes"].float() * checkpoint["scale_u"]
    fp_v, fp_u = checkpoint["fp_v"], checkpoint["fp_u"]
    for name, quant, full in (("v", v, fp_v), ("u", u, fp_u)):
        scale = float(full.abs().max()) / 3 if bool(full.abs().max()) else 1.0
        near(checkpoint[f"scale_{name}"], scale, f"Quantization scale {name}", 0)
        expected_codes = torch.round(full / scale).clamp(-3, 3).to(torch.int8)
        require(torch.equal(expected_codes, checkpoint[f"quant_{name}_codes"]),
                f"Quantization codes {name} differ")
    require(torch.equal(weights["mix.V"], v) and torch.equal(weights["mix.U"], u),
            "Checkpoint deployment factors are not recorded quantized factors")
    for field in ("V", "U", "W_target", "bias"):
        require(torch.equal(state[f"_registered_wrappers.mix.{field}"], weights[f"mix.{field}"]),
                f"Duplicate registered wrapper {field} differs")
    permutation = torch.tensor(result["operator"]["permutation"], dtype=torch.long)
    require(sorted(permutation.tolist()) == list(range(16)), "Invalid gauge permutation")

    identities = None
    largest_calibration_replay_error = 0.0
    generator = torch.Generator(device="cpu").manual_seed(result["root"] * 1000000 + 20000)
    calibration_bits = torch.randint(0, 2, (64, 2), generator=generator)
    calibration_query = torch.randint(0, 2, (64,), generator=generator)
    calibration_noise = torch.randn((64, 2), generator=generator) * 0.025
    calibration_features = (calibration_bits, calibration_query, calibration_noise)
    calibration_labels = calibration_bits.gather(1, calibration_query[:, None]).squeeze(1)
    with np.load(path / "calibration_traces.npz", allow_pickle=False) as data:
        require(len(data.files) == 496 * 5 + 4, "Unexpected calibration trace field count")
        for field, expected in zip(("bits", "query", "noise", "labels"),
                                   (*calibration_features, calibration_labels)):
            require(torch.equal(tensor(data[field]), expected),
                    f"Archived calibration {field} differs from assigned existing seed")
        for row in receipt["candidates"]:
            trace, identities = inspect_trace(data, row["id"], row["code"], 64,
                                               identities=identities)
            replay = native_replay(calibration_features, weights, v, u, trace["gate"])
            error = max_error(replay["pre_input"], trace["pre_input"],
                              f'{row["id"]} reached calibration message')
            largest_calibration_replay_error = max(largest_calibration_replay_error, error)
            score = float(F.cross_entropy(replay["logits"], calibration_labels))
            near(row["score"], score, f'{row["id"]} independently replayed CE')

    outcomes, replay_errors = {}, {}
    with np.load(path / "terminal_traces.npz", allow_pickle=False) as data:
        labels = tensor(data["labels"]).long()
        n = len(labels)
        require(n == 256, "Terminal size differs from fixed recipe")
        terminal_features = tuple(tensor(data[field]) for field in ("bits", "query", "noise"))
        require(torch.equal(labels, terminal_features[0].gather(
            1, terminal_features[1][:, None]).squeeze(1)), "Terminal labels differ from task rule")
        baseline = tensor(data["quantized_logits"]).argmax(-1) == labels
        for name, metric in result["metrics"].items():
            logits = tensor(data[name + "_logits"])
            inter = tensor(data[name + "_intermediate_logits"])
            require(bool(torch.isfinite(logits).all()) and tuple(logits.shape) == (n, 2),
                    f"{name}: invalid terminal logits")
            require(torch.equal(logits, inter[-1]), f"{name}: final and intermediate logits differ")
            correct = logits.argmax(-1) == labels
            require(torch.equal(correct, tensor(data[name + "_correct"])), f"{name}: saved bits differ")
            truth = dict(correct=int(correct.sum()), accuracy=float(correct.float().mean()),
                         recovery=int((~baseline & correct).sum()),
                         corruption=int((baseline & ~correct).sum()),
                         net_repair=int(correct.sum() - baseline.sum()))
            for field, expected in truth.items():
                near(metric[field], expected, f"{name}/{field}", 0)
            near(metric["ce"], float(F.cross_entropy(logits, labels)), f"{name}/CE")
            outcomes[name] = truth
            native = name in ("clean", "all_open_delta_one")
            vf, uf = (fp_v, fp_u) if name in ("full_precision_bpd", "fp_all_open") else (v, u)
            if name == "gauge":
                vf, uf = v[:, permutation], u[permutation]
            if name + "_raw_gate" in data.files:
                code = selected_codes.get(name, list(ZERO))
                perm = permutation if name in ("gauge", "ci_permutation_unmatched") else None
                trace, identities = inspect_trace(data, name, code, n, perm, identities)
                gates = trace["gate"]
                raw_reference = trace["raw_gate"].clamp(0, 1)
                if name == "gauge":
                    raw_reference = raw_reference[..., permutation]
                x = trace["pre_input"]
                actual = ((x @ vf) * gates) @ uf + weights["mix.bias"]
                unchanged = ((x @ vf) * raw_reference) @ uf + weights["mix.bias"]
                dose = (actual - unchanged).norm(dim=-1)
                if name in ("quantized", "full_precision_bpd", "gauge"):
                    require(torch.equal(dose, torch.zeros_like(dose)),
                            f"{name}: zero-code local write dose is nonzero")
                target_here = F.linear(x, weights["mix.W_target"], weights["mix.bias"])
                target_dose = (actual - target_here).norm(dim=-1)
                for tau, row in enumerate(metric["per_pass"]):
                    check_summary(row["local_write_displacement_vs_unbiased_gate"], dose[tau],
                                  f"{name}/pass{tau}/unbiased dose")
                    check_summary(row["local_write_displacement_vs_clean_operator"], target_dose[tau],
                                  f"{name}/pass{tau}/target dose")
            else:
                gates = torch.ones((3, n, 16))
                trace = None
            replay = native_replay(terminal_features, weights, vf, uf, gates, native=native)
            errors = {}
            for field in ("logits", "intermediate_logits", "states", "beta3"):
                errors[field] = max_error(replay[field], tensor(data[name + "_" + field]),
                                          f"{name}/{field} replay")
            if trace is not None:
                errors["pre_input"] = max_error(replay["pre_input"], trace["pre_input"],
                                               f"{name} reached message")
            replay_errors[name] = errors
        parity = float((tensor(data["all_open_delta_one_intermediate_logits"]) -
                        tensor(data["clean_intermediate_logits"])).abs().max())
        gauge = float((tensor(data["gauge_intermediate_logits"]) -
                       tensor(data["quantized_intermediate_logits"])).abs().max())
        near(audit["all_open_delta_one_max_head_error"], parity, "Recorded full-head parity", 0)
        near(audit["simultaneous_factor_ci_permutation_max_head_error"], gauge, "Recorded gauge parity", 0)
        require(audit["full_operator_parity_pass"] == (parity <= 2e-5), "Parity flag differs")
        require(audit["gauge_identity_pass"] == (gauge <= 2e-5), "Gauge flag differs")
        require(audit["baseline_error_headroom"] == int((~baseline).sum()), "Headroom count differs")
        expected_status = "AVAILABLE" if bool((~baseline).any()) else "INSTRUMENT_HEADROOM_ABSENT"
        require(audit["headroom_status"] == expected_status, "Headroom flag differs")
        gap = float((tensor(data["quantized_pre_input"]) -
                     tensor(data["all_open_delta_one_pre_input"])).abs().max())
        near(audit["online_reached_vs_clean_input_max_difference"], gap, "Reached-state gap", 0)
        agreement = float((tensor(data["full_precision_bpd_logits"]).argmax(-1) ==
                           tensor(data["clean_logits"]).argmax(-1)).float().mean())
        near(audit["full_precision_online_clean_class_agreement"], agreement, "FP online fidelity", 0)
        source_accuracy = training["source"]["source_validation_accuracy"]
        status = ("PASS" if source_accuracy >= 0.95 and agreement >= 0.95
                  else "LOW_TASK_FIDELITY_RETAINED")
        require(audit["task_fidelity_status"] == status, "Fidelity status mismatch")
    return dict(root=result["root"], status="PASS", artifact_hashes_verified=len(hashes),
                physical_calls=496, logical_calls_per_arm=172, retentions_rebuilt=44,
                calibration_rollouts_replayed=496, terminal_rollouts_replayed=len(outcomes),
                calibration_max_pre_input_error=largest_calibration_replay_error,
                terminal_replay_errors=replay_errors, selected_codes=selected_codes,
                outcomes=outcomes, parity_max=parity, gauge_max=gauge,
                headroom=int((~baseline).sum()), reached_input_gap=gap,
                result_sha256=hashlib.sha256((path / "result.json").read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base = args.run_directory.resolve()
    roots = [base] if (base / "result.json").is_file() else sorted(base.glob("root_*"))
    require(bool(roots), "No existing result roots found")
    reports = []
    for root in roots:
        try:
            reports.append(audit_root(root))
        except Exception as exc:
            reports.append(dict(root=root.name, status="FAIL",
                                exception=type(exc).__name__, error=str(exc)))
    aggregate_status = "NOT_PRESENT"
    aggregate_file = base / "aggregate.json"
    if aggregate_file.is_file() and all(row["status"] == "PASS" for row in reports):
        aggregate = read_json(aggregate_file)
        require(aggregate["roots"] == [row["root"] for row in reports], "Aggregate root list differs")
        for arm, metric in aggregate["arms"].items():
            for field in ("correct", "recovery", "corruption", "net_repair"):
                total = sum(row["outcomes"][arm][field] for row in reports)
                require(metric[field] == total, f"Aggregate {arm}/{field} differs")
            near(metric["accuracy"], sum(row["outcomes"][arm]["correct"] for row in reports) /
                 (256 * len(reports)), f"Aggregate {arm}/accuracy", 0)
        aggregate_status = "PASS"
    report = dict(status="PASS" if all(row["status"] == "PASS" for row in reports) else "FAIL",
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  independent_of_experiment_imports=True, new_roots_generated=0,
                  note="CI outputs are checked against archived reached inputs and used gates; "
                       "the CI transformer is not independently reimplemented.",
                  aggregate=aggregate_status, roots=reports)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

