"""BPD-RP-01. Fresh synthetic roots only; all prior scientific seals remain closed.

Run as a module from the repository root. ``--phase pilot`` requires a receipt
whose source_hashes match these files and whose remote_commit was supplied by
the parent workflow after publication. The receipt supplies provenance, never
scientific confirmation authorization.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Callable

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel, PassMaskSchedule
from mechinterp.bpd.config import (AblationConfig, AdversaryConfig, BPDConfig,
                                  CIConfig, MinimalityConfig, NullCalibrationConfig)
from mechinterp.bpd.trainer import BPDTrainer
from mechinterp.bpd.experiments.inkling_routing.native_model import make_batch, train_source

HERE = Path(__file__).resolve().parent
ZERO_CODE = (0, 0, 0, 0)
ARMS = ("greedy", "multistart", "beam")
PILOT_ROOTS = (17001, 17002, 17003, 17004)
SMOKE_ROOT = 17000


@dataclass(frozen=True)
class Recipe:
    source_steps: int = 260
    bpd_steps: int = 80
    source_batch: int = 192
    bpd_batch: int = 32
    calibration_n: int = 64
    terminal_n: int = 256
    width: int = 16
    passes: int = 3
    components: int = 16
    groups: int = 4
    gate_step: float = 0.25
    code_bound: int = 2
    main_lr: float = 0.001
    coeff_faith: float = 100000.0
    coeff_stoch: float = 0.5
    coeff_adv: float = 0.5
    pgd_inner_steps: int = 2
    threads: int = 1


@dataclass
class Features:
    """Only model inputs are exposed to rollouts and proposal generation."""
    bits: Tensor
    query: Tensor
    noise: Tensor

    @property
    def labels(self):
        raise RuntimeError("Label access is forbidden inside feature-only rollouts")


def features(batch) -> Features:
    return Features(batch.bits, batch.query, batch.noise)


def digest_tensor(tensor: Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_hashes() -> dict[str, str]:
    paths = [HERE / name for name in ("experiment.py", "test_experiment.py", "PROTOCOL.md")]
    paths += [HERE.parents[1] / name for name in (
        "__init__.py", "analysis.py", "components.py", "config.py", "ci.py", "trainer.py", "adversary.py", "losses.py", "nulls.py")]
    paths += [HERE.parent / "inkling_routing" / "native_model.py"]
    return {str(p.relative_to(HERE.parents[3])): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def quantize_three_bit(tensor: Tensor) -> tuple[Tensor, Tensor, float]:
    maximum = float(tensor.abs().max().item())
    scale = maximum / 3.0 if maximum else 1.0
    codes = torch.round(tensor / scale).clamp(-3, 3).to(torch.int8)
    return codes.float() * scale, codes, scale


def proposals(code: tuple[int, ...]) -> list[dict]:
    """Nine fixed slots. This function accepts codes alone, never task labels."""
    if len(code) != 4 or any(type(x) is not int or abs(x) > 2 for x in code):
        raise ValueError("A code has four integers in [-2,2]")
    out = [{"code": code, "slot": 0, "edit": "noop", "padded": False}]
    for group in range(4):
        for change in (1, -1):
            candidate = list(code)
            candidate[group] += change
            invalid = abs(candidate[group]) > 2
            out.append({"code": code if invalid else tuple(candidate), "slot": len(out),
                        "edit": f"group_{group}:{change:+d}", "padded": invalid})
    return out


def retain(candidates: list[dict], width: int) -> list[dict]:
    """Stable raw-score ties preserve input order. Dedup preserves best copy."""
    kept, seen = [], set()
    for row in sorted(candidates, key=lambda item: item["score"]):
        code = tuple(row["code"])
        if code not in seen:
            kept.append(row)
            seen.add(code)
        if len(kept) == width:
            break
    return kept


def search(score_fn: Callable[[tuple[int, ...], str], float]) -> dict:
    """A shared initial evaluation/bootstrap, then three budget-matched arms."""
    candidates, retentions = [], []
    counts = {"common": 0, **{arm: 0 for arm in ARMS}}

    def evaluate(code, arm, parent, stage, slot=0, edit="initial", padded=False):
        candidate_id = f"c{len(candidates):04d}"
        score = float(score_fn(tuple(code), candidate_id))
        if not math.isfinite(score):
            raise RuntimeError("Non-finite calibration score")
        row = dict(id=candidate_id, code=list(code), score=score, arm=arm,
                   parent=parent, stage=stage, slot=slot, edit=edit, padded=padded)
        candidates.append(row)
        counts[arm] += 1
        return row

    def expand(parent, arm, stage):
        return [evaluate(row["code"], arm, parent["id"], stage,
                         row["slot"], row["edit"], row["padded"])
                for row in proposals(tuple(parent["code"]))]

    def selection(pool, width, arm, stage):
        chosen = retain(pool, width)
        retentions.append(dict(arm=arm, stage=stage, pool=[c["id"] for c in pool],
                               selected=[c["id"] for c in chosen]))
        return chosen

    initial = evaluate(ZERO_CODE, "common", None, "initial")
    boot = expand(initial, "common", "bootstrap")
    starters = selection([initial] + boot, 3, "common", "bootstrap")

    greedy = starters[0]
    for step in range(18):
        greedy = selection([greedy] + expand(greedy, "greedy", step), 1,
                           "greedy", step)[0]

    paths = list(starters)
    for step in range(6):
        for branch in range(3):
            parent = paths[branch]
            stage = f"round_{step}:path_{branch}"
            paths[branch] = selection([parent] + expand(parent, "multistart", stage), 1,
                                      "multistart", stage)[0]
    multistart = selection(paths, 1, "multistart", "terminal")[0]

    beam = list(starters)
    for step in range(6):
        pool = list(beam)
        for parent in tuple(beam):
            pool.extend(expand(parent, "beam", step))
        beam = selection(pool, 3, "beam", step)
    chosen = {"greedy": greedy, "multistart": multistart, "beam": beam[0]}
    assert counts == {"common": 10, "greedy": 162, "multistart": 162, "beam": 162}
    return dict(candidates=candidates, retentions=retentions, selected=chosen,
                physical_counts=counts, total_physical_calls=sum(counts.values()),
                logical_calls_per_arm={arm: counts["common"] + counts[arm] for arm in ARMS},
                tie_policy="stable raw-score order; parents precede children; earlier slot wins")


class Deployment:
    """Online CI from each reached message with persistent physical V/U identities."""
    def __init__(self, model, bpd, trainer, recipe):
        self.model, self.bpd, self.trainer, self.recipe = model, bpd, trainer, recipe
        self.wrapper = bpd.wrappers["mix"]
        self.active = False
        self.trace = []
        self.hook = self.wrapper.register_forward_pre_hook(self._online_hook)

    def _online_hook(self, module, args):
        if not self.active:
            return
        x = args[0]  # Shape adapter already ran: [B,1,16].
        tau = module.calls
        ci_input = x.unsqueeze(1)  # [B,1,1,16], reached input only.
        g = self.trainer.ci_net({"mix": ci_input}, pass_offset=tau).lower_leaky["mix"][:, 0]
        expanded_bias = torch.tensor(self.code, dtype=x.dtype).repeat_interleave(4)
        original_g = (g + self.recipe.gate_step * expanded_bias).clamp(0, 1)
        used_g = original_g if self.gate_permutation is None else original_g[..., self.gate_permutation]
        module.schedule.comp_mask[:, tau] = used_g
        if self.record:
            self.trace.append(dict(pre_input=x.detach().clone(),
                                   ci_input=ci_input[:, 0].detach().clone(),
                                   raw_gate=g.detach().clone(), gate=used_g.detach().clone(),
                                   pass_index=tau, wrapper_id=id(module),
                                   v_id=id(module.V), u_id=id(module.U)))

    @torch.no_grad()
    def rollout(self, batch: Features, code=ZERO_CODE, *, record=False,
                gate_permutation=None, all_open=False, delta=0.0):
        if not isinstance(batch, Features):
            raise TypeError("Deployment accepts only Features, which blocks label access")
        b, r = batch.bits.shape[0], self.recipe
        schedule = PassMaskSchedule(torch.ones((b, r.passes, 1, r.components)),
                                    torch.full((b, r.passes, 1), float(delta)))
        self.code, self.record = tuple(code), record
        self.gate_permutation, self.trace = gate_permutation, []
        self.active = not all_open
        try:
            with self.bpd.masked_components({"mix": schedule}, record_inputs=record):
                result = self.model(batch)
                calls = self.wrapper.calls
                captured = self.bpd.collect_recorded_inputs(r.passes)["mix"] if record else None
        finally:
            self.active = False
        if calls != r.passes:
            raise AssertionError("Shared mix was not invoked at every pass")
        trace = {}
        if record:
            if all_open:
                trace = {"pre_input": captured[:, :, 0].transpose(0, 1).clone(),
                         "gate": schedule.comp_mask[:, :, 0].transpose(0, 1).clone()}
            else:
                for field in ("pre_input", "ci_input", "raw_gate", "gate"):
                    trace[field] = torch.stack([row[field][:, 0] for row in self.trace])
                if not torch.equal(trace["pre_input"], trace["ci_input"]):
                    raise AssertionError("CI input differs from reached pre-mix input")
                if not torch.equal(trace["pre_input"], captured[:, :, 0].transpose(0, 1)):
                    raise AssertionError("Hook trace differs from wrapper-captured inputs")
                trace["invocation"] = torch.tensor([[row[k] for k in
                    ("pass_index", "wrapper_id", "v_id", "u_id")] for row in self.trace])
                if len(set(row["v_id"] for row in self.trace)) != 1 or len(set(row["u_id"] for row in self.trace)) != 1:
                    raise AssertionError("Factors changed identity across passes")
        return result, trace


def configuration(seed, recipe):
    r = recipe
    return BPDConfig(c_per_module={"mix": r.components}, n_passes=r.passes,
        n_steps=r.bpd_steps, batch_size=r.bpd_batch, seq_len=1, seed=seed + 500000,
        faithfulness_warmup_steps=0, main_lr=r.main_lr, coeff_faith=r.coeff_faith,
        coeff_stoch=r.coeff_stoch, coeff_adv=r.coeff_adv, rollout_mode="closed_loop",
        ci=CIConfig(d_model=16, n_blocks=1, n_heads=2, mlp_hidden=32,
                    max_passes=3, max_seq_len=16, variant="state_only"),
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        minimality=MinimalityConfig(pool_mode="max"),
        adversary=AdversaryConfig(scope="shared_batch_and_pass", inner_steps=r.pgd_inner_steps),
        null_calibration=NullCalibrationConfig(use_tolerance_weights=False))


def fit(seed, recipe):
    r = recipe
    model, source = train_source(seed=seed, steps=r.source_steps, width=r.width,
        loops=r.passes, batch_size=r.source_batch, threads=r.threads)
    cfg = configuration(seed, r)
    torch.manual_seed(cfg.seed)
    bpd = BPDModel(model, cfg.c_per_module, max_passes=r.passes, ablation=cfg.ablation)
    wrapper = bpd.wrappers["mix"]
    # Preserve the native forward while giving the component wrapper its P axis.
    wrapper.register_forward_pre_hook(lambda module, args: (args[0].unsqueeze(1),))
    wrapper.register_forward_hook(lambda module, args, output: output.squeeze(1))

    def adapter(batch, n_passes, pass_offset, init_state):
        if n_passes != r.passes or pass_offset != 0 or init_state is not None:
            raise ValueError("RP-01 only supports full closed-loop trajectories")
        output = model(batch)
        return ({i: output["intermediate_logits"][i, :, None] for i in range(r.passes)},
                {i: output["states"][i, :, None] for i in range(r.passes)}, None, None)

    trainer = BPDTrainer(bpd, cfg, adapter, device="cpu")
    initial_v = wrapper.V.detach().clone()
    initial_u = wrapper.U.detach().clone()
    initial_ci = {name: param.detach().clone() for name, param in trainer.ci_net.named_parameters()}
    log = []
    for step in range(r.bpd_steps):
        batch = features(make_batch(seed * 1000000 + 1000 + step, r.bpd_batch))
        log.append(asdict(trainer.train_step(batch, step)))
    for parameter in bpd.parameters():
        parameter.requires_grad_(False)
    for parameter in trainer.ci_net.parameters():
        parameter.requires_grad_(False)
    trainer.ci_net.eval()
    changes = dict(v_l2=float((wrapper.V - initial_v).norm()),
                   u_l2=float((wrapper.U - initial_u).norm()),
                   ci_l2=math.sqrt(sum(float((p-initial_ci[n]).square().sum())
                       for n, p in trainer.ci_net.named_parameters())))
    training = dict(source=source, config=asdict(cfg), steps=log, parameter_change=changes,
        ci_training="Reference-conditioned: native BPDTrainer derives gates from clean trajectory inputs",
        pgd_steps=trainer.pgd.step_count,
        pgd_source_shapes={k: list(v.shape) for k, v in trainer.pgd.sources.items()},
        pgd_sources={k: v.detach().flatten().tolist() for k, v in trainer.pgd.sources.items()})
    if trainer.pgd.step_count != r.bpd_steps * (1 + r.pgd_inner_steps):
        raise AssertionError("Persistent adversary did not execute prescribed updates")
    return model, bpd, trainer, training


def summarize_tensor(x: Tensor) -> dict:
    flat = x.detach().double().flatten()
    return dict(mean=float(flat.mean()), rms=float(flat.square().mean().sqrt()),
                maximum=float(flat.max()), minimum=float(flat.min()))


def per_pass_metrics(output, clean, trace, quant_v, quant_u, bias, target_weight,
                     reference_gate_permutation=None):
    rows = []
    for tau in range(output["states"].shape[0]):
        state, ref = output["states"][tau], clean["states"][tau]
        logits, clean_logits = output["intermediate_logits"][tau], clean["intermediate_logits"][tau]
        row = dict(pass_index=tau, state_norm=summarize_tensor(state.norm(dim=-1)),
            state_rms=summarize_tensor(state.square().mean(-1).sqrt()),
            state_displacement_vs_clean=summarize_tensor((state-ref).norm(dim=-1)),
            all_head_max_abs_error=float((logits-clean_logits).abs().max()),
            all_head_rms_error=float((logits-clean_logits).square().mean().sqrt()),
            predicted_class_agreement=float((logits.argmax(-1)==clean_logits.argmax(-1)).float().mean()),
            clean_to_output_kl=float(F.kl_div(F.log_softmax(logits,-1),
                F.softmax(clean_logits,-1), reduction="batchmean")))
        if "raw_gate" in trace:
            x = trace["pre_input"][tau]
            # Local physical displacement on this arm's reached x; source operator
            # comparison includes the native bias, which cancels in subtraction.
            actual = ((x @ quant_v) * trace["gate"][tau]) @ quant_u + bias
            reference_gate = trace["raw_gate"][tau].clamp(0, 1)
            if reference_gate_permutation is not None:
                reference_gate = reference_gate[..., reference_gate_permutation]
            unchanged_here = ((x @ quant_v) * reference_gate) @ quant_u + bias
            row["local_write_displacement_vs_unbiased_gate"] = summarize_tensor((actual-unchanged_here).norm(dim=-1))
            target_here = F.linear(x, target_weight, bias)
            row["local_write_displacement_vs_clean_operator"] = summarize_tensor((actual-target_here).norm(dim=-1))
            row["local_write_norm"] = summarize_tensor(actual.norm(dim=-1))
            row["gate_sum"] = summarize_tensor(trace["gate"][tau].sum(-1))
        rows.append(row)
    return rows


def execute_root(seed: int, out: Path, recipe: Recipe) -> dict:
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    events = [{"event": "training_start", "root": seed}]
    model, bpd, trainer, training = fit(seed, recipe)
    wrapper = bpd.wrappers["mix"]
    fp_v, fp_u = wrapper.V.detach().clone(), wrapper.U.detach().clone()
    quant_v, codes_v, scale_v = quantize_three_bit(fp_v)
    quant_u, codes_u, scale_u = quantize_three_bit(fp_u)
    deployment = Deployment(model, bpd, trainer, recipe)

    def set_factors(v, u):
        with torch.no_grad():
            wrapper.V.copy_(v)
            wrapper.U.copy_(u)

    calibration_native = make_batch(seed * 1000000 + 20000, recipe.calibration_n)
    calibration = features(calibration_native)
    calibration_labels = calibration_native.labels.clone()
    events.append({"event": "calibration_labels_accessed", "n": recipe.calibration_n})
    set_factors(quant_v, quant_u)
    calibration_trace = {"bits": calibration.bits.numpy(), "query": calibration.query.numpy(),
                         "noise": calibration.noise.numpy(), "labels": calibration_labels.numpy()}
    before = {"v": digest_tensor(wrapper.V), "u": digest_tensor(wrapper.U),
              **{f"ci:{n}": digest_tensor(p) for n, p in trainer.ci_net.named_parameters()}}

    def score(code, candidate_id):
        output, trace = deployment.rollout(calibration, code, record=True)
        for key, tensor in trace.items():
            calibration_trace[f"{candidate_id}_{key}"] = tensor.cpu().numpy()
        return F.cross_entropy(output["logits"], calibration_labels).item()

    search_receipt = search(score)
    selected = {arm: tuple(search_receipt["selected"][arm]["code"]) for arm in ARMS}
    events.append({"event": "all_policies_selected", "codes": selected,
                   "candidate_calls": search_receipt["total_physical_calls"]})
    after = {"v": digest_tensor(wrapper.V), "u": digest_tensor(wrapper.U),
             **{f"ci:{n}": digest_tensor(p) for n, p in trainer.ci_net.named_parameters()}}
    if before != after:
        raise AssertionError("Parameters changed during calibration search")
    # Terminal generation is deliberately below the complete policy selection.
    terminal_native = make_batch(seed * 1000000 + 30000, recipe.terminal_n)
    terminal = features(terminal_native)
    events.append({"event": "terminal_created_after_selection", "n": recipe.terminal_n})
    terminal_labels = terminal_native.labels.clone()
    events.append({"event": "terminal_labels_accessed_after_selection"})
    terminal_arrays = {"labels": terminal_labels.numpy(), "bits": terminal.bits.numpy(),
                       "query": terminal.query.numpy(), "noise": terminal.noise.numpy()}
    with torch.no_grad(), bpd.target_capture():
        clean = model(terminal)
    outputs, traces = {"clean": clean}, {}
    # Algebraic full-operator parity is distinct from residual-disabled fidelity.
    outputs["all_open_delta_one"], traces["all_open_delta_one"] = deployment.rollout(
        terminal, record=True, all_open=True, delta=1.0)
    outputs["quantized_all_open"], traces["quantized_all_open"] = deployment.rollout(
        terminal, record=True, all_open=True, delta=0.0)
    outputs["quantized"], traces["quantized"] = deployment.rollout(terminal, record=True)
    for arm in ARMS:
        outputs[arm], traces[arm] = deployment.rollout(terminal, selected[arm], record=True)
    set_factors(fp_v, fp_u)
    outputs["full_precision_bpd"], traces["full_precision_bpd"] = deployment.rollout(terminal, record=True)
    outputs["fp_all_open"], traces["fp_all_open"] = deployment.rollout(terminal, record=True, all_open=True)

    permutation = torch.randperm(recipe.components, generator=torch.Generator().manual_seed(seed+800000))
    set_factors(quant_v[:, permutation], quant_u[permutation])
    outputs["gauge"], traces["gauge"] = deployment.rollout(terminal, record=True, gate_permutation=permutation)
    set_factors(quant_v, quant_u)
    outputs["ci_permutation_unmatched"], traces["ci_permutation_unmatched"] = deployment.rollout(
        terminal, record=True, gate_permutation=permutation)

    metrics = {}
    baseline_bits = outputs["quantized"]["logits"].argmax(-1) == terminal_labels
    for name, output in outputs.items():
        correct = output["logits"].argmax(-1) == terminal_labels
        terminal_arrays[f"{name}_correct"] = correct.numpy()
        for key in ("logits", "intermediate_logits", "states", "beta3"):
            terminal_arrays[f"{name}_{key}"] = output[key].detach().numpy()
        for key, value in traces.get(name, {}).items():
            terminal_arrays[f"{name}_{key}"] = value.numpy()
        v, u = (fp_v, fp_u) if name in ("full_precision_bpd", "fp_all_open") else (quant_v, quant_u)
        if name == "gauge":
            v, u = quant_v[:, permutation], quant_u[permutation]
        metrics[name] = dict(accuracy=float(correct.float().mean()), correct=int(correct.sum()),
            ce=float(F.cross_entropy(output["logits"], terminal_labels)),
            recovery=int((~baseline_bits & correct).sum()),
            corruption=int((baseline_bits & ~correct).sum()),
            net_repair=int(correct.sum()-baseline_bits.sum()),
            per_pass=per_pass_metrics(output, clean, traces.get(name, {}), v, u,
                                      wrapper.bias, wrapper.W_target,
                                      permutation if name == "gauge" else None))
    parity_max = float((outputs["all_open_delta_one"]["intermediate_logits"]-clean["intermediate_logits"]).abs().max())
    gauge_max = float((outputs["gauge"]["intermediate_logits"]-outputs["quantized"]["intermediate_logits"]).abs().max())
    online_gap = float((traces["quantized"]["pre_input"]-traces["all_open_delta_one"]["pre_input"]).abs().max())
    fp_agreement = float((outputs["full_precision_bpd"]["logits"].argmax(-1)==clean["logits"].argmax(-1)).float().mean())
    task_fidelity_ok = training["source"]["source_validation_accuracy"] >= .95 and fp_agreement >= .95
    audit = dict(all_open_delta_one_max_head_error=parity_max,
        full_operator_parity_pass=parity_max <= 2e-5,
        simultaneous_factor_ci_permutation_max_head_error=gauge_max,
        gauge_identity_pass=gauge_max <= 2e-5,
        online_ci_exact_input_audit=True, online_reached_vs_clean_input_max_difference=online_gap,
        shared_factor_identity_all_rollouts=True, frozen_during_search=before == after,
        pgd_update_count=training["pgd_steps"], pgd_expected_update_count=recipe.bpd_steps*3,
        nonzero_adversary_executed=training["pgd_steps"] > 0 and any(x["adv_loss"]>0 for x in training["steps"]),
        residual_policy="delta_mask=0 with zero ablation removes target residual exactly",
        proposal_label_access="proposals accepts four integer codes; deployment Features.labels raises",
        terminal_generation_follows_all_selection=True,
        source_validation_accuracy=training["source"]["source_validation_accuracy"],
        full_precision_online_clean_class_agreement=fp_agreement,
        task_fidelity_threshold=.95,
        task_fidelity_status="PASS" if task_fidelity_ok else "LOW_TASK_FIDELITY_RETAINED",
        baseline_error_headroom=int((~baseline_bits).sum()),
        headroom_status="AVAILABLE" if (~baseline_bits).any() else "INSTRUMENT_HEADROOM_ABSENT",
        physical_calibration_calls=496, logical_calibration_calls_per_arm=172,
        terminal_diagnostic_rollout_calls=len(outputs),
        diagnostic_calls_separate_from_search_budget=True)
    audit["engineering_identities_pass"] = (audit["full_operator_parity_pass"] and audit["gauge_identity_pass"])
    operator = dict(target_frobenius=float(wrapper.W_target.norm()),
        fp_factor_frobenius_error=float(((fp_v@fp_u).T-wrapper.W_target).norm()),
        quantized_factor_frobenius_error=float(((quant_v@quant_u).T-wrapper.W_target).norm()),
        quantization_operator_delta_frobenius=float((quant_v@quant_u-fp_v@fp_u).norm()),
        quantization_factor_delta_l2=math.sqrt(float((quant_v-fp_v).square().sum()+(quant_u-fp_u).square().sum())),
        quantization_scale_v=scale_v, quantization_scale_u=scale_u,
        residual_disabled=True, code_min=-3, code_max=3,
        permutation=permutation.tolist())
    np.savez_compressed(out / "calibration_traces.npz", **calibration_trace)
    np.savez_compressed(out / "terminal_traces.npz", **terminal_arrays)
    torch.save(dict(source_and_bpd=bpd.state_dict(), ci=trainer.ci_net.state_dict(),
                    fp_v=fp_v, fp_u=fp_u, quant_v_codes=codes_v, quant_u_codes=codes_u,
                    scale_v=scale_v, scale_u=scale_u), out / "checkpoint.pt")
    dump_json(out / "training.json", training)
    dump_json(out / "search.json", search_receipt)
    dump_json(out / "audit.json", audit)
    dump_json(out / "events.json", events)
    artifact_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in out.iterdir() if path.is_file()}
    result = dict(root=seed, evidence_class="PILOT_ONLY_ENGINEERING", metrics=metrics,
        selected_codes=selected, selected_calibration_ce={arm: search_receipt["selected"][arm]["score"] for arm in ARMS},
        operator=operator, audit=audit, artifact_hashes=artifact_hashes,
        elapsed_seconds=time.monotonic()-started)
    dump_json(out / "result.json", result)
    return result


def aggregate(results):
    arms = ("clean", "full_precision_bpd", "quantized", *ARMS, "ci_permutation_unmatched")
    n = 256 * len(results)
    return dict(evidence_class="PILOT_ONLY_ENGINEERING", independent_unit="independently trained source model seed",
        roots=[row["root"] for row in results], all_roots_retained=True,
        within_root_terminal_cases=256, total_terminal_cases=n,
        populations_inference="No confidence interval, p-value, power, or confirmation claim",
        arms={arm: dict(correct=sum(r["metrics"][arm]["correct"] for r in results),
                        accuracy=sum(r["metrics"][arm]["correct"] for r in results)/n,
                        recovery=sum(r["metrics"][arm]["recovery"] for r in results),
                        corruption=sum(r["metrics"][arm]["corruption"] for r in results),
                        net_repair=sum(r["metrics"][arm]["net_repair"] for r in results)) for arm in arms},
        roots_with_error_headroom=sum(r["audit"]["baseline_error_headroom"]>0 for r in results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "pilot", "hashes"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-lock", type=Path)
    args = parser.parse_args()
    hashes = source_hashes()
    if args.phase == "hashes":
        print(json.dumps(hashes, indent=2, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output required")
    if args.phase == "pilot":
        if args.source_lock is None:
            parser.error("Pilot requires the parent's verified remote source lock")
        receipt = json.loads(args.source_lock.read_text())
        if receipt.get("source_hashes") != hashes or not receipt.get("remote_commit"):
            raise RuntimeError("Missing remote commit or source hash mismatch")
        if receipt.get("evidence_class") != "PILOT_ONLY_ENGINEERING":
            raise RuntimeError("RP-01 is restricted to engineering pilot evidence")
    recipe = Recipe()
    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    roots = (SMOKE_ROOT,) if args.phase == "smoke" else PILOT_ROOTS
    args.output.mkdir(parents=True, exist_ok=False)
    dump_json(args.output / "run_manifest.json", dict(phase=args.phase, recipe=asdict(recipe),
        source_hashes=hashes, roots=roots, python=platform.python_version(),
        torch=torch.__version__, numpy=np.__version__, source_lock=str(args.source_lock)))
    results = []
    for seed in roots:
        result = execute_root(seed, args.output / f"root_{seed}", recipe)
        results.append(result)
        print(json.dumps(dict(root=seed, elapsed_seconds=result["elapsed_seconds"],
            codes=result["selected_codes"], accuracy={a: result["metrics"][a]["accuracy"]
            for a in ("clean", "full_precision_bpd", "quantized", *ARMS)}, audit=result["audit"])), flush=True)
    dump_json(args.output / "aggregate.json", aggregate(results))


if __name__ == "__main__":
    main()
