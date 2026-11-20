"""CPU instruments for actual BPD masks and a miniature native TRM recurrence.

Every initialization is an excluded engineering fixture. The four-token input
has no Sudoku score. Native operations use float32; saved matrix calculations
use float64 and carry no interval-certified roundoff guarantee.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

from .math import (
    matrix_inf_norm,
    ordered_product,
    trace_logdet_diagnostic,
    transport_diagnostic,
    walsh_diagnostic,
)


SOURCE_COMMIT = "420631c8835475c8b7a1f5f55f01a40ed4fb543e"
SEED_BASE = 2026112100
MODULE = "layers.0.down_proj"
MODEL_RECIPE = dict(hidden_size=8, num_heads=1, l_layers=1, h_cycles=1,
                    l_cycles=1, puzzle_emb_len=1, seq_len=4)
ZERO_COUNTERS = dict(training_updates=0, checkpoint_downloads=0, gpu_runs=0,
                     scientific_roots=0, confirmation_roots=0,
                     reserved_development_roots=0)
SOURCE_FILES = (
    "mechinterp/__init__.py",
    "mechinterp/bpd/__init__.py",
    "mechinterp/bpd/adapters/__init__.py",
    "mechinterp/bpd/adapters/trm.py",
    "mechinterp/bpd/adapters/huginn.py",
    "mechinterp/bpd/adapters/looped_chase.py",
    "mechinterp/bpd/components.py",
    "mechinterp/bpd/config.py",
    "mechinterp/bpd/analysis.py",
    "mechinterp/bpd/adversary.py",
    "mechinterp/bpd/ci.py",
    "mechinterp/bpd/contrast.py",
    "mechinterp/bpd/losses.py",
    "mechinterp/bpd/nulls.py",
    "mechinterp/bpd/trainer.py",
    "mechinterp/loopscope02/scripts/trm_sudoku.py",
    "mechinterp/loopscope02/scripts/sudoku_prop.py",
    "mechinterp/loopscope01/scripts/loopscope_toy.py",
    "Spectralucative/__init__.py",
    "Spectralucative/native.py",
    "Spectralucative/math.py",
    "Spectralucative/verify.py",
    "Spectralucative/PROTOCOL.json",
)
TOLERANCES = dict(all_ones_absolute=3e-5, factor_weight_absolute=2e-7,
                  delivered_write_absolute=4e-6, jvp_absolute=2e-5,
                  jvp_relative=3e-4, central_difference_absolute_l2=1e-3,
                  central_difference_relative_l2=0.02)


def validate_options(*, roots=4, steps=4, rank=3, components=4, cpu_threads=2):
    """Bound native work before any model allocation or output mutation."""
    limits = {"roots": (roots, 1, 8), "steps": (steps, 1, 8),
              "components": (components, 1, 4), "rank": (rank, 1, components),
              "cpu_threads": (cpu_threads, 1, 4)}
    for name, (value, lower, upper) in limits.items():
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


def payload_digest(payload):
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _seal(payload):
    payload["payload_sha256"] = payload_digest(payload)
    return payload


def _write_receipt(path, payload):
    # Exclusive creation preserves a prior run when the same path is requested.
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(_seal(payload), handle, sort_keys=True, separators=(",", ":"),
                  allow_nan=False)
        handle.write("\n")


def source_provenance(repo_root):
    root = Path(repo_root).resolve()
    hashes = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
              for relative in SOURCE_FILES}
    return dict(repository="MoralityLabAI/BPD", source_base_commit=SOURCE_COMMIT,
                execution_commit=os.environ.get("GITHUB_SHA"),
                source_files_sha256=hashes)


def _numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64)


def _state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)],
                                 separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _spectral_pair(matrix, step, **extra):
    z = 0.5 / max(1.0, matrix_inf_norm(matrix))
    return dict(step=step, raw=trace_logdet_diagnostic(matrix, z=1.0),
                damped=trace_logdet_diagnostic(matrix, z=z), **extra)


def _frozen_comparison(jacobians, reduced, basis):
    result = dict(depths=[], reduced_ordered_products=[], reduced_frozen_powers=[],
                  reduced_gap_inf=[], full_ordered_pushes=[], full_frozen_pushes=[],
                  full_gap_inf=[])
    for depth in range(1, len(jacobians) + 1):
        actual_a = ordered_product(reduced[:depth])
        frozen_a = np.linalg.matrix_power(reduced[0], depth)
        actual_j = ordered_product(jacobians[:depth]) @ basis
        frozen_j = np.linalg.matrix_power(jacobians[0], depth) @ basis
        result["depths"].append(depth)
        for key, value in (("reduced_ordered_products", actual_a),
                           ("reduced_frozen_powers", frozen_a),
                           ("full_ordered_pushes", actual_j),
                           ("full_frozen_pushes", frozen_j)):
            result[key].append(value.tolist())
        result["reduced_gap_inf"].append(matrix_inf_norm(actual_a - frozen_a))
        result["full_gap_inf"].append(matrix_inf_norm(actual_j - frozen_j))
    return result


def run_initialization(*, seed, steps=4, rank=3, components=4, cpu_threads=2,
                       repo_root=None):
    """Measure one fresh initialization without training or dataset access."""
    validate_options(roots=1, steps=steps, rank=rank, components=components,
                     cpu_threads=cpu_threads)
    if type(seed) is not int or not SEED_BASE <= seed < SEED_BASE + 8:
        raise ValueError("seed must belong to the excluded Spectralucative namespace")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Native study requires the CPU PyTorch dependency in requirements.txt") from error
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from mechinterp.bpd.adapters.trm import build_mini_trm
    from mechinterp.bpd.components import BPDModel, PassMaskSchedule
    from mechinterp.bpd.config import AblationConfig

    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    provenance = source_provenance(repo_root)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = build_mini_trm(**MODEL_RECIPE, seed=seed).cpu().float()
    bpd = BPDModel(model, {MODULE: components}, max_passes=2 * steps,
                   ablation=AblationConfig(mode="zero", sphere_rescale=False),
                   init_svd=True)
    for parameter in bpd.parameters():
        parameter.requires_grad_(False)
    wrapper = bpd.wrappers[MODULE]
    if any(parameter.device.type != "cpu" for parameter in bpd.parameters()):
        raise RuntimeError("The native study requires CPU parameters")
    model_hash = _state_digest(bpd)
    tokens = torch.tensor([[1, 2, 4, 7]], dtype=torch.long, device="cpu")
    with torch.no_grad():
        encoded = model.encode_inputs(tokens).detach()
    half = model.total_len * model.hidden_size
    dimension = 2 * half
    calls_per_outer = model.h_cycles * (model.l_cycles + 1)
    patch_position = model.puzzle_emb_len

    def pack(h, l):
        return torch.cat((h.reshape(-1), l.reshape(-1)))

    def unpack(state):
        shape = (1, model.total_len, model.hidden_size)
        return state[:half].reshape(shape), state[half:].reshape(shape)

    def step_fn(state):
        h, l = unpack(state)
        return pack(*model.act_step(h, l, encoded, grad_last_only=False))

    def head(state):
        h, _ = unpack(state)
        return model.lm_head(h)[:, model.puzzle_emb_len:, :][0]

    # Q embeds the declared component output-write span in the complete L state.
    writes = _numpy(wrapper.U)[:rank].T
    local_basis, triangular = np.linalg.qr(writes, mode="reduced")
    if np.min(np.abs(np.diag(triangular))) < 1e-10:
        raise RuntimeError("Declared component output writes are rank deficient")
    for column in range(rank):
        pivot = int(np.argmax(np.abs(local_basis[:, column])))
        if local_basis[pivot, column] < 0:
            local_basis[:, column] *= -1
    basis = np.zeros((dimension, rank), dtype=np.float64)
    start = half + patch_position * model.hidden_size
    basis[start:start + model.hidden_size] = local_basis
    q_native = torch.tensor(basis, dtype=torch.float32, device="cpu")

    with sdpa_kernel(SDPBackend.MATH):
        with torch.no_grad():
            bpd.clear_schedules()
            bpd.reset_wrappers()
            initial = pack(*model.init_states(1))
            # A declared single warmup keeps derivatives away from initial scale collapse.
            state0 = step_fn(initial).detach()
        states = [state0]
        jacobians, reduced, derivative_checks, thin_pushes = [], [], [], []
        for index in range(steps):
            bpd.reset_wrappers()
            jacobian = torch.func.jacrev(step_fn, chunk_size=16)(states[-1])
            j_array = _numpy(jacobian)
            if not np.all(np.isfinite(j_array)):
                raise FloatingPointError(f"Nonfinite native Jacobian at transition {index}")
            actual_pushes, finite_difference_pushes = [], []
            for column in range(rank):
                bpd.reset_wrappers()
                _, pushed = torch.func.jvp(step_fn, (states[-1],), (q_native[:, column],))
                actual_pushes.append(_numpy(pushed))
                epsilon = 1e-3
                with torch.no_grad():
                    plus = step_fn(states[-1] + epsilon * q_native[:, column])
                    minus = step_fn(states[-1] - epsilon * q_native[:, column])
                finite_difference_pushes.append((_numpy(plus) - _numpy(minus)) / (2 * epsilon))
            direct = np.column_stack(actual_pushes)
            finite_difference = np.column_stack(finite_difference_pushes)
            expected = j_array @ _numpy(q_native)
            fd_errors = np.linalg.norm(finite_difference - expected, axis=0)
            fd_limits = (TOLERANCES["central_difference_absolute_l2"] +
                         TOLERANCES["central_difference_relative_l2"] *
                         np.linalg.norm(expected, axis=0))
            derivative_checks.append(dict(
                step=index, native_basis=q_native.tolist(), direct_jvp=direct.tolist(),
                central_difference=finite_difference.tolist(), central_difference_epsilon=epsilon,
                direct_jvp_max_abs_error=float(np.max(np.abs(direct - expected))),
                direct_jvp_passed=bool(np.allclose(direct, expected,
                    atol=TOLERANCES["jvp_absolute"], rtol=TOLERANCES["jvp_relative"])),
                central_difference_error_l2=fd_errors.tolist(),
                central_difference_limits_l2=fd_limits.tolist(),
                central_difference_passed=bool(np.all(fd_errors <= fd_limits))))
            jacobians.append(j_array)
            reduced.append(basis.T @ j_array @ basis)
            thin_pushes.append(j_array @ basis)
            with torch.no_grad():
                states.append(step_fn(states[-1]).detach())

        state_arrays = [_numpy(state) for state in states]
        state_norm = float(np.linalg.norm(state_arrays[0]))
        relative_doses = [1e-4, 1e-3, 1e-2]
        finite_records = []
        for dose in relative_doses:
            requested = dose * state_norm * basis[:, 0]
            perturbed = state0 + torch.tensor(requested, dtype=torch.float32)
            actual_initial = _numpy(perturbed) - state_arrays[0]
            deltas = [actual_initial]
            perturbed_states = [_numpy(perturbed)]
            with torch.no_grad():
                bpd.clear_schedules()
                bpd.reset_wrappers()
                for index in range(steps):
                    perturbed = step_fn(perturbed)
                    perturbed_states.append(_numpy(perturbed))
                    deltas.append(_numpy(perturbed) - state_arrays[index + 1])
            predicted = [actual_initial]
            for jacobian in jacobians:
                predicted.append(jacobian @ predicted[-1])
            errors = [float(np.linalg.norm(actual - linear))
                      for actual, linear in zip(deltas, predicted)]
            finite_records.append(dict(relative_dose=dose, requested_initial_delta=requested.tolist(),
                actual_initial_delta=actual_initial.tolist(), actual_deltas=[d.tolist() for d in deltas],
                perturbed_states=[state.tolist() for state in perturbed_states],
                linearized_deltas=[d.tolist() for d in predicted], remainder_l2=errors,
                relative_remainder=[error / max(1e-15, float(np.linalg.norm(delta)))
                                    for error, delta in zip(errors, deltas)]))

        def rollout(mask_id=None):
            capture = {}

            def capture_first(_module, args, output):
                if wrapper.calls == 1:
                    capture["input"] = _numpy(args[0])[0]
                    capture["write"] = _numpy(output)[0]

            handle = wrapper.register_forward_hook(capture_first)
            if mask_id is None:
                bpd.clear_schedules()
                bpd.reset_wrappers()
                context = nullcontext()
            else:
                count = steps * calls_per_outer
                masks = torch.ones((1, count, model.total_len, components), dtype=torch.float32)
                bits = [(mask_id >> bit) & 1 for bit in range(components)]
                masks[0, 0, patch_position] = torch.tensor(bits, dtype=torch.float32)
                # All other positions/invocations execute the exact target branch.
                routes = torch.zeros((1, count, model.total_len), dtype=torch.bool)
                routes[0, 0, patch_position] = True
                context = bpd.masked_components({MODULE: PassMaskSchedule(
                    comp_mask=masks, delta_mask=None, routing_mask=routes)})
            try:
                with context, torch.no_grad():
                    current = state0.clone()
                    trajectory = []
                    for _ in range(steps):
                        current = step_fn(current)
                        trajectory.append(_numpy(current))
                    logits = _numpy(head(current))
                    if wrapper.calls != steps * calls_per_outer:
                        raise RuntimeError("Native mask invocation count changed")
            finally:
                handle.remove()
            if set(capture) != {"input", "write"}:
                raise RuntimeError("First physical component write was not observed")
            return logits, trajectory, capture

        target_logits, target_states, target_capture = rollout()
        mask_ids = list(range(1 << components))
        cube = dict(component_count=components, mask_ids=mask_ids, bits=[],
            response_definition="flattened_full_11_head_logit_delta_from_target",
            scalar_response_definition="mean_squared_full_head_logit_drift_from_target",
            scalar_responses=[], full_head_logits=[], full_head_logit_deltas=[],
            target_full_head_logits=target_logits.tolist(), physical_write_displacement_l2=[],
            physical_write_displacements=[], expected_physical_write_displacements=[],
            delivered_write_max_abs_errors=[], first_step_state_displacement_l2=[],
            final_state_displacement_l2=[], first_step_states=[], final_states=[],
            target_first_step_state=target_states[0].tolist(), target_final_state=target_states[-1].tolist(),
            target_first_write=target_capture["write"].tolist(),
            first_wrapper_input=target_capture["input"].tolist(),
            all_ones_mask_id=(1 << components) - 1,
            all_ones_tolerance=TOLERANCES["all_ones_absolute"], residual_retained=True,
            sphere_rescale=False, ablation_mode="zero",
            scope="descriptive_varying_dose_interaction_cube",
            patch_invocation=0, patch_position=patch_position,
            horizon_outer_steps=steps, calls_per_outer_step=calls_per_outer,
            routing="component at declared position and invocation; native target elsewhere")
        v_array, u_array = _numpy(wrapper.V), _numpy(wrapper.U)
        for mask_id in mask_ids:
            logits, trajectory, capture = rollout(mask_id)
            bits = np.array([(mask_id >> bit) & 1 for bit in range(components)])
            delta = (logits - target_logits).reshape(-1)
            write_delta = capture["write"] - target_capture["write"]
            expected_write = np.zeros_like(write_delta)
            expected_write[patch_position] = (
                (target_capture["input"][patch_position] @ v_array) * (bits - 1)) @ u_array
            cube["bits"].append(bits.tolist())
            cube["full_head_logits"].append(logits.tolist())
            cube["full_head_logit_deltas"].append(delta.tolist())
            cube["scalar_responses"].append(float(np.mean(delta ** 2)))
            cube["physical_write_displacements"].append(write_delta.tolist())
            cube["expected_physical_write_displacements"].append(expected_write.tolist())
            cube["physical_write_displacement_l2"].append(float(np.linalg.norm(write_delta)))
            cube["delivered_write_max_abs_errors"].append(float(np.max(np.abs(write_delta - expected_write))))
            cube["first_step_state_displacement_l2"].append(float(np.linalg.norm(trajectory[0] - target_states[0])))
            cube["final_state_displacement_l2"].append(float(np.linalg.norm(trajectory[-1] - target_states[-1])))
            cube["first_step_states"].append(trajectory[0].tolist())
            cube["final_states"].append(trajectory[-1].tolist())
        cube["all_ones_max_abs_logit_error"] = float(np.max(np.abs(cube["full_head_logit_deltas"][-1])))
        cube["all_ones_max_abs_state_error"] = float(np.max(np.abs(
            np.asarray(cube["final_states"][-1]) - target_states[-1])))

    spectral = [_spectral_pair(a, index, matrix=a.tolist()) for index, a in enumerate(reduced)]
    full_spectral = [_spectral_pair(j, index, matrix_source=f"transport.B_steps[{index}]")
                     for index, j in enumerate(jacobians)]
    transport = transport_diagnostic(reduced, jacobians, [basis] * (steps + 1))
    walsh = walsh_diagnostic(mask_ids, cube["full_head_logit_deltas"], max_degree=1)
    target_weight = _numpy(wrapper.W_target)
    residual = _numpy(wrapper.weight_delta())
    reconstruction = (v_array @ u_array).T
    factor_error = float(np.max(np.abs(reconstruction + residual - target_weight)))
    leakage = [pushed - basis @ a for pushed, a in zip(thin_pushes, reduced)]
    checks = dict(
        all_ones_full_head_parity=cube["all_ones_max_abs_logit_error"] <= TOLERANCES["all_ones_absolute"],
        all_ones_full_state_parity=cube["all_ones_max_abs_state_error"] <= TOLERANCES["all_ones_absolute"],
        residual_retained_weight_parity=factor_error <= TOLERANCES["factor_weight_absolute"],
        delivered_write_formula=max(cube["delivered_write_max_abs_errors"]) <= TOLERANCES["delivered_write_absolute"],
        autodiff_jvp_parity=all(check["direct_jvp_passed"] for check in derivative_checks),
        central_difference_jvp_sanity=all(check["central_difference_passed"] for check in derivative_checks),
        transport_identity=transport["floating_point_check_passed"],
        walsh_identities=walsh["floating_point_check_passed"],
        spectral_domain_checks=all(item["damped"]["floating_point_check_passed"] and
            (item["raw"]["floating_point_check_passed"] is not False)
            for item in spectral + full_spectral),
        model_parameters_unchanged=_state_digest(bpd) == model_hash)
    payload = dict(
        schema="BPD/Spectralucative/native-unit/v1", evidence_class="engineering_implementation_only",
        engineering_initialization_id=f"spectralucative_cpu_init_{seed}", seed=seed,
        model_initialization="random", **ZERO_COUNTERS, provenance=provenance,
        model_state_sha256=model_hash, model_recipe=dict(MODEL_RECIPE, seed=seed),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                     device="cpu", cpu_threads=cpu_threads, native_dtype="float32",
                     analysis_dtype="float64", attention_backend="math",
                     deterministic_algorithms=True, roundoff_certified=False),
        numerical_tolerances=TOLERANCES, engineering_fixture=dict(
            token_ids=tokens.tolist(), task_performance_endpoint=None, warmup_outer_steps=1,
            warmup_initial_state=_numpy(initial).tolist()),
        state=dict(order="flatten(z_h),flatten(z_l); position-major within each state",
                   dimension=dimension, half_shape=[1, model.total_len, model.hidden_size],
                   trajectory=[state.tolist() for state in state_arrays],
                   full_head_trajectory=[_numpy(head(s)).tolist() for s in states]),
        basis=dict(construction="Chosen boundary subspace: QR of first declared BPD U rows, embedded at first board position in z_l",
                   native_write_relation="U spans the pre-RMSNorm write. This coordinate lift is chosen at the outer-state boundary; post-RMSNorm and subsequent native propagation are measured by full Jacobians and leakage.",
                   rank=rank, component_ids=list(range(rank)), patch_position=patch_position,
                   matrix=basis.tolist(), orthonormality_max_abs_error=float(np.max(np.abs(basis.T @ basis - np.eye(rank))))),
        spectral=spectral, full_state_spectral=full_spectral,
        transport=dict(A_steps=[a.tolist() for a in reduced], B_steps=[j.tolist() for j in jacobians],
                       L_steps=[basis.tolist()] * (steps + 1), report=transport),
        native_pushes=dict(JQ=[push.tolist() for push in thin_pushes],
                           leakage=[escape.tolist() for escape in leakage],
                           leakage_frobenius=[float(np.linalg.norm(escape)) for escape in leakage],
                           derivative_checks=derivative_checks),
        frozen_operator_comparison=_frozen_comparison(jacobians, reduced, basis),
        finite_displacement=dict(direction=basis[:, 0].tolist(), relative_doses=relative_doses,
                                 baseline_states=[state.tolist() for state in state_arrays],
                                 state_norm_l2=state_norm, records=finite_records),
        mask_cube=cube, walsh=dict(mask_ids=mask_ids, values=cube["full_head_logit_deltas"],
                                 values_source="mask_cube.full_head_logit_deltas", report=walsh),
        factor=dict(module=MODULE, V=v_array.tolist(), U=u_array.tolist(), target=target_weight.tolist(),
                    residual=residual.tolist(), reconstruction=reconstruction.tolist(), mask=[1] * components,
                    residual_retained=True, all_ones_max_abs_weight_error=factor_error,
                    all_ones_tolerance=TOLERANCES["factor_weight_absolute"]),
        checks=checks, all_checks_passed=all(checks.values()))
    return _seal(payload)


def run_study(out, *, roots=4, steps=4, rank=3, components=4, cpu_threads=2, repo_root=None):
    validate_options(roots=roots, steps=steps, rank=rank, components=components, cpu_threads=cpu_threads)
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    folder = Path(out)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    provenance = source_provenance(repo_root)
    units, errors = [], []
    started = time.monotonic()
    raw_ineligible = 0
    for offset in range(roots):
        seed = SEED_BASE + offset
        identifier = f"spectralucative_cpu_init_{seed}"
        try:
            payload = run_initialization(seed=seed, steps=steps, rank=rank, components=components,
                                         cpu_threads=cpu_threads, repo_root=repo_root)
            filename = identifier + ".json"
            _write_receipt(folder / filename, payload)
            units.append(dict(file=filename, engineering_initialization_id=identifier, seed=seed,
                              payload_sha256=payload["payload_sha256"], all_checks_passed=payload["all_checks_passed"]))
            raw_ineligible += sum(not step["raw"]["convergence_condition"]
                                  for step in payload["full_state_spectral"])
            print(f"{identifier}: checks={payload['all_checks_passed']}", flush=True)
            if not payload["all_checks_passed"]:
                print(json.dumps({"failed_checks": [name for name, passed in payload["checks"].items()
                                                    if not passed]}, sort_keys=True), flush=True)
        except Exception as error:
            failure = dict(schema="BPD/Spectralucative/native-error/v1", engineering_initialization_id=identifier,
                           seed=seed, error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            filename = identifier + ".error.json"
            _write_receipt(folder / filename, failure)
            errors.append(dict(file=filename, engineering_initialization_id=identifier, seed=seed,
                               error_type=type(error).__name__, message=str(error)))
            print(f"{identifier}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            print(failure["traceback"], file=sys.stderr, flush=True)
    success = not errors and len(units) == roots and all(unit["all_checks_passed"] for unit in units)
    summary = dict(schema="BPD/Spectralucative/native-run/v1", evidence_class="engineering_implementation_only",
                   model_initialization="random", **ZERO_COUNTERS, engineering_root_count=roots,
                   completed_engineering_roots=len(units), status="complete" if success else "failed",
                   configuration=dict(roots=roots, steps=steps, rank=rank, components=components, cpu_threads=cpu_threads),
                   provenance=provenance, units=units, errors=errors,
                   raw_z1_ineligible_native_steps=raw_ineligible, all_checks_passed=success,
                   elapsed_seconds=time.monotonic() - started)
    _write_receipt(folder / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        summary = run_study(**vars(args))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Native study failed: {error}\n")
    print(json.dumps(dict(status=summary["status"], engineering_root_count=summary["engineering_root_count"],
                          completed_engineering_roots=summary["completed_engineering_roots"],
                          summary=str(args.out / "summary.json")), sort_keys=True))
    return 0 if summary["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
