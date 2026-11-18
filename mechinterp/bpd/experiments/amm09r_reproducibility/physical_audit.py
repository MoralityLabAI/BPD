"""Excluded-root physical and native-wrapper checks for AMM-09R.

This module never fits models. ``audit`` rejects every non-engineering phase
before requesting donor pairs. All native execution uses the repository's
float32 BPDModel; any reference-to-float32 export error is reported explicitly.
The donor-state control has the same L2 dose and an unrestricted visible write.
"""
from __future__ import annotations

import math

import torch


MASKS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))
NATIVE_TOLERANCE = 1e-5
DOSE_TOLERANCE = 1e-5
GL3_TOLERANCE = 1e-9
CANONICAL_TOLERANCE = 1e-8
IDENTIFIABLE_GAP = .01


def _max_error(left, right):
    return float((left.detach().double() - right.detach().double()).abs().max())


def _finite(tensor):
    return bool(torch.isfinite(tensor).all())


def _dark(raw, dose):
    """Independently implement the registered projection and fallback."""
    projected = raw.clone()
    projected[:, 0] = 0
    lengths = projected.norm(dim=1, keepdim=True)
    fallback = torch.zeros_like(projected)
    fallback[:, 1] = 1
    unit = torch.where(lengths > 1e-8,
                       projected / lengths.clamp_min(1e-8), fallback)
    return unit * dose


def _gauge_metrics(factor):
    v = factor.V.detach().double()
    u = factor.U.detach().double()
    weight = v @ u

    def canonical_products(matrix):
        left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        products = [(left[:, i:i + 1] * singular[i]) @ right[i:i + 1]
                    for i in range(3)]
        return singular, products

    singular, before = canonical_products(weight)
    transforms = {
        "shear": [[1., .6, 0.], [0., 1., .3], [.2, 0., 1.]],
        "rescale": [[2., 0., 0.], [0., .5, 0.], [0., 0., 1.]],
        "signed_permutation": [[0., 0., -1.], [1., 0., 0.], [0., 1., 0.]],
    }
    rows = {}
    for name, values in transforms.items():
        transform = torch.tensor(values, dtype=torch.float64)
        changed = (v @ transform) @ (torch.linalg.inv(transform) @ u)
        _, after = canonical_products(changed)
        rows[name] = {
            "max_full_operator_error": _max_error(weight, changed),
            "max_canonical_rankone_error": max(_max_error(a, b)
                                               for a, b in zip(before, after)),
        }
    gap = float(torch.min(singular[:2] - singular[1:3]))
    full_error = max(r["max_full_operator_error"] for r in rows.values())
    component_error = max(r["max_canonical_rankone_error"] for r in rows.values())
    identifiable = gap > IDENTIFIABLE_GAP
    return {
        "transforms": rows,
        "min_singular_gap": gap,
        "canonical_rankone_parity_applicable": identifiable,
        "max_full_gl3_error": full_error,
        "max_canonical_rankone_error": component_error,
        "passed": full_error <= GL3_TOLERANCE and (
            not identifiable or component_error <= CANONICAL_TOLERANCE),
    }


def audit(m, case, teacher, factor, phase, index):
    """Return JSON-safe physical audit receipts for one excluded engineering fit.

    ``m`` and ``case`` are the candidate's model and case modules. No global
    random stream is spent by native-wrapper construction. The generated donor
    pairs belong only to the registered engineering split.
    """
    if phase != "engineering":
        raise ValueError("physical audit only admits excluded engineering roots")
    if type(index) is not int or not 0 <= index < m.PHASE_N["engineering"]:
        raise ValueError("unregistered engineering root")
    if tuple(tuple(mask) for mask in m.MASKS) != MASKS:
        raise ValueError("physical audit requires exactly the six two-slot masks")
    if m.D != 6 or m.R != 3:
        raise ValueError("physical audit requires the registered 6D rank-three model")
    root = m.root_key(phase, index)

    from mechinterp.bpd.components import BPDModel, PassMaskSchedule
    from mechinterp.bpd.config import AblationConfig

    with torch.no_grad():
        recipient, donor, _ = m.donor_pairs(phase, index, m.TRAIN_PAIRS, "engineering")
        recipient = recipient.to(dtype=factor.V.dtype)
        donor = donor.to(dtype=factor.V.dtype)
        reference = case.candidate_features(factor, recipient, donor,
                                            m.family_for(index))
        n = len(recipient)
        if tuple(reference["features"].shape) != (n, 6, 18):
            raise ValueError("wrong feature/candidate shape in physical audit")
        if tuple(reference["probes"].shape) != (n, 6, 2):
            raise ValueError("wrong delayed-readout shape in physical audit")
        if len(reference["patches"]) != 6:
            raise ValueError("wrong physical patch count")
        patches = reference["patches"]
        dose = reference["dose"]
        first = reference["recipient_first"]
        donor_first = reference["donor_first"]
        full_delta = donor_first - first
        finite = all(_finite(value) for value in (
            recipient, donor, factor.V, factor.U, dose, first, donor_first,
            reference["features"], reference["probes"], *patches))
        if not finite:
            raise ArithmeticError("nonfinite engineering physical audit input")
        dose_error = max(float((patch.norm(dim=1) - dose[:, 0]).abs().max())
                         for patch in patches)
        current_write = max(float(patch[:, 0].abs().max()) for patch in patches)
        current_state = max(float(((first + patch)[:, 0] - first[:, 0]).abs().max())
                            for patch in patches)
        projection_error = 0.
        for subset, patch in zip(MASKS, patches):
            raw = ((donor - recipient) @ factor.V[:, list(subset)]) @ factor.U[list(subset), :]
            projection_error = max(projection_error, _max_error(patch, _dark(raw, dose)))

        class Base(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.core = torch.nn.Linear(6, 6, bias=False).float()
                self.core.weight.copy_(weight)

            def forward(self, value):
                return self.core(value)

        # Native constructors initialize parameters; preserve the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            native = BPDModel(
                Base(teacher.weight().detach().float()), {"core": 3}, max_passes=3,
                ablation=AblationConfig(mode="zero", sphere_rescale=False),
                init_svd=False,
            )
        wrapper = native.wrappers["core"]
        wrapper.V.copy_(factor.V.detach().float())
        wrapper.U.copy_(factor.U.detach().float())
        rec32, don32 = recipient.float(), donor.float()
        v32, u32 = factor.V.detach().float(), factor.U.detach().float()

        def ref32(value):
            return (value @ v32) @ u32

        def schedule(subset=None):
            components = torch.ones((n, 3, 1, 3), dtype=torch.float32)
            if subset is not None:
                components[:, 0, :, :] = 0
                components[:, 0, :, list(subset)] = 1
            return PassMaskSchedule(components, torch.zeros((n, 3, 1), dtype=torch.float32))

        call_records = []
        reset_ok = True
        cleanup_ok = True
        with native.masked_components({"core": schedule()}):
            reset_ok = reset_ok and wrapper.calls == 0
            a = native.target_model(rec32[:, None, :])[:, 0]
            b = native.target_model(a[:, None, :])[:, 0]
            c = native.target_model(b[:, None, :])[:, 0]
            call_records.append({"path": "recipient_full", "calls": wrapper.calls, "expected": 3})
        cleanup_ok = cleanup_ok and wrapper.mode == "target" and wrapper.schedule is None
        export_first = ref32(rec32)
        export_second = ref32(export_first)
        export_third = ref32(export_second)
        native_export_error = max(_max_error(a, export_first), _max_error(b, export_second),
                                  _max_error(c, export_third))
        reference_full_error = max(_max_error(a, first),
                                   _max_error(b[:, 0], reference["reference_two"][:, 0]),
                                   _max_error(c[:, 0], reference["reference_two"][:, 1]))

        with native.masked_components({"core": schedule()}):
            reset_ok = reset_ok and wrapper.calls == 0
            d = native.target_model(don32[:, None, :])[:, 0]
            call_records.append({"path": "donor_first", "calls": wrapper.calls, "expected": 1})
        cleanup_ok = cleanup_ok and wrapper.mode == "target" and wrapper.schedule is None
        native_dose = (d - a).norm(dim=1, keepdim=True)
        export_dose = (ref32(don32) - export_first).norm(dim=1, keepdim=True)

        subset_rows = []
        for action, subset in enumerate(MASKS):
            with native.masked_components({"core": schedule(subset)}):
                reset_ok = reset_ok and wrapper.calls == 0
                raw_native = native.target_model((don32 - rec32)[:, None, :])[:, 0]
                call_records.append({"path": f"mask_{action}_raw", "calls": wrapper.calls, "expected": 1})
            cleanup_ok = cleanup_ok and wrapper.mode == "target" and wrapper.schedule is None
            raw_export = ((don32 - rec32) @ v32[:, list(subset)]) @ u32[list(subset), :]
            raw_reference = ((donor - recipient) @ factor.V[:, list(subset)]) @ factor.U[list(subset), :]
            patch_native = _dark(raw_native, native_dose)
            patch_export = _dark(raw_export, export_dose)
            with native.masked_components({"core": schedule()}):
                reset_ok = reset_ok and wrapper.calls == 0
                r1 = native.target_model(rec32[:, None, :])[:, 0]
                r2 = native.target_model((r1 + patch_native)[:, None, :])[:, 0]
                r3 = native.target_model(r2[:, None, :])[:, 0]
                call_records.append({"path": f"mask_{action}_patched", "calls": wrapper.calls, "expected": 3})
            cleanup_ok = cleanup_ok and wrapper.mode == "target" and wrapper.schedule is None
            export2 = ref32(export_first + patch_export)
            export3 = ref32(export2)
            probes_native = torch.stack((r2[:, 0], r3[:, 0]), dim=1)
            row = {
                "action": action,
                "mask": list(subset),
                "raw_native_vs_float32_export_max_error": _max_error(raw_native, raw_export),
                "raw_native_vs_reference_max_error": _max_error(raw_native, raw_reference),
                "patch_native_vs_reference_max_error": _max_error(patch_native, patches[action]),
                "future_native_vs_float32_export_max_error": max(_max_error(r2, export2),
                                                                   _max_error(r3, export3)),
                "future_readouts_native_vs_reference_max_error": _max_error(
                    probes_native, reference["probes"][:, action, :]),
                "native_patch_current_readout_displacement": float(patch_native[:, 0].abs().max()),
                "native_patch_dose_error": float((patch_native.norm(dim=1) - native_dose[:, 0]).abs().max()),
            }
            native_export_error = max(native_export_error,
                row["raw_native_vs_float32_export_max_error"],
                row["future_native_vs_float32_export_max_error"])
            subset_rows.append(row)

        native_reference_error = max(reference_full_error, *(
            row[key] for row in subset_rows for key in (
                "raw_native_vs_reference_max_error", "patch_native_vs_reference_max_error",
                "future_readouts_native_vs_reference_max_error")))
        native_dose_error = max(row["native_patch_dose_error"] for row in subset_rows)
        native_current_write = max(row["native_patch_current_readout_displacement"]
                                   for row in subset_rows)
        calls_ok = all(row["calls"] == row["expected"] for row in call_records)
        gauge = _gauge_metrics(factor)
        full_dose_error = float((full_delta.norm(dim=1) - dose[:, 0]).abs().max())
        full_visible = full_delta[:, 0].abs()
        gates = {
            "six_frozen_nonempty_two_slot_masks": True,
            "finite": finite,
            "gauge": gauge["passed"],
            "equal_dose": dose_error <= DOSE_TOLERANCE,
            "current_readout_unchanged": current_write == 0. and current_state == 0.,
            "projection_and_normalization": projection_error <= NATIVE_TOLERANCE,
            "native_float32_export_parity": native_export_error <= NATIVE_TOLERANCE,
            "native_reference_parity_at_registered_tolerance": native_reference_error <= NATIVE_TOLERANCE,
            "native_equal_dose": native_dose_error <= DOSE_TOLERANCE,
            "native_current_readout_unchanged": native_current_write == 0.,
            "native_pass_counts": calls_ok,
            "native_pass_reset": reset_ok,
            "native_context_cleanup": cleanup_ok,
            "full_donor_equal_dose": full_dose_error <= DOSE_TOLERANCE,
        }
        result = {
            "schema": "bpd.amm09r.physical_audit.v1",
            "root": root,
            "phase": phase,
            "split": "engineering",
            "n_cases": n,
            "masks": [list(mask) for mask in MASKS],
            "reference_dtype": str(factor.V.dtype),
            "native_execution_dtype": "torch.float32",
            "reference_uses_native_dtype": factor.V.dtype == torch.float32 and factor.U.dtype == torch.float32,
            "native_tolerance": NATIVE_TOLERANCE,
            "dose_tolerance": DOSE_TOLERANCE,
            "gauge": gauge,
            "max_equal_dose_error": dose_error,
            "max_immediate_readout_write": current_write,
            "max_immediate_readout_state_displacement": current_state,
            "max_projection_and_normalization_error": projection_error,
            "zero_dose_cases": int((dose[:, 0] <= 1e-8).sum()),
            "degenerate_projection_cases_per_mask": [int(x) for x in reference["degenerate_masks"]],
            "max_native_float32_export_error": native_export_error,
            "max_native_reference_error": native_reference_error,
            "max_native_equal_dose_error": native_dose_error,
            "max_native_immediate_readout_write": native_current_write,
            "native_subset_audits": subset_rows,
            "native_call_records": call_records,
            "full_donor_control": {
                "description": "unrestricted visible donor-state write at the same L2 dose",
                "max_dose_error": full_dose_error,
                "max_immediate_readout_displacement": float(full_visible.max()),
                "mean_immediate_readout_displacement": float(full_visible.mean()),
                "cases_with_immediate_readout_displacement": int((full_visible > 0).sum()),
                "oracle_dominance_assumed": False,
            },
            "gates": gates,
            "passed": all(gates.values()),
        }
        # An accidental NaN should fail before a receipt can claim success.
        if not all(math.isfinite(value) for value in (
            dose_error, current_write, current_state, projection_error,
            native_export_error, native_reference_error, native_dose_error,
            native_current_write, full_dose_error)):
            raise ArithmeticError("nonfinite engineering physical audit output")
        return result
