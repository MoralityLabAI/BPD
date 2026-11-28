"""Dose-matched prefix interventions with an exactly preserved query state.

Directions are selected before any patched output is scored. Matching uses
the displacement after conversion to the native state dtype, including BF16
rounding. Interpolated prefix states are constructed interventions; natural
reachability of the complete patched state is not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any


ARM_NAMES = ("identity", "effect", "sham", "random", "full_donor_prefix")


def case_seed(case_id: str) -> int:
    return int.from_bytes(hashlib.sha256(("BPD-COL-02/" + case_id).encode()).digest()[:8], "big") % (2 ** 63)


def _norm(value: Any) -> float:
    import torch
    return float(torch.linalg.vector_norm(value.float()).item())


@dataclass
class PrefixPatch:
    name: str
    source: Any
    replacement_prefix: Any
    requested_dose: float
    physical_dose: float
    admission: bool
    matching_relative_error: float
    selected_visit: int = 0
    calls: int = 0
    measured_dose: float | None = None

    def __call__(self, state: Any, visit: int) -> Any:
        import torch
        if visit != self.selected_visit:
            return state
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("Selected loop boundary was patched more than once")
        actual = state.detach().cpu()
        if actual.shape != self.source.shape or not torch.equal(actual, self.source):
            raise RuntimeError("Replayed recipient state differs from its captured source")
        edited = state.clone()
        edited[:, :-1, :] = self.replacement_prefix.to(device=state.device, dtype=state.dtype)
        if not torch.equal(edited[:, -1, :], state[:, -1, :]):
            raise AssertionError("Protected final-position state changed")
        self.measured_dose = _norm(edited[:, :-1, :].detach().cpu().float() - actual[:, :-1, :].float())
        if not math.isclose(self.measured_dose, self.physical_dose, rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError("Delivered intervention dose differs from its prepared dose")
        return edited

    def metadata(self) -> dict:
        return {
            "physical_dose": self.physical_dose,
            "requested_dose": self.requested_dose,
            "dose_relative_error": self.matching_relative_error,
            "admission": self.admission,
            "selected_visit": self.selected_visit,
            "patch_calls": self.calls,
            "measured_dose": self.measured_dose,
            "protected_readout_max_abs": 0.0,
            "protected_readout_evidence": "bit-identical final-position normalized state; identical complete vocabulary head",
            "state_scope": "constructed prefix patch; last prompt position held fixed",
        }


def _matched_prefix(source: Any, direction: Any, dose: float, tolerance: float) -> tuple[Any, float, bool]:
    """Choose a scalar by monotone search after native-dtype rounding."""
    import torch
    length = _norm(direction)
    if length == 0:
        raise ValueError("Intervention direction is zero")
    unit = direction.float() / length
    baseline = source.float()

    def candidate(scale: float) -> tuple[Any, float]:
        value = (baseline + scale * unit).to(source.dtype)
        return value, _norm(value.float() - baseline)

    lo, hi = 0.0, max(2 * dose, 1e-8)
    high, measured = candidate(hi)
    for _ in range(24):
        if measured >= dose:
            break
        hi *= 2
        high, measured = candidate(hi)
    else:
        raise ValueError("Requested dose could not be bracketed")
    best, best_dose = high, measured
    for _ in range(32):
        midpoint = (lo + hi) / 2
        value, actual = candidate(midpoint)
        if abs(actual - dose) < abs(best_dose - dose):
            best, best_dose = value, actual
        if actual < dose:
            lo = midpoint
        else:
            hi = midpoint
    admitted = abs(best_dose - dose) <= max(1e-6, tolerance * dose)
    if not torch.isfinite(best).all():
        raise ValueError("Nonfinite intervention state")
    return best, best_dose, admitted


def prepare_arms(recipient: Any, effect_donor: Any, sham_donor: Any, *,
                 case_id: str, dose_fraction: float = 0.1,
                 tolerance: float = 0.01) -> dict[str, PrefixPatch]:
    import torch
    if isinstance(dose_fraction, bool) or not math.isfinite(dose_fraction) or not 0 < dose_fraction <= 1:
        raise ValueError("dose_fraction must be finite and in (0,1]")
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or not 0 < tolerance <= 0.02:
        raise ValueError("dose tolerance must be finite and in (0,0.02]")
    values = [v.detach().cpu().contiguous() for v in (recipient, effect_donor, sham_donor)]
    source, donor, sham = values
    if source.ndim != 3 or source.shape[0] != 1 or source.shape[1] < 2:
        raise ValueError("Expected batch-one [1, tokens>=2, hidden] boundary states")
    if any(v.shape != source.shape or v.dtype != source.dtype for v in values):
        raise ValueError("Donors must align with the complete recipient token/state shape")
    if any(not torch.isfinite(v).all() for v in values):
        raise ValueError("Nonfinite source or donor state")
    prefix = source[:, :-1, :]
    effect_delta = donor[:, :-1, :].float() - prefix.float()
    sham_delta = sham[:, :-1, :].float() - prefix.float()
    minimum = min(_norm(effect_delta), _norm(sham_delta))
    if minimum <= 0:
        raise ValueError("Both donor prefixes must have nonzero displacement")
    requested = dose_fraction * minimum
    generator = torch.Generator(device="cpu").manual_seed(case_seed(case_id))
    random_direction = torch.randn(prefix.shape, dtype=torch.float32, generator=generator)
    arms = {
        "identity": PrefixPatch("identity", source, prefix.clone(), 0.0, 0.0, True, 0.0),
    }
    for name, direction in (("effect", effect_delta), ("sham", sham_delta), ("random", random_direction)):
        replacement, actual, admitted = _matched_prefix(prefix, direction, requested, tolerance)
        arms[name] = PrefixPatch(name, source, replacement, requested, actual, admitted,
                                 abs(actual - requested) / requested)
    full_dose = _norm(effect_delta)
    arms["full_donor_prefix"] = PrefixPatch(
        "full_donor_prefix", source, donor[:, :-1, :].clone(), full_dose, full_dose, True, 0.0)
    return arms
