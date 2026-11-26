"""Native BPD execution for the existing HAD-05 recurrent scheduler.

The historical model writes ``(h @ V) @ U.T``.  The repository component
module writes ``(h @ V_native) @ U_native``, so conversion copies ``V`` and
the transpose of ``U``.  Open component gates and an explicit zero residual
mask preserve the trainable factor path, including its gradients.

This adapter establishes an engineering interface.  It supplies no new
search benchmark or causal-mechanism evidence.  The history still starts at
zero for each scheduler decision, as it does in the historical model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import weakref

import torch
from torch import Tensor, nn

from mechinterp.bpd.components import PassMaskSchedule, TiedComponentLinear
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.hadamard_population.had05 import F, LDT, Recurrent


@dataclass(frozen=True)
class HistoryContinuation:
    """An explicit final-pass boundary, valid for replay and interleaving.

    ``pass_index`` is global within the requested execution.  ``schedule``
    contains exactly one local pass.  Keep a continuation within the same
    parameter version when comparing counterfactual hidden states.
    """

    last: Tensor
    hidden: Tensor
    pass_index: int
    schedule: PassMaskSchedule
    owner: object


def _window(schedule: PassMaskSchedule, start: int, stop: int) -> PassMaskSchedule:
    return PassMaskSchedule(
        comp_mask=schedule.comp_mask[:, start:stop],
        delta_mask=(
            None if schedule.delta_mask is None else schedule.delta_mask[:, start:stop]
        ),
        routing_mask=(
            None
            if schedule.routing_mask is None
            else schedule.routing_mask[:, start:stop]
        ),
    )


class NativeRecurrent(nn.Module):
    """Copy a float32 ``Recurrent`` into one native, weight-tied component.

    ``forward`` and ``mid`` accept a full history mask schedule.  Masks use
    local indices ``0 .. history_length - 1``; ``pass_offset`` sets the global
    index for calibration and recording.  A missing delta mask is converted
    to zero.  An explicit nonzero delta mask opts into the native residual
    against the frozen weight snapshot taken at conversion.

    ``mid`` / ``finish`` support the original HAD-05 helpers, including
    repeated ``finish`` calls with edited hidden states.  For multiple live
    boundaries use ``boundary`` / ``resume``.  A stale legacy boundary raises
    instead of silently inheriting another history's pass index.
    """

    def __init__(self, source: Recurrent, *, max_passes: int = 64) -> None:
        super().__init__()
        if not isinstance(source, Recurrent):
            raise TypeError("source must be the historical HAD-05 Recurrent")
        if any(p.dtype != torch.float32 for p in source.parameters()):
            raise ValueError("Native HAD-05 conversion currently supports float32 parameters")
        if len({p.device for p in source.parameters()}) != 1:
            raise ValueError("All source parameters must be on the same device")
        if max_passes < 1:
            raise ValueError("max_passes must be positive")

        self.w = source.w
        self.use_ldt = source.use_ldt
        self.enc = copy.deepcopy(source.enc)
        self.ctx = copy.deepcopy(source.ctx)
        self.out = copy.deepcopy(source.out)

        # Initialize on CPU under a saved RNG state.  Conversion can therefore
        # be inserted into a paired experiment without consuming random draws.
        with torch.random.fork_rng(devices=[]):
            target = nn.Linear(self.w, self.w, bias=False, dtype=torch.float32)
            with torch.no_grad():
                target.weight.copy_((source.V @ source.U.T).T.detach().cpu())
            self.core = TiedComponentLinear(
                target,
                c=self.w,
                max_passes=max_passes,
                ablation=AblationConfig(mode="zero", sphere_rescale=False),
                init_svd=False,
            ).to(source.V.device)
        with torch.no_grad():
            self.core.V.copy_(source.V)
            self.core.U.copy_(source.U.T)
        self.core.V.requires_grad_(source.V.requires_grad)
        self.core.U.requires_grad_(source.U.requires_grad)
        self.core.mode = "component"
        self._owner = object()
        self._legacy_last: weakref.ReferenceType[Tensor] | None = None
        self._legacy_final: tuple[int, PassMaskSchedule] | None = None
        self.train(source.training)

    @classmethod
    def from_recurrent(cls, source: Recurrent, *, max_passes: int = 64) -> NativeRecurrent:
        return cls(source, max_passes=max_passes)

    @property
    def V(self) -> nn.Parameter:
        """The historical read-factor view, with one registered parameter."""
        return self.core.V

    @property
    def U(self) -> Tensor:
        """Historical output-column view for HAD-05's ``proj`` helper.

        The optimizer owns ``core.U``.  Its gradient is the transpose of the
        historical ``U`` gradient; this property is a differentiable view.
        """
        return self.core.U.T

    def prepare(self, x: Tensor) -> Tensor:
        if not self.use_ldt:
            x = x.clone()
            x[..., LDT] = 0
        return x

    def _check_history(self, x: Tensor, pass_offset: int) -> None:
        if x.ndim != 4 or x.shape[-1] != F:
            raise ValueError(f"Expected [batch, population, history, {F}] input")
        if min(x.shape[:3]) < 1:
            raise ValueError("Batch, population, and history dimensions must be positive")
        if x.dtype != torch.float32 or self.core.V.dtype != torch.float32:
            raise ValueError("Native HAD-05 execution currently supports float32")
        if x.device != self.core.V.device:
            raise ValueError("Input and model must be on the same device")
        if type(pass_offset) is not int or pass_offset < 0:
            raise ValueError("pass_offset must be a nonnegative integer")

    def _schedule(self, x: Tensor, schedule: PassMaskSchedule | None) -> PassMaskSchedule:
        batch, population, passes = x.shape[:3]
        if schedule is None:
            return PassMaskSchedule(
                comp_mask=x.new_ones((1, passes, 1, self.w)),
                delta_mask=x.new_zeros((1, passes, 1)),
            )

        def check(mask: Tensor, label: str, *, component: bool = False, route: bool = False) -> None:
            expected_ndim = 4 if component else 3
            if (
                mask.ndim != expected_ndim
                or mask.shape[0] not in (1, batch)
                or mask.shape[1] != passes
                or mask.shape[2] not in (1, population)
                or (component and mask.shape[3] != self.w)
            ):
                raise ValueError(f"{label} has an incompatible batch/pass/population/component shape")
            if mask.device != x.device:
                raise ValueError(f"{label} must be on the input device")
            if route:
                if mask.dtype != torch.bool:
                    raise ValueError("routing_mask must be boolean")
            elif mask.dtype != torch.float32 or not bool(
                (torch.isfinite(mask) & (mask >= 0) & (mask <= 1)).all()
            ):
                raise ValueError(f"{label} must contain finite float32 gates in [0, 1]")

        check(schedule.comp_mask, "comp_mask", component=True)
        delta = schedule.delta_mask
        if delta is None:
            delta = x.new_zeros((1, passes, 1))
        else:
            check(delta, "delta_mask")
        if schedule.routing_mask is not None:
            check(schedule.routing_mask, "routing_mask", route=True)
        return PassMaskSchedule(schedule.comp_mask, delta, schedule.routing_mask)

    def _start(self, schedule: PassMaskSchedule, pass_offset: int) -> None:
        if self.core.intervention is not None:
            raise ValueError("Use component masks for this adapter's component-mode execution")
        self.core.reset_pass_state(pass_offset=pass_offset)
        self.core.mode = "component"
        self.core.schedule = schedule

    def _step(self, hidden: Tensor, features: Tensor) -> Tensor:
        return torch.tanh(
            self.enc(features)
            + self.core(hidden)
            + self.ctx(hidden.mean(1, keepdim=True))
        )

    def forward(
        self,
        x: Tensor,
        *,
        schedule: PassMaskSchedule | None = None,
        pass_offset: int = 0,
    ) -> Tensor:
        self._check_history(x, pass_offset)
        x = self.prepare(x)
        self._start(self._schedule(x, schedule), pass_offset)
        hidden = x.new_zeros((*x.shape[:2], self.w))
        for index in range(x.shape[2]):
            hidden = self._step(hidden, x[:, :, index])
        return self.out(hidden).squeeze(-1)

    def boundary(
        self,
        x: Tensor,
        *,
        schedule: PassMaskSchedule | None = None,
        pass_offset: int = 0,
    ) -> HistoryContinuation:
        """Run the history prefix and retain the final pass's exact context."""
        self._check_history(x, pass_offset)
        x = self.prepare(x)
        full_schedule = self._schedule(x, schedule)
        final = x.shape[2] - 1
        self._start(_window(full_schedule, 0, final), pass_offset)
        hidden = x.new_zeros((*x.shape[:2], self.w))
        for index in range(final):
            hidden = self._step(hidden, x[:, :, index])
        return HistoryContinuation(
            last=x[:, :, -1],
            hidden=hidden,
            pass_index=pass_offset + final,
            schedule=_window(full_schedule, final, final + 1),
            owner=self._owner,
        )

    def _finish(
        self, last: Tensor, hidden: Tensor, pass_index: int, schedule: PassMaskSchedule
    ) -> Tensor:
        if hidden.shape != (*last.shape[:2], self.w):
            raise ValueError("Hidden state has an incompatible batch/population/width shape")
        if hidden.dtype != last.dtype or hidden.device != last.device:
            raise ValueError("Hidden state and final features must share dtype and device")
        # Every counterfactual finish is a fresh one-pass execution.  The native
        # wrapper indexes masks locally and calibration statistics globally.
        self._start(schedule, pass_index)
        return self.out(self._step(hidden, last)).squeeze(-1)

    def resume(self, continuation: HistoryContinuation, hidden: Tensor | None = None) -> Tensor:
        """Replay the retained final pass, optionally from an edited state."""
        if continuation.owner is not self._owner:
            raise ValueError("Continuation belongs to another model")
        return self._finish(
            continuation.last,
            continuation.hidden if hidden is None else hidden,
            continuation.pass_index,
            continuation.schedule,
        )

    def mid(
        self,
        x: Tensor,
        *,
        schedule: PassMaskSchedule | None = None,
        pass_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        continuation = self.boundary(x, schedule=schedule, pass_offset=pass_offset)
        # A weak reference preserves the original Tensor interface without
        # retaining the hidden-state graph on the model between minibatches.
        self._legacy_last = weakref.ref(continuation.last)
        self._legacy_final = (continuation.pass_index, continuation.schedule)
        return continuation.last, continuation.hidden

    def finish(self, last: Tensor, hidden: Tensor) -> Tensor:
        if (
            self._legacy_last is None
            or self._legacy_last() is not last
            or self._legacy_final is None
        ):
            raise ValueError("finish requires the latest mid boundary; use boundary/resume for interleaving")
        pass_index, schedule = self._legacy_final
        return self._finish(last, hidden, pass_index, schedule)


def native_from_recurrent(source: Recurrent, *, max_passes: int = 64) -> NativeRecurrent:
    """Copy a historical scheduler without changing its weights or RNG state."""
    return NativeRecurrent.from_recurrent(source, max_passes=max_passes)
