"""Native BPD factorized, two-state vector TRM task adapter.

One block is shared across all tokens and both high/low visits. This is a
streaming vector adaptation of the TRM recurrence, not the attention TRM.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mechinterp.bpd.components import PassMaskSchedule, TiedComponentLinear
from mechinterp.bpd.config import AblationConfig


class ComponentTRM(nn.Module):
    def __init__(self, width: int = 32, components: int = 29, low_cycles: int = 1,
                 input_dim: int = 11, values: int = 4) -> None:
        super().__init__()
        if min(width, components, low_cycles, input_dim, values) < 1 or components > width:
            raise ValueError('invalid TRM dimensions')
        self.width, self.components = width, components
        self.low_cycles, self.input_dim = low_cycles, input_dim
        self.values = values
        self.input = nn.Linear(input_dim, width)
        self.up = TiedComponentLinear(
            nn.Linear(width, 2 * width, bias=False),
            components,
            max_passes=512,
            ablation=AblationConfig(mode='zero', sphere_rescale=False),
            init_svd=True,
        )
        self.down = nn.Linear(2 * width, width, bias=False)
        self.norm = nn.RMSNorm(width)
        self.output = nn.Linear(width, values)

    def block(self, z: Tensor, *, write_patch: Tensor | None = None) -> Tensor:
        up_out = self.up(z)
        if write_patch is not None:
            up_out = up_out + write_patch
        return self.norm(z + self.down(F.silu(up_out)))

    def _validate_state_intervention(self, intervention: Tensor, b: int, t: int,
                                     x: Tensor, patch_step: int) -> None:
        if (intervention.shape != (b, 2 * self.width) or intervention.dtype != x.dtype
                or intervention.device != x.device or not bool(torch.isfinite(intervention).all())
                or type(patch_step) is not int or not 0 <= patch_step < t):
            raise ValueError('invalid joint-state intervention')

    def _validate_write_patch(self, write_patch: Tensor, b: int, t: int,
                              x: Tensor, patch_step: int, write_site: str) -> None:
        if (write_patch.shape != (b, 2 * self.width) or write_patch.dtype != x.dtype
                or write_patch.device != x.device or not bool(torch.isfinite(write_patch).all())
                or type(patch_step) is not int or not 0 <= patch_step < t
                or write_site not in ('low', 'high', 'both')):
            raise ValueError('invalid native write-site intervention')

    def rollout(self, x: Tensor, patch: Tensor | None = None, patch_step: int = 4,
                state_replacement: Tensor | None = None,
                write_patch: Tensor | None = None,
                write_site: str = 'low') -> dict[str, Tensor | int]:
        if (x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1
                or x.shape[2] != self.input_dim or not x.is_floating_point()):
            raise ValueError('expected a nonempty feature sequence')
        if patch is not None and state_replacement is not None:
            raise ValueError('exclusive interventions')
        intervention = patch if patch is not None else state_replacement
        b, t, _ = x.shape
        if intervention is not None:
            self._validate_state_intervention(intervention, b, t, x, patch_step)
        if write_patch is not None:
            self._validate_write_patch(write_patch, b, t, x, patch_step, write_site)
        visits = t * (self.low_cycles + 1)
        self.up.reset_pass_state(pass_offset=0)
        self.up.mode = 'component'
        self.up.schedule = PassMaskSchedule(
            comp_mask=x.new_ones((b, visits, self.components)),
            delta_mask=x.new_zeros((b, visits)),
        )
        zh = x.new_zeros((b, self.width))
        zl = torch.zeros_like(zh)
        e = self.input(x)
        states, low, high = [], [], []
        try:
            for j in range(t):
                if intervention is not None and j == patch_step:
                    state = (intervention if state_replacement is not None
                             else torch.cat((zh, zl), -1) + patch)
                    zh, zl = state.chunk(2, dim=-1)
                for cyc in range(self.low_cycles):
                    wp_low = (write_patch if (write_patch is not None and j == patch_step
                              and cyc == 0 and write_site in ('low', 'both')) else None)
                    zl = self.block(zl + zh + e[:, j], write_patch=wp_low)
                    low.append(zl)
                wp_high = (write_patch if (write_patch is not None and j == patch_step
                           and write_site in ('high', 'both')) else None)
                zh = self.block(zh + zl, write_patch=wp_high)
                high.append(zh)
                states.append(torch.cat((zh, zl), -1))
        finally:
            self.up.schedule = None
        return dict(
            logits=self.output(zh),
            states=torch.stack(states, 1),
            high_states=torch.stack(high, 1),
            low_states=torch.stack(low, 1),
            block_calls=self.up.calls,
        )

    def rollout_suffix(self, x_suffix: Tensor, init_state: Tensor, *,
                       prefix_steps: int = 4, patch: Tensor | None = None,
                       patch_step: int = 0, write_patch: Tensor | None = None,
                       write_site: str = 'low') -> dict[str, Tensor | int]:
        """Exact suffix-only continuation from a post-prefix joint state [z_H, z_L].

        Reuses the post-write joint state without re-executing the prefix while
        preserving exact TiedComponentLinear pass_offset parity, logits, states
        and autograd gradients with respect to both init_state and parameters.
        """
        if (x_suffix.ndim != 3 or x_suffix.shape[0] < 1 or x_suffix.shape[1] < 1
                or x_suffix.shape[2] != self.input_dim or not x_suffix.is_floating_point()):
            raise ValueError('expected a nonempty suffix feature sequence')
        if type(prefix_steps) is not int or prefix_steps < 0:
            raise ValueError('prefix_steps must be a nonnegative int')
        b, t, _ = x_suffix.shape
        if (init_state.shape != (b, 2 * self.width) or init_state.dtype != x_suffix.dtype
                or init_state.device != x_suffix.device
                or not bool(torch.isfinite(init_state).all())):
            raise ValueError('invalid initial joint state for suffix continuation')
        if patch is not None:
            self._validate_state_intervention(patch, b, t, x_suffix, patch_step)
        if write_patch is not None:
            self._validate_write_patch(write_patch, b, t, x_suffix, patch_step, write_site)

        visits = t * (self.low_cycles + 1)
        pass_offset = prefix_steps * (self.low_cycles + 1)
        self.up.reset_pass_state(pass_offset=pass_offset)
        self.up.mode = 'component'
        self.up.schedule = PassMaskSchedule(
            comp_mask=x_suffix.new_ones((b, visits, self.components)),
            delta_mask=x_suffix.new_zeros((b, visits)),
        )
        zh, zl = init_state.chunk(2, dim=-1)
        e = self.input(x_suffix)
        states, low, high = [], [], []
        try:
            for j in range(t):
                if patch is not None and j == patch_step:
                    state = torch.cat((zh, zl), -1) + patch
                    zh, zl = state.chunk(2, dim=-1)
                for cyc in range(self.low_cycles):
                    wp_low = (write_patch if (write_patch is not None and j == patch_step
                              and cyc == 0 and write_site in ('low', 'both')) else None)
                    zl = self.block(zl + zh + e[:, j], write_patch=wp_low)
                    low.append(zl)
                wp_high = (write_patch if (write_patch is not None and j == patch_step
                           and write_site in ('high', 'both')) else None)
                zh = self.block(zh + zl, write_patch=wp_high)
                high.append(zh)
                states.append(torch.cat((zh, zl), -1))
        finally:
            self.up.schedule = None
        return dict(
            logits=self.output(zh),
            states=torch.stack(states, 1),
            high_states=torch.stack(high, 1),
            low_states=torch.stack(low, 1),
            block_calls=self.up.calls,
            pass_offset=pass_offset,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.rollout(x)['logits']

    def prefix_state(self, x: Tensor, prefix_steps: int = 4) -> Tensor:
        if type(prefix_steps) is not int or not 1 <= prefix_steps <= x.shape[1]:
            raise ValueError('invalid prefix')
        return self.rollout(x[:, :prefix_steps])['states'][:, -1]

    def native_component_operator(self) -> Tensor:
        """Return exact physical operator W = V @ U of shape [width, 2*width]."""
        return self.up.V @ self.up.U

    def factor_frame_digest(self) -> str:
        """SHA-256 digest binding the current physical (V, U) factor frame."""
        import hashlib
        h = hashlib.sha256()
        for name, t in (('V', self.up.V), ('U', self.up.U)):
            h.update(name.encode('ascii'))
            h.update(str(tuple(t.shape)).encode('ascii'))
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
