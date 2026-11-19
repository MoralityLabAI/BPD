"""Native TRM training with fixed effective batches and train-only teacher caching.

The sampler draws an entire logical batch before slicing it. Blank/clue losses
use denominators from that whole batch, so accumulation preserves the objective
even when chunks contain different numbers of clues. One optimizer/EMA update
occurs per logical batch. Only fit depths 2/4/6 enter either training loss.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import time

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mechinterp.bpd.adapters.trm import build_mini_trm
from ..dt04.tasks import _DIGIT_PERMS, _ROW_BAND_PERMS
from . import FIT_DEPTHS
from .resources import (
    BoundedTensorCache,
    MemoryBudget,
    ReusableTransferBuffer,
    assert_budget,
    memory_snapshot,
)
from .techniques import OnlineTopKLinear, replace_online_factors


MODEL_CONFIGS = {
    "mini": dict(hidden_size=32, num_heads=4, l_layers=1, h_cycles=1,
                 l_cycles=1, puzzle_emb_len=1, seq_len=16),
    "dt04": dict(hidden_size=96, num_heads=4, l_layers=2, h_cycles=1,
                 l_cycles=2, puzzle_emb_len=2, seq_len=16),
    "full_7m": dict(hidden_size=512, num_heads=8, l_layers=2, h_cycles=3,
                    l_cycles=6, puzzle_emb_len=16, seq_len=81),
}

_ROW_BAND_PERMS_T = torch.tensor(_ROW_BAND_PERMS, dtype=torch.long)
_DIGIT_PERMS_T = torch.tensor(_DIGIT_PERMS, dtype=torch.long)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 480
    logical_batch: int = 128
    microbatch: int = 16
    learning_rate: float = 0.002
    ema_decay: float = 0.995
    precision: str = "fp32"

    def __post_init__(self):
        for name in ("steps", "logical_batch", "microbatch"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.logical_batch % self.microbatch:
            raise ValueError("microbatch must divide the fixed logical batch")
        if not 0 < self.learning_rate < 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("invalid learning rate or EMA decay")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16" or device.type != "cuda":
        raise ValueError("BF16 training requires a CUDA device with BF16 support")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 training requires support on the selected CUDA device")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def tensor_digest(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def model_digest(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor_digest(value).encode())
    return h.hexdigest()


def model_recipe(model: nn.Module) -> dict:
    """Include forward-affecting attributes that do not live in state_dict."""
    return {"class": f"{type(model).__module__}.{type(model).__qualname__}",
            "hidden_size": model.hidden_size, "seq_len": model.seq_len,
            "puzzle_emb_len": model.puzzle_emb_len, "h_cycles": model.h_cycles,
            "l_cycles": model.l_cycles, "layers": len(model.layers),
            "heads": [block.num_heads for block in model.layers],
            "eps": [block.eps for block in model.layers],
            "dtype": str(next(model.parameters()).dtype)}


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        self.decay = decay
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        if state.keys() != self.shadow.keys():
            raise ValueError("model structure changed during EMA")
        float_shadows = []
        float_sources = []
        for key, value in state.items():
            if value.is_floating_point():
                float_shadows.append(self.shadow[key])
                float_sources.append(value.detach())
            else:
                self.shadow[key].copy_(value)
        if float_shadows:
            torch._foreach_lerp_(float_shadows, float_sources, 1.0 - self.decay)
        self.updates += 1

    def copy_model(self, model: nn.Module) -> nn.Module:
        result = copy.deepcopy(model)
        result.load_state_dict(self.shadow, strict=True)
        return result.eval()


def fit_rollout(model: nn.Module, tokens: Tensor) -> tuple[Tensor, Tensor]:
    """Fit-depth logits and zL states only; called with train examples by distillation."""
    emb = model.encode_inputs(tokens)
    zh, zl = model.init_states(len(tokens))
    logits, states = [], []
    digits = int(model.seq_len ** 0.5)
    for depth in range(1, max(FIT_DEPTHS) + 1):
        zh, zl = model.act_step(zh, zl, emb)
        if depth in FIT_DEPTHS:
            logits.append(model.digit_logits(zh)[..., :digits])
            states.append(zl)
    return torch.stack(logits, 1), torch.stack(states, 1)


def _act_step_maybe_checkpoint(model: nn.Module, zh: Tensor, zl: Tensor, emb: Tensor, checkpoint_act: bool):
    if not checkpoint_act:
        return model.act_step(zh, zl, emb)
    return checkpoint(lambda a, b, c: model.act_step(a, b, c), zh, zl, emb, use_reentrant=False)


def backward_supervised_batch(
    model: nn.Module, tokens: Tensor, targets: Tensor, *, microbatch: int,
    precision: str = "fp32", checkpoint_act: bool = False,
) -> float:
    """Accumulate the exact whole-batch blank + 0.25 clue mean objective."""
    if type(microbatch) is not int or microbatch < 1 or len(tokens) % microbatch:
        raise ValueError("microbatch must divide batch size")
    if tokens.shape != targets.shape or tokens.ndim != 2 or len(tokens) == 0:
        raise ValueError("nonempty [batch,cells] tokens and targets required")
    blanks = tokens.eq(1)
    blank_denom = blanks.sum().clamp_min(1)
    clue_denom = (~blanks).sum().clamp_min(1)
    total = torch.zeros((), device=tokens.device)
    digits = int(model.seq_len ** 0.5)
    for offset in range(0, len(tokens), microbatch):
        bx, by = tokens[offset:offset + microbatch], targets[offset:offset + microbatch]
        blank = blanks[offset:offset + microbatch]
        with autocast_context(tokens.device, precision):
            emb = model.encode_inputs(bx)
            zh, zl = model.init_states(len(bx))
            loss = torch.zeros((), device=tokens.device)
            for depth in range(1, max(FIT_DEPTHS) + 1):
                zh, zl = _act_step_maybe_checkpoint(model, zh, zl, emb, checkpoint_act)
                if depth in FIT_DEPTHS:
                    logits = model.digit_logits(zh)[..., :digits]
                    per = F.cross_entropy(logits.reshape(-1, digits), by.reshape(-1), reduction="none").view_as(by)
                    loss = loss + (per * blank).sum() / blank_denom
                    loss = loss + 0.25 * (per * ~blank).sum() / clue_denom
            loss = loss / len(FIT_DEPTHS)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        total += loss.detach()
    return float(total.item())


class GraphedSupervisedExecutor:
    """CUDA-graphed microbatch forward+backward executor preserving whole-batch denominators.

    Captures one fixed-shape ``[microbatch, seq_len]`` chunk and replays it ``logical_batch // microbatch``
    times per update, accumulating into preallocated FP32 parameter gradients.
    """

    def __init__(self, model: nn.Module, microbatch: int, *, precision: str = "fp32"):
        dev = next(model.parameters()).device
        if dev.type != "cuda":
            raise ValueError("GraphedSupervisedExecutor requires a CUDA model")
        self.model = model
        self.dev = dev
        self.microbatch = int(microbatch)
        self.precision = precision
        self.seq_len = int(model.seq_len)
        self.digits = int(self.seq_len ** 0.5)
        self.static_bx = torch.ones((self.microbatch, self.seq_len), dtype=torch.long, device=dev)
        self.static_by = torch.zeros((self.microbatch, self.seq_len), dtype=torch.long, device=dev)
        self.static_blank_denom = torch.tensor(self.microbatch * self.seq_len, dtype=torch.long, device=dev)
        self.static_clue_denom = torch.tensor(1, dtype=torch.long, device=dev)
        self._grads: list[Tensor] = []
        for p in model.parameters():
            if p.requires_grad:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                self._grads.append(p.grad)
        self.graph = torch.cuda.CUDAGraph()
        self.static_loss = torch.zeros((), device=dev)
        self._capture()

    def _forward_backward_body(self) -> Tensor:
        blank = self.static_bx.eq(1)
        with autocast_context(self.dev, self.precision):
            emb = self.model.encode_inputs(self.static_bx)
            zh, zl = self.model.init_states(self.microbatch)
            loss = torch.zeros((), device=self.dev)
            for depth in range(1, max(FIT_DEPTHS) + 1):
                zh, zl = self.model.act_step(zh, zl, emb)
                if depth in FIT_DEPTHS:
                    logits = self.model.digit_logits(zh)[..., :self.digits]
                    per = F.cross_entropy(
                        logits.reshape(-1, self.digits), self.static_by.reshape(-1), reduction="none"
                    ).view_as(self.static_by)
                    loss = loss + (per * blank).sum() / self.static_blank_denom
                    loss = loss + 0.25 * (per * ~blank).sum() / self.static_clue_denom
            loss = loss / len(FIT_DEPTHS)
        loss.backward()
        return loss.detach()

    def _capture(self) -> None:
        prev_warn = torch.autograd.graph._get_warn_on_accumulate_grad_stream_mismatch() if hasattr(
            torch.autograd.graph, "_get_warn_on_accumulate_grad_stream_mismatch"
        ) else True
        if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
            torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        try:
            warmup_stream = torch.cuda.Stream(device=self.dev)
            warmup_stream.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    torch._foreach_zero_(self._grads)
                    tmp_loss = self._forward_backward_body()
                    del tmp_loss
            torch.cuda.current_stream(self.dev).wait_stream(warmup_stream)
            torch._foreach_zero_(self._grads)
            with torch.cuda.graph(self.graph):
                self.static_loss = self._forward_backward_body()
            torch._foreach_zero_(self._grads)
        finally:
            if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
                torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(prev_warn)

    def run(self, tokens: Tensor, targets: Tensor) -> float:
        if len(tokens) % self.microbatch or tokens.shape != targets.shape:
            raise ValueError("GraphedSupervisedExecutor input shape mismatch")
        blanks = tokens.eq(1)
        self.static_blank_denom.copy_(blanks.sum().clamp_min(1))
        self.static_clue_denom.copy_((~blanks).sum().clamp_min(1))
        torch._foreach_zero_(self._grads)
        total = torch.zeros((), device=self.dev)
        for offset in range(0, len(tokens), self.microbatch):
            self.static_bx.copy_(tokens[offset:offset + self.microbatch])
            self.static_by.copy_(targets[offset:offset + self.microbatch])
            self.graph.replay()
            total += self.static_loss
        if not bool(torch.isfinite(total)):
            raise RuntimeError("nonfinite training loss")
        return float(total.item())


def sample_augmented_batch(givens: Tensor, solutions: Tensor, count: int, gen: torch.Generator):
    """CPU sampler: draw once per optimizer update, independent of microbatch size."""
    idx = torch.randint(len(givens), (count,), generator=gen)
    g, s = givens[idx], solutions[idx]
    transpose = torch.randint(2, (count, 1, 1), generator=gen).bool()
    g, s = torch.where(transpose, g.transpose(1, 2), g), torch.where(transpose, s.transpose(1, 2), s)
    permutations = _ROW_BAND_PERMS_T
    rp = permutations[torch.randint(8, (count,), generator=gen)]
    cp = permutations[torch.randint(8, (count,), generator=gen)]
    b = torch.arange(count)[:, None, None]
    g, s = g[b, rp[:, :, None], cp[:, None, :]], s[b, rp[:, :, None], cp[:, None, :]]
    digits = _DIGIT_PERMS_T[torch.randint(24, (count,), generator=gen)]
    return digits.gather(1, g.reshape(count, 16)) + 1, digits.gather(1, s.reshape(count, 16)) - 1


class SudokuSymmetryPipeline:
    """Bounded CPU symmetry sampler with preallocated tables, cache stats, and async CUDA prefetch.

    Preserves the exact RNG draw order and bitwise output of ``sample_augmented_batch``
    while eliminating per-step permutation table construction and host allocation churn.
    """

    def __init__(
        self,
        givens: Tensor,
        solutions: Tensor,
        logical_batch: int,
        device: str | torch.device,
        *,
        budget: MemoryBudget,
        cache: BoundedTensorCache | None = None,
        pin_memory: bool = True,
        prefetch: bool = True,
        orbit_cache_batches: int = 0,
    ):
        self.dev = torch.device(device)
        self.logical_batch = int(logical_batch)
        self.budget = budget
        self.cache = cache if cache is not None else BoundedTensorCache(64 * 1024**2)
        packed_base = torch.stack((givens.contiguous(), solutions.contiguous()), dim=1)
        self.base_key = "sudoku4-base-" + tensor_digest(packed_base)
        self.base_admitted = self.cache.put_admitted(
            self.base_key, packed_base, budget, device=str(self.dev),
        )
        if self.base_admitted:
            cached_copy = self.cache.get(self.base_key)
            self.givens = cached_copy[:, 0].contiguous()
            self.solutions = cached_copy[:, 1].contiguous()
        else:
            self.givens = givens.contiguous()
            self.solutions = solutions.contiguous()
        self.batch_arange = torch.arange(self.logical_batch, dtype=torch.long)[:, None, None]
        self.orbit_cache_batches = max(0, int(orbit_cache_batches))
        self.use_cuda_prefetch = bool(prefetch and self.dev.type == "cuda" and torch.cuda.is_available())
        slots = 2 if self.use_cuda_prefetch else 1
        self.bx_buffers = [
            ReusableTransferBuffer((self.logical_batch, 16), torch.long, self.dev, pin_memory=pin_memory)
            for _ in range(slots)
        ]
        self.by_buffers = [
            ReusableTransferBuffer((self.logical_batch, 16), torch.long, self.dev, pin_memory=pin_memory)
            for _ in range(slots)
        ]
        self.copy_stream = torch.cuda.Stream(device=self.dev) if self.use_cuda_prefetch else None
        self.ready_events = [torch.cuda.Event() for _ in range(slots)] if self.use_cuda_prefetch else []
        self.consumed_events = [torch.cuda.Event() for _ in range(slots)] if self.use_cuda_prefetch else []
        self._prefetched_slot: int | None = None

    def _draw_cpu_into(self, slot: int, gen: torch.Generator, step_index: int) -> None:
        if self.orbit_cache_batches > 0:
            orbit_key = f"{self.base_key}-orbit-{step_index % self.orbit_cache_batches}"
            hit = self.cache.get(orbit_key)
            if hit is not None:
                self.bx_buffers[slot].host.copy_(hit[:, 0])
                self.by_buffers[slot].host.copy_(hit[:, 1])
                return
        count = self.logical_batch
        idx = torch.randint(len(self.givens), (count,), generator=gen)
        g, s = self.givens[idx], self.solutions[idx]
        transpose = torch.randint(2, (count, 1, 1), generator=gen).bool()
        g = torch.where(transpose, g.transpose(1, 2), g)
        s = torch.where(transpose, s.transpose(1, 2), s)
        rp = _ROW_BAND_PERMS_T[torch.randint(8, (count,), generator=gen)]
        cp = _ROW_BAND_PERMS_T[torch.randint(8, (count,), generator=gen)]
        g = g[self.batch_arange, rp[:, :, None], cp[:, None, :]]
        s = s[self.batch_arange, rp[:, :, None], cp[:, None, :]]
        digits = _DIGIT_PERMS_T[torch.randint(24, (count,), generator=gen)]
        bx_cpu = digits.gather(1, g.reshape(count, 16)) + 1
        by_cpu = digits.gather(1, s.reshape(count, 16)) - 1
        if self.orbit_cache_batches > 0:
            orbit_key = f"{self.base_key}-orbit-{step_index % self.orbit_cache_batches}"
            self.cache.put_admitted(
                orbit_key, torch.stack((bx_cpu, by_cpu), dim=1), self.budget, device=str(self.dev),
            )
        self.bx_buffers[slot].host.copy_(bx_cpu)
        self.by_buffers[slot].host.copy_(by_cpu)

    def prime(self, gen: torch.Generator, total_steps: int) -> None:
        if total_steps <= 0 or not self.use_cuda_prefetch:
            return
        self._launch_slot(0, gen, 0, wait_consumed=False)
        self._prefetched_slot = 0

    def _launch_slot(self, slot: int, gen: torch.Generator, step_index: int, *, wait_consumed: bool) -> None:
        if wait_consumed and self.use_cuda_prefetch:
            # Ensure previous GPU compute finished reading device_buf[slot] and hostDMA finished reading host[slot].
            self.consumed_events[slot].synchronize()
        self._draw_cpu_into(slot, gen, step_index)
        self.bx_buffers[slot].stage_and_send(stream=self.copy_stream)
        self.by_buffers[slot].stage_and_send(stream=self.copy_stream)
        if self.use_cuda_prefetch and self.copy_stream is not None:
            self.ready_events[slot].record(self.copy_stream)

    def next_batch(self, step_index: int, total_steps: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        if not self.use_cuda_prefetch:
            self._draw_cpu_into(0, gen, step_index)
            bx = self.bx_buffers[0].stage_and_send()
            by = self.by_buffers[0].stage_and_send()
            return bx, by
        slot = step_index % 2
        if self._prefetched_slot != slot:
            self._launch_slot(slot, gen, step_index, wait_consumed=False)
        compute_stream = torch.cuda.current_stream(self.dev)
        compute_stream.wait_event(self.ready_events[slot])
        bx_dev = self.bx_buffers[slot].device_buf
        by_dev = self.by_buffers[slot].device_buf
        if step_index + 1 < total_steps:
            next_slot = (step_index + 1) % 2
            self._launch_slot(next_slot, gen, step_index + 1, wait_consumed=(step_index + 1 >= 2))
            self._prefetched_slot = next_slot
        else:
            self._prefetched_slot = None
        return bx_dev, by_dev

    def mark_batch_consumed(self, step_index: int) -> None:
        if self.use_cuda_prefetch:
            slot = step_index % 2
            self.consumed_events[slot].record(torch.cuda.current_stream(self.dev))

    def telemetry(self) -> dict:
        return {
            "base_admitted_to_cache": self.base_admitted,
            "pinned_host_memory": self.bx_buffers[0].pinned,
            "async_cuda_prefetch": self.use_cuda_prefetch,
            "reusable_buffer_bytes": sum(b.nbytes for b in (*self.bx_buffers, *self.by_buffers)),
            "h2d_transfers": sum(b.transfers for b in (*self.bx_buffers, *self.by_buffers)),
            "orbit_cache_batches": self.orbit_cache_batches,
            "cache": self.cache.stats(),
        }


def _make_adamw(parameters, *, lr: float, weight_decay: float, dev: torch.device, allow_fused: bool):
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if allow_fused and dev.type == "cuda" and torch.optim.AdamW is getattr(torch.optim, "AdamW"):
        # Only pass fused=True on genuine unpatched torch.optim.AdamW on CUDA.
        if torch.optim.AdamW.__name__ == "AdamW":
            kwargs["fused"] = True
    return torch.optim.AdamW(parameters, **kwargs)


def train_sudoku_model(
    root: int,
    train_cases,
    *,
    config: TrainConfig,
    device: str,
    budget: MemoryBudget,
    profile: str = "dt04",
    execution_mode: str = "reference",
    checkpoint_act: bool = False,
    cuda_graph: bool | None = None,
    cache: BoundedTensorCache | None = None,
):
    if profile not in ("mini", "dt04"):
        raise ValueError("generated Sudoku4 training uses mini or dt04 profile")
    if execution_mode not in ("reference", "optimized"):
        raise ValueError("execution_mode must be reference or optimized")
    dev = torch.device(device)
    torch.manual_seed(root)
    model = build_mini_trm(**MODEL_CONFIGS[profile], seed=root * 17 + 1).to(dev).train()
    use_opt = execution_mode == "optimized"
    opt = _make_adamw(model.parameters(), lr=config.learning_rate, weight_decay=1e-4, dev=dev, allow_fused=use_opt)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.steps)
    ema = ExponentialMovingAverage(model, config.ema_decay)
    givens = torch.tensor([c.givens for c in train_cases], dtype=torch.long)
    solutions = torch.tensor([c.solution for c in train_cases], dtype=torch.long)
    gen = torch.Generator(device="cpu").manual_seed(root * 31 + 5)

    use_graph = bool(
        (dev.type == "cuda" and not checkpoint_act)
        if cuda_graph is None and use_opt
        else (cuda_graph and dev.type == "cuda" and not checkpoint_act)
    )
    graphed_exec = (
        GraphedSupervisedExecutor(model, config.microbatch, precision=config.precision)
        if use_graph else None
    )
    pipeline = (
        SudokuSymmetryPipeline(
            givens, solutions, config.logical_batch, dev,
            budget=budget, cache=cache, pin_memory=True, prefetch=True,
        )
        if use_opt else None
    )
    if pipeline is not None:
        pipeline.prime(gen, config.steps)

    trace = []
    started = time.perf_counter()
    assert_budget(memory_snapshot(device), budget)
    for update in range(config.steps):
        log_due = update == 0 or (update + 1) % 25 == 0 or update + 1 == config.steps
        if not use_opt:
            assert_budget(memory_snapshot(device), budget)
            bx, by = sample_augmented_batch(givens, solutions, config.logical_batch, gen)
            bx, by = bx.to(dev), by.to(dev)
        else:
            bx, by = pipeline.next_batch(update, config.steps, gen)  # type: ignore[union-attr]

        if graphed_exec is not None:
            loss = graphed_exec.run(bx, by)
            if pipeline is not None:
                pipeline.mark_batch_consumed(update)
        else:
            opt.zero_grad(set_to_none=True)
            loss = backward_supervised_batch(
                model, bx, by, microbatch=config.microbatch,
                precision=config.precision, checkpoint_act=checkpoint_act,
            )
            if pipeline is not None:
                pipeline.mark_batch_consumed(update)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        scheduler.step()
        ema.update(model)

        if not use_opt or log_due:
            snap = memory_snapshot(device)
            assert_budget(snap, budget)
            if log_due:
                row = {
                    "update": update + 1,
                    "loss": loss,
                    "elapsed_seconds": time.perf_counter() - started,
                    "memory": snap,
                }
                trace.append(row)
                print(json.dumps({"event": "TRAIN_UPDATE", **row}), flush=True)

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    total_elapsed = time.perf_counter() - started
    model.eval()
    receipt = {
        "config": asdict(config),
        "profile": profile,
        "execution_mode": execution_mode,
        "cuda_graph_used": use_graph,
        "activation_checkpointing": checkpoint_act,
        "updates": config.steps,
        "ema_updates": ema.updates,
        "fit_depths": list(FIT_DEPTHS),
        "elapsed_seconds": total_elapsed,
        "examples_per_second": (config.steps * config.logical_batch) / max(total_elapsed, 1e-9),
        "optimizer_updates_per_second": config.steps / max(total_elapsed, 1e-9),
        "pipeline": pipeline.telemetry() if pipeline is not None else None,
        "trace": trace,
        "train_tokens_sha256": tensor_digest(givens),
        "train_targets_sha256": tensor_digest(solutions),
        "checkpoint_selection": "fixed final raw and EMA both reported; EMA fixed as factor teacher",
    }
    return model, ema.copy_model(model), receipt


def fit_online_factors(
    teacher: nn.Module,
    train_tokens: Tensor,
    *,
    steps: int,
    components: int,
    top_k: int,
    logical_batch: int,
    microbatch: int,
    seed: int,
    budget: MemoryBudget,
    cache: BoundedTensorCache,
    execution_mode: str = "reference",
):
    """Fit V/U on current compressed activations, reusing train-only teacher targets.

    Cache keys bind exact frozen teacher tensors, inputs and fit depths. The cache
    holds logits/zL targets, never teacher-derived component masks. A miss computes
    just the requested microbatch; the same frozen teacher weights are used.
    """
    if (not isinstance(train_tokens, Tensor) or train_tokens.device.type != "cpu"
            or train_tokens.ndim != 2 or train_tokens.dtype != torch.long or len(train_tokens) == 0
            or train_tokens.shape[1] != teacher.seq_len):
        raise ValueError("nonempty CPU int64 [batch,seq_len] train tokens required")
    if (type(logical_batch) is not int or logical_batch < 1 or type(microbatch) is not int
            or microbatch < 1 or type(steps) is not int or steps < 0 or logical_batch % microbatch):
        raise ValueError("invalid factor training steps or microbatch")
    if execution_mode not in ("reference", "optimized"):
        raise ValueError("execution_mode must be reference or optimized")
    if bool(((train_tokens < 0) | (train_tokens >= 11)).any()):
        raise ValueError("train token outside native vocabulary")
    use_opt = execution_mode == "optimized"
    dev = next(teacher.parameters()).device
    device = str(dev)
    frozen_sha = model_digest(teacher)
    teacher_recipe = model_recipe(teacher)
    student = copy.deepcopy(teacher).float().eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    names = [f"layers.{i}.{site}" for i in range(len(student.layers)) for site in ("o_proj", "down_proj")]
    wrappers = replace_online_factors(student, names, components, top_k, mode="online", seed=seed)
    parameters = []
    for wrapper in wrappers.values():
        wrapper.V.requires_grad_(True)
        wrapper.U.requires_grad_(True)
        parameters.extend([wrapper.V, wrapper.U])
    opt = _make_adamw(parameters, lr=0.003, weight_decay=0.01, dev=dev, allow_fused=use_opt)
    digits = int(teacher.seq_len ** 0.5)
    logit_width = teacher.seq_len * digits
    state_width = teacher.total_len * teacher.hidden_size
    packed_width = logit_width + state_width
    key = hashlib.sha256((frozen_sha + json.dumps(teacher_recipe, sort_keys=True)
                         + tensor_digest(train_tokens) + repr(FIT_DEPTHS) + "logits-zL-fp32-v1").encode()).hexdigest()
    needed = len(train_tokens) * len(FIT_DEPTHS) * packed_width * 4
    cached = False
    snap = memory_snapshot(device)
    assert_budget(snap, budget)
    started = time.perf_counter()
    if steps and cache.get_rows(key, torch.zeros(0, dtype=torch.int64)) is not None:
        # Existing cached entry already present in BoundedTensorCache!
        cached = True
    elif steps and cache.can_admit_live(needed, budget, snapshot=snap, device=device, working_copy_multiplier=2.0):
        targets = torch.empty((len(train_tokens), len(FIT_DEPTHS), packed_width), dtype=torch.float32)
        with torch.no_grad():
            for offset in range(0, len(train_tokens), microbatch):
                if not use_opt or offset == 0:
                    assert_budget(memory_snapshot(device), budget)
                tx = train_tokens[offset:offset + microbatch].to(dev)
                logits, states = fit_rollout(teacher, tx)
                targets[offset:offset + len(tx)] = torch.cat([logits.flatten(2), states.flatten(2)], -1).cpu()
        cached = cache.put(key, targets)
        del targets

    # Reusable pinned staging buffers for zero-churn H2D in optimized mode
    tok_stage = ReusableTransferBuffer((microbatch, teacher.seq_len), torch.long, dev, pin_memory=use_opt) if use_opt else None
    tgt_stage = (
        ReusableTransferBuffer((microbatch, len(FIT_DEPTHS), packed_width), torch.float32, dev, pin_memory=True)
        if (use_opt and cached) else None
    )

    gen = torch.Generator(device="cpu").manual_seed(seed * 13 + 7)
    losses = []
    for update in range(steps):
        check_mem = (not use_opt) or update == 0 or (update + 1) % 20 == 0 or update + 1 == steps
        if check_mem:
            assert_budget(memory_snapshot(device), budget)
        indices = torch.randint(len(train_tokens), (logical_batch,), generator=gen)
        opt.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=dev)
        for offset in range(0, logical_batch, microbatch):
            batch_idx = indices[offset:offset + microbatch]
            if tok_stage is not None:
                torch.index_select(train_tokens, 0, batch_idx, out=tok_stage.host[:len(batch_idx)])
                bx = tok_stage.stage_and_send(count=len(batch_idx))
            else:
                bx = train_tokens[batch_idx].to(dev)
            packed_dev = None
            if cached:
                if tgt_stage is not None and len(batch_idx) == microbatch:
                    hit = cache.get_rows_into(key, batch_idx, tgt_stage.host)
                    if hit is not None:
                        packed_dev = tgt_stage.stage_and_send()
                else:
                    packed_cpu = cache.get_rows(key, batch_idx)
                    if packed_cpu is not None:
                        packed_dev = packed_cpu.to(dev)
            with torch.no_grad():
                if packed_dev is None:
                    target_logits, target_states = fit_rollout(teacher, bx)
                else:
                    target_logits = packed_dev[..., :logit_width].reshape(len(bx), len(FIT_DEPTHS), teacher.seq_len, digits)
                    target_states = packed_dev[..., logit_width:].reshape(len(bx), len(FIT_DEPTHS), teacher.total_len, teacher.hidden_size)
                target_logp = F.log_softmax(target_logits, -1)
                target_p = target_logp.exp()
            logits, states = fit_rollout(student, bx)
            kl = (target_p * (target_logp - F.log_softmax(logits, -1))).sum(-1).mean()
            loss = (kl + 0.05 * F.mse_loss(states, target_states)) * (len(bx) / logical_batch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite autonomous factor loss")
            loss.backward()
            total_loss += loss.detach()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        opt.step()
        losses.append(float(total_loss.item()))
        if check_mem:
            assert_budget(memory_snapshot(device), budget)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - started
    if model_digest(teacher) != frozen_sha:
        raise AssertionError("frozen teacher changed during factor fitting")
    return student.eval(), {
        "steps": steps, "components": components, "top_k": top_k,
        "logical_batch": logical_batch, "microbatch": microbatch, "fit_depths": list(FIT_DEPTHS),
        "execution_mode": execution_mode,
        "elapsed_seconds": elapsed,
        "examples_per_second": (steps * logical_batch) / max(elapsed, 1e-9) if steps else 0.0,
        "teacher_sha256": frozen_sha, "teacher_recipe": teacher_recipe,
        "train_tokens_sha256": tensor_digest(train_tokens),
        "routing": "compressed model current activation; no teacher routing schedule",
        "target_cache_admitted": cached, "target_cache_bytes_required": needed,
        "target_cache_key": key, "cache": cache.stats(), "losses": losses,
    }


def routing_control(student: nn.Module, mode: str, seed: int) -> nn.Module:
    result = copy.deepcopy(student)
    site_index = 0
    for name, module in list(result.named_modules()):
        if isinstance(module, OnlineTopKLinear):
            parent_name, _, child = name.rpartition(".")
            parent = result.get_submodule(parent_name) if parent_name else result
            replacement = OnlineTopKLinear.from_factors(module.V, module.U, module.top_k,
                                                       bias=module.bias, mode=mode, seed=seed + site_index)
            setattr(parent, child, replacement)
            site_index += 1
    return result.eval()


def ab_benchmark_training(
    root: int,
    train_cases,
    eval_tokens_cpu: Tensor,
    eval_gold_cpu: Tensor,
    *,
    ref_config: TrainConfig,
    opt_config: TrainConfig,
    device: str,
    budget: MemoryBudget,
    profile: str = "dt04",
    checkpoint_act: bool = False,
    cuda_graph: bool | None = None,
) -> dict:
    """Run controlled A/B comparison of reference vs optimized training under identical sampling."""
    from .evaluation import evaluate_policy

    if (ref_config.steps != opt_config.steps
            or ref_config.logical_batch != opt_config.logical_batch
            or ref_config.learning_rate != opt_config.learning_rate
            or ref_config.ema_decay != opt_config.ema_decay):
        raise ValueError("A/B training comparison requires identical steps, logical_batch, lr, and EMA decay")

    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    ref_raw, ref_ema, ref_rec = train_sudoku_model(
        root, train_cases, config=ref_config, device=device, budget=budget,
        profile=profile, execution_mode="reference",
    )
    ref_mem = memory_snapshot(device)
    ref_eval = evaluate_policy(ref_ema, eval_tokens_cpu, eval_gold_cpu, microbatch=ref_config.microbatch, budget=budget)

    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    cache = BoundedTensorCache(128 * 1024**2)
    opt_raw, opt_ema, opt_rec = train_sudoku_model(
        root, train_cases, config=opt_config, device=device, budget=budget,
        profile=profile, execution_mode="optimized", checkpoint_act=checkpoint_act,
        cuda_graph=cuda_graph, cache=cache,
    )
    opt_mem = memory_snapshot(device)
    opt_eval = evaluate_policy(opt_ema, eval_tokens_cpu, eval_gold_cpu, microbatch=opt_config.microbatch, budget=budget)

    ref_losses = [r["loss"] for r in ref_rec["trace"]]
    opt_losses = [r["loss"] for r in opt_rec["trace"]]
    loss_diffs = [abs(a - b) for a, b in zip(ref_losses, opt_losses)]
    param_max_diff = max(
        float((p1.detach().cpu() - p2.detach().cpu()).abs().max().item())
        for p1, p2 in zip(ref_ema.parameters(), opt_ema.parameters())
    )
    ref_d8 = ref_eval["by_depth"]["8"]["exact_board_accuracy"]
    opt_d8 = opt_eval["by_depth"]["8"]["exact_board_accuracy"]
    del ref_raw, ref_ema, opt_raw, opt_ema
    return {
        "schema": "BPD-DT05/ab-training-benchmark/v1",
        "root": root,
        "profile": profile,
        "device": device,
        "steps": ref_config.steps,
        "logical_batch": ref_config.logical_batch,
        "reference": {
            "config": asdict(ref_config),
            "elapsed_seconds": ref_rec["elapsed_seconds"],
            "examples_per_second": ref_rec["examples_per_second"],
            "optimizer_updates_per_second": ref_rec["optimizer_updates_per_second"],
            "peak_host_rss_bytes": ref_mem["host_peak_rss_bytes"],
            "current_host_rss_bytes": ref_mem["host_rss_bytes"],
            "peak_cuda_reserved_bytes": ref_mem["cuda_peak_reserved_bytes"],
            "peak_cuda_allocated_bytes": ref_mem["cuda_peak_allocated_bytes"],
            "ema_depth8_exact_board": ref_d8,
            "ema_first_valid_exact_board": ref_eval["first_valid"]["exact_board_accuracy"],
            "final_loss": ref_losses[-1],
        },
        "optimized": {
            "config": asdict(opt_config),
            "cuda_graph_used": opt_rec["cuda_graph_used"],
            "activation_checkpointing": opt_rec["activation_checkpointing"],
            "elapsed_seconds": opt_rec["elapsed_seconds"],
            "examples_per_second": opt_rec["examples_per_second"],
            "optimizer_updates_per_second": opt_rec["optimizer_updates_per_second"],
            "peak_host_rss_bytes": opt_mem["host_peak_rss_bytes"],
            "current_host_rss_bytes": opt_mem["host_rss_bytes"],
            "peak_cuda_reserved_bytes": opt_mem["cuda_peak_reserved_bytes"],
            "peak_cuda_allocated_bytes": opt_mem["cuda_peak_allocated_bytes"],
            "pipeline": opt_rec["pipeline"],
            "ema_depth8_exact_board": opt_d8,
            "ema_first_valid_exact_board": opt_eval["first_valid"]["exact_board_accuracy"],
            "final_loss": opt_losses[-1],
        },
        "speedup_ratio": ref_rec["elapsed_seconds"] / max(opt_rec["elapsed_seconds"], 1e-9),
        "numerical_parity": {
            "same_precision": ref_config.precision == opt_config.precision,
            "max_trace_loss_abs_diff": max(loss_diffs) if loss_diffs else 0.0,
            "mean_trace_loss_abs_diff": sum(loss_diffs) / max(1, len(loss_diffs)),
            "ema_parameter_max_abs_diff": param_max_diff,
            "depth8_exact_board_delta": opt_d8 - ref_d8,
            "trajectory_difference_flagged": bool(
                (max(loss_diffs) if loss_diffs else 0.0) > (1e-4 if ref_config.precision == opt_config.precision else 5e-2)
            ),
        },
    }
