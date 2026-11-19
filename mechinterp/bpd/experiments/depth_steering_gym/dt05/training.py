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

from mechinterp.bpd.adapters.trm import build_mini_trm
from ..dt04.tasks import _DIGIT_PERMS, _ROW_BAND_PERMS
from . import FIT_DEPTHS
from .resources import BoundedTensorCache, MemoryBudget, assert_budget, memory_snapshot
from .techniques import OnlineTopKLinear, replace_online_factors


MODEL_CONFIGS = {
    "mini": dict(hidden_size=32, num_heads=4, l_layers=1, h_cycles=1,
                 l_cycles=1, puzzle_emb_len=1, seq_len=16),
    "dt04": dict(hidden_size=96, num_heads=4, l_layers=2, h_cycles=1,
                 l_cycles=2, puzzle_emb_len=2, seq_len=16),
    "full_7m": dict(hidden_size=512, num_heads=8, l_layers=2, h_cycles=3,
                    l_cycles=6, puzzle_emb_len=16, seq_len=81),
}


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
        for key, value in state.items():
            if value.is_floating_point():
                self.shadow[key].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)
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


def backward_supervised_batch(
    model: nn.Module, tokens: Tensor, targets: Tensor, *, microbatch: int,
    precision: str = "fp32",
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
                zh, zl = model.act_step(zh, zl, emb)
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


def sample_augmented_batch(givens: Tensor, solutions: Tensor, count: int, gen: torch.Generator):
    """CPU sampler: draw once per optimizer update, independent of microbatch size."""
    idx = torch.randint(len(givens), (count,), generator=gen)
    g, s = givens[idx], solutions[idx]
    transpose = torch.randint(2, (count, 1, 1), generator=gen).bool()
    g, s = torch.where(transpose, g.transpose(1, 2), g), torch.where(transpose, s.transpose(1, 2), s)
    permutations = torch.tensor(_ROW_BAND_PERMS)
    rp = permutations[torch.randint(8, (count,), generator=gen)]
    cp = permutations[torch.randint(8, (count,), generator=gen)]
    b = torch.arange(count)[:, None, None]
    g, s = g[b, rp[:, :, None], cp[:, None, :]], s[b, rp[:, :, None], cp[:, None, :]]
    digits = torch.tensor(_DIGIT_PERMS)[torch.randint(24, (count,), generator=gen)]
    return digits.gather(1, g.reshape(count, 16)) + 1, digits.gather(1, s.reshape(count, 16)) - 1


def train_sudoku_model(root: int, train_cases, *, config: TrainConfig,
                       device: str, budget: MemoryBudget, profile: str = "dt04"):
    if profile not in ("mini", "dt04"):
        raise ValueError("generated Sudoku4 training uses mini or dt04 profile")
    torch.manual_seed(root)
    model = build_mini_trm(**MODEL_CONFIGS[profile], seed=root * 17 + 1).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.steps)
    ema = ExponentialMovingAverage(model, config.ema_decay)
    givens = torch.tensor([c.givens for c in train_cases], dtype=torch.long)
    solutions = torch.tensor([c.solution for c in train_cases], dtype=torch.long)
    gen = torch.Generator(device="cpu").manual_seed(root * 31 + 5)
    trace = []
    started = time.perf_counter()
    for update in range(config.steps):
        assert_budget(memory_snapshot(device), budget)
        bx, by = sample_augmented_batch(givens, solutions, config.logical_batch, gen)
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad(set_to_none=True)
        loss = backward_supervised_batch(model, bx, by, microbatch=config.microbatch, precision=config.precision)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        scheduler.step()
        ema.update(model)
        snap = memory_snapshot(device)
        assert_budget(snap, budget)
        if update == 0 or (update + 1) % 25 == 0 or update + 1 == config.steps:
            row = {"update": update + 1, "loss": loss, "elapsed_seconds": time.perf_counter() - started,
                   "memory": snap}
            trace.append(row)
            print(json.dumps({"event": "TRAIN_UPDATE", **row}), flush=True)
    model.eval()
    return model, ema.copy_model(model), {
        "config": asdict(config), "profile": profile, "updates": config.steps,
        "ema_updates": ema.updates, "fit_depths": list(FIT_DEPTHS), "trace": trace,
        "train_tokens_sha256": tensor_digest(givens), "train_targets_sha256": tensor_digest(solutions),
        "checkpoint_selection": "fixed final raw and EMA both reported; EMA fixed as factor teacher",
    }


def fit_online_factors(teacher: nn.Module, train_tokens: Tensor, *, steps: int,
                       components: int, top_k: int, logical_batch: int, microbatch: int,
                       seed: int, budget: MemoryBudget, cache: BoundedTensorCache):
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
    if bool(((train_tokens < 0) | (train_tokens >= 11)).any()):
        raise ValueError("train token outside native vocabulary")
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
    opt = torch.optim.AdamW(parameters, lr=0.003)
    digits = int(teacher.seq_len ** 0.5)
    logit_width = teacher.seq_len * digits
    state_width = teacher.total_len * teacher.hidden_size
    key = hashlib.sha256((frozen_sha + json.dumps(teacher_recipe, sort_keys=True)
                         + tensor_digest(train_tokens) + repr(FIT_DEPTHS) + "logits-zL-fp32-v1").encode()).hexdigest()
    needed = len(train_tokens) * len(FIT_DEPTHS) * (logit_width + state_width) * 4
    cached = False
    # Peak host allocation includes a working tensor and the immutable cache copy.
    snap = memory_snapshot(device)
    assert_budget(snap, budget)
    fits_host = (snap["host_rss_bytes"] + 2 * needed <= budget.max_host_rss_gib * 1024**3
                 and snap["host_available_bytes"] - 2 * needed >= budget.min_host_available_gib * 1024**3)
    if steps and needed <= cache.max_bytes and fits_host:
        targets = torch.empty((len(train_tokens), len(FIT_DEPTHS), logit_width + state_width))
        with torch.no_grad():
            for offset in range(0, len(train_tokens), microbatch):
                assert_budget(memory_snapshot(device), budget)
                tx = train_tokens[offset:offset + microbatch].to(dev)
                logits, states = fit_rollout(teacher, tx)
                targets[offset:offset + len(tx)] = torch.cat([logits.flatten(2), states.flatten(2)], -1).cpu()
        cached = cache.put(key, targets)
        del targets
    gen = torch.Generator(device="cpu").manual_seed(seed * 13 + 7)
    losses = []
    for update in range(steps):
        assert_budget(memory_snapshot(device), budget)
        indices = torch.randint(len(train_tokens), (logical_batch,), generator=gen)
        opt.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=dev)
        for offset in range(0, logical_batch, microbatch):
            batch_idx = indices[offset:offset + microbatch]
            bx = train_tokens[batch_idx].to(dev)
            packed = cache.get_rows(key, batch_idx) if cached else None
            with torch.no_grad():
                if packed is None:
                    target_logits, target_states = fit_rollout(teacher, bx)
                else:
                    packed = packed.to(dev)
                    target_logits = packed[..., :logit_width].reshape(len(bx), len(FIT_DEPTHS), teacher.seq_len, digits)
                    target_states = packed[..., logit_width:].reshape(len(bx), len(FIT_DEPTHS), teacher.total_len, teacher.hidden_size)
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
        assert_budget(memory_snapshot(device), budget)
    if model_digest(teacher) != frozen_sha:
        raise AssertionError("frozen teacher changed during factor fitting")
    return student.eval(), {
        "steps": steps, "components": components, "top_k": top_k,
        "logical_batch": logical_batch, "microbatch": microbatch, "fit_depths": list(FIT_DEPTHS),
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
