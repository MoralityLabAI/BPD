"""INK-BPD-02: executable PyTorch recurrent block routing, patched before full continuation.

The trained model is a small PUBLIC SYNTHETIC POSITIVE-CONTROL, not Inkling
weights. One physical writer, feedback and core module execute across passes.
The hook acts on a real tensor immediately before the next block reader.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class Batch:
    bits: Tensor
    query: Tensor
    noise: Tensor

    @property
    def labels(self) -> Tensor:
        return self.bits.gather(1, self.query[:, None]).squeeze(1)


def make_batch(seed: int, n: int) -> Batch:
    if n < 1:
        raise ValueError("batch size must be positive")
    g = torch.Generator(device="cpu").manual_seed(seed)
    return Batch(
        bits=torch.randint(0, 2, (n, 2), generator=g),
        query=torch.randint(0, 2, (n,), generator=g),
        noise=torch.randn((n, 2), generator=g) * .025,
    )


def delivered_rotation(x: Tensor, pair: tuple[int, int], dose: float) -> tuple[Tensor, Tensor]:
    """Rotate physical writer output by fixed per-sample L2 distance.

    Ineligible samples become no-op here and MUST be excluded from any scored
    matched comparison using the pre-registered joint-admission mask.
    """
    if x.ndim != 2:
        raise ValueError("writer tensor must have shape [batch, width]")
    i, j = pair
    if i == j or min(i, j) < 0 or max(i, j) >= x.shape[1]:
        raise ValueError("invalid coordinate pair")
    if isinstance(dose, bool) or not math.isfinite(dose) or dose <= 0:
        raise ValueError("positive finite physical dose required")
    radius = torch.linalg.vector_norm(x[:, [i, j]], dim=-1)
    admitted = radius >= dose * .50001
    angle = 2 * torch.asin(torch.clamp(dose / (2 * radius.clamp_min(1e-12)), max=1.))
    cosine, sine = torch.cos(angle), torch.sin(angle)
    xi, xj = x[:, i], x[:, j]
    y = x.clone()
    y[:, i] = torch.where(admitted, cosine * xi - sine * xj, xi)
    y[:, j] = torch.where(admitted, sine * xi + cosine * xj, xj)
    return y, admitted


class TinyCrossLoopRouter(nn.Module):
    """Learned two-slot pointer router with native recurrent continuation.

    This is a miniature architectural analog, with a shared *physical* writer
    at every pass and query-dependent dot-product selection. Its layers are
    ordinary torch modules; a changed predecessor state propagates through
    later feedback and readout rather than being algebraically rescored.
    """
    def __init__(self, width: int = 16, loops: int = 3) -> None:
        super().__init__()
        if width < 8 or loops < 2:
            raise ValueError("width >= 8 and loops >= 2 required")
        self.width, self.loops = width, loops
        self.writer = nn.Linear(4, width)
        self.reader = nn.Embedding(2, width)
        self.feedback = nn.Linear(width, width, bias=False)
        self.core = nn.Linear(width, width)
        self.mix = nn.Linear(width, width)
        self.head = nn.Linear(width, 2)

    def _source(self, batch: Batch) -> Tensor:
        bit = batch.bits.float() * 2 - 1
        n = bit.shape[0]
        z = torch.zeros_like(bit)
        one = torch.ones_like(bit)
        # One content bit plus distinct slot identity for the shared writer.
        first = torch.stack([bit[:, 0] + batch.noise[:, 0], one[:, 0], z[:, 0], one[:, 0]], -1)
        second = torch.stack([bit[:, 1] + batch.noise[:, 1], z[:, 1], one[:, 1], one[:, 1]], -1)
        return torch.stack([first, second], dim=1).reshape(n, 2, 4)

    def forward(self, batch: Batch, *, hook_pass: int = 1,
                pair: tuple[int, int] | None = None, dose: float = .16,
                mode: str = "full", gate_control: str = "none") -> dict[str, Tensor]:
        if mode not in ("full", "selector_only", "content_only"):
            raise ValueError("mode must be full/selector_only/content_only")
        if gate_control not in ("none", "invert_at_hook", "invert_all"):
            raise ValueError("gate_control must be none/invert_at_hook/invert_all")
        if pair is not None and gate_control != "none":
            raise ValueError("gate-inversion control cannot be mixed with physical writer patch")
        if pair is not None and not 0 <= hook_pass < self.loops:
            raise ValueError("hook pass out of range")
        src = self._source(batch)
        state = torch.zeros((src.shape[0], self.width), dtype=src.dtype, device=src.device)
        q = self.reader(batch.query)
        dose_valid = torch.ones(src.shape[0], dtype=torch.bool, device=state.device)
        betas: list[Tensor] = []
        saved: list[Tensor] = []
        readout: list[Tensor] = []
        at_hook: dict[str, Tensor] = {}
        for step in range(self.loops):
            # Identical physical writer weights execute in every loop.
            written = self.writer(src) + .12 * self.feedback(state).unsqueeze(1)
            old3 = written[:, 1]
            reader3, content3 = old3, old3
            if pair is not None and step == hook_pass:
                patched3, dose_valid = delivered_rotation(old3, pair, dose)
                if mode in ("full", "selector_only"):
                    reader3 = patched3
                if mode in ("full", "content_only"):
                    content3 = patched3
                at_hook = {"clean3":old3, "patched3":patched3,
                           "clean2":written[:, 0]}
            choose = torch.stack([written[:, 0], reader3], 1)
            content = torch.stack([written[:, 0], content3], 1)
            normalized = F.normalize(choose, p=2, dim=-1) * math.sqrt(self.width)
            scores = (normalized * q[:, None, :]).sum(-1) / math.sqrt(self.width)
            gate = F.softmax(scores, dim=-1)
            # Deliberately strong DIRECT gate intervention. This is an
            # unmatched-dose positive control and is scored separately from
            # the physically dose-matched writer interventions.
            if gate_control == "invert_all" or (gate_control == "invert_at_hook" and step == hook_pass):
                gate = torch.flip(gate, dims=(-1,))
            message = (content * gate[:, :, None]).sum(1)
            state = torch.tanh(.35 * self.core(state) + self.mix(message))
            betas.append(gate[:, 1])
            saved.append(state)
            readout.append(self.head(state))
        return {"logits":readout[-1], "intermediate_logits":torch.stack(readout),
                "beta3":torch.stack(betas), "states":torch.stack(saved),
                "admitted":dose_valid, **at_hook}


def train_source(*, seed: int = 2701, steps: int = 260,
                 width: int = 16, loops: int = 3, batch_size: int = 192,
                 threads: int = 2) -> tuple[TinyCrossLoopRouter, dict]:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model=TinyCrossLoopRouter(width=width,loops=loops)
    opt=torch.optim.AdamW(model.parameters(), lr=.008, weight_decay=.0001)
    model.train()
    for step in range(steps):
        batch=make_batch(seed*100000+step, batch_size)
        result=model(batch)
        loss=F.cross_entropy(result["logits"],batch.labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        batch=make_batch(seed*1000000+7,256)
        logits=model(batch)["logits"]
        accuracy=(logits.argmax(-1)==batch.labels).float().mean().item()
    return model, {"seed":seed,"steps":steps,"source_validation_accuracy":accuracy,
                   "parameters":sum(p.numel() for p in model.parameters()),
                   "width":width,"loops":loops, "threads":threads}
