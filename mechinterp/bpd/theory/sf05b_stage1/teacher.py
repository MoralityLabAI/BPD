"""SF05B Stage 1 trained shared recurrent teacher and physical BPD factor fit.

All teacher/factor/policy training is restricted to Stage-0 F_train. Heldout
event-family diagnostics run only after the Stage-1 protocol and source lock.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from mechinterp.bpd.theory.sf05b_source_families.source_families import (
    HistoryBatch, MAX_WRITES, WIDTH as EVENT_WIDTH, make_batch,
)

HIDDEN = 32
INNER = 64
COMPONENTS = 32
TEACHER_UPDATES = 2600
FACTOR_UPDATES = 450
BATCH_SIZE = 128


def generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


@dataclass
class States:
    before_last: torch.Tensor
    prequery: torch.Tensor
    final_output: torch.Tensor
    intermediate_readout: torch.Tensor


class Teacher(nn.Module):
    """A single shared nonlinear physical transition on write and query events."""

    def __init__(self, seed: int):
        super().__init__()
        g = generator(seed + 1741)
        self.A = nn.Parameter(torch.randn(HIDDEN, INNER, generator=g)*.1)
        self.B = nn.Parameter(torch.randn(EVENT_WIDTH, INNER, generator=g)*.1)
        self.bias = nn.Parameter(torch.zeros(INNER))
        self.C = nn.Parameter(torch.randn(INNER, HIDDEN, generator=g)*.1)

    def features(self, state: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        return torch.tanh(state @ self.A + event @ self.B + self.bias)

    def step(self, state: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        return state + self.features(state, event) @ self.C

    def rollout(self, batch: HistoryBatch, events: torch.Tensor | None = None) -> States:
        ev = batch.events if events is None else events
        if ev.shape != batch.events.shape:
            raise ValueError("source-donor event shape must match recipient")
        state = torch.zeros(len(batch.target), HIDDEN, device=ev.device, dtype=ev.dtype)
        before_last = torch.zeros_like(state)
        visible = []
        for j in range(MAX_WRITES):
            is_last = j == batch.lengths-1
            before_last = torch.where(is_last[:, None], state, before_last)
            new_state = self.step(state, ev[:, j, :])
            state = torch.where(batch.event_mask[:, j, None], new_state, state)
            visible.append(state[:, 0])
        final = self.step(state, batch.query)
        return States(before_last, state, final[:, 0], torch.stack(visible, dim=1))


class Factors(nn.Module):
    """Full-rank, trainable physical V,U with no learned sparse gates."""

    def __init__(self, seed: int):
        super().__init__()
        g = generator(seed + 88200)
        self.V = nn.Parameter(torch.randn(INNER, COMPONENTS, generator=g)*.1)
        self.U = nn.Parameter(torch.randn(COMPONENTS, HIDDEN, generator=g)*.1)

    def weight(self) -> torch.Tensor:
        return self.V @ self.U


def train_teacher(seed: int, updates: int=TEACHER_UPDATES,
                  batch_size: int=BATCH_SIZE) -> Teacher:
    torch.set_num_threads(1)
    teacher = Teacher(seed)
    opt = torch.optim.Adam(teacher.parameters(), lr=.012)
    for update in range(updates):
        batch = make_batch(seed*100000+update, batch_size, "train")
        states = teacher.rollout(batch)
        final_loss = (states.final_output-batch.target).square().mean()
        early_loss = ((states.intermediate_readout*batch.event_mask).square().sum()
                      / batch.event_mask.sum().clamp_min(1))
        loss = (4*final_loss + .15*early_loss
                + .0001*(teacher.A.square().mean()+teacher.C.square().mean()))
        if not torch.isfinite(loss):
            raise RuntimeError(f"teacher nonfinite loss at seed {seed}, update {update}")
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(teacher.parameters(), 5.)
        opt.step()
    teacher.eval().requires_grad_(False)
    return teacher


def train_factors(seed: int, teacher: Teacher, updates: int=FACTOR_UPDATES,
                  batch_size: int=BATCH_SIZE) -> Factors:
    factors = Factors(seed)
    opt = torch.optim.Adam(factors.parameters(), lr=.035)
    true_weight = teacher.C.detach()
    for update in range(updates):
        batch = make_batch(seed*170000 + 300000 + update, batch_size, "train")
        with torch.no_grad():
            state = torch.zeros(batch_size, HIDDEN)
            features = []
            for j in range(MAX_WRITES):
                f = teacher.features(state, batch.events[:, j, :])
                features.append(f*batch.event_mask[:, j, None])
                proposed = state + f@true_weight
                state = torch.where(batch.event_mask[:, j, None], proposed, state)
            features.append(teacher.features(state, batch.query))
            feature_matrix = torch.cat(features, dim=0)
        reconstruction = factors.weight()
        loss = (1.5*(reconstruction-true_weight).square().mean()
                + (feature_matrix @ reconstruction
                   - feature_matrix @ true_weight).square().mean())
        if not torch.isfinite(loss):
            raise RuntimeError(f"factor nonfinite loss at seed {seed}, update {update}")
        opt.zero_grad()
        loss.backward()
        opt.step()
    factors.eval().requires_grad_(False)
    return factors


def recipient_donor_edit(teacher: Teacher, factors: Factors, batch: HistoryBatch,
                         donor_events: torch.Tensor, states: States
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return donor physical state, readout-dark factor edit and feature difference."""
    row = torch.arange(len(batch.target))
    last = batch.lengths-1
    rec_event = batch.events[row, last]
    donor_event = donor_events[row, last]
    with torch.no_grad():
        feature_delta = (teacher.features(states.before_last, donor_event)
                         - teacher.features(states.before_last, rec_event))
        physical_write = feature_delta @ factors.weight()
        raw_dark = physical_write.clone()
        raw_dark[:, 0] = 0
        donor_state = teacher.step(states.before_last, donor_event)
        if not torch.equal((states.prequery + raw_dark)[:,0], states.prequery[:,0]):
            raise AssertionError("current readout changed")
    return donor_state, raw_dark, feature_delta


def public_query(batch: HistoryBatch) -> torch.Tensor:
    """Keep the same public value while querying the public role U."""
    q = batch.query.clone()
    q[:, 5:9] = 0.
    q[:, 8] = 1.
    return q


@torch.no_grad()
def competence(teacher: Teacher, seed: int, family: str,
               n: int=512) -> dict:
    batch = make_batch(seed, n, family)
    states = teacher.rollout(batch)
    err = states.final_output - batch.target
    blind = batch.source_blind_conditional_mean - batch.target
    public_mask = batch.query_role==3
    active = batch.event_mask
    return {
        "source_family": family, "n": n,
        "mae":float(err.abs().mean()),
        "mse":float(err.square().mean()),
        "blind_mae":float(blind.abs().mean()),
        "blind_mse":float(blind.square().mean()),
        "public_mae":float(err[public_mask].abs().mean()),
        "public_n":int(public_mask.sum()),
        "prequery_abs_readout":float(states.prequery[:, 0].abs().mean()),
        "mean_valid_write_abs_readout":float(
            ((states.intermediate_readout.abs()*active).sum()/active.sum()).item()
        ),
    }
