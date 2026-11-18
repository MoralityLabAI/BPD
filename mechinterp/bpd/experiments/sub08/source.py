"""Registered CPU source training for BPD-SUB-08; no causal interventions."""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .oracle import partition, VOCAB, SEQ_LEN

DESIGN_ROOTS = (1890, 1891, 1892)
PROSPECTIVE_ROOTS = tuple(range(1900, 1908))
SMOKE_ROOT = 1889
CANDIDATES = ((48, 2400), (48, 3600), (64, 2400), (64, 3600))
CHECKPOINTS = (2400, 3600)
BATCH_SIZE = 128
LR = .003


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_runtime():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Registered source requires Python3.12")
    if torch.__version__ != "2.5.1+cpu" or np.__version__ != "2.3.5":
        raise RuntimeError("Registered source requires torch2.5.1+cpu and numpy2.3.5")
    return {
        "python": platform.python_version(), "python_minor": "3.12",
        "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "device": "cpu", "dtype": "float32",
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def tensor_split(name, role):
    allowed = {"train", "validation"}
    if role == "prospective":
        allowed.add("heldout")
    if role not in {"design", "prospective", "smoke"} or name not in allowed:
        raise ValueError("Split access is outside the registered role")
    seq, labels, ids = partition(name)
    x = F.one_hot(torch.as_tensor(seq, dtype=torch.long, device="cpu"), VOCAB).float()
    return x, torch.as_tensor(labels, dtype=torch.long, device="cpu"), ids


class TiedGRU(nn.Module):
    def __init__(self, width):
        super().__init__()
        if width not in (48, 64):
            raise ValueError("Width outside the registered candidate family")
        self.width = width
        self.cell = nn.GRUCell(VOCAB, width, device="cpu", dtype=torch.float32)
        self.head = nn.Linear(width, 4, device="cpu", dtype=torch.float32)

    def rollout(self, x):
        if x.ndim != 3 or x.shape[1:] != (SEQ_LEN, VOCAB):
            raise ValueError("Expected [batch,6,4] one-hot inputs")
        h = x.new_zeros(len(x), self.width)
        states = []
        for t in range(SEQ_LEN):
            h = self.cell(x[:, t], h)
            states.append(h)
        return self.head(h), states


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def prefix_consistency(ids, logits, prefix_length):
    # Sequence IDs are six base-four digits in lexicographic enumeration.
    groups = {}
    maximum = 0.0
    for sequence_id, row in zip(ids, logits):
        prefix = int(sequence_id) // (4 ** (6 - prefix_length))
        if prefix in groups:
            maximum = max(maximum, float(np.max(np.abs(row - groups[prefix]))))
        else:
            groups[prefix] = row
    return maximum


def summarize(split):
    logits = np.asarray(split["logits"], dtype=np.float64)
    labels = np.asarray(split["labels"], dtype=np.int64)
    predictions = logits.argmax(axis=1)
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    centered = logits - logits.max(axis=1, keepdims=True)
    nll = np.log(np.exp(centered).sum(axis=1)) - centered[np.arange(len(labels)), labels]
    counts = confusion.sum(axis=1)
    result = {
        "n": len(labels), "correct": int(np.trace(confusion)),
        "accuracy": float((predictions == labels).mean()), "nll": float(nll.mean()),
        "confusion": confusion.tolist(), "class_counts": counts.tolist(),
        "class_recall": (np.diag(confusion) / counts).tolist(),
    }
    tokens = np.asarray(split["ids"], dtype=np.int64) % 4
    result["last_token_accuracy"] = [float((predictions[tokens == t] == labels[tokens == t]).mean()) for t in range(4)]
    for length in (2, 4):
        key = f"prefix{length}_logits"
        if key in split:
            result[f"prefix{length}_accuracy"] = float((np.argmax(split[key], axis=1) == labels).mean())
    return result


@torch.no_grad()
def evaluate(model, name, role):
    x, y, ids = tensor_split(name, role)
    logits, states = model.rollout(x)
    record = {"ids": [int(i) for i in ids], "labels": y.tolist(), "logits": logits.tolist(),
              "predictions": logits.argmax(dim=1).tolist(),
              "ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()}
    if name == "heldout":
        for length in (2, 4):
            prefix_logits = model.head(states[length - 1])
            record[f"prefix{length}_logits"] = prefix_logits.tolist()
            record[f"prefix{length}_predictions"] = prefix_logits.argmax(dim=1).tolist()
    return record


@torch.no_grad()
def implementation_checks(model, role):
    name = "heldout" if role == "prospective" else "validation"
    x, _, ids = tensor_split(name, role)
    logits, states = model.rollout(x)
    h = torch.zeros((len(x), model.width), dtype=torch.float32, device="cpu")
    # Independently iterate the public cell/head path at full physical width.
    for token in x.unbind(dim=1):
        h = model.cell(token, h)
    direct = model.head(h)
    return {
        "full_width_parity_max": float((logits - direct).abs().max()),
        "prefix2_consistency_max": prefix_consistency(ids, model.head(states[1]).numpy(), 2),
        "prefix4_consistency_max": prefix_consistency(ids, model.head(states[3]).numpy(), 4),
        "weights_finite": all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()),
    }


def train_trajectory(seed, width, updates, role, on_checkpoint):
    if role == "design":
        if seed not in DESIGN_ROOTS or updates != 3600:
            raise ValueError("Design trajectory outside frozen roster/budget")
        checkpoints = CHECKPOINTS
    elif role == "prospective":
        if seed not in PROSPECTIVE_ROOTS or (width, updates) not in CANDIDATES:
            raise ValueError("Prospective source outside frozen family")
        checkpoints = (updates,)
    elif role == "smoke":
        if seed != SMOKE_ROOT or updates > 16 or updates < 1:
            raise ValueError("Smoke exceeds excluded budget")
        checkpoints = (updates,)
    else:
        raise ValueError("Unknown source-training role")
    runtime = setup_runtime()
    torch.manual_seed(seed)
    model = TiedGRU(width)
    initial_hash = state_digest(model)
    x, y, _ = tensor_split("train", role)
    generator = torch.Generator(device="cpu").manual_seed(seed + 44)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, betas=(.9, .999), eps=1e-8,
        weight_decay=0, amsgrad=False, foreach=False, fused=False,
    )
    trace = []
    completed = 0
    try:
        for step in range(1, updates + 1):
            idx = torch.randint(len(x), (BATCH_SIZE,), generator=generator, device="cpu")
            logits, _ = model.rollout(x[idx])
            loss = F.cross_entropy(logits, y[idx])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite or absent training gradient")
            optimizer.step()
            completed = step
            if step % 200 == 0 or step in checkpoints:
                trace.append({"updates": step, "loss": float(loss.detach())})
            if step in checkpoints:
                model.eval()
                on_checkpoint(model, step, {
                    "initial_state_sha256": initial_hash, "trace": list(trace),
                    "runtime": runtime, "completed_updates": completed,
                    "final_training_loss": float(loss.detach()),
                })
                model.train()
    except Exception as error:
        error.completed_updates = completed
        error.initial_state_sha256 = initial_hash
        error.trace = trace
        raise

