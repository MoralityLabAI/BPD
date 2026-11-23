"""Pinned BF16 safetensors affine replay and fixed next-pass K tangent."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from archive_inputs import bound, bound_json, require


INDEX = "model.safetensors.index.json"
CONFIG = "config.json"
GAMMA = "model.norm.weight"
ETA = "model.layers.0.input_layernorm.weight"
QUERY_K = "model.layers.0.self_attn.k_proj.weight"
DOWN = "model.layers.21.mlp.down_proj.weight"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def decode_bf16(data):
    return (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def round_bf16(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)


class Shard:
    """Access only declared BF16 tensors through a verified read-only handle."""

    def __init__(self, stream, expected, check):
        stat = os.fstat(stream.fileno())
        digest = sha256()
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
            check()
        require(digest.hexdigest() == expected, "checkpoint shard hash mismatch")
        self.stream = stream
        self.stat = stat
        stream.seek(0)
        header_len = int.from_bytes(stream.read(8), "little")
        require(0 < header_len < 10 * 1024 * 1024, "safetensors header exceeds the replay bound")
        self.header = json.loads(stream.read(header_len))
        self.offset = 8 + header_len
        self.verify_unchanged()

    def verify_unchanged(self):
        stat = os.fstat(self.stream.fileno())
        require((stat.st_size, stat.st_mtime_ns) == (self.stat.st_size, self.stat.st_mtime_ns),
                "checkpoint shard changed during replay")

    def row_shape(self, name, shape):
        entry = self.header[name]
        start, end = entry["data_offsets"]
        require(
            entry["dtype"] == "BF16" and tuple(entry["shape"]) == shape
            and type(start) is int and type(end) is int
            and 0 <= start < end <= self.stat.st_size - self.offset
            and end - start == 2 * int(np.prod(shape)),
            "tensor dtype, shape or byte range disagrees with the pinned layout",
        )
        return start, end

    def tensor(self, name, shape, row_range=None):
        start, end = self.row_shape(name, shape)
        if row_range is None:
            out_shape = shape
        else:
            lo, hi = row_range
            require(len(shape) == 2 and 0 <= lo < hi <= shape[0], "invalid safetensors row range")
            stride = 2 * shape[1]
            end = start + hi * stride
            start += lo * stride
            out_shape = (hi - lo, shape[1])
        self.stream.seek(self.offset + start)
        payload = self.stream.read(end - start)
        require(len(payload) == end - start, "incomplete safetensors tensor payload")
        values = decode_bf16(payload).reshape(out_shape)
        require(np.isfinite(values).all(), "nonfinite values in pinned tensor")
        return values


def double_rms_jvp(carrier, writes, gamma, eta, eps):
    """Continuous derivative of RMSNorm_eta(RMSNorm_gamma(carrier))."""
    x = np.asarray(carrier, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)
    h = np.asarray(eta, dtype=np.float64)
    metric = np.square(g) + eps
    scale = 1 / np.sqrt(np.mean(np.square(x) * metric) + eps ** 2)
    coupling = (x * metric) @ writes / x.size
    return (g * h)[:, None] * (scale * writes - scale ** 3 * x[:, None] * coupling)


def replay_affine(shard, occurrences, inp, out, check):
    rows = np.concatenate([occurrence.x.reshape(-1, inp) for occurrence in occurrences.values()]).astype(np.float64)
    archived = np.concatenate([occurrence.native.reshape(-1, out) for occurrence in occurrences.values()])
    replay = np.empty(archived.shape, dtype=np.float32)
    for lo in range(0, out, 192):
        check()
        hi = min(lo + 192, out)
        w = shard.tensor(DOWN, (out, inp), (lo, hi)).astype(np.float64)
        replay[:, lo:hi] = round_bf16(rows @ w.T)
    residual = replay.astype(np.float64) - archived
    equal = replay.view(np.uint32) == archived.view(np.uint32)
    denom = float(np.linalg.norm(archived.astype(np.float64)))
    per_pass = {}
    start = 0
    for phase, occurrence in occurrences.items():
        count = occurrence.x.shape[0] * occurrence.x.shape[1]
        stop = start + count
        mask = equal[start:stop]
        delta = residual[start:stop]
        per_pass[str(phase)] = {
            "case_writes": count,
            "bit_exact_case_writes": int(mask.all(axis=1).sum()),
            "bit_exact_elements": int(mask.sum()),
            "elements": mask.size,
            "max_absolute_error": float(np.max(np.abs(delta))),
            "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(archived[start:stop].astype(np.float64))),
        }
        start = stop
    input_bits = rows.astype(np.float32).view(np.uint32)
    output_bits = archived.view(np.uint32)
    return {
        "method": "float64 affine contraction followed by float32 and BF16 round-to-nearest-even",
        "case_writes": len(archived),
        "bit_exact_case_writes": int(equal.all(axis=1).sum()),
        "bit_exact_elements": int(equal.sum()),
        "elements": equal.size,
        "max_absolute_error": float(np.max(np.abs(residual))),
        "relative_l2": float(np.linalg.norm(residual) / denom) if denom else None,
        "all_archived_input_values_bf16_representable": bool(np.all((input_bits & 0xFFFF) == 0)),
        "all_archived_output_values_bf16_representable": bool(np.all((output_bits & 0xFFFF) == 0)),
        "passes": per_pass,
    }


class FixedK:
    def __init__(self, gamma, eta, eps, k):
        self.gamma, self.eta, self.eps, self.k = gamma, eta, eps, k

    def __call__(self, carrier, writes):
        return self.k @ double_rms_jvp(carrier, writes, self.gamma, self.eta, self.eps)


def checkpoint_replay(snapshot: Path, revision, pinned, tokenizer_config_sha256, occurrences, check):
    require(snapshot.name == revision == pinned["revision"], "checkpoint revision mismatch")
    locks = pinned["locked_files_sha256"]
    bound(snapshot / "tokenizer_config.json", tokenizer_config_sha256)
    config = bound_json(snapshot / CONFIG, locks[CONFIG])
    index = bound_json(snapshot / INDEX, locks[INDEX])["weight_map"]
    inp, out = config["intermediate_size"], config["hidden_size"]
    head_size = config.get("head_dim", out // config["num_attention_heads"])
    k_rows = head_size * config["num_key_value_heads"]
    require(
        config["num_hidden_layers"] == 22 and inp == 10752 and out == 3072 and k_rows == 1024
        and set(index[name] for name in (GAMMA, ETA, QUERY_K, DOWN)) == set(SHARDS)
        and DOWN.removesuffix(".weight") + ".bias" not in index,
        "checkpoint layout does not match the registered affine sites",
    )
    with (snapshot / SHARDS[0]).open("rb") as f0, (snapshot / SHARDS[1]).open("rb") as f1:
        shards = {name: Shard(stream, locks[name], check) for name, stream in zip(SHARDS, (f0, f1), strict=True)}
        gamma = shards[index[GAMMA]].tensor(GAMMA, (out,)).astype(np.float64)
        eta = shards[index[ETA]].tensor(ETA, (out,)).astype(np.float64)
        k = shards[index[QUERY_K]].tensor(QUERY_K, (k_rows, out)).astype(np.float64)
        affine = replay_affine(shards[index[DOWN]], occurrences, inp, out, check)
        for shard in shards.values():
            shard.verify_unchanged()
    return affine, FixedK(gamma, eta, config["rms_norm_eps"], k), {
        "checkpoint_revision": revision,
        "checkpoint_file_sha256": {name: locks[name] for name in (CONFIG, INDEX, *SHARDS)},
        "tokenizer_config_sha256": tokenizer_config_sha256,
    }
