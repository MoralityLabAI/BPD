"""Explicit byte protocol for component activations.

The header is a little-endian uint32 row count, uint16 component count,
and uint8 bit depth. Values use MSB-first fixed-width unsigned codes at
2/4/8 bits, or little-endian IEEE754 floats at 32 bits. Codebook scales
are shared setup and never adapted to readout data.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

HEADER = struct.Struct("<IHB")


@dataclass(frozen=True)
class Message:
    payload: bytes
    clipped_values: int
    values: int

    @property
    def nbits(self) -> int:
        return 8 * len(self.payload)


def pack_codes(codes: np.ndarray, bits: int) -> bytes:
    if bits not in (2, 4, 8):
        raise ValueError("fixed code depth must be 2, 4 or 8")
    flat = np.asarray(codes).reshape(-1)
    if np.any(flat < 0) or np.any(flat >= (1 << bits)) or np.any(flat != np.floor(flat)):
        raise ValueError("code outside range")
    result = bytearray(math.ceil(flat.size * bits / 8))
    for i, value in enumerate(flat):
        shift = 8 - bits - (i * bits % 8)
        result[i * bits // 8] |= int(value) << shift
    return bytes(result)


def unpack_codes(payload: bytes, count: int, bits: int) -> np.ndarray:
    if bits not in (2, 4, 8) or count < 0:
        raise ValueError("invalid code depth or count")
    if len(payload) != math.ceil(count * bits / 8):
        raise ValueError("payload size does not match code count")
    unused = len(payload) * 8 - count * bits
    if unused and payload and payload[-1] & ((1 << unused) - 1):
        raise ValueError("nonzero padding bits")
    mask = (1 << bits) - 1
    return np.array([(payload[i * bits // 8] >> (8 - bits - i * bits % 8)) & mask
                     for i in range(count)], dtype=np.float32)


def encode(activations: np.ndarray, scales: np.ndarray, bits: int) -> Message:
    z = np.asarray(activations, dtype=np.float32)
    if z.ndim != 2 or not np.all(np.isfinite(z)):
        raise ValueError("activations must be a finite matrix")
    n, rank = z.shape
    if rank < 1 or rank > 65535 or n < 1 or n > 65535:
        raise ValueError("protocol shape exceeds bounds")
    if bits not in (2, 4, 8, 32):
        raise ValueError("unsupported bit depth")
    header = HEADER.pack(n, rank, bits)
    if bits == 32:
        return Message(header + z.astype("<f4").tobytes(), 0, z.size)
    scales = np.asarray(scales, dtype=np.float32)
    if scales.shape != (rank,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("invalid shared quantizer scales")
    clipped = int(np.sum(np.abs(z) > scales))
    levels = (1 << bits) - 1
    codes = np.rint((np.clip(z / scales, -1, 1) + 1) * (levels / 2)).astype(np.int64)
    return Message(header + pack_codes(codes, bits), clipped, z.size)


def decode(payload: bytes, scales: np.ndarray, *, expected_rank: int) -> np.ndarray:
    if len(payload) < HEADER.size:
        raise ValueError("truncated header")
    n, rank, bits = HEADER.unpack_from(payload)
    if n < 1 or n > 65535 or rank != expected_rank or rank < 1:
        raise ValueError("invalid message shape")
    data = payload[HEADER.size:]
    if bits == 32:
        if len(data) != 4 * n * rank:
            raise ValueError("invalid float payload length")
        out = np.frombuffer(data, dtype="<f4").reshape(n, rank).copy()
        if not np.all(np.isfinite(out)):
            raise ValueError("invalid float payload values")
        return out
    scales = np.asarray(scales, dtype=np.float32)
    if scales.shape != (rank,) or np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("invalid shared quantizer scales")
    codes = unpack_codes(data, n * rank, bits).reshape(n, rank)
    return (codes / ((1 << bits) - 1) * 2 - 1) * scales


def action_reply(action: int, n_candidates: int) -> bytes:
    if not 0 <= action < n_candidates <= 65535:
        raise ValueError("action outside announced candidate set")
    return struct.pack("<H", action)
