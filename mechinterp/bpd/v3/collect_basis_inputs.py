"""BASIS-01 local-only clean TRM module-vector exporter.

One board per rollout; hooks on four physical shared matrices record a seeded
nonadaptive sample of their true prelinear inputs at every recurrent call.
No gradients, gates, dither or checkpoint downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mechinterp.bpd.v3.collect_native_dither import (  # noqa: E402
    admit_cohort, file_sha256, load_local_trm,
)

DEFAULT_SITES = (
    "layers.0.o_proj", "layers.0.down_proj",
    "layers.1.o_proj", "layers.1.down_proj",
)
NEAR_ZERO_THRESHOLDS = (0.0, 0.001, 0.01, 0.1)


def position_seed(seed: int, board: int, tau: int) -> int:
    digest = hashlib.sha256(f"basis01:{seed}:{board}:{tau}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def fixed_positions(*, seed: int, board: int, n_passes: int,
                    prefix_len: int, total_len: int, positions_per_call: int,
                    device: torch.device) -> tuple[np.ndarray, list[torch.Tensor]]:
    available = total_len - prefix_len
    if available < positions_per_call or positions_per_call < 1 or n_passes < 1:
        raise ValueError("insufficient board-token positions or invalid sample budget")
    pos = np.empty((n_passes, positions_per_call), dtype=np.int64)
    for tau in range(n_passes):
        rng = np.random.default_rng(position_seed(seed, board, tau))
        pos[tau] = prefix_len + rng.choice(available, size=positions_per_call,
                                          replace=False)
    return pos, [torch.tensor(row, dtype=torch.long, device=device) for row in pos]


@torch.no_grad()
def capture_module_inputs(
    model: Any, encoded_inputs: np.ndarray, *, seed: int = 20261117,
    n_passes: int = 63, positions_per_call: int = 2,
    sites: tuple[str, ...] = DEFAULT_SITES,
) -> dict[str, dict[str, np.ndarray]]:
    """Return W, inputs and repeated-measurement metadata for each module site.

    Native admission (checksum and full roster) belongs to the CLI. This pure
    collector accepts a small explicitly provided fake TRM for CPU testing.
    """
    x = np.asarray(encoded_inputs)
    if x.ndim != 2 or len(x) < 1 or not np.issubdtype(x.dtype, np.integer):
        raise ValueError("encoded_inputs must be a nonempty integer [board,token] array")
    if x.min() < 1 or x.max() > 10:
        raise ValueError("Sudoku token IDs must lie in [1,10]")
    if len({row.tobytes() for row in np.ascontiguousarray(x)}) != len(x):
        raise ValueError("duplicate puzzle roots")
    if len(set(sites)) != len(sites) or not sites:
        raise ValueError("site names must be nonempty and unique")

    dev = model.embed_tokens.weight.device
    modules = dict(model.named_modules())
    for name in sites:
        if name not in modules or not isinstance(modules[name], torch.nn.Linear):
            raise ValueError(f"missing linear module: {name}")

    prefix_len = int(model.puzzle_emb_len)
    total_len = int(model.total_len)
    if x.shape[1] != total_len - prefix_len:
        raise ValueError("input length does not match model's board-token count")
    cyc = int(model.l_cycles) + 1
    if cyc < 2 or n_passes % cyc != 0:
        raise ValueError("calls must contain complete recurrence periods")
    meta = {"board_idx": [], "pass_idx": [], "call_type": [], "token_pos": []}
    all_inputs = {name: [] for name in sites}

    for board, tokens in enumerate(x):
        pos, selectors = fixed_positions(
            seed=seed, board=board, n_passes=n_passes,
            prefix_len=prefix_len, total_len=total_len,
            positions_per_call=positions_per_call, device=dev,
        )
        buffers: dict[str, list[torch.Tensor]] = {name: [] for name in sites}
        counts = {name: 0 for name in sites}
        handles = []

        def register(name: str) -> None:
            def before(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
                tau = counts[name]
                if tau >= n_passes:
                    raise RuntimeError(f"extra forward invocation at {name}")
                vector = args[0]
                if vector.ndim != 3 or vector.shape[0] != 1 or vector.shape[1] != total_len:
                    raise RuntimeError(f"unexpected input layout at {name}")
                buffers[name].append(
                    vector[0].index_select(0, selectors[tau]).detach().float()
                )
                counts[name] += 1
            handles.append(modules[name].register_forward_pre_hook(before))

        try:
            for name in sites:
                register(name)
            inp = torch.from_numpy(tokens.astype(np.int64)).view(1, -1).to(dev)
            x_emb = model.encode_inputs(inp)
            z_h, z_l = model.init_states(1)
            for tau in range(n_passes):
                if (tau % cyc) < model.l_cycles:
                    z_l = model.run_l_level(z_l, z_h + x_emb)
                else:
                    z_h = model.run_l_level(z_h, z_l)
        finally:
            for handle in handles:
                handle.remove()

        for name in sites:
            if counts[name] != n_passes:
                raise RuntimeError(
                    f"hook counts do not close for {name}: {counts[name]} vs {n_passes}"
                )
            arr = torch.stack(buffers[name], dim=0).cpu().numpy()
            if arr.shape != (n_passes, positions_per_call, modules[name].in_features):
                raise RuntimeError(f"module input dimension mismatch for {name}")
            all_inputs[name].append(arr.reshape(-1, arr.shape[-1]))
        rows = n_passes * positions_per_call
        meta["board_idx"].append(np.full(rows, board, dtype=np.int32))
        meta["pass_idx"].append(np.repeat(np.arange(n_passes, dtype=np.int16),
                                           positions_per_call))
        meta["call_type"].append(np.repeat(
            np.arange(n_passes, dtype=np.int8) % cyc, positions_per_call))
        meta["token_pos"].append(pos.reshape(-1).astype(np.int16))

    metadata = {key: np.concatenate(value, axis=0) for key, value in meta.items()}
    result = {}
    for name in sites:
        weight = modules[name].weight.detach().float().cpu().numpy().copy()
        inputs = np.concatenate(all_inputs[name], axis=0)
        if weight.shape[1] != inputs.shape[1] or not (
            np.isfinite(inputs).all() and np.isfinite(weight).all()
        ):
            raise ValueError(f"module inputs/weights invalid: {name}")
        result[name] = {"W": weight, "inputs": inputs, **metadata}
    return result


def phase_sparsity(inputs: np.ndarray, call_type: np.ndarray,
                   thresholds: tuple[float, ...] = NEAR_ZERO_THRESHOLDS) -> dict:
    x = np.asarray(inputs, np.float64)
    phases = np.asarray(call_type)
    if x.ndim != 2 or phases.shape != (x.shape[0],):
        raise ValueError("phase sparsity inputs/labels do not match")
    scale = float(np.sqrt(np.mean(x ** 2)))
    scale = max(scale, 1e-12)
    out = {}
    for phase in sorted(int(y) for y in np.unique(phases)):
        part = x[phases == phase]
        out[str(phase)] = {
            "n_vectors": int(part.shape[0]),
            "fraction_abs_le_rms_scale": {
                str(t): float(np.mean(np.abs(part) <= t * scale))
                for t in thresholds
            },
        }
    return {"global_input_rms": scale, "per_call_type": out}


def write_capture_files(
    captures: dict[str, dict[str, np.ndarray]], out_dir: Path, *,
    checkpoint_key: str, checkpoint_sha256: str,
    roster_sha256: str, seed: int, n_boards: int,
    n_passes: int, positions_per_call: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sites = {}
    for name, data in captures.items():
        stem = name.replace(".", "_")
        path = out_dir / (stem + ".npz")
        np.savez_compressed(
            path, **data,
            checkpoint_sha256=np.array(checkpoint_sha256),
            roster_sha256=np.array(roster_sha256),
        )
        sites[name] = {
            "npz_path": path.as_posix(),
            "npz_sha256": file_sha256(path),
            "weight_shape": list(data["W"].shape),
            "input_shape": list(data["inputs"].shape),
            "sparsity": phase_sparsity(data["inputs"], data["call_type"]),
        }
    manifest = {
        "protocol": "BASIS-01-MODULES", "status": "DEVELOPMENT_ONLY",
        "n_boards": n_boards, "n_passes": n_passes,
        "positions_per_call": positions_per_call,
        "seed": seed, "checkpoint_key": checkpoint_key,
        "checkpoint_sha256": checkpoint_sha256, "roster_sha256": roster_sha256,
        "sampling": "independent deterministic board-token positions per board and call; shared across modules",
        "sites": sites,
        "evidence": "clean target inputs; no gates, no native basis fit or task intervention",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-file", type=Path, required=True)
    p.add_argument("--checkpoint-key", choices=("gaoxin492", "yagizdevre"), required=True)
    p.add_argument("--inputs-npz", type=Path, required=True)
    p.add_argument("--expected-roster-sha256", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    p.add_argument("--max-boards", type=int, help="development-only subset")
    p.add_argument("--positions-per-call", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261117)
    args = p.parse_args()

    if not args.inputs_npz.is_file():
        p.error("local roster is missing")
    roster_sha = file_sha256(args.inputs_npz)
    with np.load(args.inputs_npz, allow_pickle=False) as obj:
        if "input_ids" not in obj.files:
            p.error("input_ids missing from local NPZ")
        inputs = obj["input_ids"]
    if inputs.ndim != 2 or inputs.shape[1] != 81:
        p.error("native TRM requires input_ids[B,81]")
    if args.expected_roster_sha256.lower() != roster_sha.lower():
        p.error("input roster hash mismatch")
    if args.max_boards is not None:
        if args.max_boards < 1:
            p.error("max-boards must be positive")
        inputs = inputs[:args.max_boards]
    try:
        admit_cohort(len(inputs), max_boards=args.max_boards,
                     expected_roster_sha256=args.expected_roster_sha256,
                     observed_roster_sha256=roster_sha)
    except ValueError as err:
        p.error(str(err))

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, checkpoint_sha = load_local_trm(
        args.checkpoint_file, args.checkpoint_key, dtype=dtype, device=args.device
    )
    capture = capture_module_inputs(
        model, inputs, seed=args.seed, n_passes=63,
        positions_per_call=args.positions_per_call,
    )
    manifest = write_capture_files(
        capture, args.out_dir, checkpoint_key=args.checkpoint_key,
        checkpoint_sha256=checkpoint_sha, roster_sha256=roster_sha,
        seed=args.seed, n_boards=len(inputs), n_passes=63,
        positions_per_call=args.positions_per_call,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
