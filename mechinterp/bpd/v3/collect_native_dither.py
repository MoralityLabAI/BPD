"""FLOOR-01 local-only native recurrent dither collector.

Requires an explicit SHA-pinned TRM checkpoint and a local, unsealed board roster.
The module contains no downloader and cannot read any protected ARG-03 test data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_SIGMA = 2**-8 / math.sqrt(12.0)
DEFAULT_READOUTS = (20, 41, 62)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1 << 20), b''):
            h.update(buf)
    return h.hexdigest()


def substream_seed(seed: int, board: int, draw: int) -> int:
    digest = hashlib.sha256(f'floor01:{seed}:{board}:{draw}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') % (2**63 - 1)


def dither_updated(state: torch.Tensor, generator: torch.Generator, sigma: float) -> torch.Tensor:
    """Bounded uniform additive dither in f32, cast back to native dtype."""
    if sigma < 0 or not math.isfinite(sigma):
        raise ValueError('sigma must be finite and nonnegative')
    if sigma == 0:
        return state
    z = state.float()
    rms = z.pow(2).mean(dim=-1, keepdim=True).sqrt()
    uniform = torch.rand(z.shape, dtype=torch.float32, device=z.device, generator=generator)
    return (z + (uniform * 2.0 - 1.0) * (math.sqrt(3) * sigma * rms)).to(state.dtype)


@torch.no_grad()
def rollout_single(model: Any, encoded_input: torch.Tensor, *, seed: int,
                   sigma: float, readouts: tuple[int, ...] = DEFAULT_READOUTS) -> np.ndarray:
    """Return [R,cell,9] for one root under one realized dither stream."""
    if encoded_input.ndim != 2 or encoded_input.shape[0] != 1:
        raise ValueError('one board per rollout is required for arithmetic comparability')
    period = int(model.l_cycles) + 1
    if len(readouts) == 0 or tuple(sorted(set(readouts))) != readouts:
        raise ValueError('readouts must be nonempty, strictly increasing')
    if any((t + 1) % period != 0 for t in readouts):
        raise ValueError('every readout must follow a z_H commit')
    if encoded_input.device != model.embed_tokens.weight.device:
        raise ValueError('model and input must share device')
    try:
        gen = torch.Generator(device=encoded_input.device)
        gen.manual_seed(seed)
    except RuntimeError as exc:
        raise ValueError('device does not support a deterministic torch generator') from exc
    x_emb = model.encode_inputs(encoded_input)
    z_h, z_l = model.init_states(1)
    output = []
    milestones = set(readouts)
    for tau in range(readouts[-1] + 1):
        if (tau % period) < model.l_cycles:
            z_l = dither_updated(model.run_l_level(z_l, z_h + x_emb), gen, sigma)
        else:
            z_h = dither_updated(model.run_l_level(z_h, z_l), gen, sigma)
            if tau in milestones:
                logits = model.digit_logits(z_h).float()
                if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != 9:
                    raise ValueError('digit logits must have shape [1,cells,9]')
                output.append(logits[0].cpu().numpy())
    if len(output) != len(readouts):
        raise AssertionError('missing readout')
    return np.stack(output, axis=0)


@torch.no_grad()
def collect(model: Any, encoded_inputs: np.ndarray, *, k: int = 8,
            sigma: float = DEFAULT_SIGMA, seed: int = 20261117,
            readouts: tuple[int, ...] = DEFAULT_READOUTS,
            compare_clean_repeat: bool = True) -> dict[str, np.ndarray | float]:
    """Native target dither ensemble. No clean-run gates or future-state feedback."""
    x = np.asarray(encoded_inputs)
    if x.ndim != 2 or len(x) == 0 or k < 2:
        raise ValueError('input shape [B,cells] and K>=2 required')
    if not np.issubdtype(x.dtype, np.integer) or x.min() < 1 or x.max() > 10:
        raise ValueError('input_ids must be pre-encoded Sudoku token IDs in [1,10]')
    if len({row.tobytes() for row in np.ascontiguousarray(x)}) != x.shape[0]:
        raise ValueError('duplicate puzzle roots in input roster')
    dev = model.embed_tokens.weight.device
    target = []
    static = []
    clean_diff = 0.0
    for board, ids in enumerate(x):
        inp = torch.from_numpy(ids.astype(np.int64)).view(1, -1).to(dev)
        clean = rollout_single(model, inp, seed=substream_seed(seed, board, -1),
                               sigma=0, readouts=readouts)
        if compare_clean_repeat:
            again = rollout_single(model, inp, seed=substream_seed(seed, board, -2),
                                   sigma=0, readouts=readouts)
            clean_diff = max(clean_diff, float(np.max(np.abs(clean - again))))
        static.append(clean)
        samples = [rollout_single(model, inp, seed=substream_seed(seed, board, j),
                                  sigma=sigma, readouts=readouts) for j in range(k)]
        target.append(np.stack(samples, axis=0))
    a = np.stack(target, axis=0)
    b = np.stack(static, axis=0)
    if a.shape[0:3] != (x.shape[0], k, len(readouts)) or a.shape[3:] != b.shape[2:]:
        raise AssertionError('collector output dimensional mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite model logits')
    return {'target_logits': a, 'static_logits': b, 'max_clean_repeat_logit_difference': clean_diff}


def load_local_trm(path: Path, key: str, *, dtype: torch.dtype, device: str):
    """Verify the exact named checkpoint before loading. No HF helper is invoked."""
    from mechinterp.loopscope02.scripts.trm_sudoku import PureSudokuTRM, TRM_CHECKPOINTS
    if key not in TRM_CHECKPOINTS:
        raise ValueError('unknown pinned checkpoint key')
    if not path.is_file():
        raise FileNotFoundError(f'local checkpoint absent: {path}')
    observed = file_sha256(path)
    expected = TRM_CHECKPOINTS[key]['sha256']
    if observed != expected:
        raise ValueError(f'checkpoint hash mismatch for {key}: {observed} != {expected}')
    weights = torch.load(path, map_location='cpu', weights_only=True)
    model = PureSudokuTRM.from_state_dict(weights, dtype=dtype).to(device).eval()
    return model, observed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint-file', type=Path, required=True)
    ap.add_argument('--checkpoint-key', choices=('gaoxin492', 'yagizdevre'), required=True)
    ap.add_argument('--inputs-npz', type=Path, required=True, help='Contains input_ids [B,81]')
    ap.add_argument('--out', type=Path, required=True, help='Output paired logits .npz')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--dtype', choices=('bf16', 'float32'), default='bf16')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--sigma', type=float, default=DEFAULT_SIGMA)
    ap.add_argument('--seed', type=int, default=20261117)
    ap.add_argument('--expected-roster-sha256')
    ap.add_argument('--max-boards', type=int, help='Development subset only')
    args = ap.parse_args()
    if args.k < 2 or args.sigma < 0 or not math.isfinite(args.sigma):
        ap.error('invalid K or sigma')
    if not args.inputs_npz.is_file():
        ap.error('input roster must exist locally')
    roster_sha = file_sha256(args.inputs_npz)
    if args.expected_roster_sha256 and roster_sha != args.expected_roster_sha256:
        ap.error('input roster hash mismatch')
    with np.load(args.inputs_npz, allow_pickle=False) as d:
        if 'input_ids' not in d:
            ap.error('input roster is missing input_ids')
        inputs = d['input_ids']
    if inputs.ndim != 2 or inputs.shape[1] != 81:
        ap.error('native TRM requires encoded input_ids [B,81]')
    if args.max_boards is not None:
        if args.max_boards < 1:
            ap.error('max-boards must be positive')
        inputs = inputs[:args.max_boards]
    if len(inputs) != 183 and args.max_boards is None:
        ap.error('native cohort must contain 183 WB-Stall roots, or supply --max-boards for development')
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    model, sha = load_local_trm(args.checkpoint_file, args.checkpoint_key,
                                dtype=dtype, device=args.device)
    result = collect(model, inputs, k=args.k, sigma=args.sigma, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, target_logits=result['target_logits'],
                        static_logits=result['static_logits'],
                        readout_passes=np.array(DEFAULT_READOUTS, dtype=np.int32))
    manifest = {
        'protocol': 'FLOOR-01_NATIVE_COLLECTION',
        'status': 'DEVELOPMENT_ONLY' if args.max_boards is not None else 'NATIVE_COHORT_UNCONFIRMED',
        'checkpoint_key': args.checkpoint_key, 'checkpoint_sha256': sha,
        'input_npz_sha256': roster_sha, 'output_npz_sha256': file_sha256(args.out),
        'n_boards': len(inputs), 'k': args.k, 'sigma_relative_rms': args.sigma,
        'seed': args.seed, 'dtype': args.dtype, 'device': args.device,
        'readout_passes': list(DEFAULT_READOUTS), 'batch_size': 1,
        'max_clean_repeat_logit_difference': result['max_clean_repeat_logit_difference'],
        'target_shape': list(result['target_logits'].shape),
        'static_shape': list(result['static_logits'].shape),
        'noise_law': 'per-call post-update uniform ±sqrt(3)*sigma*token_RMS in float32, cast to model dtype',
        'notes': 'Injected dither is an explicit stochastic intervention, not a measured GPU rounding law.',
    }
    outfile = args.out.with_suffix('.manifest.json')
    outfile.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
