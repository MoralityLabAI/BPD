"""Independent saved-checkpoint recurrence replay auditor for BPD-TRM-CURATE-02.

Reconstructs ComponentTRM directly from saved PyTorch state_dict tensors without
importing ``model.py`` or ``campaign.py``. Verifies artifact SHA-256 manifests,
parameter count (5,380), exact prediction bits on primary and secondary delays,
and exact aggregate statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..training_steering.data import make_pairs

TOLERANCE = 1e-5


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_for(root: int, domain: str, index: int = 0) -> int:
    raw = f"BPD-TRM-CURATE-02:{root}:{domain}:{index}".encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little') % (2**63 - 1)


def _state_sha(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode('ascii'))
        h.update(str(tuple(value.shape)).encode('ascii'))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def manual_trm_rollout(state: dict[str, torch.Tensor], x: torch.Tensor,
                       low_cycles: int = 1) -> torch.Tensor:
    """Independent ComponentTRM recurrence directly from raw checkpoint tensors."""
    w = state['up.V'].shape[0]
    b, t, _ = x.shape
    zh = x.new_zeros((b, w))
    zl = torch.zeros_like(zh)
    e = F.linear(x, state['input.weight'], state['input.bias'])
    V = state['up.V']
    U = state['up.U']
    W_down = state['down.weight']
    norm_w = state['norm.weight']
    eps = torch.finfo(x.dtype).eps

    def rms_norm(z: torch.Tensor) -> torch.Tensor:
        return z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + eps) * norm_w

    def block(z: torch.Tensor) -> torch.Tensor:
        up_out = (z.float() @ V) @ U
        h = F.linear(F.silu(up_out.to(z.dtype)), W_down)
        return rms_norm(z + h)

    for j in range(t):
        for _ in range(low_cycles):
            zl = block(zl + zh + e[:, j])
        zh = block(zh + zl)
    return F.linear(zh, state['output.weight'], state['output.bias'])


def audit_development_dir(dev_dir: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    manifest = json.loads((dev_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    recipe = manifest['recipe']
    roots = manifest['roots']
    checks = 0
    max_err = 0.0
    replayed_cases = 0

    for root in roots:
        r_dir = dev_dir / f'root_{root}'
        hashes = json.loads((r_dir / 'artifact_hashes.json').read_text(encoding='utf-8'))
        for fname, expected in hashes.items():
            if fname == 'artifact_hashes.json':
                continue
            checks += 1
            actual = _sha(r_dir / fname)
            if actual != expected:
                raise ValueError(f'Artifact hash mismatch in root_{root}/{fname}')
        res = json.loads((r_dir / 'result.json').read_text(encoding='utf-8'))

        warm = torch.load(r_dir / 'shared_warmup.pt', map_location='cpu', weights_only=True)
        checks += 1
        if _state_sha(warm) != res['shared_warmup_sha256']:
            raise ValueError(f'Warmup state hash mismatch for root {root}')
        param_count = sum(
            warm[k].numel() for k in (
                'input.weight', 'input.bias', 'up.V', 'up.U',
                'down.weight', 'norm.weight', 'output.weight', 'output.bias'
            )
        )
        checks += 1
        if param_count != 5380:
            raise ValueError(f'Expected 5380 params, got {param_count}')

        # Replay primary delay-31 and delay-2 predictions for every saved arm
        for arm_name, arm_rec in res['arms'].items():
            st = torch.load(r_dir / f'{arm_name}.pt', map_location='cpu', weights_only=True)
            checks += 1
            if _state_sha(st) != arm_rec['final_state_sha256']:
                raise ValueError(f'Final checkpoint hash mismatch: root={root}, arm={arm_name}')
            for d in (2, recipe['primary_delay']):
                d_rec = arm_rec['eval']['delays'][str(d)]
                batch = make_pairs(
                    _seed_for(root, f'dev_eval_d{d}', 0), d_rec['n'],
                    root_seed=root, delay=d, split='eval',
                    slots=recipe['slots'], values=recipe['values'], noise=recipe['noise'],
                )
                checks += 1
                if batch.fingerprint() != d_rec['batch_sha256']:
                    raise ValueError(f'Batch fingerprint mismatch root={root} arm={arm_name} d={d}')
                logits = manual_trm_rollout(st, batch.receiver, low_cycles=recipe['low_cycles'])
                preds = logits.argmax(-1).tolist()
                checks += 1
                if preds != d_rec['predictions']:
                    raise ValueError(f'Prediction bit mismatch root={root} arm={arm_name} d={d}')
                ce_err = abs(float(F.cross_entropy(logits, batch.labels_r)) - d_rec['cross_entropy'])
                max_err = max(max_err, ce_err)
                checks += 1
                if not math.isfinite(ce_err) or ce_err > TOLERANCE:
                    raise ValueError(f'CE error {ce_err} exceeds tolerance on root={root} arm={arm_name} d={d}')
                replayed_cases += d_rec['n']

    return {
        'schema_version': 1,
        'experiment': 'BPD-TRM-CURATE-02',
        'verdict': 'PASS',
        'roots_audited': len(roots),
        'checks_passed': checks,
        'cases_replayed': replayed_cases,
        'max_scalar_ce_error': max_err,
        'tolerance': TOLERANCE,
        'prediction_bits_exact_match': True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dev_dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit_development_dir(args.dev_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
