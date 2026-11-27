"""Independent saved-checkpoint and outcome replay for SCCG-10 MMS-TRM.

Rebuilds the ComponentTRM recurrence from raw state_dict tensors (no import of
trm_curation.model, the trainer or native.py), regenerates every evaluation
batch from its seed domain, and checks: artifact SHA-256 manifest, warmup and
final checkpoint hashes, parameter count 5380, per-case correctness bits,
correct counts, and cross-entropy (|err| <= 1e-5).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..trm_curation.data import edge_batch
from ..trm_curation.scheduler import edge_views
from ..training_steering.data import make_pairs
from ..training_steering_stability.data import binding_batch

TOL = 1e-5
DOMAIN = 'SCCG10-MMS-TRM'


def _seed(root, domain, i=0):
    return int.from_bytes(hashlib.sha256(f'{DOMAIN}:{root}:{domain}:{i}'.encode()).digest()[:8], 'little') % (2**63 - 1)


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _state_sha(state) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode('ascii'))
        h.update(str(tuple(value.shape)).encode('ascii'))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def manual_logits(state, x: torch.Tensor) -> torch.Tensor:
    V, U = state['up.V'], state['up.U']
    w = V.shape[0]
    eps = torch.finfo(x.dtype).eps

    def block(z):
        h = F.linear(F.silu((z.float() @ V) @ U), state['down.weight'])
        y = z + h
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps) * state['norm.weight']
    zh = x.new_zeros((x.shape[0], w))
    zl = torch.zeros_like(zh)
    e = F.linear(x, state['input.weight'], state['input.bias'])
    for j in range(x.shape[1]):
        zl = block(zl + zh + e[:, j])
        zh = block(zh + zl)
    return F.linear(zh, state['output.weight'], state['output.bias'])


def _check_eval_tune(state, root, recipe, rec) -> tuple[int, float, int]:
    tune = edge_views(recipe['split_seed'])['tune']
    checks, err, cases = 0, 0.0, 0
    for d, r in rec.items():
        b = edge_batch(tune, root=root, seed=_seed(root, f'tune_d{d}'), delay=int(d), split='train',
                       noise=recipe['noise'])
        x = torch.cat((b.receiver, b.hybrid)); y = torch.cat((b.labels_r, b.labels_h))
        lg = manual_logits(state, x)
        correct = int((lg.argmax(-1) == y).sum())
        if correct != r['correct']:
            raise ValueError(f'tune correct mismatch root={root} d={d}: {correct} vs {r["correct"]}')
        e = abs(float(F.cross_entropy(lg, y)) - r['cross_entropy'])
        if not math.isfinite(e) or e > TOL:
            raise ValueError(f'tune CE error {e}')
        checks += 2; err = max(err, e); cases += len(y)
    return checks, err, cases


def _check_final(state, root, recipe, rec) -> tuple[int, float, int]:
    checks, err, cases = 0, 0.0, 0
    for d, r in rec['delays'].items():
        b = make_pairs(_seed(root, f'final_dev_d{d}'), r['n'], root_seed=root, delay=int(d), split='eval',
                       slots=recipe['slots'], values=recipe['values'], noise=recipe['noise'])
        if b.fingerprint() != r['batch_sha256']:
            raise ValueError('final batch fingerprint mismatch')
        lg = manual_logits(state, b.receiver)
        bits = (lg.argmax(-1) == b.labels_r).int().tolist()
        if bits != r['correct'] or sum(bits) != r['correct_count']:
            raise ValueError(f'final correctness bits mismatch root={root} d={d}')
        e = abs(float(F.cross_entropy(lg, b.labels_r)) - r['cross_entropy'])
        if e > TOL:
            raise ValueError(f'final CE error {e}')
        checks += 3; err = max(err, e); cases += r['n']

    class _R:
        slots, values, noise = recipe['slots'], recipe['values'], recipe['noise']
    for d, r in rec['binding'].items():
        x, y, meta = binding_batch(root, f'sccg10_final_bind_d{d}', r['n'], int(d), 'eval', _R)
        if meta['batch_sha256'] != r['batch_sha256']:
            raise ValueError('binding fingerprint mismatch')
        lg = manual_logits(state, x)
        if int((lg.argmax(-1) == y).sum()) != r['correct_count']:
            raise ValueError('binding count mismatch')
        checks += 2; cases += r['n']
    return checks, err, cases


def replay_arms_dir(d: Path, root: int, recipe: dict, warm_path: Path) -> dict[str, Any]:
    res = json.loads((d / 'arms_result.json').read_text(encoding='utf-8'))
    warm = torch.load(warm_path, weights_only=True)
    checks, err, cases = 0, 0.0, 0
    if _state_sha(warm) != res['warm_sha256']:
        raise ValueError('warm checkpoint hash mismatch')
    if sum(v.numel() for k, v in warm.items() if k in ('input.weight', 'input.bias', 'up.V', 'up.U',
                                                        'down.weight', 'norm.weight', 'output.weight',
                                                        'output.bias')) != 5380:
        raise ValueError('parameter count')
    checks += 2
    for name, a in res['arms'].items():
        st = torch.load(d / a['checkpoint_file'], weights_only=True)
        if _state_sha(st) != a['final_state_sha256']:
            raise ValueError(f'final hash mismatch {name}')
        checks += 1
        if 'eval_tune' in a:
            c, e, n = _check_eval_tune(st, root, recipe, a['eval_tune'])
        else:
            c, e, n = _check_final(st, root, recipe, a['eval'])
        checks += c; err = max(err, e); cases += n
    hashes = d / 'artifact_hashes.json'
    if hashes.is_file():
        for f, h in json.loads(hashes.read_text(encoding='utf-8')).items():
            if _sha_file(d / f) != h:
                raise ValueError(f'artifact hash mismatch {f}')
            checks += 1
    return {'dir': str(d).replace('\\', '/'), 'verdict': 'PASS', 'checks': checks, 'cases': cases,
            'max_ce_error': err}


def write_artifact_hashes(d: Path) -> None:
    d = Path(d)
    out = {p.name: _sha_file(p) for p in sorted(d.iterdir()) if p.is_file() and p.name != 'artifact_hashes.json'}
    (d / 'artifact_hashes.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n', encoding='utf-8')


def replay_root(out: Path, recipe) -> dict:
    raise SystemExit('use: python -m ...sccg10_native_mms.replay <results_dir>')


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results', type=Path, help='engineering or development results directory')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    reports = []
    for rd in sorted(a.results.glob('root_*')):
        root = int(rd.name.split('_')[1])
        warm = rd / 'warmup.pt'
        rec = json.loads((rd / 'warmup_receipt.json').read_text(encoding='utf-8'))['recipe']
        for sub in sorted(p for p in rd.iterdir() if p.is_dir() and (p / 'arms_result.json').is_file()):
            r = replay_arms_dir(sub, root, rec, warm)
            reports.append(r)
            print(json.dumps(r), flush=True)
    summary = {'schema': 'SCCG10-replay/v1', 'verdict': 'PASS', 'dirs': reports,
               'checks': sum(r['checks'] for r in reports), 'cases': sum(r['cases'] for r in reports),
               'max_ce_error': max((r['max_ce_error'] for r in reports), default=0.0),
               'independence': 'manual recurrence from raw tensors; no import of model.py, native.py or trainer.py'}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
