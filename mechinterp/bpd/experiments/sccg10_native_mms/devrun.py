"""Frozen development runner: verifies the source lock, then per root runs the
fresh warmup, the fixed-CPU arms and the fixed-update arms, all inside the
6 GiB job cap, writing artifact hashes per run directory.

    python -m mechinterp.bpd.experiments.sccg10_native_mms.devrun --freeze DEV_FREEZE.json --roots 29301 29302
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch

from ..trm_curation.memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap
from .common import REPO, dump_json, recipe_from, train_warmup
from .replay import write_artifact_hashes


CRLF, LF = bytes((13, 10)), bytes((10,))


def lf_sha256(path) -> str:
    """SHA-256 of file bytes with CRLF normalized to LF (checkouts may use autocrlf)."""
    return hashlib.sha256(Path(path).read_bytes().replace(CRLF, LF)).hexdigest()


def source_hashes(files) -> dict[str, str]:
    return {f: lf_sha256(REPO / f) for f in files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--roots', type=int, nargs='+', required=True)
    a = ap.parse_args(argv)
    fz = json.loads(a.freeze.read_text(encoding='utf-8'))
    now = source_hashes(fz['source_hashes'].keys())
    bad = [f for f, h in fz['source_hashes'].items() if now[f] != h]
    if bad:
        raise SystemExit(f'source lock violated: {bad}')
    if any(r not in fz['development_roots'] for r in a.roots):
        raise SystemExit('root not in frozen development roster')
    recipe = recipe_from(fz['recipe'])
    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    out = REPO / fz['output_dir']
    cap = ProcessTreeMemoryCap(DEFAULT_CAP_BYTES)
    cap.__enter__()
    from .trainer import run_arms
    try:
        for root in a.roots:
            rd = out / f'root_{root}'
            rd.mkdir(parents=True, exist_ok=True)
            if not (rd / 'warmup.pt').is_file():
                model, rec = train_warmup(root, recipe)
                torch.save(model.state_dict(), rd / 'warmup.pt')
                rec['recipe'] = recipe.as_dict()
                rec['source_lock_sha256'] = fz['source_lock_sha256']
                dump_json(rd / 'warmup_receipt.json', rec)
            for name in ('fixed_cpu', 'fixed_update'):
                sub = rd / name
                if (sub / 'arms_result.json').is_file():
                    continue
                sub.mkdir(parents=True, exist_ok=True)
                cfg = dict(fz['arm_configs'][name])
                cfg['eval'] = 'final_development'
                run_arms(root, rd / 'warmup.pt', sub, recipe, cfg)
                write_artifact_hashes(sub)
    finally:
        acc = cap.accounting()
        tag = '_'.join(map(str, a.roots))
        dump_json(out / f'memory_accounting_{tag}.json', asdict(acc))
        print(json.dumps({'roots': a.roots, 'memory_cap': acc.enforcement_mechanism, 'enforced': acc.enforced,
                          'peak_job_mib': round((acc.peak_job_commit_bytes or 0) / 2**20, 1),
                          'cpu_s': round(acc.process_cpu_seconds, 1)}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
