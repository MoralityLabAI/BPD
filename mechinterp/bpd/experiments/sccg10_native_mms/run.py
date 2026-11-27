"""SCCG-10 MMS-TRM command-line stages, each inside the fail-closed 6 GiB job cap.

Stages: warmup (fresh root, 6000 updates), drift (checkpoint drift instrument),
arms (paired trainer arms), replay (independent checkpoint/outcome replay).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

from ..trm_curation.memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap
from .common import Recipe, dump_json, recipe_from, train_warmup


def _recipe(args) -> Recipe:
    if args.recipe:
        return recipe_from(json.loads(Path(args.recipe).read_text(encoding='utf-8')))
    return Recipe()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('warmup', 'drift', 'arms', 'replay'))
    p.add_argument('--root', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--recipe', type=Path)
    p.add_argument('--warm', type=Path, help='warmup checkpoint for drift/arms')
    p.add_argument('--config', type=Path, help='JSON stage config (arms, caps, thresholds)')
    p.add_argument('--no-cap', action='store_true', help='unit tests only')
    args = p.parse_args(argv)
    recipe = _recipe(args)
    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    cap = None if args.no_cap else ProcessTreeMemoryCap(DEFAULT_CAP_BYTES)
    if cap is not None:
        cap.__enter__()
    try:
        if args.stage == 'warmup':
            model, receipt = train_warmup(args.root, recipe)
            torch.save(model.state_dict(), args.out / 'warmup.pt')
            receipt['recipe'] = recipe.as_dict()
            dump_json(args.out / 'warmup_receipt.json', receipt)
        elif args.stage == 'drift':
            from .drift import run_drift
            cfg = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
            run_drift(args.root, args.warm, args.out, recipe, **cfg)
        elif args.stage == 'arms':
            from .trainer import run_arms
            cfg = json.loads(args.config.read_text(encoding='utf-8'))
            run_arms(args.root, args.warm, args.out, recipe, cfg)
        elif args.stage == 'replay':
            from .replay import replay_root
            replay_root(args.out, recipe)
    finally:
        if cap is not None:
            acc = cap.accounting()
            dump_json(args.out / f'memory_accounting_{args.stage}.json', asdict(acc))
            print(json.dumps({'memory_cap': acc.enforcement_mechanism, 'enforced': acc.enforced,
                              'peak_job_mib': round((acc.peak_job_commit_bytes or 0) / 2**20, 1),
                              'cpu_s': round(acc.process_cpu_seconds, 2)}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
