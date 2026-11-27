"""Independent checkpoint and outcome replay for SCCG-10 engineering receipts.

1. Every saved economics checkpoint: recompute its state hash and the ordinary
   steering-disabled delay accuracies from the saved weights, compare to the receipt.
2. Deterministic re-execution: retrain fixed-update A_ordinary and C_mech_fresh
   from the saved warmup and compare final state hashes.
3. Drift checkpoints: recompute state hashes from the saved weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechinterp.bpd.experiments.trm_curation.campaign import eval_model_ordinary
from mechinterp.bpd.experiments.trm_curation.memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap

from .common import ENGINEERING_ROOTS, HERE, accounting_dict, dump_json, load_checkpoint, model_state_sha256, recipe
from .economics import Setup, run_arm

ENG = HERE / 'results' / 'engineering'


def replay_root(root: int, *, rerun: bool, prefix: str = 'economics') -> dict:
    rc = recipe()
    torch.set_num_threads(rc.threads)
    torch.use_deterministic_algorithms(True)
    edir = ENG / f'{prefix}_{root}'
    rep = json.loads((edir / 'economics.json').read_text(encoding='utf-8'))
    out = dict(root=root, checkpoints={}, reruns={})
    for tag, res in rep['results'].items():
        m = load_checkpoint(edir / f'{tag}.pt')
        sha_ok = model_state_sha256(m) == res['final_state_sha256']
        ev = eval_model_ordinary(m, root, rc, split='eval', domain_prefix='sccg10_eng_eval')
        counts_ok = all(ev['delays'][d]['correct_count'] == res['eval']['delays'][d]['correct_count']
                        for d in res['eval']['delays'])
        bind_ok = all(ev['binding'][d]['correct_count'] == res['eval']['binding'][d]['correct_count']
                      for d in res['eval']['binding'])
        out['checkpoints'][tag] = dict(state_sha_match=sha_ok, delay_counts_match=counts_ok, binding_counts_match=bind_ok)
    if rerun:
        setup = Setup(root, ENG / f'warmup_{root}' / 'warmup.pt')
        for arm in ('A_ordinary', 'C_mech_fresh'):
            res, _ = run_arm(arm, setup, budget='updates', limit=rep['windows'],
                             preview_mode=rep.get('preview_mode', 'transactional'))
            want = rep['results'][f'updates_{arm}']['final_state_sha256']
            out['reruns'][arm] = dict(rerun_sha=res['final_state_sha256'], receipt_sha=want,
                                      match=res['final_state_sha256'] == want)
    return out


def replay_drift(name: str) -> dict:
    d = ENG / name
    r = json.loads((d / 'drift.json').read_text(encoding='utf-8'))
    out = {}
    for t, ck in r['checkpoints'].items():
        p = d / f'ckpt_u{int(t):03d}.pt'
        if p.is_file():
            out[t] = model_state_sha256(load_checkpoint(p)) == ck['state_sha256']
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-rerun', action='store_true')
    a = ap.parse_args()
    cap = ProcessTreeMemoryCap(DEFAULT_CAP_BYTES)
    with cap:
        rep = dict(economics={f'{p}_{r}': replay_root(r, rerun=not a.no_rerun, prefix=p)
                              for p in ('economics', 'economics_batched') for r in ENGINEERING_ROOTS
                              if (ENG / f'{p}_{r}' / 'economics.json').is_file()},
                   drift={n.name: replay_drift(n.name) for n in sorted(ENG.glob('drift_*')) if (n / 'drift.json').is_file()})
        ok = all(all(v.values()) for r in rep['economics'].values() for v in r['checkpoints'].values())
        ok &= all(v['match'] for r in rep['economics'].values() for v in r['reruns'].values())
        ok &= all(all(v.values()) for v in rep['drift'].values())
        rep['all_match'] = bool(ok)
        rep['memory'] = accounting_dict(cap)
        dump_json(ENG / 'replay.json', rep)
        print(json.dumps(dict(all_match=rep['all_match'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
