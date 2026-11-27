"""Common 6000-update ordinary warmup for fresh SCCG-10 roots.

Same generator, architecture and optimizer recipe as CURATE-02 Phase A
(TRM_w32_r29 only). Runs under the fail-closed 6 GiB process-tree cap.
Development roots may be warmed, but their held EVAL partition is never
scored here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechinterp.bpd.experiments.trm_curation.campaign import eval_model_ordinary, train_warmup_model
from mechinterp.bpd.experiments.trm_curation.memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap
from mechinterp.bpd.experiments.trm_curation.scheduler import edge_views
from mechinterp.bpd.experiments.trm_curation.signal_bank import seed_for

from .common import (DEVELOPMENT_ROOTS, ENGINEERING_ROOTS, accounting_dict, dump_json, make_model,
                     model_state_sha256, recipe, source_hashes)


def strip(ev):
    return {
        'delays': {k: {fk: fv for fk, fv in v.items() if fk not in ('predictions', 'labels', 'correct')}
                   for k, v in ev['delays'].items()},
        'binding': ev['binding'],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--held-eval', action='store_true',
                   help='also score the historical EVAL partition (engineering roots only)')
    a = p.parse_args()
    if a.root not in ENGINEERING_ROOTS + DEVELOPMENT_ROOTS:
        raise SystemExit('root not allocated to SCCG-10')
    if a.held_eval and a.root not in ENGINEERING_ROOTS:
        raise SystemExit('held EVAL scoring of development roots is reserved for the frozen protocol')
    rc = recipe()
    cap = ProcessTreeMemoryCap(DEFAULT_CAP_BYTES)
    with cap:
        torch.set_num_threads(rc.threads)
        torch.use_deterministic_algorithms(True)
        fit = edge_views(rc.split_seed)['fit']
        torch.manual_seed(seed_for(a.root, 'init_TRM_w32_r29', 0))
        model = make_model()
        info = train_warmup_model(model, 'TRM_w32_r29', a.root, fit, rc, log_every=1000)
        ev_tr = strip(eval_model_ordinary(model, a.root, rc, split='train', domain_prefix='sccg10_warm_train'))
        gate = (ev_tr['delays']['2']['accuracy'] >= .8 and ev_tr['delays']['7']['accuracy'] >= .8
                and ev_tr['binding']['2']['accuracy'] >= .75 and ev_tr['binding']['7']['accuracy'] >= .75)
        rec = {**info, 'train_eval': ev_tr, 'competence_gate_passed': gate, 'source_hashes': source_hashes()}
        if a.held_eval:
            rec['held_eval'] = strip(eval_model_ordinary(model, a.root, rc, split='eval',
                                                         domain_prefix='sccg10_warm_eval'))
        a.out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), a.out / 'warmup.pt')
        rec['checkpoint_state_sha256'] = model_state_sha256(model)
        rec['memory'] = accounting_dict(cap)
        dump_json(a.out / 'warmup.json', rec)
        print(json.dumps({'root': a.root, 'cpu_s': round(info['cpu_seconds'], 1), 'gate': gate,
                          'tr_d31': ev_tr['delays']['31']['accuracy']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
