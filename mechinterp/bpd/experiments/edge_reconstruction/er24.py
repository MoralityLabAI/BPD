"""ER-24: calibration-only RSS/time lookup and held-out policy replay.
RSS predictions are empirical guards, not hard memory guarantees.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WIDTHS = (64, 128, 256)
CAL = (38, 39)
TEST = (40, 41)
CONFIGS = (('eager', 32, 8), ('eager', 64, 8), ('dense', 64, 8),
           ('stream', 32, 4), ('stream', 64, 8), ('stream', 128, 16))
BUDGETS = (96, 112, 128, 160, 192, 224)
GUARD = 4096


def configs(w):
    return CONFIGS if w != 256 else (CONFIGS[1], CONFIGS[2], CONFIGS[4])


def key(r):
    return r['mode'], r['batch'], r['block']


def prepare(path):
    from er23 import prepare as make
    for w in WIDTHS:
        make(path, [w], CAL + TEST, 1 if w == 256 else 2)


def collect(path, split):
    for w in WIDTHS:
        for root in CAL if split == 'cal' else TEST:
            ks = configs(w)
            for mode, batch, block in (ks[::-1] if root % 2 else ks):
                args = [sys.executable, str(HERE / 'er23.py'), 'bench',
                        str(path / f'{w}_{root}.npz'), '--mode', mode,
                        '--batch', str(batch), '--block', str(block), '--repeats', '1']
                r = json.loads(subprocess.check_output(args, env=dict(os.environ,
                               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')))
                r.update(root=root, split=split)
                print(json.dumps(r), flush=True)


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def fit(rows):
    table = {}
    for w in WIDTHS:
        table[w] = {}
        for k in configs(w):
            rr = [r for r in rows if r['width'] == w and key(r) == k]
            assert {r['root'] for r in rr} == set(CAL)
            table[w][k] = dict(peak_kib=max(r['peak_kib'] for r in rr) + GUARD,
                               ms=float(np.median([e['ms'] for r in rr for e in r['rows']])))
    return table


def select(table, w, budget):
    eligible = [(v['ms'], k) for k, v in table[w].items() if v['peak_kib'] <= budget]
    return min(eligible)[1] if eligible else None


def evaluate(cal, test):
    table = fit(cal)
    trials = []
    for w in WIDTHS:
        for root in TEST:
            rr = {key(r): r for r in test if r['width'] == w and r['root'] == root}
            assert set(rr) == set(configs(w))
            ref = rr[('eager', 64, 8)]
            for r in rr.values():
                assert len(r['rows']) == len(ref['rows'])
                for e, a in zip(r['rows'], ref['rows']):
                    assert e['best'] == a['best'] and abs(e['score'] - a['score']) < 1e-10
            for budget in BUDGETS:
                lim = budget * 1024
                chosen = select(table, w, lim)
                actual = rr[chosen] if chosen else None
                feasible = [(k, r) for k, r in rr.items() if r['peak_kib'] <= lim]
                for i, a in enumerate(ref['rows']):
                    oracle = min((r['rows'][i]['ms'], k) for k, r in feasible) if feasible else None
                    trials.append(dict(width=w, root=root, episode=i, budget_mib=budget,
                        chosen=chosen, actual_peak_kib=actual['peak_kib'] if actual else None,
                        ms=actual['rows'][i]['ms'] if actual else None,
                        fit=actual['peak_kib'] <= lim if actual else None,
                        oracle_config=oracle[1] if oracle else None, oracle_ms=oracle[0] if oracle else None,
                        oracle_feasible=bool(feasible), eager64_ms=a['ms'],
                        eager64_fits=ref['peak_kib'] <= lim, hit=a['best'] == a['hidden']))
    accepted = [r for r in trials if r['chosen']]
    ok = [r for r in accepted if r['fit']]
    return dict(widths=WIDTHS, cal_roots=CAL, test_roots=TEST, guard_kib=GUARD,
                calibration={str(w): [dict(config=k, **v) for k, v in t.items()] for w, t in table.items()},
                trials=trials, summary=dict(trials=len(trials), accepted=len(accepted),
                    violations=len(accepted)-len(ok), abstentions=len(trials)-len(accepted),
                    false_abstentions=sum(r['chosen'] is None and r['oracle_feasible'] for r in trials),
                    median_regret=float(np.median([r['ms']/r['oracle_ms'] for r in ok])),
                    max_regret=float(max(r['ms']/r['oracle_ms'] for r in ok)),
                    median_eager_speedup=float(np.median([r['eager64_ms']/r['ms'] for r in ok if r['eager64_fits']]))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'cal', 'test', 'evaluate'])
    p.add_argument('path', type=Path)
    p.add_argument('--test', type=Path)
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.path)
    elif a.action in ('cal', 'test'):
        collect(a.path, a.action)
    else:
        print(json.dumps(evaluate(load(a.path), load(a.test)), indent=2))
