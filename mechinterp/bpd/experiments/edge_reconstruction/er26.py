"""ER-26: sorted refinement cost bound and online incumbent-contraction pilot."""
import argparse
import json
import time
import numpy as np
from er25 import world, screen
from er23 import Bank, cost, EPS

WIDTHS = (32, 64, 128)
ROOTS = (48, 49, 50, 51)
BATCHES = (16, 64, 256)
STATIC = {32: 64, 64: 64, 128: 16}  # frozen ER-25 calibration choices
MODES = ('eager', 'sort16', 'sort64', 'sort256', 'online')


def finish(lo, eligible, incumbent, score, evaluate, mode):
    """Sorted exact search; adaptive policy starts small, grows on a plateau."""
    order = eligible[np.argsort(lo[eligible], kind='stable')]
    sorted_lo = lo[order]
    initial = len(order)
    best, calls, pos = incumbent, 1, 0
    batch = 16 if mode == 'online' else int(mode[4:])
    trace = []
    while pos < len(order):
        stop = int(np.searchsorted(sorted_lo, score + EPS, side='right'))
        if stop <= pos:
            break
        remaining = stop - pos
        ids = order[pos:min(pos + batch, stop)]
        vals = evaluate(ids)
        assert np.all(lo[ids] <= vals + EPS)
        m = float(vals.min())
        k = int(ids[vals == m].min())
        old = score
        if m < score or (m == score and k < best):
            score, best = m, k
        calls += len(ids)
        pos += len(ids)
        new_stop = int(np.searchsorted(sorted_lo, score + EPS, side='right'))
        contraction = (stop - new_stop) / remaining
        trace.append([batch, len(ids), stop, new_stop, old, score])
        if mode == 'online':
            # A large contraction indicates that the optimum is still being approached.
            # A plateau favors amortizing exact-kernel dispatch.
            if contraction >= .25:
                batch = 16
            else:
                batch = min(256, batch * 2)
    return dict(best=best, score=score, calls=calls, initial=initial,
                trace=trace, max_batch=max((r[0] for r in trace), default=0))


def audit_synthetic():
    """Exhaustive integer fixtures including ties and an extraneous incumbent."""
    from itertools import product
    checks = 0
    for n in (1, 2, 3, 4):
        for exact in product(range(3), repeat=n):
            exact = np.array(exact, dtype=float)
            opt = int(exact.argmin())
            for lower in product(*(range(int(v)+1) for v in exact)):
                lo = np.array(lower, dtype=float)
                sstar = int((lo <= exact[opt]+EPS).sum())
                for incumbent in range(n):
                    eligible = np.flatnonzero((lo <= exact[incumbent]+EPS) & (np.arange(n) != incumbent))
                    for mode in ('sort1', 'sort2', 'sort3', 'online'):
                        r = finish(lo, eligible, incumbent, exact[incumbent], lambda ids: exact[ids], mode)
                        assert r['best'] == opt and r['score'] == exact[opt]
                        # Initial incumbent plus at most one batch of extraneous work.
                        assert sstar <= r['calls'] <= sstar + r['max_batch'] + (r['max_batch']==0)
                        checks += 1
    return checks


def run_one(args,y,mode):
    t = time.perf_counter()
    bank = Bank(*args)
    if mode == 'eager':
        # Materialize only scalar costs for audit, not full predictions.
        costs = np.concatenate([cost(bank.exact(np.arange(i,min(i+64,bank.d**2))),y)
                                for i in range(0,bank.d**2,64)])
        k = int(costs.argmin())
        return dict(best=k, score=float(costs[k]), calls=len(costs), ms=1000*(time.perf_counter()-t)), costs
    lo,eligible,k,score = screen(bank,y)
    s = time.perf_counter()
    r = finish(lo,eligible,k,score,lambda ids:cost(bank.exact(ids),y),mode)
    end = time.perf_counter()
    r.update(ms=1000*(end-t), screen_ms=1000*(s-t), finish_ms=1000*(end-s))
    return r,lo


def run(width,roots):
    for root in roots:
        for episode,(args,y,hidden,noise) in enumerate(world(width,root)):
            timing = {m:[] for m in MODES}
            outputs = {}
            aux = {}
            for rep in range(2):
                shift = (root+episode+rep)%len(MODES)
                for mode in MODES[shift:]+MODES[:shift]:
                    r,a = run_one(args,y,mode)
                    timing[mode].append(r['ms'])
                    if rep == 0:
                        outputs[mode],aux[mode] = r,a
                    else:
                        assert outputs[mode]['best']==r['best'] and abs(outputs[mode]['score']-r['score'])<1e-10
            ref = outputs['eager']
            costs = aux['eager']
            for mode,r in outputs.items():
                r['ms'] = float(np.median(timing[mode]))
                assert r['best']==ref['best'] and abs(r['score']-ref['score'])<1e-10
                if mode!='eager':
                    lo=aux[mode]
                    assert np.all(lo<=costs+EPS)
                    sstar=int((lo<=ref['score']+EPS).sum())
                    assert sstar<=r['calls']<=sstar+max(1,r['max_batch'])
                    r['sstar']=sstar
                    r['excess']=r['calls']-sstar
            static=f'sort{STATIC[width]}'
            print(json.dumps(dict(width=width,root=root,episode=episode,noise=noise,hidden=hidden,
                hit=ref['best']==hidden,static=static,arms=outputs)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--width',type=int,choices=WIDTHS)
    p.add_argument('--roots',type=int,nargs='+',default=ROOTS); p.add_argument('--audit',action='store_true')
    a=p.parse_args()
    if a.audit: print(json.dumps(dict(exhaustive_checks=audit_synthetic())))
    else: run(a.width,a.roots)
