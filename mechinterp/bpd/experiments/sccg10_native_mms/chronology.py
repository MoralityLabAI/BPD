"""Signed chronological Jacobian transport along the native suffix (MATH_REVIEW §3).

For one recurrence step z -> step(z, e_t) on the joint state [z_H, z_L] (one low
visit, one high visit, shared factors), measure the exact per-step Jacobians D_t
along the clean suffix trajectory at two checkpoints, their spectral norms M_t,
the per-step drift eps_t = ||D'_t - D_t||, the actual end-to-end product drift
||prod D'_t - prod D_t||, and the chronological bound
prod(M_t + eps_t) - prod(M_t). Everything is an exact measurement at the listed
checkpoints on a TRAIN panel; it is not a uniform statement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import Recipe, dump_json, make_trm
from .drift import panel_batches, panel_edges, stream_batch
from . import native


def _block(model, z):
    # component-mode native block with all components on and no delta path:
    # up(z) = z V U (parity with rollout_suffix states is asserted in run()).
    up = (z.float() @ model.up.V) @ model.up.U
    return model.norm(z + model.down(F.silu(up)))


def step_fn(model, e_t):
    def f(state):
        zh, zl = state[: model.width], state[model.width:]
        zl = _block(model, zl + zh + e_t)
        zh = _block(model, zh + zl)
        return torch.cat((zh, zl))
    return f


def suffix_jacobians(model, panel, delay):
    x = panel.batch.receiver[:1]
    seq = torch.cat((x[:, :4 + delay], x[:, -1:]), 1)[:, 4:]
    s = native.post_write_state(model, panel)
    e = model.input(seq)[0].detach()
    Ds, states = [], [s]
    for t in range(e.shape[0]):
        f = step_fn(model, e[t])
        D = torch.autograd.functional.jacobian(f, s).detach().double().numpy()
        Ds.append(D)
        with torch.no_grad():
            s = f(s)
        states.append(s)
    with torch.no_grad():
        ref = model.rollout_suffix(seq, native.post_write_state(model, panel)[None], prefix_steps=4)['states'][0]
    got = torch.stack(states[1:])
    if not torch.allclose(got, ref, rtol=1e-4, atol=1e-5):
        raise AssertionError('manual step does not match native rollout_suffix')
    return Ds


def run(root: int, warm_path, out: Path, recipe: Recipe, *, n_edges=6, delay=31, checkpoints=(0, 1, 8)):
    model = make_trm(recipe)
    model.load_state_dict(torch.load(warm_path, weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=recipe.curation_learning_rate, weight_decay=recipe.weight_decay)
    edges = panel_edges(root, recipe)[:n_edges]
    panels = [native.make_panel(b) for b in panel_batches(root, recipe, edges)]
    Jref, rows, step = None, [], 0
    per_ck = {}
    for ck in checkpoints:
        while step < ck:
            model.train()
            xb, yb = stream_batch(root, step, recipe)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip); opt.step()
            step += 1
        model.eval()
        per_ck[ck] = [suffix_jacobians(model, p, delay) for p in panels]
    base = per_ck[checkpoints[0]]
    for ck in checkpoints[1:]:
        for i, (D0s, D1s) in enumerate(zip(base, per_ck[ck])):
            M = [np.linalg.norm(D, 2) for D in D0s]
            eps = [np.linalg.norm(D1 - D0, 2) for D0, D1 in zip(D0s, D1s)]
            P0, P1 = np.eye(D0s[0].shape[0]), np.eye(D0s[0].shape[0])
            prefix = []
            for D0, D1 in zip(D0s, D1s):
                P0, P1 = D0 @ P0, D1 @ P1
                prefix.append((np.linalg.norm(P0, 2), np.linalg.norm(P1 - P0, 2)))
            bound = float(np.prod(np.add(M, eps)) - np.prod(M))
            for t in (2, 7, 15, delay):
                if t <= len(prefix):
                    pass
            rows.append({'checkpoint': ck, 'edge': i, 'steps': len(M), 'M_max': float(max(M)),
                         'M_geomean': float(np.exp(np.mean(np.log(M)))), 'eps_max': float(max(eps)),
                         'eps_mean': float(np.mean(eps)),
                         'product_norm_by_t': {str(t): prefix[t - 1][0] for t in (1, 3, 8, 16, len(prefix))},
                         'product_drift_by_t': {str(t): prefix[t - 1][1] for t in (1, 3, 8, 16, len(prefix))},
                         'chronological_bound_full': bound,
                         'actual_full_drift': float(prefix[-1][1])})
    res = {'root': root, 'delay': delay, 'checkpoints': list(checkpoints), 'rows': rows,
           'scope': 'exact per-step Jacobians on TRAIN panel edges at listed checkpoints'}
    dump_json(Path(out) / 'chronology.json', res)
    return res


if __name__ == '__main__':
    import argparse
    from ..trm_curation.memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=int, required=True)
    ap.add_argument('--warm', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    ProcessTreeMemoryCap(DEFAULT_CAP_BYTES).__enter__()
    r = run(a.root, a.warm, a.out, Recipe())
    for row in r['rows']:
        print(json.dumps({k: row[k] for k in ('checkpoint', 'edge', 'M_max', 'M_geomean', 'eps_max', 'chronological_bound_full', 'actual_full_drift')}))
