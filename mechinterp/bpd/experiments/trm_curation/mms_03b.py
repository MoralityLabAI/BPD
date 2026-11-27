"""CURATE-03B MMS profiling: detached cache tiers under fixed-update checkpoint parity.

For each tier (0, 1, 4, 16, 64, 256 MiB) runs the same fixed-window arm from the
same warmup on an engineering root, and records CPU, cache occupancy,
recomputations avoided, process-tree RSS and the final checkpoint. Parity:
every tier must end on a byte-identical checkpoint and identical selections.

``reuse_potential`` reads pilot window logs and measures how often a raw
per-edge gradient could be served from cache: same parent checkpoint as an
earlier window (after rejection or abstention) *and* the same candidate input
(edge, delay, nuisance variant). Keys would also bind objective, factor frame,
data recipe and optimizer provenance; a checkpoint change invalidates all.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .campaign import CampaignRecipe, dump_json
from .memory_cap import DEFAULT_CAP_BYTES, ProcessTreeMemoryCap
from .pilot import ArmRunner, HERE, _rss_tree, prepare_root, source_hashes

TIERS = (0, 1, 4, 16, 64, 256)


def sweep(root: int, recipe: CampaignRecipe, arms, windows: int, out_dir: Path) -> dict:
    wp = HERE / 'results' / 'engineering' / 'phase_a' / f'root_{root}' / 'TRM_w32_r29.pt'
    prep = prepare_root(root, recipe, out_dir / f'root_{root}', warm_path=wp)
    res = {}
    for arm in arms:
        rows = {}
        for tier in TIERS:
            rn = ArmRunner(arm, root, prep['warm'], recipe, prep['projectors'], prep['fb_edges'],
                           prep['fit_edges'], mms_mib=tier, budget_cpu=None, fixed_windows=windows,
                           record_rows=False)
            s = rn.run()
            rows[str(tier)] = {'cpu_seconds': s['cpu_seconds'], 'final_state_sha256': s['final_state_sha256'],
                               'selections': [w.get('selected') for w in rn.windows],
                               'committed': s['committed_updates'], 'cache': s['cache_stats'],
                               'feedback_cache_hits': sum(1 for w in rn.windows
                                                          if (w.get('feedback') or {}).get('cache_hit')),
                               'cpu_by_category': s['cpu_by_category'],
                               'peak_tree_rss_bytes': max(s['peak_tree_rss_bytes'], _rss_tree())}
            print(json.dumps({'arm': arm, 'tier': tier, 'cpu': round(s['cpu_seconds'], 2),
                              'hits': s['cache_stats']['hits']}), flush=True)
        ref = rows['0']
        res[arm] = {
            'windows': windows, 'tiers': {k: {kk: vv for kk, vv in v.items() if kk != 'selections'}
                                          for k, v in rows.items()},
            'checkpoint_parity': all(v['final_state_sha256'] == ref['final_state_sha256'] for v in rows.values()),
            'selection_parity': all(v['selections'] == ref['selections'] for v in rows.values()),
            'cpu_speedup_vs_0': {k: ref['cpu_seconds'] / v['cpu_seconds'] for k, v in rows.items()},
            'peak_cache_bytes': {k: v['cache']['peak_total_bytes'] for k, v in rows.items()},
        }
    return res


def reuse_potential(dev_dir: Path, roots, arms) -> dict:
    out = {}
    for arm in arms:
        same_ckpt = total = repeat_inputs = candidates_at_repeat = 0
        for r in roots:
            p = dev_dir / f'root_{r}' / f'{arm}_windows.jsonl'
            if not p.is_file():
                continue
            seen: dict[str, set] = {}
            for line in p.read_text(encoding='utf-8').splitlines():
                w = json.loads(line)
                ck = w.get('checkpoint_sha256')
                if ck is None:
                    continue
                total += 1
                keys = {(json.dumps(e), w['delay'], w['window'] % 4) for e in w['acquisition_order']}
                if ck in seen:
                    same_ckpt += 1
                    candidates_at_repeat += len(keys)
                    repeat_inputs += len(keys & seen[ck])
                    seen[ck] |= keys
                else:
                    seen = {ck: keys}
        out[arm] = {'windows': total, 'windows_reusing_parent_checkpoint': same_ckpt,
                    'candidate_inputs_at_reused_checkpoint': candidates_at_repeat,
                    'cacheable_per_edge_gradients': repeat_inputs,
                    'cacheable_fraction_of_all_candidates': repeat_inputs / max(16 * total, 1)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--root', type=int, default=28000)
    ap.add_argument('--windows', type=int, default=40)
    ap.add_argument('--arms', nargs='+', default=['D_analytic_mech', 'G_joint_full16'])
    ap.add_argument('--dev', type=Path)
    ap.add_argument('--dev-roots', type=int, nargs='*', default=[])
    ap.add_argument('--no-cap', action='store_true')
    a = ap.parse_args()
    recipe = CampaignRecipe(threads=2)
    cap = None if a.no_cap else ProcessTreeMemoryCap(DEFAULT_CAP_BYTES)
    if cap is not None:
        cap.__enter__()
    try:
        a.out.mkdir(parents=True, exist_ok=True)
        res = {'experiment': 'BPD-TRM-CURATE-03B-MMS', 'evidence_class': 'ENGINEERING_SPENT_ROOTS',
               'root': a.root, 'tiers_mib': list(TIERS), 'sweep': sweep(a.root, recipe, a.arms, a.windows, a.out),
               'source_sha256': source_hashes()}
        if a.dev:
            res['per_edge_gradient_reuse_potential'] = reuse_potential(
                a.dev, a.dev_roots, ['B_transactional', 'C_analytic_ord', 'D_analytic_mech', 'E_full16_gate',
                                     'F_early12_gate', 'G_joint_full16', 'H_joint_early12', 'I_shuffled'])
        dump_json(a.out / 'mms_summary.json', res)
    finally:
        if cap is not None:
            dump_json(a.out / 'memory_accounting_mms.json', asdict(cap.accounting()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
