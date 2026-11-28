"""Export successful TRAIN-only mechanistic curation decisions in DataGoblin format.

Strict split custody:
- Only TRAIN fit-view semantic edges are admitted (verified against semantic_edges('train')
  and disjoint from semantic_edges('eval')).
- Zero evaluation labels, zero development readout metrics, zero held-out rows enter
  the exported training JSONL files.
- Emits candidate_decisions.jsonl, curation_preferences.jsonl, and manifest.json
  with SHA-256 provenance, predicted & realized utility, mechanistic features, and
  charged CPU resource costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .reference import semantic_edges


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    ).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')


def export_datagoblin(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Export verified TRAIN-only curation decisions and preference pairs to ``out_dir``."""
    train_set = set(semantic_edges('train'))
    eval_set = set(semantic_edges('eval'))

    root_dirs = sorted(p for p in source_dir.glob('root_*') if p.is_dir())
    if not root_dirs:
        raise ValueError(f'No root_* directories found under {source_dir}')

    out_dir.mkdir(parents=True, exist_ok=True)
    decisions: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}

    for rdir in root_dirs:
        dec_path = rdir / 'train_curation_decisions.jsonl'
        if not dec_path.is_file():
            continue
        src_sha = _sha256_bytes(dec_path)
        source_hashes[rdir.name] = src_sha
        by_window: dict[tuple[int, int], list[dict]] = {}

        for line in dec_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('split') != 'train' or row.get('training_eligible') is not True:
                raise ValueError(f'Split custody violation in {dec_path}')
            e_raw = row['edge']
            edge_tuple = (tuple(e_raw[0]), tuple(e_raw[1]), int(e_raw[2]))
            if edge_tuple not in train_set or edge_tuple in eval_set:
                raise ValueError(f'Edge outside TRAIN semantic partition: {edge_tuple}')
            for num_k in ('train_loss', 'predicted_gain', 'realized_gain', 'utility', 'update_norm', 'charged_cpu_seconds'):
                v = row[num_k]
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError(f'Non-finite numeric field {num_k}')

            rec_id = _digest_obj({
                'root': row['root'], 'window': row['window'],
                'checkpoint_sha256': row['checkpoint_sha256'], 'edge': row['edge'],
            })
            dg_row = {
                'schema_version': 'datagoblin.trm_curation_decision.v1',
                'record_id': rec_id,
                'split': 'train',
                'training_eligible': True,
                'evaluation_leakage': False,
                'provenance': {
                    'experiment': 'BPD-TRM-CURATE-02',
                    'root': row['root'],
                    'arm': row['arm'],
                    'window': row['window'],
                    'epoch': row['epoch'],
                    'checkpoint_sha256': row['checkpoint_sha256'],
                    'source_file_sha256': src_sha,
                },
                'input': {
                    'edge_receiver': e_raw[0],
                    'edge_hybrid': e_raw[1],
                    'edit_slot': int(e_raw[2]),
                    'delay': int(row['delay']),
                },
                'mechanistic_features': {
                    'train_loss': float(row['train_loss']),
                    'predicted_gain': float(row['predicted_gain']),
                    'update_norm': float(row['update_norm']),
                    'predicted_utility': float(row['utility']),
                },
                'outcome': {
                    'selected': bool(row['selected']),
                    'selection_reason': row['selection_reason'],
                    'window_accepted': bool(row['window_accepted']),
                    'realized_gain': float(row['realized_gain']),
                    'realized_utility': float(row['realized_gain']) / max(float(row['charged_cpu_seconds']), 1e-9),
                    'sign_match': bool((row['predicted_gain'] > 0) == (row['realized_gain'] > 0)),
                    'window_cal_before': row['window_cal_before'],
                    'window_cal_after': row['window_cal_after'],
                },
                'resource_cost': {
                    'charged_cpu_seconds': float(row['charged_cpu_seconds']),
                },
            }
            decisions.append(dg_row)
            by_window.setdefault((row['root'], row['window']), []).append(dg_row)

        # Build verified preference pairs within each window where a chosen candidate
        # strictly beat a rejected pool candidate on realized calibration gain
        for (root_id, win_id), w_rows in sorted(by_window.items()):
            pos = [r for r in w_rows if r['outcome']['selected'] and r['outcome']['realized_gain'] > 0]
            neg = [r for r in w_rows if not r['outcome']['selected'] and r['outcome']['realized_gain'] <= 0]
            if pos and neg:
                best_pos = max(pos, key=lambda r: r['outcome']['realized_gain'])
                worst_neg = min(neg, key=lambda r: r['outcome']['realized_gain'])
                preferences.append({
                    'schema_version': 'datagoblin.trm_curation_preference.v1',
                    'split': 'train',
                    'training_eligible': True,
                    'evaluation_leakage': False,
                    'root': root_id,
                    'window': win_id,
                    'checkpoint_sha256': best_pos['provenance']['checkpoint_sha256'],
                    'chosen': {
                        'record_id': best_pos['record_id'],
                        'input': best_pos['input'],
                        'mechanistic_features': best_pos['mechanistic_features'],
                        'realized_gain': best_pos['outcome']['realized_gain'],
                    },
                    'rejected': {
                        'record_id': worst_neg['record_id'],
                        'input': worst_neg['input'],
                        'mechanistic_features': worst_neg['mechanistic_features'],
                        'realized_gain': worst_neg['outcome']['realized_gain'],
                    },
                    'realized_margin': best_pos['outcome']['realized_gain'] - worst_neg['outcome']['realized_gain'],
                })

    dec_out = out_dir / 'candidate_decisions.jsonl'
    pref_out = out_dir / 'curation_preferences.jsonl'
    _write_jsonl(dec_out, decisions)
    _write_jsonl(pref_out, preferences)

    successful = [r for r in decisions if r['outcome']['selected'] and r['outcome']['realized_gain'] > 0]
    manifest = {
        'schema_version': 'datagoblin.trm_curation_export_manifest.v1',
        'experiment': 'BPD-TRM-CURATE-02',
        'train_only': True,
        'evaluation_rows_exported': 0,
        'roots_exported': sorted(source_hashes.keys()),
        'total_candidate_decisions': len(decisions),
        'successful_selected_decisions': len(successful),
        'verified_preference_pairs': len(preferences),
        'source_sha256': source_hashes,
        'files_sha256': {
            'candidate_decisions.jsonl': _sha256_bytes(dec_out),
            'curation_preferences.jsonl': _sha256_bytes(pref_out),
        },
        'claim_boundary': (
            'Strictly TRAIN fit-view semantic edges only; zero evaluation or development '
            'held-out outcomes included. Utility and realized gain measured on TRAIN feedback view.'
        ),
    }
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    m = export_datagoblin(args.source, args.out)
    print(json.dumps({
        'decisions': m['total_candidate_decisions'],
        'successful_selected': m['successful_selected_decisions'],
        'preferences': m['verified_preference_pairs'],
        'train_only': m['train_only'],
    }, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
