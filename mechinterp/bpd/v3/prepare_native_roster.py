"""Build an encoded WB-Stall input roster from the existing LOOPSCOPE-02 benchmark bundle.

This reads only the named public benchmark bundle and never accesses ARG-03-N.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_stall(data: dict, expected_n: int = 183) -> tuple[np.ndarray, list[str], list[str]]:
    items = data.get('items')
    if not isinstance(items, list):
        raise ValueError('bundle items must be a list')
    rows = [x for x in items if x.get('prop_solved') is False]
    if len(rows) != expected_n:
        raise ValueError(f'WB-Stall roster requires {expected_n} roots, got {len(rows)}')
    questions, item_ids = [], []
    for item in rows:
        q = item.get('question')
        if not isinstance(q, str) or len(q) != 81 or any(c not in '.123456789' for c in q):
            raise ValueError(f'invalid 81-char Sudoku root: {item.get("item_id")}')
        questions.append(q)
        item_ids.append(str(item['item_id']))
    if len(set(questions)) != len(questions) or len(set(item_ids)) != len(item_ids):
        raise ValueError('duplicated roots or IDs')
    tokens = np.array([[1 if c == '.' else int(c) + 1 for c in q] for q in questions], dtype=np.int64)
    return tokens, item_ids, [hashlib.sha256(q.encode()).hexdigest() for q in questions]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle-json', type=Path,
                    default=Path('mechinterp/loopscope02/results/benchmark_bundle.json'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.bundle_json.read_text(encoding='utf-8'))
    encoded, ids, hashes = encode_stall(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, input_ids=encoded, item_ids=np.array(ids),
                        root_sha256=np.array(hashes))
    meta = {'n_boards': int(encoded.shape[0]), 'label': 'WB-Stall',
            'input_ids_shape': list(encoded.shape),
            'bundle_json_sha256': sha(args.bundle_json), 'roster_npz_sha256': sha(args.out),
            'item_ids': ids, 'root_sha256': hashes}
    args.out.with_suffix('.manifest.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in meta.items() if k not in ('item_ids', 'root_sha256')}, indent=2))


if __name__ == '__main__':
    main()
