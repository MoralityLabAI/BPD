"""Workstream F: DataGoblin TRAIN-only export of curriculum and mechanism decisions.

Reads per-arm ``*_decisions.jsonl`` (curriculum candidates) and
``*_mechanism.jsonl`` (atlas admission records) and writes:

* candidate_decisions.jsonl  one row per candidate per curated window, with
  predicted value, selection reason, realized window value, cost, verifier
  outcome and scope label (heuristic first-order proposal verified by the
  transactional acceptance test, or unaudited).
* mechanism_admissions.jsonl one row per feedback-edge admission decision,
  with ladder level, reason, estimated (source) and realized (verified) gain,
  physical dose and scope label (native finite verification vs none).
* resource_ledger.json       per arm CPU by category and work counts.

Fail-closed custody: every row must carry split=train and view in {fit} for
candidates; mechanism rows are TRAIN-feedback edges. No tune, final-development
or reserved rows are ever read. Mathematical certificates and heuristic
predictions are labelled separately via ``scope``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CERT_SCOPES = {'NATIVE_FINITE_VERIFIED_ON_PANEL_AT_CHECKPOINT', 'NATIVE_FINITE_VERIFIED_AT_SOURCE'}


def export(results: Path, out: Path, source_commit: str = 'unknown') -> dict:
    def repo_rel(p: Path) -> str:
        parts = Path(p).resolve().parts
        i = max(k for k, part in enumerate(parts) if part == 'mechinterp')
        return '/'.join(parts[i:])
    out.mkdir(parents=True, exist_ok=True)
    n_c = n_m = 0
    ledger = {}
    with (out / 'candidate_decisions.jsonl').open('w', encoding='utf-8', newline='\n') as fc, \
            (out / 'mechanism_admissions.jsonl').open('w', encoding='utf-8', newline='\n') as fm:
        for res in sorted(results.glob('root_*/*/arms_result.json')):
            run = res.parent
            data = json.loads(res.read_text(encoding='utf-8'))
            for arm, summ in data['arms'].items():
                ledger[f'{run.parent.name}/{run.name}/{arm}'] = {'ledger': summ['ledger'], 'cpu_used': summ['cpu_used'],
                                                                 'windows': summ['windows']}
                dp = run / f'{arm}_decisions.jsonl'
                if dp.is_file():
                    for line in dp.open(encoding='utf-8'):
                        r = json.loads(line)
                        if r.get('split') != 'train' or r.get('view') != 'fit':
                            raise ValueError(f'custody violation in {dp}')
                        r['source_run'] = f'{run.parent.name}/{run.name}'
                        r['intended_split'] = 'train'
                        r['provenance_refs'] = [f'https://github.com/MoralityLabAI/BPD/blob/{source_commit}/{repo_rel(dp)}']
                        r['certificate_kind'] = 'NONE_HEURISTIC_PROPOSAL'
                        r['verifier_outcome'] = ('ACCEPTED' if r['window_accepted'] else 'REJECTED_ROLLED_BACK') if r['window_audited'] else 'UNAUDITED'
                        fc.write(json.dumps(r, sort_keys=True) + '\n')
                        n_c += 1
                mp = run / f'{arm}_mechanism.jsonl'
                if mp.is_file():
                    for line in mp.open(encoding='utf-8'):
                        r = json.loads(line)
                        r.update({'root': data['root'], 'arm': arm, 'source_run': f'{run.parent.name}/{run.name}',
                                  'split': 'train', 'view': 'feedback', 'intended_split': 'train',
                                  'provenance_refs': [f'https://github.com/MoralityLabAI/BPD/blob/{source_commit}/{repo_rel(mp)}'],
                                  'certificate_kind': 'NATIVE_FINITE_VERIFICATION_ON_FIXED_PANEL' if r.get('scope') in CERT_SCOPES else 'NONE',
                                  'uniform_certificate': False, 'training_eligible': True})
                        fm.write(json.dumps(r, sort_keys=True, default=str) + '\n')
                        n_m += 1
    (out / 'resource_ledger.json').write_text(json.dumps(ledger, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    man = {'candidate_rows': n_c, 'mechanism_rows': n_m, 'splits_read': ['train/fit', 'train/feedback'],
           'excluded': ['train/tune', 'eval (final development)', 'ARG-03-N and reserved cohorts'],
           'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
                     if p.suffix in ('.jsonl', '.json') and p.name != 'manifest.json'}}
    (out / 'manifest.json').write_text(json.dumps(man, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--source-commit', default='unknown')
    a = ap.parse_args()
    print(json.dumps(export(a.results, a.out, a.source_commit), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
