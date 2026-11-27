"""Write the prospective development freeze (source lock, recipe, arms, roots, statistics)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import DEVELOPMENT_ROOTS, REPO, Recipe, dump_json
from .devrun import lf_sha256

LOCKED = [
    'mechinterp/bpd/experiments/sccg10_native_mms/' + f for f in (
        '__init__.py', 'admission.py', 'atlas.py', 'common.py', 'devrun.py', 'economics.py',
        'ladder.py', 'mechanism.py', 'native.py', 'replay.py', 'support.py', 'trainer.py')
] + [
    'mechinterp/bpd/experiments/trm_curation/' + f for f in (
        '__init__.py', 'cache.py', 'data.py', 'memory_cap.py', 'model.py', 'observer.py', 'reference.py',
        'scheduler.py', 'signal_bank.py', 'transaction.py')
] + [
    'mechinterp/bpd/experiments/training_steering/data.py',
    'mechinterp/bpd/experiments/training_steering_stability/data.py',
    'mechinterp/bpd/experiments/training_steering_stability/stats.py',
    'mechinterp/bpd/components.py', 'mechinterp/bpd/config.py', 'mechinterp/bpd/__init__.py',
    'mechinterp/bpd/experiments/repair_portfolio/evidence/preflight.py',
]
ANALYSIS_TOOLS = ['mechinterp/bpd/experiments/sccg10_native_mms/' + f
                  for f in ('analyze.py', 'export_datagoblin.py', 'freeze.py', 'DEV_PROTOCOL.md')]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--decisions', type=Path, required=True, help='JSON with engineering-derived choices')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    dec = json.loads(a.decisions.read_text(encoding='utf-8'))
    hashes = {f: lf_sha256(REPO / f) for f in LOCKED}
    lock = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    recipe = Recipe().as_dict()
    fz = {'schema': 'SCCG10-MMS-TRM/dev-freeze/v1', 'evidence_class': 'DEVELOPMENT_ONLY',
          'source_hashes': hashes, 'source_lock_sha256': lock, 'recipe': recipe,
          'analysis_tool_hashes_at_freeze': {f: lf_sha256(REPO / f) for f in ANALYSIS_TOOLS},
          'hash_normalization': 'sha256 of file bytes with CRLF replaced by LF',
          'development_roots': list(DEVELOPMENT_ROOTS), 'output_dir': dec['output_dir'],
          'arm_configs': dec['arm_configs'], 'arm_roles': dec['arm_roles'],
          'statistics': dec['statistics'], 'engineering_basis': dec['engineering_basis'],
          'primary_endpoint': 'ordinary steering-disabled final-development delay-31 accuracy (512 cases/root), '
                              'D minus A at fixed total CPU, paired by independent model root',
          'custody': {'reads': ['TRAIN fit/feedback views (training, scoring, acceptance)',
                                'historical EVAL tuple partition with fresh SCCG10 seeds (final development only)'],
                      'never': ['TRAIN tune view for development decisions', 'ARG-03-N', 'any reserved cohort'],
                      'historical_roots_replay_only': [28000, 28001] + list(range(28101, 28109))},
          'behavioral_gate': 'FAIL on root 29200 (results/engineering/BEHAVIORAL_GATE_RESULT.json): causal auxiliary cross closed',
          'stop_rules': 'Report every root including failures; no reselection of primary; no promotion.'}
    dump_json(a.out, fz)
    print(json.dumps({'source_lock_sha256': lock, 'files': len(hashes)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
