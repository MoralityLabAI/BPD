"""DataGoblin TRAIN-only export of curriculum and intervention decisions, plus resource ledger.

Two record kinds, both drawn only from TRAIN partitions (fit-view proposals,
feedback-view panel and calibration cohort):

* ``curriculum``: per-candidate decisions of C_mech_fresh and D_atlas in the
  fixed-CPU engineering runs (predicted value, selection reason, window
  verification outcome, observer provenance, cost);
* ``intervention``: per-edge cached/restricted/live proposal outcomes from the
  drift instrument (estimated linear gain, realized finite gain, collateral,
  physical dose, admission verdict and its scope).

``scope`` distinguishes mathematical certificates from heuristic predictions.
No record here carries a certificate: native outputs have no uniform bounds
(MATH_REVIEW sections 2.2 and 2.8), so every intervention admission verdict is
``HEURISTIC_MEASURED_INPUTS`` and every outcome is ``VERIFIED_FINITE_ON_DECLARED_QUERIES``.
Rows with ``training_eligible`` false must not enter DataGoblin training intake.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from .common import ENGINEERING_ROOTS, HERE, dump_json

ENG = HERE / 'results' / 'engineering'
OUT = HERE / 'results' / 'datagoblin_export'
DRIFT = {28000: 'drift_historical_28000', 28001: 'drift_historical_28001', 29200: 'drift_29200', 29201: 'drift_29201'}


def curriculum_rows():
    for prefix, mode in (('economics', 'transactional'), ('economics_batched', 'batched')):
        yield from _curriculum(prefix, mode)


def _curriculum(prefix, mode):
    for root in ENGINEERING_ROOTS:
        p = ENG / f'{prefix}_{root}' / 'train_decisions.jsonl'
        if not p.is_file():
            continue
        for line in p.read_text(encoding='utf-8').splitlines():
            d = json.loads(line)
            if d.get('split') != 'train':
                raise ValueError('non-TRAIN curriculum record')
            yield dict(kind='curriculum', root=d['root'], arm=d['arm'], window=d['window'], edge=d['edge'],
                       delay=d['delay'], checkpoint_sha256=d['checkpoint_sha256'],
                       decision='selected' if d['selected'] else 'not_selected', reason=d['reason'],
                       estimated_value=d['predicted_gain'], realized_window_value=d['window_realized_gain'],
                       window_predicted_value=d['window_predicted_gain'], verifier=d['verifier'],
                       verifier_outcome='accepted' if d['window_accepted'] else 'rolled_back',
                       provenance=dict(observer=d['observer'], observer_age=d['observer_age'],
                                       atlas_decision=d['atlas_decision']),
                       physical_dose=d['dose'], cost_cpu_seconds=d['preview_cpu_seconds'],
                       scope=d['scope'], preview_mode=mode, split='train', training_eligible=True)


def intervention_rows():
    for root, name in DRIFT.items():
        p = ENG / name / 'drift.json'
        if not p.is_file():
            continue
        r = json.loads(p.read_text(encoding='utf-8'))
        panel = r['panel']
        for t, ck in r['checkpoints'].items():
            for i, e in enumerate(ck['edges']):
                for sp, s in e['spaces'].items():
                    adm = s.get('admission_k4')
                    for key, o in s.items():
                        if not isinstance(o, dict) or 'target_gain' not in o or key == 'dose_grid':
                            continue
                        kind = key.split('_')[0]
                        yield dict(kind='intervention', root=root, historical_root=root in (28000, 28001),
                                   checkpoint_update=int(t), checkpoint_sha256=ck['state_sha256'],
                                   factor_frame=ck['factor_frame'], edge=panel[i], action_space=sp,
                                   proposal=key, proposal_source=kind,
                                   estimated_value=o.get('linear_gain'), realized_value=o['target_gain'],
                                   protected_change_l2=o['protected_l2'], target_flipped=o['target_flipped'],
                                   corruption=[o['corruption_numerator'], o['corruption_denominator']],
                                   physical_dose=o['dose'], frame_residual=o.get('frame_residual'),
                                   admission=(dict(verdict=adm['verdict'], scope=adm['scope'])
                                              if (adm and key == 'cached_k4' and 'scope' in adm) else None),
                                   cost_cpu_seconds=e['cost'], scope='VERIFIED_FINITE_ON_DECLARED_QUERIES',
                                   split='train', training_eligible=root not in (28000, 28001))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in (('curriculum_decisions.jsonl.gz', curriculum_rows()),
                       ('intervention_decisions.jsonl.gz', intervention_rows())):
        n = 0
        # mtime=0 keeps the gzip bytes deterministic
        with gzip.GzipFile(OUT / name, 'wb', mtime=0) as fh:
            for row in rows:
                fh.write((json.dumps(row, sort_keys=True) + '\n').encode('utf-8'))
                n += 1
        counts[name] = n
    ledger = {}
    summ = ENG / 'summary.json'
    if summ.is_file():
        s = json.loads(summ.read_text(encoding='utf-8'))
        for root, r in s['economics'].items():
            for tag, a in r['arms'].items():
                ledger[f'{root}:{tag}'] = dict(cpu_seconds=a['cpu'], windows=a['windows'], counters=a['counters'],
                                               atlas=a['atlas'], cache_hits=a['cache_hits'], overrun=a['overrun'])
            ledger[f'{root}:memory'] = r['memory']
        for root, r in s['drift_per_root'].items():
            ledger[f'{root}:drift_memory'] = r['memory']
    dump_json(OUT / 'manifest.json', dict(counts=counts, split='train_only',
                                          excluded=['tune', 'development', 'eval', 'reserved test', 'ARG-03-N'],
                                          historical_roots_training_eligible=False, resource_ledger=ledger))
    print(json.dumps(counts))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
