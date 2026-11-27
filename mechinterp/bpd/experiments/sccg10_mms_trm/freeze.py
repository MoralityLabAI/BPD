"""Write DEV_LOCK.json: frozen development roster, budgets, thresholds and source hashes.

Run once, after engineering and before any development outcome exists. The
economics trainer refuses development roots unless the live source hashes and
budget/roster equal this lock.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .atlas import Policy
from .common import DEVELOPMENT_ROOTS, HERE, dump_json, file_sha256, source_hashes
from .economics import ARMS, NO_AUDIT_WINDOWS

LOCK = HERE / 'DEV_LOCK.json'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview-mode', choices=('transactional', 'batched'), required=True)
    ap.add_argument('--rationale', required=True)
    a = ap.parse_args()
    if LOCK.exists():
        raise SystemExit('DEV_LOCK.json already exists; a freeze is never rewritten')
    if any((HERE / 'results' / 'development').glob('*/economics.json')):
        raise SystemExit('development outcomes already exist; cannot freeze after outcomes')
    summary = HERE / 'results' / 'engineering' / 'summary.json'
    lock = dict(
        schema='BPD-SCCG10-MMS-TRM/dev-lock/v1',
        evidence_class='DEVELOPMENT_ONLY',
        roots=list(DEVELOPMENT_ROOTS),
        frozen=dict(cpu_limit=100.0, windows=80, arms=list(ARMS), atlas_tiers=[16.0], preview_mode=a.preview_mode,
                    preview_mode_rationale=a.rationale,
                    policy=asdict(Policy()), no_audit_windows=NO_AUDIT_WINDOWS,
                    warmup_updates=6000, threads=2, memory_cap_bytes=6442450944),
        endpoints=dict(
            primary='fixed-CPU ordinary steering-disabled delay-31 accuracy, D_atlas_mms16 minus A_ordinary, per root',
            secondary=['cpu D_atlas_mms16 - C_mech_fresh', 'cpu D_atlas_mms16 - B_ord_grad',
                       'cpu D_atlas_mms16 - E_shuffled_mms16'],
            family='Holm over primary + 3 secondary, exact two-sided root sign-flip',
            descriptive=['fixed-update contrasts', 'd2/d7/d63/d127', 'binding d2/d7/d31', 'feedback losses',
                         'panel hybrid flips/corruption', 'atlas reuse/refresh/trace residuals', 'CPU ledger'],
            unit='independently initialized TRM model root; probes, panels, checkpoints, queries are not units',
            practical_floor_pp=5.0, floor_status='planning value inherited from CURATE-02, not established',
            sham='E_shuffled_mms16 minus A_ordinary (matched shuffled mechanism), one per root',
            preflight='EvalMath strict mechinterp-power preflight; heldout_sealed=false (historical EVAL '
                      'partition is exposed), so the decision is REDESIGN_OR_PILOT_ONLY regardless of outcome'),
        causal_auxiliary='closed: CURATE-02 behavioral selective-control admission failed; no fresh gate passed',
        engineering_summary_sha256=file_sha256(summary) if summary.is_file() else None,
        source_hashes=source_hashes(),
    )
    dump_json(LOCK, lock)
    print(json.dumps({'lock': str(LOCK.name), 'files': len(lock['source_hashes'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
