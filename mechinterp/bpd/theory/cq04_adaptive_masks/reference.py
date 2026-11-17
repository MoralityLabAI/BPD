"""Write-once source-locked results and independent/native counts for CQ-04."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .experiment import SEEDS,ENGINEERING_SEEDS,cohort,N_TEST

HERE=Path(__file__).resolve().parent
LOCK=HERE/'results'/'LOCKED_COUNTS.json'
FILES=('PROTOCOL.md','experiment.py','reference.py')

def blob_sha(path):
    data=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def header():return {p:blob_sha(HERE/p) for p in FILES}

def compact(result):
    return {'schema':result['schema'],'seeds':result['seeds'],
        'success_counts':result['success_counts'],
        'rows':[{'seed':r['seed'],
                 'counts':{k:v['correct'] for k,v in r['metrics'].items()},
                 'costs':{k:v['mean_components'] for k,v in r['metrics'].items()},
                 'by_source':{k:v['by_source'] for k,v in r['metrics'].items()},
                 'donor':{k:r['donor'][k] for k in ('eligible','clean','same_mask','success')},
                 'gauge_masked_accuracy_after_shear':r['gauge_masked_accuracy_after_shear']}
                 for r in result['rows']]}

def report(result):
    rows=result['rows']; diffs=result['per_root_primary_diffs']
    seeds_positive=sum(d>0 for d in diffs)
    n_gate=sum(r['metrics']['adaptive']['correct']>=.9*N_TEST and
              r['metrics']['adaptive']['mean_components']<=3.0 and
              (r['metrics']['adaptive']['correct']-r['metrics']['static_mix']['correct'])>=.15*N_TEST
              for r in rows)
    return {'pilot_threshold_met':n_gate>=6,'pilot_threshold_roots':n_gate,
            'positive_effect_roots':seeds_positive,
            'mean_root_diff':sum(diffs)/len(diffs),
            'mean_slots':result['mean_adaptive_cost'],
            'donor_eligible':sum(r['donor']['eligible'] for r in rows),
            'donor_success':{k:sum(r['donor']['success'][k] for r in rows) for k in rows[0]['donor']['success']},
            'evidence_class':'PILOT ONLY: 8 independent roots, uncalibrated sham tail'}

def check(result,locked):
    assert locked['file_blobs']==header(), 'source/protocol/reference changed since locked counts'
    actual=compact(result); frozen=locked['counts']
    assert actual['schema']==frozen['schema']
    assert actual['seeds']==frozen['seeds']==list(SEEDS)
    assert actual['success_counts']==frozen['success_counts']
    for r,s in zip(actual['rows'],frozen['rows']):
        assert r['seed']==s['seed'] and r['counts']==s['counts']
        assert r['by_source']==s['by_source'] and r['donor']==s['donor']
        assert r['gauge_masked_accuracy_after_shear']==s['gauge_masked_accuracy_after_shear']
        for k in r['costs']:
            assert abs(r['costs'][k]-s['costs'][k])<=.005
    for row in result['rows']:
        assert row['donor']['max_equal_dose_error']<=1e-5
        assert row['gauge_full_error']<=2e-6
        if result['backend']=='native':
            assert row['native_parity_max'] is not None and row['native_parity_max']<=1e-5
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['native','independent'],default='independent')
    p.add_argument('--engineering',action='store_true');p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    result=cohort(args.backend,ENGINEERING_SEEDS if args.engineering else SEEDS)
    result['registered_pilot_status']=report(result)
    if args.check:
        if args.engineering:raise ValueError('engineering runs cannot be checked against prospective lock')
        assert check(result,json.loads(LOCK.read_text()))
        print('CQ04',args.backend,'source-locked count replay PASS')
    if args.out:
        if args.out.exists():raise FileExistsError(args.out)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'backend':args.backend,'seeds':result['seeds'],'counts':result['success_counts'],
                      'pilot_status':result['registered_pilot_status']},indent=2))

if __name__=='__main__':main()
