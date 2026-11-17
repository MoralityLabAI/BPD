"""BPD-CQ-03 immutable receipt creator and source-locked CPU verifier."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .experiment import SEEDS, cohort

HERE=Path(__file__).resolve().parent
CHECK_FILES=('PROTOCOL.md','experiment.py','reference.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header():return {'protocol_sha256':sha(HERE/'PROTOCOL.md'), 'source_sha256':{name:sha(HERE/name) for name in CHECK_FILES if name!='reference.py'},'seed_roster':list(SEEDS)}


def summary(result):
    return {'eligible_total':result['eligible_total'],'success_counts':result['success_counts'],
            'clean_by_seed':[x['clean_n'] for x in result['rows']],
            'observable_rank_by_seed':[x['delayed_observable_rank'] for x in result['rows']],
            'gauge_singleton_drift_by_seed':[x['gauge_singleton_drift'] for x in result['rows']],
            'max_norm_error':max(x['max_norm_error'] for x in result['rows']),
            'max_full_gauge_error':max(x['gauge_full_write_error'] for x in result['rows'])}


def evaluate_pass(result):
    rows=result['rows']
    return {
        'engineering_valid':len(rows)==8 and [x['seed'] for x in rows]==list(SEEDS)
            and all(x['max_norm_error']<=1e-5 and x['gauge_full_write_error']<=2e-6
                    and x['delayed_observable_rank']>=1 and x['clean_n']<=128
                    and (x['native_parity_max'] is None or x['native_parity_max']<=1e-5)
                    for x in rows),
        'science_gate':sum(x['clean_n']>=110
            and x['transfer']['full_state']>=.95*x['clean_n']
            and x['transfer']['dark_native']-x['transfer']['random_dark']>=.20*x['clean_n']
            for x in rows)>=6,
        'seed_passes':sum(x['clean_n']>=110
            and x['transfer']['full_state']>=.95*x['clean_n']
            and x['transfer']['dark_native']-x['transfer']['random_dark']>=.20*x['clean_n']
            for x in rows),
    }


def build(backend):
    result=cohort(backend)
    result['source_lock']=header()
    result['summary']=summary(result)
    result['registered_outcome']=evaluate_pass(result)
    return result


def verify(actual,frozen,backend):
    assert actual['backend']==frozen['backend']==backend
    assert actual['source_lock']==frozen['source_lock']==header()
    assert actual['schema']==frozen['schema']=='bpd.cq03.learned_recurrence.v1'
    assert actual['seeds']==frozen['seeds']==list(SEEDS)
    assert actual['registered_outcome']==frozen['registered_outcome']
    # Per-seed finite exact counts must replay. Small float host variance in
    # trained factors is tolerated only where it does not change counts.
    for a,b in zip(actual['rows'],frozen['rows']):
        assert a['seed']==b['seed'] and a['clean_n']==b['clean_n']
        assert a['transfer']==b['transfer'] and a['current_visible_preserved']==b['current_visible_preserved']
        assert a['delayed_observable_rank']==b['delayed_observable_rank']
        assert abs(a['gauge_singleton_drift']-b['gauge_singleton_drift'])<.005
        assert a['max_norm_error']<=1e-5
        if backend=='native':assert a['native_parity_max'] is not None and a['native_parity_max']<=1e-5
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['independent','native'],default='independent')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--check',action='store_true')
    a=p.parse_args();current=build(a.backend)
    if a.check:
        frozen=json.loads(a.out.read_text(encoding='utf-8'))
        verify(current,frozen,a.backend)
        print('BPD-CQ03',a.backend,'locked CPU replay PASS',current['summary'],current['registered_outcome'])
    else:
        if a.out.exists():raise FileExistsError(f'write-once immutable receipt exists: {a.out}')
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(current,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        print('BPD-CQ03',a.backend,'written',a.out,current['summary'],current['registered_outcome'])

if __name__=='__main__':main()
