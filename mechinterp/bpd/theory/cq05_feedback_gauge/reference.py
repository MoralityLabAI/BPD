"""CQ-05 prospective pilot source-lock and native count replay verifier."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from .experiment import PILOT_SEEDS,cohort,TOL

HERE=Path(__file__).resolve().parent
LOCK=HERE/'results'/'LOCKED_COUNTS.json'
FILES=('PROTOCOL.md','experiment.py','reference.py')

def blob_sha(path):
    data=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def source_hashes():return {name:blob_sha(HERE/name) for name in FILES}

def compact(run):
    return {'schema':run['schema'],'seeds':run['seeds'],'cases_per_seed':run['cases_per_seed'],
        'totals':run['totals'],'donor_totals':run['donor_totals'],
        'rows':[{'seed':r['seed'],'scores':r['scores'],
                 'donor':{'n':r['donor']['n'],'clean_pairs':r['donor']['clean_pairs'],
                          'same_plan':r['donor']['same_plan'],
                          'clean_same_plan':r['donor']['clean_same_plan'],
                          'zero_dose':r['donor']['zero_dose'],
                          'pass_two_switches':r['donor']['pass_two_switches'],
                          'success_all_pairs':r['donor']['success_all_pairs']},
                 'gauge_switches':r['gauge']['max_feedback_mask_switches_after_canonicalization']}
                 for r in run['rows']]}

def science(run):
    rows=run['rows'];n=len(rows)
    def donor_delta(r,mode='feedback',comp='random_dark'):
        x=r['donor']['success_all_pairs'][mode]
        return (x['dark_native']-x[comp])/r['donor']['n']
    root_diffs=[donor_delta(r) for r in rows]
    feedback_gain=[(r['donor']['success_all_pairs']['feedback']['dark_native']
                   -r['donor']['success_all_pairs']['frozen']['dark_native'])/r['donor']['n']
                   for r in rows]
    task_ok=sum(r['scores']['feedback']['correct']>=.9*run['cases_per_seed'] and
                r['scores']['feedback']['mean_slots']<=3 for r in rows)
    return {'pilot_only':True,'task_gate_roots':task_ok,'task_gate_met':task_ok>=6,
        'donor_mechanistic_minimum':0.15,
        'primary_donor_root_differences':root_diffs,
        'primary_donor_roots_at_floor':sum(d>=.15 for d in root_diffs),
        'donor_direction_gate_met':sum(d>=.15 for d in root_diffs)>=6,
        'feedback_vs_frozen_donor_root_differences':feedback_gain,
        'positive_feedback_vs_frozen_roots':sum(d>0 for d in feedback_gain),
        'evidence_class':'prospective synthetic pilot, no 59-root sham tolerance or separate confirmation'}

def verify(result,locked):
    if locked['source_sha256']!=source_hashes():
        raise AssertionError('CQ05 source or protocol changed after locked pilot')
    now=compact(result);before=locked['counts']
    if now['schema']!=before['schema'] or now['seeds']!=before['seeds'] or now['cases_per_seed']!=before['cases_per_seed']:
        raise AssertionError('CQ05 cohort mismatch')
    if now['totals']!=before['totals'] or now['donor_totals']!=before['donor_totals']:
        raise AssertionError('CQ05 score mismatch')
    for a,b in zip(now['rows'],before['rows']):
        if a['seed']!=b['seed'] or a['donor']!=b['donor'] or a['gauge_switches']!=b['gauge_switches']:
            raise AssertionError(f"CQ05 donor/geometry mismatch for {a['seed']}")
        for name in a['scores']:
            x=a['scores'][name];y=b['scores'][name]
            if x['correct']!=y['correct'] or x['by_source']!=y['by_source']:
                raise AssertionError(f"CQ05 score mismatch for {a['seed']} {name}")
            if abs(x['mean_slots']-y['mean_slots'])>1e-4:
                raise AssertionError('component budget mismatch')
    for r in result['rows']:
        if r['donor']['max_equal_dose_error']>1e-5:raise AssertionError('physical dose mismatch')
        if r['gauge']['max_full_operator_delta']>2e-6:raise AssertionError('GL2 invariant failed')
        if r['geometry']['singular_gap']>.001:
            if r['gauge']['max_canonical_masked_readout_delta']>1e-5:
                raise AssertionError('canonical gauge readout drift')
            if r['gauge']['max_feedback_mask_switches_after_canonicalization']!=0:
                raise AssertionError('canonical gauge mask switch')
    return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=('independent','native'),default='independent')
    p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path)
    a=p.parse_args();result=cohort(a.backend)
    result['registered_pilot_status']=science(result)
    if a.check:
        verify(result,json.loads(LOCK.read_text()))
        print('CQ05',a.backend,'locked pilot count replay PASS')
    if a.out:
        if a.out.exists():raise FileExistsError(a.out)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'backend':a.backend,'success':result['totals'],
           'donor':result['donor_totals'],'science':result['registered_pilot_status']},indent=2))

if __name__=='__main__':main()
