"""BPD-CQ-06 write-once source-locked staged evaluation and independent-native parity.

Use 'development' and 'sham' before preflight. Confirmation is blocked unless
EvalMath's actual canonical preflight returns READY_FOR_CONFIRMATION on 59
independent sham/development models, before any confirmation root is executed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from math import comb
from fractions import Fraction
from pathlib import Path
import torch
from . import experiment as E
from . import evalmath_preflight as EM

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
SOURCE_FILES=('PROTOCOL.md','REGISTRATION.json','AMENDMENTS.md','experiment.py','reference.py','evalmath_preflight.py')


def git_blob_sha(path):
    content=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_lock():return {name:git_blob_sha(ROOT/name) for name in SOURCE_FILES}


def validate_roster(run,phase):
    roster={'development':E.DEVELOPMENT,'sham':E.SHAM,'confirmation':E.CONFIRMATION}[phase]
    assert run['seeds']==list(roster),f'{phase}: roster changed'
    assert len(set(roster))==len(roster)
    assert set(E.DEVELOPMENT).isdisjoint(E.SHAM)
    assert set(E.DEVELOPMENT).isdisjoint(E.CONFIRMATION)
    assert set(E.SHAM).isdisjoint(E.CONFIRMATION)
    assert all(r['donor']['n_pairs']==128 and r['donor']['all_pairs_scored'] for r in run['rows'])
    assert all(r['donor']['per_root_target_flipped'] and r['donor']['first_target_retained'] for r in run['rows'])
    assert all(r['donor']['max_equal_dose_error']<=1e-5 for r in run['rows'])
    assert all(r['donor']['max_immediate_readout_dark_effect']<=1e-5 for r in run['rows'])
    if phase!='sham':
        assert all(r['task']['myopic_same_first_cue_pairs']['pairs_exactly_opposite_second_label'] for r in run['rows'])
        assert all(r['task']['myopic_same_first_cue_pairs']['pairs_identical_first_cue'] for r in run['rows'])
        assert all(r['task']['myopic_same_first_cue_pairs']['correct']==128 for r in run['rows'])
        for r in run['rows']:
            g=r['gauge']
            assert g['max_full_operator_shear_error']<=2e-6
            if g['min_singular_gap_after_shear']>.001:
                assert g['max_recanonicalized_logit_drift']<=1e-5
                assert g['recanonicalized_mask_switches']==0
    if run['backend']=='native':
        assert all(r['native_max_parity_error'] is not None and r['native_max_parity_error']<=1e-5 for r in run['rows'])


def compact(run):
    out={'schema':run['schema'],'phase':run['phase'],'seeds':run['seeds'],
         'n_donor_pairs_per_root':run['n_donor_pairs_per_root'],
         'n_task_cases_per_root':run['n_task_cases_per_root'],'rows':[]}
    for row in run['rows']:
        rd=row['donor']
        item={'seed':row['seed'],'counts':rd['counts'],'switches':rd['switches'],
              'zero_dose':rd['zero_dose'],'clean_first_recipient':rd['clean_first_recipient'],
              'clean_first_donor':rd['clean_first_donor'],
              'offmanifold':rd['out_of_tanh_range_by_arm'],
              'pair_outcomes':rd['pair_outcomes']}
        if 'task' in row:
            item['task']=row['task']
            item['gauge_switches']=row['gauge']['recanonicalized_mask_switches']
        out['rows'].append(item)
    return out


def verify(run,frozen):
    assert frozen['source_lock']==source_lock(), 'source changed after first run'
    assert compact(run)==compact(frozen['run']), 'per-root or per-pair replay differs'
    validate_roster(run,run['phase'])
    return True


def load(phase):
    path=RESULTS/f'cq06_{phase}.json'
    return json.loads(path.read_text())


def write_once(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(content,f,sort_keys=True,indent=2);f.write('\n')
    return path


def build_preflight():
    sham=load('sham')
    assert sham['source_lock']==source_lock(), 'the sham calibration source changed'
    run=sham['run'];validate_roster(run,'sham')
    assert len(run['rows'])>=59
    pilot=[]
    for row in run['rows']:
        c=row['donor']['counts']['feedback']
        effect=Fraction(c['dark_bpd']-c['manifold_dark_1'],128)
        sham_effect=Fraction(c['manifold_dark_1']-c['manifold_dark_2'],128)
        pilot.append({'unit_id':f"trained-root-{row['seed']}",
                      'paired_effect':str(effect),
                      'sham_effects':[str(sham_effect)]})
    spec={'schema_version':1,'experiment_id':'BPD-CQ-06-59ROOT-MANIFOLD-PREFLIGHT',
          'independent_unit':'one independently rotated and retrained physical/BPD/feedback world',
          'alpha':'0.05','scientific_min_effect':'0.15','target_effect':'0.30',
          'measurement_resolution':'1/128','sham_quantile':'0.95','sham_confidence':'0.95',
          'planned_comparisons':2,'confirm_units_planned':26,
          'pilot_from_development_only':True,'heldout_sealed':True,
          'units_independent':True,'pilot':pilot}
    result=EM.preflight(spec)
    result['sham_source_sha256']=sha256(RESULTS/'cq06_sham.json')
    result['registered_source_lock']=source_lock()
    return spec,result


def sham_gate():
    p=RESULTS/'cq06_preflight_result.json'
    if not p.exists():raise RuntimeError('Confirmation blocked: no pre-data EvalMath preflight receipt')
    r=json.loads(p.read_text())
    if r['decision']!='READY_FOR_CONFIRMATION':
        raise RuntimeError('Confirmation blocked by EvalMath: '+r['decision'])
    if r['registered_source_lock']!=source_lock():raise RuntimeError('Confirmation blocked: source changed after preflight')
    if r['sham_source_sha256']!=sha256(RESULTS/'cq06_sham.json'):
        raise RuntimeError('Confirmation blocked: sham receipt changed')
    if r['pilot_worlds']<59 or r['planned_confirmation_worlds']!=26:
        raise RuntimeError('Confirmation blocked: insufficient independent calibration worlds')
    return r


def sign_test_greater(xs,margin):
    good=sum(x>margin for x in xs)
    n=len(xs)
    return {'positive_roots':good,'n_roots':n,
            'one_sided_exact_p':float(Fraction(sum(comb(n,k) for k in range(good,n+1)),2**n))}


def bootstrap_lower(xs,draws=10000,quantile=.025,seed=777):
    gen=torch.Generator().manual_seed(seed)
    val=torch.tensor(xs,dtype=torch.float64)
    n=len(xs)
    samples=[]
    for j in range(0,draws,1000):
        idx=torch.randint(0,n,(min(1000,draws-j),n),generator=gen)
        samples.append(val[idx].mean(1))
    return float(torch.quantile(torch.cat(samples),quantile))


def decision(run,preflight):
    roots=run['rows']
    diffs=[(r['donor']['counts']['feedback']['dark_bpd']-
            r['donor']['counts']['feedback']['manifold_dark_1'])/128 for r in roots]
    fb=[(r['donor']['counts']['feedback']['dark_bpd']-
         r['donor']['counts']['frozen']['dark_bpd'])/128 for r in roots]
    floor=max(.15,preflight['upper_sham_quantile'],1/128)
    p=sign_test_greater(diffs,floor)
    q=sign_test_greater(fb,0.)
    lb=bootstrap_lower(diffs,seed=777)
    lb2=bootstrap_lower(fb,seed=778)
    model_gate=sum(r['task']['teacher']['both_correct']>=.9*256 and
                   r['task']['full']['both_correct']>=.9*256 and
                   r['task']['feedback']['both_correct']>=.9*256 and
                   r['task']['feedback']['mean_active_slots']<=7.0
                   for r in roots)
    primary=p['one_sided_exact_p']<=.025 and lb>floor
    secondary=q['one_sided_exact_p']<=.025 and lb2>0.
    return {'source':'CQ06 registered 26-root synthetic nonlinear delayed-state family',
            'effective_sham_mechanism_floor':floor,
            'primary_mean_effect':sum(diffs)/len(diffs),'primary_root_effects':diffs,
            'primary_root_sign_test':p,'primary_cluster_bootstrap_lower_97p5':lb,
            'primary_gate':primary,
            'secondary_feedback_mean_effect':sum(fb)/len(fb),
            'secondary_feedback_root_effects':fb,'secondary_root_sign_test':q,
            'secondary_cluster_bootstrap_lower_97p5':lb2,'secondary_gate':secondary,
            'model_gate_roots':model_gate,'model_gate':model_gate>=21,
            'confirmatory_primary_positive_within_synthetic_family':primary and model_gate>=21,
            'failure_or_negative_results_retained':True,
            'scope':'only freshly seeded rotated nonlinear tanh parity worlds, not natural-language transformers'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('development','sham','confirmation','preflight'),required=True)
    parser.add_argument('--backend',choices=('independent','native'),default='independent')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.phase=='preflight':
        if args.backend!='independent':raise ValueError('preflight uses independent development/sham receipt only')
        spec,result=build_preflight()
        write_once(RESULTS/'cq06_preflight_input.json',spec)
        write_once(RESULTS/'cq06_preflight_result.json',result)
        print(json.dumps(result,indent=2,sort_keys=True))
        return 0 if result['decision']=='READY_FOR_CONFIRMATION' else 2
    if args.phase=='confirmation':gate=sham_gate()
    else:gate=None
    run=E.cohort(args.phase,args.backend,include_pairs=True)
    validate_roster(run,args.phase)
    result={'source_lock':source_lock(),'run':run,'stats':E.stats(run)}
    if gate is not None:result['registered_decision']=decision(run,gate)
    if args.check:
        frozen=load(args.phase)
        assert frozen['run']['backend']=='independent'
        verify(run,frozen)
        print(f'CQ-06 {args.phase} {args.backend}: source-locked raw paired replay PASS')
    else:
        if args.backend!='independent':raise ValueError('Only independent backend may write initial frozen receipt')
        write_once(args.out or RESULTS/f'cq06_{args.phase}.json',result)
        print(f'CQ-06 {args.phase}: write-once receipt saved')
    print(json.dumps({'phase':args.phase,'backend':args.backend,'stats':result['stats'],
          'registered_decision':result.get('registered_decision')},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
