"""AMM-05: prospectively registered case-conditioned BPD strategy transfer.

Source-only engineering tests are restricted to excluded engineering roots.
Confirmation requires a sealed release manifest *before* any model access.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any
import torch
from . import models as m, case

NAME=m.NAME
SCHEMA=m.SCHEMA
ROOT=Path(__file__).resolve().parent
F=case.FEATURE_COUNT
LAMBDA_MEMORY=10.
LAMBDA_LOCAL=2.
LAMBDA_RESIDUAL=20.
JUDGE_FLIP=.35
N_LOCAL=4
BOOTSTRAPS=2000
SCIENCE_FLOOR=.08
ARMS=('memory_calibrated','local_ridge','local_count','memory_only',
      'geometry_rule','pooled','random_component','random_dark',
      'random_dark_b','sham','full_donor')
COMPARISONS={'memory_vs_local_ridge':'local_ridge',
             'memory_vs_local_count':'local_count',
             'memory_vs_geometry_rule':'geometry_rule'}


def source_sha256()->dict[str,str]:
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('models.py','case.py','experiment.py')}


def ridge(X:torch.Tensor,Y:torch.Tensor,lam:float,intercept_penalty:bool=False)->list[float]:
    if X.ndim!=2 or X.shape[1]!=F or Y.shape!=(len(X),):
        raise ValueError('wrong ridge dimensions')
    X=X.double();Y=Y.double()
    penalty=torch.eye(F,dtype=torch.float64)*lam
    if not intercept_penalty:penalty[0,0]=0.
    w=torch.linalg.solve(X.T@X+penalty,X.T@Y)
    if not bool(torch.isfinite(w).all()):raise ArithmeticError('nonfinite ridge')
    return w.tolist()


def predict(features:list,weights:list[float])->torch.Tensor:
    X=torch.tensor(features,dtype=torch.float64)
    if X.ndim!=3 or X.shape[1:]!=(2,F):raise ValueError('invalid per-case features')
    return X@torch.tensor(weights,dtype=torch.float64)


def train():
    rows=[];xs=[];ys=[];totals=[0,0]
    for i in range(m.PHASE_N['training']):
        model,f,metrics=m.fit('training',i)
        scored=case.assay('training',i,f,'training_pairs',m.TRAIN_PAIRS)
        feats=torch.tensor(scored['features'],dtype=torch.float64)
        outcomes=torch.tensor([[int(scored['candidate_bits'][f'component_{c}'][j]) for c in (0,1)]
                               for j in range(m.TRAIN_PAIRS)],dtype=torch.float64)
        xs.append(feats.reshape(-1,F));ys.append(outcomes.reshape(-1))
        for c in (0,1):totals[c]+=scored['candidate_hits'][f'component_{c}']
        rows.append({'root':m.root_key('training',i),'family':m.family_for(i),
                     'fitting':metrics,'assay':scored})
    weights=ridge(torch.cat(xs),torch.cat(ys),LAMBDA_MEMORY)
    out={'schema':SCHEMA+'/training_memory','experiment':NAME,'phase':'training',
         'independent_training_roots':m.PHASE_N['training'],
         'training_pairs_per_root':m.TRAIN_PAIRS,
         'component_training_hits':{'0':totals[0],'1':totals[1]},
         'pooled_action':0 if totals[0]>=totals[1] else 1,
         'coefficients':weights,'memory_ridge_lambda':LAMBDA_MEMORY,
         'source_sha256':source_sha256(),'rows':rows}
    out['sha256']=m.digest(out)
    return out


def validate_memory(memory:dict):
    if not isinstance(memory,dict) or memory.get('schema')!=SCHEMA+'/training_memory':
        raise ValueError('AMM-05 frozen training policy required')
    if memory.get('source_sha256')!=source_sha256():
        raise ValueError('model and memory source mismatch')
    if memory.get('sha256')!=m.digest({k:v for k,v in memory.items() if k!='sha256'}):
        raise ValueError('training memory digest failed')
    if memory.get('independent_training_roots')!=64 or len(memory.get('rows',[]))!=64:
        raise ValueError('training source-root count invalid')
    if {row['root'] for row in memory['rows']}!={m.root_key('training',i) for i in range(64)}:
        raise ValueError('training root membership invalid')
    if len(memory.get('coefficients',[]))!=F or not all(math.isfinite(x) for x in memory['coefficients']):
        raise ValueError('invalid memory weights')
    return True


def judge_report(phase:str,index:int,cal:dict):
    if cal['n']!=N_LOCAL or cal['split']!='local_calibration':
        raise ValueError('invalid calibration cohort')
    bits={}
    for c in (0,1):
        key=f'component_{c}'
        rng=m.gen(m.root_key(phase,index),'judge_noise',c)
        corrupt=(torch.rand(N_LOCAL,generator=rng)<JUDGE_FLIP).tolist()
        clean=cal['candidate_bits'][key]
        bits[key]=''.join(str(int((ch=='1')!=flip)) for ch,flip in zip(clean,corrupt))
    return {'reported_bits':bits,
            'reported_hits':{k:v.count('1') for k,v in bits.items()},
            'n_feedback_labels':2*N_LOCAL,'flip_probability':JUDGE_FLIP}


def calibration_weights(cal_features:list,feedback:dict,memory:dict):
    X=torch.tensor(cal_features,dtype=torch.float64).reshape(-1,F)
    labels=torch.tensor([[int(feedback['reported_bits'][f'component_{c}'][j])
                         for c in (0,1)] for j in range(N_LOCAL)],dtype=torch.float64).reshape(-1)
    corrected=(labels-JUDGE_FLIP)/(1.-2.*JUDGE_FLIP)
    local_weights=ridge(X,corrected,LAMBDA_LOCAL)
    mem_prior=predict(cal_features,memory['coefficients']).clamp(0,1).reshape(-1)
    residual=ridge(X,corrected-mem_prior,LAMBDA_RESIDUAL,intercept_penalty=True)
    return local_weights,residual


def choose(feats:list,geo:dict,feedback:dict,memory:dict,local_weights:list,
           residual_weights:list,root:str)->dict[str,list[int]]:
    """No donor targets, scored test labels, or donor continuation are passed."""
    x=torch.tensor(feats,dtype=torch.float64)
    prior=predict(feats,memory['coefficients']).clamp(0,1)
    residual=predict(feats,residual_weights)
    local=predict(feats,local_weights)
    n=len(feats)
    def argmax_each(y):
        return torch.argmax(y,dim=1).tolist()
    hits=feedback['reported_hits']
    if hits['component_0']>hits['component_1']:fixed=0
    elif hits['component_1']>hits['component_0']:fixed=1
    else:fixed=0 if geo['signed_gains'][0]>=geo['signed_gains'][1] else 1
    return {
        'memory_calibrated':argmax_each(prior+residual),
        'local_ridge':argmax_each(local),
        'local_count':[fixed]*n,
        'memory_only':argmax_each(prior),
        'geometry_rule':argmax_each(x[:,:,5]),
        'pooled':[memory['pooled_action']]*n,
        'random_component':[m.seed(root,'random_component',i)%2 for i in range(n)]
    }


def score_one(phase:str,index:int,memory:dict)->dict:
    _,f,metrics=m.fit(phase,index)
    root=m.root_key(phase,index)
    geo=m.geometry(f,m.family_for(index))
    cal=case.assay(phase,index,f,'local_calibration',N_LOCAL)
    feedback=judge_report(phase,index,cal)
    local_weights,residual_weights=calibration_weights(cal['features'],feedback,memory)
    held=case.assay(phase,index,f,'heldout_test',m.PAIRS)
    decisions=choose(held['features'],geo,feedback,memory,local_weights,residual_weights,root)
    bits={}
    for arm,actions in decisions.items():
        bits[arm]=''.join(held['candidate_bits'][f'component_{c}'][i]
                          for i,c in enumerate(actions))
    for key in ('random_dark','random_dark_b','sham','full_donor'):
        bits[key]=held['candidate_bits'][key]
    repeated=case.assay(phase,index,f,'repeatability',m.PAIRS)
    repeat_dec=choose(repeated['features'],geo,feedback,memory,local_weights,residual_weights,root)
    repeat_local_bits=''.join(repeated['candidate_bits'][f'component_{c}'][i]
                              for i,c in enumerate(repeat_dec['local_ridge']))
    judge_flips={key:sum(a!=b for a,b in zip(cal['candidate_bits'][key],feedback['reported_bits'][key]))
                 for key in ('component_0','component_1')}
    return {'root':root,'family':m.family_for(index),'fitting':metrics,
            'calibration_clean_audit':cal,'noisy_feedback':feedback,
            'calibration_judge_audit_flip_counts':judge_flips,
            'local_weights':local_weights,'memory_residual_weights':residual_weights,
            'heldout':held,'policy_actions':decisions,'policy_bits':bits,
            'policy_hits':{k:v.count('1') for k,v in bits.items()},
            'repeatability':{'assay':repeated,
                             'local_ridge_actions':repeat_dec['local_ridge'],
                             'local_ridge_bits':repeat_local_bits,
                             'local_ridge_hits':repeat_local_bits.count('1')}}


def admit(phase:str,memory:dict|None,preflight:dict|None=None,release:dict|None=None):
    """Fail closed before any confirmation model fitting or root generation."""
    if phase not in ('development','confirmation'):
        raise ValueError('evaluation phase must be development or confirmation')
    validate_memory(memory)
    if phase=='confirmation':
        pf=preflight
        if not isinstance(pf,dict) or any((
            pf.get('schema')!='EvalMath/mechinterp-power-preflight/v1',
            pf.get('decision')!='READY_FOR_CONFIRMATION',
            pf.get('experiment_id')!=NAME,
            pf.get('pilot_worlds')!=64,
            pf.get('planned_confirmation_worlds')!=128,
            pf.get('planned_comparisons')!=3,
            pf.get('mean_pilot_effect',0)<=pf.get('mechanistic_variation_floor',1),
            bool(pf.get('reasons')),
        )):
            raise ValueError('corrected strict EvalMath admission required')
        if not isinstance(release,dict) or any((
            release.get('release_authorized') is not True,
            release.get('experiment')!=NAME,
            release.get('source_sha256')!=source_sha256(),
            release.get('memory_sha256')!=memory['sha256'],
            release.get('preflight_sha256')!=m.digest(pf),
            release.get('independent_confirmation_roots')!=128,
            release.get('registered_population')!='AMM05/confirmation/0..127',
            release.get('development_oracle_headroom_rate',0)<.08,
            not isinstance(release.get('approval_commit'),str),
        )):
            raise ValueError('independent explicit confirmation release manifest required')


def preflight_input(run:dict):
    if run['phase']!='development' or run['n_independent_roots']!=64:
        raise ValueError('64 separate development roots required')
    rows=[]
    for r in run['rows']:
        hits=r['policy_hits']
        rows.append({'unit_id':r['root'],
                     'paired_effect':f"{hits['memory_calibrated']-hits['local_ridge']}/{m.PAIRS}",
                     'sham_effects':[f"{hits['local_ridge']-r['repeatability']['local_ridge_hits']}/{m.PAIRS}"]})
    return {'schema_version':1,'experiment_id':NAME,
            'independent_unit':'one separately trained physical recurrent model plus BPD factor pair',
            'alpha':'0.05','scientific_min_effect':'0.08',
            'target_effect':'0.20','measurement_resolution':f'1/{m.PAIRS}',
            'sham_quantile':'0.95','sham_confidence':'0.95',
            'planned_comparisons':3,'confirm_units_planned':128,
            'pilot_from_development_only':True,'heldout_sealed':True,
            'units_independent':True,'pilot':rows}


def root_sign_p(diffs:list[int])->float:
    n=sum(x!=0 for x in diffs);k=sum(x>0 for x in diffs)
    if n==0:return 1.0
    return sum(math.comb(n,i) for i in range(k,n+1))/2**n


def holm(ps:dict[str,float])->dict[str,float]:
    adjusted={};floor=0.
    for j,key in enumerate(sorted(ps,key=lambda z:(ps[z],z))):
        floor=max(floor,min(1.,(len(ps)-j)*ps[key]))
        adjusted[key]=floor
    return adjusted


def bootstrap(diffs:list[int],label:str)->list[float]:
    rng=random.Random(m.seed('bootstrap',label))
    n=len(diffs)
    vals=sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/(n*m.PAIRS) for _ in range(BOOTSTRAPS))
    return [vals[50],vals[1949]]


def analyze(run:dict,pf:dict)->dict:
    if run['phase']!='confirmation':raise ValueError('confirmation phase only')
    diffs={k:[r['policy_hits']['memory_calibrated']-r['policy_hits'][v] for r in run['rows']]
           for k,v in COMPARISONS.items()}
    p={k:root_sign_p(v) for k,v in diffs.items()}
    adj=holm(p)
    out={k:{'mean_effect':sum(v)/(len(v)*m.PAIRS),
            'root_bootstrap_95_ci':bootstrap(v,k),
            'positive_roots':sum(x>0 for x in v),'negative_roots':sum(x<0 for x in v),
            'tied_roots':sum(x==0 for x in v),
            'root_sign_p_one_sided':p[k],'holm_adjusted_p':adj[k]} for k,v in diffs.items()}
    floor=max(SCIENCE_FLOOR,pf['mechanistic_variation_floor'])
    err=max(r['heldout']['diagnostics']['max_equal_dose_error'] for r in run['rows'])
    vis=max(r['heldout']['diagnostics']['max_immediate_readout_displacement'] for r in run['rows'])
    primary=out['memory_vs_local_ridge']
    return {'schema':SCHEMA+'/analysis','comparisons':out,
            'registered_floor':floor,'max_equal_dose_error':err,
            'max_visible_drift':vis,
            'registered_primary_pass':bool(primary['root_bootstrap_95_ci'][0]>floor
                                           and primary['holm_adjusted_p']<.05
                                           and err<=1e-5 and vis<=1e-5)}


def evaluate(phase:str,memory:dict|None=None,pf:dict|None=None,release:dict|None=None)->dict:
    admit(phase,memory,pf,release)
    assert memory is not None
    rows=[score_one(phase,i,memory) for i in range(m.PHASE_N[phase])]
    totals={k:sum(r['policy_hits'][k] for r in rows) for k in ARMS}
    out={'schema':SCHEMA+'/cohort','experiment':NAME,'phase':phase,
         'source_sha256':source_sha256(),'training_memory_sha256':memory['sha256'],
         'n_independent_roots':len(rows),'n_calibration_pairs_per_root':N_LOCAL,
         'calibration_feedback_bits_per_root':N_LOCAL*2,
         'n_test_pairs_per_root':m.PAIRS,'test_cases':m.PAIRS*len(rows),
         'totals':totals,'rows':rows}
    if phase=='development':
        out['preflight_input']=preflight_input(out)
        out['posthoc_oracle_headroom_hits']=sum(
            r['heldout']['diagnostics']['casewise_oracle_hits_audit_only']-r['policy_hits']['local_ridge']
            for r in rows)
        out['posthoc_oracle_headroom_rate']=out['posthoc_oracle_headroom_hits']/out['test_cases']
    else:
        out['preflight_sha256']=m.digest(pf)
        out['release_sha256']=m.digest(release)
        out['analysis']=analyze(out,pf)
    return out


def compact(run:dict)->dict:
    if run['phase']=='training':
        return {'schema':SCHEMA+'/training_lock','source_sha256':run['source_sha256'],
                'memory_sha256':run['sha256'],'pooled_action':run['pooled_action'],
                'independent_roots':64,'train_candidate_hits':run['component_training_hits'],
                'train_rows_sha256':m.digest([{'root':r['root'],
                    'candidate_bits':r['assay']['candidate_bits']} for r in run['rows']])}
    out={'schema':SCHEMA+'/cohort_lock','phase':run['phase'],
         'source_sha256':run['source_sha256'],
         'memory_sha256':run['training_memory_sha256'],
         'independent_roots':run['n_independent_roots'],'test_cases':run['test_cases'],
         'totals':run['totals'],
         'all_case_bits_sha256':m.digest([{'root':r['root'],
               'clean_calibration':r['calibration_clean_audit']['candidate_bits'],
               'noisy_calibration':r['noisy_feedback']['reported_bits'],
               'candidate_bits':r['heldout']['candidate_bits'],
               'policy_actions':r['policy_actions'],'policy_bits':r['policy_bits'],
               'repeat_bits':r['repeatability']['local_ridge_bits']} for r in run['rows']]),
         'full_receipt_sha256':m.digest(run)}
    if run['phase']=='development':
        out['headroom_rate']=run['posthoc_oracle_headroom_rate']
        out['preflight_input_sha256']=m.digest(run['preflight_input'])
    else:
        out['preflight_sha256']=run['preflight_sha256']
        out['analysis']=run['analysis']
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('engineering','training','development','confirmation'))
    parser.add_argument('--memory',type=Path)
    parser.add_argument('--preflight',type=Path)
    parser.add_argument('--release',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--check-lock',type=Path)
    args=parser.parse_args()
    if args.phase=='training':result=train()
    elif args.phase=='engineering':
        rows=[]
        for i in range(2):
            _,f,metrics=m.fit('engineering',i)
            rows.append({'root':m.root_key('engineering',i),'fit':metrics,
                        'cases':case.assay('engineering',i,f,'engineering_test',m.PAIRS),
                        'gauge':m.gauge_check(f,m.family_for(i))})
        result={'phase':'engineering','rows':rows}
    else:
        if args.memory is None:raise ValueError('frozen memory path required')
        memory=json.loads(args.memory.read_text())
        pf=json.loads(args.preflight.read_text()) if args.preflight else None
        release=json.loads(args.release.read_text()) if args.release else None
        result=evaluate(args.phase,memory,pf,release)
    if args.check_lock:
        if args.phase=='engineering':raise ValueError('engineering has no locked cohort')
        locked=json.loads(args.check_lock.read_text())
        if compact(result)!=locked:raise AssertionError('AMM-05 source-locked result mismatch')
        print('AMM-05',args.phase,'full raw replay PASS')
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x') as stream:
            json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n')
    print(json.dumps({'experiment':NAME,'phase':args.phase,
                      'n_roots':result.get('n_independent_roots'),
                      'totals':result.get('totals'),
                      'memory_hash':result.get('sha256')}))


if __name__=='__main__':main()
