"""AMM-03 prospective model-conditioned memory, strictly phase-gated.

Run from the BPD repository root. The registration in PROTOCOL.md precedes this
source; confirmatory roots require a successful independent EvalMath preflight.
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

from . import models as m

NAME=m.NAME
SCHEMA=m.SCHEMA
PAIRS=m.PAIRS
CAL_PAIRS=m.CAL_PAIRS
TRAIN_PAIRS=m.TRAIN_PAIRS
ARMS=m.ARMS
COMPARISONS=m.COMPARISONS
ROOT=Path(__file__).resolve().parent
PRIOR_PSEUDOCASES=8
RIDGE_LAMBDA=.01
BOOTSTRAP_REPS=2000


def source_sha256() -> dict[str,str]:
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('models.py','experiment.py')}


def trained_model_root(phase:str,index:int):
    model,factors,diagnostics=m.fit(phase,index)
    geometry=m.geometry(factors,m.family_for(index))
    return model,factors,diagnostics,geometry


def training_memory():
    """Cross-root ridge memory; training phase exclusively."""
    rows=[]
    for i in range(m.PHASE_N['training']):
        _,factors,diag,geo=trained_model_root('training',i)
        assay=m.assay('training',i,factors,'training_pairs',TRAIN_PAIRS)
        rows.append({'root':m.root_key('training',i),'family':m.family_for(i),
                     'geometry':geo,'teacher_factor_diagnostics':diag,
                     'training_assay':assay})
    # Every model, rather than every repeated donor pair, has two equal-weight rows.
    x=[];y=[]
    for r in rows:
        for c in (0,1):
            x.append(r['geometry']['features'][c])
            y.append(r['training_assay']['candidate_hits'][f'component_{c}']/TRAIN_PAIRS)
    X=torch.tensor(x,dtype=torch.float64)
    Y=torch.tensor(y,dtype=torch.float64)
    penalty=torch.diag(torch.tensor([0.,1.,1.,1.,1.,1.,1.],dtype=torch.float64))*RIDGE_LAMBDA
    w=torch.linalg.solve(X.T@X+penalty,X.T@Y)
    hits={str(c):sum(r['training_assay']['candidate_hits'][f'component_{c}'] for r in rows) for c in (0,1)}
    out={'schema':SCHEMA+'/training_memory','experiment':NAME,'phase':'training',
         'source_sha256':source_sha256(),'independent_roots':64,
         'training_pairs_per_root':TRAIN_PAIRS,'training_labels_per_root':2*TRAIN_PAIRS,
         'ridge_lambda':RIDGE_LAMBDA,'ridge_coefficients':w.tolist(),
         'source_counts':{f:sum(r['family']==f for r in rows) for f in m.FAMILIES},
         'component_training_hits':hits,
         'pooled_action':0 if hits['0']>=hits['1'] else 1,
         'rows':rows}
    out['sha256']=m.digest(out)
    return out


def validate_memory(memory:dict):
    if not isinstance(memory,dict) or memory.get('schema')!=SCHEMA+'/training_memory':
        raise ValueError('frozen AMM-03 training memory required')
    body={k:v for k,v in memory.items() if k!='sha256'}
    if memory.get('sha256') != m.digest(body):
        raise ValueError('training memory hash invalid')
    if memory.get('source_sha256') != source_sha256():
        raise ValueError('training memory source mismatch')
    if memory.get('independent_roots')!=64 or len(memory.get('rows',[]))!=64:
        raise ValueError('training root count invalid')
    if {x.get('root') for x in memory['rows']} != {m.root_key('training',i) for i in range(64)}:
        raise ValueError('training roots reused or missing')
    if len(memory.get('ridge_coefficients',[]))!=7 or any(not math.isfinite(z) for z in memory['ridge_coefficients']):
        raise ValueError('invalid ridge coefficients')
    if memory.get('ridge_lambda') != RIDGE_LAMBDA or memory.get('pooled_action') not in (0,1):
        raise ValueError('policy settings mismatch')
    return True


def ridge_predictions(geometry:dict,memory:dict) -> list[float]:
    w=memory['ridge_coefficients']
    return [max(0.,min(1.,sum(u*v for u,v in zip(feature,w)))) for feature in geometry['features']]


def choose_actions(root:str,geometry:dict,calibration:dict,memory:dict) -> dict[str,int]:
    priors=ridge_predictions(geometry,memory)
    hits=[calibration['candidate_hits'][f'component_{c}'] for c in (0,1)]
    post=[(PRIOR_PSEUDOCASES*priors[c]+hits[c])/(PRIOR_PSEUDOCASES+CAL_PAIRS) for c in (0,1)]
    chooser=lambda values: 0 if values[0]>=values[1] else 1
    return {
        'memory_calibrated':chooser(post),
        'local_only':chooser(hits),
        'memory_only':chooser(priors),
        'pooled':memory['pooled_action'],
        'geometry_rule':chooser(geometry['signed_gains']),
        'random_component':m.seed(root,'random_component')%2,
    }


def score_model(phase:str,index:int,memory:dict) -> dict[str,Any]:
    """New root may only be called after top-level admission."""
    _,factors,metrics,geo=trained_model_root(phase,index)
    root=m.root_key(phase,index)
    cal=m.assay(phase,index,factors,'local_calibration',CAL_PAIRS)
    decisions=choose_actions(root,geo,cal,memory)
    scored=m.assay(phase,index,factors,'heldout_test',PAIRS)
    bits={}
    for policy,choice in decisions.items():
        bits[policy]=scored['candidate_bits'][f'component_{choice}']
    for policy in ('random_dark','random_dark_b','sham','full_donor'):
        bits[policy]=scored['candidate_bits'][policy]
    counts={name:bits[name].count('1') for name in ARMS}
    return {'root':root,'family':m.family_for(index),'geometry':geo,
            'teacher_factor_diagnostics':metrics,
            'calibration':cal,'policy_actions':decisions,
            'memory_prior_rates':ridge_predictions(geo,memory),
            'heldout':scored,'policy_bits':bits,'policy_hits':counts}


def admission(phase:str,memory:dict|None,preflight:dict|None):
    """Hard stop before ANY phase-specific model fit or donor generation."""
    if phase not in ('development','confirmation'):
        raise ValueError('unknown evaluation phase')
    validate_memory(memory)
    if phase=='confirmation':
        pf=preflight
        if not isinstance(pf,dict) or (
            pf.get('decision') != 'READY_FOR_CONFIRMATION'
            or pf.get('experiment_id')!=NAME
            or pf.get('planned_confirmation_worlds') != m.PHASE_N['confirmation']
            or pf.get('planned_comparisons') != len(COMPARISONS)
            or pf.get('pilot_worlds') != m.PHASE_N['development']
            or pf.get('upper_sham_quantile') is None
            or bool(pf.get('reasons'))
        ):
            raise ValueError('independent strict EvalMath preflight required before confirmation')


def evaluate(phase:str,memory:dict|None=None,preflight:dict|None=None) -> dict[str,Any]:
    admission(phase,memory,preflight)
    assert memory is not None
    rows=[score_model(phase,i,memory) for i in range(m.PHASE_N[phase])]
    out={'schema':SCHEMA+'/cohort','experiment':NAME,'phase':phase,
         'source_sha256':source_sha256(),'training_memory_sha256':memory['sha256'],
         'n_independent_roots':len(rows),'local_calibration_pairs_per_root':CAL_PAIRS,
         'labeled_local_calibration_interventions_per_root':2*CAL_PAIRS,
         'n_test_pairs_per_root':PAIRS,'test_cases_total':len(rows)*PAIRS,
         'totals':{a:sum(r['policy_hits'][a] for r in rows) for a in ARMS},
         'rows':rows}
    if phase=='development':
        out['preflight_input']=preflight_input(out)
    else:
        out['preflight_sha256']=m.digest(preflight)
        out['analysis']=analyze(out,preflight)
    return out


def preflight_input(run:dict) -> dict:
    if run.get('phase')!='development' or run.get('n_independent_roots')!=64:
        raise ValueError('requires exactly 64 independent development roots')
    entries=[]
    for r in run['rows']:
        c=r['policy_hits']
        entries.append({'unit_id':r['root'],
                        'paired_effect':f"{c['memory_calibrated']-c['local_only']}/{PAIRS}",
                        'sham_effects':[f"{c['random_dark']-c['random_dark_b']}/{PAIRS}"]})
    return {'schema_version':1,'experiment_id':NAME,
            'independent_unit':'one fresh model root including teacher and trained BPD factorization',
            'alpha':'0.05','scientific_min_effect':'0.10','target_effect':'0.20',
            'measurement_resolution':f'1/{PAIRS}',
            'sham_quantile':'0.95','sham_confidence':'0.95',
            'planned_comparisons':len(COMPARISONS),
            'confirm_units_planned':m.PHASE_N['confirmation'],
            'pilot_from_development_only':True,'heldout_sealed':True,
            'units_independent':True,'pilot':entries}


def root_sign_p(diffs:list[int]) -> float:
    k=sum(x>0 for x in diffs);n=sum(x!=0 for x in diffs)
    if n==0:return 1.
    return sum(math.comb(n,j) for j in range(k,n+1))/(2**n)


def holm(ps:dict[str,float]) -> dict[str,float]:
    adjusted={};floor=0.
    for i,key in enumerate(sorted(ps,key=lambda name:(ps[name],name))):
        floor=max(floor,min(1.,ps[key]*(len(ps)-i)))
        adjusted[key]=floor
    return adjusted


def bootstrap(diffs:list[int],name:str) -> list[float]:
    rng=random.Random(m.seed('bootstrap',name))
    n=len(diffs)
    arr=sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/(n*PAIRS) for _ in range(BOOTSTRAP_REPS))
    return [arr[50],arr[1949]]


def analyze(run:dict,pf:dict) -> dict:
    if run['phase']!='confirmation':raise ValueError('confirmation only')
    differences={name:[r['policy_hits']['memory_calibrated']-r['policy_hits'][control] for r in run['rows']]
                 for name,control in COMPARISONS.items()}
    ps={name:root_sign_p(x) for name,x in differences.items()}
    ap=holm(ps)
    effects={name:{'mean_root_effect':sum(x)/(len(x)*PAIRS),
                   'root_bootstrap_95_ci':bootstrap(x,name),
                   'positive_roots':sum(t>0 for t in x),
                   'negative_roots':sum(t<0 for t in x),
                   'tied_roots':sum(t==0 for t in x),
                   'exact_one_sided_sign_p':ps[name],
                   'holm_adjusted_p':ap[name]}
             for name,x in differences.items()}
    floor=max(.10,pf['mechanistic_variation_floor'])
    equal=max(r['heldout']['dose_max_abs_error'] for r in run['rows'])
    visible=max(r['heldout']['current_readout_max_abs_error'] for r in run['rows'])
    p=effects['memory_vs_local']
    return {'schema':SCHEMA+'/analysis','evidence':'synthetic trained rank-two tied linear BPD',
            'comparisons':effects,'mechanistic_floor':floor,
            'max_active_dose_error':equal,'max_current_readout_displacement':visible,
            'registered_primary_gate':bool(p['root_bootstrap_95_ci'][0]>floor and p['holm_adjusted_p']<.05
                                            and equal<=1e-5 and visible<=1e-5)}


def compact(run:dict) -> dict:
    """Small stable lock; full scored root traces reproduce from source and seed."""
    if run.get('phase')=='training':
        return {'experiment':NAME,'phase':'training','training_memory_sha256':run['sha256'],
                'pooled_action':run['pooled_action'],'ridge_coefficients':run['ridge_coefficients'],
                'source_sha256':run['source_sha256']}
    out={'schema':SCHEMA+'/lock','experiment':NAME,'phase':run['phase'],
         'source_sha256':run['source_sha256'],
         'training_memory_sha256':run['training_memory_sha256'],
         'independent_roots':run['n_independent_roots'],'n_cases':run['test_cases_total'],
         'totals':run['totals'],'full_raw_receipt_sha256':m.digest(run)}
    if run['phase']=='development':
        out['preflight_input_sha256']=m.digest(run['preflight_input'])
    else:
        out['preflight_sha256']=run['preflight_sha256']
        out['analysis']=run['analysis']
    return out


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('engineering','training','development','confirmation'),required=True)
    p.add_argument('--memory',type=Path)
    p.add_argument('--preflight',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--check-lock',type=Path)
    a=p.parse_args()
    if a.phase=='training':run=training_memory()
    elif a.phase=='engineering':
        run={'phase':'engineering','rows':[]}
        for i in range(2):
            _,f,metrics,geo=trained_model_root('engineering',i)
            row=m.assay('engineering',i,f,'engineering_test',PAIRS)
            run['rows'].append({'root':m.root_key('engineering',i),'geometry':geo,
                                'teacher_factor_diagnostics':metrics,'assay':row,
                                'gauge':m.gauge_check(f,m.family_for(i))})
    else:
        if a.memory is None:raise ValueError('requires --memory')
        memory=json.loads(a.memory.read_text())
        pf=json.loads(a.preflight.read_text()) if a.preflight else None
        run=evaluate(a.phase,memory,pf)
    if a.check_lock:
        if a.phase=='engineering':raise ValueError('engineering is not result-locked')
        locked=json.loads(a.check_lock.read_text())
        if compact(run)!=locked:raise AssertionError('AMM-03 source-locked replay mismatch')
        print(NAME,a.phase,'full raw digest replay PASS')
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open('x') as out:
            json.dump(run,out,sort_keys=True,indent=2,allow_nan=False);out.write('\n')
    summary={'experiment':NAME,'phase':a.phase}
    if 'totals' in run:
        summary.update({'totals':run['totals'],'roots':run['n_independent_roots']})
    if 'pooled_action' in run:
        summary.update({'pooled_action':run['pooled_action'],'training_memory_sha256':run['sha256']})
    print(json.dumps(summary,sort_keys=True))


if __name__=='__main__':cli()
