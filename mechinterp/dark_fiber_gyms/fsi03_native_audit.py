"""FSI-03: retrospective root-aware audit of actual trained TRM Gramian receipts.

Reads published 120-board native LOOPSCOPE-04 receipts; no GPU, no access
or modification of ARG-03's sealed held-out 168-case test. All endpoints are
exploratory and the existing publication claims remain unchanged.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REL='supplements/loopscope04_arg03_receipts/loopscope04/'
FILES={
    'roster':REL+'rep_eval_strata.json',
    'gaoxin':REL+'gaoxin492_rep_gramian_t4_p8_k4_h12.json',
    'yagiz':REL+'yagizdevre_rep_gramian_t4_p8_k4_h12.json'
}
ANGLES=('0.04','0.15','0.35','0.75')
METRICS=('exact_solve_at_h12','board_any_flip')


def load_fsi():
    filename=os.environ.get('EVALMATH_FSI_FILE')
    if not filename or not Path(filename).exists():
        raise RuntimeError('Set EVALMATH_FSI_FILE to pinned FSI-01 source')
    spec=importlib.util.spec_from_file_location('bpd_fsi03_module',filename)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def case_value(direction,metric):
    if metric=='board_any_flip':return int(direction['board_flips_vs_clean_at_h12']>0)
    return int(direction['exact_solve_at_h12'])


def compute_pair(a,b,fsi):
    res=fsi.exact_mcnemar(a,b)
    return {'wins':res['candidate_only'],'losses':res['baseline_only'],
        'ties':res['ties'],'effect_exact':str((res['candidate_only']-res['baseline_only'])),
        'effect_pp':100*(res['candidate_only']-res['baseline_only'])/len(a),
        'p_exact':str(res['p_exact']),'p_float':float(res['p_exact'])}


def audit(root=ROOT):
    files={name:(root/path).read_bytes() for name,path in FILES.items()}
    roster=json.loads(files['roster'])
    if roster['n_eval']!=120 or len(roster['rows'])!=120:
        raise ValueError('representative board roster not 120')
    ids=[x['case_id'] for x in roster['rows']]
    if len(set(ids))!=120:raise ValueError('duplicate source boards')
    fsi=load_fsi()
    comparisons=[];summaries={}
    for name in ('gaoxin','yagiz'):
        doc=json.loads(files[name]);rows=doc['rows']
        if doc['mode']!='gramian' or len(rows)!=120:
            raise ValueError(f'{name} unexpected receipt type/count')
        byid={r['case_id']:r for r in rows}
        if len(byid)!=120 or set(byid)!=set(ids):raise ValueError('root roster mismatch')
        summaries[name]={'checkpoint_sha256':doc['checkpoint_sha256'],'n_boards':120,'angles':{}}
        for ang in ANGLES:
            ref={x:{} for x in ('high_eig','low_eig','random_dir')}
            for case_id in ids:
                directions=byid[case_id]['causal_direction_tests']['angle_'+ang]['directions']
                for key in ref:ref[key][case_id]=directions[key]
            means={key:sum(ref[key][cid]['lag12_readout_l2'] for cid in ids)/120 for key in ref}
            high_minus_random=sum(ref['high_eig'][cid]['lag12_readout_l2']-
                                  ref['random_dir'][cid]['lag12_readout_l2'] for cid in ids)/120
            effects={}
            for metric in METRICS:
                a=[case_value(ref['high_eig'][cid],metric) for cid in ids]
                b=[case_value(ref['random_dir'][cid],metric) for cid in ids]
                row=compute_pair(a,b,fsi)
                row.update({'checkpoint':name,'angle':ang,'metric':metric,
                            'candidate_total':sum(a),'random_total':sum(b)})
                comparisons.append(row)
                effects[metric]={k:row[k] for k in ('wins','losses','effect_pp','p_exact','p_float')}
            summaries[name]['angles'][ang]={
              'mean_lag12_l2':means,'high_minus_random_mean_lag12':high_minus_random,
              'high_gt_random_lag12':sum(ref['high_eig'][cid]['lag12_readout_l2']>
                ref['random_dir'][cid]['lag12_readout_l2'] for cid in ids),
              'effects':effects,
            }
    adjusted=fsi.holm_adjust([fsi.binomial_two_sided(row['wins'],row['wins']+row['losses']) for row in comparisons])
    for row,adj in zip(comparisons,adjusted):
        row['holm_exact']=str(adj);row['holm_float']=float(adj)
    min_p=min(row['p_float'] for row in comparisons)
    maximum_effect=max(row['effect_pp'] for row in comparisons)
    return {'schema':'BPD-FSI03/native120-retrospective-v1',
      'evidence_level':'published native trained TRM checkpoint causal receipt, retrospective; not new model evaluation',
      'independent_units_declared':120,'repeat_checkpoints_are_not_extra_roots':True,
      'scope':'two real 7M TRM checkpoints, 120 matched unique source boards, 4 fixed historical angles',
      'source_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
      'recorded_checkpoint_contrasts':summaries,
      'binary_comparisons':comparisons,
      'family':{'size':len(comparisons),'nominal_p_below_005':sum(row['p_float']<.05 for row in comparisons),
                'holm_p_below_005':sum(row['holm_float']<.05 for row in comparisons),
                'smallest_unadjusted_p':min_p,
                'largest_positive_effect_pp':maximum_effect,
                'mechanistic_effect_floor_pp':10.0,
                'any_beat_10pp':any(row['effect_pp']>10 for row in comparisons)},
      'interpretation':'The chosen high-Gramian direction can increase continuous delayed readout response relative to equal-angle random directions without a corresponding 10 percentage-point task improvement. Existing cases are exploratory, not sealed or pre-registered for this audit. Population inference assumes independent source boards, not two times 120 independent checkpoint rows.',
      'heldout_seal':'ARG-03-N 168-case test not read; no new network weights accessed'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    r=audit(args.root)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print('BPD-FSI03',r['independent_units_declared'],'boards',len(r['binary_comparisons']),'historical paired comparisons;',r['family'])
if __name__=='__main__':main()
