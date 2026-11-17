"""BPD-CQ-01 deterministic exact-rational source audit, no model or sealed test."""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from random import Random

from .quotient import (matrix,eye,mul,rank,all_masks,invariant_masks,first_detected,
                       invariant_to_horizon,response_matrices,zero,closed_under_intersection,
                       delayed_shear,nonlattice_four,capacity_witness,powers_times_readout)

ROOT=Path(__file__).resolve().parent
SEED=2026111801
TWISTS=(-7,-5,-3,-2,-1,1,2,3,5,7)


def sha(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def random_frame(rng:Random):
    n=4
    r=eye(n)
    for _ in range(4):
        i,j=rng.sample(range(n),2)
        coeff=rng.choice(TWISTS)
        e=[[int(a==b)+(coeff if (a,b)==(i,j) else 0) for b in range(n)] for a in range(n)]
        r=mul(r,matrix(e))
    perm=rng.sample(range(n),n)
    t=matrix([[int(j==perm[i]) for j in range(n)] for i in range(n)])
    k=rng.randrange(n)
    c=matrix([[int(j==k) for j in range(n)]])
    return dict(r=r,m=eye(n),u=eye(n),t=t,c=c,s=eye(n))


def run(*, engineering=False)->dict:
    rng=Random(SEED if not engineering else 246813579)
    nworlds=8 if engineering else 384
    worlds=[]
    for idx in range(nworlds):
        if engineering:
            cohort=('delayed','nonlattice','random')[idx%3]
        else:
            cohort='delayed' if idx<128 else ('nonlattice' if idx<256 else 'random')
        twist=rng.choice(TWISTS)
        case=delayed_shear(twist) if cohort=='delayed' else (nonlattice_four(twist) if cohort=='nonlattice' else random_frame(rng))
        d=len(case['r'])
        masks=[set(invariant_masks(**case,horizon=k)) for k in range(d+2)]
        if any(not masks[k+1]<=masks[k] for k in range(d+1)):
            raise AssertionError('observable mask set increased with horizon')
        if masks[d-1]!=masks[d+1]:
            raise AssertionError('Cayley-Hamilton stabilization violated')
        for bits in all_masks(d):
            all_responses=response_matrices(**case,bits=bits,horizon=d-1)
            for horizon in range(d):
                expected=all(zero(a) for a in all_responses[:horizon+1])
                if invariant_to_horizon(**case,bits=bits,horizon=horizon)!=expected:
                    raise AssertionError('Gramian/direct-response disagreement')
        detected=Counter('never' if first_detected(**case,bits=bits) is None else str(first_detected(**case,bits=bits)) for bits in all_masks(d))
        if cohort=='delayed' and (len(masks[0]),len(masks[1]))!=(4,2):
            raise AssertionError('2D delayed source theorem failed')
        if cohort=='nonlattice' and (len(masks[0]),len(masks[1]),closed_under_intersection(masks[1]))!=(16,6,False):
            raise AssertionError('4D observational non-Boolean theorem failed')
        obs=[*powers_times_readout(case['c'],case['t'],case['u'],d-1)]
        rank_k=[rank(matrix([row for z in obs[:k+1] for row in z])) for k in range(d)]
        worlds.append(dict(index=idx,cohort=cohort,twist=twist if cohort!='random' else None,
                           dimension=d,invariant_counts=[len(masks[k]) for k in range(d)],
                           delayed_detection=dict(sorted(detected.items())),
                           horizon1_mask_boolean=closed_under_intersection(masks[1]),
                           observable_rank=rank_k,
                           saturation_at_dimension_minus_one=masks[d-1]==masks[-1]))
    counts={}
    for cohort in ('delayed','nonlattice','random'):
        group=[w for w in worlds if w['cohort']==cohort]
        if not group:continue
        counts[cohort]={'n':len(group),'invariant_counts_by_horizon_pattern':dict(sorted(Counter(','.join(map(str,w['invariant_counts'])) for w in group).items())),
                        'non_boolean_at_horizon1':sum(not w['horizon1_mask_boolean'] for w in group),
                        'saturation_passed':sum(w['saturation_at_dimension_minus_one'] for w in group),
                        'observability_rank_at_last_horizon':dict(sorted(Counter(str(w['observable_rank'][-1]) for w in group).items()))}
    return {'schema':'BPD-CQ-01/exact-rational-reference/v1','evidence':'source_only_mathematical_fixtures',
            'prospective_extended_seed':SEED,'engineering_run':engineering,'n_worlds':len(worlds),
            'base_linear_observable_horizon':'0_to_state_dimension_minus_one',
            'no_sealed_arg03_test':True,'neural_models_evaluated':0,
            'source_sha256':{p:sha(ROOT/p) for p in ('quotient.py','reference.py','PROTOCOL.md')},
            'capacity':capacity_witness(),'cohorts':counts,'worlds':worlds}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--engineering',action='store_true')
    args=ap.parse_args()
    res=run(engineering=args.engineering)
    text=json.dumps(res,sort_keys=True,indent=2)+'\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding='utf-8')!=text:
            raise SystemExit('source-locked BPD-CQ-01 reference missing or changed')
        print('BPD-CQ-01 verified:',res['n_worlds'],'exact rational worlds')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x',encoding='utf-8',newline='\n') as handle:
            handle.write(text)
        print('BPD-CQ-01 generated',res['n_worlds'],'worlds:',args.out)

if __name__=='__main__': main()
