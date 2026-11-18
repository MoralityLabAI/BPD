"""EvalMath-compatible CQ08 two-stage power gate with separate sham worlds.

For calibration, use 8 development effect roots plus 59 independent sham roots.
The upper 95/95 nonparametric tolerance bound is the smallest order statistic
whose binomial coverage is at least .95; for 59 roots this is the sample max.
The normal power estimate is *planning only*; the cohort stays sealed until an
explicitly committed confirmation-release lock is approved and versioned.
"""
from __future__ import annotations
import argparse,json,math
from fractions import Fraction
from math import comb
from pathlib import Path
from statistics import NormalDist,stdev

Q=Fraction(95,100)

def tolerance(shams,quantile=Q,confidence=Q):
    n=len(shams)
    for k in range(1,n+1):
        cdf=sum((Fraction(comb(n,j))*quantile**j*(1-quantile)**(n-j) for j in range(k)),Fraction(0))
        if cdf>=confidence:return sorted(shams)[k-1],k
    return None,None

def preflight(development,calibration=None,confirmation_roots=32):
    observed=[float(v) for v in development['summary']['primary_effects_by_root']]
    assert len(observed)==8
    dev_sham=[float(v) for v in development['summary']['random_random_abs_shams_by_root']]
    independent=[] if calibration is None else [float(v) for v in calibration['summary']['random_random_abs_shams_by_root']]
    assert calibration is None or len(independent)==59
    floor,rank=tolerance(independent)
    mech_floor=max(.15,1/128,floor or 0.)
    sd=stdev(observed)
    z=NormalDist().inv_cdf(1-.05/(2*2))+NormalDist().inv_cdf(.8)
    excess=.30-mech_floor
    required=max(2,math.ceil((z*sd/excess)**2)) if excess>0 else None
    blockers=[]
    if len(independent)<59 or floor is None:blockers.append('sham_quantile_uncalibrated')
    if excess<=0:blockers.append('target_below_mechanistic_variation_floor')
    if required is None or required>confirmation_roots:blockers.append('confirmation_underpowered')
    # A separate explicit unlock file must be committed after registration.
    return {'schema':'EvalMath/CQ08-separate-sham-preflight/v1',
      'decision':'REDESIGN_OR_PILOT_ONLY' if blockers else 'READY_FOR_CONFIRMATION',
      'reasons':blockers,'independent_unit':'one new physical recurrence, BPD fit, and nuisance world',
      'development_roots':len(observed),'sham_calibration_roots':len(independent),
      'confirmation_roots_planned':confirmation_roots,
      'mean_development_effect':sum(observed)/len(observed),'development_root_sd':sd,
      'observed_worst_development_sham':max(dev_sham),
      'calibrated_95th_percentile_95pct_confidence_bound':floor,
      'calibration_order_statistic_rank':rank,
      'minimum_scientific_effect':.15,
      'effective_variation_floor':mech_floor,
      'target_effect':.30,'planned_comparisons':2,'alpha':.05,
      'approx_roots_80pct_power':required,
      'approx_resolvable_effect':mech_floor+z*sd/math.sqrt(confirmation_roots),
      'source':'8 exposed development roots and distinct 59-root random-vs-random sham population',
      'limits':'Normal approximation is planning only; the sealed confirmation remains unopened until an explicit post-preflight release.'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--sham',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--strict',action='store_true')
    a=p.parse_args();dev=json.loads(a.development.read_text())
    sham=json.loads(a.sham.read_text()) if a.sham else None
    outcome=preflight(dev,sham)
    s=json.dumps(outcome,indent=2,sort_keys=True)+'\n'
    if a.out:
        if a.out.exists():raise FileExistsError(a.out)
        a.out.write_text(s)
    else:print(s,end='')
    return 2 if a.strict and outcome['decision']!='READY_FOR_CONFIRMATION' else 0
if __name__=='__main__':raise SystemExit(main())
