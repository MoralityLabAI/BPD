"""CQ-08 preregistered independent-root confirmation evaluator.

Run only after the separate GitHub release manifest is committed. Original
physical/BPD experiment, training, tests and source archives are immutable.
"""
from __future__ import annotations
import argparse, base64, gzip, hashlib, json, math
from pathlib import Path
import numpy as np
from .experiment import ROOT, CONFIRMATION, DONOR_CASES, cohort, summarize
from .reference import registration, load_frozen, sha, git_blob
from .power import preflight

UNLOCK=ROOT/'CONFIRM_UNLOCK.json'
SHAMFILE=ROOT/'results'/'cq08_sham_independent.json.gz.b64'
SHAMLOCK=ROOT/'results'/'SHAM_LOCK.json'
PREFLIGHT=ROOT/'results'/'preflight_after_59_shams.json'
FIELDS=('selected_off','random_off_1','random_off_2')

def check_release():
    lock=registration()
    if not UNLOCK.exists():raise RuntimeError('Prospective confirmation remains sealed')
    release=json.loads(UNLOCK.read_text())
    assert release['schema']=='bpd.cq08.confirmation.release.v1'
    assert release['registration_git_blob_sha']==git_blob(ROOT/'REGISTRATION.json')
    assert release['experiment_source_git_blob']==git_blob(ROOT/'experiment.py')
    assert release['development_receipt_sha256']==lock['development_receipt_blob_sha256']
    assert release['confirmed_seed_roster']==lock['confirmation_roots']==list(CONFIRMATION)
    assert release['scientific_min_effect']==.15 and release['planned_comparisons']==2
    assert release['cases_per_seed']==DONOR_CASES==128
    assert release['sham_archive_sha256']==sha(SHAMFILE)
    assert release['preflight_report_sha256']==sha(PREFLIGHT)
    shamlock=json.loads(SHAMLOCK.read_text())
    assert shamlock['confirmed_release']=='NOT RELEASED'
    assert shamlock['sham_roots']==59 and shamlock['sham_seed_roster']==lock['sham_roots']
    sham=json.loads(gzip.decompress(base64.b64decode(SHAMFILE.read_text())))
    assert sham['run']['seeds']==lock['sham_roots']
    dev=load_frozen()
    outcome=preflight(dev,sham)
    assert outcome==json.loads(PREFLIGHT.read_text())
    assert outcome['decision']=='READY_FOR_CONFIRMATION'
    assert outcome['sham_calibration_roots']==59 and outcome['confirmation_roots_planned']==32
    assert outcome['calibrated_95th_percentile_95pct_confidence_bound']<=.15
    return release

def binom_sign_p(n,k):
    return float(sum(math.comb(n,j) for j in range(k,n+1))/2**n)

def infer(run,release):
    assert run['phase']=='confirmation'
    assert run['seeds']==list(CONFIRMATION)
    roots=run['rows'];assert len(roots)==32
    assert all(row['donor']['pairs']==128 for row in roots)
    effects=[]
    for rand in ('random_off_1','random_off_2'):
        effects.append(np.array([(r['donor']['damage_counts_itt']['selected_off']-
                    r['donor']['damage_counts_itt'][rand])/DONOR_CASES for r in roots],dtype=np.float64))
    counts=[int((a>.15).sum()) for a in effects]
    p=[binom_sign_p(32,k) for k in counts]
    ordered=sorted(range(2),key=lambda i:p[i]);holm=[0.,0.];adj=0.
    for rank,i in enumerate(ordered):
        adj=max(adj,min(1.,(2-rank)*p[i]));holm[i]=adj
    rng=np.random.default_rng(20261118)
    # Whole independent model roots are bootstrap sampling units.
    sampled=rng.integers(0,32,size=(20000,32))
    ci=[np.quantile(x[sampled].mean(axis=1),[.025,.975]).tolist() for x in effects]
    primary_gate=bool(effects[0].mean()>=.15 and holm[0]<.05)
    tech={'complete_roots':len(roots),
          'feasibility_roots':sum(r['donor']['eligible_match']>=.8*128 for r in roots),
          'box_violation_max':max(r['donor']['max_state_box_violation'] for r in roots),
          'source_replay_error_max':max(r['donor']['max_source_replay_error'] for r in roots),
          'full_donor_correct':sum(r['donor']['full_donor_correct'] for r in roots),
          'full_bpd_two_readout_correct':sum(r['task']['bpd_full_both_correct'] for r in roots),
          'teacher_two_readout_correct':sum(r['task']['teacher_both_correct'] for r in roots)}
    tech['engineering_gate']=bool(tech['feasibility_roots']==32 and tech['box_violation_max']==0
                and tech['source_replay_error_max']<=1e-6 and tech['full_donor_correct']==4096
                and tech['full_bpd_two_readout_correct']==8192 and tech['teacher_two_readout_correct']==8192)
    dose={k:{'per_root_mean':[r['donor']['ablation_write_norms'][k]['mean'] for r in roots],
         'mean_of_root_means':float(np.mean([r['donor']['ablation_write_norms'][k]['mean'] for r in roots]))}
          for k in FIELDS}
    return {'schema':'bpd.cq08.confirmation.inference.v1',
            'n_model_roots':32,'n_donor_cases':4096,'science_minimum':.15,
            'effect_means':[float(x.mean()) for x in effects],
            'root_effects':[x.tolist() for x in effects],
            'roots_above_min_effect':counts,
            'one_sided_exact_sign_p':p,'holm_corrected_p':holm,
            'cluster_bootstrap_95pct_intervals':ci,
            'primary_gate':primary_gate,'secondary_direction_pass':bool(holm[1]<.05),
            'engineering':tech,'ablation_physical_write_norms':dose,
            'provenance':{'registered_release_schema':release['schema'],
                'registration_commit':release['registration_commit'],
                'registered_sham_abs_root_threshold':release['calibrated_upper_sham_abs_root_effect'],
                'root_resample_seed':20261118,'bootstrap_resamples':20000},
            'scientific_boundary':'Equal ablation cardinality, unequal physical ablation doses; synthetic disconnected core/nuisance architecture.'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--run',action='store_true')
    p.add_argument('--verify',type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if sum([a.preflight,a.run,a.verify is not None])!=1:raise SystemExit('choose exactly one of --preflight, --run, --verify')
    release=check_release()
    if a.preflight:
        print(json.dumps({'release':'REGISTERED','n_roots':32,'decision':'READY_FOR_CONFIRMATION','physical_dose_match':False},indent=2));return
    if a.run:
        if a.out is None:raise SystemExit('--out is required to create a write-once confirmation receipt')
        if a.out.exists():raise FileExistsError(a.out)
        raw=cohort('confirmation','independent',CONFIRMATION,include_pairs=True)
        result={'run':raw,'summary':summarize(raw),'inference':infer(raw,release)}
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    else:
        result=json.loads(a.verify.read_text())
        assert result['inference']==infer(result['run'],release)
    print(json.dumps({'summary':result['summary'],'inference':{k:v for k,v in result['inference'].items() if k not in ('root_effects','ablation_physical_write_norms')}},indent=2))

if __name__=='__main__':main()
