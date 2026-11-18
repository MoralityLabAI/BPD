"""Native BPD masked execution audit of released CQ-08 confirmation roots.

Recompute each model once on hosted CPU, score the exact same learned state
with standalone and native BPD wrappers, and compare to the immutable local
confirmation receipt. Freeze every mismatch as a source-compatible failure.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json
from pathlib import Path
import torch
from .experiment import (ROOT,CONFIRMATION,configure,train_teacher,train_factors,
    canonicalize,calibration_block,NativeModel,score_donors,paired_z_task,native_parity)
from .confirmation_release import check_release,infer
from .reference import git_blob

RESULTS=ROOT/'results'
LOCK=RESULTS/'CONFIRMATION_LOCK.json'
ARCHIVE=RESULTS/'cq08_confirmation_independent.json.gz.b64'

def load_original():
    release=check_release()
    lock=json.loads(LOCK.read_text())
    if lock['release_commit']!='0be2c8bbe10335f9597c31b12d688bca522f9117':raise AssertionError('release commit mismatch')
    if lock['original_source_experiment_blob']!=git_blob(ROOT/'experiment.py'):raise AssertionError('source hash mismatch')
    raw=ARCHIVE.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==lock['gzip_base64_sha256']
    decoded=gzip.decompress(base64.b64decode(raw))
    assert hashlib.sha256(decoded).hexdigest()==lock['raw_json_sha256']
    original=json.loads(decoded)
    assert original['run']['seeds']==lock['roster']==list(CONFIRMATION)
    assert original['inference']==infer(original['run'],release)
    return original

def audit(out: Path):
    configure()
    original=load_original()
    hosted=[]
    for row in original['run']['rows']:
        seed=row['seed']
        torch.manual_seed(seed)
        teacher=train_teacher(seed);f=train_factors(seed,teacher);canonicalize(f)
        sel,_=calibration_block(seed,teacher,f)
        if sel!=row['calibration_selected']:
            raise AssertionError(f'calibration component selection changed: root {seed}')
        native=NativeModel(teacher,f)
        p=native_parity(seed,teacher,f,native)
        if p>1e-5:raise AssertionError(f'native physical output parity failed {seed}: {p}')
        a=score_donors(seed,teacher,f,sel,'independent',None,True)
        b=score_donors(seed,teacher,f,sel,'native',native,True)
        ct=paired_z_task(seed,teacher,f)
        nt=paired_z_task(seed,teacher,f,'native',native)
        if ct!=nt:raise AssertionError(f'task backend parity failed {seed}')
        names=['eligible_match','damage_counts_itt','selected_off_correct','random_off_1_correct','random_off_2_correct']
        for k in names:
            if a[k]!=b[k]:raise AssertionError(f'hosted backend count mismatch {seed}/{k}')
        for k in ('match','damage_selected','damage_random1','damage_random2','donor_full_correct'):
            if a['pair_data'][k]!=b['pair_data'][k]:raise AssertionError(f'hosted native pair mismatch {seed}/{k}')
        diff={}
        for k in names:
            if a[k]!=row['donor'][k]:diff[k]={'frozen':row['donor'][k],'hosted':a[k]}
        for k in ('match','damage_selected','damage_random1','damage_random2'):
            x,y=a['pair_data'][k],row['donor']['pair_data'][k]
            if x!=y:diff[k]={'different_pair_indices':[i for i,(v,w) in enumerate(zip(x,y)) if v!=w]}
        hosted.append({'seed':seed,'hosted_backend_parity':True,'native_parity_max':p,
                       'original_frozen_count_diffs':diff,'original_task_equal':ct==row['task'],
                       'hosted_counts':a['damage_counts_itt'],'matched':a['eligible_match']})
        print('NATIVE_ROOT',seed,'PASS','frozen_differences',len(diff),flush=True)
    result={'schema':'bpd.cq08.confirmation.hosted_native_audit.v1',
            'roots':32,'backend_native_parity':'PASS',
            'frozen_cpu_exact_count_parity':'PASS' if all(not r['original_frozen_count_diffs'] for r in hosted) else 'FAIL',
            'n_roots_with_cross_host_drift':sum(bool(r['original_frozen_count_diffs']) for r in hosted),
            'rows':hosted}
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.out)
if __name__=='__main__':main()
