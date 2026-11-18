"""Checkpoint-frozen replay of CQ-08 development roots, including native BPD masks.

Engineering audit only. The training source, source-locks, and original outcome
receipt remain unchanged. Trained weights are stored as no-pickle NPZ arrays;
this replay diagnoses floating-point optimizer drift between CPU hosts.
"""
from __future__ import annotations
import argparse,base64,hashlib,io,json
from pathlib import Path
import numpy as np
import torch
from .experiment import (ROOT,DEVELOPMENT,PhysicalTeacher,Factors,train_teacher,
                         train_factors,canonicalize,calibration_block,score_donors,
                         paired_z_task,NativeModel,native_parity,configure)
from .reference import load_frozen,registration

DATA=ROOT/'checkpoints'/'cq08_dev_models.npz.b64'
MANIFEST=ROOT/'checkpoints'/'MANIFEST.json'
FIELDS=('A','B','A_n','B_n','bias')

def as_record(seed,t,f,sel):
    d={f'{seed}/teacher/{p}':getattr(t,p).detach().numpy() for p in FIELDS}
    d[f'{seed}/teacher/head_weight']=t.head.weight.detach().numpy()
    d[f'{seed}/teacher/nuisance_target_map']=t.nuisance_target_map.detach().numpy()
    d[f'{seed}/factor/V']=f.V.detach().numpy()
    d[f'{seed}/factor/U']=f.U.detach().numpy()
    d[f'{seed}/selected']=np.asarray(sel,dtype=np.int64)
    return d

def unpack(d,seed):
    t=PhysicalTeacher(seed);f=Factors(seed)
    with torch.no_grad():
        for attr in FIELDS:getattr(t,attr).copy_(torch.from_numpy(d[f'{seed}/teacher/{attr}']))
        t.head.weight.copy_(torch.from_numpy(d[f'{seed}/teacher/head_weight']))
        t.nuisance_target_map.copy_(torch.from_numpy(d[f'{seed}/teacher/nuisance_target_map']))
        f.V.copy_(torch.from_numpy(d[f'{seed}/factor/V']))
        f.U.copy_(torch.from_numpy(d[f'{seed}/factor/U']))
    return t,f,[int(x) for x in d[f'{seed}/selected']]

def compare_one(seed,t,f,sel,original,with_native=True):
    with torch.no_grad():
        score=score_donors(seed,t,f,sel,include_pairs=True)
        task=paired_z_task(seed,t,f)
        d0=original['donor']
        for key in ('eligible_match','selected_off_correct','random_off_1_correct','random_off_2_correct',
                    'damage_counts_itt','full_donor_correct',
                    'matched_nuisance_recipient_label_correct','matched_nuisance_donor_label_correct'):
            if score[key]!=d0[key]:raise AssertionError(f'checkpoint mismatch seed {seed} / {key}: {score[key]} versus {d0[key]}')
        for key in ('match','damage_selected','damage_random1','damage_random2'):
            if score['pair_data'][key]!=d0['pair_data'][key]:raise AssertionError(f'checkpoint mismatch seed {seed} pair {key}')
        if task!=original['task']:raise AssertionError(f'checkpoint task mismatch seed {seed}')
        parity=None
        if with_native:
            n=NativeModel(t,f)
            parity=native_parity(seed,t,f,n)
            if parity>1e-5:raise AssertionError(f'native parity {seed}: {parity}')
            donor_native=score_donors(seed,t,f,sel,'native',n,True)
            for key in ('damage_counts_itt','eligible_match','selected_off_correct','random_off_1_correct','random_off_2_correct'):
                if donor_native[key]!=score[key]:raise AssertionError(f'native counts differ {seed}/{key}')
            for key in ('damage_selected','damage_random1','damage_random2'):
                if donor_native['pair_data'][key]!=score['pair_data'][key]:raise AssertionError(f'native pair differs {seed}/{key}')
    return {'seed':seed,'selected':sel,'matched':score['eligible_match'],
            'damage':score['damage_counts_itt'],'native_parity_max':parity}

def write():
    configure();frozen=load_frozen();arrays={};rows=[]
    for seed,reference_row in zip(DEVELOPMENT,frozen['run']['rows']):
        t=train_teacher(seed);f=train_factors(seed,t);canonicalize(f)
        sel,_=calibration_block(seed,t,f)
        if sel!=reference_row['calibration_selected']:raise AssertionError(f'calibration changed {seed}')
        rows.append(compare_one(seed,t,f,sel,reference_row,with_native=False))
        arrays.update(as_record(seed,t,f,sel))
    if DATA.exists():raise FileExistsError(DATA)
    DATA.parent.mkdir(parents=True,exist_ok=True)
    b=io.BytesIO();np.savez_compressed(b,**arrays)
    packed=base64.b64encode(b.getvalue()).decode()+'\n'
    DATA.write_text(packed)
    manifest={'schema':'bpd.cq08.frozen_development_checkpoint_bundle.v1',
              'roster':list(DEVELOPMENT),'arrays':len(arrays),
              'npz_sha256':hashlib.sha256(b.getvalue()).hexdigest(),
              'base64_text_sha256':hashlib.sha256(packed.encode()).hexdigest(),
              'source_reg_git_sha':'6960a1fc45d76a1e04866344a25a0e41c5c24489',
              'notice':'Independent engineering checkpoints, no post-selection source changes; original raw outcome counts unchanged.'}
    MANIFEST.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return rows

def check():
    configure();registration();frozen=load_frozen()
    manifest=json.loads(MANIFEST.read_text())
    encoded=DATA.read_text()
    if hashlib.sha256(encoded.encode()).hexdigest()!=manifest['base64_text_sha256']:raise AssertionError('checkpoint blob hash')
    raw=base64.b64decode(encoded)
    if hashlib.sha256(raw).hexdigest()!=manifest['npz_sha256']:raise AssertionError('checkpoint NPZ content hash')
    rows=[]
    with np.load(io.BytesIO(raw),allow_pickle=False) as data:
        for seed,reference_row in zip(DEVELOPMENT,frozen['run']['rows']):
            t,f,sel=unpack(data,seed)
            if sel!=reference_row['calibration_selected']:raise AssertionError('component selection was changed')
            rows.append(compare_one(seed,t,f,sel,reference_row,with_native=True))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');p.add_argument('--check',action='store_true')
    a=p.parse_args()
    if a.write==a.check:raise SystemExit('choose exactly one of --write or --check')
    result=write() if a.write else check()
    print(json.dumps({'mode':'wrote-checkpoints' if a.write else 'checkpoint-frozen native replay PASS','rows':result},indent=2))
if __name__=='__main__':main()
