"""Immutable replay for CQ-07 excluded-root *development-only* geometry audit."""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json
from pathlib import Path
from . import geometry as G

HERE=Path(__file__).resolve().parent
LOCK=HERE/'results/LOCKED_SUMMARY.json'
RAW=HERE/'results/cq07_development_independent.json.gz.b64'
SOURCE_FILES=(
    ('PROTOCOL.md',HERE/'PROTOCOL.md'),
    ('geometry.py',HERE/'geometry.py'),
    ('reference.py',HERE/'reference.py'),
    ('CQ06/experiment.py',HERE.parent/'cq06_nonlinear_feedback'/'experiment.py'),
)


def git_blob_sha256_misnomer(path):
    """Git blob SHA-1, named clearly in lock; avoid conflating with SHA256."""
    content=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()


def source_header():return {name:git_blob_sha256_misnomer(p) for name,p in SOURCE_FILES}


def read_raw():
    data=gzip.decompress(base64.b64decode(RAW.read_text(encoding='ascii')))
    return data,json.loads(data)


def compact(run):
    return {'schema':run['schema'],'seeds':run['seeds'],'pairs_per_root':run['pairs_per_root'],
            'totals':run['totals'],'feasibility_totals':run['feasibility_totals'],
            'rows':[{'seed':r['seed'],'counts':r['counts'],'feasibility':r['feasibility'],
                     'min_singular_value':r['model_geometry']['combined_min_singular_value'],
                     'rank':r['model_geometry']['combined_input_plus_bpd_rank'],
                     'pair_masks':{k:r['pair_outcomes'][k] for k in ('bpd','null_1','null_2','full_donor','sham',
                         'patch_feasible','null1_matched','null2_matched','joint_patch_null1',
                         'matched_null1_target_x_sign')}}
                    for r in run['rows']]}


def check(run,locked):
    if source_header()!=locked['source_git_blob_sha1']:
        raise AssertionError('CQ-07 engineering source/protocol changed from immutable lock')
    raw,data=read_raw()
    if hashlib.sha256(raw).hexdigest()!=locked['development_raw_sha256']:
        raise AssertionError('compressed development receipt content changed')
    if compact(data)!=locked['counts']:
        raise AssertionError('immutable summary disagrees with archived raw receipt')
    a=compact(run);b=locked['counts']
    for k in ('schema','seeds','pairs_per_root','totals','feasibility_totals'):
        if a[k]!=b[k]:raise AssertionError(f'CQ-07 {k} changed')
    for r,s in zip(a['rows'],b['rows']):
        for k in ('seed','counts','feasibility','rank','pair_masks'):
            if r[k]!=s[k]:raise AssertionError(f'CQ-07 root {r["seed"]} {k} changed')
        if abs(r['min_singular_value']-s['min_singular_value'])>1e-3:
            raise AssertionError(f'CQ-07 rank witness drift for {r["seed"]}')
    assert all(row['model_geometry']['zero_intersection_certificate'] for row in run['rows'])
    assert all(row['feasibility']['range_violations_bpd']==0 for row in run['rows'])
    if run['backend']=='native':
        assert all(row['max_native_state_parity_error']<=1e-5 for row in run['rows'])
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=('independent','native'),default='independent')
    p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path)
    opts=p.parse_args()
    result=G.cohort(G.DEVELOPMENT,backend=opts.backend)
    if opts.check:
        locked=json.loads(LOCK.read_text())
        check(result,locked)
        print('BPD-CQ07 source-locked excluded-root',opts.backend,'replay PASS')
    if opts.out:
        if opts.out.exists():raise FileExistsError('write-once run output exists')
        opts.out.parent.mkdir(parents=True,exist_ok=True)
        opts.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'backend':opts.backend,'totals':result['totals'],
           'feasibility':result['feasibility_totals']},sort_keys=True,indent=2))

if __name__=='__main__':main()
