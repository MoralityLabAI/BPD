"""Source-locked CQ-08 replay, independent/native count parity and write-once receipts."""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json
from pathlib import Path
from .experiment import ROOT,DEVELOPMENT,SHAM,CONFIRMATION,cohort,summarize

LOCK=ROOT/'REGISTRATION.json'
RECEIPT=ROOT/'results'/'cq08_development_independent.json.gz.b64'
SUMMARY=ROOT/'results'/'LOCKED_DEVELOPMENT.json'
SOURCES=('PROTOCOL.md','experiment.py','reference.py','power.py','tests/test_experiment.py')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def git_blob(p):
    data=Path(p).read_bytes()
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()

def source_locks():return {p:git_blob(ROOT/p) for p in SOURCES if (ROOT/p).exists()}

def registration():
    lock=json.loads(LOCK.read_text(encoding='utf-8'))
    for name,digest in source_locks().items():
        if lock['source_git_blob_sha1'][name]!=digest:
            raise AssertionError('CQ08 source changed after prospective registration: '+name)
    return lock

def load_frozen():
    lock=registration()
    if sha(RECEIPT)!=lock['development_receipt_blob_sha256']:
        raise AssertionError('CQ08 locked development blob hash mismatch')
    data=json.loads(gzip.decompress(base64.b64decode(RECEIPT.read_text())).decode('utf-8'))
    if hashlib.sha256(json.dumps(data,sort_keys=True,indent=2).encode('utf-8')).hexdigest()!=lock['development_receipt_content_sha256']:
        raise AssertionError('CQ08 original uncompressed JSON changed')
    return data

def compare_rows(actual,frozen,backend):
    if [r['seed'] for r in actual['run']['rows']]!=[r['seed'] for r in frozen['run']['rows']]:
        raise AssertionError('seed roster changed')
    if actual['summary']!=frozen['summary']:
        # Exact count and effect checks; platform-dependent floats can differ slightly.
        for k in ('roots','n_donor_pairs','jointly_matched','full_donor_correct','damage_itt',
                  'roots_feasibility_at_least_80pct','teacher_task_hits','full_bpd_task_hits','paired_z_myopic_hits'):
            assert actual['summary'][k]==frozen['summary'][k],f'failed summary {k}'
        for k in ('primary_effects_by_root','second_effects_by_root','random_random_abs_shams_by_root'):
            assert actual['summary'][k]==frozen['summary'][k],f'failed per-root {k}'
    for a,b in zip(actual['run']['rows'],frozen['run']['rows']):
        assert a['calibration_selected']==b['calibration_selected']
        assert a['task']==b['task']
        for k in ('pairs','eligible_match','full_donor_correct','selected_off_correct',
                  'random_off_1_correct','random_off_2_correct',
                  'matched_nuisance_recipient_label_correct','matched_nuisance_donor_label_correct',
                  'damage_counts_itt','pair_data'):
            if k=='pair_data':
                for name in ('match','donor_full_correct','damage_selected','damage_random1',
                             'damage_random2','nuisance_own_correct'):
                    assert a['donor'][k][name]==b['donor'][k][name],(a['seed'],name)
            else:
                assert a['donor'][k]==b['donor'][k],(a['seed'],k)
        assert a['donor']['max_state_box_violation']==0.
        assert a['donor']['max_source_replay_error']<=1e-6
        assert a['gauge']['max_full_operator_shear_error']<=2e-6
        if a['gauge']['min_singular_gap']>=.001:
            assert a['gauge']['max_recanonical_masked_logit_drift']<=1e-5
        if backend=='native':
            assert a['native_parity_max'] is not None and a['native_parity_max']<=1e-5
    return True

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('development','sham','confirmation'),default='development')
    parser.add_argument('--backend',choices=('independent','native'),default='independent')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    lock=registration()
    if args.phase=='confirmation':
        raise RuntimeError('sealed CQ08 confirmation: release requires separate explicit unlock')
    if args.phase=='development':
        run=cohort('development',args.backend,DEVELOPMENT)
        result={'run':run,'summary':summarize(run)}
        if args.check:
            frozen=load_frozen()
            compare_rows(result,frozen,args.backend)
            print('CQ08',args.backend,'locked development exact count replay PASS')
    else:
        run=cohort('sham',args.backend,SHAM)
        result={'run':run,'summary':summarize(run)}
        print('CQ08 sham calibration, NOT confirmation; external 59-root preflight still needed')
    if args.out:
        if args.out.exists():raise FileExistsError(f'write-once CQ08 receipt already exists: {args.out}')
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'phase':args.phase,'backend':args.backend,'summary':result['summary']},indent=2))

if __name__=='__main__':main()
