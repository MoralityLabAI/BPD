"""Audit independent/native hosted replay and compare against immutable CPU source receipt.

This diagnostic is deliberately separate from the prospective experiment source.
Any count disagreement is archived as a failure, never silently tolerated.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .reference import load_frozen

COUNTS=('eligible_match','source_match_available','first_head_ok','first_clean_correct',
        'full_donor_correct','selected_off_correct','random_off_1_correct','random_off_2_correct',
        'damage_counts_itt','matched_nuisance_recipient_label_correct',
        'matched_nuisance_donor_label_correct','recipient_sham_donor_label_correct')
PAIR=('match','donor_full_correct','damage_selected','damage_random1','damage_random2','nuisance_own_correct')

def audit(independent,native,frozen):
    assert independent['run']['seeds']==native['run']['seeds']==frozen['run']['seeds']
    findings=[]
    for a,b,c in zip(independent['run']['rows'],native['run']['rows'],frozen['run']['rows']):
        n=a['seed'];issues=[]
        for key in COUNTS:
            x,y,z=a['donor'][key],b['donor'][key],c['donor'][key]
            if x!=y:issues.append({'field':key,'scope':'hosted_backend_mismatch','independent':x,'native':y})
            if x!=z:issues.append({'field':key,'scope':'frozen_host_mismatch','frozen':z,'hosted':x})
        for key in PAIR:
            x=a['donor']['pair_data'][key];y=b['donor']['pair_data'][key];z=c['donor']['pair_data'][key]
            if x!=y:
                issues.append({'field':key,'scope':'hosted_backend_mismatch','pair_ids':[i for i,(v,w) in enumerate(zip(x,y)) if v!=w]})
            if x!=z:
                issues.append({'field':key,'scope':'frozen_host_mismatch','pair_ids':[i for i,(v,w) in enumerate(zip(x,z)) if v!=w]})
        if a['task']!=b['task']:
            issues.append({'field':'task','scope':'hosted_backend_mismatch','independent':a['task'],'native':b['task']})
        if a['task']!=c['task']:
            issues.append({'field':'task','scope':'frozen_host_mismatch','frozen':c['task'],'hosted':a['task']})
        if b['native_parity_max'] is None or b['native_parity_max']>1e-5:
            issues.append({'field':'native_parity_max','scope':'hosted_native_failure','value':b['native_parity_max']})
        if issues:findings.append({'seed':n,'issues':issues})
    return {'schema':'bpd.cq08.cross_environment_audit.v1',
        'backend_parity':'PASS' if not any(i['scope']=='hosted_backend_mismatch' for f in findings for i in f['issues']) else 'FAIL',
        'frozen_receipt_parity':'PASS' if not any(i['scope']=='frozen_host_mismatch' for f in findings for i in f['issues']) else 'FAIL',
        'n_seeds_with_findings':len(findings),'findings':findings}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--independent',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=audit(json.loads(a.independent.read_text()),json.loads(a.native.read_text()),load_frozen())
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'backend_parity':result['backend_parity'],
                      'frozen_receipt_parity':result['frozen_receipt_parity'],
                      'n_seeds_with_findings':result['n_seeds_with_findings'],
                      'summary':[{'seed':f['seed'],'n_issues':len(f['issues'])} for f in result['findings']]},indent=2))
    return 0 if result['backend_parity']=='PASS' and result['frozen_receipt_parity']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
