"""Independent checks of archive result receipts and all injected children."""
import argparse, collections, json, itertools
from pathlib import Path
import numpy as np
from audit import bitgram
from archive import Archive
from had05 import Trajectory

def scalar_r(x):
    m=len(x[0]); return [sum(int(s[i])*int(s[(i+t)%m]) for s in x for i in range(m)) for t in range(1,m//2+1)]

def run(path,near=False):
    data=json.loads(Path(path).read_text()); a=data['args']; seen={}; checks=certs=injections=injected_solutions=0
    for z in data['rows']:
        key=(z['root'],z['m'],z['case']); prev=seen.setdefault(key,z['start_hash']); m=z['m']; assert prev==z['start_hash']; assert z['candidate_units']==z['archive_candidate_units']+z['search_candidate_scores']; assert z['candidate_units']<=a['rounds']*a['chunk']*4*m; assert z['solved']==(z['certificate'] is not None); checks+=4
        if 'archive_max_payload' in z: assert z['archive_payload_bytes']<=z['archive_max_payload']; checks+=1
        if z['solved']:
            x=z['certificate']; assert len(x)==4 and all(len(s)==m and set(s)<={-1,1} for s in x); assert not any(scalar_r(x)); checks+=1+bitgram(x); certs+=1
        if near:
            for idx,b in enumerate(z['injections'],1):
                x=b['sequences']; r=scalar_r(x); assert len(x)==4 and all(len(s)==m and set(s)<={-1,1} for s in x); assert sum(v*v for v in r)==b['energy']<b['replaced_best']; assert sum(sum(s)**2 for s in x)==4*m; assert r[0]==r[-1]==0; assert b['generation']==idx; assert b['work']<=z['candidate_units']; checks+=6
                if m%3==0:
                    q=0
                    for s in x:
                        u,v,w=[sum(s[k::3]) for k in range(3)]; q+=u*u+v*v+w*w-u*v-v*w-w*u
                    assert q==4*m; checks+=1
                injections+=1
            if z.get('solution_injection'):
                assert z['solved'] and z['solution_kind']=='native' and z['solution_injection']<=len(z['injections']); injected_solutions+=1; checks+=1
    return dict(paired_populations=len(seen),certificates=certs,injected_children=injections,solutions_from_injected_children=injected_solutions,checks=checks)

def positive():
    # Independent disjoint-source library fixture: one known verified order-12 construction.
    x=[[1,1,1],[1,1,-1],[1,1,-1],[1,1,-1]]
    ar=Archive(3,8,1)
    # Only two distinct AC signatures are needed; repetition is legal in the GS array.
    ar.add(x[0],(0,0,0),100); ar.add(x[1],(1,0,0),100)
    assert ar.cert is not None and not any(scalar_r(ar.cert)) and len(set(s[0] for s in ar.sources))==2
    return dict(disjoint_source_fixture=True,gram_checks=bitgram(ar.cert))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dir',default='.'); a=p.parse_args(); d=Path(a.dir); out=dict(exact=run(d/'archive_results.json'),near=run(d/'near_results.json',True),positive_control=positive()); (d/'archive_audit.json').write_text(json.dumps(out,indent=2)); print(out)
