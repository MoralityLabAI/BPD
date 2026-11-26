"""Recheck serialized certificates with scalar autocorrelations and bit-packed Gram."""
import json, argparse
from pathlib import Path

def run(base):
    counts={}; checks=0
    for name in ('engineering','engineering_extra','results','learn_results'):
        r=json.loads((base/(name+'.json')).read_text()); c=0
        for a in r['rows']:
            cert=a['certificate']; assert a['solved']==(cert is not None)
            if not cert: continue
            m=a['m']; assert len(cert)==4 and all(len(s)==m and set(s)<={-1,1} for s in cert)
            for t in range(1,m): assert sum(s[i]*s[(i+t)%m] for s in cert for i in range(m))==0; checks+=1
            A,B,C,D=cert; rows=[]
            specs=[[(A,1,False,False),(B,1,False,True),(C,1,False,True),(D,1,False,True)],[(B,-1,False,True),(A,1,False,False),(D,-1,True,True),(C,1,True,True)],[(C,-1,False,True),(D,1,True,True),(A,1,False,False),(B,-1,True,True)],[(D,-1,False,True),(C,-1,True,True),(B,1,True,True),(A,1,False,False)]]
            for br in range(4):
                for i in range(m):
                    row=0
                    for s,sgn,trans,rev in specs[br]:
                        for j in range(m):
                            k=m-1-j if rev else j; idx=(i-k if trans else k-i)%m
                            row=(row<<1)|(sgn*s[idx]==1)
                    rows.append(row)
            for i,u in enumerate(rows):
                for v in rows[i+1:]: assert (u^v).bit_count()==2*m; checks+=1
            c+=1
        counts[name]=c
    return {'verified_certificates':counts,'integer_assertions':checks,'method':'direct cyclic sums and bit-packed Gram via Hamming distances, independently constructed block rows'}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dir',default='.'); a=p.parse_args(); base=Path(a.dir); result=run(base); (base/'independent_audit.json').write_text(json.dumps(result,indent=2)); print(result)
