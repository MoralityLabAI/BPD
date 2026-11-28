"""HAD-07B: exact local support/margin lift; full correlation remains unsolved."""
from itertools import product
from functools import lru_cache
from pathlib import Path
import json
from joint_quotient import paf, orbits

MASKS=[m for m in product((0,1),repeat=4) if sum(m)%2==0]


def degrees():
    layers=[{(0,0,0,0):None}]
    for _ in range(11):
        nxt={}
        for x in layers[-1]:
            for m in MASKS:
                y=tuple(a+b for a,b in zip(x,m))
                nxt.setdefault(y,(x,m))
        layers.append(nxt)
    return layers


def row_options(u,v):
    # One fixed singleton and 11 pairs; exactly one channel is active at singleton.
    assert (u+v)%2==1
    active=v%2
    out={}
    for sign in (-1,1):
        a=(u-(1-active)*sign)//2
        b=(v-active*sign)//2
        for k in range(12):
            if abs(b)<=k and (k-b)%2==0 and abs(a)<=11-k and (11-k-a)%2==0:
                out.setdefault(k,(sign,a,b))
    return out


def column(u,v,origin,layers):
    active=tuple(x%2 for x in v)
    if (origin and sum(active)!=3) or (not origin and sum(active)%2):
        return {}
    banks=[row_options(a,b) for a,b in zip(u,v)]
    out={}
    for k in product(*banks):
        if k in layers[-1]:
            out.setdefault(sum(active)+2*sum(k),(k,[b[x] for b,x in zip(banks,k)],active))
    return out


def materialize(spec,layers):
    k,options,active=spec
    masks=[]
    for i in range(11,0,-1):
        k,m=layers[i][k];masks.append(m)
    masks.reverse()
    U=[[0]*23 for _ in range(4)];V=[[0]*23 for _ in range(4)]
    for f,(sgn,a,b) in enumerate(options):
        (V if active[f] else U)[f][0]=sgn
        for ch,s,flag in [(U,a,0),(V,b,1)]:
            pos=[i+1 for i,m in enumerate(masks) if m[f]==flag]
            nplus=(len(pos)+s)//2
            assert 0<=nplus<=len(pos)
            for j,t in enumerate(pos):ch[f][t]=ch[f][-t]=1 if j<nplus else -1
    return U,V


def feasible_masses(u,v,layers):
    cols=[column([r[i] for r in u],[r[i] for r in v],i==0,layers) for i in range(3)]
    return next(((a,b,c) for a,b,c in product(*cols) if a+6*b+6*c==599),None)


def lift(u,v,layers):
    # The remaining group is the direct product H6 on Z13 and inversion on Z23.
    group=[1,139,183,185,231,277,321,367,413,415,459,597]
    assert {(g%13,g%23) for g in group}=={(pow(4,k,13),s) for k in range(6) for s in (1,22)}
    O=orbits(13,{pow(4,k,13) for k in range(6)})
    cols=[column([r[i] for r in u],[r[i] for r in v],i==0,layers) for i in range(3)]
    found=next(((a,b,c) for a,b,c in product(*cols) if a+6*b+6*c==599),None)
    if found is None:return dict(status='EXCLUDED_BY_LOCAL_LIFT',column_masses=[sorted(c) for c in cols])
    U=[[0]*299 for _ in range(4)];V=[[0]*299 for _ in range(4)]
    for i,mass in enumerate(found):
        a,b=materialize(cols[i][mass],layers)
        for t in range(299):
            if t%13 in O[i]:
                for f in range(4):U[f][t]=a[f][t%23];V[f][t]=b[f][t%23]
    for f in range(4):
        assert all(abs(a)+abs(b)==1 for a,b in zip(U[f],V[f]))
        assert all(U[f][t]==U[f][g*t%299] and V[f][t]==V[f][g*t%299] for g in group for t in range(299))
        for i,o in enumerate(O):
            for j in o:
                assert sum(U[f][j::13])==u[f][i] and sum(V[f][j::13])==v[f][i]
    assert sum(abs(v) for row in V for v in row)==599
    assert all(sum(abs(row[t]) for row in V)%2==int(t==0) for t in range(299))
    assert sum(abs(row[0]) for row in V)==3
    ru=[sum(paf(row,t) for row in U)-(597 if t==0 else -2) for t in range(299)]
    rv=[sum(paf(row,t) for row in V)-(599 if t==0 else 0) for t in range(299)]
    assert all(sum(ru[j::13])==sum(rv[j::13])==0 for j in range(13))
    return dict(status='FEASIBLE_LOCAL_LIFT',column_masses=[sorted(c) for c in cols],
                selected_masses=found,energy_U=sum(r*r for r in ru[1:150]),
                energy_V=sum(r*r for r in rv[1:150]),U=U,V=V)


def main():
    base=Path(__file__).parent
    q=json.loads((base/'joint_receipt.json').read_text())['quotient']
    layers=degrees();results=[]
    @lru_cache(maxsize=None)
    def masses(u,v,origin):
        return tuple(column(u,v,origin,layers))
    for i,row in enumerate(q['systems']):
        u=row['U']; count=0; hit=None
        for v in row['all_V']:
            cols=[masses(tuple(r[j] for r in u),tuple(r[j] for r in v),j==0) for j in range(3)]
            if any(a+6*b+6*c==599 for a,b,c in product(*cols)):
                count+=1
                if hit is None:hit=v
        result=lift(u,hit if hit is not None else row['V_witness'],layers)
        assert (result['status']=='FEASIBLE_LOCAL_LIFT')==bool(count)
        result.update(id=i,quotient_count=len(row['all_V']),local_lift_count=count,V_witness=hit)
        results.append(result)
    out=dict(scope='LOCAL_MARGIN_AND_SUPPORT_LIFTS_NOT_HADAMARD_CERTIFICATES',
             reachable_degree_vectors=len(layers[-1]),results=results)
    (base/'lift_receipt.json').write_text(json.dumps(out,indent=2)+'\n')
    print('reachable degree vectors',len(layers[-1]))
    for r in results:print(r['id'],r['status'],r['local_lift_count'],r['quotient_count'],r.get('selected_masses'),r.get('energy_U'),r.get('energy_V'))

if __name__=='__main__':main()
