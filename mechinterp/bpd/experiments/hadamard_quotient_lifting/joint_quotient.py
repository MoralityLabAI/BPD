"""HAD-07A: exact projection and support sieve, no model training.

All counts use Python integers. A quotient witness is only a relaxation.
"""
from collections import Counter, defaultdict
from itertools import product
from math import isqrt, log2, comb
import json
from pathlib import Path


def orbits(n, group):
    left = set(range(n))
    result = []
    while left:
        i = min(left)
        orbit = sorted({i * a % n for a in group})
        assert set(orbit) <= left
        left.difference_update(orbit)
        result.append(orbit)
    return result


def paf(x, t):
    return sum(v * x[(i + t) % len(x)] for i, v in enumerate(x))


def add(x, y):
    return tuple(a + b for a, b in zip(x, y))


def sub(x, y):
    return tuple(a - b for a, b in zip(x, y))


def expand(values, orbit, n):
    row = [0] * n
    for v, block in zip(values, orbit):
        for i in block:
            row[i] = v
    return row


def support_counts():
    group = [1, 139, 183, 185, 231, 277, 321, 367, 413, 415, 459, 597]
    assert len({g % 299 for g in group}) == 12
    orbit = orbits(299, group)
    weights = [len(o) for o in orbit]
    assert weights[0] == 1 and all(w % 2 == 0 for w in weights[1:])
    assert sum(weights) == 299
    def count(origin, allowed):
        dp = Counter({0: 1})
        for index, w in enumerate(weights):
            nxt = Counter()
            for k, val in dp.items():
                for j in origin if index == 0 else allowed:
                    if k + j * w <= 599:
                        nxt[k + j * w] += val * comb(4, j)
            dp = nxt
        return dp[599]
    baseline = count(range(5), range(5))
    parity = count([3], [0, 2, 4])
    assert 0 < parity < baseline
    return dict(orbit_sizes=dict(Counter(weights)), orbits=len(weights),
                support_systems_mass_only=baseline, support_systems_parity=parity,
                log2_mass_only=log2(baseline), log2_parity=log2(parity),
                removed_bits=log2(baseline)-log2(parity))


def quotient(all_witnesses=False):
    # q=299, compress both ternary channels by the kernel of size 23.
    # U target (553,-46,-46), V target (599,0,0) on the three shift classes.
    orbit = orbits(13, {pow(4, k, 13) for k in range(6)})
    weights = [len(o) for o in orbit]
    assert weights == [1, 6, 6]
    def sig(values):
        r = expand(values, orbit, 13)
        return tuple(paf(r, t) for t in (0, 1, 2))
    def norm(values):
        return sum(w*x*x for w,x in zip(weights, values))
    def u_bank(s):
        bound = isqrt(553//6)
        out = []
        for b,c in product(range(-bound,bound+1), repeat=2):
            a = (s-6*(b+c),b,c)
            if max(map(abs,a)) > 23 or norm(a) > 553:
                continue
            if s == 0 and a < tuple(-x for x in a):
                continue
            out.append((a,sig(a)))
        return sorted(out)
    A,B = u_bank(1),u_bank(0)
    pairs = defaultdict(list)
    for i,(a,x) in enumerate(B):
        for b,y in B[i:]:
            z = add(x,y)
            if z[0] <= 553:
                pairs[z].append((a,b))
    u_systems = []
    for a,x in A:
        for b,y in B:
            for c,d in pairs.get(sub(sub((553,-46,-46),x),y),[]):
                if b <= c <= d:
                    u_systems.append((a,b,c,d))
    assert len(u_systems)==len(set(u_systems))
    # Independent loop over canonical zero-sum triples checks completeness.
    from itertools import combinations_with_replacement
    target = (553,-46,-46)
    amap = defaultdict(list)
    for a,x in A: amap[x].append(a)
    independent = set()
    for (b,x),(c,y),(d,z) in combinations_with_replacement(B,3):
        for a in amap.get(sub(sub(sub(target,x),y),z),[]):
            independent.add((a,b,c,d))
    assert independent == set(u_systems)
    V = []
    bound = isqrt(599//6)
    for v in product(range(-23,24),range(-bound,bound+1),range(-bound,bound+1)):
        if v >= tuple(-x for x in v) and norm(v)<=599:
            V.append((v,sig(v)))
    receipts = []
    for u in u_systems:
        banks = [[(v,s) for v,s in V if all((a+b)%2==1 and abs(a)+abs(b)<=23
                    for a,b in zip(row,v))] for row in u]
        # Count labeled V rows with independent sign normalization. Multiple V
        # orderings on equal U rows are kept; this is not a joint orbit count.
        counts = [Counter(s for _,s in bank) for bank in banks]
        reps = [{s:v for v,s in bank} for bank in banks]
        right = Counter()
        witness = {}
        for x,n in counts[2].items():
            for y,m in counts[3].items():
                z=add(x,y)
                if z[0]<=599:
                    right[z]+=n*m
                    witness.setdefault(z,(reps[2][x],reps[3][y]))
        count=0
        hit=None
        for x,n in counts[0].items():
            for y,m in counts[1].items():
                k=sub(sub((599,0,0),x),y)
                if k in right:
                    count+=n*m*right[k]
                    if hit is None: hit=(reps[0][x],reps[1][y],*witness[k])
        assert count>0 and hit is not None
        ue=[expand(a,orbit,13) for a in u]
        ve=[expand(a,orbit,13) for a in hit]
        assert sorted(map(sum,ue))==[0,0,0,1]
        for t in range(13):
            assert sum(paf(r,t) for r in ue)==(553 if t==0 else -46)
            assert sum(paf(r,t) for r in ve)==(599 if t==0 else 0)
        for a,b in zip(ue,ve):
            assert all((x+y)%2==1 and abs(x)+abs(y)<=23 for x,y in zip(a,b))
        record=dict(U=u,V_witness=hit,V_bank_sizes=list(map(len,banks)),
                    V_labeled_sign_normalized_count=count)
        if all_witnesses:
            bysig=[]
            for bank in banks:
                d=defaultdict(list)
                for v,s in bank:d[s].append(v)
                bysig.append(d)
            needed={sub(sub((599,0,0),x),y) for x in counts[0] for y in counts[1]}
            rp=defaultdict(list)
            for x in counts[2]:
                for y in counts[3]:
                    z=add(x,y)
                    if z in needed:rp[z].append((x,y))
            allv=[]
            for x in counts[0]:
                for y in counts[1]:
                    k=sub(sub((599,0,0),x),y)
                    for z,w in rp.get(k,[]):
                        allv.extend(product(bysig[0][x],bysig[1][y],bysig[2][z],bysig[3][w]))
            assert len(allv)==count==len(set(allv))
            record['all_V']=allv
        receipts.append(record)
    return dict(scope='COMPLETE_U_QUOTIENT_ENUMERATION_AND_V_RELAXATION',
                p=13, fibre=23, orbit=orbit, U_bank_sizes=[len(A),len(B)],
                V_bank_size=len(V), canonical_U_count=len(u_systems),
                U_with_V_relaxation=sum(r['V_labeled_sign_normalized_count']>0 for r in receipts),
                systems=receipts)


def symmetric_controls(q):
    # Exhaustive ordered row-pair join. At every valid B4 pair complement,
    # XOR of support masks must be 1 and exactly three V rows occupy origin.
    orbit=orbits(q,[1,q-1]); bank=[]
    for a in product([-1,1],repeat=len(orbit)):
        A=expand(a,orbit,q)
        for b in product([-1,1],repeat=len(orbit)):
            B=expand(b,orbit,q)
            u=[(x+y)//2 for x,y in zip(A,B)]
            v=[(x-y)//2 for x,y in zip(A,B)]
            s=tuple(paf(u,t) for t in range(q//2+1))+tuple(paf(v,t) for t in range(q//2+1))
            mask=sum((x!=0)<<i for i,x in enumerate(v))
            bank.append((s,mask,int(v[0]!=0)))
    pairs=defaultdict(Counter)
    for s,a,x in bank:
        for t,b,y in bank:
            pairs[add(s,t)][(a^b,x+y)]+=1
    target=(2*q-1,)+(-2,)*(q//2)+(2*q+1,)+(0,)*(q//2)
    count=0;checks=0
    for key,states in pairs.items():
        for (mask,n),a in states.items():
            for (other,m),b in pairs.get(sub(target,key),{}).items():
                assert mask^other==1 and n+m==3
                count+=a*b;checks+=1
    return dict(q=q,ordered_binary_rows=len(bank),ordered_solutions=count,
                signature_pairs_checked=checks)


def main():
    result=dict(scope='EXACT_MATH_AND_FINITE_RELAXATIONS_ONLY',support=support_counts(),
                quotient=quotient(all_witnesses=True),controls=[symmetric_controls(q) for q in (3,5,7)])
    out=Path(__file__).with_name('joint_receipt.json')
    out.write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='quotient'},indent=2))
    print('U systems',result['quotient']['canonical_U_count'],'with V',result['quotient']['U_with_V_relaxation'])
    print('V counts',[r['V_labeled_sign_normalized_count'] for r in result['quotient']['systems']])

if __name__=='__main__':main()
