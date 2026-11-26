"""SCCG-06 exact dynamic quotient atlas; constructed development only."""
from collections import Counter
from fractions import Fraction as F
from itertools import combinations
import argparse
import hashlib
import json
import random
import sys
import time

N, K, PRIME = 14, 4, 65521
ROOTS, EPOCHS = 8, 12


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def kernel(rows, n):
    a = [[F(v) for v in row] for row in rows]
    pivots = []
    for j in range(n):
        r = len(pivots)
        i = next((i for i in range(r, len(a)) if a[i][j]), None)
        if i is None:
            continue
        a[r], a[i] = a[i], a[r]
        d = a[r][j]
        a[r] = [v/d for v in a[r]]
        for i in range(len(a)):
            if i != r and a[i][j]:
                d = a[i][j]
                a[i] = [x-d*y for x, y in zip(a[i], a[r])]
        pivots.append(j)
    out = []
    for j in range(n):
        if j not in pivots:
            v = [F(0)]*n
            v[j] = F(1)
            for i, p in enumerate(pivots):
                v[p] = -a[i][j]
            out.append(tuple(v))
    return tuple(out)


def restrict(rows, S):
    return tuple(tuple(row[i] for i in S) for row in rows)


def full_mod(rows, n):
    """Full rank modulo PRIME is an exact rational full-rank certificate."""
    basis = {}
    for row in rows:
        v = [x % PRIME for x in row]
        for j in range(n):
            if not v[j]:
                continue
            if j in basis:
                f = v[j]
                v = [(x-f*y) % PRIME for x, y in zip(v, basis[j])]
            else:
                f = pow(v[j], -1, PRIME)
                basis[j] = [x*f % PRIME for x in v]
                break
        if len(basis) == n:
            return True
    return False


def scan(rows, supports, counts):
    result = {}
    for S in supports:
        m = restrict(rows, S)
        counts['modular_screens'] += 1
        if full_mod(m, len(S)):
            continue
        counts['full_rational_kernels'] += 1
        z = kernel(m, len(S))
        if z:
            result[S] = z
    return result


class Atlas:
    def __init__(self, D, supports):
        self.D = tuple(map(tuple, D))
        self.n = len(D[0])
        self.counts = Counter()
        self.base = scan(self.D, supports, self.counts)
        self.index = [set() for _ in range(self.n)]
        for S in self.base:
            for i in S:
                self.index[i].add(S)
        self.last = None
        self.active = {}

    def update(self, D, E, sparse):
        if tuple(map(tuple, D)) != self.D:
            raise ValueError('fixed quotient changed; rebuild atlas')
        E = tuple(map(tuple, E))
        if any(len(row) != self.n for row in E):
            raise ValueError('mutable row width changed')
        if sparse and self.last is not None and len(E) == len(self.last):
            changed = [i for i in range(self.n)
                       if any(a[i] != b[i] for a,b in zip(E,self.last))]
            todo = set().union(*(self.index[i] for i in changed))
        else:
            todo = self.base.keys()
        self.counts['candidate_refreshes'] += len(todo)
        self.counts['candidate_reuses'] += len(self.base)-len(todo)
        for S in sorted(todo):
            Z = self.base[S]
            reduced = tuple(tuple(dot(row,z) for z in Z) for row in restrict(E,S))
            self.counts['reduced_kernels'] += 1
            K = kernel(reduced,len(Z))
            lifted = tuple(tuple(sum(v[j]*Z[j][i] for j in range(len(Z)))
                                 for i in range(len(S))) for v in K)
            if lifted:
                self.active[S] = lifted
            else:
                self.active.pop(S,None)
        self.last = E
        return dict(self.active)


def digest(result):
    # Canonical support/nullity fingerprint, independent of basis choice.
    return hashlib.sha256(json.dumps([(S,len(z)) for S,z in sorted(result.items())],
                                    separators=(',',':')).encode()).hexdigest()


def verify(D,E,direct,actual):
    if {S:len(z) for S,z in direct.items()} != {S:len(z) for S,z in actual.items()}:
        raise ArithmeticError('support or nullity mismatch')
    for S,Z in actual.items():
        if any(dot(row,z) for row in restrict(D+E,S) for z in Z):
            raise ArithmeticError('non-null lifted vector')
        if kernel(tuple(zip(*Z)),len(Z)):
            raise ArithmeticError('dependent lifted basis')


def world(seed):
    rng = random.Random(seed)
    planted = seed % 4 != 0
    nbase = N-2 if planted else N
    columns = [[t**j for j in range(8)] for t in rng.sample(range(-9,10),nbase)]
    groups = []
    if planted:
        ix = rng.sample(range(nbase),6)
        for triple in (ix[:3],ix[3:]):
            columns.append([-sum(columns[i][j] for i in triple) for j in range(8)])
            groups.append(triple+[len(columns)-1])
    order = list(range(N)); rng.shuffle(order)
    inv = {i:j for j,i in enumerate(order)}
    groups = [tuple(inv[i] for i in S) for S in groups]
    D = tuple(zip(*(columns[i] for i in order)))
    row = [rng.randint(-3,3) for _ in range(N)]
    epochs = []
    for t in range(EPOCHS):
        if t % 4 == 0:
            row = [rng.randint(-3,3) for _ in range(N)]
        elif t % 4 == 3:
            row[rng.randrange(N)] += 1
        if groups:
            for i,S in enumerate(groups):
                if t % 4 == 0 or (t+i)%2==0:
                    row[S[-1]] = -sum(row[j] for j in S[:-1]) + int((t//2+i)%2)
        epochs.append((tuple(row),))
    return D,epochs


def edge_cases():
    # Nullity 3, then 2, then 1, then 3; two variable rows.
    D = ((1,1,1,1),)
    S = [(0,1,2,3)]
    a = Atlas(D,S)
    for E in (((0,0,0,0),),((0,1,0,0),),((0,1,0,0),(0,0,1,0)),()):
        ref = scan(D+E,S,Counter())
        verify(D,E,ref,a.update(D,E,True))
    # A rationally full-rank support can be deficient modulo PRIME.
    D2=((PRIME,0),(0,1))
    if scan(D2,[(0,1)],Counter()):
        raise ArithmeticError('modular deficiency treated as proof')
    try:
        a.update(((1,1,2,1),),(),True)
    except ValueError:
        pass
    else:
        raise ArithmeticError('stale quotient accepted')
    # Non-coordinate update direction U=(1,2,3)^T. L annihilates U.
    C=((1,2,3),(0,1,1),(2,0,1))
    def quotient(c):
        return tuple(tuple(y-f*x for x,y in zip(c[0],row))
                     for f,row in zip((2,3),c[1:]))
    D3=quotient(C)
    for v in ((0,0,0),(2,-3,1),(-1,1,1)):
        Ct=tuple(tuple(x+f*y for x,y in zip(row,v)) for f,row in zip((1,2,3),C))
        if quotient(Ct)!=D3:
            raise ArithmeticError('annihilator invariance failed')
        ref=scan(Ct,[(0,1,2)],Counter())
        a3=Atlas(D3,[(0,1,2)])
        verify(D3,(Ct[0],),ref,a3.update(D3,(Ct[0],),True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output');args=ap.parse_args()
    edge_cases()
    supports=list(combinations(range(N),K))
    modes=('rebuild','dense','sparse')
    totals={m:Counter() for m in modes};times={m:0. for m in modes}
    compile_seconds={m:0. for m in modes};rows=[];candidates=0
    births=deaths=0;cache_bytes=0
    for root in range(41000,41000+ROOTS):
        D,epochs=world(root);answers={};atlases={}
        order=modes if root%2==0 else modes[::-1]
        for mode in order:
            start=time.perf_counter()
            a=None if mode=='rebuild' else Atlas(D,supports)
            compile_seconds[mode]+=time.perf_counter()-start
            if a:
                atlases[mode]=a
            if mode=='dense':
                candidates+=len(a.base)
                # Serialized rational base only, not process memory or full index.
                payload=[(S,[[str(v) for v in z] for z in Z]) for S,Z in sorted(a.base.items())]
                cache_bytes=max(cache_bytes,len(json.dumps(payload,separators=(',',':')).encode()))
            counts=Counter();ans=[]
            start=time.perf_counter()
            for E in epochs:
                ans.append(scan(D+E,supports,counts) if a is None else a.update(D,E,mode=='sparse'))
            times[mode]+=time.perf_counter()-start
            totals[mode].update(counts if a is None else a.counts)
            answers[mode]=ans
        prev=set()
        for t,E in enumerate(epochs):
            ref=answers['rebuild'][t]
            for mode in ('dense','sparse'):
                verify(D,E,ref,answers[mode][t])
            live=set(ref);births+=len(live-prev);deaths+=len(prev-live);prev=live
            rows.append(dict(root=root,epoch=t,active=len(live),digest=digest(ref)))
    result=dict(scope='CONSTRUCTED_DEVELOPMENT_ONLY',roots=ROOTS,epochs=EPOCHS,
                supports_per_epoch=len(supports),support_epoch_decisions=ROOTS*EPOCHS*len(supports),
                quotient_candidates=candidates,births_including_initial=births,deaths=deaths,
                max_serialized_base_bytes=cache_bytes,rows=rows,
                arms={m:dict(counts=dict(totals[m]),compile_seconds=compile_seconds[m],
                             update_seconds=times[m],total_seconds=compile_seconds[m]+times[m]) for m in modes},
                python=sys.version.split()[0])
    if args.output:
        with open(args.output,'w') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2,sort_keys=True))


if __name__=='__main__':main()
